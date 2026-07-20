"""
競馬予想分析ツール Streamlit WebUI
起動: streamlit run src/app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

MASTER_CSV     = Path(__file__).parent.parent / "data" / "processed" / "master.csv"
MASTER_PARQUET = Path(__file__).parent.parent / "data" / "processed" / "master.parquet"
MODEL_PATH     = Path(__file__).parent.parent / "data" / "processed" / "lgbm_model.pkl"
DATA_DIR       = Path(__file__).parent.parent / "data"
INPUT_DIR      = Path.home() / "Downloads"
LAST_PRED_PATH = Path(__file__).parent.parent / "data" / "processed" / "last_pred.parquet"

def _download_master_from_gdrive() -> tuple[bool, str]:
    """
    Google Drive から master.csv をダウンロードする。
    Secrets に gdrive_master_csv_url が必要。
    戻り値: (成功フラグ, メッセージ)
    """
    try:
        url = st.secrets.get("gdrive_master_csv_url", "")
    except Exception:
        return False, "Secrets が設定されていません"
    if not url:
        return False, "Secrets に gdrive_master_csv_url が設定されていません"
    try:
        import re as _re_gd
        import gdown
        import gc
        MASTER_CSV.parent.mkdir(parents=True, exist_ok=True)
        _m = _re_gd.search(r'/d/([a-zA-Z0-9_-]+)', url)
        if not _m:
            return False, "Google Drive の URL からファイルIDを取得できませんでした"
        _file_id = _m.group(1)
        # 一時CSVにダウンロード
        _csv_tmp = MASTER_CSV.with_suffix('.csv.tmp')
        gdown.download(id=_file_id, output=str(_csv_tmp), quiet=False)
        if not _csv_tmp.exists() or _csv_tmp.stat().st_size < 1024 * 1024:
            if _csv_tmp.exists():
                _csv_tmp.unlink()
            return False, "ダウンロードされたファイルが小さすぎます（Google Drive の共有設定を確認してください）"
        # CSV → Parquet に変換（メモリ効率のため5万行ずつ処理）
        import pyarrow as _pa
        import pyarrow.parquet as _pq
        _pq_writer = None
        _ref_schema = None
        try:
            for _chunk in pd.read_csv(_csv_tmp, encoding='utf-8-sig', chunksize=50000, low_memory=False):
                _tbl = _pa.Table.from_pandas(_chunk, preserve_index=False)
                if _ref_schema is None:
                    _ref_schema = _tbl.schema
                    _pq_writer = _pq.ParquetWriter(str(MASTER_PARQUET), _ref_schema, compression='snappy')
                elif _tbl.schema != _ref_schema:
                    # チャンク間の型不一致を先頭チャンクのスキーマに合わせてキャスト
                    _cols = []
                    for _fn in _ref_schema.names:
                        _col = _tbl.column(_fn)
                        _rtype = _ref_schema.field(_fn).type
                        if _col.type != _rtype:
                            _col = _col.cast(_rtype)
                        _cols.append(_col)
                    _tbl = _pa.table(dict(zip(_ref_schema.names, _cols)), schema=_ref_schema)
                _pq_writer.write_table(_tbl)
                del _chunk, _tbl
                gc.collect()
        finally:
            if _pq_writer:
                _pq_writer.close()
            _csv_tmp.unlink(missing_ok=True)  # 251MB CSV を削除
            gc.collect()
        if MASTER_PARQUET.exists() and MASTER_PARQUET.stat().st_size > 1024 * 1024:
            _mb_csv = 251  # 元CSVサイズ
            _mb_pq  = MASTER_PARQUET.stat().st_size // 1024 // 1024
            return True, f"ダウンロード・変換成功（CSV {_mb_csv}MB → Parquet {_mb_pq}MB）"
        else:
            return False, "Parquet 変換に失敗しました"
    except Exception as _e:
        import traceback as _tb
        return False, f"{_e}\n{_tb.format_exc()}"

VENUE_MAP = {
    '東': '東京', '中': '中山', '京': '京都', '阪': '阪神',
    '名': '中京', '小': '小倉', '新': '新潟', '福': '福島',
    '函': '函館', '札': '札幌',
}
VENUE_ORDER = ['東京','中山','札幌','函館','福島','新潟','中京','阪神','京都','小倉']

def load_master_full():
    # メモリ削減: DB検索は直近5年のみ集計（個人運用向け）。6桁日付(YYMMDD)文字列で
    # 行フィルタして読込む（2016-2026は辞書順=時系列順）。非対応時は全読込→5年絞り。
    try:
        _cut_s = (pd.Timestamp.now().normalize() - pd.DateOffset(years=5)).strftime('%y%m%d')
        df = pd.read_parquet(MASTER_PARQUET, filters=[('日付', '>=', _cut_s)])
        if df.empty:
            raise ValueError("日付フィルタが空")
    except Exception:
        df = pd.read_parquet(MASTER_PARQUET)
        from features import filter_recent_years as _fry
        df = _fry(df)
    df['距離'] = pd.to_numeric(df['距離'], errors='coerce')
    df['着順_num'] = pd.to_numeric(
        df['着順'].astype(str).str.translate(str.maketrans('０１２３４５６７８９','0123456789')),
        errors='coerce'
    )
    df['人気'] = pd.to_numeric(df['人気'], errors='coerce')
    df['前走上り3F'] = pd.to_numeric(df['前走上り3F'], errors='coerce')
    df['_venue_name'] = df['開催'].astype(str).str.extract(r'\d+([^\d])')[0].map(VENUE_MAP)
    pos4c = pd.to_numeric(df['前4角'].astype(str).str.translate(
        str.maketrans('０１２３４５６７８９','0123456789')), errors='coerce')
    prev_horses = pd.to_numeric(df['前走頭数'], errors='coerce')
    df['_style_ratio'] = pos4c / prev_horses
    def _pay(s):
        try:
            v = str(s).strip()
            if v.startswith('(') or v in ('', 'nan', 'None', 'NaN'):
                return np.nan
            f = float(v)
            return f if (f == f and f > 0) else np.nan
        except Exception:
            return np.nan
    df['_tan_pay'] = df['単勝配当'].apply(_pay) if '単勝配当' in df.columns else np.nan
    df['_fuku_pay'] = df['複勝配当'].apply(_pay) if '複勝配当' in df.columns else np.nan
    return df

def _build_honmei_lines(info: dict) -> str:
    """本命◎/対抗○/妙味★ を1頭1行のHTMLで返す（ランキング表の印と一致）。"""
    if not info:
        return ''
    rows = []
    for key, disp, color, badge in [('本命', '本命', '#f1c40f', '◎'),
                                     ('対抗', '対抗', '#2ecc71', '○'),
                                     ('穴',   '妙味', '#9b59b6', '★')]:
        h = info.get(key)
        if h:
            ev_str   = f' EV{h["ev"]:+.0f}%'  if pd.notna(h.get('ev'))  else ''
            rows.append(
                f'<div style="margin:2px 0;">'
                f'<span style="color:{color};font-weight:bold;font-size:1.0em;">{badge}{disp}</span>'
                f'<span style="color:white;margin-left:6px;">{h["name"]}（{h["pop"]}人気）</span>'
                f'<span style="color:#aaa;font-size:0.85em;">{ev_str}</span>'
                f'</div>'
            )
    return ''.join(rows)


def parse_venue(kai_str: str) -> str:
    """
    '1東3' → '東京'  （Target形式）
    '東京' → '東京'  （netkeiba scrape形式 / フォールバック）
    """
    import re
    s = str(kai_str)
    m = re.search(r'\d+([^\d])', s)
    if m:
        return VENUE_MAP.get(m.group(1), '')
    # 略称の直接マッチ
    if s in VENUE_MAP:
        return VENUE_MAP[s]
    # フル名がそのまま入っている場合
    if s in VENUE_ORDER:
        return s
    return ''

st.set_page_config(page_title="競馬予想分析ツール", page_icon="🏇", layout="wide")

# モバイル判定: User-Agent から推定（st.context.headers）。取得不可なら PC 扱い。
# 用途: 馬カードの既定表示を PC=詳細 / スマホ=圧縮 に出し分ける。
_is_mobile = False
try:
    _ua = ''
    _hdrs = getattr(st.context, 'headers', None)
    if _hdrs:
        _ua = _hdrs.get('User-Agent', '') or _hdrs.get('user-agent', '') or ''
    _is_mobile = any(_k in _ua for _k in ('Mobile', 'Android', 'iPhone', 'iPad', 'iPod', 'Windows Phone'))
except Exception:
    _is_mobile = False

# スマホでも R選択ボタンを 6列グリッド（2行×6）に保つCSS。
# Streamlitは狭い画面で st.columns を縦積みにするため、rbtn_grid_* キーの
# コンテナ内のカラムだけ横並び（6等分）を強制する。
st.markdown("""
<style>
[class*="st-key-rbtn_grid_"] [data-testid="stHorizontalBlock"]{
    flex-wrap: nowrap !important;
    gap: 4px !important;
}
[class*="st-key-rbtn_grid_"] [data-testid="stColumn"]{
    min-width: 0 !important;
    flex: 1 1 0 !important;
}
[class*="st-key-rbtn_grid_"] [data-testid="stColumn"] button{
    padding-left: 2px !important;
    padding-right: 2px !important;
    min-width: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ── 起動時 master.parquet 自動ダウンロード ───────────────────────────────
# 条件: 存在しない OR 列数が古いバージョン（99列未満）
def _master_needs_update() -> bool:
    if not MASTER_PARQUET.exists():
        return True
    try:
        import pyarrow.parquet as _pq
        _meta = _pq.read_metadata(str(MASTER_PARQUET))
        return _meta.num_columns < 99   # クラス名/クラス_num/天気を追加した版は99列
    except Exception:
        return True

if _master_needs_update() and 'master_dl_attempted' not in st.session_state:
    st.session_state['master_dl_attempted'] = True
    try:
        _url_chk = st.secrets.get("gdrive_master_csv_url", "")
    except Exception:
        _url_chk = ""
    if _url_chk:
        _dl_reason = "初回" if not MASTER_PARQUET.exists() else "データ更新"
        with st.spinner(f"📥 master.csv をダウンロード中（{_dl_reason}・約30秒）..."):
            _dl_ok, _dl_msg = _download_master_from_gdrive()
        if _dl_ok:
            st.success(f"✅ {_dl_msg}")
        else:
            st.warning(f"⚠️ master.csv の自動取得に失敗しました: {_dl_msg}")

# ── モバイル最適化CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
/* ===== フォント ===== */
html, body, [class*="css"], .stMarkdown, .stButton > button,
.stSelectbox, .stRadio, .stDataFrame, .stTextInput,
[data-testid="stMarkdownContainer"] {
    font-family: "Meiryo", "メイリオ", "Hiragino Kaku Gothic ProN",
                 "ヒラギノ角ゴ ProN W3", "Yu Gothic", "游ゴシック",
                 sans-serif !important;
}

/* ===== 共通 ===== */
.block-container { padding-top: 1rem !important; }

/* プルダウン（selectbox）の文字を中央揃え */
[data-testid="stSelectbox"] select,
[data-testid="stSelectbox"] > div > div {
    text-align: center !important;
    text-align-last: center !important;
}

/* ===== スマホ (〜768px) ===== */
@media screen and (max-width: 768px) {

    .block-container {
        padding-left: 0.4rem !important;
        padding-right: 0.4rem !important;
    }
    /* タブを小さく */
    button[data-baseweb="tab"] {
        font-size: 0.68em !important;
        padding: 4px 3px !important;
    }
    /* ボタン全般を小さく */
    .stButton > button {
        font-size: 0.78em !important;
        padding: 3px 2px !important;
        min-height: 2rem !important;
        line-height: 1.2 !important;
    }
    /* テーブル横スクロール */
    [data-testid="stDataFrame"] {
        overflow-x: auto !important;
    }
    /* 見出し縮小 */
    h1 { font-size: 1.2em !important; }
    h2 { font-size: 1.0em !important; }
    h3 { font-size: 0.95em !important; }
}
</style>
""", unsafe_allow_html=True)

st.title("🏇 競馬予想分析ツール")

# ── 前回の予測結果を自動ロード ───────────────────────────────────────
if 'pred_df' not in st.session_state and LAST_PRED_PATH.exists():
    try:
        _auto = pd.read_parquet(LAST_PRED_PATH)
        # 馬番が全NaN = fix前に生成された壊れたparquet → 破棄して再実行を促す
        if '馬番' in _auto.columns and _auto['馬番'].isna().mean() > 0.5:
            LAST_PRED_PATH.unlink(missing_ok=True)
            st.session_state['_stale_parquet'] = True
        else:
            st.session_state['pred_df'] = _auto
            st.session_state['is_upcoming'] = True
            _mtime = LAST_PRED_PATH.stat().st_mtime
            import datetime as _dt
            _ts = _dt.datetime.fromtimestamp(_mtime).strftime('%Y/%m/%d %H:%M')
            st.session_state['_auto_load_ts'] = _ts
    except Exception:
        pass

# セッションにpred_dfがあっても馬番が全NaN（ホットリロード前から残存）なら警告
if 'pred_df' in st.session_state and not st.session_state.get('_stale_parquet'):
    _chk = st.session_state['pred_df']
    if '馬番' in _chk.columns and _chk['馬番'].isna().mean() > 0.5:
        LAST_PRED_PATH.unlink(missing_ok=True)
        del st.session_state['pred_df']
        st.session_state['_stale_parquet'] = True

# ── 管理者/閲覧モード判定 ───────────────────────────────────────────────
# URL に ?admin=<ADMIN_KEY> を付けると管理者モード（全機能）。付けなければ閲覧モード
# （回収率トラッキング・次走狙いタブを非表示）。管理者だけがパラメータ付きURLを使う運用。
# ※コードは公開リポジトリにあるため厳密なアクセス制御ではなく「表示の仕切り」。
#   厳密に限定したい場合は Streamlit ダッシュボードの限定公開かパスワード方式を使うこと。
ADMIN_KEY = "asari-admin"
try:
    _admin_param = st.query_params.get("admin", "")
except Exception:
    try:
        _qp = st.experimental_get_query_params().get("admin", [""])
        _admin_param = _qp[0] if isinstance(_qp, list) else _qp
    except Exception:
        _admin_param = ""
_is_admin = (_admin_param == ADMIN_KEY)

if _is_admin:
    tab1, tab8, tab7, tab4, tab5, tab6 = st.tabs(
        ["📊 レース予測", "🗓 全レース一覧", "✅ 予測精度", "🔍 データベース検索", "📈 回収率トラッキング", "📝 馬ノート"])
else:
    tab1, tab8, tab7, tab4 = st.tabs(["📊 レース予測", "🗓 全レース一覧", "✅ 予測精度", "🔍 データベース検索"])
    tab5 = tab6 = None


# ============================================================
# Tab 1: レース予測
# ============================================================
with tab1:
    st.subheader("レース予測")
    if _is_admin:
        if st.session_state.get('_stale_parquet'):
            st.warning("⚠️ 保存済みの予測データが古いバージョンで生成されたため削除しました。出馬表CSVを再アップロードして「予測実行」を押してください。")

        data_source = st.radio(
            "データソース",
            ["📋 Target出馬表CSV（今週分）", "📁 Target過去5CSV（分析用）", "🌐 netkeibaから取得"],
            horizontal=True,
        )

        # ── データ読み込みパネル ───────────────────────────────────────────
        with st.expander("📂 データ読み込み設定", expanded='pred_df' not in st.session_state):

            if data_source == "📋 Target出馬表CSV（今週分）":
                st.markdown("**Targetソフト → エクスポート → 出馬表** で出力したCSVをアップロードしてください。")
                st.caption("複数日分（例: 0620.csv + 0621.csv）を同時にアップロード可能です。")
                col_up1, col_up2 = st.columns(2)
                with col_up1:
                    uploaded = st.file_uploader(
                        "出馬表CSV（cp932/Shift-JIS）",
                        type=['csv'],
                        accept_multiple_files=True,
                        key='shutuba_upload'
                    )
                with col_up2:
                    uploaded_maesou = st.file_uploader(
                        "前走CSV（オプション・予測精度向上）",
                        type=['csv'],
                        accept_multiple_files=True,
                        key='maesou_upload',
                        help="Targetソフト → エクスポート → 前走 で出力したCSV。アップロードすると前走着順・着差・コーナー位置などが予測に反映されます。"
                    )
                # ── コース区分手動入力（芝レース一括適用）─────────────────────
                with st.expander("🏟 コース区分設定（芝レース・一括適用）", expanded=False):
                    st.caption(
                        "JRAサイト等で今週の開催コース（A/B/C/D）を確認して入力してください。"
                        "　芝レースのみ適用されます（ダートは影響なし）。土日2日間は同じ設定でOKです。"
                    )
                    _course_opts = ['未指定', 'A', 'B', 'C', 'D']
                    _c_cols = st.columns(5)
                    for _ci, _vn in enumerate(VENUE_ORDER):
                        _sk = f'course_type_{_vn}'
                        _c_cols[_ci % 5].selectbox(
                            _vn, _course_opts,
                            key=_sk
                        )

                if uploaded and st.button("🔍 予測実行", type="primary", key="run_shutuba"):
                    if not MODEL_PATH.exists():
                        st.error("モデルが未学習です。train.py を実行してください。")
                    else:
                        with st.spinner("出馬表を読み込んで予測中..."):
                            try:
                                from load_shutuba_target import (
                                    load_shutuba_target,
                                    load_maesou_target,
                                    merge_maesou_into_shutuba,
                                )
                                import importlib, pipeline_target as _pt
                                if 'pipeline_target' not in st.session_state.get('_reloaded_mods', set()):
                                    importlib.reload(_pt)
                                    _rm = st.session_state.get('_reloaded_mods', set())
                                    _rm.add('pipeline_target')
                                    st.session_state['_reloaded_mods'] = _rm
                                predict_both_from_df = _pt.predict_both_from_df
                                frames = []
                                for uf in uploaded:
                                    raw = uf.read()
                                    df_part = load_shutuba_target(
                                        file_bytes=raw,
                                        filename=uf.name
                                    )
                                    df_part['_src_file'] = uf.name
                                    frames.append(df_part)
                                shutuba_df = pd.concat(frames, ignore_index=True)

                                if uploaded_maesou:
                                    mframes = []
                                    for mf in uploaded_maesou:
                                        mframes.append(load_maesou_target(
                                            file_bytes=mf.read(), filename=mf.name))
                                    maesou_df = pd.concat(mframes, ignore_index=True)
                                    shutuba_df = merge_maesou_into_shutuba(shutuba_df, maesou_df)
                                    st.caption(f"前走データ {len(maesou_df)}頭分をマージしました。")

                                # コース区分を手動設定で適用（芝レースのみ）
                                if 'コース区分' not in shutuba_df.columns:
                                    shutuba_df['コース区分'] = pd.Series(pd.NA, index=shutuba_df.index, dtype='object')
                                else:
                                    shutuba_df['コース区分'] = shutuba_df['コース区分'].astype(object)
                                _n_course_applied = 0
                                for _vn2 in VENUE_ORDER:
                                    _ck = st.session_state.get(f'course_type_{_vn2}', '未指定')
                                    if _ck != '未指定':
                                        _vm = shutuba_df['開催'].astype(str).apply(parse_venue) == _vn2
                                        _tm = shutuba_df['芝・ダ'].astype(str).str.startswith('芝')
                                        shutuba_df.loc[_vm & _tm, 'コース区分'] = _ck
                                        _n_course_applied += int((_vm & _tm).sum())
                                if _n_course_applied:
                                    _course_summary = ', '.join(
                                        f'{v}:{st.session_state.get("course_type_" + v, "未指定")}'
                                        for v in VENUE_ORDER
                                        if st.session_state.get('course_type_' + v, '未指定') != '未指定'
                                    )
                                    st.caption(f"🏟 コース区分（{_course_summary}）を芝{_n_course_applied}頭に適用しました。")

                                # 日付ごとに独立して予測する。特徴量の累積集計(_prior_sum等)は
                                # 入力バッチの行順に依存するため、複数日を同時に読み込むと
                                # ある日の予測が他の日の行に汚染される（例: 7/4を足すと7/5の
                                # 本命が入れ替わる）。日付単位で切って予測すれば各日が独立する。
                                if '日付' in shutuba_df.columns and shutuba_df['日付'].astype(str).nunique() > 1:
                                    _dparts = []
                                    for _dkey, _dgrp in shutuba_df.groupby(
                                            shutuba_df['日付'].astype(str), sort=True):
                                        _dparts.append(predict_both_from_df(_dgrp.copy()))
                                    pred_df = pd.concat(_dparts, ignore_index=True)
                                else:
                                    pred_df = predict_both_from_df(shutuba_df)
                                st.session_state['pred_df'] = pred_df
                                st.session_state['is_upcoming'] = True
                                LAST_PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
                                pred_df.to_parquet(LAST_PRED_PATH, index=False)
                                st.success(f"{len(pred_df)}頭のデータを読み込みました（{len(uploaded)}ファイル）")
                            except Exception as e:
                                st.error(f"エラー: {e}")
                                import traceback; st.code(traceback.format_exc())

            elif data_source == "📁 Target過去5CSV（分析用）":
                csv_dir = st.text_input(
                    "Targetエクスポート先フォルダ",
                    value=str(INPUT_DIR),
                    help="基本/基本2/タイム/前走/生産データ の5CSVが置かれているフォルダ"
                )
                run_pred = st.button("🔍 予測実行", type="primary", key="run_5csv")
                if run_pred:
                    if not MODEL_PATH.exists():
                        st.error("モデルが未学習です。train.py を実行してください。")
                    else:
                        with st.spinner("CSVを読み込んでいます..."):
                            try:
                                from pipeline_target import load_and_predict
                                pred_df = load_and_predict(csv_dir)
                                st.session_state['pred_df'] = pred_df
                                st.session_state['is_upcoming'] = False
                                st.success(f"{len(pred_df)}頭のデータを読み込みました")
                            except Exception as e:
                                st.error(f"エラー: {e}")
                                import traceback; st.code(traceback.format_exc())
            else:
                from datetime import datetime, timedelta
                today = datetime.today()
                date_candidates = []
                for i in range(10):
                    d = today + timedelta(days=i)
                    if d.weekday() in (5, 6):
                        date_candidates.append(d.strftime('%Y%m%d'))
                if not date_candidates:
                    date_candidates = [(today + timedelta(days=i)).strftime('%Y%m%d') for i in range(3)]

                sel_date = st.selectbox("開催日", date_candidates,
                    format_func=lambda x: f"{x[:4]}/{x[4:6]}/{x[6:]} ({'土' if datetime.strptime(x,'%Y%m%d').weekday()==5 else '日' if datetime.strptime(x,'%Y%m%d').weekday()==6 else ''})")
                fetch_list = st.button("📋 レース一覧を取得")
                if fetch_list:
                    with st.spinner("netkeiba から取得中..."):
                        try:
                            from scrape_shutuba import get_race_list
                            st.session_state['race_list'] = get_race_list(sel_date)
                        except Exception as e:
                            st.error(f"取得エラー: {e}")

                if 'race_list' in st.session_state and st.session_state['race_list']:
                    races = st.session_state['race_list']
                    race_opts = {f"{r['venue']} R{r['r']} {r['race_name']}": r['race_id'] for r in races}
                    sel_race_label = st.selectbox("レースを選択", list(race_opts.keys()))
                    if st.button("🔍 出馬表取得 → 予測", type="primary"):
                        if not MODEL_PATH.exists():
                            st.error("モデルが未学習です。train.py を実行してください。")
                        else:
                            with st.spinner("出馬表取得 → 予測中..."):
                                try:
                                    from scrape_shutuba import get_shutuba
                                    from pipeline_target import predict_both_from_df
                                    shutuba_df = get_shutuba(race_opts[sel_race_label], kaisai_date=sel_date)
                                    if shutuba_df.empty:
                                        st.error("出馬表が取得できませんでした（レース未確定の可能性）")
                                    else:
                                        pred_df = predict_both_from_df(shutuba_df)
                                        # _race_id はfeature pipelineで消えるので再付与
                                        if '_race_id' in shutuba_df.columns:
                                            pred_df['_race_id'] = str(shutuba_df['_race_id'].iloc[0])
                                        # 既存pred_dfとマージ（全R蓄積）
                                        _existing = st.session_state.get('pred_df')
                                        if _existing is not None and not _existing.empty:
                                            _rid_new = pred_df['_race_id'].iloc[0] if '_race_id' in pred_df.columns else None
                                            if _rid_new:
                                                _existing = _existing[_existing.get('_race_id', pd.Series(dtype=str)) != _rid_new] if '_race_id' in _existing.columns else _existing
                                            pred_df = pd.concat([_existing, pred_df], ignore_index=True)
                                        st.session_state['pred_df'] = pred_df
                                        st.session_state['is_upcoming'] = True
                                        LAST_PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
                                        pred_df.to_parquet(LAST_PRED_PATH, index=False)
                                        st.success(f"✅ {len(pred_df)}頭のデータを蓄積しました（{pred_df['Ｒ'].nunique() if 'Ｒ' in pred_df.columns else '?'}R分）")
                                except Exception as e:
                                    st.error(f"エラー: {e}")
                                    import traceback; st.code(traceback.format_exc())

    # ── レース選択ナビゲーション ──────────────────────────────────────
    if 'pred_df' not in st.session_state:
        if _is_admin:
            st.info("上の「データ読み込み設定」からデータを読み込んでください。")
        else:
            st.info("📭 まだ予測が公開されていません。管理者が予測を公開するとここに表示されます。")
        st.stop()

    if '_auto_load_ts' in st.session_state:
        if _is_admin:
            st.caption(f"💾 前回の予測結果を自動ロードしました（保存日時: {st.session_state['_auto_load_ts']}）　新しいCSVを読み込むには上のパネルから予測実行してください。")
        else:
            st.caption(f"💾 予測（更新: {st.session_state['_auto_load_ts']}）")

    pred_df = st.session_state['pred_df'].copy()

    # メタ列を付与
    # 日付を8桁(YYYYMMDD)に正規化。Target CSVは6桁(YYMMDD)で来ることがあり、
    # 単純な zfill(8) だと '260621'→'00260621'(0026年) となって日付選択から漏れていた
    # （その結果、本来の全レースが表示されず紛れ込みの数頭だけ表示される不具合があった）。
    _raw_date = pred_df['日付'].astype(str).str.strip().str.replace(r'\D', '', regex=True)
    pred_df['_date_str']   = _raw_date.where(_raw_date.str.len() != 6, '20' + _raw_date).str.zfill(8)
    pred_df['_year']       = pred_df['_date_str'].str[:4]
    pred_df['_month']      = pred_df['_date_str'].str[4:6].str.lstrip('0')
    pred_df['_day']        = pred_df['_date_str'].str[6:8].str.lstrip('0')
    pred_df['_venue_name'] = pred_df['開催'].astype(str).apply(parse_venue)
    pred_df['_r_num']      = pd.to_numeric(pred_df['Ｒ'], errors='coerce').fillna(0).astype(int)
    pred_df['_race_name']  = pred_df['レース名'].fillna('') if 'レース名' in pred_df.columns else ''

    # ── 全レースを pred_log へ自動記録（管理者のみ・予測ロード時に1回）──
    # 「Tab1で開いたレースだけ」でなく、開かなくても全レースを無条件で記録する。
    # 印はpred_rank/穴rank/人気から算出（信頼度テーブル不要）、オッズは埋め込み済みを利用。
    if _is_admin and not pred_df.empty and 'pred_score' in pred_df.columns:
        try:
            _al_sig = (f"{len(pred_df)}|{','.join(sorted(pred_df['_date_str'].unique()))}"
                       f"|{pd.to_numeric(pred_df['pred_score'], errors='coerce').sum():.1f}")
            if st.session_state.get('_autolog_sig') != _al_sig:
                from reliability import assign_marks as _am_al, build_buy_tickets as _bt_al
                from result_tracker import save_pred_logs_bulk as _spb_al, load_pred_log as _lpl_al
                from scrape_odds import build_race_id as _bri_al
                _lpl_cur = _lpl_al()
                # 「本命が入っている」レースだけスキップする。本命が空の行が残って
                # いる場合は再保存して埋め直す（重複排除は本命ありを優先するため
                # 上書きされる）。
                if not _lpl_cur.empty and 'honmei' in _lpl_cur.columns:
                    _hon_ok = (_lpl_cur['honmei'].astype(str).str.strip()
                               .replace({'nan': '', 'None': '', 'NaN': '', '<NA>': ''}) != '')
                    _exist_al = set(_lpl_cur.loc[_hon_ok, 'race_id'].astype(str))
                else:
                    _exist_al = set()
                _al_items = []
                for (_ald8, _alv, _alr), _alg in pred_df.groupby(['_date_str', '開催', '_r_num']):
                    if not _alr or int(_alr) <= 0:
                        continue
                    try:
                        _alrid = _bri_al(str(_ald8), str(_alv), int(_alr)) or f"{_ald8}_{_alv}_{_alr}"
                    except Exception:
                        _alrid = f"{_ald8}_{_alv}_{_alr}"
                    if str(_alrid) in _exist_al:
                        continue
                    _g = _alg.copy()
                    _g['pred_rank'] = _g['pred_score'].rank(ascending=False, method='min').astype(int)
                    if 'pred_score_anaba' in _g.columns:
                        _g['pred_rank_anaba'] = _g['pred_score_anaba'].rank(ascending=False, method='min').astype(int)
                    _pv = 99
                    for _c in ('人気_live', '人気'):
                        if _c in _g.columns:
                            _s = pd.to_numeric(_g[_c], errors='coerce')
                            if _s.notna().any():
                                _pv = _s.fillna(99).astype(int)
                                break
                    _g['_pop_int'] = _pv
                    _g = _am_al(_g)
                    _rnm = str(_g['_race_name'].iloc[0]) if '_race_name' in _g.columns else ''
                    _al_items.append((_alrid, str(_ald8), parse_venue(str(_alv)),
                                      int(_alr), _rnm, _g, _bt_al(_g)))
                st.session_state['_autolog_last'] = _spb_al(_al_items) if _al_items else 0
                st.session_state['_autolog_skip'] = len(_exist_al)
                st.session_state['_autolog_sig'] = _al_sig
                st.session_state.pop('_autolog_err', None)
        except Exception as _al_err:
            # 以前は握りつぶしていたため、記録に失敗しても画面に何も出ず
            # 「反映されない」原因が分からなかった。理由を表示する。
            import traceback as _tb_al
            st.session_state['_autolog_err'] = f"{type(_al_err).__name__}: {_al_err}"
            st.session_state['_autolog_trace'] = _tb_al.format_exc()[-1500:]
    if _is_admin and st.session_state.get('_autolog_err'):
        st.warning(f"⚠️ 予測ログの自動記録に失敗しました → {st.session_state['_autolog_err']}")
        with st.expander("詳細（エラー内容）"):
            st.code(st.session_state.get('_autolog_trace', ''))
    elif _is_admin and st.session_state.get('_autolog_sig'):
        _n_new = st.session_state.get('_autolog_last', 0)
        _n_skip = st.session_state.get('_autolog_skip', 0)
        st.caption(f"🧾 表示中の全レースを予測ログへ自動記録しました"
                   f"（今回の追加 {_n_new}R ／ 記録済みのためスキップ {_n_skip}R）。"
                   "回収率トラッキングで集計できます。")

    from datetime import datetime as _dt
    from collections import defaultdict as _ddict
    _DOW = ['月','火','水','木','金','土','日']

    # 全開催日をdatetimeに変換してリスト化（正規化済み8桁列を使用）
    _all_dates_raw = pred_df['_date_str'].str.extract(r'^(\d{8})$')[0].dropna().unique()
    _all_dates = []
    for _ds in sorted(_all_dates_raw):
        try:
            _all_dates.append(_dt.strptime(_ds, '%Y%m%d'))
        except ValueError:
            pass
    _all_dates = sorted(set(_all_dates))
    if not _all_dates:
        st.warning("予測データに有効な開催日情報がありません。")
        st.write({
            '原因': '日付列が8桁(YYYYMMDD)で取得できていません',
            'pred_df の日付 生値': pred_df['日付'].astype(str).unique().tolist()[:10],
            'pred_df の日付 桁数': pred_df['日付'].astype(str).str.len().unique().tolist(),
            'pred_df 行数': int(len(pred_df)),
        })
        st.stop()

    # ISO週でグループ化（月〜日）
    _week_groups = _ddict(list)
    for _d in _all_dates:
        _iso = _d.isocalendar()
        _week_groups[(_iso[0], _iso[1])].append(_d)
    _week_keys = sorted(_week_groups.keys())

    # 週インデックス初期化（今日に最も近い週）、範囲外なら再計算
    _today = _dt.today()
    def _best_week_idx():
        return min(range(len(_week_keys)),
                   key=lambda i: min(abs((_d - _today).days) for _d in _week_groups[_week_keys[i]]))
    if ('nav_week_idx' not in st.session_state
            or st.session_state['nav_week_idx'] >= len(_week_keys)):
        st.session_state['nav_week_idx'] = _best_week_idx()

    _wi = max(0, min(st.session_state['nav_week_idx'], len(_week_keys) - 1))
    _cur_dates = sorted(_week_groups[_week_keys[_wi]])

    # 前週/次週ナビ + 週ラベル
    _w0, _w1 = _cur_dates[0], _cur_dates[-1]
    _wlabel = (f"{_w0.month}月{_w0.day}日({_DOW[_w0.weekday()]}) 〜 "
               f"{_w1.month}月{_w1.day}日({_DOW[_w1.weekday()]})")
    _nc1, _nc2, _nc3 = st.columns([1, 5, 1])
    with _nc1:
        if _wi > 0 and st.button("◀ 前週", key='nav_prev_week', use_container_width=True):
            st.session_state['nav_week_idx'] = _wi - 1
            st.rerun()
    with _nc2:
        st.markdown(
            f"<div style='text-align:center;color:#ccc;font-size:0.92em;padding-top:6px;'>"
            f"📅 {_wlabel}</div>", unsafe_allow_html=True)
    with _nc3:
        if _wi < len(_week_keys) - 1 and st.button("次週 ▶", key='nav_next_week', use_container_width=True):
            st.session_state['nav_week_idx'] = _wi + 1
            st.rerun()

    # 日付ボタン
    _date_labels = [f"{_d.month}月{_d.day}日({_DOW[_d.weekday()]})" for _d in _cur_dates]
    _label_to_date = dict(zip(_date_labels, _cur_dates))

    if 'sel_date_label' not in st.session_state or st.session_state['sel_date_label'] not in _date_labels:
        # 今日以降の最初の日、なければ最終日
        _today = _dt.today().date()
        _future = [_d for _d in _cur_dates if _d.date() >= _today]
        _init_date = _future[0] if _future else _cur_dates[-1]
        st.session_state['sel_date_label'] = f"{_init_date.month}月{_init_date.day}日({_DOW[_init_date.weekday()]})"

    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
    _dcols = st.columns(len(_date_labels))
    for _col, _lbl in zip(_dcols, _date_labels):
        _is_sel = st.session_state['sel_date_label'] == _lbl
        if _col.button(_lbl, key=f'nav_date_{_lbl}',
                       type='primary' if _is_sel else 'secondary',
                       use_container_width=True):
            st.session_state['sel_date_label'] = _lbl
            st.rerun()

    _sel_date = _label_to_date[st.session_state['sel_date_label']]
    df_d = pred_df[
        (pred_df['_year']  == str(_sel_date.year)) &
        (pred_df['_month'] == str(_sel_date.month)) &
        (pred_df['_day']   == str(_sel_date.day))
    ]

    # ④ 競馬場（開催ごとにタブ表示）
    venues_in_month = [v for v in VENUE_ORDER if v in df_d['_venue_name'].unique()]
    if not venues_in_month:
        st.warning("この日のデータがありません。")
        st.write({
            '選択中の日付': str(_sel_date.date()),
            'この日付の行数 df_d': int(len(df_d)),
            'df_d の _venue_name 一覧': df_d['_venue_name'].astype(str).unique().tolist(),
            'pred_df 全体の開催 生値': sorted(pred_df['開催'].astype(str).unique().tolist())[:15],
            'pred_df 全体の _venue_name': sorted(pred_df['_venue_name'].astype(str).unique().tolist()),
            'pred_df の日付一覧': sorted(pred_df['_date_str'].unique().tolist())[:10],
        })
        st.stop()

    venue_tabs = st.tabs(venues_in_month)

    for v_tab, v_name in zip(venue_tabs, venues_in_month):
        with v_tab:
            df_v = df_d[df_d['_venue_name'] == v_name]

            # ④ 開催回（同一場が複数開催ある場合）
            kaisai_nums = sorted(df_v['開催'].astype(str).str.extract(r'^(\d)')[0].dropna().unique())
            if len(kaisai_nums) > 1:
                sel_kai_label = st.radio(
                    "開催", [f"第{k}回" for k in kaisai_nums],
                    horizontal=True, key=f'kai_{v_name}'
                )
                sel_kai = kaisai_nums[int(sel_kai_label.replace('第','').replace('回','')) - 1]
                df_v = df_v[df_v['開催'].astype(str).str.startswith(sel_kai)]

            # ⑤ レース番号ボタン
            r_nums = sorted(df_v['_r_num'].dropna().unique().astype(int))
            if not r_nums:
                st.info("レースデータがありません。")
                continue

            state_key = f'sel_r_{v_name}'
            if state_key not in st.session_state or st.session_state[state_key] not in r_nums:
                st.session_state[state_key] = r_nums[0]

            # 各Rの芝/ダ情報を事前に取得
            r_surf_map = {}
            for r in r_nums:
                _rdf = df_v[df_v['_r_num'] == r]
                if not _rdf.empty and '芝・ダ' in _rdf.columns:
                    _surf_val = str(_rdf['芝・ダ'].iloc[0])
                    r_surf_map[r] = '芝' if _surf_val.startswith('芝') else 'ダ'
                else:
                    r_surf_map[r] = ''

            # ── レース選択: PC・スマホ共通で 2行×6列ボタングリッド ───────────
            # （スマホでも縦積みにならないよう st-key-rbtn_grid_* のCSSで横並び強制）
            st.markdown("<div style='margin-top:12px;margin-bottom:4px;color:#aaa;font-size:0.85em;'>🏁 レース選択</div>", unsafe_allow_html=True)

            _per_row = 6
            with st.container(key=f'rbtn_grid_{v_name}'):
                for _i in range(0, len(r_nums), _per_row):
                    _row_rs = r_nums[_i:_i + _per_row]
                    r_cols = st.columns(_per_row)
                    for col, r in zip(r_cols, _row_rs):
                        is_sel = st.session_state[state_key] == r
                        _s = r_surf_map.get(r, '')
                        surf_emoji = '🌿' if _s == '芝' else ('🟤' if _s == 'ダ' else '')
                        label = f"{r}R {surf_emoji}" if _s else f"{r}R"
                        if col.button(label, key=f'rbtn_{v_name}_{r}',
                                      type="primary" if is_sel else "secondary",
                                      use_container_width=True):
                            st.session_state[state_key] = r
                            st.rerun()

            # ── 全R一括オッズ取得（メインボタン・目立たせる）────────────────
            if st.button(f"⚡ 全レースのオッズを取得（{len(r_nums)}R分）",
                         key=f'bulk_odds_{v_name}',
                         type="primary", use_container_width=True,
                         help="この競馬場の全レースの単勝オッズをまとめて取得し、EV・買い判定に反映します"):
                from scrape_odds import build_race_id as _brod, fetch_odds_tan as _fot
                from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _asc2

                # _race_id列があれば直接引く（日付の誤抽出を回避）
                _rid_map_bulk = {}
                if '_race_id' in df_v.columns:
                    for _r2 in r_nums:
                        _rdf2 = df_v[df_v['_r_num'] == _r2]
                        if not _rdf2.empty and '_race_id' in _rdf2.columns:
                            _rid_map_bulk[_r2] = str(_rdf2['_race_id'].iloc[0])
                if not _rid_map_bulk:
                    _date0 = str(df_v['日付'].iloc[0]) if not df_v.empty else ''
                    # 日付を8桁(YYYYMMDD)に正規化。6桁(YYMMDD)だと
                    # build_race_id のNetkeiba一覧照合・推定が壊れオッズ取得0件になる
                    _d0 = ''.join(ch for ch in _date0 if ch.isdigit())
                    if len(_d0) == 6:
                        _date0 = '20' + _d0
                    elif len(_d0) == 8:
                        _date0 = _d0
                    _kai0  = str(df_v['開催'].iloc[0]) if not df_v.empty else ''

                def _fetch_odds_r(r):
                    try:
                        if r in _rid_map_bulk:
                            rid = _rid_map_bulk[r]
                        else:
                            rid = _brod(_date0, _kai0, r)
                        return r, _fot(rid) if rid else None
                    except Exception:
                        return r, None

                with st.spinner(f"オッズ取得中... ({len(r_nums)}R)"):
                    with _TPE(max_workers=8) as _ex2:
                        _futs2 = {_ex2.submit(_fetch_odds_r, r): r for r in r_nums}
                        _ok = 0
                        for _fut2 in _asc2(_futs2):
                            _r2, _od = _fut2.result()
                            if _od is not None and not _od.empty:
                                _now = __import__('datetime').datetime.now().strftime('%H:%M:%S')
                                st.session_state[f'live_odds_{v_name}_{_r2}'] = _od
                                st.session_state[f'live_odds_time_{v_name}_{_r2}'] = _now
                                _ok += 1
                # 取得したオッズを永続DF(pred_df)に埋め込み、レース切替後も確実に維持する。
                # （session_state個別キーに加え、DF自身にも持たせることで確実に残す）
                try:
                    _pdf_o = st.session_state.get('pred_df')
                    if _pdf_o is not None and not _pdf_o.empty and '馬番' in _pdf_o.columns \
                            and '_venue_name' in _pdf_o.columns:
                        _pdf_o = _pdf_o.copy()
                        _Zh = str.maketrans('０１２３４５６７８９', '0123456789')
                        _cur_ds = str(df_v['_date_str'].iloc[0]) if '_date_str' in df_v.columns and not df_v.empty else None
                        _pdf_o['_ub_n'] = pd.to_numeric(_pdf_o['馬番'].astype(str).str.translate(_Zh), errors='coerce')
                        for _c in ('単勝オッズ_live', '人気_live'):
                            if _c not in _pdf_o.columns:
                                _pdf_o[_c] = np.nan
                        for _rr in r_nums:
                            _od2 = st.session_state.get(f'live_odds_{v_name}_{_rr}')
                            if _od2 is None or _od2.empty:
                                continue
                            _ubn = pd.to_numeric(_od2['馬番'], errors='coerce')
                            _om = dict(zip(_ubn, pd.to_numeric(_od2['単勝オッズ'], errors='coerce')))
                            _pm = dict(zip(_ubn, pd.to_numeric(_od2['人気'], errors='coerce')))
                            _mask = (_pdf_o['_venue_name'] == v_name) & \
                                    (pd.to_numeric(_pdf_o['_r_num'], errors='coerce') == _rr)
                            if _cur_ds is not None and '_date_str' in _pdf_o.columns:
                                _mask = _mask & (_pdf_o['_date_str'].astype(str) == _cur_ds)
                            _pdf_o.loc[_mask, '単勝オッズ_live'] = _pdf_o.loc[_mask, '_ub_n'].map(_om)
                            _pdf_o.loc[_mask, '人気_live'] = _pdf_o.loc[_mask, '_ub_n'].map(_pm)
                        _pdf_o.drop(columns=['_ub_n'], inplace=True, errors='ignore')
                        st.session_state['pred_df'] = _pdf_o
                except Exception:
                    pass
                st.success(f"✅ {_ok}/{len(r_nums)}R のオッズを取得しました")

            sel_r = st.session_state[state_key]
            show_df = df_v[df_v['_r_num'] == sel_r].copy()

            # 同一馬の重複行を除去（複数ファイル読み込み時など）
            show_df = show_df.drop_duplicates(subset=['馬名'], keep='first')

            # pred_rank を show_df 内で再計算（高スコア=好走なので降順で1位から振り直し）
            if 'pred_score' in show_df.columns and not show_df.empty:
                show_df['pred_rank'] = show_df['pred_score'].rank(ascending=False, method='min').astype(int)
            if 'pred_score_anaba' in show_df.columns and not show_df.empty:
                show_df['pred_rank_anaba'] = show_df['pred_score_anaba'].rank(ascending=False, method='min').astype(int)

            if show_df.empty:
                st.info("データがありません。")
                with st.expander("🔍 診断情報（表示されない原因の特定用）", expanded=True):
                    st.write({
                        '選択中レース sel_r': int(sel_r),
                        'sel_r の型': str(type(sel_r).__name__),
                        'この会場の _r_num 一覧': sorted(pd.to_numeric(df_v['_r_num'], errors='coerce').dropna().astype(int).unique().tolist()),
                        '_r_num の型': str(df_v['_r_num'].dtype),
                        'この会場の行数': int(len(df_v)),
                        'pred_df 総行数': int(len(pred_df)),
                        'pred_df の日付一覧': sorted(pred_df['日付'].astype(str).unique().tolist())[:10],
                        'pred_df の開催一覧': sorted(pred_df['開催'].astype(str).unique().tolist())[:10],
                        '選択中の会場 v_name': str(v_name),
                        '選択中の日付': str(_sel_date.date()),
                    })
                continue

            # ── オッズ入力 ────────────────────────────────────────────
            live_odds_key = f'live_odds_{v_name}_{sel_r}'

            # 単発の「自動取得」「リセット」ボタンは廃止（上の全R一括取得に集約）。
            # オッズ取得済みの時刻表示のみ残す。
            _ot = st.session_state.get(f'live_odds_time_{v_name}_{sel_r}')
            if _ot:
                st.caption(f"📡 オッズ取得済 {_ot}")

            show_top_n = 3  # 上位3頭をハイライト（固定）

            from pred_utils import (softmax_probs, estimate_fuku_probs,
                                     calc_ev, calc_ev_live, implied_odds,
                                     confidence_score, recommend_bet,
                                     get_reasons, POPULAR_STATS)

            # ── ライブオッズを show_df にマージ ──────────────────────
            # 優先度1: parquetに埋め込み済みの live オッズ列
            # 優先度2: session_state（手動オッズ取得ボタン）
            has_live_odds = False
            if '単勝オッズ_live' in show_df.columns and show_df['単勝オッズ_live'].notna().any():
                has_live_odds = True
            else:
                _live_odds = st.session_state.get(live_odds_key)
                if _live_odds is not None and not _live_odds.empty:
                    _lo = _live_odds.copy()
                    _lo['馬番'] = pd.to_numeric(_lo['馬番'], errors='coerce')
                    if '馬番' not in show_df.columns or show_df['馬番'].isna().all():
                        st.warning("⚠️ 馬番データがありません。出馬表CSVを再アップロードして「予測実行」を押してください（以前の予測データは古いバージョンで生成されています）。")
                    else:
                        _ZEN2HAN = str.maketrans('０１２３４５６７８９', '0123456789')
                        show_df['馬番'] = pd.to_numeric(
                            show_df['馬番'].astype(str).str.translate(_ZEN2HAN),
                            errors='coerce'
                        )
                        # pred_dfに埋め込んだ(全NaNの)live列があるとmergeで列名衝突するため除去
                        show_df = show_df.drop(columns=['単勝オッズ_live', '人気_live'], errors='ignore')
                        show_df = show_df.merge(
                            _lo[['馬番', '単勝オッズ', '人気']].rename(
                                columns={'単勝オッズ': '単勝オッズ_live', '人気': '人気_live'}),
                            on='馬番', how='left'
                        )
                        # 単勝オッズがあれば has_live_odds=True。人気_live が空（出馬表段階等）でも
                        # 単勝オッズの順位で人気を補完する（人気=オッズ昇順の順位）。
                        _ol = pd.to_numeric(show_df.get('単勝オッズ_live'), errors='coerce')
                        if _ol is not None and _ol.notna().any():
                            _pl = pd.to_numeric(show_df.get('人気_live'), errors='coerce')
                            if _pl is None or _pl.notna().sum() == 0 or (_pl.fillna(0) <= 0).all():
                                show_df['人気_live'] = _ol.rank(method='min').astype('Int64')
                            has_live_odds = True
                        else:
                            show_df = show_df.drop(columns=['単勝オッズ_live', '人気_live'], errors='ignore')

            # ── EV計算（show_df が空でなければ常に実行）────────────────
            if not show_df.empty:
                probs = softmax_probs(show_df['pred_score'])
                show_df = show_df.copy()
                show_df['_win_prob'] = probs.values
                # 複勝確率: モデル勝率を isotonic 較正した実複勝%（OOS ECE 1.45pt）。
                # 未生成時は estimate_fuku_probs 相当(3×勝率clip)にフォールバック。
                try:
                    from fuku_calibration import calibrated_fuku_prob
                    show_df['_fuku_prob'] = calibrated_fuku_prob(probs).values
                except Exception:
                    show_df['_fuku_prob'] = estimate_fuku_probs(probs).values
                # 推定単勝倍率 (= 1 / 勝利確率)
                show_df['推定単勝倍率'] = show_df['_win_prob'].apply(implied_odds)

                if has_live_odds:
                    # 実人気で _pop_int を更新（範囲: 1〜18）
                    show_df['_pop_int'] = pd.to_numeric(
                        show_df['人気_live'], errors='coerce'
                    ).fillna(
                        pd.to_numeric(show_df['人気'], errors='coerce').fillna(9)
                    ).clip(1, 18).astype(int)
                    # 単勝EV: 実オッズ使用（model_prob × actual_odds × 100 - 100）
                    show_df['EV単勝'] = show_df.apply(
                        lambda r: calc_ev_live(r['_win_prob'],
                                               float(r['単勝オッズ_live'])
                                               if pd.notna(r.get('単勝オッズ_live')) else 0.0),
                        axis=1)
                    # 複勝EV: 複勝オッズ未取得のため POPULAR_STATS ベース推定
                    show_df['EV複勝'] = show_df.apply(
                        lambda r: calc_ev(r['_fuku_prob'],
                                          int(np.clip(r['_pop_int'], 1, 18)), 'fuku'),
                        axis=1)
                    _ev_source = '実オッズ'
                else:
                    show_df['_pop_int'] = pd.to_numeric(
                        show_df['人気'], errors='coerce'
                    ).fillna(9).clip(1, 18).astype(int)
                    show_df['EV単勝'] = show_df.apply(
                        lambda r: calc_ev(r['_win_prob'],
                                          int(np.clip(r['_pop_int'], 1, 18)), 'tan'),
                        axis=1)
                    show_df['EV複勝'] = show_df.apply(
                        lambda r: calc_ev(r['_fuku_prob'],
                                          int(np.clip(r['_pop_int'], 1, 18)), 'fuku'),
                        axis=1)
                    _ev_source = '推定（人気別平均配当）'

                show_df['人気乖離'] = show_df['_pop_int'] - show_df['pred_rank'].fillna(99)

            # ── 本命スコア・印・買い目 計算 ──────────────────────────────
            _honmei_info = {}
            _buy_tickets = {}
            if not show_df.empty and '_win_prob' in show_df.columns:
                try:
                    import importlib, reliability as _rel
                    importlib.reload(_rel)
                    from reliability import (calc_honmei_score, honmei_summary,
                                             assign_marks, build_buy_tickets,
                                             RELIABILITY_CACHE, MARK_COLORS)
                    if not RELIABILITY_CACHE.exists() and 'reliability_built' not in st.session_state:
                        with st.spinner("信頼度テーブルを初回構築中（1回のみ・約30秒）..."):
                            from reliability import rebuild_reliability_cache
                            rebuild_reliability_cache()
                        st.session_state['reliability_built'] = True
                    show_df = calc_honmei_score(show_df, show_df.iloc[0])
                    show_df = assign_marks(show_df)
                    _honmei_info = honmei_summary(show_df)
                    _buy_tickets = build_buy_tickets(show_df)
                    # ※ 予測ログへの記録はここでは行わない。
                    #    予測ロード時に「全レースを一括記録」する処理（上部の自動記録）が
                    #    表示中の全レースを漏れなく保存するため、ここで1レースずつ保存すると
                    #    別IDの重複行を作り「開いたレースだけ増える」ように見える原因になる。
                except Exception as _rel_err:
                    st.warning(f"印/買い目計算エラー: {_rel_err}")

            # ── 穴馬判定（厳格条件: 最大2頭）────────────────────────────
            if has_live_odds and '人気_live' in show_df.columns:
                _real_pop = pd.to_numeric(show_df['人気_live'], errors='coerce')
            elif '人気' in show_df.columns:
                _real_pop = pd.to_numeric(show_df['人気'], errors='coerce')
            else:
                _real_pop = pd.Series(dtype=float, index=show_df.index)

            _real_pop_known = _real_pop.notna() & (_real_pop > 0)
            _pred_rank_s    = show_df['pred_rank'].fillna(99).astype(int)
            _has_anaba_col  = 'pred_rank_anaba' in show_df.columns
            _anaba_rank_s   = (show_df['pred_rank_anaba'].fillna(99).astype(int)
                               if _has_anaba_col else pd.Series(99, index=show_df.index))
            _drift_s        = pd.to_numeric(show_df.get('人気乖離', pd.Series(dtype=float)),
                                            errors='coerce').fillna(0)

            # 基本条件: 通常モデル4位以内 AND 人気6番以下（市場が過小評価する妙味馬）
            # ※旧条件(通常1-2位×穴1-3位×人気7+)は検証(192R)で年4回・的中0%・市場比-12.9ptと
            #   機能不全。緩めた本条件は同人気の市場複勝率を+6.4pt上回る(n=84)。穴モデルは
            #   選定に寄与しないためフィルタから除外し、妙味=通常モデルと市場の乖離で判定。
            _cond_base = _real_pop_known & (_real_pop >= 6) & (_pred_rank_s <= 4)
            # 特上条件: 基本 AND 乖離大（人気 − 通常順位 ≥ 5＝モデルと市場の評価が大きく食い違う）
            _cond_tokujou = _cond_base & (_drift_s >= 5)

            # 見落とし防止(recall重視): 特上(乖離大)は最大2頭に絞り、溢れた分と残りの
            # 基本条件該当は全頭「候補」として残す（漏れなく提示）。
            # ※+EV狙いではなく「モデルは評価するが市場が人気薄にした馬」を取捨の材料に
            #   漏れなく出す用途（検証でこの層は同人気の複勝率を+6.4pt上回る）。
            show_df['_is_tokujou'] = _cond_tokujou
            show_df['_drift_tmp'] = _drift_s
            _tq = show_df[show_df['_is_tokujou']].sort_values(['_drift_tmp', 'pred_rank'], ascending=[False, True])
            if len(_tq) > 2:
                show_df.loc[_tq.index[2:], '_is_tokujou'] = False   # 溢れた特上→下で候補に回る
            show_df['_is_anaba'] = _cond_base & ~show_df['_is_tokujou']
            show_df.drop(columns=['_drift_tmp'], inplace=True)

            # 集計
            tokujou_horses = show_df[show_df['_is_tokujou']]['馬名'].tolist()
            anaba_horses   = show_df[show_df['_is_anaba']]['馬名'].tolist()

            # ── コース別ペース傾向（master.csvがある場合）──────────────
            _pace_prof = {}
            _horse_apts = {}
            try:
                if 'pace_analysis' not in st.session_state.get('_reloaded_mods', set()):
                    import importlib, pace_analysis as _pa
                    importlib.reload(_pa)
                    _rm = st.session_state.get('_reloaded_mods', set())
                    _rm.add('pace_analysis')
                    st.session_state['_reloaded_mods'] = _rm
                from pace_analysis import (course_pace_profile, horse_pace_aptitude,
                                           race_name_profile, pace_fit_score)
                from pipeline_target import MASTER_PARQUET as _MCSV, MASTER_CSV as _MCSV2
                _master_exists = _MCSV.exists() or _MCSV2.exists()
                if _master_exists and not show_df.empty:
                    _row0   = show_df.iloc[0]
                    # 芝/ダート判定: 文字列列 → is_turf数値列 → r_surf_map の順でフォールバック
                    _surf_raw2 = str(_row0.get('芝・ダ', ''))
                    if _surf_raw2.startswith('芝'):
                        _is_turf = True
                    elif _surf_raw2.startswith('ダ'):
                        _is_turf = False
                    else:
                        _it_val = _row0.get('is_turf')
                        if pd.notna(_it_val):
                            _is_turf = int(_it_val) == 1
                        else:
                            _is_turf = r_surf_map.get(sel_r, '') == '芝'
                    _dist_v  = int(pd.to_numeric(_row0.get('dist_num', _row0.get('距離', 0)), errors='coerce') or 0)
                    _vmap   = {'東': '東京', '中': '中山', '京': '京都', '阪': '阪神',
                               '名': '中京', '小': '小倉', '新': '新潟', '福': '福島',
                               '函': '函館', '札': '札幌'}
                    _kai = str(_row0.get('開催', ''))
                    import re as _re
                    _vm = _re.search(r'\d+([^\d])', _kai)
                    _vname = _vmap.get(_vm.group(1), '') if _vm else ''
                    if _vname and _dist_v:
                        _pace_prof = course_pace_profile(_vname, _is_turf, _dist_v)

                    # レース名で特殊傾向を取得（G1等）
                    _rname_kw = str(_row0.get('レース名', ''))
                    _race_name_prof = {}
                    if _rname_kw and len(_rname_kw) >= 3:
                        _race_name_prof = race_name_profile(_rname_kw)

                    # 各馬のペース適性（最大10頭まで、重い処理なので制限）
                    for _hname in show_df['馬名'].head(10):
                        _apt = horse_pace_aptitude(str(_hname))
                        if _apt:
                            _horse_apts[str(_hname)] = _apt
            except Exception:
                pass

            # ── レース概要バナー ──────────────────────────────────────
            race_row = show_df.iloc[0] if not show_df.empty else None
            if race_row is not None:
                # 日付 → M/DD 形式
                _date_raw = str(race_row.get('日付', ''))
                try:
                    _m = int(_date_raw[4:6]); _d = int(_date_raw[6:8])
                    r_date = f"{_m}/{_d}"
                except Exception:
                    r_date = _date_raw

                r_venue = v_name  # 既にparse済みの競馬場名
                # 芝/ダート表示（複数ソースからフォールバック）
                _surf_raw = str(race_row.get('芝・ダ', ''))
                if _surf_raw.startswith('芝'):
                    r_surf = '芝'
                elif _surf_raw.startswith('ダ'):
                    r_surf = 'ダート'
                else:
                    # is_turf列（features.pyが生成する0/1）を使う
                    _is_turf_val = race_row.get('is_turf')
                    if pd.notna(_is_turf_val):
                        r_surf = '芝' if int(_is_turf_val) == 1 else 'ダート'
                    else:
                        # r_surf_map（Rボタン用に取得済み）からも参照
                        _sm = r_surf_map.get(sel_r, '')
                        r_surf = '芝' if _sm == '芝' else ('ダート' if _sm == 'ダ' else '')
                r_dist  = race_row.get('dist_num', race_row.get('距離', ''))
                r_baba  = str(race_row.get('馬場状態', ''))
                _heads_raw = race_row.get('horses_num', race_row.get('頭数', ''))
                r_heads = int(_heads_raw) if str(_heads_raw).replace('.','').isdigit() else len(show_df)
                # レース名（NaN・空文字・'nan'を除外）
                _rname_raw = race_row.get('レース名', '')
                r_name = str(_rname_raw) if _rname_raw and str(_rname_raw) not in ('', 'nan') else ''

                conf = confidence_score(show_df)

                # 特上穴馬がいる場合は自信度をブースト（穴馬の存在は+αの価値）
                conf_display = conf
                if tokujou_horses:
                    conf_display = min(conf + 0.15, 1.0)
                elif anaba_horses:
                    conf_display = min(conf + 0.05, 1.0)
                stars = '★' * round(conf_display * 5) + '☆' * (5 - round(conf_display * 5))

                top3_ev = show_df[show_df['pred_rank'] <= 3]['EV単勝'].dropna().tolist()

                # 穴馬を考慮した推奨馬券
                if tokujou_horses:
                    if top3_ev and max(top3_ev) >= 50:
                        rec_bet = f"単勝・複勝（穴★ {tokujou_horses[0]}）"
                    else:
                        rec_bet = f"複勝・ワイド（穴★ {tokujou_horses[0]}）"
                elif anaba_horses:
                    rec_bet = recommend_bet(conf, top3_ev) + f"（穴◎ {anaba_horses[0]}）"
                else:
                    rec_bet = recommend_bet(conf, top3_ev)

                # 穴情報ライン
                anaba_line = ''
                if tokujou_horses:
                    names_str = '・'.join(tokujou_horses)
                    anaba_line = f'<br><span style="color:#f39c12;font-size:0.9em;">🌟 特上穴馬: {names_str}</span>'
                elif anaba_horses:
                    names_str = '・'.join(anaba_horses)
                    anaba_line = f'<br><span style="color:#c39bd3;font-size:0.9em;">💜 穴馬候補: {names_str}</span>'

                # ペース傾向ライン（avg_pciがNaNでも有利脚質データがあれば表示）
                pace_line = ''
                _has_pace = (_pace_prof and
                             (_pace_prof.get('avg_pci') is not None or
                              _pace_prof.get('yuri_style') is not None))
                if _has_pace:
                    _pc   = _pace_prof.get('avg_pci')
                    _pl   = _pace_prof.get('pci_label', '')
                    _pcol = _pace_prof.get('pci_color', '#888')
                    _fwr  = _pace_prof.get('front_win_rate')
                    _awr  = _pace_prof.get('agari_win_rate')
                    _nr   = _pace_prof.get('n_races', 0)
                    _ys   = _pace_prof.get('yuri_style')
                    _yc   = _pace_prof.get('yuri_color', '#aaa')
                    _yuri_html = (f'<span title="このコース・距離で過去好走しやすい脚質（過去10年の傾向）。先行有利なら前目の馬、差し有利なら末脚型が買い。" style="color:{_yc};font-weight:bold;cursor:help;">🏇 有利脚質: {_ys}</span>&nbsp;|&nbsp;'
                                  if _ys else '')
                    _fwr_txt = f"先行勝率{_fwr:.0f}%" if _fwr is not None else ''
                    _awr_txt = f"上り最速勝率{_awr:.0f}%" if _awr is not None else ''
                    _dr = _pace_prof.get('dist_range', '')
                    _dr_txt = f'({_dr}・{_nr}R・過去10年)' if _dr else f'({_nr}R・過去10年)'
                    _pace_txt = (f'<span title="このコースの平均的なペース傾向（PCIベース）。前傾=ハイペースで持続力勝負、後傾=スローで瞬発力勝負。" style="cursor:help;">ペース</span>: <span style="color:{_pcol};font-weight:bold;">{_pl}</span>&nbsp;|&nbsp;'
                                 if _pc is not None else '')
                    _stats_txt = '&nbsp;/&nbsp;'.join(t for t in [_fwr_txt, _awr_txt] if t)
                    pace_line = (
                        f'<div style="margin-top:6px;padding-top:6px;border-top:1px solid #2a2a4e;font-size:0.85em;">'
                        f'{_yuri_html}'
                        f'{_pace_txt}'
                        f'{_stats_txt}&nbsp;'
                        f'<span style="color:#555;">{_dr_txt}</span>'
                        f'</div>'
                    )

                _dist_str = f"{int(r_dist)}m" if str(r_dist).replace('.','').isdigit() else f"{r_dist}m"
                _rname_str = f"　{r_name}" if r_name and r_name not in ('', 'nan') else ''

                # Netkeiba から発走時刻・馬場・天候・レース名を取得（session_stateでキャッシュ）
                # Target CSVアップロード時は _race_id が無いので 日付+開催+R から構築する
                _rid_for_info = ''
                if '_race_id' in show_df.columns and not show_df.empty and pd.notna(show_df['_race_id'].iloc[0]):
                    _rid_for_info = str(show_df['_race_id'].iloc[0])
                else:
                    try:
                        from scrape_odds import build_race_id as _bri
                        _date_s = str(show_df['日付'].iloc[0]) if '日付' in show_df.columns and not show_df.empty else ''
                        if len(_date_s) == 6:
                            _date_s = '20' + _date_s
                        _kaisai_s = str(show_df['開催'].iloc[0]) if '開催' in show_df.columns and not show_df.empty else ''
                        _rid_for_info = _bri(_date_s, _kaisai_s, int(sel_r)) or ''
                    except Exception:
                        _rid_for_info = ''

                _net_info = {}
                if _rid_for_info:
                    _ri_key = f'race_info_{_rid_for_info}'
                    if _ri_key not in st.session_state:
                        try:
                            from scrape_odds import fetch_race_info as _fri
                            st.session_state[_ri_key] = _fri(_rid_for_info)
                        except Exception:
                            st.session_state[_ri_key] = {}
                    _net_info = st.session_state.get(_ri_key, {})

                _start_time = _net_info.get('time', '')
                _net_baba   = _net_info.get('baba', '')
                _net_tenki  = _net_info.get('tenki', '')
                # レース名: CSV優先、無ければNetkeibaから補完
                _net_name = _net_info.get('name', '')
                if not r_name and _net_name:
                    r_name = _net_name
                    _rname_str = f"　{r_name}"

                # コース区分（A/B/C/D）
                _course_disp = race_row.get('コース区分', '')
                if not _course_disp or str(_course_disp) in ('', 'nan', 'None'):
                    _course_disp = st.session_state.get(f'course_type_{v_name}', '未指定')
                    if _course_disp == '未指定':
                        _course_disp = ''

                # サブ行: 発走時刻 / 芝距離 / 頭数 / コース / 天候 / 馬場
                _sub_parts = []
                if _start_time:
                    _sub_parts.append(f'<span style="color:#fff;">{_start_time}発走</span>')
                _sub_parts.append(f'{r_surf}{_dist_str}')
                _sub_parts.append(f'{r_heads}頭')
                if _course_disp and r_surf == '芝':
                    _sub_parts.append(f'<span style="color:#7eb8f7;">{_course_disp}コース</span>')
                if _net_tenki:
                    _sub_parts.append(f'天候:{_net_tenki}')
                if _net_baba:
                    _sub_parts.append(f'馬場:{_net_baba}')
                elif r_baba and r_baba not in ('', 'nan'):
                    _sub_parts.append(f'馬場:{r_baba}')
                _sub_line = ' / '.join(_sub_parts)

                _bottom_line_parts = []
                if tokujou_horses:
                    _bottom_line_parts.append(f'<span style="color:#f39c12;">🌟 特上穴馬: {"・".join(tokujou_horses)}</span>')
                elif anaba_horses:
                    _bottom_line_parts.append(f'<span style="color:#c39bd3;">💜 穴馬候補: {"・".join(anaba_horses)}</span>')
                _bottom_line_parts.append(
                    f'<span style="background:#2980b9;padding:2px 10px;border-radius:4px;font-size:0.9em;">💡推奨: {rec_bet}</span>'
                )
                _bottom_line = '&nbsp;&nbsp;'.join(_bottom_line_parts)

                # 💧 道悪適性ライン（道悪◎/○の馬を列挙。当日が重・不良なら強調）
                doaku_line = ''
                if 'doaku_score' in show_df.columns and show_df['doaku_score'].notna().any():
                    _dk = show_df[['馬名', 'doaku_score']].dropna(subset=['doaku_score'])
                    _maru = _dk[_dk['doaku_score'] >= 0.42]['馬名'].astype(str).tolist()
                    _wa = _dk[(_dk['doaku_score'] >= 0.34) & (_dk['doaku_score'] < 0.42)]['馬名'].astype(str).tolist()
                    if _maru or _wa:
                        _wet_now = any(x in str(_net_baba) for x in ('重', '不良'))
                        _emph = '<span style="color:#e67e22;">（当日道悪！）</span>' if _wet_now else ''
                        _pp = []
                        if _maru:
                            _pp.append(f'<span style="color:#5dade2;font-weight:bold;">道悪◎ {"・".join(_maru)}</span>')
                        if _wa:
                            _pp.append(f'<span style="color:#85c1e9;">道悪○ {"・".join(_wa)}</span>')
                        doaku_line = (
                            '<div style="margin-top:6px;padding-top:6px;border-top:1px solid #2a2a4e;font-size:0.85em;">'
                            '<span title="重・不良馬場での複勝率（標本が少ない馬は血統の道悪適性で補正）。'
                            '◎=道悪で明確に好走／○=やや得意。当日が重・不良なら特に注目。" style="cursor:help;">'
                            f'💧 道悪適性</span>{_emph}: ' + '　'.join(_pp) + '</div>'
                        )

                # 🐴 馬ノート（有効なメモがこのレースの出走馬にあれば表示）
                watch_line = ''
                try:
                    import race_notes as _whp
                    if '馬名' in show_df.columns:
                        _race_names = list(show_df['馬名'].astype(str).map(_whp.normalize_name))
                        _wnames = list(set(_race_names))
                        try:
                            _wm = pd.read_parquet(MASTER_PARQUET, columns=['馬名', '日付_dt'],
                                                  filters=[('馬名', 'in', _wnames)])
                            _wm['日付_dt'] = pd.to_datetime(_wm['日付_dt'], errors='coerce')
                            _wlast = _wm.groupby(_wm['馬名'].map(_whp.normalize_name))['日付_dt'].max().to_dict()
                        except Exception:
                            _wlast = {}
                        _hit = _whp.active_notes_for_horses(_race_names, _wlast)
                        if _hit:
                            _EVC = {'次走注目': '#f1c40f', '危険(過剰人気警戒)': '#e74c3c',
                                    '度外視': '#3498db', '中立': '#c9d1d9'}
                            _lines = []
                            for _nm, _nt in _hit.items():
                                _ev = str(_nt.get('評価', '中立') or '中立')
                                _aim = int(pd.to_numeric(_nt.get('狙い度', 2), errors='coerce') or 2)
                                _tg = str(_nt.get('タグ', '') or '')
                                _detail = ' / '.join(x for x in [_tg, str(_nt.get('メモ', '') or '')] if x)
                                _lines.append(
                                    f'<b>{_nm}</b> '
                                    f'<span style="color:{_EVC.get(_ev, "#c9d1d9")};">[{_ev}{"★" * _aim}]</span>'
                                    + (f' <span style="color:#adbac7;">{_detail}</span>' if _detail else ''))
                            watch_line = (
                                '<div style="margin-top:6px;padding-top:6px;border-top:1px solid #2a2a4e;font-size:0.86em;">'
                                '🐴 <span title="馬ノート（結果回顧で記録した独自メモ）。次走を迎えると自動で消化されます。" '
                                'style="color:#f1c40f;font-weight:bold;cursor:help;">馬ノート</span>: '
                                + '　'.join(_lines) + '</div>'
                            )
                except Exception:
                    watch_line = ''

                st.markdown(f"""
<div style="background:#1a1a2e;border-radius:10px;padding:14px 20px;margin-bottom:12px;color:white;">
<div style="font-size:1.15em;font-weight:bold;">
  {r_venue}{sel_r}R{_rname_str}
</div>
<div style="color:#aaa;font-size:0.88em;margin-top:3px;">{_sub_line}</div>
<div style="color:#f1c40f;margin-top:6px;"><span title="本命◎の信頼度（レース質×人気帯ごとの過去の複勝率から算出）。★が多いほど本命が堅い。" style="cursor:help;">自信度</span>: {stars}</div>
<div style="margin-top:8px;padding-top:8px;border-top:1px solid #2a2a4e;">
{_build_honmei_lines(_honmei_info)}
</div>
<div style="margin-top:8px;padding-top:6px;border-top:1px solid #2a2a4e;font-size:0.9em;">
  {_bottom_line}
</div>
{pace_line}
{doaku_line}
{watch_line}
</div>
""", unsafe_allow_html=True)

            # ── 当該コースの好調データ（騎手/種牡馬/調教師 勝率TOP・出走メンバー限定）──
            # ※機械的な買い目表示は廃止し、出走メンバー内の当該コース好調データに差し替え。
            _cperf = {}
            try:
                import re as _re_cs
                from course_stats import course_top_performers
                _cv = str(show_df['開催'].iloc[0]) if '開催' in show_df.columns else ''
                _cvm = _re_cs.search(r'\d+([^\d])', _cv)
                _cvenue = _cvm.group(1) if _cvm else ''
                _cturf = int(pd.to_numeric(show_df.get('is_turf', pd.Series([1])).iloc[0], errors='coerce') or 0) == 1
                _cdist = int(pd.to_numeric(show_df.get('dist_num', show_df.get('距離', pd.Series([0]))).iloc[0], errors='coerce') or 0)
                if _cdist > 0:
                    _cperf = course_top_performers(show_df, _cvenue, _cturf, _cdist)
            except Exception:
                _cperf = {}
            if _cperf and any(_cperf.get(k) for k in ('騎手', '種牡馬', '調教師')):
                def _cs_block(cat, icon, color):
                    items = _cperf.get(cat, [])
                    if not items:
                        inner = '<div style="color:#6e7681;font-size:0.85em;padding:4px 0;">該当データなし</div>'
                    else:
                        inner = ''
                        for x in items:
                            _h = x['horses'][0] if x.get('horses') else ''
                            inner += (
                                f'<div style="margin:7px 0;line-height:1.3;">'
                                f'<span style="color:#e6edf3;font-size:0.98em;font-weight:600;">{x["name"]}</span>'
                                f'<span style="margin-left:6px;"><b style="color:{color};font-size:1.1em;">{x["rate"]*100:.0f}%</b>'
                                f'<span style="color:#6e7681;font-size:0.8em;">（{x["n"]}）</span></span><br>'
                                f'<span style="color:#8b949e;font-size:0.82em;">→ {_h}</span></div>')
                    return (
                        f'<div style="flex:1 1 170px;min-width:150px;background:#161b22;'
                        f'border-radius:8px;padding:10px 12px;">'
                        f'<div style="color:{color};font-weight:bold;font-size:0.98em;margin-bottom:6px;'
                        f'border-bottom:1px solid {color}44;padding-bottom:5px;">{icon} {cat}</div>'
                        f'{inner}</div>')
                st.markdown(
                    f'<div style="background:#0d1117;border:1px solid #30363d;border-radius:10px;'
                    f'padding:12px 16px;margin-bottom:12px;">'
                    f'<div style="color:#f1c40f;font-weight:bold;font-size:1.02em;margin-bottom:3px;" '
                    f'title="このコース(場×芝ダ×距離)で過去勝率が高い、出走中の騎手・種牡馬・調教師のTOP3。">'
                    f'\U0001F3C6 このコースの好調データ（{_cperf.get("course_label", "")}・出走メンバー内）'
                    f'<span style="color:#8b949e;font-size:0.75em;cursor:help;">　ⓘ</span></div>'
                    f'<div style="color:#6e7681;font-size:0.76em;margin-bottom:9px;">'
                    f'％＝このコースでの勝率　／　（数字）＝騎乗・出走数（標本）</div>'
                    f'<div style="display:flex;gap:10px;flex-wrap:wrap;align-items:stretch;">'
                    + _cs_block('騎手', '\U0001F3C7', '#58a6ff')
                    + _cs_block('種牡馬', '\U0001F9EC', '#2ecc71')
                    + _cs_block('調教師', '\U0001F3EB', '#c39bd3')
                    + '</div></div>',
                    unsafe_allow_html=True)

            # ── 穴馬推奨セクション（最大2頭、一目でわかるバナー）────────
            _anaba_rec_rows = show_df[show_df['_is_tokujou'] | show_df['_is_anaba']].sort_values(
                ['_is_tokujou', 'pred_rank'], ascending=[False, True])

            if not _anaba_rec_rows.empty:
                _ab_cards = []
                for _, _abr in _anaba_rec_rows.iterrows():
                    _ab_name   = str(_abr.get('馬名', ''))
                    _ab_pop    = int(_abr.get('_pop_int', 0))
                    _ab_odds   = _abr.get('単勝オッズ_live', None)
                    _ab_ev     = _abr.get('EV単勝', float('nan'))
                    _ab_drift  = int(_abr.get('人気乖離', 0))
                    _ab_rank   = int(_abr.get('pred_rank', 99))
                    _ab_arank  = int(_abr.get('pred_rank_anaba', 99)) if 'pred_rank_anaba' in show_df.columns else None
                    _ab_tokujou = bool(_abr.get('_is_tokujou', False))
                    _ab_pop_str = f'{_ab_pop}番人気' if _ab_pop > 0 else '人気不明'
                    _ab_odds_str = f'{float(_ab_odds):.1f}倍' if pd.notna(_ab_odds) else '--'
                    _ab_ev_str  = f'EV{_ab_ev:+.0f}%' if pd.notna(_ab_ev) else 'EV---'
                    _ab_ev_col  = '#2ecc71' if pd.notna(_ab_ev) and _ab_ev >= 0 else '#e74c3c'
                    _ab_arank_str = f' / 穴モデル{_ab_arank}位' if _ab_arank is not None else ''
                    _ab_fp = _abr.get('_fuku_prob', float('nan'))
                    _ab_fp_str = f'予想複勝{float(_ab_fp)*100:.0f}%' if pd.notna(_ab_fp) else ''
                    if _ab_tokujou:
                        _border = '#f39c12'
                        _badge  = '🌟 特上穴馬'
                        _badge_bg = '#7d4e00'
                        _badge_color = '#f39c12'
                    else:
                        _border = '#8e44ad'
                        _badge  = '💜 穴馬推奨'
                        _badge_bg = '#3d1a5c'
                        _badge_color = '#c39bd3'
                    _ab_cards.append(f"""
<div style="flex:1;min-width:220px;background:#1a1025;border:2px solid {_border};
     border-radius:10px;padding:12px 16px;">
  <div style="background:{_badge_bg};color:{_badge_color};font-weight:bold;
       font-size:0.9em;padding:2px 10px;border-radius:4px;display:inline-block;
       margin-bottom:8px;">{_badge}</div>
  <div style="font-size:1.3em;font-weight:bold;color:white;margin-bottom:4px;">{_ab_name}</div>
  <div style="color:#aaa;font-size:0.88em;">
    {_ab_pop_str} &nbsp;|&nbsp; {_ab_odds_str} &nbsp;|&nbsp;
    <span style="color:{_ab_ev_col};">{_ab_ev_str}</span>
  </div>
  <div style="color:#888;font-size:0.8em;margin-top:4px;">
    通常モデル{_ab_rank}位{_ab_arank_str} &nbsp;|&nbsp; 人気乖離+{_ab_drift}
    {(' &nbsp;|&nbsp; <span style="color:#c39bd3;">' + _ab_fp_str + '</span>') if _ab_fp_str else ''}
  </div>
</div>""")
                st.markdown(f"""
<div style="margin-bottom:6px;">
  <div style="color:#f39c12;font-weight:bold;font-size:1.05em;margin-bottom:8px;"
       title="通常モデルが上位(4位以内)に評価しているのに市場が人気薄(6番人気以下)にしている馬。買い推奨(+EV)ではなく、取捨で見落としやすい妙味馬を漏れなく提示するための注意リスト。検証でこの層は同人気の市場複勝率を+6.4pt上回る（ただし控除率の壁で単体の回収率は100%未満）。">
    🎯 見落とし注意リスト（通常モデル上位×人気薄）<span style="color:#8b949e;font-size:0.72em;cursor:help;">　ⓘ</span>
  </div>
  <div style="display:flex;gap:12px;flex-wrap:wrap;">
    {''.join(_ab_cards)}
  </div>
</div>
""", unsafe_allow_html=True)
                st.caption('※買い推奨（+EV）ではなく、取捨で見落としやすい「モデルは評価するが人気薄」の馬を漏れなく提示する参考リストです。')
            else:
                if _real_pop_known.any():
                    st.markdown(
                        '<div style="color:#555;font-size:0.85em;margin-bottom:10px;">'
                        '🔍 このレースに穴馬候補なし（条件: 通常モデル4位以内 ＋ 6番人気以下）'
                        '</div>', unsafe_allow_html=True)

            # 穴馬モデルランクが使えるか
            has_anaba = 'pred_rank_anaba' in show_df.columns

            # ── 馬ごとのカード表示 ────────────────────────────────────
            # 枠番カラー（JRA公式）
            _WAKU_COLORS = {
                1: ('#FFFFFF', '#000000'),  # 白地・黒文字
                2: ('#000000', '#FFFFFF'),  # 黒地・白文字
                3: ('#EE0000', '#FFFFFF'),  # 赤地・白文字
                4: ('#0066CC', '#FFFFFF'),  # 青地・白文字
                5: ('#FFD700', '#000000'),  # 黄地・黒文字
                6: ('#008000', '#FFFFFF'),  # 緑地・白文字
                7: ('#FF8C00', '#000000'),  # オレンジ地・黒文字
                8: ('#FF69B4', '#000000'),  # ピンク地・黒文字
            }

            def _umaban_to_waku(ub, n):
                """馬番 ub → 枠番(1-8)。JRA方式: 8頭以下は馬番=枠番。9頭以上は外枠
                (大きい枠)から2頭ずつ詰める（17頭で8枠3頭、18頭で7・8枠3頭）。"""
                n = max(1, int(n))
                if n <= 8:
                    counts = [1] * n + [0] * (8 - n)
                elif n <= 16:
                    ones = 16 - n                       # 内側の1頭枠の数
                    counts = [1] * ones + [2] * (8 - ones)
                else:                                   # 17,18頭
                    threes = n - 16                     # 外側の3頭枠の数
                    counts = [2] * (8 - threes) + [3] * threes
                cum = 0
                for frame in range(1, 9):
                    cum += counts[frame - 1]
                    if ub <= cum:
                        return frame
                return 8

            def _waku_html(umaban_val, n_horses):
                try:
                    ub = int(umaban_val)
                except (TypeError, ValueError):
                    return ''
                waku = _umaban_to_waku(ub, n_horses) if n_horses else min((ub + 1) // 2, 8)
                bg_c, fg_c = _WAKU_COLORS.get(waku, ('#555', '#fff'))
                return (
                    f'<span style="display:inline-block;background:{bg_c};color:{fg_c};'
                    f'border-radius:50%;width:24px;height:24px;line-height:24px;'
                    f'text-align:center;font-size:0.82em;font-weight:bold;'
                    f'border:1px solid #444;margin-right:2px;">{ub}</span>'
                )

            show_df_sorted = show_df.sort_values('pred_rank')
            # 馬カードの表示密度: 既定は PC=詳細 / スマホ=圧縮。トグルで切替可。
            _detail_cards = st.toggle("🔍 詳細表示（EV・全タグ）", value=(not _is_mobile),
                                      key=f'detail_{v_name}')
            # 枠番算出に使う頭数（最大馬番＝出走頭数。欠場で穴があっても枠は不変）
            _n_horses = (int(pd.to_numeric(show_df_sorted['馬番'], errors='coerce').max())
                         if '馬番' in show_df_sorted.columns and show_df_sorted['馬番'].notna().any()
                         else 0)

            for _, row in show_df_sorted.iterrows():
                rank       = int(row.get('pred_rank', 99))
                rank_anaba = int(row.get('pred_rank_anaba', 99)) if has_anaba else None
                pop        = int(row.get('_pop_int', 99))
                name       = str(row.get('馬名', ''))
                umaban_raw = row.get('馬番', None)
                umaban_html = _waku_html(umaban_raw, _n_horses)
                jock       = str(row.get('騎手', ''))
                ev_t       = row.get('EV単勝', float('nan'))
                ev_f       = row.get('EV複勝', float('nan'))
                drift      = int(row.get('人気乖離', 0))
                odds_live  = row.get('単勝オッズ_live', None)
                is_tokujou = bool(row.get('_is_tokujou', False))
                is_anaba   = bool(row.get('_is_anaba', False))

                if pd.notna(odds_live):
                    odds_html = (f'<span style="color:#3498db;font-size:0.85em;margin-left:4px;">'
                                 f'📡 {float(odds_live):.1f}倍</span>')
                else:
                    _impl = row.get('推定単勝倍率')
                    odds_html = (f'<span style="color:#888;font-size:0.82em;margin-left:4px;">'
                                 f'推定≈{float(_impl):.1f}倍</span>'
                                 ) if pd.notna(_impl) else ''

                # 印バッジ
                _mark       = str(row.get('_mark', ''))
                _fuku_rate  = row.get('_fuku_rate')
                _confidence = row.get('_confidence')
                # モデルの校正済み複勝%（取捨の参考。市場人気でなくモデル評価ベース）
                _fp = row.get('_fuku_prob')
                if pd.notna(_fp):
                    _fp = float(_fp)
                    _fp_col = '#2ecc71' if _fp >= 0.5 else ('#3498db' if _fp >= 0.3 else ('#aaa' if _fp >= 0.15 else '#777'))
                    _fp_html = (f'<span title="モデルが推定するこの馬の複勝(3着内)確率。レース内スコアを過去実績で較正した値（人気ではなくモデル評価）。取捨の目安。" '
                                f'style="color:{_fp_col};font-weight:bold;font-size:0.85em;cursor:help;">予想複勝{_fp*100:.0f}%</span>')
                else:
                    _fp_html = ''
                try:
                    from reliability import MARK_COLORS
                    _mark_color = MARK_COLORS.get(_mark, '#555')
                except Exception:
                    _mark_color = '#555'
                # 市場ベースの「馬券内%」表示は廃止（モデル評価の「予想複勝%」に一本化）。
                # _fuku_rate は信頼度(_confidence)算出には引き続き内部利用。
                conf_label = ''
                if pd.notna(_confidence):
                    conf_label = '信頼高' if _confidence >= 0.7 else ('信頼中' if _confidence >= 0.4 else '信頼低')
                mark_label = {'◎': '本命', '○': '対抗', '▲': '単穴', '△': '連下', '★': '妙味'}.get(_mark, '')
                _mark_crit = {
                    '◎': '通常モデル1位', '○': '通常モデル2位', '▲': '通常モデル3位',
                    '△': '通常モデル4〜6位', '★': '連下(通常4〜6位)のうち人気6番以下で穴馬モデル最上位＝妙味',
                }.get(_mark, '')
                honmei_html = ''
                if _mark:
                    _arank_t = rank_anaba if rank_anaba is not None else '—'
                    _mark_title = (f"{_mark}{mark_label}：{_mark_crit}"
                                   f" ｜ 通常モデル{rank}位・穴馬モデル{_arank_t}位"
                                   + (f"・予想複勝{float(_fp)*100:.0f}%" if pd.notna(_fp) else '')
                                   + (f"・{conf_label}" if conf_label else ''))
                    honmei_html = (
                        f'<span title="{_mark_title}" style="background:{_mark_color};color:#111;padding:1px 10px;'
                        f'border-radius:4px;font-size:1.0em;font-weight:bold;margin-left:6px;cursor:help;">'
                        f'{_mark}{mark_label}</span>'
                    )

                reasons = get_reasons(row, show_df_sorted, top_n=4)

                # 背景色・ボーダー（特上穴馬は金オレンジ枠）
                if is_tokujou:
                    bg     = '#2d1f00'
                    border = '#f39c12'
                elif is_anaba:
                    bg     = '#1f0d2e'
                    border = '#8e44ad'
                elif rank == 1:
                    bg     = '#1a3a1a'
                    border = '#f1c40f'
                elif rank <= show_top_n:
                    bg     = '#1a2a1a'
                    border = '#555'
                else:
                    bg     = '#1a1a1a'
                    border = '#555'

                # EV色
                ev_color  = '#2ecc71' if pd.notna(ev_t) and ev_t >= 50  else ('#f39c12' if pd.notna(ev_t) and ev_t >= 0 else '#e74c3c')
                fev_color = '#2ecc71' if pd.notna(ev_f) and ev_f >= 50  else ('#f39c12' if pd.notna(ev_f) and ev_f >= 0 else '#e74c3c')
                ev_t_str  = f'{ev_t:+.0f}%' if pd.notna(ev_t) else '---'
                ev_f_str  = f'{ev_f:+.0f}%' if pd.notna(ev_f) else '---'

                # バッジ
                if is_tokujou:
                    anaba_badge = '<span style="background:#f39c12;color:#1a1a1a;padding:1px 8px;border-radius:4px;font-size:0.85em;font-weight:bold;margin-left:6px;">🌟穴★</span>'
                elif is_anaba:
                    anaba_badge = '<span style="background:#8e44ad;color:white;padding:1px 7px;border-radius:4px;font-size:0.8em;margin-left:6px;">◎穴</span>'
                else:
                    anaba_badge = ''

                # 穴馬モデル順位の表示
                anaba_rank_html = ''
                if rank_anaba is not None:
                    color = '#c39bd3' if rank_anaba <= 3 else '#666'
                    anaba_rank_html = f'<span style="color:{color};font-size:0.82em;margin-left:6px;">穴モデル:{rank_anaba}位</span>'

                # 特徴タグ（カテゴリ別色分け）
                _TAG_MAP = {
                    '馬の平均着順が良い':         ('馬◎着順',   '#1a6b3c', '#2ecc71'),
                    '馬の過去勝率が高い':         ('馬◎勝率',   '#1a6b3c', '#2ecc71'),
                    '馬の複勝率が高い':           ('馬◎複勝',   '#1a6b3c', '#2ecc71'),
                    'このコースで好成績':         ('コース◎',   '#1a3d6b', '#58a6ff'),
                    'このコースの勝率が高い':     ('コース勝率', '#1a3d6b', '#58a6ff'),
                    'このコースの複勝率が高い':   ('コース複勝', '#1a3d6b', '#58a6ff'),
                    '脚質がこのコースに合う':     ('脚質適性◎', '#1a3d6b', '#58a6ff'),
                    '先行有利コースで先行脚質':   ('先行◎',     '#1a3d6b', '#58a6ff'),
                    '騎手の勝率が高い':           ('騎手◎',     '#6b3d1a', '#f39c12'),
                    '騎手の平均着順が良い':       ('騎手◎着順', '#6b3d1a', '#f39c12'),
                    '調教師の勝率が高い':         ('調教師◎',   '#6b3d1a', '#f39c12'),
                    '前走上り3Fが速い':           ('前走上り◎', '#4b1a6b', '#c39bd3'),
                    '直近3走の上り平均が速い':    ('上り安定',   '#4b1a6b', '#c39bd3'),
                    '前走着差が少ない（接戦）':   ('前走接戦',   '#4b1a6b', '#c39bd3'),
                    '直近3走の着差が少ない':      ('着差安定',   '#4b1a6b', '#c39bd3'),
                    '前走タイムが優秀':           ('前走タイム◎','#4b1a6b', '#c39bd3'),
                    '前走着順が良い':             ('前走◎',     '#4b1a6b', '#c39bd3'),
                    '直近3走の着順が安定':        ('近走安定',   '#4b1a6b', '#c39bd3'),
                    '前走4角で前目につけた':      ('前走先行',   '#4b1a6b', '#c39bd3'),
                    '距離延長・短縮が合う':       ('距離適性◎', '#3d3d1a', '#d4ac0d'),
                    'モデル総合スコアが上位':     ('総合◎',     '#1a1a3d', '#8888ff'),
                    # 血統(産駒)適性（ティール）
                    '血統が芝/ダに適性':          ('血統芝ダ',   '#0e4b4b', '#48c9b0'),
                    '血統がこの距離に合う':       ('血統距離',   '#0e4b4b', '#48c9b0'),
                    '産駒がこの条件(芝ダ×距離)得意': ('産駒◎',    '#0e4b4b', '#48c9b0'),
                    '血統が道悪巧者':             ('道悪血統',   '#0e4b4b', '#48c9b0'),
                    '母父が芝/ダに適性':          ('母父芝ダ',   '#0e4b4b', '#48c9b0'),
                    '母父がこの距離に合う':       ('母父距離',   '#0e4b4b', '#48c9b0'),
                    '母父産駒がこの条件得意':     ('母父産駒◎', '#0e4b4b', '#48c9b0'),
                    '母父が道悪巧者':             ('母父道悪',   '#0e4b4b', '#48c9b0'),
                    # 騎手の条件別適性（オレンジ）
                    '騎手が芝/ダ得意':            ('騎手芝ダ',   '#6b3d1a', '#f39c12'),
                    '騎手がこの距離得意':         ('騎手距離',   '#6b3d1a', '#f39c12'),
                    # 展開不利の巻き返し（パープル）
                    '前走は展開負け→今回は先行向き': ('展開向く', '#4b1a6b', '#c39bd3'),
                    '前走は展開負け→今回は差し向き': ('展開向く', '#4b1a6b', '#c39bd3'),
                    # 前走大敗の度外視・妙味（ゴールド）
                    '前走大敗を度外視・実績馬（巻き返し妙味）': ('💰実績馬妙味', '#2d1f00', '#f1c40f'),
                }
                # 根拠タグの開発者向けヘルプ（ロジック説明）。get_reasonsは全て
                # 「その馬の値が出走馬平均より優位」で判定される（フィールド内相対比較）。
                _REASON_HELP = {
                    '馬の平均着順が良い': '通算平均着順が出走馬平均より良い (horse_avg_chaku≤平均×0.8)',
                    'このコースで好成績': 'このコース(場×芝ダ×距離帯)の平均着順が良い (course_avg_chaku)',
                    '直近3走の上り平均が速い': '直近3走の上り3F平均が速い (agari_avg3)',
                    '前走上り3Fが速い': '前走の上り3Fが出走馬平均より速い (prev_agari)',
                    '馬の過去勝率が高い': '通算勝率が出走馬平均の1.3倍以上 (horse_win_rate)',
                    'このコースの勝率が高い': 'このコースの勝率が平均の1.5倍以上 (course_win_rate)',
                    '騎手の勝率が高い': '騎手の通算勝率が平均の1.3倍以上 (jockey_win_rate)',
                    '調教師の勝率が高い': '調教師の通算勝率が高い (trainer_win_rate)',
                    '馬の複勝率が高い': '通算複勝率が平均の1.3倍以上 (horse_fuku_rate)',
                    'このコースの複勝率が高い': 'このコースの複勝率が平均の1.5倍以上 (course_fuku_rate)',
                    '直近3走の着順が安定': '直近3走の平均着順が良い (chaku_avg3)',
                    '直近3走の着差が少ない': '直近3走の平均着差が小さい (chakusa_avg3)',
                    '前走着差が少ない（接戦）': '前走の着差タイムが小さい=接戦 (prev_chakusa_t)',
                    '前走着順が良い': '前走着順が出走馬平均より上位 (prev_chakujun)',
                    '前走4角で前目につけた': '前走の4角通過順が前 (prev_pos_4c)',
                    '脚質がこのコースに合う': '脚質とコース先行有利度が適合 (style_course_fit)',
                    '先行有利コースで先行脚質': '先行有利コース×先行脚質 (pace_front_ratio)',
                    '産駒がこの条件(芝ダ×距離)得意': '種牡馬産駒のこの芝ダ×距離帯の複勝率が高い (sire_sd_fuku)',
                    '母父産駒がこの条件得意': '母父産駒のこの条件の複勝率が高い (bms_sd_fuku)',
                    '血統が芝/ダに適性': '種牡馬の芝ダ別複勝率が高い (sire_surf_fuku)',
                    '血統がこの距離に合う': '種牡馬の距離帯別複勝率が高い (sire_dist_fuku)',
                    '母父が芝/ダに適性': '母父の芝ダ別複勝率が高い (bms_surf_fuku)',
                    '母父がこの距離に合う': '母父の距離帯別複勝率が高い (bms_dist_fuku)',
                    '騎手が芝/ダ得意': '騎手の芝ダ別複勝率が高い (jockey_surf_fuku)',
                    '騎手がこの距離得意': '騎手の距離帯別複勝率が高い (jockey_dist_fuku)',
                    '前走は展開負け→今回は先行向き': '前走で差し有利展開に先行し負け→今回先行有利 (senko_revenge_fit)',
                    '前走は展開負け→今回は差し向き': '前走で先行有利展開に差し負け→今回差し有利 (sashi_revenge_fit)',
                    '前走大敗を度外視・実績馬（巻き返し妙味）': '前走大敗だが過去実績あり→度外視の妙味 (flop_rebound)',
                }
                from pred_utils import reason_strength as _rstr
                _tag_spans = []
                for r_label in reasons:
                    _tag = _TAG_MAP.get(r_label)
                    if _tag:
                        _short, _bg, _fg = _tag
                    else:
                        _short, _bg, _fg = r_label[:6], '#333', '#aaa'
                    # 強度で視覚差別化: 強=●太字・くっきり / 中=通常 / 弱=控えめ(半透明)
                    _st = _rstr(r_label)
                    _strmk = '● ' if _st == 3 else ''
                    _wt = 'bold' if _st == 3 else 'normal'
                    _op = '1' if _st >= 2 else '0.55'
                    _rhelp = _REASON_HELP.get(r_label, r_label)
                    _tier_lbl = {3: '強い根拠', 2: '中程度の根拠', 1: '弱い根拠'}[_st]
                    _tag_spans.append(
                        f'<span title="[{_tier_lbl}] {_rhelp}" style="background:{_bg};color:{_fg};border:1px solid {_fg};'
                        f'border-radius:3px;padding:1px 6px;font-size:0.75em;white-space:nowrap;cursor:help;'
                        f'font-weight:{_wt};opacity:{_op};">'
                        f'{_strmk}{_short}</span>'
                    )
                # ⑦ 昇級初戦の過剰人気警戒（赤タグ・前走勝ち上がり→今走昇級）
                _cuf = row.get('class_up_first')
                if pd.notna(_cuf) and float(_cuf) == 1.0:
                    _tag_spans.append(
                        '<span title="前走で勝ち上がり今走が昇級初戦(class_up_first=1)。過剰人気になりやすく妙味薄の警戒タグ。" '
                        'style="background:#5a1a1a;color:#e74c3c;border:1px solid #e74c3c;'
                        'border-radius:3px;padding:1px 6px;font-size:0.75em;white-space:nowrap;cursor:help;">'
                        '⚠昇級初戦</span>'
                    )
                reasons_html = ' '.join(_tag_spans) if _tag_spans else ''

                # 脚質バッジ（脚質のみ表示。ペース選好/適合判定/上り平均は「判定データ不足」等が
                # 出て不統一だったため廃止。脚質はコースPCI非依存で馬の過去4角位置から常時表示）
                apt = _horse_apts.get(name, {})
                pace_apt_html = ''
                if apt:
                    _style = apt.get('style', '')
                    _n = apt.get('n_races', 0)
                    if _n >= 3 and _style and _style not in ('不明', ''):
                        pace_apt_html = (
                            f'<span title="この馬の脚質（過去の平均4角位置から判定）。" '
                            f'style="color:#9db4d0;font-size:0.78em;margin-left:6px;cursor:help;">'
                            f'🏇 {_style}</span>'
                        )

                _rank_icon = '🥇' if rank==1 else ('🥈' if rank==2 else ('🥉' if rank==3 else f'{rank}位'))
                _name_color = '#f39c12' if is_tokujou else 'white'
                _rank_color = '#f1c40f' if rank==1 else 'white'
                _drift_str = f"{'+' if drift>0 else ''}{drift}"
                _live_icon = '📡' if pd.notna(odds_live) else ''
                if not _detail_cards:
                    # 圧縮カード（2行）: 印/馬番/馬名/予想複勝% ＋ 補助1行（人気・オッズ・騎手・穴・脚質・強タグ2）
                    _tags2 = ' '.join(_tag_spans[:2]) if _tag_spans else ''
                    _ar = f'・穴{rank_anaba}位' if rank_anaba is not None else ''
                    _mk_c = (f'<span style="background:{_mark_color};color:#111;font-weight:bold;'
                             f'font-size:0.9em;padding:0 6px;border-radius:4px;white-space:nowrap;">{_mark}</span>'
                             if _mark else '')
                    _fp_c = (f'<span style="color:{_fp_col};font-weight:bold;font-size:0.9em;white-space:nowrap;">複勝{float(_fp) * 100:.0f}%</span>'
                             if pd.notna(_fp) else '')
                    st.markdown(
                        f'<div style="background:{bg};border-radius:8px;padding:6px 10px;margin-bottom:3px;border-left:4px solid {border};">'
                        f'<div style="display:flex;align-items:center;gap:6px;">'
                        f'<span style="font-weight:bold;color:{_rank_color};white-space:nowrap;font-size:0.9em;min-width:28px;">{_rank_icon}</span>'
                        f'{umaban_html}'
                        f'{_mk_c}'
                        f'<span style="font-size:1.02em;font-weight:bold;color:{_name_color};flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{name}</span>'
                        f'{_fp_c}'
                        f'</div>'
                        f'<div style="font-size:0.8em;color:#8b949e;margin-top:2px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">'
                        f'{"---" if pop == 0 else str(pop) + "人気"} ・ {odds_html} ・ {jock}{_ar}{pace_apt_html}　{_tags2}'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div style="background:{bg};border-radius:8px;padding:5px 10px;margin-bottom:4px;border-left:4px solid {border};">'
                        f'<div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">'
                        f'<span style="font-size:1.15em;font-weight:bold;color:{_rank_color};white-space:nowrap;">{_rank_icon}</span>'
                        f'{umaban_html}'
                        f'<span style="font-size:1.05em;font-weight:bold;color:{_name_color};">{name}</span>'
                        f'{anaba_badge}'
                        f'{honmei_html}'
                        f'{pace_apt_html}'
                        f'</div>'
                        f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-top:2px;">'
                        f'<span style="color:#aaa;font-size:0.85em;">{jock}</span>'
                        f'<span style="color:#aaa;font-size:0.85em;">{"---" if pop == 0 else f"{pop}番人気"}</span>'
                        f'{odds_html}'
                        f'{anaba_rank_html}'
                        f'{_fp_html}'
                        f'<span style="color:#aaa;font-size:0.82em;">乖離{_drift_str}</span>'
                        f'<span style="color:{ev_color};font-weight:bold;font-size:0.9em;margin-left:auto;">単EV:{ev_t_str}{_live_icon}</span>'
                        f'<span style="color:{fev_color};font-weight:bold;font-size:0.9em;">複EV:{ev_f_str}</span>'
                        f'</div>'
                        + (f'<div style="margin-top:3px;display:flex;flex-wrap:wrap;gap:4px;">{reasons_html}</div>' if reasons_html else '')
                        + '</div>',
                        unsafe_allow_html=True
                    )

            # ── SNS用バナー生成（1:1 PNG 2枚: 予測サマリー / コースデータ）──
            with st.expander("📣 SNS用バナーを作成（1:1 PNG・2枚）"):
                if st.button("🖼 バナーを生成", key=f'gen_banner_{v_name}_{sel_r}',
                             help="この画面の予測から、SNS投稿用の正方形バナー2枚を作成します"):
                    try:
                        from banner import make_banners
                        _bds = str(show_df['日付'].iloc[0]) if '日付' in show_df.columns and not show_df.empty else ''
                        _bd8 = ('20' + _bds) if len(_bds) == 6 else _bds
                        try:
                            _bdt = __import__('datetime').datetime.strptime(_bd8, '%Y%m%d')
                            _bdow = '月火水木金土日'[_bdt.weekday()]
                            _bdate = f"{_bdt.year}.{_bdt.month}.{_bdt.day}（{_bdow}）"
                        except Exception:
                            _bdate = _bds
                        _btitle = f"{v_name}{sel_r}R"
                        if r_name and str(r_name) not in ('', 'nan'):
                            _btitle += f"　{r_name}"
                        _bsurf = '芝' if r_surf == '芝' else ('ダ' if r_surf in ('ダート', 'ダ') else '')
                        _bdist_s = f"{int(r_dist)}" if str(r_dist).replace('.', '').isdigit() else str(r_dist)
                        _bsub = f"{_bsurf}{_bdist_s}m ・ {r_heads}頭"
                        _bconf = int(round(conf_display * 5))
                        with st.spinner("バナーを生成中…"):
                            _b1, _b2 = make_banners(
                                show_df, venue_abbr=_cvenue, venue_full=v_name,
                                is_turf=_cturf, dist=_cdist, date_label=_bdate,
                                title=_btitle, subtitle=_bsub, confidence=_bconf,
                                n_horses=r_heads)
                        # SNS投稿テキスト（バナーと同じ複勝率）。印数は出走頭数で変える:
                        #   9頭以下=◎〇★ / 10〜14頭=◎〇▲★ / 15頭以上=◎〇▲△★（★=妙味で常に最後）
                        try:
                            _md = f"{_bdt.month:02d}/{_bdt.day:02d}"
                        except Exception:
                            _md = _bds
                        _rn = str(r_name) if r_name and str(r_name) not in ('', 'nan') else ''
                        _hd = ' '.join(x for x in [_md, v_name, f"{sel_r}R", _rn] if x)
                        _plines = [_hd, '']
                        _sdf_p = show_df.sort_values('pred_rank')
                        try:
                            _nh = int(r_heads)
                        except Exception:
                            _nh = len(show_df)
                        # (印, 表示頭数)。15頭以上は△を2頭まで出す（該当が1頭なら1頭）。
                        if _nh >= 15:
                            _mk_seq = [('◎', 1), ('○', 1), ('▲', 1), ('△', 2), ('★', 1)]
                        elif _nh >= 10:
                            _mk_seq = [('◎', 1), ('○', 1), ('▲', 1), ('★', 1)]
                        else:
                            _mk_seq = [('◎', 1), ('○', 1), ('★', 1)]
                        for _mk, _cnt in _mk_seq:
                            _mr = _sdf_p[_sdf_p.get('_mark', pd.Series('', index=_sdf_p.index)).astype(str) == _mk]
                            for _, _rr in _mr.head(_cnt).iterrows():
                                _fpp = pd.to_numeric(_rr.get('_fuku_prob'), errors='coerce')
                                _fps = f" {int(round(_fpp * 100))}%" if pd.notna(_fpp) else ''
                                _dmk = '〇' if _mk == '○' else _mk   # 対抗は漢数字ゼロで表記
                                _plines.append(f"{_dmk} {_rr.get('馬名', '')}{_fps}")
                        _plines += ['', 'よろしくお願いいたします😎', '', '※数字は予測複勝率']
                        _post_text = '\n'.join(_plines)
                        st.session_state[f'banner_imgs_{v_name}_{sel_r}'] = (_b1, _b2, _post_text)
                    except Exception as _be:
                        st.error(f"バナー生成エラー: {_be}")
                        import traceback as _tb; st.code(_tb.format_exc())
                _bimg_key = f'banner_imgs_{v_name}_{sel_r}'
                if _bimg_key in st.session_state:
                    _bpack = st.session_state[_bimg_key]
                    _bi1, _bi2 = _bpack[0], _bpack[1]
                    _post_txt = _bpack[2] if len(_bpack) > 2 else ''
                    st.caption("画像を右クリック／長押しで保存、または下のボタンでダウンロードできます。")
                    _bc1, _bc2 = st.columns(2)
                    with _bc1:
                        st.image(_bi1, caption="① 予測サマリー")
                        st.download_button("① 予測サマリーを保存", _bi1,
                                           file_name=f"{v_name}{sel_r}R_予測.png", mime="image/png",
                                           key=f'dlb1_{v_name}_{sel_r}', use_container_width=True)
                    with _bc2:
                        st.image(_bi2, caption="② コースデータ")
                        st.download_button("② コースデータを保存", _bi2,
                                           file_name=f"{v_name}{sel_r}R_コース.png", mime="image/png",
                                           key=f'dlb2_{v_name}_{sel_r}', use_container_width=True)
                    if _post_txt:
                        st.markdown("##### 📝 投稿用テキスト")
                        st.caption("右上のコピーアイコン📋でコピーできます。")
                        st.code(_post_txt, language=None)
                        import urllib.parse as _up
                        _xurl = "https://twitter.com/intent/tweet?text=" + _up.quote(_post_txt)
                        st.link_button("𝕏 Xにこのテキストで投稿", _xurl, use_container_width=True)
                        st.caption("※ Xボタンは投稿画面にテキストを引き継ぎます。画像はX側の仕様で自動添付できないため、"
                                   "上のバナーを保存してX投稿画面で手動添付してください。")

            # ── EV一覧バーチャート ───────────────────────────────────
            if 'EV単勝' in show_df_sorted.columns:
                _ev_src_label = _ev_source if '_ev_source' in dir() else '推定'
                _ev_icon = '📡' if '実オッズ' in _ev_src_label else '📊'
                st.markdown(f"#### 単勝EV一覧　{_ev_icon} {_ev_src_label}")
                ev_fig_df = show_df_sorted[['馬名', 'EV単勝', '人気']].copy()
                ev_fig_df['色'] = ev_fig_df['EV単勝'].apply(
                    lambda x: '高EV(+50%↑)' if x >= 50 else ('プラスEV' if x >= 0 else 'マイナスEV'))
                _ev_title = (
                    "単勝 期待値（EV）一覧　— 実オッズ使用（精度高）"
                    if '実オッズ' in _ev_src_label
                    else "単勝 期待値（EV）一覧　— 人気別平均配当ベース推定（目安）"
                )
                fig_ev = px.bar(
                    ev_fig_df, x='馬名', y='EV単勝',
                    color='色',
                    color_discrete_map={'高EV(+50%↑)': '#2ecc71', 'プラスEV': '#f39c12', 'マイナスEV': '#e74c3c'},
                    title=_ev_title,
                    labels={'EV単勝': 'EV (%)', '馬名': ''},
                )
                fig_ev.add_hline(y=0, line_dash='dash', line_color='white', opacity=0.4)
                fig_ev.update_xaxes(tickangle=30)
                fig_ev.update_layout(showlegend=True)
                st.plotly_chart(fig_ev)
                if '実オッズ' not in _ev_src_label:
                    st.caption("💡 ライブオッズ取得ボタンでリアルタイムオッズを読み込むと、より正確なEVで表示されます。")


# ============================================================
# Tab 4: データベース検索
# ============================================================
with tab4:
    st.subheader("データベース検索")
    st.caption("条件を絞って馬・種牡馬の実績をランキング表示します。10年分のデータから集計します。")

    if not MASTER_PARQUET.exists():
        st.warning("master.csv が見つかりません。build_dataset.py を実行してください。")
    else:
      try:
        # 競馬場略称 → 正式名称マッピング
        # ── フィルタパネル ──────────────────────────────────────────────
        st.markdown("### 絞り込み条件")
        st.caption("※ データは「🔍 検索実行」を押したときのみ読み込みます（251MB のため自動読込を省略）")
        fc1, fc2, fc3, fc4 = st.columns(4)

        with fc1:
            venue_opts = ['すべて'] + VENUE_ORDER
            sel_venue = st.selectbox("競馬場", venue_opts)

        with fc2:
            surf_opts = ['すべて', '芝', 'ダ']
            sel_surf = st.selectbox("芝・ダート", surf_opts)

        with fc3:
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                dist_lo = st.number_input("距離 最小(m)", min_value=0, max_value=9999,
                                          value=800, step=100)
            with d_col2:
                dist_hi = st.number_input("距離 最大(m)", min_value=0, max_value=9999,
                                          value=3600, step=100)

        with fc4:
            style_opts = {
                'すべて': (0.0, 1.0),
                '逃げ・先行 (前4角 上位30%)': (0.0, 0.35),
                '差し (中団 30~70%)': (0.35, 0.70),
                '追い込み (後方 70%~)': (0.70, 1.0),
            }
            sel_style_label = st.selectbox("脚質（前走4角位置）", list(style_opts.keys()))
            style_lo, style_hi = style_opts[sel_style_label]

        fc5, fc6, fc7 = st.columns(3)
        with fc5:
            sel_sire = st.text_input("種牡馬（部分一致）", value="")
        with fc6:
            sel_bms = st.text_input("母父馬（部分一致）", value="")
        with fc7:
            baba_opts = ['すべて', '良', '稍', '重', '不']
            sel_baba = st.selectbox("馬場状態", baba_opts)

        # 集計単位と並び順
        sa1, sa2, sa3 = st.columns(3)
        with sa1:
            group_by = st.radio("集計単位", ["馬名", "種牡馬", "母父馬", "騎手", "調教師"], horizontal=True)
        with sa2:
            sort_by = st.selectbox("並び順", ["勝率", "複勝率", "単勝回収値", "複勝回収値", "平均着順", "平均上り3F", "出走数"])
        with sa3:
            min_races = st.slider("最低出走数", 1, 30, 5)

        run_search = st.button("🔍 検索実行", type="primary")

        if run_search:
            master = load_master_full()
            # ── フィルタ適用 ──────────────────────────────────────────
            filt = master.copy()

            if sel_venue != 'すべて':
                filt = filt[filt['_venue_name'] == sel_venue]
            if sel_surf != 'すべて':
                filt = filt[filt['芝・ダ'] == sel_surf]
            filt = filt[filt['距離'].between(dist_lo, dist_hi)]
            if sel_style_label != 'すべて':
                filt = filt[filt['_style_ratio'].between(style_lo, style_hi)]
            if sel_sire.strip() and '種牡馬' in filt.columns:
                filt = filt[filt['種牡馬'].astype(str).str.contains(sel_sire.strip(), na=False)]
            if sel_bms.strip() and '母父馬' in filt.columns:
                filt = filt[filt['母父馬'].astype(str).str.contains(sel_bms.strip(), na=False)]
            if sel_baba != 'すべて':
                filt = filt[filt['馬場状態'] == sel_baba]

            # 着順が確定しているレコードのみ集計
            filt = filt[filt['着順_num'].notna() & (filt['着順_num'] >= 1)]

            if len(filt) == 0:
                st.warning("条件に一致するデータがありません。")
            else:
                # ── 集計 ──────────────────────────────────────────────
                grp_col = group_by
                if grp_col not in filt.columns:
                    st.error(f"列 '{grp_col}' が見つかりません。")
                else:
                    agg = filt.groupby(grp_col).agg(
                        出走数=('着順_num', 'count'),
                        勝利数=('着順_num', lambda x: (x == 1).sum()),
                        複勝数=('着順_num', lambda x: (x <= 3).sum()),
                        avg_chaku=('着順_num', 'mean'),
                        avg_agari=('前走上り3F', 'mean'),
                        _tan_sum=('_tan_pay', lambda x: x.where(filt.loc[x.index,'着順_num']==1).sum()),
                        _fuku_sum=('_fuku_pay', lambda x: x.where(filt.loc[x.index,'着順_num']<=3).sum()),
                    ).reset_index()

                    agg['勝率']      = agg['勝利数'] / agg['出走数']
                    agg['複勝率']     = agg['複勝数'] / agg['出走数']
                    agg['平均着順']    = agg['avg_chaku']
                    agg['平均上り3F']  = agg['avg_agari']
                    # 単勝回収値・複勝回収値（100円賭けた場合の平均回収額）
                    agg['単勝回収値']  = (agg['_tan_sum']  / agg['出走数']).round(1)
                    agg['複勝回収値']  = (agg['_fuku_sum'] / agg['出走数']).round(1)

                    agg = agg[agg['出走数'] >= min_races]

                    sort_col_map = {
                        '勝率':     ('勝率',     False),
                        '複勝率':    ('複勝率',    False),
                        '単勝回収値': ('単勝回収値', False),
                        '複勝回収値': ('複勝回収値', False),
                        '平均着順':   ('平均着順',   True),
                        '平均上り3F': ('平均上り3F', True),
                        '出走数':    ('出走数',    False),
                    }
                    s_col, s_asc = sort_col_map[sort_by]
                    agg = agg.sort_values(s_col, ascending=s_asc).reset_index(drop=True)
                    agg.index += 1

                    disp = agg[[grp_col, '出走数', '勝利数', '複勝数',
                                '勝率', '複勝率', '単勝回収値', '複勝回収値',
                                '平均着順', '平均上り3F']].copy()

                    def color_rate(val):
                        if not isinstance(val, float): return ''
                        if val >= 0.25: return 'background-color: #2ecc7144'
                        if val >= 0.15: return 'background-color: #f39c1244'
                        return ''

                    def color_roi(val):
                        if not isinstance(val, float): return ''
                        if val >= 100: return 'background-color: #2ecc7144'
                        if val >= 80:  return 'background-color: #f39c1244'
                        return 'background-color: #e74c3c22'

                    dist_label = f"{dist_lo}m〜{dist_hi}m"
                    st.markdown(f"**{len(agg)}件ヒット** （条件: {sel_venue} / {sel_surf} / {dist_label} / {sel_style_label}）")
                    st.caption("単勝回収値・複勝回収値: 100円賭けた場合の平均回収額（100円以上でプラス収支）")

                    st.dataframe(
                        disp.style
                            .format({'勝率': '{:.1%}', '複勝率': '{:.1%}',
                                     '単勝回収値': '{:.1f}円', '複勝回収値': '{:.1f}円',
                                     '平均着順': '{:.2f}', '平均上り3F': '{:.1f}'})
                            .map(color_rate, subset=['勝率', '複勝率'])
                            .map(color_roi,  subset=['単勝回収値', '複勝回収値']),
                        height=500,
                    )

                    # ── 上位20のバーチャート ──────────────────────────
                    top20 = agg.head(20)
                    y_col = sort_by if sort_by in top20.columns else '勝率'
                    fig_db = px.bar(
                        top20, x=grp_col, y=y_col,
                        color='勝率',
                        color_continuous_scale=['#e8f4f8','#2980b9'],
                        title=f"上位20（{sort_by}順）",
                        text='出走数',
                    )
                    fig_db.update_traces(texttemplate='%{text}走', textposition='outside')
                    fig_db.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_db)

                    # ── 脚質×着順分布（散布図）─────────────────────────
                    if sel_style_label == 'すべて' and grp_col == '馬名':
                        st.markdown("#### 脚質スコア vs 平均着順（散布図）")
                        style_agg = filt.groupby('馬名').agg(
                            出走数=('着順_num','count'),
                            平均着順=('着順_num','mean'),
                            avg_style=('_style_ratio','mean'),
                        ).reset_index()
                        style_agg = style_agg[style_agg['出走数'] >= min_races]
                        fig_sc = px.scatter(
                            style_agg, x='avg_style', y='平均着順',
                            size='出走数', hover_name='馬名',
                            labels={'avg_style': '脚質スコア（0=逃げ 1=追い込み）', '平均着順': '平均着順'},
                            title='脚質スコア vs 平均着順（バブルサイズ=出走数）',
                            color='平均着順',
                            color_continuous_scale='RdYlGn_r',
                        )
                        fig_sc.update_yaxes(autorange='reversed')
                        st.plotly_chart(fig_sc)
      except Exception as _e4:
          st.warning(f"データベース検索の読み込みエラー: {_e4}")


# ============================================================
# Tab 5: 回収率トラッキング
# ============================================================
def _render_tracking_tab():
    st.subheader("📈 回収率トラッキング")
    st.caption("「レース予測」タブで表示した予測の本命◎を自動記録し、確定結果（master）と"
               "突合して単勝・複勝の回収率を集計します。")

    import result_tracker as _rt   # reloadしない（master突合テーブルのキャッシュを保持）

    if not _rt.MASTER_PARQUET.exists():
        st.error("master.parquet が見つかりません。結果の突合ができません。")
        return

    _pred = _rt.load_pred_log()
    if _pred.empty:
        st.info("予測ログがまだありません。「📊 レース予測」タブで予測を表示すると、"
                "そのレースの本命◎が自動で記録されます。")
        return

    try:
        _mres = _rt._load_master_results()
        _mlast = sorted(_mres['_d'].unique())[-1] if not _mres.empty else '―'
    except Exception:
        _mlast = '―'

    # ── 期間フィルタ ────────────────────────────────────────────────
    _pdates = sorted(_pred['date'].astype(str).unique())
    _c1, _c2 = st.columns([1, 3])
    with _c1:
        _period = st.selectbox('集計期間', ['全期間', '直近1ヶ月', '直近3ヶ月'], key='roi_period')
    _pf = _pred.copy()
    if _period != '全期間' and _pdates:
        _mn = 1 if _period == '直近1ヶ月' else 3
        _cut = (pd.to_datetime(_pdates[-1], format='%Y%m%d')
                - pd.DateOffset(months=_mn)).strftime('%Y%m%d')
        _pf = _pf[_pf['date'].astype(str) >= _cut]

    _res = _rt.calc_honmei_roi(_pf)
    _sts, _detail = _res['stats'], _res['detail']
    with _c2:
        st.caption(f"master最新日: {_mlast}　｜　記録 {_sts['n_pred']}R ／ "
                   f"本命あり {_sts['n_honmei']}R ／ 結果と突合 **{_sts['n_matched']}R**"
                   + (f" ／ 未突合 {_sts['n_unmatched']}R" if _sts['n_unmatched'] else ""))

    if _detail.empty:
        st.warning("結果と突合できたレースがまだありません。"
                   "master に該当日の結果が追加されているかご確認ください。")
        return

    # ── サマリー ────────────────────────────────────────────────────
    st.markdown("### 回収率サマリー（本命◎）")
    _tan, _fuku = _res['summary'].iloc[0], _res['summary'].iloc[1]

    def _rc(v):
        return '#2ecc71' if v >= 100 else ('#e67e22' if v >= 80 else '#e74c3c')

    _m1, _m2, _m3, _m4 = st.columns(4)
    _m1.metric("単勝 回収率", f"{_tan['回収率']}%", help="本命◎に単勝100円を賭け続けた場合")
    _m2.metric("単勝 的中率", f"{_tan['的中率']}%",
               help=f"{_tan['的中']}/{_tan['的中'] + _tan['外れ']}レース")
    _m3.metric("複勝 回収率", f"{_fuku['回収率']}%", help="本命◎に複勝100円を賭け続けた場合")
    _m4.metric("複勝 的中率", f"{_fuku['的中率']}%",
               help=f"{_fuku['的中']}/{_fuku['的中'] + _fuku['外れ']}レース")
    st.markdown(
        f"<span style='color:{_rc(_tan['回収率'])};font-weight:bold;'>単勝 {_tan['回収率']}%</span>"
        f" <span style='color:#8b949e;font-size:0.85em;'>（投資{_tan['投資額']:,}円 → "
        f"回収{_tan['回収額']:,}円 / 収支{_tan['回収額'] - _tan['投資額']:+,}円）</span>　　"
        f"<span style='color:{_rc(_fuku['回収率'])};font-weight:bold;'>複勝 {_fuku['回収率']}%</span>"
        f" <span style='color:#8b949e;font-size:0.85em;'>（投資{_fuku['投資額']:,}円 → "
        f"回収{_fuku['回収額']:,}円 / 収支{_fuku['回収額'] - _fuku['投資額']:+,}円）</span>",
        unsafe_allow_html=True)
    st.caption("単位100円。回収率100%超で利益。控除率（単勝・複勝とも約20%）があるため、"
               "80%前後が「平均的に賭けた場合」の水準です。")

    # ── 日別 ────────────────────────────────────────────────────────
    st.markdown("### 日別回収率")
    _daily = _rt.calc_honmei_daily(_detail)
    st.dataframe(
        _daily, hide_index=True, use_container_width=True,
        column_config={
            '日付': st.column_config.TextColumn('日付', width='small'),
            'レース数': st.column_config.NumberColumn('R数', width='small'),
            '単勝的中': st.column_config.NumberColumn('単的中', width='small'),
            '単勝回収率': st.column_config.NumberColumn('単回収%', format='%.1f'),
            '複勝的中': st.column_config.NumberColumn('複的中', width='small'),
            '複勝回収率': st.column_config.NumberColumn('複回収%', format='%.1f'),
        })

    # ── 記録状況（日付別）: どこで欠けているかを可視化 ──────────────
    with st.expander("🔎 記録状況（日付別）— 反映されない時はここを確認"):
        _pl = _pred.copy()
        _pl['_h'] = (_pl['honmei'].astype(str).str.strip()
                     .replace({'nan': '', 'None': '', 'NaN': '', '<NA>': ''}) != '')
        _stat = (_pl.groupby('date')
                 .agg(記録R=('race_id', 'size'), 本命あり=('_h', 'sum')).reset_index()
                 .rename(columns={'date': '日付'}))
        _stat['日付'] = _stat['日付'].astype(str)
        _mt = (_detail.groupby('日付').size().rename('突合R').reset_index()
               if not _detail.empty else pd.DataFrame(columns=['日付', '突合R']))
        _mt['日付'] = _mt['日付'].astype(str)
        _stat = _stat.merge(_mt, on='日付', how='left')
        _stat['突合R'] = _stat['突合R'].fillna(0).astype(int)
        st.dataframe(_stat.sort_values('日付', ascending=False), hide_index=True,
                     use_container_width=True)
        st.caption("「記録R」＝予測ログに入っているレース数（通常は36＝12R×3場）。"
                   "「本命あり」＝集計に使える行。「突合R」＝masterの結果と照合できたレース数。"
                   "記録Rが少ない→予測を管理者モードで読み込み直す／"
                   "突合Rが少ない→masterにその日の結果が未追加、が原因です。")

    # ── 明細 ────────────────────────────────────────────────────────
    with st.expander(f"📋 レース明細（{len(_detail)}R）"):
        st.dataframe(
            _detail.sort_values(['日付', '開催', 'R'], ascending=[False, True, True]),
            hide_index=True, use_container_width=True,
            column_config={
                '日付': st.column_config.TextColumn('日付', width='small'),
                'R': st.column_config.NumberColumn('R', width='small'),
                '着順': st.column_config.NumberColumn('着', width='small'),
                '単払戻': st.column_config.NumberColumn('単払戻', format='%d'),
                '複払戻': st.column_config.NumberColumn('複払戻', format='%d'),
            })

    # ── 予測ログ管理 ────────────────────────────────────────────────
    st.divider()
    with st.expander("💾 予測ログの管理（バックアップ／復元）"):
        st.caption("予測ログはCloudの再デプロイで消えます。定期的にダウンロードし、"
                   "消えたらアップロードで復元してください。")
        _b1, _b2 = st.columns(2)
        with _b1:
            st.download_button("⬇️ 予測ログをCSV保存", key='dl_predlog',
                               data=_pred.to_csv(index=False).encode('utf-8-sig'),
                               file_name="pred_log.csv", mime="text/csv")
        with _b2:
            _up = st.file_uploader("予測ログCSVを復元", type='csv', key='up_predlog')
            if _up:
                _dfu = pd.read_csv(_up, dtype=str)
                _rt._write_pred_log(_dfu)
                st.success(f"予測ログを復元しました（{len(_dfu)}件）。")
                st.rerun()
        if st.button("🗑️ 予測ログをクリア", key='clear_predlog'):
            _rt.PRED_LOG_PATH.unlink(missing_ok=True)
            st.success("予測ログを削除しました。")
            st.rerun()


# ============================================================
# Tab 6: 馬ノート（回顧DB・管理者メモ）
# ============================================================
def _render_watch_tab():
    st.subheader("📝 馬ノート（回顧DB）")
    st.caption("映像・パドック等、データに見えない不利/見どころ/状態をレース単位で記録。"
               "次にその馬が出走する際、レース予測にタグ表示します（独自データの蓄積）。")
    try:
        import importlib, datetime as _dtw
        import sheets_store as _ss   # reloadしない（認証/データのキャッシュを再実行間で保持）
        import race_notes as _wh
        importlib.reload(_wh)

        # ── 保存先ステータス（Googleスプレッドシート直結なら永続化）──────────
        _stat = _wh.storage_status()
        if _stat['mode'] == 'sheets':
            st.success("🟢 保存先: **Googleスプレッドシート**（永続化済み・再デプロイでも消えません）。"
                       "シートを直接編集すれば一括で追加・更新・削除できます。", icon="✅")
        elif _stat['mode'] == 'sheets_error':
            st.error(f"🔴 スプレッドシートに接続できません: {_stat['error']}\n\n"
                     "secretsの設定内容と、サービスアカウントへのシート共有（編集者）をご確認ください。"
                     "接続できるまではローカル保存にフォールバックします。")
        else:
            st.warning("🟡 保存先: **ローカル（parquet）**。Cloudは再デプロイで消えます。"
                       "下の「💾 保存先の設定」からGoogleスプレッドシート直結にすると永続化＆一括更新が可能です。")

        # 馬名候補: master 直近30日の出走馬
        _recent_names = []
        try:
            _cut = (_dtw.date.today() - _dtw.timedelta(days=30)).strftime('%y%m%d')
            _rn = pd.read_parquet(MASTER_PARQUET, columns=['馬名', '日付'],
                                  filters=[('日付', '>=', _cut)])
            _recent_names = sorted(_rn['馬名'].dropna().astype(str).map(_wh.normalize_name).unique())
        except Exception:
            _recent_names = []

        # ── 📊 メモ馬の成績トラッキング（次走）──────────────────────────────
        with st.expander("📊 メモ馬の成績トラッキング（次走の結果で“あなたの目”を検証）", expanded=False):
            st.caption("メモした馬が『次走』でどれだけ走ったかを集計。複勝率・回収率で、評価やタグにエッジがあるかを検証します。"
                       "（次走を終えたメモだけが対象。溜まるほど精度が上がります）")
            try:
                import importlib as _il, note_tracking as _ntk
                _il.reload(_ntk)
                _tk = _ntk.evaluate()
                if _tk.get('raw_n', 0) == 0:
                    st.info("まだ集計対象がありません（メモした馬が次走を終えると集計されます）。メモを溜めていきましょう。")
                else:
                    _ov = _tk['overall']
                    st.markdown(f"**全体（次走を終えたメモ {_ov['n']}件）**")
                    _m1, _m2, _m3, _m4, _m5 = st.columns(5)
                    _m1.metric("複勝率", f"{_ov['複勝率']}%")
                    _m2.metric("勝率", f"{_ov['勝率']}%")
                    _m3.metric("平均人気", f"{_ov['平均人気']}")
                    _m4.metric("単回収率", f"{_ov['単回収率']}%")
                    _m5.metric("複回収率", f"{_ov['複回収率']}%")
                    st.caption("※回収率100%超なら控除率の壁を越えた＝“あなたの目”にエッジがある可能性。サンプルが少ないうちは参考値。")

                    def _roi_style(v):
                        if not isinstance(v, (int, float)):
                            return ''
                        return 'color:#2ecc71;font-weight:bold' if v >= 100 else ('color:#e67e22' if v >= 70 else 'color:#e74c3c')
                    _fmt = {c: '{:.1f}%' for c in ['複勝率', '勝率', '単回収率', '複回収率']}
                    _fmt['平均人気'] = '{:.1f}'
                    if not _tk['by_eval'].empty:
                        st.markdown("**評価別**")
                        st.dataframe(_tk['by_eval'].style.format(_fmt, na_rep='—')
                                     .map(_roi_style, subset=['複回収率']), hide_index=True, use_container_width=True)
                    if not _tk['by_aim'].empty:
                        st.markdown("**狙い度別**")
                        st.dataframe(_tk['by_aim'].style.format(_fmt, na_rep='—')
                                     .map(_roi_style, subset=['複回収率']), hide_index=True, use_container_width=True)
                    if not _tk['by_tag'].empty:
                        st.markdown("**タグ別**（サンプル数の多い順）")
                        st.dataframe(_tk['by_tag'].style.format(_fmt, na_rep='—')
                                     .map(_roi_style, subset=['複回収率']), hide_index=True, use_container_width=True)
            except Exception as _tke:
                st.caption(f"（トラッキング集計をスキップ: {_tke}）")

        # ── 🔍 結果回顧（結果＋モデル評価を見ながらメモ） ──────────────────
        st.markdown("### 🔍 結果回顧（結果を見ながらメモ）")
        st.caption("masterに結果がある日のレースを選択し、各馬に評価・タグ・メモを記録します。"
                   "🎯列は脚余し候補の目安（上がり上位×着外）です。")
        try:
            _dates_av = sorted(pd.read_parquet(MASTER_PARQUET, columns=['日付'])['日付'].astype(str).unique())[-50:][::-1]
        except Exception:
            _dates_av = []
        if not _dates_av:
            st.info('masterに結果データがありません。')
        else:
            _rc1, _rc2, _rc3 = st.columns(3)
            with _rc1:
                _rdate = st.selectbox('日付', _dates_av,
                                      format_func=lambda d: (f"20{d[:2]}/{d[2:4]}/{d[4:]}" if len(d) == 6 else d),
                                      key='review_date')
            _dval = int(_rdate) if str(_rdate).isdigit() else _rdate
            _dayrows = pd.read_parquet(MASTER_PARQUET, filters=[('日付', '==', _dval)])
            _dayrows['_v'] = _dayrows['開催'].astype(str).apply(parse_venue)
            with _rc2:
                _venues = [v for v in VENUE_ORDER if v in _dayrows['_v'].unique()]
                _rven = st.selectbox('競馬場', _venues, key='review_venue') if _venues else None
            _vrows = _dayrows[_dayrows['_v'] == _rven] if _rven else _dayrows.iloc[0:0]
            with _rc3:
                _rs = sorted(pd.to_numeric(_vrows['Ｒ'], errors='coerce').dropna().astype(int).unique())
                _rr = st.selectbox('R', _rs, key='review_r') if _rs else None
            _race = _vrows[pd.to_numeric(_vrows['Ｒ'], errors='coerce') == _rr].copy() if _rr else _vrows.iloc[0:0]
            if _race.empty:
                st.info('レースを選択してください。')
            else:
                _kai = str(_race['開催'].iloc[0])
                _race['着'] = pd.to_numeric(_race['着順_num'], errors='coerce')
                _race['上り'] = pd.to_numeric(_race['上り3F'], errors='coerce')
                _race['候補'] = ((_race['上り'].rank(method='min') <= 3) & (_race['着'] >= 4)).map({True: '🎯', False: ''})

                def _tsuka(r):
                    xs = [str(int(r[c])) for c in ['2角', '3角', '4角']
                          if c in r and pd.notna(r[c]) and str(r[c]).replace('.', '').isdigit()]
                    return '-'.join(xs)
                _race['通過'] = _race.apply(_tsuka, axis=1)

                _rid = f"{_rdate}_{_kai}_{_rr}"
                _pk = f'_review_pred_{_rid}'
                if _pk not in st.session_state:
                    with st.spinner('モデル順位を計算中…（数秒）'):
                        from pipeline_target import predict_race_in_master as _prim
                        st.session_state[_pk] = _prim(_rdate, _kai, str(_rr))
                _pred = st.session_state[_pk]
                if not _pred.empty:
                    _race = _race.merge(_pred, on='馬名', how='left')
                else:
                    _race['pred_rank'] = np.nan
                    _race['pred_rank_anaba'] = np.nan

                _race = _race.sort_values('着').reset_index(drop=True)
                _md = pd.to_datetime('20' + _rdate, format='%Y%m%d') if len(_rdate) == 6 else pd.to_datetime(_rdate)

                # ── 実オッズ・払戻を結合し、印と買い目の回収結果を算出（odds_storeがある時のみ）──
                _marks_map, _bet_rows, _real_odds = {}, [], None
                try:
                    from odds_store import attach_odds as _attach_odds
                    from reliability import assign_marks as _assign_marks, evaluate_race_bets as _eval_bets
                    _mk = _attach_odds(_race)
                    if 'tan_odds' in _mk.columns and _mk['tan_odds'].notna().any():
                        _mk['_pop_int'] = pd.to_numeric(_mk['人気'], errors='coerce')
                        _mk = _assign_marks(_mk)
                        _mk['着'] = pd.to_numeric(_mk['着順_num'], errors='coerce')
                        _marks_map = dict(zip(_mk['馬名'].astype(str), _mk['_mark']))
                        _real_odds = dict(zip(pd.to_numeric(_mk['馬番'], errors='coerce'),
                                              pd.to_numeric(_mk['tan_odds'], errors='coerce')))
                        _bet_rows = _eval_bets(_mk)
                except Exception:
                    pass

                _rname_rv = str(_race['レース名'].iloc[0]) if 'レース名' in _race.columns and not _race.empty else ''
                _rdate8 = ('20' + _rdate) if len(str(_rdate)) == 6 else str(_rdate)

                def _fmt_time(sec):
                    try:
                        sec = float(sec)
                        return f"{int(sec // 60)}:{sec % 60:04.1f}"
                    except Exception:
                        return ''

                def _fmt_odds(v):
                    import re as _re
                    try:
                        return float(_re.sub(r'[()（）]', '', str(v)))
                    except Exception:
                        return np.nan

                _umaban_s = pd.to_numeric(_race['馬番'], errors='coerce')
                # オッズ: odds_storeの実単勝オッズを優先、無ければmasterの単勝配当参照値
                if _real_odds:
                    _odds_col = _umaban_s.map(_real_odds)
                else:
                    _odds_col = _race['単勝配当'].map(_fmt_odds) if '単勝配当' in _race.columns else np.nan
                _disp = pd.DataFrame({
                    '着': _race['着'].astype('Int64'),
                    '印': _race['馬名'].astype(str).map(_marks_map).fillna('') if _marks_map else '',
                    '馬番': _umaban_s.astype('Int64'),
                    '馬名': _race['馬名'].astype(str),
                    '騎手': _race['騎手'].astype(str) if '騎手' in _race.columns else '',
                    '人気': pd.to_numeric(_race['人気'], errors='coerce').astype('Int64'),
                    'オッズ': _odds_col,
                    'タイム': _race['走破秒'].map(_fmt_time) if '走破秒' in _race.columns else '',
                    '着差': _race['着差'].astype(str) if '着差' in _race.columns else '',
                    '上り': _race['上り'].round(1),
                    '通過': _race['通過'],
                    '通常': pd.to_numeric(_race['pred_rank'], errors='coerce').astype('Int64'),
                    '穴': pd.to_numeric(_race['pred_rank_anaba'], errors='coerce').astype('Int64'),
                    '候補': _race['候補'],
                })
                # ① 一括メモ入力: 結果を見ながら「メモ」列に各馬の印象を自由入力。
                # 着・馬番・馬名を左固定、結果列は編集不可、メモ列だけ編集可。
                # タグ付けはこのメモをAIが構造化（②）。タグ修正は📋登録済み（③）で行う。
                _disp['メモ'] = ''
                if _is_mobile:
                    _col_order = ['着', '馬番', '馬名', 'メモ', '通常', '穴', '印', '人気',
                                  'オッズ', '上り', '通過', '着差', 'タイム', '騎手', '候補']
                else:
                    _col_order = ['着', '印', '馬番', '馬名', '騎手', '人気', 'オッズ', 'タイム',
                                  '着差', '上り', '通過', '通常', '穴', '候補', 'メモ']
                _edited = st.data_editor(
                    _disp,
                    column_order=_col_order,
                    column_config={
                        '着':   st.column_config.NumberColumn('着', width='small', pinned=True),
                        '馬番': st.column_config.NumberColumn('馬番', width='small', pinned=True),
                        '馬名': st.column_config.TextColumn('馬名', width='small', pinned=True),
                        '印':   st.column_config.TextColumn('印', width='small'),
                        '騎手': st.column_config.TextColumn('騎手', width='small'),
                        '人気': st.column_config.NumberColumn('人気', width='small'),
                        'オッズ': st.column_config.NumberColumn('オッズ', format='%.1f', width='small'),
                        'タイム': st.column_config.TextColumn('タイム', width='small'),
                        '着差': st.column_config.TextColumn('着差', width='small'),
                        '上り': st.column_config.NumberColumn('上り', format='%.1f', width='small'),
                        '通過': st.column_config.TextColumn('通過', width='small'),
                        '通常': st.column_config.NumberColumn('通常', width='small'),
                        '穴':   st.column_config.NumberColumn('穴', width='small'),
                        '候補': st.column_config.TextColumn('候補', width='small'),
                        'メモ': st.column_config.TextColumn('メモ（見たまま自由に）', width='large'),
                    },
                    disabled=['着', '印', '馬番', '馬名', '騎手', '人気', 'オッズ', 'タイム',
                              '着差', '上り', '通過', '通常', '穴', '候補'],
                    hide_index=True, use_container_width=True, key=f'editor_{_rid}',
                )
                st.caption('着・馬番・馬名は左固定。気になった馬の「メモ」欄に印象を自由入力し、'
                           '下の「🪄」でタグ化して保存します。🎯＝脚余し候補の目安。')

                # ── このレースの買い目回収結果（実配当）──────────────────
                if _bet_rows:
                    _bdf = pd.DataFrame(_bet_rows)
                    _tb = int(_bdf['投資'].sum()); _tr = int(_bdf['払戻'].sum())
                    _roi = _tr / _tb * 100 if _tb else 0.0
                    _rc = '#2ecc71' if _roi >= 100 else ('#e67e22' if _roi >= 50 else '#e74c3c')
                    st.markdown(
                        f"#### 💴 このレースの買い目回収（実配当）　"
                        f"<span style='color:{_rc};font-weight:bold;'>回収率 {_roi:.0f}%</span> "
                        f"<span style='color:#8b949e;font-size:0.85em;'>（投資{_tb:,}円→回収{_tr:,}円 / 収支{_tr-_tb:+,}円）</span>",
                        unsafe_allow_html=True)
                    _show_bdf = _bdf.copy()
                    _show_bdf['的中'] = _show_bdf['的中'].map({True: '✅', False: '―'})
                    st.dataframe(
                        _show_bdf[['券種', '点数', '投資', '払戻', '損益', '的中']],
                        column_config={
                            '点数': st.column_config.NumberColumn('点数', width='small'),
                            '投資': st.column_config.NumberColumn('投資', format='%d円'),
                            '払戻': st.column_config.NumberColumn('払戻', format='%d円'),
                            '損益': st.column_config.NumberColumn('損益', format='%d円'),
                            '的中': st.column_config.TextColumn('的中', width='small'),
                        },
                        hide_index=True, use_container_width=True)
                    st.caption('印（◎○▲△★）に基づくテンプレ買い目を、このレースの実際の単勝オッズ・払戻で精算した結果です。'
                               '単位100円。★=連下内の妙味馬（相手に含む）。過去実績であり将来を保証しません。')

                import llm_assist as _llm_rv

                # ② メモをAIが評価・タグ（複数可）・狙い度に構造化して保存。タグ修正は③で。
                _sv1, _sv2 = st.columns(2)
                with _sv1:
                    if st.button('🪄 AI構造化して保存（メモのある全馬）', type='primary', key=f'llmsave_{_rid}',
                                 help='メモを書いた各馬について、AIが評価・タグ（複数可）・狙い度を判定し、メモと一緒に保存します。'
                                      'タグは後で「📋 登録済み馬ノート」で修正できます。'):
                        _memorows = [(str(_er['馬名']), str(_er.get('メモ', '') or '').strip())
                                     for _, _er in _edited.iterrows() if str(_er.get('メモ', '') or '').strip()]
                        if not _llm_rv.available():
                            st.error('APIキー未設定です。secretsに ANTHROPIC_API_KEY を登録してください。')
                        elif not _memorows:
                            st.warning('メモが空です。まず結果表の「メモ」欄に各馬の印象を入力してください。')
                        else:
                            _n, _fail = 0, 0
                            with st.spinner(f'AIが{len(_memorows)}件のメモを解析中…'):
                                for _nm, _mmo in _memorows:
                                    try:
                                        _s = _llm_rv.suggest_from_memo(_mmo, _wh.EVAL_OPTIONS, _wh.ALL_TAGS)
                                        _wh.add_note(_nm, _rdate8, 評価=_s['評価'], 狙い度=_s['狙い度'],
                                                     タグ=_s['タグ'], メモ=_mmo, 開催=_kai, Ｒ=_rr,
                                                     レース名=_rname_rv, ソース='LLM')
                                        _n += 1
                                    except Exception:
                                        _fail += 1
                            st.success(f'AI構造化して {_n}頭を保存しました。' + (f'（{_fail}件失敗）' if _fail else '')
                                       + ' タグ・メモの修正は下の「📋 登録済み馬ノート」で行えます。')
                            st.rerun()
                with _sv2:
                    if st.button('💾 メモだけ保存（タグ無し）', key=f'savereview_{_rid}',
                                 help='AIを使わずメモをそのまま保存します。評価・タグは後で「📋 登録済み馬ノート」で付けられます。'):
                        _saved = 0
                        for _, _er in _edited.iterrows():
                            _mmo = str(_er.get('メモ', '') or '').strip()
                            if _mmo:
                                _wh.add_note(_er['馬名'], _rdate8, 評価='中立', 狙い度=2,
                                             タグ=[], メモ=_mmo, 開催=_kai, Ｒ=_rr,
                                             レース名=_rname_rv, ソース='回顧')
                                _saved += 1
                        st.success(f'{_saved}頭のメモを保存しました。評価・タグは📋登録済みで付けられます。')
                        st.rerun()
                st.caption('操作の流れ：①メモ欄に入力 → ②「🪄」でAIがタグ化して保存 → '
                           '③下の「📋 登録済み馬ノート」でタグ・評価・メモを修正。')

        st.divider()
        st.markdown("### ➕ 1頭を詳細登録（🪄LLM補助あり）")
        import llm_assist as _llm
        st.session_state.setdefault('note_ev', '中立')
        st.session_state.setdefault('note_aim', 2)
        for _grp in _wh.TAG_GROUPS:
            st.session_state.setdefault(f'note_tag_{_grp}', [])

        _nc1, _nc2 = st.columns([3, 1])
        with _nc1:
            _sel = st.selectbox('馬名（直近30日の出走馬から選択）', options=[''] + _recent_names, index=0, key='note_sel')
            _txt = st.text_input('リストに無い場合は手入力（入力時はこちら優先）', key='note_txt')
        with _nc2:
            _memo_d = st.date_input('レース日（メモ対象）', value=_dtw.date.today(), key='note_date')
        _memo = st.text_area('観察メモ（自由記述）', key='note_memo_txt',
                             placeholder='例: 4角で前が壁、外に出して伸びたが届かず。次走は流れ次第で一変も。')

        _lb1, _lb2 = st.columns([1, 3])
        with _lb1:
            if st.button('🪄 メモからタグ提案', key='note_llm_btn'):
                if not str(_memo).strip():
                    st.warning('先に観察メモを入力してください。')
                elif not _llm.available():
                    st.error('APIキー未設定です。secretsに ANTHROPIC_API_KEY を登録してください。')
                else:
                    try:
                        with st.spinner('AIがメモを解析中…'):
                            _sug = _llm.suggest_from_memo(_memo, _wh.EVAL_OPTIONS, _wh.ALL_TAGS)
                        st.session_state['note_ev'] = _sug['評価']
                        st.session_state['note_aim'] = _sug['狙い度']
                        for _grp, _opts in _wh.TAG_GROUPS.items():
                            st.session_state[f'note_tag_{_grp}'] = [t for t in _sug['タグ'] if t in _opts]
                        st.session_state['note_llm_summary'] = _sug.get('要約', '') or '（要約なし）'
                        st.rerun()
                    except Exception as _le:
                        st.error(f'LLM提案エラー: {_le}')
        with _lb2:
            if st.session_state.get('note_llm_summary'):
                st.caption(f"🪄 AI提案を反映しました（要約: {st.session_state['note_llm_summary']}）。下で編集して登録できます。")

        _fc1, _fc2 = st.columns([2, 1])
        with _fc1:
            _ev = st.selectbox('評価', _wh.EVAL_OPTIONS, key='note_ev')
        with _fc2:
            _aim = st.radio('狙い度', [3, 2, 1], format_func=lambda x: '★' * x, key='note_aim', horizontal=True,
                            help='★1=軽め（一応チェック／条件が向けば・半信半疑）　'
                                 '★2=標準（次走で買い候補・妙味あり）　'
                                 '★3=本気（次走で本命〜対抗級に狙う／明確な巻き返し材料）')
        _tags = []
        for _grp, _opts in _wh.TAG_GROUPS.items():
            _tags += st.multiselect(f'タグ（{_grp}）', _opts, key=f'note_tag_{_grp}')

        if st.button('登録', type='primary', key='note_save_btn'):
            _name = (_txt or _sel).strip()
            if not _name:
                st.error('馬名を選択または手入力してください。')
            else:
                _src = 'LLM' if st.session_state.get('note_llm_summary') else '手動'
                _wh.add_note(_name, _memo_d, 評価=_ev, 狙い度=_aim, タグ=_tags, メモ=_memo, ソース=_src)
                st.session_state.pop('note_llm_summary', None)
                st.success(f'「{_name}」の馬ノートを登録しました。')
                st.rerun()

        st.divider()
        st.markdown("### 📋 登録済み馬ノート")
        _wdf = _wh.load_notes()
        if _wdf.empty:
            st.info('まだ馬ノートがありません。上でメモを保存すると、ここに一覧されます。')
        else:
            # 状態（有効/消化済み）を付与
            _names = _wdf['馬名'].map(_wh.normalize_name).unique().tolist()
            _last = {}
            try:
                _mm = pd.read_parquet(MASTER_PARQUET, columns=['馬名', '日付_dt'],
                                      filters=[('馬名', 'in', _names)])
                _mm['日付_dt'] = pd.to_datetime(_mm['日付_dt'], errors='coerce')
                _last = _mm.groupby(_mm['馬名'].map(_wh.normalize_name))['日付_dt'].max().to_dict()
            except Exception:
                _last = {}
            _wdf = _wh.annotate_active(_wdf, _last)

            def _md8(d):
                d = str(d)
                return f"{d[:4]}/{d[4:6]}/{d[6:]}" if len(d) == 8 else d

            # ── フィルタ（検索・状態・タグ）──
            _fc1, _fc2, _fc3 = st.columns([2, 1, 2])
            with _fc1:
                _nkey = st.text_input('🔍 検索（馬名・タグ・メモ）', key='notes_search').strip()
            with _fc2:
                _stf = st.selectbox('状態', ['全て', '🟢 有効', '⚪ 消化済み'], key='notes_statef')
            with _fc3:
                _tagf = st.multiselect('タグで絞込', _wh.ALL_TAGS, key='notes_tagf')
            _f = _wdf.copy()
            if _nkey:
                _f = _f[_f['馬名'].astype(str).str.contains(_nkey, na=False)
                        | _f['タグ'].astype(str).str.contains(_nkey, na=False)
                        | _f['メモ'].astype(str).str.contains(_nkey, na=False)]
            if _stf.startswith('🟢'):
                _f = _f[_f['状態'] == '有効']
            elif _stf.startswith('⚪'):
                _f = _f[_f['状態'] == '消化済み']
            if _tagf:
                _f = _f[_f['タグ'].astype(str).apply(lambda s: any(t in str(s).split('・') for t in _tagf))]

            if _f.empty:
                st.info('条件に合う馬ノートがありません。フィルタを緩めてください。')
            else:
                # ── 一覧テーブル（ヘッダクリックで並び替え・🔍検索・列幅調整が標準装備）──
                _tbl = pd.DataFrame({
                    '状態': _f['状態'].map({'有効': '🟢', '消化済み': '⚪'}).fillna(''),
                    '馬名': _f['馬名'].astype(str),
                    '評価': _f['評価'].astype(str),
                    '狙い': pd.to_numeric(_f['狙い度'], errors='coerce').fillna(2).astype(int).map(lambda n: '★' * n),
                    'タグ': _f['タグ'].astype(str),
                    'メモ': _f['メモ'].astype(str),
                    '日付': _f['日付'].map(_md8),
                    '場R': _f.apply(lambda r: f"{r['開催']}{r['Ｒ']}R" if str(r.get('開催', '')) else '', axis=1),
                    '元': _f['ソース'].astype(str),
                }).sort_values('日付', ascending=False)
                st.caption(f'{len(_f)}件。ヘッダをクリックで並び替え／右上🔍で表内検索／列幅ドラッグ可。'
                           '状態・馬名は左固定、横スクロールでタグ・メモを確認。')
                st.dataframe(
                    _tbl,
                    column_config={
                        '状態': st.column_config.TextColumn('状態', width='small', pinned=True),
                        '馬名': st.column_config.TextColumn('馬名', width='medium', pinned=True),
                        '評価': st.column_config.TextColumn('評価', width='small'),
                        '狙い': st.column_config.TextColumn('狙い', width='small'),
                        'タグ': st.column_config.TextColumn('タグ', width='large'),
                        'メモ': st.column_config.TextColumn('メモ', width='large'),
                        '日付': st.column_config.TextColumn('日付', width='small'),
                        '場R': st.column_config.TextColumn('場R', width='small'),
                        '元': st.column_config.TextColumn('元', width='small'),
                    },
                    hide_index=True, use_container_width=True, key='notes_table',
                )

                # ── 編集パネル：1件を選んでチップ編集（複数タグ）──
                st.markdown("**✏️ 編集する馬ノートを選択**（馬名で打ち込み検索できます）")
                _idmap = {}
                for _, _r in _f.sort_values('日付', ascending=False).iterrows():
                    # ラベルは 馬名×日付×場R（＝upsertキー）で安定。編集で値が変わってもラベルは不変。
                    _lab = f"{_r['馬名']}（{_md8(_r['日付'])} {_r['開催']}{_r['Ｒ']}R）"
                    _idmap[_lab] = _r
                _opts_list = ['—'] + list(_idmap.keys())
                # 削除等で選択中のラベルが消えたら'—'へ（ウィジェット生成前に補正）
                if st.session_state.get('notes_editpick') not in _opts_list:
                    st.session_state['notes_editpick'] = '—'
                _pick = st.selectbox('馬ノート', _opts_list, key='notes_editpick', label_visibility='collapsed')
                if _pick != '—' and _pick in _idmap:
                    _r = _idmap[_pick]
                    _nid = str(_r['id'])
                    _ev0 = str(_r.get('評価', '中立') or '中立')
                    _aim0 = int(pd.to_numeric(_r.get('狙い度', 2), errors='coerce') or 2)
                    _tags0 = str(_r.get('タグ', '') or '')
                    _memo0 = str(_r.get('メモ', '') or '')
                    _ek = f"edit_{_nid}"
                    _e1, _e2 = st.columns([3, 2])
                    with _e1:
                        st.session_state.setdefault(f'{_ek}_ev', _ev0 if _ev0 in _wh.EVAL_OPTIONS else None)
                        st.pills('評価', _wh.EVAL_OPTIONS, selection_mode='single', key=f'{_ek}_ev')
                    with _e2:
                        st.session_state.setdefault(f'{_ek}_aim', _aim0 if _aim0 in (1, 2, 3) else 2)
                        st.pills('狙い★', [1, 2, 3], selection_mode='single',
                                 format_func=lambda x: '★' * x, key=f'{_ek}_aim',
                                 help='★1=軽め／★2=標準／★3=本気')
                    _cur = set(_tags0.split('・'))
                    for _grp, _opts in _wh.TAG_GROUPS.items():
                        st.session_state.setdefault(f'{_ek}_tag_{_grp}', [t for t in _opts if t in _cur])
                        st.pills(_grp, _opts, selection_mode='multi', key=f'{_ek}_tag_{_grp}')
                    st.session_state.setdefault(f'{_ek}_memo', _memo0)
                    st.text_input('メモ', key=f'{_ek}_memo')
                    _b1, _b2, _b3 = st.columns([1, 1, 3])
                    with _b1:
                        if st.button('💾 更新', type='primary', key=f'upd_{_nid}'):
                            _nev = st.session_state.get(f'{_ek}_ev') or '中立'
                            _naim = st.session_state.get(f'{_ek}_aim') or 2
                            _ntags = []
                            for _grp in _wh.TAG_GROUPS:
                                _ntags += list(st.session_state.get(f'{_ek}_tag_{_grp}') or [])
                            _nmemo = str(st.session_state.get(f'{_ek}_memo', '') or '').strip()
                            _wh.add_note(_r['馬名'], _r['日付'], 評価=_nev, 狙い度=int(_naim),
                                         タグ=_ntags, メモ=_nmemo, 開催=_r.get('開催', ''), Ｒ=_r.get('Ｒ', ''),
                                         レース名=_r.get('レース名', ''), ソース=str(_r.get('ソース', '編集') or '編集'))
                            st.success(f"{_r['馬名']} を更新しました。")
                            st.rerun()
                    with _b2:
                        if st.button('🗑 削除', key=f'del_{_nid}'):
                            _wh.delete_note(_r['id'])
                            st.rerun()

        # ── 保存先の設定（Googleスプレッドシート直結）＋CSV一括取込 ─────────────
        st.divider()
        with st.expander("💾 保存先の設定（Googleスプレッドシート直結）／CSV一括取込", expanded=(_stat['mode'] != 'sheets')):
            if _stat['mode'] == 'sheets':
                st.success("接続OK。馬ノートはGoogleスプレッドシートに保存されています。"
                           "シートを直接開いて行を追加・編集・削除すれば、そのままアプリに反映されます。")
                try:
                    _sheet_url = str(_ss._get_secret('race_notes_sheet', ''))
                    if _sheet_url.startswith('http'):
                        st.markdown(f"🔗 [馬ノートのスプレッドシートを開く]({_sheet_url})")
                except Exception:
                    pass
            else:
                st.markdown(
                    "**永続化＋スプレッドシート一括更新の初期設定（1回だけ）:**\n\n"
                    "1. Google Cloud で新規プロジェクト → **Google Sheets API** と **Google Drive API** を有効化\n"
                    "   （組織で鍵発行が禁止されている場合は、**個人の@gmail.comアカウント**でプロジェクトを作成）\n"
                    "2. **サービスアカウント**を作成 → 鍵(JSON)を作成しダウンロード\n"
                    "3. Googleスプレッドシートを新規作成し、URLをコピー\n"
                    "4. そのシートを、サービスアカウントのメール "
                    "（`xxx@xxx.iam.gserviceaccount.com`）に**編集者**として共有\n"
                    "5. Streamlit Cloud の **Settings → Secrets** に貼り付け（ローカルは "
                    "`.streamlit/secrets.toml`）。**下の「かんたん方式」がおすすめ**:\n"
                )
                st.markdown("**◆ かんたん方式（鍵JSONをまるごと貼る・失敗しにくい）**")
                st.code(
                    "gcp_service_account_json = '''\n"
                    "<ここにダウンロードした鍵JSONファイルの中身をまるごと貼り付け>\n"
                    "'''\n\n"
                    'race_notes_sheet = "https://docs.google.com/spreadsheets/d/XXXX/edit"',
                    language='toml')
                st.caption("鍵JSONの中身（{ から } まで全部）を、上のように三連クオート ''' … ''' の中へ丸ごと貼るだけ。"
                           "改行やprivate_keyの整形を気にせず済みます。")
                st.markdown("**◆ 従来方式（項目ごとに書く）**")
                st.code(
                    '[gcp_service_account]\n'
                    'type = "service_account"\n'
                    'project_id = "..."\n'
                    'private_key = "-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n"\n'
                    'client_email = "xxx@xxx.iam.gserviceaccount.com"\n'
                    '# 他の項目も鍵JSONの通りに...\n\n'
                    'race_notes_sheet = "https://docs.google.com/spreadsheets/d/XXXX/edit"',
                    language='toml')
                st.caption("設定後にアプリを再起動すると🟢になります。"
                           "※各行は必ず「名前 = 値」の形。名前のない値だけの行があるとTOMLエラーになります。"
                           "※認証情報はsecretsにのみ置き、gitには絶対に上げないでください。")

            st.markdown("**CSVで一括取込・更新（既存の内容を置き換え）**")
            st.caption("スプレッドシートから書き出したCSV（列: 馬名/日付/開催/Ｒ/レース名/評価/狙い度/タグ/メモ 等）を"
                       "アップロードすると全件を置き換えます（Sheets設定時はシートにも反映）。バックアップからの復元にも使えます。")
            _all_notes = _wh.load_notes()
            _bk1, _bk2 = st.columns(2)
            with _bk1:
                if not _all_notes.empty:
                    st.download_button("⬇️ 現在の馬ノートをCSV保存", key='dl_notes',
                                       data=_all_notes.to_csv(index=False).encode('utf-8-sig'),
                                       file_name="race_notes.csv", mime="text/csv")
            with _bk2:
                _upn = st.file_uploader("CSVを取込（全件置換）", type='csv', key='up_notes')
                if _upn:
                    _dfu = pd.read_csv(_upn, dtype=str)
                    _n = _wh.replace_all(_dfu)
                    st.success(f"馬ノートを取り込みました（{_n}件・全件置換）。")
                    st.rerun()
    except Exception as _wh_err:
        import traceback as _tbw
        st.error(f"馬ノートタブ エラー: {_wh_err}")
        st.code(_tbw.format_exc())


# ── 管理者専用タブの描画（閲覧モードでは tab5/tab6 は生成されず非表示）─────────
if _is_admin and tab5 is not None and tab6 is not None:
    with tab5:
        _render_tracking_tab()
    with tab6:
        _render_watch_tab()


# ── ✅ 予測精度モニタリング（閲覧者含む全員に表示）──────────────────────────
with tab7:
    st.subheader("✅ 予測精度モニタリング")
    st.caption("過去レースを現行モデルでリーク無し再予測した実績。モデルがどれだけ当たっているかの継続チェック。")
    try:
        from accuracy_monitor import (load_accuracy_log, summary_metrics,
                                       weekly_trend, calibration_bins, _filter_period)
        _alog = load_accuracy_log()
    except Exception as _ae:
        _alog = None
        st.error(f"精度ログ読込エラー: {_ae}")

    if _alog is None or _alog.empty:
        st.info("精度ログが未生成です。`python src/accuracy_monitor.py` で生成できます（学習時に自動再生成）。")
    else:
        _pmap = {'直近1ヶ月': 1, '直近3ヶ月': 3, '直近1年': 12}
        _psel = st.radio("期間", list(_pmap.keys()), index=1, horizontal=True, key='acc_period')
        _adf = _filter_period(_alog, _pmap[_psel])
        _s = summary_metrics(_adf)

        _c1, _c2, _c3, _c4, _c5 = st.columns(5)
        _c1.metric("◎本命 複勝率", f"{_s['hon_fuku']*100:.0f}%")
        _c2.metric("◎本命 勝率", f"{_s['hon_win']*100:.0f}%")
        _c3.metric("上位3頭 複勝率", f"{_s['top3_any']*100:.0f}%",
                   help="◎○▲のうち1頭以上が3着内に来た割合")
        _c4.metric("順位相関", f"{_s['corr']:.2f}",
                   help="予測順位×実着順のスピアマン相関。1に近いほど順位が的中。")
        _c5.metric("対象レース", f"{_s['n_races']:,}")

        # 週別トレンド
        _w = weekly_trend(_adf)
        if not _w.empty:
            st.markdown("##### 開催週ごとの ◎本命 複勝率")
            import plotly.graph_objects as _go
            _fig = _go.Figure()
            _fig.add_trace(_go.Scatter(x=_w['_week'], y=(_w['hon_fuku']*100).round(1),
                                       mode='lines+markers', name='◎複勝率',
                                       line=dict(color='#3498db', width=2.5)))
            _fig.add_hline(y=float((_adf[_adf['pop'] == 1]['chk'] <= 3).mean()*100),
                           line_dash='dash', line_color='#888',
                           annotation_text='市場(1番人気)基準', annotation_position='top left')
            _fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                               yaxis_title='複勝率(%)', showlegend=False,
                               yaxis=dict(range=[30, 80]))
            st.plotly_chart(_fig, use_container_width=True)

        # 較正
        _cal = calibration_bins(_adf)
        if not _cal.empty:
            st.markdown("##### 予想複勝% の的中較正（予測 vs 実際）")
            st.caption("予測と実際のズレが小さいほど「予想複勝%」が信頼できる。")
            import plotly.graph_objects as _go2
            # x は「予測複勝%の帯」。等間隔のカテゴリとして並べる
            # （数値軸だと帯の平均値の位置に不均等配置され、空白の帯ができてしまう）
            if 'lo' in _cal.columns:
                _labels = [f"{r.lo*100:.0f}〜{r.hi*100:.0f}%" for r in _cal.itertuples()]
            else:
                _labels = [f"{r.pred*100:.0f}%" for r in _cal.itertuples()]
            _ns = _cal['n'].tolist()
            _fig2 = _go2.Figure()
            _fig2.add_trace(_go2.Bar(x=_labels, y=(_cal['pred']*100).round(1), name='予測',
                                     marker_color='#95a5a6', customdata=_ns,
                                     hovertemplate='帯 %{x}<br>予測 %{y:.1f}%<br>頭数 %{customdata}<extra></extra>'))
            _fig2.add_trace(_go2.Bar(x=_labels, y=(_cal['actual']*100).round(1), name='実際',
                                     marker_color='#3498db', customdata=_ns,
                                     hovertemplate='帯 %{x}<br>実際 %{y:.1f}%<br>頭数 %{customdata}<extra></extra>'))
            _fig2.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                                barmode='group', xaxis_title='予測複勝%の帯（同数ずつ6分割）',
                                yaxis_title='複勝率(%)',
                                xaxis=dict(type='category'),
                                legend=dict(orientation='h', y=1.1))
            st.plotly_chart(_fig2, use_container_width=True)
            st.caption('各帯を「予測の平均（灰）」と「実際に3着内に来た割合（青）」で比較。'
                       '同じ高さなら予想複勝%が信頼できる。青が低い＝過大評価、青が高い＝過小評価。')

        # 穴モデル併記
        st.markdown("##### 穴モデル・妙味の精度（参考）")
        _a1, _a2 = st.columns(2)
        _a1.metric("穴モデル1位 複勝率", f"{_s['anaba1_fuku']*100:.0f}%",
                   help="穴馬モデルの最上位馬が3着内に来た割合")
        _a2.metric("★妙味馬 複勝率", f"{_s['star_fuku']*100:.0f}%",
                   help=f"連下×人気薄の妙味馬（★）の複勝率。対象{_s['star_n']}頭")
        st.caption("※検証では機械的な馬券は控除率の壁で+EVになりません。この精度は「当てる・取捨の参考」の指標です。")





# -- 開催の俯瞰（鉄板馬・妙味馬）／閲覧者含む全員に表示 --
with tab8:
    st.subheader("🗓 開催の俯瞰（鉄板馬・妙味馬）")
    st.caption("公開中の予測から、鉄板馬（本命の予想複勝80%以上）と妙味馬（見落とし注意）を発走時刻順に一覧。")
    _pdf = st.session_state.get('pred_df')
    if _pdf is None or _pdf.empty:
        st.info("予測データがありません。（管理者が予測を公開するとここに一覧が出ます）")
    else:
        from pred_utils import softmax_probs as _sfx_cr
        try:
            from fuku_calibration import calibrated_fuku_prob as _cfp_cr
        except Exception:
            _cfp_cr = None
        _cr = _pdf.copy()
        _cr['_ds'] = _cr['日付'].astype(str)
        _dates_cr = sorted(_cr['_ds'].unique())
        _dsel = st.selectbox("日付", _dates_cr, index=len(_dates_cr) - 1,
                             format_func=lambda d: (f"20{d[2:4]}/{d[4:6]}/{d[6:]}" if len(d) == 8
                                                    else (f"20{d[:2]}/{d[2:4]}/{d[4:]}" if len(d) == 6 else d)),
                             key='cross_date')
        _day = _cr[_cr['_ds'] == _dsel].copy()
        _day['pop'] = pd.to_numeric(_day['人気'], errors='coerce')
        _day['_umaban'] = pd.to_numeric(_day['馬番'], errors='coerce')
        _day['rk'] = _day['開催'].astype(str) + '_' + _day['Ｒ'].astype(str)
        _day['pred_rank'] = _day.groupby('rk')['pred_score'].rank(ascending=False, method='first')
        _day['win_prob'] = _day.groupby('rk')['pred_score'].transform(_sfx_cr)
        _day['fp'] = (_cfp_cr(_day['win_prob']).values if _cfp_cr is not None
                      else (_day['win_prob'] * 3).clip(upper=0.95).values)

        _date8 = ('20' + _dsel) if len(_dsel) == 6 else _dsel
        _tk = f'_cross_times_{_dsel}'
        _ok = f'_cross_odds_{_dsel}'
        _pk = f'_cross_pop_{_dsel}'
        if st.button("🕐 発走時刻・単勝オッズを取得", key=f'fetch_to_{_dsel}',
                     help="Netkeibaから各レースの発走時刻・単勝オッズ・人気を取得します"):
            from scrape_odds import build_race_id as _bri_t, fetch_race_info as _fri_t, fetch_odds_tan as _fot_t
            from concurrent.futures import ThreadPoolExecutor as _TPEt, as_completed as _asct
            _rid_of = {}
            for _rkx in sorted(_day['rk'].unique()):
                _kx, _rx = _rkx.rsplit('_', 1)
                try:
                    _rid_of[_rkx] = _bri_t(_date8, _kx, int(_rx))
                except Exception:
                    _rid_of[_rkx] = None

            def _fetch_one(_rk, _rid):
                _t, _od, _pp = '', {}, {}
                try:
                    _t = _fri_t(_rid).get('time', '')
                except Exception:
                    pass
                try:
                    _df = _fot_t(_rid)
                    if _df is not None and not _df.empty:
                        _od = dict(zip(_df['馬番'].astype(int), _df['単勝オッズ'].astype(float)))
                        _pp = dict(zip(_df['馬番'].astype(int), pd.to_numeric(_df['人気'], errors='coerce')))
                except Exception:
                    pass
                return _rk, _t, _od, _pp
            _tmap, _omap, _pmap = {}, {}, {}
            with st.spinner("発走時刻・単勝オッズを取得中…"):
                with _TPEt(max_workers=8) as _ext:
                    _futs = {_ext.submit(_fetch_one, _rk, _rid): _rk for _rk, _rid in _rid_of.items() if _rid}
                    for _f in _asct(_futs):
                        try:
                            _rk2, _t2, _o2, _p2 = _f.result()
                            _tmap[_rk2] = _t2
                            _omap[_rk2] = _o2
                            _pmap[_rk2] = _p2
                        except Exception:
                            pass
            st.session_state[_tk] = _tmap
            st.session_state[_ok] = _omap
            st.session_state[_pk] = _pmap
        _times = st.session_state.get(_tk, {})
        _odds_map = st.session_state.get(_ok, {})
        _pop_map = st.session_state.get(_pk, {})

        # 実効人気: 出馬表の人気(あれば) → 無ければ取得オッズ由来の人気
        def _eff_pop(_rk, _umaban, _csv_pop):
            if pd.notna(_csv_pop) and _csv_pop > 0:
                return int(_csv_pop)
            try:
                _p = _pop_map.get(_rk, {}).get(int(_umaban))
                return int(_p) if pd.notna(_p) and _p > 0 else 0
            except Exception:
                return 0
        _day['eff_pop'] = _day.apply(lambda r: _eff_pop(r['rk'], r['_umaban'], r['pop']), axis=1)
        _has_pop = (_day['eff_pop'] > 0).any()

        def _ostr(_rk, _umaban):
            try:
                _o = _odds_map.get(_rk, {}).get(int(_umaban))
                return f'{_o:.1f}倍' if _o else ''
            except Exception:
                return ''

        _tetsu, _myomi_all = [], []
        for _rk, _g in _day.groupby('rk'):
            _hon = _g[_g['pred_rank'] == 1].iloc[0]
            _ven = parse_venue(str(_hon['開催']))
            _rno = int(pd.to_numeric(_hon['Ｒ'], errors='coerce') or 0)
            _t = _times.get(_rk, '')
            _fpv = round(float(_hon['fp']) * 100)
            if _fpv >= 80:
                _tetsu.append({'time': _t, '会場R': f'{_ven}{_rno}R', 'R': _rno, '会場': _ven,
                               '馬名': str(_hon['馬名']),
                               '人気': int(_hon['eff_pop']) if _hon['eff_pop'] > 0 else 0,
                               '予想複勝': _fpv, 'オッズ': _ostr(_rk, _hon['_umaban'])})
            for _, _mr in _g[(_g['pred_rank'] <= 4) & (_g['eff_pop'] >= 6)].sort_values('pred_rank').iterrows():
                _myomi_all.append({'time': _t, '会場R': f'{_ven}{_rno}R', 'R': _rno, '会場': _ven,
                                   '馬名': str(_mr['馬名']), '人気': int(_mr['eff_pop']),
                                   '通常順位': int(_mr['pred_rank']),
                                   '予想複勝': round(float(_mr['fp']) * 100),
                                   'オッズ': _ostr(_rk, _mr['_umaban'])})

        def _chrono(rows):
            df = pd.DataFrame(rows)
            if df.empty:
                return df
            if df['time'].astype(str).str.len().gt(0).any():
                return df.assign(_x=df['time'].replace('', '99:99')).sort_values('_x')
            return df.sort_values(['R', '会場'])
        _tdf = _chrono(_tetsu)
        _mdf = _chrono(_myomi_all)

        _m1, _m2, _m3 = st.columns(3)
        _m1.metric("レース数", f"{_day['rk'].nunique()}")
        _m2.metric("🔥 鉄板馬", f"{len(_tetsu)}", help="本命の予想複勝率が80%以上")
        _m3.metric("💡 妙味馬", f"{len(_myomi_all)}", help="通常4位以内×人気6番以下の見落とし注意馬")
        if not _times:
            st.caption("※発走時刻・単勝オッズ・人気は上のボタンで取得（未取得時は会場・R順、出馬表段階は人気未確定）。")

        def _pstr(_p):
            return f'（{_p}人気）' if _p and _p > 0 else '（人気未定）'

        # レスポンシブCSS: PCは1行、スマホ(≤640px)は2行に折り返し馬名を独立行に。
        # 固定幅要素で馬名領域が潰れ1文字ずつ改行される問題を解消する。
        st.markdown("""
<style>
.crow{display:flex;align-items:center;gap:8px;padding:5px 8px;border-bottom:1px solid #21262d;flex-wrap:wrap;}
.crow-main{display:flex;align-items:center;gap:8px;flex:1 1 auto;min-width:0;}
.crow-time{min-width:42px;color:#8b949e;font-size:0.88em;}
.crow-venue{min-width:60px;font-weight:bold;color:#e6edf3;}
.crow-name{flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.crow-detail{display:flex;align-items:center;gap:8px;flex:0 0 auto;}
.crow-mid{min-width:44px;font-size:0.84em;}
.crow-fp{min-width:70px;text-align:right;font-weight:bold;color:#2ecc71;}
.crow-odds{min-width:52px;text-align:right;color:#f1c40f;font-size:0.9em;}
@media (max-width:640px){
  .crow-detail{flex:1 1 100%;padding-left:44px;}
  .crow-fp{min-width:auto;text-align:left;}
  .crow-odds{min-width:auto;text-align:left;margin-left:auto;}
}
</style>
""", unsafe_allow_html=True)

        def _row(_r, _name_color, _pop_color, _mid):
            _tstr = _r['time'] if _r['time'] else '—'
            _od = f'<span class="crow-odds">{_r["オッズ"]}</span>' if _r['オッズ'] else '<span class="crow-odds"></span>'
            return (
                f'<div class="crow">'
                f'<div class="crow-main">'
                f'<span class="crow-time">{_tstr}</span>'
                f'<span class="crow-venue">{_r["会場R"]}</span>'
                f'<span class="crow-name" style="color:{_name_color};">{_r["馬名"]}'
                f'<span style="color:{_pop_color};font-size:0.83em;">{_pstr(_r["人気"])}</span></span>'
                f'</div>'
                f'<div class="crow-detail">'
                f'{_mid}'
                f'<span class="crow-fp">複勝{_r["予想複勝"]}%</span>'
                f'{_od}</div>'
                f'</div>')

        st.markdown("##### 🔥 鉄板馬（本命の予想複勝80%以上）")
        if _tdf.empty:
            st.caption("この日は鉄板馬（複勝80%以上）はありません。")
        else:
            for _, _r in _tdf.iterrows():
                st.markdown(_row(_r, '#e6edf3', '#8b949e',
                                 '<span class="crow-mid" style="color:#e74c3c;">🔥本命</span>'),
                            unsafe_allow_html=True)

        st.markdown("##### 💡 妙味馬（見落とし注意・通常上位×人気薄）")
        if not _has_pop:
            st.caption("人気が未確定です。上の「発走時刻・単勝オッズを取得」ボタンで人気を反映すると妙味馬が判定されます。")
        elif _mdf.empty:
            st.caption("この日は妙味馬（通常4位以内×人気6番以下）はありません。")
        else:
            for _, _r in _mdf.iterrows():
                st.markdown(_row(_r, '#e6edf3', '#e67e22',
                                 f'<span class="crow-mid" style="color:#8b949e;">通常{_r["通常順位"]}位</span>'),
                            unsafe_allow_html=True)
