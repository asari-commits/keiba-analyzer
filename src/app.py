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

def load_master_summary():
    df = pd.read_parquet(MASTER_PARQUET)
    return df.head(5000)

def load_master_full():
    df = pd.read_parquet(MASTER_PARQUET)
    df['距離'] = pd.to_numeric(df['距離'], errors='coerce')
    df['着順_num'] = pd.to_numeric(
        df['着順'].astype(str).str.translate(str.maketrans('０１２３４５６７８９','0123456789')),
        errors='coerce'
    )
    df['人気'] = pd.to_numeric(df['人気'], errors='coerce')
    df['前走上り3F'] = pd.to_numeric(df['前走上り3F'], errors='coerce')
    df['_venue_name'] = df['開催'].astype(str).str.extract(r'\d([^\d]+)\d')[0].map(VENUE_MAP)
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

def _build_honmei_line(info: dict) -> str:
    """本命◎/対抗○/穴△ サマリー行のHTML"""
    if not info:
        return ''
    parts = []
    for label, color, badge in [('本命', '#f1c40f', '◎'), ('対抗', '#2ecc71', '○'), ('穴', '#c39bd3', '△')]:
        h = info.get(label)
        if h:
            ev_str   = f" EV{h['ev']:+.0f}%" if pd.notna(h.get('ev')) else ''
            fuku_str = f" 馬券内{h['fuku']:.0%}" if h.get('fuku') else ''
            conf     = h.get('conf', 0)
            conf_str = f" 信頼{'高' if conf >= 0.7 else ('中' if conf >= 0.4 else '低')}" if conf else ''
            parts.append(
                f'<span style="color:{color};font-weight:bold;">{badge}{label}</span>'
                f'<span style="color:white;margin-left:2px;">{h["name"]}（{h["pop"]}人気）</span>'
                f'<span style="color:#aaa;font-size:0.85em;">{fuku_str}{conf_str}{ev_str}</span>'
            )
    if not parts:
        return ''
    race_type = info.get('race_type', '')
    type_str = f'<span style="color:#666;font-size:0.8em;margin-left:8px;">[{race_type}]</span>' if race_type else ''
    return '<br><span style="font-size:0.9em;">' + '&nbsp;&nbsp;'.join(parts) + type_str + '</span>'


def parse_venue(kai_str: str) -> str:
    """
    '1東3' → '東京'  （Target形式）
    '東京' → '東京'  （netkeiba scrape形式 / フォールバック）
    """
    import re
    s = str(kai_str)
    m = re.search(r'\d([^\d]+)\d', s)
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

# モバイル判定（streamlit_js_eval は廃止・PC固定）
_is_mobile = False

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
        if '馬番' in _auto.columns and _auto['馬番'].isna().all():
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
    if '馬番' in _chk.columns and _chk['馬番'].isna().all():
        LAST_PRED_PATH.unlink(missing_ok=True)
        del st.session_state['pred_df']
        st.session_state['_stale_parquet'] = True

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 レース予測", "💰 回収率シミュレーション", "📋 データ確認", "🔍 データベース検索", "📈 回収率トラッキング"])


# ============================================================
# Tab 1: レース予測
# ============================================================
with tab1:
    st.subheader("レース予測")
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
        st.info("上の「データ読み込み設定」からデータを読み込んでください。")
        st.stop()

    if '_auto_load_ts' in st.session_state:
        st.caption(f"💾 前回の予測結果を自動ロードしました（保存日時: {st.session_state['_auto_load_ts']}）　新しいCSVを読み込むには上のパネルから予測実行してください。")

    pred_df = st.session_state['pred_df'].copy()

    # メタ列を付与
    pred_df['_date_str']   = pred_df['日付'].astype(str).str.zfill(8)
    pred_df['_year']       = pred_df['_date_str'].str[:4]
    pred_df['_month']      = pred_df['_date_str'].str[4:6].str.lstrip('0')
    pred_df['_day']        = pred_df['_date_str'].str[6:8].str.lstrip('0')
    pred_df['_venue_name'] = pred_df['開催'].astype(str).apply(parse_venue)
    pred_df['_r_num']      = pd.to_numeric(pred_df['Ｒ'], errors='coerce').fillna(0).astype(int)
    pred_df['_race_name']  = pred_df['レース名'].fillna('') if 'レース名' in pred_df.columns else ''

    st.markdown("---")

    from datetime import datetime as _dt
    _DOW = ['月','火','水','木','金','土','日']

    def _btn_grid(label, options, state_key, cols=6):
        """ボタングリッドで1つを選択。選択中は primary、それ以外は secondary。"""
        if state_key not in st.session_state or st.session_state[state_key] not in options:
            st.session_state[state_key] = options[0]
        st.markdown(f"<div style='color:#aaa;font-size:0.85em;margin-bottom:4px;'>{label}</div>",
                    unsafe_allow_html=True)
        rows = [options[i:i+cols] for i in range(0, len(options), cols)]
        for row in rows:
            btn_cols = st.columns(len(row))
            for col, opt in zip(btn_cols, row):
                is_sel = st.session_state[state_key] == opt
                if col.button(opt, key=f'nav_{state_key}_{opt}',
                              type='primary' if is_sel else 'secondary',
                              use_container_width=True):
                    st.session_state[state_key] = opt
                    st.rerun()
        return st.session_state[state_key]

    # ① 年
    years = sorted(pred_df['_year'].dropna().unique(), reverse=True)
    sel_year = _btn_grid("📅 年", list(map(str, years)), 'sel_year', cols=len(years))
    df_y = pred_df[pred_df['_year'] == sel_year]

    st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)

    # ② 月
    months = sorted(df_y['_month'].dropna().unique(), key=lambda x: int(x) if x.isdigit() else 0)
    month_opts = [f"{m}月" for m in months]
    sel_month_label = _btn_grid("📅 月", month_opts, 'sel_month', cols=len(month_opts))
    sel_month_num = sel_month_label.replace('月', '')
    df_m = df_y[df_y['_month'] == sel_month_num]

    st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)

    # ③ 日（開催日）
    days = sorted(df_m['_day'].dropna().unique(), key=lambda x: int(x) if x.isdigit() else 0)
    if not days:
        st.warning("この月のデータがありません。")
        st.stop()

    def day_label(d):
        try:
            dt = _dt.strptime(f"{sel_year}{sel_month_num.zfill(2)}{d.zfill(2)}", '%Y%m%d')
            dow = _DOW[dt.weekday()]
            return f"{int(d)}日({dow})"
        except Exception:
            return f"{int(d)}日"

    day_opts = [day_label(d) for d in days]
    if 'sel_day' not in st.session_state or st.session_state['sel_day'] not in day_opts:
        st.session_state['sel_day'] = day_opts[0]

    st.markdown("<div style='color:#aaa;font-size:0.85em;margin-bottom:6px;'>📅 開催日</div>",
                unsafe_allow_html=True)
    day_cols = st.columns(len(day_opts))
    for col, opt in zip(day_cols, day_opts):
        is_sel = st.session_state['sel_day'] == opt
        if col.button(opt, key=f'nav_sel_day_{opt}',
                      type='primary' if is_sel else 'secondary',
                      use_container_width=True):
            st.session_state['sel_day'] = opt
            st.rerun()
    sel_day_label = st.session_state['sel_day']
    sel_day_num = sel_day_label.split('日')[0].zfill(2)
    df_d = df_m[df_m['_day'] == sel_day_num.lstrip('0')]

    # ④ 競馬場（開催ごとにタブ表示）
    venues_in_month = [v for v in VENUE_ORDER if v in df_d['_venue_name'].unique()]
    if not venues_in_month:
        st.warning("この日のデータがありません。")
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

            # ── レース選択: PC=ボタン1行 / スマホ=プルダウン ───────────
            st.markdown("<div style='margin-top:12px;margin-bottom:4px;color:#aaa;font-size:0.85em;'>🏁 レース選択</div>", unsafe_allow_html=True)

            _r_opts = {
                (f"{r}R 🌿芝" if r_surf_map.get(r) == '芝' else (f"{r}R 🟤ダ" if r_surf_map.get(r) == 'ダ' else f"{r}R")): r
                for r in r_nums
            }

            if _is_mobile:
                # スマホ: プルダウン
                _cur_label = next((lb for lb, rv in _r_opts.items() if rv == st.session_state[state_key]), list(_r_opts.keys())[0])
                _sel_label = st.selectbox("レース番号", list(_r_opts.keys()),
                                          index=list(_r_opts.keys()).index(_cur_label),
                                          key=f'rsel_{v_name}', label_visibility='collapsed')
                if _r_opts[_sel_label] != st.session_state[state_key]:
                    st.session_state[state_key] = _r_opts[_sel_label]
                    st.rerun()
            else:
                # PC: ボタン1行
                r_cols = st.columns(max(len(r_nums), 1))
                for col, r in zip(r_cols, r_nums):
                    is_sel = st.session_state[state_key] == r
                    _s = r_surf_map.get(r, '')
                    surf_emoji = '🌿' if _s == '芝' else ('🟤' if _s == 'ダ' else '')
                    label = f"{r}R {surf_emoji}" if _s else f"{r}R"
                    if col.button(label, key=f'rbtn_{v_name}_{r}',
                                  type="primary" if is_sel else "secondary",
                                  use_container_width=True):
                        st.session_state[state_key] = r
                        st.rerun()

            # ── 全R一括オッズ取得 ───────────────────────────────────────
            if st.button(f"⚡ 全R一括オッズ取得 ({len(r_nums)}R分)",
                         key=f'bulk_odds_{v_name}',
                         help="この競馬場の全レースのオッズをまとめて取得します"):
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
                st.success(f"✅ {_ok}/{len(r_nums)}R のオッズを取得しました")

            sel_r = st.session_state[state_key]
            show_df = df_v[df_v['_r_num'] == sel_r].copy()

            # 同一馬の重複行を除去（複数ファイル読み込み時など）
            show_df = show_df.drop_duplicates(subset=['馬名'], keep='first')

            # pred_rank を show_df 内で再計算（スコア順に1位から振り直し）
            if 'pred_score' in show_df.columns and not show_df.empty:
                show_df['pred_rank'] = show_df['pred_score'].rank(ascending=False, method='min').astype(int)
            if 'pred_score_anaba' in show_df.columns and not show_df.empty:
                show_df['pred_rank_anaba'] = show_df['pred_score_anaba'].rank(ascending=False, method='min').astype(int)

            if show_df.empty:
                st.info("データがありません。")
                continue

            # ── オッズ入力 ────────────────────────────────────────────
            live_odds_key = f'live_odds_{v_name}_{sel_r}'

            with st.expander("📊 オッズ・人気（自動取得 or 手動入力）", expanded=st.session_state.get(live_odds_key) is None):
                # 自動取得ボタン
                _ao1, _ao2, _ao3 = st.columns([2, 2, 6])
                with _ao1:
                    if st.button("🔄 自動取得", key=f'fetch_odds_{v_name}_{sel_r}',
                                 help="Netkeibaから単勝オッズを自動取得します"):
                        try:
                            from scrape_odds import build_race_id, _SESSION, _parse_tan_json
                            import requests as _req

                            if '_race_id' in show_df.columns and not show_df.empty:
                                _race_id = str(show_df['_race_id'].iloc[0])
                            else:
                                _date_str = str(show_df['日付'].iloc[0]) if not show_df.empty else ''
                                _kaisai   = str(show_df['開催'].iloc[0]) if not show_df.empty else ''
                                _race_id  = build_race_id(_date_str, _kaisai, sel_r)

                            if not _race_id:
                                st.error("レースIDを構築できませんでした（場所コード不明）")
                            else:
                                _api_url = (f"https://race.netkeiba.com/api/api_get_jra_odds.html"
                                            f"?race_id={_race_id}&type=1&action=init")
                                with st.spinner(f"取得中... ({_race_id})"):
                                    try:
                                        _resp = _SESSION.get(_api_url, timeout=10)
                                        from scrape_odds import _parse_tan_json
                                        _odds = _parse_tan_json(_resp.json())
                                    except _req.exceptions.RequestException as _re:
                                        st.error(f"接続エラー: {_re}")
                                        _odds = None
                                if _odds is None:
                                    pass
                                elif _odds.empty:
                                    st.warning("オッズが取得できませんでした（発売前 or レースID不一致）")
                                else:
                                    st.session_state[live_odds_key] = _odds
                                    st.session_state[f'live_odds_time_{v_name}_{sel_r}'] = \
                                        __import__('datetime').datetime.now().strftime('%H:%M:%S')
                                    st.rerun()
                        except Exception as _e:
                            st.error(f"取得エラー: {_e}")
                with _ao2:
                    if st.button("🗑️ リセット", key=f'clear_odds_{v_name}_{sel_r}'):
                        st.session_state.pop(live_odds_key, None)
                        st.rerun()
                with _ao3:
                    _ot = st.session_state.get(f'live_odds_time_{v_name}_{sel_r}')
                    if _ot:
                        st.caption(f"📡 取得済 {_ot}")

                st.divider()
                st.caption("自動取得できない場合は下表に手動入力して「✅ 反映」を押してください。")

                # 現在セッションに保存済みの値があれば初期値として使う
                _saved_odds = st.session_state.get(live_odds_key)
                _horses_for_input = show_df[['馬番', '馬名']].drop_duplicates('馬番').sort_values('馬番') if '馬番' in show_df.columns else show_df[['馬名']].drop_duplicates()

                if _saved_odds is not None and not _saved_odds.empty and '馬番' in _horses_for_input.columns:
                    _init_df = _horses_for_input.merge(_saved_odds[['馬番', '人気', '単勝オッズ']], on='馬番', how='left')
                else:
                    _init_df = _horses_for_input.copy()
                    _init_df['人気']     = None
                    _init_df['単勝オッズ'] = None

                _init_df = _init_df.reset_index(drop=True)
                # 型を明示
                _init_df['人気']     = pd.to_numeric(_init_df['人気'],     errors='coerce')
                _init_df['単勝オッズ'] = pd.to_numeric(_init_df['単勝オッズ'], errors='coerce')

                _edited_odds = st.data_editor(
                    _init_df,
                    column_config={
                        '馬番':     st.column_config.NumberColumn('馬番',     disabled=True, width='small'),
                        '馬名':     st.column_config.TextColumn('馬名',       disabled=True, width='medium'),
                        '人気':     st.column_config.NumberColumn('人気',     min_value=1, max_value=28, step=1, width='small'),
                        '単勝オッズ': st.column_config.NumberColumn('単勝オッズ', min_value=1.0, format='%.1f', width='small'),
                    },
                    hide_index=True,
                    use_container_width=False,
                    key=f'odds_editor_{v_name}_{sel_r}',
                )

                _btn_col1, _btn_col2 = st.columns([2, 6])
                with _btn_col1:
                    if st.button("✅ 反映", key=f'apply_odds_{v_name}_{sel_r}', type='primary'):
                        _filled = _edited_odds.dropna(subset=['人気', '単勝オッズ'])
                        if _filled.empty:
                            st.warning("人気と単勝オッズを1行以上入力してください。")
                        else:
                            _filled = _filled.rename(columns={'単勝オッズ': '単勝オッズ'})
                            st.session_state[live_odds_key] = _filled[['馬番', '単勝オッズ', '人気']].copy()
                            st.session_state[f'live_odds_time_{v_name}_{sel_r}'] = \
                                __import__('datetime').datetime.now().strftime('%H:%M:%S')
                            st.rerun()
            _ot = st.session_state.get(f'live_odds_time_{v_name}_{sel_r}')
            if _ot:
                st.caption(f"📡 オッズ反映済 {_ot} — EV計算・サマリーテーブルに適用中")

            show_top_n = 3  # 上位3頭をハイライト（固定）

            from pred_utils import (softmax_probs, calc_ev, confidence_score,
                                     recommend_bet, get_reasons, POPULAR_STATS)

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
                        show_df = show_df.merge(
                            _lo[['馬番', '単勝オッズ', '人気']].rename(
                                columns={'単勝オッズ': '単勝オッズ_live', '人気': '人気_live'}),
                            on='馬番', how='left'
                        )
                        if '人気_live' in show_df.columns and show_df['人気_live'].notna().any():
                            has_live_odds = True
                        else:
                            show_df = show_df.drop(columns=['単勝オッズ_live', '人気_live'], errors='ignore')

            # ── EV計算（show_df が空でなければ常に実行）────────────────
            if not show_df.empty:
                probs = softmax_probs(show_df['pred_score'])
                show_df = show_df.copy()
                show_df['_win_prob'] = probs.values

                if has_live_odds:
                    # 実人気で _pop_int を更新
                    show_df['_pop_int'] = pd.to_numeric(
                        show_df['人気_live'], errors='coerce'
                    ).fillna(
                        pd.to_numeric(show_df['人気'], errors='coerce').fillna(0)
                    ).astype(int)
                    # 実オッズ使用EV: model_prob × actual_odds × 100 - 100
                    def _ev_live(r):
                        o = r.get('単勝オッズ_live')
                        if pd.notna(o) and float(o) > 0:
                            return float(max(r['_win_prob'] * float(o) * 100 - 100, -100.0))
                        return float('nan')
                    show_df['EV単勝'] = show_df.apply(_ev_live, axis=1)
                    # 複勝は引き続き POPULAR_STATS ベース（複勝オッズ未取得）
                    show_df['EV複勝'] = show_df.apply(
                        lambda r: calc_ev(r['_win_prob'] * 2.5, r['_pop_int'], 'fuku'), axis=1)
                else:
                    show_df['_pop_int'] = pd.to_numeric(
                        show_df['人気'], errors='coerce').fillna(0).astype(int)
                    show_df['EV単勝'] = show_df.apply(
                        lambda r: calc_ev(r['_win_prob'], max(int(r['_pop_int']), 1), 'tan'), axis=1)
                    show_df['EV複勝'] = show_df.apply(
                        lambda r: calc_ev(r['_win_prob'] * 2.5, max(int(r['_pop_int']), 1), 'fuku'), axis=1)

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
                    # 予測ログを自動保存
                    try:
                        from result_tracker import save_pred_log as _spl
                        _date_str2 = str(show_df['日付'].iloc[0]) if not show_df.empty else ''
                        _kaisai2   = str(show_df['開催'].iloc[0]) if not show_df.empty else ''
                        from scrape_odds import build_race_id as _bri
                        _rid2 = _bri(_date_str2, _kaisai2, sel_r) or f"{_date_str2}_{_kaisai2}_{sel_r}"
                        _rname2 = str(show_df['レース名'].iloc[0]) if 'レース名' in show_df.columns and not show_df.empty else ''
                        _spl(_rid2, _date_str2, v_name, sel_r, _rname2, show_df, _buy_tickets)
                    except Exception:
                        pass
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

            # 基本条件: 通常モデル1〜2位 AND 穴馬モデル1〜3位 AND 7番人気以上
            _cond_base = _real_pop_known & (_real_pop >= 7) & (_pred_rank_s <= 2) & (_anaba_rank_s <= 3)
            # 特上条件: 基本 AND 通常モデル1位 AND 乖離5以上
            _cond_tokujou = _cond_base & (_pred_rank_s == 1) & (_drift_s >= 5)

            show_df['_is_tokujou'] = _cond_tokujou
            show_df['_is_anaba']   = _cond_base & ~_cond_tokujou

            # 各カテゴリ最大1頭（通常モデル順位でソート）
            _tq = show_df[show_df['_is_tokujou']].sort_values('pred_rank')
            if len(_tq) > 1:
                show_df.loc[_tq.index[1:], '_is_tokujou'] = False
            _an = show_df[show_df['_is_anaba']].sort_values(['pred_rank', 'pred_rank_anaba' if _has_anaba_col else 'pred_rank'])
            if len(_an) > 1:
                show_df.loc[_an.index[1:], '_is_anaba'] = False

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
                from pipeline_target import MASTER_PARQUET as _MCSV
                if _MCSV.exists() and not show_df.empty:
                    _row0   = show_df.iloc[0]
                    _is_turf = str(_row0.get('芝・ダ', '')).startswith('芝')
                    _dist_v  = int(pd.to_numeric(_row0.get('dist_num', _row0.get('距離', 0)), errors='coerce') or 0)
                    _vmap   = {'東': '東京', '中': '中山', '京': '京都', '阪': '阪神',
                               '名': '中京', '小': '小倉', '新': '新潟', '福': '福島',
                               '函': '函館', '札': '札幌'}
                    _kai = str(_row0.get('開催', ''))
                    import re as _re
                    _vm = _re.search(r'\d([^\d]+)\d', _kai)
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

                # ペース傾向ライン
                pace_line = ''
                if _pace_prof and _pace_prof.get('avg_pci') is not None:
                    _pc  = _pace_prof['avg_pci']
                    _pl  = _pace_prof['pci_label']
                    _pcol = _pace_prof['pci_color']
                    _fwr = _pace_prof.get('front_win_rate')
                    _awr = _pace_prof.get('agari_win_rate')
                    _nr  = _pace_prof.get('n_races', 0)
                    _fwr_txt = f"先行勝率{_fwr:.0f}%" if _fwr is not None else ''
                    _awr_txt = f"上り最速勝率{_awr:.0f}%" if _awr is not None else ''
                    pace_line = (
                        f'<br><span style="font-size:0.85em;">'
                        f'📊 コースPCI: <span style="color:{_pcol};font-weight:bold;">'
                        f'{_pc} ({_pl})</span>'
                        f'&nbsp;|&nbsp;{_fwr_txt}&nbsp;|&nbsp;{_awr_txt}'
                        f'&nbsp;<span style="color:#555;">({_nr}R)</span></span>'
                    )

                _dist_str  = f"{int(r_dist)}m" if str(r_dist).replace('.','').isdigit() else f"{r_dist}m"
                _baba_str  = f" {r_baba}馬場" if r_baba and r_baba not in ('', 'nan') else ''
                _rname_str = f"　{r_name}" if r_name and r_name not in ('', 'nan') else ''
                st.markdown(f"""
<div style="background:#1a1a2e;border-radius:10px;padding:14px 20px;margin-bottom:12px;color:white;">
<span style="font-size:1.1em;font-weight:bold;">{r_date} {r_venue}{sel_r}R</span>
<span style="font-size:1.0em;color:#f1c40f;margin-left:8px;">{r_name if r_name and r_name not in ('', 'nan') else ''}</span>
&nbsp;&nbsp;
<span style="color:#aaa;">{r_surf}{_dist_str}{_baba_str}　{r_heads}頭</span>
<br>
<span style="color:#f1c40f;">自信度: {stars}</span>
&nbsp;&nbsp;
<span style="background:#2980b9;padding:2px 10px;border-radius:4px;">推奨: {rec_bet}</span>
{anaba_line}
{pace_line}
{_build_honmei_line(_honmei_info)}
</div>
""", unsafe_allow_html=True)

            # ── 買い目テンプレート表示 ──────────────────────────────────
            if _buy_tickets:
                from reliability import MARK_COLORS
                _tan  = _buy_tickets.get('単勝', [])
                _bar  = _buy_tickets.get('馬連', [])
                _3f   = _buy_tickets.get('三連複_fmtn', {})

                # 馬名→馬番の辞書
                _umaban_d = {}
                if '馬番' in show_df.columns:
                    _umaban_d = {str(r['馬名']): r['馬番']
                                 for _, r in show_df.iterrows()
                                 if pd.notna(r.get('馬番'))}

                def _ub_str(name):
                    ub = _umaban_d.get(str(name))
                    if ub is None:
                        return ''
                    try:
                        return f'<span style="color:#ccc;font-size:0.85em;margin-right:2px;">({int(ub)})</span>'
                    except (ValueError, TypeError):
                        return ''

                def _mark_html(mark, name, pop, ev=None):
                    c = MARK_COLORS.get(mark, '#aaa')
                    ev_str = f'<span style="color:#aaa;font-size:0.8em;">EV{ev:+.0f}%</span>' if ev is not None and pd.notna(ev) else ''
                    return (f'<span style="color:{c};font-weight:bold;font-size:1.1em;">{mark}</span>'
                            f'{_ub_str(name)}'
                            f'<span style="color:white;margin-left:2px;">{name}（{pop}人）</span>{ev_str}')

                def _names_str(names):
                    if not names:
                        return '<span style="color:#666;">未選出</span>'
                    parts = []
                    marks_d = show_df.set_index('馬名')['_mark'].to_dict() if '_mark' in show_df.columns else {}
                    pop_d   = show_df.set_index('馬名')['_pop_int'].to_dict() if '_pop_int' in show_df.columns else {}
                    for n in names:
                        mk = marks_d.get(n, '')
                        c  = MARK_COLORS.get(mk, '#aaa')
                        parts.append(
                            f'<span style="color:{c};font-weight:bold;">{mk}</span>'
                            f'{_ub_str(n)}'
                            f'<span style="color:white;">{n}（{int(pop_d.get(n,99))}人）</span>'
                        )
                    return '　'.join(parts)

                _marks_d = show_df.set_index('馬名')['_mark'].to_dict() if '_mark' in show_df.columns else {}
                tan_html  = '　'.join([_mark_html('◎', h['馬名'], h['pop'], h['ev']) for h in _tan]) or '未選出'
                bar_lines = ''.join([
                    f'<div>{_mark_html("◎", b["馬名1"], b["pop1"])}'
                    f'<span style="color:#aaa;margin:0 4px;">－</span>'
                    f'{_mark_html(_marks_d.get(b["馬名2"],""), b["馬名2"], b["pop2"])}'
                    f'</div>'
                    for b in _bar
                ]) or '<span style="color:#666;">未選出</span>'

                col1_html = _names_str(_3f.get('1列', []))
                col2_html = _names_str(_3f.get('2列', []))
                col3_html = _names_str(_3f.get('3列', []))
                n_tickets = _3f.get('点数', 0)

                st.markdown(f"""
<div style="background:#0d1117;border:1px solid #30363d;border-radius:10px;padding:14px 20px;margin-bottom:12px;">
<div style="color:#58a6ff;font-weight:bold;font-size:1.0em;margin-bottom:8px;">🎯 買い目テンプレート</div>
<table style="width:100%;border-collapse:collapse;font-size:0.9em;">
<tr style="border-bottom:1px solid #21262d;">
  <td style="color:#8b949e;padding:4px 8px;width:120px;">単勝 1点</td>
  <td style="padding:4px 8px;">{tan_html}</td>
</tr>
<tr style="border-bottom:1px solid #21262d;">
  <td style="color:#8b949e;padding:4px 8px;">馬連 {len(_bar)}点</td>
  <td style="padding:4px 8px;">{bar_lines}</td>
</tr>
<tr>
  <td style="color:#8b949e;padding:4px 8px;vertical-align:top;">三連複<br>{n_tickets}点</td>
  <td style="padding:4px 8px;">
    <div><span style="color:#8b949e;font-size:0.85em;">1列</span>　{col1_html}</div>
    <div><span style="color:#8b949e;font-size:0.85em;">2列</span>　{col2_html}</div>
    <div><span style="color:#8b949e;font-size:0.85em;">3列</span>　{col3_html}</div>
  </td>
</tr>
</table>
<div style="color:#555;font-size:0.75em;margin-top:6px;">※ 馬連・三連複のEVはStep2（オッズ取得後）に表示予定</div>
</div>
""", unsafe_allow_html=True)

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
  </div>
</div>""")
                st.markdown(f"""
<div style="margin-bottom:16px;">
  <div style="color:#f39c12;font-weight:bold;font-size:1.05em;margin-bottom:8px;">
    🎯 穴馬推奨（通常モデル1〜2位 ＋ 穴馬モデル1〜3位 ＋ 7番人気以上）
  </div>
  <div style="display:flex;gap:12px;flex-wrap:wrap;">
    {''.join(_ab_cards)}
  </div>
</div>
""", unsafe_allow_html=True)
            else:
                if _real_pop_known.any():
                    st.markdown(
                        '<div style="color:#555;font-size:0.85em;margin-bottom:10px;">'
                        '🔍 このレースに穴馬候補なし（条件: 通常モデル1〜2位 ＋ 穴馬モデル1〜3位 ＋ 7番人気以上）'
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

            def _waku_html(umaban_val):
                try:
                    ub = int(umaban_val)
                except (TypeError, ValueError):
                    return ''
                waku = min((ub + 1) // 2, 8)
                bg_c, fg_c = _WAKU_COLORS.get(waku, ('#555', '#fff'))
                return (
                    f'<span style="display:inline-block;background:{bg_c};color:{fg_c};'
                    f'border-radius:50%;width:24px;height:24px;line-height:24px;'
                    f'text-align:center;font-size:0.82em;font-weight:bold;'
                    f'border:1px solid #444;margin-right:2px;">{ub}</span>'
                )

            show_df_sorted = show_df.sort_values('pred_rank')

            # ── 一覧サマリーテーブル ──────────────────────────────────
            _tbl_cols  = {'pred_rank': '予測順位'}
            if '馬番' in show_df_sorted.columns:
                _tbl_cols['馬番'] = '馬番'
            _tbl_cols['馬名'] = '馬名'
            _tbl_cols['_pop_int'] = '人気'
            if has_live_odds and '単勝オッズ_live' in show_df_sorted.columns:
                _tbl_cols['単勝オッズ_live'] = '単勝オッズ'
            _tbl_cols['EV単勝']  = 'EV単勝(%)'
            _tbl_cols['EV複勝']  = 'EV複勝(%)'
            _tbl_df = (show_df_sorted[list(_tbl_cols.keys())]
                       .rename(columns=_tbl_cols)
                       .reset_index(drop=True))
            if '人気' in _tbl_df.columns:
                _tbl_df['人気'] = _tbl_df['人気'].apply(lambda x: '--' if x == 0 else int(x))
            if '単勝オッズ' in _tbl_df.columns:
                _tbl_df['単勝オッズ'] = _tbl_df['単勝オッズ'].apply(
                    lambda x: f'{float(x):.1f}' if pd.notna(x) else '--')
            for _ec in ['EV単勝(%)', 'EV複勝(%)']:
                if _ec in _tbl_df.columns:
                    _tbl_df[_ec] = _tbl_df[_ec].apply(
                        lambda x: f'{x:+.0f}' if pd.notna(x) else '--')
            st.dataframe(_tbl_df, hide_index=True)

            for _, row in show_df_sorted.iterrows():
                rank       = int(row.get('pred_rank', 99))
                rank_anaba = int(row.get('pred_rank_anaba', 99)) if has_anaba else None
                pop        = int(row.get('_pop_int', 99))
                name       = str(row.get('馬名', ''))
                umaban_raw = row.get('馬番', None)
                umaban_html = _waku_html(umaban_raw)
                jock       = str(row.get('騎手', ''))
                ev_t       = row.get('EV単勝', float('nan'))
                ev_f       = row.get('EV複勝', float('nan'))
                drift      = int(row.get('人気乖離', 0))
                odds_live  = row.get('単勝オッズ_live', None)
                is_tokujou = bool(row.get('_is_tokujou', False))
                is_anaba   = bool(row.get('_is_anaba', False))

                odds_html  = (f'<span style="color:#3498db;font-size:0.85em;margin-left:4px;">'
                              f'📡 {float(odds_live):.1f}倍</span>'
                              if pd.notna(odds_live) else '')

                # 印バッジ
                _mark       = str(row.get('_mark', ''))
                _fuku_rate  = row.get('_fuku_rate')
                _confidence = row.get('_confidence')
                try:
                    from reliability import MARK_COLORS
                    _mark_color = MARK_COLORS.get(_mark, '#555')
                except Exception:
                    _mark_color = '#555'
                fuku_rate_str = f'馬券内{float(_fuku_rate):.0%}' if pd.notna(_fuku_rate) else ''
                conf_label = ''
                if pd.notna(_confidence):
                    conf_label = '信頼高' if _confidence >= 0.7 else ('信頼中' if _confidence >= 0.4 else '信頼低')
                mark_label = {'◎': '本命', '○': '対抗', '▲': '単穴', '△': '連下', '★': '妙味'}.get(_mark, '')
                honmei_html = ''
                if _mark:
                    honmei_html = (
                        f'<span style="background:{_mark_color};color:#111;padding:1px 10px;'
                        f'border-radius:4px;font-size:1.0em;font-weight:bold;margin-left:6px;">'
                        f'{_mark}{mark_label}</span>'
                        f'<span style="color:#888;font-size:0.8em;margin-left:4px;">'
                        f'{fuku_rate_str}　{conf_label}</span>'
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

                reasons_html = ' ／ '.join(reasons)

                # ペース適性バッジ
                apt = _horse_apts.get(name, {})
                pace_apt_html = ''
                if apt and _pace_prof and _pace_prof.get('avg_pci') is not None:
                    _fit_score, _fit_label = pace_fit_score(apt, _pace_prof)
                    _pref = apt.get('pref_pace', '')
                    _style = apt.get('style', '')
                    _avg_agari = apt.get('avg_agari')
                    _n = apt.get('n_races', 0)
                    if _n >= 3:
                        _agari_txt = f'上り平均{_avg_agari}秒' if _avg_agari else ''
                        _fit_color = '#2ecc71' if _fit_score >= 0.15 else ('#f39c12' if _fit_score >= 0.08 else '#888')
                        pace_apt_html = (
                            f'<span style="color:{_fit_color};font-size:0.78em;margin-left:6px;">'
                            f'🏇 {_style} / {_pref} / {_fit_label}'
                            f'{(" / " + _agari_txt) if _agari_txt else ""}'
                            f'</span>'
                        )

                _rank_icon = '🥇' if rank==1 else ('🥈' if rank==2 else ('🥉' if rank==3 else f'{rank}位'))
                _name_color = '#f39c12' if is_tokujou else 'white'
                _rank_color = '#f1c40f' if rank==1 else 'white'
                _drift_str = f"{'+' if drift>0 else ''}{drift}"
                _live_icon = '📡' if pd.notna(odds_live) else ''
                st.markdown(
                    f'<div style="background:{bg};border-radius:8px;padding:8px 12px;margin-bottom:6px;border-left:4px solid {border};">'
                    f'<div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">'
                    f'<span style="font-size:1.2em;font-weight:bold;color:{_rank_color};white-space:nowrap;">{_rank_icon}</span>'
                    f'{umaban_html}'
                    f'<span style="font-size:1.05em;font-weight:bold;color:{_name_color};">{name}</span>'
                    f'{anaba_badge}'
                    f'{honmei_html}'
                    f'</div>'
                    f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-top:4px;">'
                    f'<span style="color:#aaa;font-size:0.85em;">{jock}</span>'
                    f'<span style="color:#aaa;font-size:0.85em;">{"---" if pop == 0 else f"{pop}番人気"}</span>'
                    f'{odds_html}'
                    f'{anaba_rank_html}'
                    f'<span style="color:#aaa;font-size:0.82em;">乖離{_drift_str}</span>'
                    f'<span style="color:{ev_color};font-weight:bold;font-size:0.9em;margin-left:auto;">単EV:{ev_t_str}{_live_icon}</span>'
                    f'<span style="color:{fev_color};font-weight:bold;font-size:0.9em;">複EV:{ev_f_str}</span>'
                    f'</div>'
                    + (f'<div style="margin-top:2px;">{pace_apt_html}</div>' if pace_apt_html else '')
                    + f'<div style="color:#95a5a6;font-size:0.78em;margin-top:3px;line-height:1.4;">📌 {reasons_html}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # ── EV一覧バーチャート ───────────────────────────────────
            if 'EV単勝' in show_df_sorted.columns:
                st.markdown("#### 単勝EV一覧")
                ev_fig_df = show_df_sorted[['馬名', 'EV単勝', '人気']].copy()
                ev_fig_df['色'] = ev_fig_df['EV単勝'].apply(
                    lambda x: '高EV(+50%↑)' if x >= 50 else ('プラスEV' if x >= 0 else 'マイナスEV'))
                fig_ev = px.bar(
                    ev_fig_df, x='馬名', y='EV単勝',
                    color='色',
                    color_discrete_map={'高EV(+50%↑)': '#2ecc71', 'プラスEV': '#f39c12', 'マイナスEV': '#e74c3c'},
                    title="単勝 期待値（EV）一覧　※100円賭けたとき平均でXX円多く/少なく戻る想定",
                    labels={'EV単勝': 'EV (%)', '馬名': ''},
                )
                fig_ev.add_hline(y=0, line_dash='dash', line_color='white', opacity=0.4)
                fig_ev.update_xaxes(tickangle=30)
                fig_ev.update_layout(showlegend=True)
                st.plotly_chart(fig_ev)


# ============================================================
# Tab 2: 回収率シミュレーション
# ============================================================
with tab2:
    st.subheader("回収率シミュレーション")
    st.caption("学習済みモデルをテストセット（直近20%のレース）に適用し、各馬券種の回収率を計算します。")

    run_sim = st.button("▶ シミュレーション実行", type="primary")

    if run_sim:
        if not MODEL_PATH.exists() or not MASTER_PARQUET.exists():
            st.error("モデルまたはデータが見つかりません。train.py を実行してください。")
        else:
            with st.spinner("特徴量構築 → 予測 → 集計中（数分かかります）..."):
                try:
                    from features import build_features
                    from simulate import predict_testset, simulate

                    df = pd.read_csv(MASTER_CSV, encoding='utf-8-sig', low_memory=False)
                    df['日付_dt'] = pd.to_datetime(df['日付_dt'], errors='coerce')
                    df = df.sort_values('日付_dt').reset_index(drop=True)

                    feat_df = build_features(df, verbose=False)
                    test_df = predict_testset(feat_df)
                    results = simulate(test_df, df)
                    st.session_state['sim_results'] = results
                    st.session_state['sim_test_df'] = test_df

                except Exception as e:
                    st.error(f"エラー: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    if 'sim_results' in st.session_state:
        results  = st.session_state['sim_results']
        test_df  = st.session_state['sim_test_df']

        # ── サマリーテーブル ──
        st.markdown("### 馬券種別 回収率一覧")
        rows = []
        for name, r in results.items():
            rows.append({
                '戦略': name,
                '購入数': f"{r['bets']:,}",
                '的中数': f"{r['hits']:,}",
                '的中率': f"{r['hit_rate']:.1%}",
                '投資額': f"{r['invested']:,}円",
                '回収額': f"{r['returned']:,}円",
                '回収率': r['roi'],
            })
        sum_df = pd.DataFrame(rows)

        def color_roi(val):
            if isinstance(val, float):
                color = '#2ecc71' if val >= 0 else '#e74c3c'
                return f'color: {color}; font-weight: bold'
            return ''

        st.dataframe(
            sum_df.style
                  .format({'回収率': '{:+.1f}%'})
                  .map(color_roi, subset=['回収率']),
            hide_index=True,
        )

        # ── 回収率バーチャート ──
        roi_df = pd.DataFrame({
            '戦略': list(results.keys()),
            '回収率': [r['roi'] for r in results.values()],
        })
        fig_roi = px.bar(
            roi_df, x='戦略', y='回収率',
            color='回収率',
            color_continuous_scale=['#e74c3c', '#f39c12', '#2ecc71'],
            color_continuous_midpoint=0,
            title="馬券種別 回収率（%）",
            labels={'回収率': '回収率 (%)'},
        )
        fig_roi.add_hline(y=0, line_dash='dash', line_color='gray')
        fig_roi.add_hline(y=-25, line_dash='dot', line_color='red',
                          annotation_text='JRA控除率ライン(-25%)')
        fig_roi.update_xaxes(tickangle=20)
        st.plotly_chart(fig_roi)

        # ── 累積損益グラフ ──
        st.markdown("### 累積損益推移（単勝_予測1位）")
        if '単勝_予測1位' in results:
            _df = test_df.copy()
            _df['race_key'] = _df['日付'].astype(str) + _df['開催'].astype(str) + _df['Ｒ'].astype(str)

            from simulate import _parse_payout
            cum_rows = []
            cum_pnl = 0
            for _, grp in _df.groupby('race_key'):
                target = grp[grp['pred_rank'] == 1]
                if target.empty:
                    continue
                row = target.iloc[0]
                cum_pnl -= 100
                if row['着順_num'] == 1:
                    pay = _parse_payout(row.get('単勝配当'))
                    if pay:
                        cum_pnl += pay
                cum_rows.append({'date': row.get('日付_dt', row.get('日付')), 'cum_pnl': cum_pnl})

            if cum_rows:
                pnl_df = pd.DataFrame(cum_rows)
                fig_pnl = px.line(
                    pnl_df, x='date', y='cum_pnl',
                    title='累積損益推移（単勝_予測1位 / 100円ずつ）',
                    labels={'date': '日付', 'cum_pnl': '累積損益（円）'},
                )
                fig_pnl.add_hline(y=0, line_dash='dash', line_color='gray')
                st.plotly_chart(fig_pnl)

        # ── 人気帯別 単勝回収率 ──
        st.markdown("### 人気帯別 単勝回収率（予測1位馬）")
        pop_rows = []
        _df2 = test_df.copy()
        _df2['race_key'] = _df2['日付'].astype(str) + _df2['開催'].astype(str) + _df2['Ｒ'].astype(str)
        for pop_range, label in [((1,1),'1番人気'),(( 2,3),'2-3番人気'),((4,6),'4-6番人気'),((7,99),'7番人気以下')]:
            bets = hits = inv = ret = 0
            for _, grp in _df2.groupby('race_key'):
                t = grp[grp['pred_rank'] == 1]
                if t.empty: continue
                row = t.iloc[0]
                pop = pd.to_numeric(row.get('人気'), errors='coerce')
                if pd.isna(pop) or not (pop_range[0] <= pop <= pop_range[1]): continue
                bets += 1; inv += 100
                if row['着順_num'] == 1:
                    pay = _parse_payout(row.get('単勝配当'))
                    if pay: hits += 1; ret += pay
            if bets > 0:
                pop_rows.append({'人気帯': label, '購入数': bets, '的中率': hits/bets,
                                 '回収率': (ret - inv) / inv * 100})
        if pop_rows:
            pop_df = pd.DataFrame(pop_rows)
            fig_pop = px.bar(pop_df, x='人気帯', y='回収率', color='回収率',
                             color_continuous_scale=['#e74c3c','#f39c12','#2ecc71'],
                             color_continuous_midpoint=0,
                             text='的中率',
                             title='人気帯別 単勝回収率（予測1位馬）')
            fig_pop.update_traces(texttemplate='的中率 %{text:.1%}', textposition='outside')
            fig_pop.add_hline(y=0, line_dash='dash', line_color='gray')
            st.plotly_chart(fig_pop)


# ============================================================
# Tab 3: データ確認
# ============================================================
with tab3:
    st.subheader("データ確認")

    if MASTER_PARQUET.exists():
        try:
            df_sample = load_master_summary()
            st.metric("総レコード数（推定）", "約480,000行（10年分）")
            st.metric("特徴量列数", f"{len(df_sample.columns)}列")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**芝・ダ分布**")
                st.bar_chart(df_sample['芝・ダ'].value_counts())
            with col2:
                st.markdown("**馬場状態分布**")
                st.bar_chart(df_sample['馬場状態'].value_counts())
            with col3:
                st.markdown("**距離分布**")
                st.bar_chart(df_sample['距離'].value_counts().head(10))

            st.markdown("**先頭10行**")
            st.dataframe(df_sample.head(10))
        except Exception as _e3:
            st.warning(f"データ確認の読み込みエラー: {_e3}")
    else:
        st.warning("master.csv が見つかりません。build_dataset.py を実行してください。")


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
with tab5:
    st.subheader("📈 回収率トラッキング")
    st.caption("予測時の印・買い目は自動保存されます。レース確定後に結果を登録してROIを計算します。")

    # ── master.csv ステータス & 取得 ─────────────────────────────────────
    _mc_ok = MASTER_PARQUET.exists()
    if _mc_ok:
        _mc_size = MASTER_PARQUET.stat().st_size // 1024
        st.success(f"✅ master.csv 読み込み済み ({_mc_size:,} KB) — 予測モデルは正常動作中")
    else:
        st.error("⚠️ **master.csv が見つかりません** — 全頭ランク1位になる場合はこれが原因です。")

        # Google Drive からダウンロード（Secrets に URL が設定済みの場合）
        _gdrive_url_set = False
        try:
            _gdrive_url_set = bool(st.secrets.get("gdrive_master_csv_url", ""))
        except Exception:
            pass

        if _gdrive_url_set:
            st.info("📥 Google Drive の URL が Secrets に設定されています。ボタンを押してダウンロードしてください。")
            if st.button("☁️ Google Drive から master.csv をダウンロード", type='primary', key='dl_master_gdrive'):
                with st.spinner("Google Drive からダウンロード中... （239MBのため数分かかります）"):
                    _dl_ok, _dl_msg = _download_master_from_gdrive()
                if _dl_ok:
                    st.success(f"✅ {_dl_msg}　← **ページをリロード（F5）してください**")
                else:
                    st.error(f"❌ ダウンロード失敗: {_dl_msg}")
                    st.markdown(
                        "**確認事項:**\n"
                        "1. Google Drive のファイルが「リンクを知っている全員が閲覧可能」に設定されているか\n"
                        "2. Secrets の `gdrive_master_csv_url` が正しいか\n"
                        "3. 以下のアップロードから手動でアップロード（分割してもOK）"
                    )
        else:
            st.warning("Streamlit Cloud の Secrets に `gdrive_master_csv_url` を設定するか、下のアップロードを使ってください。")

        with st.expander("📂 master.csv 手動アップロード（200MB超の場合は分割不可のため Google Drive 推奨）"):
            _up_mc = st.file_uploader(
                "master.csv をアップロード",
                type=['csv'],
                key='upload_master_csv',
            )
            if _up_mc is not None:
                MASTER_PARQUET.parent.mkdir(parents=True, exist_ok=True)
                import io as _io, gc as _gc2
                with st.spinner("CSV → Parquet 変換中..."):
                    import pyarrow as _pa2
                    import pyarrow.parquet as _pq2
                    _pw2 = None
                    for _ck2 in pd.read_csv(_io.BytesIO(_up_mc.read()), encoding='utf-8-sig', chunksize=50000):
                        _t2 = _pa2.Table.from_pandas(_ck2, preserve_index=False)
                        if _pw2 is None:
                            _pw2 = _pq2.ParquetWriter(str(MASTER_PARQUET), _t2.schema, compression='snappy')
                        _pw2.write_table(_t2)
                        del _ck2, _t2; _gc2.collect()
                    if _pw2: _pw2.close()
                _mb2 = MASTER_PARQUET.stat().st_size // 1024 // 1024
                st.success(f"✅ master.parquet を保存しました ({_mb2} MB)")
                st.rerun()
    st.divider()

    import importlib
    _tab5_err_msg = None
    pred_log   = pd.DataFrame()
    result_log = pd.DataFrame()
    try:
        import result_tracker as _rt
        importlib.reload(_rt)
        from result_tracker import (
            load_pred_log, load_result_log, save_result, calc_roi
        )
        import scrape_result as _sr
        importlib.reload(_sr)
        from scrape_result import fetch_race_result
        pred_log   = load_pred_log()
        result_log = load_result_log()
    except Exception as _tab5_init_err:
        import traceback as _tb
        _tab5_err_msg = traceback.format_exc()
        st.error(f"⚠️ Tab5 初期化エラー: {_tab5_init_err}\n\n```\n{_tab5_err_msg}\n```")

    # ── 結果登録セクション ──────────────────────────────────────────
    st.markdown("### 結果登録")

    rc1, rc2 = st.columns([3, 1])
    with rc1:
        _result_method = st.radio(
            "登録方法", ["Netkeibaから自動取得", "手動入力"],
            horizontal=True, key='result_method'
        )

    if _result_method == "Netkeibaから自動取得":
        st.markdown("#### 日付指定で全レース一括処理")
        st.caption("① 予測を一括実行してpred_logへ保存　② 結果を一括取得してresult_logへ保存　の順で実行してください。")

        # ── 「Tab1予測済み」→ pred_log 一括保存ボタン ────────────────────
        _has_last_pred = LAST_PRED_PATH.exists()
        if _has_last_pred:
            try:
                _lp_all   = pd.read_parquet(LAST_PRED_PATH)
                _lp_cols  = [c for c in ['日付', '開催', 'Ｒ'] if c in _lp_all.columns]
                _lp_info  = _lp_all[_lp_cols].drop_duplicates() if _lp_cols else pd.DataFrame()
                _lp_races = len(_lp_info)
                _lp_date  = str(_lp_info['日付'].iloc[0]) if '日付' in _lp_info.columns and not _lp_info.empty else '不明'
            except Exception:
                _has_last_pred = False
                _lp_races = 0
                _lp_date  = '不明'
            st.info(
                f"📋 **Tab1で予測済みデータあり** — {_lp_date} / {_lp_races}レース\n\n"
                "Target CSV でTab1予測した結果を、そのまま pred_log に一括保存できます（再予測・master.csv 不要）。"
            )
            if st.button("📋 Tab1予測済みデータ → pred_log 一括保存（オッズ自動取得）", key='save_from_lastpred', type='primary'):
                from reliability import calc_honmei_score, assign_marks, build_buy_tickets
                from result_tracker import save_pred_log as _spl_lp
                from pred_utils import softmax_probs as _sfmax_lp
                from scrape_odds import get_race_ids_for_date as _grid_lp, fetch_odds_tan as _fot_lp
                from concurrent.futures import ThreadPoolExecutor, as_completed as _asc_lp
                import re as _re_lp

                _lp_df = pd.read_parquet(LAST_PRED_PATH)
                _lp_ok, _lp_ng = [], []

                def _kaisai_to_abbr_lp(k):
                    m = _re_lp.search(r'\d([^\d]+)\d', str(k))
                    return m.group(1) if m else str(k)

                _lp_df['_va'] = _lp_df['開催'].apply(_kaisai_to_abbr_lp)
                _lp_df['_rn'] = pd.to_numeric(_lp_df['Ｒ'], errors='coerce')

                def _calc_pop_int(df):
                    for col in ('人気_live', '人気'):
                        if col in df.columns:
                            s = pd.to_numeric(df[col], errors='coerce')
                            if s.notna().any():
                                return s.fillna(0).astype(int)
                    return pd.Series(0, index=df.index, dtype=int)

                # ── オッズを並列取得 ─────────────────────────────────────
                _lp_date_str = str(_lp_df['日付'].iloc[0]) if '日付' in _lp_df.columns else ''
                _lp_id_map   = {}
                _lp_odds_map = {}
                if _lp_date_str:
                    with st.spinner(f"レースID・オッズを取得中（{_lp_date_str}）..."):
                        try:
                            _lp_id_map = _grid_lp(_lp_date_str)
                        except Exception:
                            _lp_id_map = {}

                    if _lp_id_map:
                        def _fetch_lp_odds(args):
                            key, rid = args
                            try:
                                return key, _fot_lp(rid)
                            except Exception:
                                return key, None

                        _lp_prog = st.progress(0.0, text="オッズ取得中...")
                        _lp_total = len(_lp_id_map)
                        _lp_done  = 0
                        with ThreadPoolExecutor(max_workers=8) as _ex_lp:
                            _lp_futs = {_ex_lp.submit(_fetch_lp_odds, item): item
                                        for item in _lp_id_map.items()}
                            for _lp_fut in _asc_lp(_lp_futs):
                                _lp_key, _lp_ods = _lp_fut.result()
                                _lp_odds_map[_lp_key] = _lp_ods
                                _lp_done += 1
                                _lp_prog.progress(_lp_done / _lp_total,
                                                  text=f"オッズ取得中... {_lp_done}/{_lp_total}")
                        _lp_prog.empty()

                # ── レースごとにオッズマージ → pred_log 保存 ────────────
                for (_va, _rn_f), _grp in _lp_df.groupby(['_va', '_rn']):
                    _rn = int(_rn_f) if not (isinstance(_rn_f, float) and pd.isna(_rn_f)) else None
                    if _rn is None:
                        continue
                    try:
                        _show = _grp.copy()

                        # race_id 解決（parquet埋め込み優先 → IDマップから逆引き）
                        if '_race_id' in _show.columns and str(_show['_race_id'].iloc[0]) not in ('nan', ''):
                            _rid = str(_show['_race_id'].iloc[0])
                        else:
                            _rid = _lp_id_map.get((_va, _rn), f"{_lp_date_str}_{_va}_{_rn}")

                        # オッズマージ（parquet埋め込み済みでなければ取得分を使う）
                        _has_embedded = ('単勝オッズ_live' in _show.columns
                                         and _show['単勝オッズ_live'].notna().any())
                        if not _has_embedded:
                            _ods = _lp_odds_map.get((_va, _rn))
                            if _ods is not None and not _ods.empty:
                                _show['馬番'] = pd.to_numeric(_show['馬番'], errors='coerce')
                                _o2 = _ods.copy()
                                _o2['馬番'] = pd.to_numeric(_o2['馬番'], errors='coerce')
                                _show = _show.merge(
                                    _o2[['馬番', '単勝オッズ', '人気']].rename(
                                        columns={'単勝オッズ': '単勝オッズ_live', '人気': '人気_live'}),
                                    on='馬番', how='left')
                                _vf2 = VENUE_MAP.get(_va, _va)
                                st.session_state[f'live_odds_{_vf2}_{_rn}'] = _ods
                                st.session_state[f'live_odds_time_{_vf2}_{_rn}'] = \
                                    __import__('datetime').datetime.now().strftime('%H:%M:%S')

                        _show['_win_prob'] = _sfmax_lp(_show['pred_score']).values
                        _show['_pop_int']  = _calc_pop_int(_show)
                        _show['pred_rank'] = pd.to_numeric(_show['pred_rank'], errors='coerce').fillna(99).astype(int)
                        _show = calc_honmei_score(_show, _show.iloc[0])
                        _show = assign_marks(_show)
                        _bkt  = build_buy_tickets(_show)
                        _d    = str(_show['日付'].iloc[0]) if '日付' in _show.columns else ''
                        _rnm  = str(_show['レース名'].iloc[0]) if 'レース名' in _show.columns else ''
                        _spl_lp(_rid, _d, _va, _rn, _rnm, _show, _bkt)
                        _lp_ok.append((_va, _rn))
                    except Exception as _lpe:
                        _lp_ng.append((_va, _rn, str(_lpe)))

                # last_pred.parquet もオッズ込みで更新
                if _lp_ok and _lp_odds_map:
                    try:
                        _lp_df.to_parquet(LAST_PRED_PATH, index=False)
                    except Exception:
                        pass

                if _lp_ok:
                    _n_odds_lp = sum(1 for (_va2, _rn2) in _lp_ok
                                     if f'live_odds_{VENUE_MAP.get(_va2, _va2)}_{_rn2}' in st.session_state)
                    st.success(f"✅ {len(_lp_ok)}レース pred_log 保存 / オッズ取得: {_n_odds_lp}R")
                if _lp_ng:
                    st.warning(f"⚠️ {len(_lp_ng)}レースは失敗")
                    with st.expander("詳細"):
                        for _va2, _rn2, _e2 in _lp_ng:
                            st.write(f"- {_va2} {_rn2}R → {_e2}")

        st.divider()

        # ── データソース選択 ───────────────────────────────────────────────
        _pred_src = st.radio(
            "予測データソース（新規予測）",
            ["📋 Target CSV（前走データあり・精度高）", "🌐 Netkeiba自動取得（前走データなし・低精度）"],
            horizontal=True, key='bulk_pred_src'
        )
        _use_target_csv = _pred_src.startswith("📋")

        # ── Target CSV モード: ファイルアップロード ──────────────────────
        if _use_target_csv:
            _tgt_col1, _tgt_col2 = st.columns(2)
            with _tgt_col1:
                _bulk_shutuba_csv = st.file_uploader(
                    "出馬表CSV（Target エクスポート）",
                    type=['csv'], accept_multiple_files=True, key='bulk_shutuba_csv',
                    help="Targetソフト → エクスポート → 出馬表 で出力したCSV（cp932）"
                )
            with _tgt_col2:
                _bulk_maesou_csv = st.file_uploader(
                    "前走CSV（オプション・精度向上）",
                    type=['csv'], accept_multiple_files=True, key='bulk_maesou_csv',
                    help="Targetソフト → エクスポート → 前走 で出力したCSV"
                )
            _tgt_csv_ready = bool(_bulk_shutuba_csv)
        else:
            _bulk_shutuba_csv = None
            _bulk_maesou_csv  = None
            _tgt_csv_ready    = True

        _date_col, _btn_col1, _btn_col2 = st.columns([2, 3, 3])
        with _date_col:
            _bulk_date = st.date_input("対象日", value=None, key='bulk_date_input',
                                       label_visibility='collapsed')
        with _btn_col1:
            if _use_target_csv:
                _bulk_pred_btn = st.button(
                    "① Target CSV 全レース予測 → pred_log保存",
                    key='bulk_pred_btn',
                    disabled=(_bulk_date is None or not _tgt_csv_ready),
                    type='primary',
                    help="Target CSVを使って全レース予測し、pred_logに一括保存します"
                )
            else:
                _bulk_pred_btn = st.button(
                    "① 全レース予測 → pred_log保存",
                    key='bulk_pred_btn',
                    disabled=_bulk_date is None,
                    help="⚠️ Netkeiba経由のため前走データなし。精度が低い点に注意。"
                )
        with _btn_col2:
            _bulk_fetch_btn = st.button("② 全レース結果を一括取得",
                                        key='bulk_date_btn', type='primary',
                                        disabled=_bulk_date is None)

        # ── ① 一括予測 → pred_log ──────────────────────────────────────
        if _bulk_pred_btn and _bulk_date is not None:
            from scrape_odds import get_race_ids_for_date, fetch_odds_tan as _fot_bulk
            from pipeline_target import predict_both_from_df
            from pred_utils import softmax_probs as _sfmax
            from reliability import calc_honmei_score, assign_marks, build_buy_tickets
            from result_tracker import save_pred_log as _spl_bulk
            import re as _re

            _date_str = _bulk_date.strftime('%Y%m%d')

            if not MODEL_PATH.exists():
                st.error("モデルが未学習です。train.py を実行してください。")

            # ══════════════════════════════════════════════════════
            # モード A: Target CSV 使用（前走データあり・精度高）
            # ══════════════════════════════════════════════════════
            elif _use_target_csv and _bulk_shutuba_csv:
                from load_shutuba_target import (load_shutuba_target as _lst,
                                                  load_maesou_target as _lmt,
                                                  merge_maesou_into_shutuba as _mms)
                from concurrent.futures import ThreadPoolExecutor, as_completed as _asc

                # Step1: Target CSV を読み込み
                with st.spinner("Target CSV を読み込み中..."):
                    _tc_frames = []
                    for _uf in _bulk_shutuba_csv:
                        _tc_df = _lst(file_bytes=_uf.read(), filename=_uf.name,
                                      date_str=_date_str)
                        _tc_frames.append(_tc_df)
                    _shutuba_all = pd.concat(_tc_frames, ignore_index=True)

                    if _bulk_maesou_csv:
                        _mf_frames = []
                        for _mf in _bulk_maesou_csv:
                            _mf_frames.append(_lmt(file_bytes=_mf.read(), filename=_mf.name))
                        _maesou_all = pd.concat(_mf_frames, ignore_index=True)
                        _shutuba_all = _mms(_shutuba_all, _maesou_all)
                        st.caption(f"前走データ {len(_maesou_all)}頭分をマージしました。")

                # Step2: 全レース一括予測（master.csv は1回だけ読み込み）
                with st.spinner(f"全{len(_shutuba_all)}頭を予測中..."):
                    _pred_all = predict_both_from_df(_shutuba_all)

                # Step3: venue_abbr を '1東1' → '東' に変換
                def _kaisai_to_abbr(k):
                    m = _re.search(r'\d([^\d]+)\d', str(k))
                    return m.group(1) if m else str(k)

                _pred_all['_venue_abbr'] = _pred_all['開催'].apply(_kaisai_to_abbr)
                _pred_all['_r_num'] = pd.to_numeric(_pred_all['Ｒ'], errors='coerce')

                # Step4: Netkeiba からレースIDを取得（オッズ用）
                with st.spinner("レースID・オッズを取得中..."):
                    _id_map = get_race_ids_for_date(_date_str)

                _race_groups_list = [
                    (va, int(rn))
                    for va, rn in _pred_all.groupby(['_venue_abbr', '_r_num']).groups.keys()
                    if not (isinstance(rn, float) and pd.isna(rn))
                ]
                _race_id_lookup = {(va, rn): _id_map.get((va, rn)) for va, rn in _race_groups_list}

                # Step5: オッズを並列取得
                _odds_cache = {}

                def _fetch_odds_only(args):
                    key, rid = args
                    if rid is None:
                        return key, None
                    try:
                        return key, _fot_bulk(rid)
                    except Exception:
                        return key, None

                _total = len(_race_id_lookup)
                _prog_p = st.progress(0.0, text=f"オッズ取得中... 0 / {_total}")
                _done_count = 0
                with ThreadPoolExecutor(max_workers=8) as _ex:
                    _futs = {_ex.submit(_fetch_odds_only, item): item
                             for item in _race_id_lookup.items()}
                    for _fut in _asc(_futs):
                        _key, _ods = _fut.result()
                        _odds_cache[_key] = _ods
                        _done_count += 1
                        _prog_p.progress(_done_count / _total,
                                         text=f"オッズ取得中... {_done_count} / {_total}")

                # Step6: レースごとに odds マージ → pred_log 保存
                _pred_ok, _pred_ng = [], []
                _bulk_show_frames = []

                for (_va, _rn_f), _grp in _pred_all.groupby(['_venue_abbr', '_r_num']):
                    _rn = int(_rn_f) if not (isinstance(_rn_f, float) and pd.isna(_rn_f)) else None
                    if _rn is None:
                        continue
                    _race_id = _race_id_lookup.get((_va, _rn))
                    _odds_df = _odds_cache.get((_va, _rn))
                    try:
                        _show = _grp.copy()
                        _show['_race_id'] = _race_id or f"{_date_str}_{_va}_{_rn}"
                        if _odds_df is not None and not _odds_df.empty:
                            _show['馬番'] = pd.to_numeric(_show.get('馬番'), errors='coerce')
                            _o2 = _odds_df.copy()
                            _o2['馬番'] = pd.to_numeric(_o2['馬番'], errors='coerce')
                            _show = _show.merge(
                                _o2[['馬番', '単勝オッズ', '人気']].rename(
                                    columns={'単勝オッズ': '単勝オッズ_live', '人気': '人気_live'}),
                                on='馬番', how='left')
                            _vf2 = VENUE_MAP.get(_va, _va)
                            st.session_state[f'live_odds_{_vf2}_{_rn}'] = _odds_df
                            st.session_state[f'live_odds_time_{_vf2}_{_rn}'] = \
                                __import__('datetime').datetime.now().strftime('%H:%M:%S')
                        _show['_win_prob'] = _sfmax(_show['pred_score']).values
                        _pop_s = next(
                            (pd.to_numeric(_show[c], errors='coerce') for c in ('人気_live', '人気')
                             if c in _show.columns and pd.to_numeric(_show[c], errors='coerce').notna().any()),
                            pd.Series(0, index=_show.index)
                        )
                        _show['_pop_int'] = _pop_s.fillna(0).astype(int)
                        _show['pred_rank'] = pd.to_numeric(_show['pred_rank'], errors='coerce').fillna(99).astype(int)
                        _show = calc_honmei_score(_show, _show.iloc[0])
                        _show = assign_marks(_show)
                        _bkt = build_buy_tickets(_show)
                        _d   = str(_show['日付'].iloc[0]) if '日付' in _show.columns else _date_str
                        _rnm = str(_show['レース名'].iloc[0]) if 'レース名' in _show.columns else ''
                        if _race_id:
                            _spl_bulk(_race_id, _d, _va, _rn, _rnm, _show, _bkt)
                        _bulk_show_frames.append(_show)
                        _pred_ok.append((_race_id or f"{_va}{_rn}R", _va, _rn))
                    except Exception as _pe:
                        _pred_ng.append((_race_id or f"{_va}{_rn}R", _va, _rn, str(_pe)))

                if _bulk_show_frames:
                    _bulk_all = pd.concat(_bulk_show_frames, ignore_index=True)
                    LAST_PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
                    _bulk_all.to_parquet(LAST_PRED_PATH, index=False)
                    st.session_state['pred_df'] = _bulk_all
                    st.session_state['is_upcoming'] = True

                _prog_p.empty()
                if _pred_ok:
                    st.success(f"✅ {len(_pred_ok)}レース予測完了（Target CSV使用・前走データあり）")
                if _pred_ng:
                    st.warning(f"⚠️ {len(_pred_ng)}レースは失敗しました")
                    with st.expander("失敗レース詳細"):
                        for _rid2, _va2, _rn2, _err2 in _pred_ng:
                            st.write(f"- {_va2} {_rn2}R ({_rid2}) → {_err2}")
                if _pred_ok:
                    st.rerun()

            # ══════════════════════════════════════════════════════
            # モード B: Netkeiba 自動取得（前走データなし・低精度）
            # ══════════════════════════════════════════════════════
            else:
                from scrape_shutuba import get_shutuba
                with st.spinner(f"{_bulk_date} のレースIDを取得中..."):
                    _id_map = get_race_ids_for_date(_date_str)

                if not _id_map:
                    st.warning("レースIDが取得できませんでした。")
                elif not MASTER_PARQUET.exists():
                    st.error(
                        "⚠️ master.csv が見つかりません。"
                        "このまま実行すると全頭ランク1位になります。\n\n"
                        "**「回収率トラッキング」タブから master.csv をアップロードしてください。**"
                    )
                else:
                    from concurrent.futures import ThreadPoolExecutor, as_completed as _asc

                    _id_list = sorted(_id_map.items(), key=lambda x: (x[0][0], x[0][1]))
                    _total   = len(_id_list)
                    _prog_p  = st.progress(0.0, text=f"出馬表を並列取得中... 0 / {_total}")

                    # ── Step1: 出馬表＋オッズを並列HTTP取得（最大8並列） ─────
                    _shutuba_map = {}  # race_id → (va, rn, sdf, odds_df, error)

                    def _fetch_sdf(args):
                        (v_abbr, r_num), race_id = args
                        try:
                            sdf = get_shutuba(race_id, kaisai_date=_date_str)
                        except Exception as e:
                            return race_id, v_abbr, r_num, None, None, str(e)
                        try:
                            odds_df = _fot_bulk(race_id)
                        except Exception:
                            odds_df = None
                        return race_id, v_abbr, r_num, sdf, odds_df, None

                    _done_count = 0
                    with ThreadPoolExecutor(max_workers=8) as _ex:
                        _futs = {_ex.submit(_fetch_sdf, item): item for item in _id_list}
                        for _fut in _asc(_futs):
                            _rid, _va, _rn, _sdf, _odds_df, _err = _fut.result()
                            _shutuba_map[_rid] = (_va, _rn, _sdf, _odds_df, _err)
                            _done_count += 1
                            _prog_p.progress(_done_count / _total,
                                             text=f"出馬表取得中... {_done_count} / {_total}")

                    # ── Step2: 予測・保存（直列・軽量） ─────────────────
                    _pred_ok, _pred_ng = [], []
                    _bulk_show_frames = []
                    for _i, ((v_abbr, r_num), race_id) in enumerate(_id_list):
                        _va, _rn, _sdf, _odds_df, _err = _shutuba_map.get(
                            race_id, (v_abbr, r_num, None, None, "未取得"))
                        _prog_p.progress((_i + 1) / _total,
                                         text=f"予測中... {_i+1} / {_total}  ({_va} {_rn}R)")
                        if _err or _sdf is None or _sdf.empty:
                            _pred_ng.append((race_id, _va, _rn, _err or "出馬表が空"))
                            continue
                        try:
                            _pdf  = predict_both_from_df(_sdf)
                            _show = _pdf.copy()
                            _show['_race_id'] = race_id
                            if _odds_df is not None and not _odds_df.empty:
                                _show['馬番'] = pd.to_numeric(_show.get('馬番'), errors='coerce')
                                _o2 = _odds_df.copy()
                                _o2['馬番'] = pd.to_numeric(_o2['馬番'], errors='coerce')
                                _show = _show.merge(
                                    _o2[['馬番', '単勝オッズ', '人気']].rename(
                                        columns={'単勝オッズ': '単勝オッズ_live', '人気': '人気_live'}),
                                    on='馬番', how='left')
                                _vf2 = VENUE_MAP.get(_va, _va)
                                st.session_state[f'live_odds_{_vf2}_{_rn}'] = _odds_df
                                st.session_state[f'live_odds_time_{_vf2}_{_rn}'] = \
                                    __import__('datetime').datetime.now().strftime('%H:%M:%S')
                            _show['_win_prob'] = _sfmax(_show['pred_score']).values
                            _pop_s2 = next(
                                (pd.to_numeric(_show[c], errors='coerce') for c in ('人気_live', '人気')
                                 if c in _show.columns and pd.to_numeric(_show[c], errors='coerce').notna().any()),
                                pd.Series(0, index=_show.index)
                            )
                            _show['_pop_int'] = _pop_s2.fillna(0).astype(int)
                            _show['pred_rank'] = pd.to_numeric(_show['pred_rank'], errors='coerce').fillna(99).astype(int)
                            _show = calc_honmei_score(_show, _show.iloc[0])
                            _show = assign_marks(_show)
                            _bkt  = build_buy_tickets(_show)
                            _d    = str(_show['日付'].iloc[0]) if '日付' in _show.columns else _date_str
                            _rnm  = str(_show['レース名'].iloc[0]) if 'レース名' in _show.columns else ''
                            _spl_bulk(race_id, _d, _va, _rn, _rnm, _show, _bkt)
                            _bulk_show_frames.append(_show)
                            _pred_ok.append((race_id, _va, _rn))
                        except Exception as _pe:
                            _pred_ng.append((race_id, _va, _rn, str(_pe)))
                    if _bulk_show_frames:
                        _bulk_all = pd.concat(_bulk_show_frames, ignore_index=True)
                        LAST_PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
                        _bulk_all.to_parquet(LAST_PRED_PATH, index=False)
                        st.session_state['pred_df'] = _bulk_all
                        st.session_state['is_upcoming'] = True
                    _prog_p.empty()
                    if _pred_ok:
                        _n_odds = sum(
                            1 for ((va, rn), rid) in _id_list
                            if f'live_odds_{VENUE_MAP.get(va, va)}_{rn}' in st.session_state
                        )
                        st.success(f"✅ {len(_pred_ok)}レース予測完了 / オッズ取得: {_n_odds}R")
                    if _pred_ng:
                        st.warning(f"⚠️ {len(_pred_ng)}レースは失敗しました")
                        with st.expander("失敗レース詳細"):
                            for _rid, _va, _rn, _err in _pred_ng:
                                st.write(f"- {_va} {_rn}R ({_rid}) → {_err}")
                    if _pred_ok:
                        st.rerun()

        if _bulk_fetch_btn and _bulk_date is not None:
            from scrape_odds import get_race_ids_for_date
            from scrape_result import fetch_all_results
            _date_str = _bulk_date.strftime('%Y%m%d')
            with st.spinner(f"{_bulk_date} のレースIDを取得中..."):
                _id_map = get_race_ids_for_date(_date_str)  # {(venue_abbr, r_num): race_id}

            if not _id_map:
                st.warning("レースIDが取得できませんでした。日付を確認してください。")
            else:
                _all_ids = list(_id_map.values())
                st.info(f"{len(_all_ids)}レースを取得します...")
                _prog2 = st.progress(0.0, text=f"取得中... 0 / {len(_all_ids)}")

                def _cb2(done, total, rid):
                    _prog2.progress(done / total, text=f"取得中... {done} / {total}  ({rid})")

                _res_map = fetch_all_results(_all_ids, progress_cb=_cb2)
                _prog2.empty()
                _ok2, _ng2 = [], []
                for _rid, _res in _res_map.items():
                    # venue/r_num をIDマップから逆引き
                    _venue_r = next(((v, r) for (v, r), i in _id_map.items() if i == _rid), ('', ''))
                    _lbl2 = f"{_bulk_date} {_venue_r[0]} {_venue_r[1]}R  ({_rid})"
                    if _res.get('error'):
                        _ng2.append((_lbl2, _res['error']))
                    else:
                        _h    = _res['horses']
                        _fuku = _res.get('fuku', [])
                        save_result(
                            race_id        = _rid,
                            chaku1         = _h[0]['name'] if len(_h) > 0 else '',
                            chaku2         = _h[1]['name'] if len(_h) > 1 else '',
                            chaku3         = _h[2]['name'] if len(_h) > 2 else '',
                            tan_pay        = _res.get('tan'),
                            fuku1_pay      = _fuku[0] if len(_fuku) > 0 else None,
                            fuku2_pay      = _fuku[1] if len(_fuku) > 1 else None,
                            fuku3_pay      = _fuku[2] if len(_fuku) > 2 else None,
                            baren_pay      = _res.get('baren'),
                            sanrenpuku_pay = _res.get('sanrenpuku'),
                        )
                        _ok2.append((_lbl2, _res))

                if _ok2:
                    st.success(f"✅ {len(_ok2)}レースを登録しました。")
                    with st.expander("登録内容を確認"):
                        for _lbl2, _res in _ok2:
                            _h     = _res['horses']
                            _names = ' / '.join(h['name'] for h in _h)
                            st.write(
                                f"**{_lbl2}**　1-2-3着: {_names}　"
                                f"単勝:{_res.get('tan')}円　複勝:{_res.get('fuku')}　"
                                f"馬連:{_res.get('baren')}円　三連複:{_res.get('sanrenpuku')}円"
                            )
                if _ng2:
                    st.warning(f"⚠️ {len(_ng2)}レースは取得できませんでした")
                    with st.expander("失敗レース詳細"):
                        for _lbl2, _err in _ng2:
                            st.write(f"- {_lbl2} → {_err}")
                if _ok2:
                    st.rerun()

        st.markdown("---")
        st.markdown("#### 予測ログと紐づいたレースのみ取得")
        if pred_log.empty:
            st.info("予測ログがありません。上の「日付指定で全レース一括取得」をご利用ください。")
        else:
            registered_ids = set(result_log['race_id'].tolist()) if not result_log.empty else set()
            unregistered   = pred_log[~pred_log['race_id'].isin(registered_ids)]

            # 全レースの辞書（ラベル → race_id）
            _all_race_options = {}
            for _, row in pred_log.iterrows():
                d = str(row.get('date', ''))
                label = f"{d[:4]}/{d[4:6]}/{d[6:]} {row.get('venue','')} {row.get('r_num','')}R {row.get('race_name','')}"
                _all_race_options[label] = row['race_id']

            # 未登録のみ
            _unreg_options = {lb: rv for lb, rv in _all_race_options.items()
                              if rv not in registered_ids}

            col_a, col_b = st.columns(2)
            with col_a:
                st.caption(f"未登録: {len(_unreg_options)}件  /  全体: {len(_all_race_options)}件")
                _fetch_new = st.button(
                    f"🔄 未登録 {len(_unreg_options)}件 を取得",
                    key='fetch_new_btn', type='primary',
                    disabled=len(_unreg_options) == 0,
                )
            with col_b:
                _fetch_all_flag = st.button(
                    f"🔁 全 {len(_all_race_options)}件 を上書き再取得",
                    key='fetch_all_btn',
                    help="払戻が正しく取れていない場合はこちらで全レース取り直してください",
                )

            _target = (_unreg_options if _fetch_new else
                       _all_race_options if _fetch_all_flag else None)

            if _target is not None:
                from scrape_result import fetch_all_results
                _prog    = st.progress(0.0, text=f"取得中... 0 / {len(_target)}")
                _ok, _ng = [], []

                def _cb(done, total, rid):
                    _prog.progress(done / total,
                                   text=f"取得中... {done} / {total}  ({rid})")

                _all_res = fetch_all_results(list(_target.values()), progress_cb=_cb)
                for _rid, _res in _all_res.items():
                    _lbl = next((lb for lb, rv in _target.items() if rv == _rid), _rid)
                    if _res.get('error'):
                        _ng.append((_rid, _lbl, _res['error']))
                    else:
                        _h    = _res['horses']
                        _fuku = _res.get('fuku', [])
                        save_result(
                            race_id        = _rid,
                            chaku1         = _h[0]['name'] if len(_h) > 0 else '',
                            chaku2         = _h[1]['name'] if len(_h) > 1 else '',
                            chaku3         = _h[2]['name'] if len(_h) > 2 else '',
                            tan_pay        = _res.get('tan'),
                            fuku1_pay      = _fuku[0] if len(_fuku) > 0 else None,
                            fuku2_pay      = _fuku[1] if len(_fuku) > 1 else None,
                            fuku3_pay      = _fuku[2] if len(_fuku) > 2 else None,
                            baren_pay      = _res.get('baren'),
                            sanrenpuku_pay = _res.get('sanrenpuku'),
                        )
                        _ok.append((_rid, _lbl, _res))
                _prog.empty()
                if _ok:
                    st.success(f"✅ {len(_ok)}レースを登録しました。")
                    with st.expander("登録内容を確認"):
                        for _rid, _lbl, _res in _ok:
                            _h     = _res['horses']
                            _names = ' / '.join(h['name'] for h in _h)
                            st.write(
                                f"**{_lbl}**　"
                                f"1-2-3着: {_names}　"
                                f"単勝:{_res.get('tan')}円　"
                                f"複勝:{_res.get('fuku')}　"
                                f"馬連:{_res.get('baren')}円　"
                                f"三連複:{_res.get('sanrenpuku')}円"
                            )
                if _ng:
                    st.warning(f"⚠️ {len(_ng)}レースは取得できませんでした")
                    with st.expander("失敗レースの詳細（結果未確定 or レースID不一致）"):
                        for _rid, _lbl, _err in _ng:
                            st.write(f"- {_lbl}　({_rid})　→　{_err}")
                if _ok:
                    st.rerun()

    else:  # 手動入力
        st.markdown("#### 手動入力")
        if pred_log.empty:
            st.info("予測ログがありません。")
        else:
            registered_ids = set(result_log['race_id'].tolist()) if not result_log.empty else set()
            _race_options2 = {}
            for _, row in pred_log.iterrows():
                d = str(row.get('date', ''))
                label = f"{d[:4]}/{d[4:6]}/{d[6:]} {row.get('venue','')} {row.get('r_num','')}R {row.get('race_name','')}"
                _race_options2[label] = row['race_id']

            sel_manual_label = st.selectbox(
                "レース", list(_race_options2.keys()), key='sel_manual_race'
            )
            sel_manual_id = _race_options2[sel_manual_label]

            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                m_c1 = st.text_input("1着馬名", key='m_c1')
                m_tan = st.number_input("単勝払戻(円)", min_value=0, value=0, step=10, key='m_tan')
                m_fuku1 = st.number_input("複勝払戻1着(円)", min_value=0, value=0, step=10, key='m_f1')
            with mc2:
                m_c2 = st.text_input("2着馬名", key='m_c2')
                m_baren = st.number_input("馬連払戻(円)", min_value=0, value=0, step=10, key='m_bar')
                m_fuku2 = st.number_input("複勝払戻2着(円)", min_value=0, value=0, step=10, key='m_f2')
            with mc3:
                m_c3 = st.text_input("3着馬名", key='m_c3')
                m_3f = st.number_input("三連複払戻(円)", min_value=0, value=0, step=10, key='m_3f')
                m_fuku3 = st.number_input("複勝払戻3着(円)", min_value=0, value=0, step=10, key='m_f3')

            if st.button("💾 結果を保存", key='save_manual_result', type='primary'):
                if not m_c1:
                    st.error("1着馬名を入力してください。")
                else:
                    save_result(
                        race_id       = sel_manual_id,
                        chaku1        = m_c1, chaku2 = m_c2, chaku3 = m_c3,
                        tan_pay       = m_tan or None,
                        fuku1_pay     = m_fuku1 or None,
                        fuku2_pay     = m_fuku2 or None,
                        fuku3_pay     = m_fuku3 or None,
                        baren_pay     = m_baren or None,
                        sanrenpuku_pay= m_3f or None,
                    )
                    st.success("結果を保存しました。")
                    st.rerun()

    st.divider()

    # ── ROI サマリー ────────────────────────────────────────────────
    st.markdown("### 回収率サマリー")
    pred_log2   = load_pred_log()
    result_log2 = load_result_log()
    roi_data = calc_roi(pred_log2, result_log2)
    summary_df = roi_data['summary']
    detail_df  = roi_data['detail']

    if summary_df.empty:
        st.info("まだ結果登録済みのレースがありません。")
    else:
        def _color_roi(val):
            if not isinstance(val, (int, float)): return ''
            if val >= 100: return 'color: #2ecc71; font-weight: bold'
            if val >= 70:  return 'color: #f39c12'
            return 'color: #e74c3c'

        st.dataframe(
            summary_df.style
                .format({'回収率': '{:.1f}%', '投資額': '{:,}円', '回収額': '{:,}円'})
                .map(_color_roi, subset=['回収率']),
            hide_index=True,
        )

        # 回収率バーチャート
        fig_roi = px.bar(
            summary_df, x='券種', y='回収率',
            color='回収率',
            color_continuous_scale=['#e74c3c', '#f39c12', '#2ecc71'],
            range_color=[0, 200],
            text='回収率',
            title='券種別回収率',
        )
        fig_roi.add_hline(y=100, line_dash='dash', line_color='white', annotation_text='100%（±0）')
        fig_roi.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig_roi)

        # 詳細テーブル
        with st.expander("レース別詳細"):
            if not detail_df.empty:
                st.dataframe(detail_df, hide_index=True)
            else:
                st.info("詳細データなし")

    st.divider()

    # ── 予測ログ管理 ────────────────────────────────────────────────
    with st.expander("予測ログ / 結果ログ（管理用）"):
        st.markdown("**予測ログ**")
        if not pred_log2.empty:
            st.dataframe(pred_log2, hide_index=True)
        else:
            st.caption("データなし")
        st.markdown("**結果ログ**")
        if not result_log2.empty:
            st.dataframe(result_log2, hide_index=True)
        else:
            st.caption("データなし")

        # ── ダウンロード ─────────────────────────────────────────────
        st.markdown("**バックアップ（ダウンロード）**")
        st.caption("再デプロイ後にデータが消えた場合はアップロードで復元できます。")
        _dl1, _dl2 = st.columns(2)
        with _dl1:
            if not pred_log2.empty:
                st.download_button(
                    "⬇️ 予測ログ保存",
                    data=pred_log2.to_csv(index=False).encode('utf-8-sig'),
                    file_name="pred_log.csv",
                    mime="text/csv",
                    key='dl_pred_log',
                )
        with _dl2:
            if not result_log2.empty:
                st.download_button(
                    "⬇️ 結果ログ保存",
                    data=result_log2.to_csv(index=False).encode('utf-8-sig'),
                    file_name="result_log.csv",
                    mime="text/csv",
                    key='dl_result_log',
                )

        # ── アップロード（復元） ──────────────────────────────────────
        st.markdown("**復元（アップロード）**")
        _ul1, _ul2 = st.columns(2)
        with _ul1:
            _up_pred = st.file_uploader("予測ログCSVをアップロード", type='csv', key='up_pred_log')
            if _up_pred:
                import result_tracker as _rt_up1
                _df_up = pd.read_csv(_up_pred, dtype=str)
                # r_num だけ int に戻す
                if 'r_num' in _df_up.columns:
                    _df_up['r_num'] = pd.to_numeric(_df_up['r_num'], errors='coerce').astype('Int64')
                _df_up.to_parquet(_rt_up1.PRED_LOG_PATH, index=False)
                st.success(f"予測ログを復元しました（{len(_df_up)}件）。")
                st.rerun()
        with _ul2:
            _up_res = st.file_uploader("結果ログCSVをアップロード", type='csv', key='up_result_log')
            if _up_res:
                import result_tracker as _rt_up2
                _df_up2 = pd.read_csv(_up_res, dtype=str)
                _num_cols = ['tan_pay','fuku1_pay','fuku2_pay','fuku3_pay','baren_pay','sanrenpuku_pay']
                for _c in _num_cols:
                    if _c in _df_up2.columns:
                        _df_up2[_c] = pd.to_numeric(_df_up2[_c], errors='coerce')
                _df_up2.to_parquet(_rt_up2.RESULT_LOG_PATH, index=False)
                st.success(f"結果ログを復元しました（{len(_df_up2)}件）。")
                st.rerun()

        # ── クリア ──────────────────────────────────────────────────
        st.markdown("**削除**")
        col_del1, col_del2 = st.columns(2)
        with col_del1:
            if st.button("🗑️ 予測ログをクリア", key='clear_pred_log'):
                import result_tracker as _rt2
                _rt2.PRED_LOG_PATH.unlink(missing_ok=True)
                st.success("予測ログを削除しました。")
                st.rerun()
        with col_del2:
            if st.button("🗑️ 結果ログをクリア", key='clear_result_log'):
                import result_tracker as _rt3
                _rt3.RESULT_LOG_PATH.unlink(missing_ok=True)
                st.success("結果ログを削除しました。")
                st.rerun()
