"""
Targetエクスポートの5CSVを読み込み、特徴量を構築してモデルで予測する。
predict_testset とは異なり「未来レース」の予測用（着順不明）。
"""
import pickle
import re
import gc
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_PATH       = Path(__file__).parent.parent / "data" / "processed" / "lgbm_model.pkl"
MODEL_ANABA_PATH = Path(__file__).parent.parent / "data" / "processed" / "lgbm_model_anaba.pkl"
MASTER_CSV       = Path(__file__).parent.parent / "data" / "processed" / "master.csv"
MASTER_PARQUET   = Path(__file__).parent.parent / "data" / "processed" / "master.parquet"

import sys
sys.path.insert(0, str(Path(__file__).parent))
from features import build_features, FEATURE_COLS

# masterから読み込む列（不要な文字列列を除外してメモリ節約・型エラー防止）
_MASTER_NEEDED_COLS = [
    # メタ情報
    '日付_dt', '日付', '開催', 'Ｒ', '馬番', '馬名', '騎手', '調教師',
    # レース条件
    '芝・ダ', '馬場状態', '距離', '頭数', '斤量', 'コース区分',
    # 結果
    '着順_num', '人気',
    # 前走データ
    '前走着順', '前走人気', '前走上り3F', '前走走破タイム', '前走着差タイム',
    '前走馬番', '前走頭数', '前2角', '前3角', '前4角',
    '前距離', '間隔', '前走馬場状態', '前芝・ダ', '替', '前走斤量',
    # ペース
    'PCI', '上り3F',
    # タイム偏差（走破秒=Target既変換済み）
    '走破秒', '上3F地点差',
    # 馬属性
    '馬体重', '馬体重増減', 'キャリア', '性別', '年齢', '種牡馬', '母父馬',
    # クラス（コース区分は上のレース条件に記載済み）
    'クラス_num',
]


CSV_NAMES = {
    'kihon':   '基本',
    'kihon2':  '基本2',
    'time':    'タイム',
    'maesou':  '前走',
    'seisan':  '生産データ',
}

PREFER_FROM_SEISAN = {'種牡馬', '種牡馬コード', '母父馬', '母父馬コード'}


def _find_csv(folder: Path, keyword: str) -> Path | None:
    """フォルダ内でkeywordを含むCSVを探す（大文字小文字無視）"""
    for p in folder.glob('*.csv'):
        if keyword in p.stem:
            return p
    return None


def load_target_csvs(folder: str | Path) -> pd.DataFrame:
    """Targetの5CSVを結合してDataFrameを返す"""
    folder = Path(folder)

    frames = {}
    for key, keyword in CSV_NAMES.items():
        p = _find_csv(folder, keyword)
        if p is None:
            raise FileNotFoundError(f"'{keyword}' を含むCSVが見つかりません: {folder}")
        frames[key] = pd.read_csv(p, encoding='cp932', low_memory=False)

    # 行数チェック
    n = len(frames['kihon'])
    for key, df in frames.items():
        if len(df) != n:
            raise ValueError(f"{key}: 行数不一致 ({len(df)} vs {n})")

    df = frames['kihon'].copy()
    for key in ['kihon2', 'time', 'maesou', 'seisan']:
        other = frames[key]
        dupe_cols = set(df.columns) & set(other.columns) - PREFER_FROM_SEISAN
        dupe_cols_strict = set(df.columns) & set(other.columns)
        if key == 'seisan':
            drop = dupe_cols_strict - PREFER_FROM_SEISAN
        else:
            drop = dupe_cols_strict
        other = other.drop(columns=list(drop), errors='ignore')
        df = pd.concat([df, other], axis=1)

    df = df.loc[:, ~df.columns.duplicated(keep='last')]
    df = _convert_types(df)
    return df


def _convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """走破タイム・着順の型変換"""
    def parse_time(v):
        try:
            s = str(int(v)).zfill(4)
            m, sec, f = int(s[:-3]), int(s[-3:-1]), int(s[-1])
            return m * 60 + sec + f / 10
        except Exception:
            return np.nan

    if '走破タイム' in df.columns:
        df['走破タイム'] = df['走破タイム'].apply(
            lambda x: parse_time(x) if pd.notna(x) and str(x).strip() not in ('', '0') else np.nan
        )

    tr = str.maketrans('０１２３４５６７８９', '0123456789')
    if '着順' in df.columns:
        def to_chaku(v):
            try:
                return int(str(v).translate(tr))
            except Exception:
                return np.nan
        df['着順_num'] = df['着順'].apply(to_chaku)

    if '日付' in df.columns:
        df['日付_dt'] = pd.to_datetime(df['日付'].astype(str), format='%Y%m%d', errors='coerce')

    return df


def _norm_name(s: pd.Series) -> pd.Series:
    """build_features と同じ馬名正規化（先頭/末尾のマーカー・空白除去）。"""
    return (s.astype(str)
            .str.replace(r'^[\s　$*＊◇◆☆★・]+', '', regex=True)
            .str.replace(r'[\s　]+$', '', regex=True))


def _get_store(master: pd.DataFrame):
    """集計ストアを取得。master.parquet より新しいキャッシュがあれば再利用、
    なければ構築して保存する。"""
    import feature_store as _fs
    sp = _fs.STORE_PATH
    try:
        if (sp.exists() and MASTER_PARQUET.exists()
                and sp.stat().st_mtime >= MASTER_PARQUET.stat().st_mtime):
            st = _fs.load_store(sp)
            if st is not None:
                return st
    except Exception:
        pass
    st = _fs.build_store(master)
    try:
        _fs.save_store(st, sp)
    except Exception:
        pass
    return st


def _build_features_via_store(new_df: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """高速パス: 出走馬の master 履歴だけに build_features を掛け、馬個体非依存の
    集計（騎手・血統・コース等）は事前計算ストアで上書きする。48万行の再計算を排除。
    メモリ削減: ストアがキャッシュ済みなら master 全体は読まず、出走馬の行だけ読む。"""
    import feature_store as _fs
    if not MASTER_PARQUET.exists():
        raise RuntimeError("master.parquet が無いためストア方式は不可")

    import pyarrow.parquet as _pq_tmp
    _avail = _pq_tmp.read_schema(str(MASTER_PARQUET)).names
    _cols = list(dict.fromkeys(c for c in _MASTER_NEEDED_COLS if c in _avail))

    nd = new_df.copy()
    nd['日付_dt'] = pd.to_datetime(nd.get('日付_dt', pd.NaT), errors='coerce')
    nd['馬名'] = _norm_name(nd['馬名'])
    horse_names = list(set(nd['馬名'].dropna()))

    # ストア: master の行数が一致するキャッシュがあれば再利用（git同梱でも確実）。
    # 無ければ全master読込で構築（ローカル開発時など）。
    sp = _fs.STORE_PATH
    store = None
    try:
        if sp.exists():
            _st = _fs.load_store(sp)
            _nrows = _pq_tmp.read_metadata(str(MASTER_PARQUET)).num_rows
            if _st is not None and _st.get('_meta', {}).get('n_master') == _nrows:
                store = _st
    except Exception:
        store = None
    if store is None:
        master_full = pd.read_parquet(MASTER_PARQUET, columns=_cols)
        master_full['日付_dt'] = pd.to_datetime(master_full['日付_dt'], errors='coerce')
        # メモリ削減: 集計だけなら float32 で十分（OOM対策）
        _f64 = master_full.select_dtypes(include=['float64']).columns
        if len(_f64):
            master_full[_f64] = master_full[_f64].astype('float32')
        store = _fs.build_store(master_full)
        try:
            _fs.save_store(store, sp)
        except Exception:
            pass
        del master_full
        gc.collect()

    # 出走馬の履歴行だけを読み込む（行フィルタでメモリ削減。非対応時は全読込で絞る）
    try:
        sub = pd.read_parquet(MASTER_PARQUET, columns=_cols,
                              filters=[('馬名', 'in', horse_names)])
        if sub.empty and horse_names:
            raise ValueError("フィルタ読込が空（馬名形式差の可能性）→全読込にフォールバック")
    except Exception:
        _m = pd.read_parquet(MASTER_PARQUET, columns=_cols)
        _m['馬名'] = _norm_name(_m['馬名'])
        sub = _m[_m['馬名'].isin(set(horse_names))].copy()
        del _m
        gc.collect()
    sub['日付_dt'] = pd.to_datetime(sub['日付_dt'], errors='coerce')
    sub['馬名'] = _norm_name(sub['馬名'])
    # 学習・ストアと同じ直近5年の窓に揃える（馬履歴も5年に限定）
    try:
        _cut = pd.to_datetime(store.get('_meta', {}).get('cutoff'))
        if pd.notna(_cut):
            sub = sub[sub['日付_dt'] >= _cut]
    except Exception:
        pass
    gc.collect()

    combined_pre = pd.concat([sub, nd], ignore_index=True)
    new_row_mask = combined_pre.index >= (len(combined_pre) - len(nd))
    combined = combined_pre.sort_values('日付_dt').reset_index(drop=True)
    original_order = combined_pre.sort_values('日付_dt').index
    new_indices = [i for i, orig in enumerate(original_order) if new_row_mask[orig]]
    del combined_pre, sub
    gc.collect()

    feat = build_features(combined, verbose=False).iloc[new_indices].reset_index(drop=True)
    del combined
    gc.collect()
    feat = _fs.apply_store(feat, nd, store)
    return feat, list(range(len(feat)))


def _build_features_and_slice(new_df: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """特徴量を構築し新規行を返す。まず高速なストア方式を試し、
    失敗時は従来のフル再計算にフォールバックする。"""
    try:
        return _build_features_via_store(new_df)
    except Exception as _e:
        print(f"  [store] フォールバック（フル再計算）: {_e}")
        return _build_features_full(new_df)


def _build_features_full(new_df: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """
    過去データ全体と結合して特徴量を構築し、新規行のインデックスを返す（フォールバック）。
    戻り値: (新規行のみの feat_df, インデックスリスト)
    """
    new_df = new_df.copy()
    new_df['日付_dt'] = pd.to_datetime(new_df.get('日付_dt', pd.NaT), errors='coerce')

    _master_path = MASTER_PARQUET if MASTER_PARQUET.exists() else (MASTER_CSV if MASTER_CSV.exists() else None)
    if _master_path is not None:
        if _master_path.suffix == '.parquet':
            import pyarrow.parquet as _pq_tmp
            _avail = _pq_tmp.read_schema(str(_master_path)).names
            _load_cols = list(dict.fromkeys(c for c in _MASTER_NEEDED_COLS if c in _avail))
            master = pd.read_parquet(_master_path, columns=_load_cols)
        else:
            master = pd.read_csv(_master_path, encoding='utf-8-sig', low_memory=False,
                                 usecols=lambda c: c in _MASTER_NEEDED_COLS)
        master['日付_dt'] = pd.to_datetime(master['日付_dt'], errors='coerce')
        # 学習・ストアと同じ直近5年に限定（一貫性とメモリ削減）
        from features import filter_recent_years as _fry
        master = _fry(master)
        combined_pre = pd.concat([master, new_df], ignore_index=True)
        del master
        gc.collect()
        new_row_mask = combined_pre.index >= (len(combined_pre) - len(new_df))
        combined = combined_pre.sort_values('日付_dt').reset_index(drop=True)
        original_order = combined_pre.sort_values('日付_dt').index
        new_indices = [i for i, orig in enumerate(original_order) if new_row_mask[orig]]
        del combined_pre
        gc.collect()
    else:
        combined = new_df.sort_values('日付_dt').reset_index(drop=True)
        new_indices = list(range(len(combined)))

    # メモリ削減: 48万行の float64 → float32 にダウンキャスト（クラウドのOOM対策）。
    # 検証: 最新日36レースで本命(rank1)は float64 と完全一致、中位順位の入替のみ。
    _f64 = combined.select_dtypes(include=['float64']).columns
    if len(_f64):
        combined[_f64] = combined[_f64].astype('float32')
    feat_df = build_features(combined, verbose=False)
    del combined
    gc.collect()
    # メモリ削減: 予測に必要なのは新規行のみ。48万行の feat_df をここで解放し、
    # 新規18頭分だけを返す（_apply_model が2回呼ばれても大きな DF を保持しない）。
    sliced = feat_df.iloc[new_indices].reset_index(drop=True)
    del feat_df
    gc.collect()
    return sliced, list(range(len(sliced)))


def _apply_model(feat_df: pd.DataFrame, new_indices: list[int],
                 model_path: Path) -> pd.DataFrame:
    """特徴量DFの新規行にモデル予測を適用してスコア付きDFを返す"""
    pred_target = feat_df.iloc[new_indices].copy()

    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    model, use_cols = data['model'], data['features']

    _missing = [c for c in use_cols if c not in pred_target.columns]
    for _c in _missing:
        pred_target[_c] = 0.0
    X = pred_target[use_cols].fillna(0).astype(float)
    pred_target['pred_score'] = model.predict(X)

    race_key = (pred_target['日付'].astype(str) +
                pred_target['開催'].astype(str) +
                pred_target['Ｒ'].astype(str))
    # make_label が「1着=最大ラベル」のため lambdarank は「高スコア=好走」を学習する。
    # よって降順ランクで rank=1 が本命（最高スコア=最上位）。
    # 検証(6/14): 高スコア本命=勝率11%/複勝率39%/平均5.1着、逆向きだと勝率0%/平均11着だった。
    pred_target['pred_rank'] = pred_target.groupby(race_key)['pred_score'].rank(ascending=False, method='min')

    # 日付を「8桁(YYYYMMDD)の文字列」に統一する。理由:
    #  ① parquet保存時の型推論エラー防止（int/str混在で pyarrow が失敗していた）
    #  ② 6桁(YYMMDD '260621')のままだと日付選択(0026年扱い)・race_id生成(netkeiba形式に
    #     ならず結果と結合不可)が壊れる。Target CSVは6桁で来ることがあるため8桁化する。
    if '日付' in pred_target.columns:
        _d = pred_target['日付'].astype(str).str.strip().str.replace(r'\D', '', regex=True)
        pred_target['日付'] = _d.where(_d.str.len() != 6, '20' + _d)
    return pred_target


def _build_and_predict(new_df: pd.DataFrame,
                       model_path: Path = None) -> pd.DataFrame:
    """共通処理: new_df を過去データと結合して特徴量構築→予測"""
    if model_path is None:
        model_path = MODEL_PATH
    feat_df, new_indices = _build_features_and_slice(new_df)
    return _apply_model(feat_df, new_indices, model_path)


def predict_from_df(new_df: pd.DataFrame,
                    use_anaba: bool = False) -> pd.DataFrame:
    """
    出馬表DataFrame（get_shutubaの出力）を受け取り予測スコア付きDFを返す。
    use_anaba=True で穴馬モデルを使用。
    両モデルの結果を結合したい場合は predict_both_from_df() を使う。
    """
    if not MODEL_PATH.exists():
        raise RuntimeError("モデルが未学習。train.py を実行してください。")
    path = MODEL_ANABA_PATH if use_anaba else MODEL_PATH
    return _build_and_predict(new_df, model_path=path)


def predict_both_from_df(new_df: pd.DataFrame) -> pd.DataFrame:
    """
    通常モデルと穴馬モデル両方で予測し、
    pred_score / pred_rank に加えて
    pred_score_anaba / pred_rank_anaba 列も付与して返す。
    """
    if not MODEL_PATH.exists():
        raise RuntimeError("モデルが未学習。train.py を実行してください。")

    feat_df, new_indices = _build_features_and_slice(new_df)
    result = _apply_model(feat_df, new_indices, MODEL_PATH)

    if MODEL_ANABA_PATH.exists():
        anaba = _apply_model(feat_df, new_indices, MODEL_ANABA_PATH)
        result['pred_score_anaba'] = anaba['pred_score'].values
        result['pred_rank_anaba']  = anaba['pred_rank'].values

    return result


def predict_race_in_master(date6: str, kaisai: str, rnum) -> pd.DataFrame:
    """master内の過去レースを現行モデルで再予測し、馬名×pred_rank/pred_rank_anaba を返す。
    前走ベースの特徴量のみ使うためleakなし（当該レースの結果は予測に使われない）。"""
    empty = pd.DataFrame(columns=['馬名', 'pred_rank', 'pred_rank_anaba'])
    # master.parquet の 日付 は int64。型を合わせてフィルタ（不一致時は全読込でフォールバック）
    _dv = int(date6) if str(date6).isdigit() else date6
    try:
        day = pd.read_parquet(MASTER_PARQUET, filters=[('日付', '==', _dv)])
    except Exception:
        _m = pd.read_parquet(MASTER_PARQUET)
        day = _m[_m['日付'].astype(str) == str(date6)]
    race = day[(day['開催'].astype(str) == str(kaisai)) & (day['Ｒ'].astype(str) == str(rnum))]
    if race.empty:
        return empty
    names = race['馬名'].dropna().astype(str).unique().tolist()
    try:
        hist = pd.read_parquet(MASTER_PARQUET, filters=[('馬名', 'in', names)])
    except Exception:
        _m = pd.read_parquet(MASTER_PARQUET)
        hist = _m[_m['馬名'].astype(str).isin(names)]
    feat = build_features(hist, verbose=False)
    sel = ((feat['日付'].astype(str) == str(date6)) &
           (feat['開催'].astype(str) == str(kaisai)) &
           (feat['Ｒ'].astype(str) == str(rnum)))
    pt = feat[sel].copy()
    if pt.empty:
        return empty
    for path, suf in [(MODEL_PATH, ''), (MODEL_ANABA_PATH, '_anaba')]:
        if not Path(path).exists():
            pt['pred_rank' + suf] = np.nan
            continue
        with open(path, 'rb') as f:
            d = pickle.load(f)
        model, uc = d['model'], d['features']
        for c in uc:
            if c not in pt.columns:
                pt[c] = 0.0
        score = model.predict(pt[uc].fillna(0).astype(float))
        pt['pred_rank' + suf] = pd.Series(score, index=pt.index).rank(ascending=False, method='min').astype(int)
    return pt[['馬名', 'pred_rank', 'pred_rank_anaba']].reset_index(drop=True)


def load_and_predict(csv_dir: str | Path) -> pd.DataFrame:
    """5CSVを読み込み → 過去データと結合 → 特徴量構築 → 予測スコア付きDFを返す"""
    if not MODEL_PATH.exists():
        raise RuntimeError("モデルが未学習。train.py を実行してください。")
    new_df = load_target_csvs(csv_dir)
    feat_df, new_indices = _build_features_and_slice(new_df)
    result = _apply_model(feat_df, new_indices, MODEL_PATH)
    if MODEL_ANABA_PATH.exists():
        anaba = _apply_model(feat_df, new_indices, MODEL_ANABA_PATH)
        result['pred_score_anaba'] = anaba['pred_score'].values
        result['pred_rank_anaba']  = anaba['pred_rank'].values
    return result
