"""
Targetエクスポートの5CSVを読み込み、特徴量を構築してモデルで予測する。
predict_testset とは異なり「未来レース」の予測用（着順不明）。
"""
import pickle
import re
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


def _build_features_and_slice(new_df: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """
    過去データと結合して特徴量を構築し、新規行のインデックスを返す。
    戻り値: (feat_df全体, 新規行のインデックスリスト)
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
        combined_pre = pd.concat([master, new_df], ignore_index=True)
        new_row_mask = combined_pre.index >= len(master)
        combined = combined_pre.sort_values('日付_dt').reset_index(drop=True)
        original_order = combined_pre.sort_values('日付_dt').index
        new_indices = [i for i, orig in enumerate(original_order) if new_row_mask[orig]]
    else:
        combined = new_df.sort_values('日付_dt').reset_index(drop=True)
        new_indices = list(range(len(combined)))

    feat_df = build_features(combined, verbose=False)
    return feat_df, new_indices


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
    pred_target['pred_rank'] = pred_target.groupby(race_key)['pred_score'].rank(ascending=False, method='min')
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
