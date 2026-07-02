"""
複勝率キャリブレーション。

モデルのレース内 softmax 勝率(win_prob) を、実際の複勝(3着内)確率に
isotonic 回帰で較正する。「取捨の参考」用に、各馬の信頼できる複勝%を出す。

既存の estimate_fuku_probs（3×勝率の近似）より較正誤差が小さいことを
out-of-sample で確認済（ECE 1.45pt vs 1.96pt, 2021-2026）。

キャリブレータはモデル依存の派生物なので、モデル再学習時に build し直すこと
（make_deploy_data / train 後に `python src/fuku_calibration.py`）。
"""
from functools import lru_cache
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

CALIB_PATH = Path(__file__).parent.parent / "data" / "processed" / "fuku_calibration.pkl"


def build_fuku_calibration(verbose: bool = True):
    """master を現行モデルで予測→ win_prob と実複勝から isotonic を学習し保存する。"""
    from sklearn.isotonic import IsotonicRegression
    from features import build_features, filter_recent_years
    from pred_utils import softmax_probs

    proc = Path(__file__).parent.parent / "data" / "processed"
    model = pickle.load(open(proc / 'lgbm_model.pkl', 'rb'))
    m = pd.read_parquet(proc / 'master.parquet')
    m['日付_dt'] = pd.to_datetime(m['日付_dt'], errors='coerce')
    feat = build_features(filter_recent_years(m), verbose=False)
    feat['chk'] = pd.to_numeric(feat['着順_num'], errors='coerce')
    feat['rk'] = (feat['日付'].astype(str) + '_' + feat['開催'].astype(str)
                  + '_' + feat['Ｒ'].astype(str))
    feat = feat[feat.groupby('rk')['chk'].transform(lambda s: s.notna().sum()) >= 5].copy()
    X = feat.copy()
    for c in model['features']:
        if c not in X.columns:
            X[c] = np.nan
    feat['_s'] = model['model'].predict(X[model['features']])
    feat['win_prob'] = feat.groupby('rk')['_s'].transform(softmax_probs)
    valid = feat.dropna(subset=['chk', 'win_prob'])
    y = (valid['chk'] <= 3).astype(int).values

    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso.fit(valid['win_prob'].values, y)

    CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIB_PATH, 'wb') as f:
        pickle.dump({'iso': iso, 'n': int(len(valid))}, f)
    if verbose:
        print(f'複勝キャリブレーション保存: {CALIB_PATH}  (n={len(valid):,})')
    return iso


@lru_cache(maxsize=1)
def _load_calibrator():
    if not CALIB_PATH.exists():
        return None
    try:
        return pickle.load(open(CALIB_PATH, 'rb'))['iso']
    except Exception:
        return None


def calibrated_fuku_prob(win_probs) -> pd.Series:
    """win_prob(レース内softmax勝率)→ 較正済み複勝%(0〜1)。

    キャリブレータ未生成時は estimate_fuku_probs 相当のフォールバック(3×勝率clip)。
    """
    wp = pd.Series(win_probs).astype(float)
    iso = _load_calibrator()
    if iso is None:
        return (wp * 3.0).clip(upper=0.95)
    return pd.Series(iso.predict(wp.values), index=wp.index)


if __name__ == '__main__':
    build_fuku_calibration()
