"""
予測精度モニタリング。

過去レースを現行モデルでリーク無し再予測し、per-horse の成績ログを
`accuracy_log.parquet` に保存する（モデル依存の派生物なので学習後に再生成）。
アプリはこれを読むだけで精度カード/週別トレンド/較正を表示できる。
"""
from functools import lru_cache
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

ACC_LOG_PATH = Path(__file__).parent.parent / "data" / "processed" / "accuracy_log.parquet"
_PROC = Path(__file__).parent.parent / "data" / "processed"


def build_accuracy_log(months: int = 12, verbose: bool = True):
    """直近 months ヶ月の過去レースを現行モデルで再予測し per-horse ログを保存。"""
    from features import build_features, filter_recent_years
    from pred_utils import softmax_probs
    from fuku_calibration import calibrated_fuku_prob

    N = pickle.load(open(_PROC / 'lgbm_model.pkl', 'rb'))
    A = pickle.load(open(_PROC / 'lgbm_model_anaba.pkl', 'rb'))
    m = pd.read_parquet(_PROC / 'master.parquet')
    m['日付_dt'] = pd.to_datetime(m['日付_dt'], errors='coerce')
    feat = build_features(filter_recent_years(m), verbose=False)

    feat['chk'] = pd.to_numeric(feat['着順_num'], errors='coerce')
    feat['pop'] = pd.to_numeric(feat['人気'], errors='coerce')
    feat['日付'] = feat['日付'].astype(str)
    # 直近 months ヶ月に限定（6桁 YYMMDD 前提）
    _d = pd.to_datetime('20' + feat['日付'], format='%Y%m%d', errors='coerce')
    cutoff = _d.max() - pd.DateOffset(months=months)
    H = feat[_d >= cutoff].copy()
    H['rk'] = H['日付'] + '_' + H['開催'].astype(str) + '_' + H['Ｒ'].astype(str)
    H = H[H.groupby('rk')['chk'].transform(lambda s: s.notna().sum()) >= 5].copy()

    for mdl, nm in [(N, 'sN'), (A, 'sA')]:
        X = H.copy()
        for c in mdl['features']:
            if c not in X.columns:
                X[c] = np.nan
        H[nm] = mdl['model'].predict(X[mdl['features']])
    H['rN'] = H.groupby('rk')['sN'].rank(ascending=False, method='first').astype(int)
    H['rA'] = H.groupby('rk')['sA'].rank(ascending=False, method='first').astype(int)
    H['win_prob'] = H.groupby('rk')['sN'].transform(softmax_probs)
    H['fp'] = calibrated_fuku_prob(H['win_prob']).values

    # ★妙味 = 連下(通常4-6位)×人気6+ の中で穴モデル最上位（assign_marks と同ロジック）
    band = H[(H['rN'] >= 4) & (H['rN'] <= 6) & (H['pop'] >= 6)]
    H['is_star'] = False
    if not band.empty:
        star_idx = band.groupby('rk')['rA'].idxmin()
        H.loc[star_idx, 'is_star'] = True

    log = H[['日付', 'rk', 'rN', 'rA', 'pop', 'chk', 'fp', 'is_star']].copy()
    ACC_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log.to_parquet(ACC_LOG_PATH, index=False)
    if verbose:
        print(f'精度ログ保存: {ACC_LOG_PATH}  ({log["rk"].nunique():,}レース / {len(log):,}行)')
    return log


@lru_cache(maxsize=1)
def load_accuracy_log():
    if not ACC_LOG_PATH.exists():
        return None
    return pd.read_parquet(ACC_LOG_PATH)


def _filter_period(df, months):
    if months is None:
        return df
    d = pd.to_datetime('20' + df['日付'].astype(str), format='%Y%m%d', errors='coerce')
    return df[d >= (d.max() - pd.DateOffset(months=months))]


def summary_metrics(df) -> dict:
    """カード用サマリー。"""
    g = df.dropna(subset=['chk'])
    hon = g[g['rN'] == 1]
    res = {
        'n_races': int(g['rk'].nunique()),
        'hon_fuku': float((hon['chk'] <= 3).mean()) if len(hon) else np.nan,
        'hon_win':  float((hon['chk'] == 1).mean()) if len(hon) else np.nan,
    }
    # 上位3頭（◎○▲）で1頭以上が複勝圏
    top3 = g[g['rN'] <= 3]
    any_hit = top3.assign(_h=(top3['chk'] <= 3)).groupby('rk')['_h'].max()
    res['top3_any'] = float(any_hit.mean()) if len(any_hit) else np.nan
    # 順位相関（レース毎 spearman(rN, chk) 平均 / 負ほど良い）
    corrs = g.groupby('rk').apply(lambda x: x['rN'].corr(x['chk'], method='spearman'))
    res['corr'] = float(corrs.mean())
    # 穴モデル
    ana1 = g[g['rA'] == 1]
    res['anaba1_fuku'] = float((ana1['chk'] <= 3).mean()) if len(ana1) else np.nan
    star = g[g['is_star']]
    res['star_fuku'] = float((star['chk'] <= 3).mean()) if len(star) else np.nan
    res['star_n'] = int(len(star))
    return res


def weekly_trend(df) -> pd.DataFrame:
    """開催週ごとの ◎複勝率・◎勝率・対象レース数。"""
    g = df.dropna(subset=['chk']).copy()
    d = pd.to_datetime('20' + g['日付'].astype(str), format='%Y%m%d', errors='coerce')
    g['_week'] = d.dt.to_period('W').apply(lambda p: p.start_time)
    hon = g[g['rN'] == 1]
    out = hon.groupby('_week').apply(lambda x: pd.Series({
        'hon_fuku': (x['chk'] <= 3).mean(),
        'hon_win':  (x['chk'] == 1).mean(),
        'n': x['rk'].nunique(),
    })).reset_index()
    return out.sort_values('_week')


def calibration_bins(df, n_bins: int = 6) -> pd.DataFrame:
    """予想複勝% の較正（予測平均 vs 実際）。"""
    g = df.dropna(subset=['chk', 'fp']).copy()
    g['top3'] = (g['chk'] <= 3).astype(int)
    try:
        g['_b'] = pd.qcut(g['fp'], n_bins, duplicates='drop')
    except Exception:
        return pd.DataFrame(columns=['pred', 'actual', 'n'])
    out = g.groupby('_b', observed=True).agg(pred=('fp', 'mean'),
                                             actual=('top3', 'mean'),
                                             n=('fp', 'size')).reset_index(drop=True)
    return out


if __name__ == '__main__':
    build_accuracy_log()
