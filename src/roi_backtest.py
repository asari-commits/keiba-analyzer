"""
実配当ベースの回収率(ROI)バックテスト。

現行モデルで master の過去レースを予測し、odds_store の実オッズ・払戻と突き合わせて
各買い方の回収率を算出する。in-sample（学習期間内）評価になる点に注意。
out-of-sample を見たい場合は cutoff_date で「その日以降のみ」を対象にし、
学習期間を切ったモデルで別途評価すること。

使い方:
    python src/roi_backtest.py            # 直近1年
    python src/roi_backtest.py 250701     # 指定日(6桁)以降
"""
import sys
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import build_features, filter_recent_years  # noqa: E402
from pred_utils import softmax_probs  # noqa: E402
from odds_store import load_odds_store  # noqa: E402

_PROC = Path(__file__).parent.parent / "data" / "processed"


def _score(from_date6: int):
    """master を予測し、(日付,開催,Ｒ,馬番,人気,着順,rN,rA,win_prob) を返す。"""
    N = pickle.load(open(_PROC / 'lgbm_model.pkl', 'rb'))
    A = pickle.load(open(_PROC / 'lgbm_model_anaba.pkl', 'rb'))
    m = pd.read_parquet(_PROC / 'master.parquet')
    m['日付_dt'] = pd.to_datetime(m['日付_dt'], errors='coerce')
    feat = build_features(filter_recent_years(m), verbose=False)
    H = feat[feat['日付'] >= from_date6].copy()
    H['chk'] = pd.to_numeric(H['着順_num'], errors='coerce')
    H['pop'] = pd.to_numeric(H['人気'], errors='coerce')
    H['馬番_n'] = pd.to_numeric(H['馬番'], errors='coerce').astype('Int64')
    H['rk'] = H['日付'].astype(str) + '_' + H['開催'].astype(str) + '_' + H['Ｒ'].astype(str)
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
    return H


def _merge_odds(H: pd.DataFrame) -> pd.DataFrame:
    horse, race = load_odds_store()
    if horse is None:
        raise SystemExit('odds_store が未生成です。先に `python src/odds_store.py <csv>` を実行してください。')
    horse = horse.rename(columns={'R': '_R', '馬番': '馬番_n'})
    H['_R'] = pd.to_numeric(H['Ｒ'], errors='coerce').astype('Int64')
    m = H.merge(horse[['日付', '開催', '_R', '馬番_n', 'rkey',
                       'tan_odds', 'tan_pay', 'fuku_pay']],
                on=['日付', '開催', '_R', '馬番_n'], how='left')
    if race is not None:
        m = m.merge(race, on='rkey', how='left')
    return m.dropna(subset=['tan_odds', 'chk'])


def _fmt(bet, recv, hits):
    return f'回収率{recv / bet * 100:5.1f}%  投資{int(bet):>9} 回収{int(recv):>9} 的中{hits}' if bet else 'n=0'


def template_roi(m: pd.DataFrame, scheme: str = 'NEW'):
    """テンプレ買い目（◎→相手流し）の券種別ROI。scheme='NEW'(相手5頭)/'OLD'(相手6頭)。"""
    tot = {k: [0, 0, 0] for k in ['単勝', '複勝', '馬連', '三連複', '三連単']}
    for _, g in m.groupby('rk'):
        g = g.dropna(subset=['chk'])
        hon = g[g['rN'] == 1]
        if hon.empty:
            continue
        hon = hon.iloc[0]
        hb = hon['馬番_n']
        w = {k: g.loc[g['chk'] == k, '馬番_n'] for k in (1, 2, 3)}
        if w[1].empty:
            continue
        w1 = w[1].iloc[0]
        w2 = w[2].iloc[0] if len(w[2]) else None
        w3 = w[3].iloc[0] if len(w[3]) else None
        aite = list(g.loc[(g['rN'] >= 2) & (g['rN'] <= 6), '馬番_n'])
        if scheme == 'OLD':
            outer = g[g['rN'] >= 7]
            if len(outer):
                aite.append(outer.loc[outer['rA'].idxmin(), '馬番_n'])
        aite = [a for a in aite if pd.notna(a) and a != hb]
        aset = set(aite)
        nA = len(aite)
        um, sp, st = g['umaren'].iloc[0], g['sanpuku'].iloc[0], g['santan'].iloc[0]
        tot['単勝'][0] += 100
        if hb == w1 and pd.notna(hon['tan_pay']):
            tot['単勝'][1] += hon['tan_pay']; tot['単勝'][2] += 1
        tot['複勝'][0] += 100
        if pd.notna(hon['fuku_pay']):
            tot['複勝'][1] += hon['fuku_pay']; tot['複勝'][2] += 1
        if nA >= 1:
            tot['馬連'][0] += nA * 100
            if hb in {w1, w2} and ({w1, w2} - {hb}).issubset(aset) and pd.notna(um):
                tot['馬連'][1] += um; tot['馬連'][2] += 1
        if nA >= 2 and w2 is not None and w3 is not None:
            tot['三連複'][0] += len(list(combinations(aite, 2))) * 100
            if hb in {w1, w2, w3} and ({w1, w2, w3} - {hb}).issubset(aset) and pd.notna(sp):
                tot['三連複'][1] += sp; tot['三連複'][2] += 1
            tot['三連単'][0] += nA * (nA - 1) * 100
            if hb == w1 and w2 in aset and w3 in aset and pd.notna(st):
                tot['三連単'][1] += st; tot['三連単'][2] += 1
    return tot


def single_roi(m: pd.DataFrame):
    """単一馬を単勝/複勝で買う戦略群のROI。"""
    md = m.copy()
    md['EV'] = md['win_prob'] * md['tan_odds']
    md['drift'] = md['pop'] - md['rN']

    def roi(sel):
        n = len(sel)
        if n == 0:
            return None
        tr = sel.loc[sel['chk'] == 1, 'tan_pay'].sum() / (n * 100) * 100
        fr = sel['fuku_pay'].fillna(0).sum() / (n * 100) * 100
        return tr, fr, sel['pop'].mean(), n

    band = md[(md['rN'] >= 4) & (md['rN'] <= 6) & (md['pop'] >= 6)]
    star = band.loc[band.groupby('rk')['rA'].idxmin()] if len(band) else band
    strategies = {
        '◎(通常1位)': md[md['rN'] == 1],
        '◎ & EV>=1.0': md[(md['rN'] == 1) & (md['EV'] >= 1.0)],
        '相手(通常2-6位)': md[(md['rN'] >= 2) & (md['rN'] <= 6)],
        '★馬(連下内妙味)': star,
        'バナー基本(通4以内×人気6+)': md[(md['rN'] <= 4) & (md['pop'] >= 6)],
        'バナー特上(+乖離5)': md[(md['rN'] <= 4) & (md['pop'] >= 6) & (md['drift'] >= 5)],
    }
    return {k: roi(v) for k, v in strategies.items()}


def main():
    from_date6 = int(sys.argv[1]) if len(sys.argv) > 1 else 250701
    print(f'=== ROIバックテスト（日付 {from_date6} 以降・現行モデル/in-sample）===')
    H = _score(from_date6)
    m = _merge_odds(H)
    print(f'対象 {m["rk"].nunique():,}レース / {len(m):,}行\n')

    print('--- テンプレ買い目（◎→相手流し）---')
    for scheme in ('OLD', 'NEW'):
        t = template_roi(m, scheme)
        print(f'[{scheme} {"相手6頭" if scheme=="OLD" else "相手5頭"}]')
        for k, (bet, recv, hits) in t.items():
            print(f'  {k}: {_fmt(bet, recv, hits)}')

    print('\n--- 単一馬 単勝/複勝ROI ---')
    for k, r in single_roi(m).items():
        if r:
            tr, fr, pop, n = r
            print(f'  {k:<26} 単勝{tr:5.1f}% 複勝{fr:5.1f}% 人気{pop:.1f} n={n}')


if __name__ == '__main__':
    main()
