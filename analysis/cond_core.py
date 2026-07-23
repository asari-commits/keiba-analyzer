"""条件別 有利データ抽出：共通処理

master.parquet を読み、回収率ベースの集計を行うための前処理と集計関数。
- 単勝配当: 勝ち馬のみ数値(円/100円)。非勝ち馬は "(オッズ)" 表記
- 複勝配当: 3着内(小頭数は2着内)のみ数値
"""
from pathlib import Path
import numpy as np
import pandas as pd

MASTER = Path(__file__).resolve().parent.parent / "data" / "processed" / "master.parquet"

VENUE = {'札': '札幌', '函': '函館', '福': '福島', '新': '新潟', '東': '東京',
         '中': '中山', '名': '中京', '京': '京都', '阪': '阪神', '小': '小倉'}

# 開催日目の記号 (1-9, A=10, B=11, ...)
_DAY = {**{str(i): i for i in range(1, 10)},
        **{c: 10 + i for i, c in enumerate('ABCDEFGH')}}


def load() -> pd.DataFrame:
    df = pd.read_parquet(MASTER)

    # ---- 配当 → 回収額(円/100円賭け) ----
    tan = df['単勝配当'].astype(str)
    is_win_payout = ~tan.str.contains(r'\(', na=True)
    df['tan_ret'] = np.where(is_win_payout, pd.to_numeric(tan, errors='coerce'), 0.0)
    df['tan_ret'] = df['tan_ret'].fillna(0.0)
    df['fuku_ret'] = df['複勝配当'].fillna(0.0)
    # 単勝オッズ（全馬）：勝ち馬は配当/100、それ以外は括弧内
    odds_paren = pd.to_numeric(tan.str.extract(r'\(([\d.]+)\)')[0], errors='coerce')
    df['odds'] = np.where(is_win_payout, pd.to_numeric(tan, errors='coerce') / 100.0, odds_paren)

    # ---- 着順 ----
    df['rank'] = pd.to_numeric(df['着順_num'], errors='coerce')
    df['is_win'] = (df['rank'] == 1).astype(float)
    df['is_fuku'] = (df['fuku_ret'] > 0).astype(float)
    df['is_2'] = (df['rank'] <= 2).astype(float)

    # ---- 開催の分解 ----
    kai = df['開催'].astype(str)
    ex = kai.str.extract(r'^(\d+)([^\d])(.+)$')
    df['kaiji'] = pd.to_numeric(ex[0], errors='coerce')          # 何回開催
    df['venue'] = ex[1].map(VENUE)
    df['nichime'] = ex[2].map(_DAY)                              # 何日目
    # 開催週（2日で1週）
    df['week'] = np.ceil(df['nichime'] / 2)

    # ---- コース ----
    df['turf'] = np.where(df['芝・ダ'].astype(str).str.startswith('芝'), '芝', 'ダ')
    df['course'] = df['venue'] + df['turf'] + df['距離'].astype(str)
    df['baba'] = df['馬場状態'].astype(str).str.strip()
    df['month'] = df['日付_dt'].dt.month
    df['year'] = df['日付_dt'].dt.year

    # ---- 枠番 ----
    df['waku'] = np.ceil(df['馬番'] / df['頭数'] * 8).clip(1, 8)
    # 正確な枠番（JRA方式）
    df['waku'] = _calc_waku(df['馬番'], df['頭数'])
    df['waku_grp'] = pd.cut(df['waku'], [0, 2, 4, 6, 8],
                            labels=['1-2枠(内)', '3-4枠(中内)', '5-6枠(中外)', '7-8枠(外)'])

    # ---- 脚質（4角通過順 / 頭数）----
    c4 = pd.to_numeric(df['4角'], errors='coerce')
    ratio = c4 / df['頭数']
    df['pos_ratio'] = ratio
    df['kyakushitsu'] = np.select(
        [c4 == 1, ratio <= 1 / 3, ratio <= 2 / 3, ratio > 2 / 3],
        ['逃げ', '先行', '中団', '追込'], default=None)

    # ---- 人気帯 ----
    df['ninki'] = pd.to_numeric(df['人気'], errors='coerce')
    df['ninki_grp'] = pd.cut(df['ninki'], [0, 1, 3, 6, 9, 99],
                             labels=['1番人気', '2-3番人気', '4-6番人気', '7-9番人気', '10番人気以下'])

    df['class'] = df['クラス名'].astype(str)

    # ---- 障害レース判定（平地分析から除外する）----
    rn = df['レース名'].astype(str)
    df['jump'] = rn.str.contains('障|ジャ|大障|ハードル', na=False) | df['class'].str.contains('ＪＧ', na=False)
    return df


def _calc_waku(umaban: pd.Series, tosu: pd.Series) -> pd.Series:
    """JRA方式の枠番。頭数に応じて各枠の頭数が変わる。"""
    u = umaban.to_numpy()
    n = tosu.to_numpy()
    out = np.zeros(len(u))
    for i in range(len(u)):
        ni, ui = n[i], u[i]
        if ni <= 8:
            out[i] = ui
            continue
        base = ni // 8
        extra = ni % 8          # 後ろの extra 枠が base+1 頭
        cum, w = 0, 0
        for k in range(1, 9):
            cnt = base + (1 if k > 8 - extra else 0)
            cum += cnt
            if ui <= cum:
                w = k
                break
        out[i] = w
    return pd.Series(out, index=umaban.index)


def agg(df: pd.DataFrame, by, min_n: int = 200) -> pd.DataFrame:
    """回収率ベースの集計。by は列名 or 列名リスト。"""
    g = df.groupby(by, observed=True)
    r = g.agg(n=('is_win', 'size'),
              win=('is_win', 'sum'),
              fuku=('is_fuku', 'sum'),
              tan_ret=('tan_ret', 'sum'),
              fuku_ret=('fuku_ret', 'sum'),
              avg_ninki=('ninki', 'mean'))
    r['勝率%'] = r['win'] / r['n'] * 100
    r['複勝率%'] = r['fuku'] / r['n'] * 100
    r['単回収率%'] = r['tan_ret'] / r['n']
    r['複回収率%'] = r['fuku_ret'] / r['n']
    r = r[r['n'] >= min_n]
    return r.sort_values('単回収率%', ascending=False)


def split_check(df: pd.DataFrame, by, min_n: int = 100) -> pd.DataFrame:
    """前半期間・後半期間で分けて再現性を確認する。"""
    mid = df['日付_dt'].quantile(0.5)
    a = agg(df[df['日付_dt'] <= mid], by, min_n=min_n)
    b = agg(df[df['日付_dt'] > mid], by, min_n=min_n)
    j = a[['n', '単回収率%', '複回収率%', '複勝率%']].join(
        b[['n', '単回収率%', '複回収率%', '複勝率%']], lsuffix='_前半', rsuffix='_後半', how='inner')
    return j


if __name__ == '__main__':
    d = load()
    print('rows', len(d), 'races', d.groupby(['日付', '開催', 'Ｒ']).ngroups)
    print('全体 単回収率', d['tan_ret'].mean(), '複回収率', d['fuku_ret'].mean())
    print(agg(d, 'ninki_grp').to_string())
