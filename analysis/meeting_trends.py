# -*- coding: utf-8 -*-
"""開催日の傾向（枠・脚質・騎手・種牡馬）をまとめる。偶然の範囲かどうか分かるよう、例年値と人気からの期待値を並べる。

  python analysis/meeting_trends.py 20261010 20261011 [--ref 20261003,20261004]

入力: data/raw/day_results/<日付>.csv（fetch_day_results.py の出力。枠・騎手ID・馬IDを含む版）
種牡馬: master の最新の記録 → 無い馬は data/raw/day_results/sire_cache.csv（netkeiba の血統表から取得）
例年値: master 全期間（枠は馬番と頭数から JRA の規則で計算）
人気からの期待: master 全期間の「人気→勝率・複勝率」
"""
import argparse, os, re, sys
import numpy as np, pandas as pd

sys.stdout.reconfigure(encoding='utf-8'); pd.set_option('display.width', 260); pd.set_option('display.max_rows', 200)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAY = os.path.join(ROOT, 'data', 'raw', 'day_results')
n_ = lambda s: pd.to_numeric(s, errors='coerce')
KEY = ['日付', '開催', 'Ｒ']
V1 = {'東': '東京', '京': '京都', '中': '中山', '阪': '阪神', '名': '中京', '新': '新潟', '札': '札幌', '函': '函館', '福': '福島', '小': '小倉'}


def waku_of(uma, n):
    """JRA の枠番: 8頭以下は馬番＝枠。9頭以上は外の枠から2頭ずつ入る。"""
    if pd.isna(uma) or pd.isna(n):
        return np.nan
    uma, n = int(uma), int(n)
    if n <= 8:
        return uma
    sizes = [1] * 8
    for i in range(n - 8):
        sizes[7 - (i % 8)] += 1
    s = 0
    for w, k in enumerate(sizes, 1):
        s += k
        if uma <= s:
            return w
    return 8


def clean_sire(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ''
    s = re.sub(r'&nbsp;|\s*\((米|英|愛|仏|豪|独|加|亜|伊|新|南|日)\)', '', str(s or '')).strip()
    return s


def load_days(days):
    X = []
    for d in days:
        T = pd.read_csv(os.path.join(DAY, f'{d}.csv'), encoding='utf-8-sig', dtype={'馬ID': str, '騎手ID': str})
        T['日'] = d
        X.append(T)
    T = pd.concat(X, ignore_index=True)
    T = T[T['芝ダ'].isin(['芝', 'ダ']) & T['着'].notna()].copy()
    T['馬名'] = T['馬名'].str.strip()
    T['rel4'] = T['pos4'] / T['頭数']
    T['上順'] = T.groupby(['日', '場', 'R'])['上3F'].rank(method='min')
    T['帯'] = pd.cut(T['rel4'], [0, .2, .45, .7, 9], labels=['A', 'B', 'C', 'D'])
    T['逃げ'] = T['pos4'] <= 1
    T['w'] = (T['着'] == 1).astype(int); T['q'] = (T['着'] <= 2).astype(int); T['f'] = (T['着'] <= 3).astype(int)
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('days', nargs='+')
    ap.add_argument('--ref', default='')
    a = ap.parse_args()
    T = load_days(a.days)
    R = load_days(a.days + [d for d in a.ref.split(',') if d]) if a.ref else T

    M = pd.read_parquet(os.path.join(ROOT, 'data', 'processed', 'master.parquet'),
                        columns=['日付', '開催', 'Ｒ', '馬名', '着順_num', '人気', '頭数', '馬番', '4角', '芝・ダ', '種牡馬', '母父馬'])
    M['着'] = n_(M['着順_num']); M = M.dropna(subset=['着'])
    for c in ['人気', '頭数', '馬番', '4角']:
        M[c] = n_(M[c])
    M['場'] = M['開催'].astype(str).str.extract(r'([中名新札函小京阪東福])')[0].map(V1)
    M['f'] = (M['着'] <= 3).astype(int); M['w'] = (M['着'] == 1).astype(int)
    M['枠'] = [waku_of(u, n) for u, n in zip(M['馬番'], M['頭数'])]
    exp = M.groupby('人気')[['w', 'f']].mean()
    # 種牡馬
    last = M.sort_values('日付').drop_duplicates('馬名', keep='last').set_index('馬名')
    cache = pd.read_csv(os.path.join(DAY, 'sire_cache.csv'), encoding='utf-8-sig', dtype=str) \
        if os.path.exists(os.path.join(DAY, 'sire_cache.csv')) else pd.DataFrame(columns=['馬名', '種牡馬', '母父'])
    cs = cache.set_index('馬名')
    for D in (T, R):
        D['種牡馬'] = D['馬名'].map(last['種牡馬']).fillna(D['馬名'].map(cs['種牡馬'])).map(clean_sire)
        D['母父'] = D['馬名'].map(last['母父馬']).fillna(D['馬名'].map(cs['母父'])).map(clean_sire)
        D['期待複'] = D['人気'].map(exp['f']); D['期待勝'] = D['人気'].map(exp['w'])

    def rate(g):
        n = len(g)
        return pd.Series(dict(頭数=n, 勝=int(g['w'].sum()), 連=int(g['q'].sum()), 複=int(g['f'].sum()),
                              勝率=round(g['w'].mean() * 100, 1), 複勝率=round(g['f'].mean() * 100, 1),
                              人気からの期待複勝率=round(g['期待複'].mean() * 100, 1),
                              複勝の超過=round((g['f'] - g['期待複']).mean() * 100, 1),
                              単回収=round(g.loc[g['w'] == 1, 'オッズ'].sum() * 100 / n) if n else 0,
                              平均人気=round(g['人気'].mean(), 1)))

    days = '・'.join(f"{int(d[4:6])}/{int(d[6:])}" for d in a.days)
    print(f"■ 対象: {days}  {T.groupby(['日', '場', 'R']).ngroups}R / {len(T)}頭（障害を除く）")

    # 1. 枠
    print("\n" + "=" * 100 + "\n1. 枠番別（複勝率%。括弧は例年＝その場・路面のmaster全期間）")
    for (v, sd), g in T.groupby(['場', '芝ダ']):
        base = M[(M['場'] == v) & (M['芝・ダ'] == sd)].groupby('枠')['f'].mean() * 100
        s = g.groupby('枠').agg(n=('f', 'size'), w=('w', 'sum'), f=('f', 'sum'))
        print(f" 【{v}{sd}】{g.groupby(['日', 'R']).ngroups}R")
        print("   " + "  ".join(f"{int(i)}枠 {int(r['w'])}-{int(r['f'])}/{int(r['n'])}={r['f']/r['n']*100:.0f}%({base.get(i, np.nan):.0f})"
                                for i, r in s.iterrows()))
        g = g.assign(位置=pd.cut(g['馬番'] / g['頭数'], [0, 1/3, 2/3, 9], labels=['内', '中', '外']))
        s2 = g.groupby('位置', observed=True).agg(n=('f', 'size'), w=('w', 'sum'), f=('f', 'sum'))
        print("   馬番の位置: " + "  ".join(f"{i} {int(r['w'])}勝 複{r['f']/r['n']*100:.0f}%({int(r['f'])}/{int(r['n'])})" for i, r in s2.iterrows()))

    # 2. 脚質
    print("\n" + "=" * 100 + "\n2. 脚質（4角位置）別（複勝率%。括弧は例年）")
    M['帯'] = pd.cut(M['4角'] / M['頭数'], [0, .2, .45, .7, 9], labels=['A', 'B', 'C', 'D'])
    M['逃げ'] = M['4角'] <= 1
    for (v, sd), g in T.groupby(['場', '芝ダ']):
        b = M[(M['場'] == v) & (M['芝・ダ'] == sd)]
        bb = b.groupby('帯', observed=True)['f'].mean() * 100
        s = g.groupby('帯', observed=True).agg(n=('f', 'size'), w=('w', 'sum'), f=('f', 'sum'))
        ng = g[g['逃げ']]
        print(f" 【{v}{sd}】 " + "  ".join(f"{i} {int(r['w'])}勝 {r['f']/r['n']*100:.0f}%({bb[i]:.0f})" for i, r in s.iterrows())
              + f"  ／ 4角先頭 {int(ng['w'].sum())}勝{int(ng['f'].sum())}複/{len(ng)}（例年 複{b[b['逃げ']]['f'].mean()*100:.0f}%）")
        W = g[g['w'] == 1]
        print(f"   勝ち馬の4角位置 {W['rel4'].round(2).tolist()}  上がり順 {W['上順'].astype(int).tolist()}  人気 {W['人気'].astype(int).tolist()}")

    # 3. 騎手
    print("\n" + "=" * 100 + "\n3. 騎手（3騎乗以上。複勝の超過＝実際の複勝率−人気から期待される複勝率）")
    jk = T.groupby('騎手ID').apply(rate)
    jk['騎手'] = T.groupby('騎手ID')['騎手'].first(); jk['場'] = T.groupby('騎手ID')['場'].agg(lambda s: '・'.join(sorted(set(s))))
    if a.ref:
        jr = R.groupby('騎手ID').apply(rate)
        jk['参考期間 複勝の超過'] = jr['複勝の超過']; jk['参考期間 騎乗'] = jr['頭数']
    jk = jk[jk['頭数'] >= 3].sort_values('複勝の超過', ascending=False)
    cols = ['騎手', '場', '頭数', '勝', '連', '複', '勝率', '複勝率', '人気からの期待複勝率', '複勝の超過', '単回収', '平均人気'] + \
           (['参考期間 騎乗', '参考期間 複勝の超過'] if a.ref else [])
    print(jk[cols].to_string(index=False))

    # 4. 種牡馬
    print("\n" + "=" * 100 + "\n4. 種牡馬（3頭以上）")
    for sd in ['芝', 'ダ', None]:
        g = T if sd is None else T[T['芝ダ'] == sd]
        st = g.groupby('種牡馬').apply(rate)
        st = st[st['頭数'] >= 3].sort_values('複勝の超過', ascending=False)
        print(f" 【{'全体' if sd is None else sd}】")
        print(st[['頭数', '勝', '連', '複', '勝率', '複勝率', '人気からの期待複勝率', '複勝の超過', '単回収', '平均人気']].to_string())
    print("\n 母父（全体・4頭以上）")
    bm = T.groupby('母父').apply(rate)
    bm = bm[bm['頭数'] >= 4].sort_values('複勝の超過', ascending=False)
    print(bm[['頭数', '勝', '複', '複勝率', '人気からの期待複勝率', '複勝の超過', '単回収']].to_string())
    print(f"\n 種牡馬が分からない馬: {(T['種牡馬'] == '').sum()}頭")


if __name__ == '__main__':
    main()
