# -*- coding: utf-8 -*-
"""クッション値・含水率が4角バイアス・上がり・時計を実際にどう変えるかを検証する。

  python analysis/baba_effect.py

前提: analysis/fetch_jra_baba.py で data/processed/jra_baba.parquet を作ってあること。
master と (場, 日付) で突合するので、2021年以降のレースだけが対象になる。

出力の読み方:
  レース上り中位 = そのレースの上り3Fの中位値の平均 ＝「馬場の速さ」の代理指標
  勝馬rel4       = 勝ち馬の4角通過順÷頭数の平均。小さい＝前から勝っている
"""
import sys
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8'); pd.set_option('display.width', 320)
n_ = lambda s: pd.to_numeric(s, errors='coerce'); KEY = ['日付', '開催', 'Ｒ']
R = r"C:\Users\asari\Downloads\Claude\keiba-analyzer\data\processed"

M = pd.read_parquet(R + r"\master.parquet")
M['着'] = n_(M['着順_num']); M = M.dropna(subset=['着'])
for c in ['頭数', '4角', '馬番', '人気']:
    M[c] = n_(M[c])
M['距'] = n_(M['距離'].astype(str).str.extract(r'(\d+)')[0])
M['ag'] = n_(M['上り3F'])
M['ven1'] = M['開催'].astype(str).str.extract(r'([中名新札函小京阪東福])')[0]
V1 = {'中': '中山', '名': '中京', '新': '新潟', '札': '札幌', '函': '函館',
      '小': '小倉', '京': '京都', '阪': '阪神', '東': '東京', '福': '福島'}
M['場'] = M['ven1'].map(V1)
M['日付'] = M['日付'].astype(str)
M['rel4'] = M['4角'] / M['頭数']
M['上順'] = M.groupby(KEY)['ag'].rank(method='min')
M['win'] = (M['着'] == 1).astype(int); M['fuku'] = (M['着'] <= 3).astype(int)
M['ba'] = M['馬場状態'].astype(str).str[0]

B = pd.read_parquet(R + r"\jra_baba.parquet")
B['日付'] = B['日付'].astype(str)
D = M.merge(B[['場', '日付', 'クッション値', '芝含水率_ゴール前', 'ダ含水率_ゴール前', '使用コース']],
            on=['場', '日付'], how='inner')
print(f"突合: {len(D):,}頭 / {D.groupby(KEY).ngroups:,}R  （masterの{len(D)/len(M)*100:.0f}%）")
D['4角帯'] = pd.cut(D['rel4'], [0, .2, .45, .7, 9], labels=['A', 'B', 'C', 'D'])

# レース単位の「馬場の速さ」指標：そのレースの上り3Fの中位値
rl = D.groupby(KEY).agg(上り中位=('ag', 'median')).reset_index()
D = D.merge(rl, on=KEY, how='left')


def report(S, bincol, bins, labels, title):
    S = S.copy()
    S['帯'] = pd.cut(S[bincol], bins, labels=labels)
    print(f"\n{'='*112}\n■ {title}")
    rows = []
    for k, g in S.groupby('帯', observed=True):
        W = g[g['win'] == 1]
        a = g[g['4角帯'] == 'A']; bb = g[g['4角帯'] == 'B']
        c = g[g['4角帯'] == 'C']; dd = g[g['4角帯'] == 'D']
        rows.append({bincol: k, 'R数': g.groupby(KEY).ngroups, '頭数': len(g),
                     'A帯%': round(a['win'].mean()*100, 1), 'B帯%': round(bb['win'].mean()*100, 1),
                     'C帯%': round(c['win'].mean()*100, 1), 'D帯%': round(dd['win'].mean()*100, 1),
                     'D帯複%': round(dd['fuku'].mean()*100, 1),
                     '勝馬rel4': round(W['rel4'].mean(), 3), '勝馬上り順': round(W['上順'].mean(), 2),
                     '上り1位率': round((W['上順'] == 1).mean()*100, 0),
                     'レース上り中位': round(g.groupby(KEY)['上り中位'].first().mean(), 2),
                     '1人気%': round(g[g['人気'] == 1]['win'].mean()*100, 1)})
    print(pd.DataFrame(rows).to_string(index=False))


S = D[(D['芝・ダ'] == '芝') & (D['ba'] == '良')]
report(S, 'クッション値', [0, 8.5, 9.0, 9.5, 10.0, 99], ['〜8.5', '8.5-9.0', '9.0-9.5', '9.5-10.0', '10.0〜'],
       f"芝・良馬場のみ　クッション値の帯別（{len(S):,}頭）  ※硬い＝数値が高い"
       "／注: 場をまたぐと比較にならない（京都は東京より常に0.5〜1.9高い）")
report(S, '芝含水率_ゴール前', [0, 9, 11, 13, 15, 99], ['〜9%', '9-11%', '11-13%', '13-15%', '15%〜'],
       f"芝・良馬場のみ　含水率（ゴール前）の帯別（{len(S):,}頭）")

SD = D[(D['芝・ダ'] == 'ダ')]
report(SD, 'ダ含水率_ゴール前', [0, 3, 5, 8, 12, 99], ['〜3%', '3-5%', '5-8%', '8-12%', '12%〜'],
       f"ダート（全馬場状態）　含水率（ゴール前）の帯別（{len(SD):,}頭）")

print("\n" + "=" * 112 + "\n■ 「良馬場」の中にどれだけ幅があるか（芝・馬場状態ごとの含水率）")
g = D[D['芝・ダ'] == '芝'].groupby(KEY).first().reset_index()
print(g.groupby('ba').agg(R数=('芝含水率_ゴール前', 'size'), 含水率平均=('芝含水率_ゴール前', 'mean'),
                          最小=('芝含水率_ゴール前', 'min'), 最大=('芝含水率_ゴール前', 'max'),
                          クッション平均=('クッション値', 'mean')).round(2).to_string())

print("\n" + "=" * 112 + "\n■ 場×月の基準（これと比べて乖離を見る。絶対値では判断しない）")
for sd, col in (('芝', '芝含水率_ゴール前'), ('芝', 'クッション値'), ('ダ', 'ダ含水率_ゴール前')):
    gg = D[D['芝・ダ'] == sd].groupby(KEY).first().reset_index()
    gg['月'] = gg['日付'].str[2:4]
    print(f"\n  【{sd} {col}】")
    print(gg.pivot_table(index='月', columns='場', values=col, aggfunc='mean').round(1).to_string())
