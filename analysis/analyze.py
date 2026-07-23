"""条件別 有利データ抽出（回収率ベース）— 全4軸の集計を実行し results.pkl に保存"""
import pickle
import numpy as np
import pandas as pd
import cond_core as C

pd.set_option('display.width', 300)
pd.set_option('display.max_rows', 200)

d = C.load()
d = d[~d['jump']].copy()          # 平地のみ
BASE_TAN = d['tan_ret'].mean()
BASE_FUKU = d['fuku_ret'].mean()
R = {'meta': {'rows': len(d), 'races': d.groupby(['日付', '開催', 'Ｒ']).ngroups,
              'from': str(d['日付_dt'].min().date()), 'to': str(d['日付_dt'].max().date()),
              'base_tan': BASE_TAN, 'base_fuku': BASE_FUKU}}


def show(title, df, cols=('n', '勝率%', '複勝率%', '単回収率%', '複回収率%', 'avg_ninki')):
    print(f'\n=== {title} ===')
    print(df[list(cols)].round(1).to_string())


# ---------------------------------------------------------------
# 1. コース別 × 枠順 / 脚質
# ---------------------------------------------------------------
# 距離帯（コース単位だとサンプルが割れるため）
d['dist_grp'] = pd.cut(d['距離'], [0, 1400, 1800, 2200, 9999],
                       labels=['短距離~1400', 'マイル1401-1800', '中距離1801-2200', '長距離2201~'])

R['waku_course'] = C.agg(d, ['venue', 'turf', 'dist_grp'], min_n=0)  # placeholder

# 枠：コース(場×芝ダ×距離帯)ごとに内外差を出す
rows = []
for (v, t, dg), sub in d.groupby(['venue', 'turf', 'dist_grp'], observed=True):
    if len(sub) < 2000:
        continue
    a = C.agg(sub, 'waku_grp', min_n=300)
    if len(a) < 4:
        continue
    inner = a.loc['1-2枠(内)'] if '1-2枠(内)' in a.index else None
    outer = a.loc['7-8枠(外)'] if '7-8枠(外)' in a.index else None
    if inner is None or outer is None:
        continue
    rows.append({'コース': f'{v}{t}{dg}', 'n': len(sub),
                 '内枠_複回収': inner['複回収率%'], '外枠_複回収': outer['複回収率%'],
                 '内枠_複勝率': inner['複勝率%'], '外枠_複勝率': outer['複勝率%'],
                 '内-外(複回収)': inner['複回収率%'] - outer['複回収率%']})
R['waku_bias'] = pd.DataFrame(rows).sort_values('内-外(複回収)', ascending=False)
show('コース別 内枠 vs 外枠（複勝回収率差）', R['waku_bias'].set_index('コース'),
     cols=('n', '内枠_複回収', '外枠_複回収', '内枠_複勝率', '外枠_複勝率', '内-外(複回収)'))

# 脚質：コースごとの逃げ・先行有利度
rows = []
for (v, t, dg), sub in d.groupby(['venue', 'turf', 'dist_grp'], observed=True):
    if len(sub) < 2000:
        continue
    a = C.agg(sub, 'kyakushitsu', min_n=100)
    if '逃げ' not in a.index or '追込' not in a.index:
        continue
    nige, senko = a.loc['逃げ'], a.loc['先行']
    oi = a.loc['追込']
    rows.append({'コース': f'{v}{t}{dg}', 'n': len(sub),
                 '逃げ_複勝率': nige['複勝率%'], '逃げ_単回収': nige['単回収率%'],
                 '先行_複勝率': senko['複勝率%'], '追込_複勝率': oi['複勝率%'],
                 '逃げ-追込': nige['複勝率%'] - oi['複勝率%']})
R['kyaku_bias'] = pd.DataFrame(rows).sort_values('逃げ-追込', ascending=False)
show('コース別 逃げ有利度', R['kyaku_bias'].set_index('コース'),
     cols=('n', '逃げ_複勝率', '逃げ_単回収', '先行_複勝率', '追込_複勝率', '逃げ-追込'))

# ---------------------------------------------------------------
# 2. 馬場状態・季節・開催週
# ---------------------------------------------------------------
R['baba'] = C.agg(d, ['turf', 'baba'], min_n=1000)
show('馬場状態別', R['baba'])

R['baba_kyaku'] = C.agg(d, ['turf', 'baba', 'kyakushitsu'], min_n=500)
show('馬場状態 × 脚質', R['baba_kyaku'])

R['week'] = C.agg(d, ['turf', 'week'], min_n=1000)
show('開催週（1週=2日）別', R['week'])

R['week_waku'] = C.agg(d[d['turf'] == '芝'], ['week', 'waku_grp'], min_n=500)
show('芝：開催週 × 枠', R['week_waku'])

R['month'] = C.agg(d, ['turf', 'month'], min_n=1000)
show('月別', R['month'])

# ---------------------------------------------------------------
# 3. 種牡馬・騎手・調教師の得意条件
# ---------------------------------------------------------------
for col, key, mn in [('種牡馬', 'sire', 150), ('騎手', 'jockey', 150), ('調教師', 'trainer', 150)]:
    a = C.agg(d, [col, 'turf'], min_n=mn)
    a = a[a['単回収率%'] > BASE_TAN]
    R[key] = a
    show(f'{col} × 芝ダ（単回収率トップ20 / n>={mn}）', a.head(20))

# 距離帯まで見る（サンプル確保のため n>=80）
R['sire_dist'] = C.agg(d, ['種牡馬', 'turf', 'dist_grp'], min_n=100).head(30)
show('種牡馬 × 芝ダ × 距離帯（n>=100 トップ30）', R['sire_dist'])

R['jockey_venue'] = C.agg(d, ['騎手', 'venue', 'turf'], min_n=100).head(30)
show('騎手 × 開催場 × 芝ダ（n>=100 トップ30）', R['jockey_venue'])

# ---------------------------------------------------------------
# 4. 人気帯・オッズ妙味
# ---------------------------------------------------------------
R['ninki'] = C.agg(d, 'ninki_grp', min_n=100)
show('人気帯別', R['ninki'])

R['ninki_class'] = C.agg(d, ['class', 'ninki_grp'], min_n=500)
show('クラス × 人気帯', R['ninki_class'])

R['ninki_kyaku'] = C.agg(d, ['ninki_grp', 'kyakushitsu'], min_n=500)
show('人気帯 × 脚質', R['ninki_kyaku'])

# 前走人気と今回人気のギャップ（人気落ち＝妙味？）
d['prev_ninki'] = pd.to_numeric(d['前走人気'], errors='coerce')
d['ninki_move'] = d['ninki'] - d['prev_ninki']
d['move_grp'] = pd.cut(d['ninki_move'], [-99, -5, -2, 1, 4, 99],
                       labels=['人気大幅上昇', '人気上昇', '横ばい', '人気ダウン', '人気大幅ダウン'])
R['ninki_move'] = C.agg(d, 'move_grp', min_n=500)
show('前走比 人気の動き', R['ninki_move'])

# 休養明け間隔
d['kankaku_grp'] = pd.cut(d['間隔'], [0, 2, 4, 8, 16, 999],
                          labels=['連闘~2週', '3-4週', '5-8週', '9-16週', '17週~(長期休養明け)'])
R['kankaku'] = C.agg(d, ['turf', 'kankaku_grp'], min_n=500)
show('レース間隔', R['kankaku'])

with open('results.pkl', 'wb') as f:
    pickle.dump(R, f)
print('\nsaved results.pkl')
