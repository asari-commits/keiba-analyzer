# -*- coding: utf-8 -*-
"""⑥上がり順予測 と ⑦統合スコア が食い違ったとき、どちらが正しいかの検証。

  python analysis/check_divergence.py

循環を避けるため:
  - ロジスティック回帰は 〜75%期 で学習
  - 期待勝率のクロス表は 75〜87.5%期 で作成
  - 評価は 87.5%期以降 のみ（どちらにも使っていない期間）
"""
import sys, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore'); sys.stdout.reconfigure(encoding='utf-8')
pd.set_option('display.width', 340)
DATA = r"C:\Users\asari\Downloads\Claude\keiba-analyzer\data\processed"
AG = dict(const=0.4003, p_relag5=0.50028, p_rel4_3=-0.03872, p_pci5=-0.00234, cls差=0.01712, 距差100=0.01276)
FEATS = ['pred_relag', 'p_rel4_3', 'p_push5', 'p_pci5', 'sire_r', 'bms_r',
         'career', 'weight', 'cls差', '距差100', 'n_field', 'rest_w']
n_ = lambda s: pd.to_numeric(s, errors='coerce')
KEY = ['日付', '開催', 'Ｒ']

print("読み込み…")
M = pd.read_parquet(DATA + r"\master.parquet")
M['着'] = n_(M['着順_num']); M = M.dropna(subset=['着'])
M['距'] = n_(M['距離'].astype(str).str.extract(r'(\d+)')[0])
for c in ['頭数', '3角', '4角', 'クラス_num', 'キャリア', '馬体重', '人気']:
    M[c] = n_(M[c])
M['ag'] = n_(M['上り3F']); M['pci'] = n_(M['PCI'])
M['rel4'] = (M['4角'] / M['頭数']).round(3); M['rel3'] = (M['3角'] / M['頭数']).round(3)
M['push'] = (M['rel3'] - M['rel4']).round(3)
M['上順'] = M.groupby(KEY)['ag'].rank(method='min')
M['relag'] = (M['上順'] / M['頭数']).round(3)
M['dt'] = pd.to_datetime('20' + M['日付'].astype(str), format='%Y%m%d', errors='coerce')
M['tan'] = n_(M['単勝配当'].astype(str).str.replace(r'[()（）,]', '', regex=True)).fillna(0)
M['fuk'] = n_(M['複勝配当'].astype(str).str.replace(r'[()（）,]', '', regex=True)).fillna(0)
M['distbin'] = (M['距'] // 400 * 400)
M = M.dropna(subset=['dt', 'relag', 'rel4', '人気']).sort_values(['馬名', 'dt']).reset_index(drop=True)

g = M.groupby('馬名', sort=False)
M['p_relag5'] = g['relag'].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
M['p_rel4_3'] = g['rel4'].transform(lambda s: s.shift(1).rolling(3, min_periods=1).median())
M['p_push5'] = g['push'].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
M['p_pci5'] = g['pci'].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
M['cls差'] = M['クラス_num'] - g['クラス_num'].transform(lambda s: s.shift(1))
M['距差100'] = (M['距'] - g['距'].transform(lambda s: s.shift(1))) / 100.0
M['rest_w'] = (M['dt'] - g['dt'].transform(lambda s: s.shift(1))).dt.days / 7.0
M['career'] = M['キャリア']; M['weight'] = M['馬体重']; M['n_field'] = M['頭数']
M['pred_relag'] = (AG['const'] + AG['p_relag5'] * M['p_relag5'].fillna(.5) + AG['p_rel4_3'] * M['p_rel4_3'].fillna(.5)
                   + AG['p_pci5'] * M['p_pci5'].fillna(47.5) + AG['cls差'] * M['cls差'].fillna(0)
                   + AG['距差100'] * M['距差100'].fillna(0)).clip(.02, 1.0)
M['win'] = (M['着'] == 1).astype(int)

C1, C2 = M['dt'].quantile(0.75), M['dt'].quantile(0.875)
tr = M[M['dt'] <= C1]
sire = M[M['dt'] <= C1].groupby(['芝・ダ', 'distbin', '種牡馬'])['win'].agg(['size', 'mean'])
sire = sire[sire['size'] >= 20]['mean'].to_dict()
bms = M[M['dt'] <= C1].groupby(['芝・ダ', 'distbin', '母父馬'])['win'].agg(['size', 'mean'])
bms = bms[bms['size'] >= 20]['mean'].to_dict()
base = tr['win'].mean()
M['sire_r'] = [sire.get(k, base) for k in zip(M['芝・ダ'], M['distbin'], M['種牡馬'])]
M['bms_r'] = [bms.get(k, base) for k in zip(M['芝・ダ'], M['distbin'], M['母父馬'])]

D = M.dropna(subset=['p_relag5', 'p_rel4_3']).copy()
med = D[D['dt'] <= C1][FEATS].median()
for c in FEATS:
    D[c] = D[c].fillna(med[c])
mu, sd = D[D['dt'] <= C1][FEATS].mean(), D[D['dt'] <= C1][FEATS].std().replace(0, 1)
clf = LogisticRegression(max_iter=2000).fit(((D[D['dt'] <= C1][FEATS] - mu) / sd).values, D[D['dt'] <= C1]['win'].values)
D['p'] = clf.predict_proba(((D[FEATS] - mu) / sd).values)[:, 1]
D['模型順'] = D.groupby(KEY)['p'].rank(ascending=False, method='min')
D['市場順'] = D.groupby(KEY)['人気'].rank(method='min')
D['上り順予測'] = D.groupby(KEY)['pred_relag'].rank(method='min')       # ← ⑥
D['頭数'] = D.groupby(KEY)['win'].transform('size')

pb = lambda r: '1人気' if r <= 1 else ('2人気' if r <= 2 else ('3人気' if r <= 3 else ('4-5' if r <= 5 else ('6-8' if r <= 8 else '9↓'))))
mb = lambda r: '模型1位' if r <= 1 else ('模型2-3' if r <= 3 else ('模型4-6' if r <= 6 else '模型7↓'))
mid = D[(D['dt'] > C1) & (D['dt'] <= C2)].copy()
mid['pk'] = mid['市場順'].apply(pb); mid['mk'] = mid['模型順'].apply(mb)
ct = mid.groupby(['pk', 'mk'])['win'].agg(['size', 'mean'])
ct = ct[ct['size'] >= 60]['mean'].to_dict()

te = D[D['dt'] > C2].copy()
te['pk'] = te['市場順'].apply(pb); te['mk'] = te['模型順'].apply(mb)
te['期待勝率'] = [ct.get(k, np.nan) for k in zip(te['pk'], te['mk'])]
te = te.dropna(subset=['期待勝率'])
te['⑦順'] = te.groupby(KEY)['期待勝率'].rank(ascending=False, method='min')
print(f"学習〜{C1.date()} / クロス表{C1.date()}〜{C2.date()} / 評価{C2.date()}〜  評価n={len(te):,} / {te.groupby(KEY).ngroups:,}R\n")


def stat(df, lab):
    return dict(区分=lab, n=len(df), 勝率=round(df['win'].mean() * 100, 1),
                複勝率=round((df['着'] <= 3).mean() * 100, 1),
                単ROI=round(df['tan'].mean(), 0), 複ROI=round(df['fuk'].mean(), 0))


print("【1】⑥上がり順予測の順位 × ⑦期待勝率の順位 → 実勝率%")
te['⑥帯'] = pd.cut(te['上り順予測'], [0, 1, 3, 6, 99], labels=['⑥1位', '⑥2-3', '⑥4-6', '⑥7↓'])
te['⑦帯'] = pd.cut(te['⑦順'], [0, 1, 3, 6, 99], labels=['⑦1位', '⑦2-3', '⑦4-6', '⑦7↓'])
print(te.pivot_table(index='⑥帯', columns='⑦帯', values='win', aggfunc=lambda s: round(s.mean() * 100, 1), observed=True).to_string())
print("頭数:"); print(te.pivot_table(index='⑥帯', columns='⑦帯', values='win', aggfunc='size', observed=True).to_string())

print("\n【2】食い違いの方向別（⑥が上・⑦が下 / その逆）")
te['差'] = te['⑦順'] - te['上り順予測']      # 正 = ⑥のほうが高評価
rows = [stat(te[te['差'] >= 5], '⑥が5ランク以上 高評価'), stat(te[te['差'].between(2, 4)], '⑥が2-4ランク高'),
        stat(te[te['差'].abs() <= 1], 'ほぼ一致'), stat(te[te['差'].between(-4, -2)], '⑦が2-4ランク高'),
        stat(te[te['差'] <= -5], '⑦が5ランク以上 高評価')]
print(pd.DataFrame(rows).to_string(index=False))

print("\n【3】各流派の『本命』の成績（レース内1位に推した馬）")
r = [stat(te.loc[te.groupby(KEY)['pred_relag'].idxmin()], '⑥ 上がり順予測の1位'),
     stat(te.loc[te.groupby(KEY)['期待勝率'].idxmax()], '⑦ 期待勝率の1位'),
     stat(te.loc[te.groupby(KEY)['p'].idxmax()], '模型スコアの1位'),
     stat(te[te['市場順'] == 1], '市場（1番人気）')]
print(pd.DataFrame(r).to_string(index=False))

print("\n【4】⑥1位 かつ ⑦下位 の馬（今回3レースで勝ったパターン）")
r2 = [stat(te[(te['上り順予測'] == 1) & (te['⑦順'] == 1)], '⑥1位 かつ ⑦1位'),
      stat(te[(te['上り順予測'] == 1) & (te['⑦順'].between(2, 3))], '⑥1位 かつ ⑦2-3位'),
      stat(te[(te['上り順予測'] == 1) & (te['⑦順'].between(4, 6))], '⑥1位 かつ ⑦4-6位'),
      stat(te[(te['上り順予測'] == 1) & (te['⑦順'] >= 7)], '⑥1位 かつ ⑦7位以下')]
print(pd.DataFrame(r2).to_string(index=False))

print("\n【5】人気帯を揃えた比較（⑥の順位は市場に対して情報を足すか）")
for lab, lo, hi in [('1人気', 1, 1), ('2-3人気', 2, 3), ('4-6人気', 4, 6), ('7人気以下', 7, 99)]:
    sub = te[te['市場順'].between(lo, hi)].copy()
    if len(sub) < 500: continue
    sub['⑥半'] = np.where(sub['上り順予測'] <= sub['頭数'] / 3, '⑥上位1/3', '⑥それ以外')
    print(f"\n[{lab}]")
    print(pd.DataFrame([stat(sub[sub['⑥半'] == k], k) for k in ['⑥上位1/3', '⑥それ以外']]).to_string(index=False))
