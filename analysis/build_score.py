# -*- coding: utf-8 -*-
"""5ファクターを1つの勝率スコアに束ねる。学習＋時系列検証＋係数の書き出し。

  python analysis/build_score.py

出力: analysis/score_model.json   （race_factors.py がこれを読んで ⑦統合スコア を出す）

方針
  - 特徴量はすべてレース前に確定するもの（過去走の shift 済み集計）のみ
  - 血統の勝率は「学習期間だけ」で作り、検証期間には当てはめるだけ（リーク防止）
  - オッズは入れない。市場と独立な意見を作り、あとで突き合わせて妙味を測るため
"""
import json, sys, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore'); sys.stdout.reconfigure(encoding='utf-8')
pd.set_option('display.width', 340)
DATA = r"C:\Users\asari\Downloads\Claude\keiba-analyzer\data\processed"
OUT = r"C:\Users\asari\Downloads\Claude\keiba-analyzer\analysis\score_model.json"
n_ = lambda s: pd.to_numeric(s, errors='coerce')
AG = dict(const=0.4003, p_relag5=0.50028, p_rel4_3=-0.03872, p_pci5=-0.00234, cls差=0.01712, 距差100=0.01276)

FEATS = ['pred_relag', 'p_rel4_3', 'p_push5', 'p_pci5', 'sire_r', 'bms_r',
         'career', 'weight', 'cls差', '距差100', 'n_field', 'rest_w']

print("読み込み中…")
M = pd.read_parquet(DATA + r"\master.parquet")
M['着'] = n_(M['着順_num']); M = M.dropna(subset=['着'])
M['距'] = n_(M['距離'].astype(str).str.extract(r'(\d+)')[0])
for c in ['頭数', '3角', '4角', 'クラス_num', 'キャリア', '馬体重']:
    M[c] = n_(M[c])
M['ag'] = n_(M['上り3F']); M['pci'] = n_(M['PCI'])
M['rel4'] = (M['4角'] / M['頭数']).round(3); M['rel3'] = (M['3角'] / M['頭数']).round(3)
M['push'] = (M['rel3'] - M['rel4']).round(3)
M['上順'] = M.groupby(['日付', '開催', 'Ｒ'])['ag'].rank(method='min')
M['relag'] = (M['上順'] / M['頭数']).round(3)
M['dt'] = pd.to_datetime('20' + M['日付'].astype(str), format='%Y%m%d', errors='coerce')
M['tan'] = n_(M['単勝配当'].astype(str).str.replace(r'[()（）,]', '', regex=True)).fillna(0)
M['distbin'] = (M['距'] // 400 * 400)
M = M.dropna(subset=['dt', 'relag', 'rel4']).sort_values(['馬名', 'dt']).reset_index(drop=True)

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

CUT = M['dt'].quantile(0.75)
tr_mask, te_mask = M['dt'] <= CUT, M['dt'] > CUT
print(f"学習 〜{CUT.date()} / 検証 {CUT.date()}〜")

# --- 血統の勝率は学習期間のみで作る（芝ダ×400m帯ごと） ---
def rate_table(df, col):
    t = df.groupby(['芝・ダ', 'distbin', col])['win'].agg(['size', 'mean'])
    t = t[t['size'] >= 20]['mean']
    return t.to_dict()


sire = rate_table(M[tr_mask], '種牡馬'); bms = rate_table(M[tr_mask], '母父馬')
base = M[tr_mask]['win'].mean()
key = list(zip(M['芝・ダ'], M['distbin'], M['種牡馬']))
M['sire_r'] = [sire.get(k, base) for k in key]
key2 = list(zip(M['芝・ダ'], M['distbin'], M['母父馬']))
M['bms_r'] = [bms.get(k, base) for k in key2]
print(f"血統テーブル: 種牡馬{len(sire)}件 / 母父{len(bms)}件（各n>=20、学習期間のみ）")

D = M.dropna(subset=['p_relag5', 'p_rel4_3']).copy()
for c in FEATS:
    D[c] = D[c].fillna(D.loc[tr_mask.reindex(D.index, fill_value=False), c].median() if c in D else 0)
tr, te = D[D['dt'] <= CUT], D[D['dt'] > CUT]
print(f"学習 {len(tr):,}行 / 検証 {len(te):,}行  （勝率 基準 {tr['win'].mean()*100:.1f}%）\n")

mu, sd = tr[FEATS].mean(), tr[FEATS].std().replace(0, 1)
Xtr = ((tr[FEATS] - mu) / sd).values; Xte = ((te[FEATS] - mu) / sd).values
clf = LogisticRegression(max_iter=2000, C=1.0)
clf.fit(Xtr, tr['win'].values)
te = te.copy(); te['p'] = clf.predict_proba(Xte)[:, 1]
# レース内で正規化 → 出走各馬の勝率の合計が1になる
te['score'] = te.groupby(['日付', '開催', 'Ｒ'])['p'].transform(lambda s: s / s.sum())

print("■ 係数（標準化後。正=勝ちやすい方向）")
co = pd.Series(clf.coef_[0], index=FEATS).sort_values(key=abs, ascending=False)
print(co.round(4).to_string())

print("\n■ 検証期間：スコア10分位ごとの実績")
te['dec'] = pd.qcut(te['score'], 10, labels=False, duplicates='drop') + 1
t = te.groupby('dec').agg(n=('win', 'size'), 勝率=('win', lambda s: round(s.mean() * 100, 1)),
                          平均スコア=('score', lambda s: round(s.mean() * 100, 1)),
                          単ROI=('tan', lambda s: round(s.mean(), 0)))
print(t.to_string())

print("\n■ レース内で最上位に評価した馬の成績（＝この模型の本命）")
top = te.loc[te.groupby(['日付', '開催', 'Ｒ'])['score'].idxmax()]
print(f"  n={len(top):,}  勝率{top['win'].mean()*100:.1f}%  単ROI{top['tan'].mean():.0f}")
print("\n■ 参考：市場（1番人気）")
te['人気'] = n_(te['人気'])
pop = te[te['人気'] == 1]
print(f"  n={len(pop):,}  勝率{pop['win'].mean()*100:.1f}%  単ROI{pop['tan'].mean():.0f}")

print("\n■ 妙味の検証：模型のスコア vs 市場の支持率")
te['mkt'] = te.groupby(['日付', '開催', 'Ｒ'])['人気'].transform(lambda s: 1.0 / s)
te['mkt'] = te.groupby(['日付', '開催', 'Ｒ'])['mkt'].transform(lambda s: s / s.sum())
te['edge'] = te['score'] - te['mkt']
te['eb'] = pd.qcut(te['edge'], 5, labels=['①模型<市場', '②', '③', '④', '⑤模型>市場'])
print(te.groupby('eb', observed=True).agg(n=('win', 'size'), 勝率=('win', lambda s: round(s.mean() * 100, 1)),
                                          単ROI=('tan', lambda s: round(s.mean(), 0))).to_string())

json.dump(dict(feats=FEATS, coef=clf.coef_[0].tolist(), intercept=float(clf.intercept_[0]),
               mu=mu.to_dict(), sd=sd.to_dict(), ag=AG, base=float(base),
               sire={f"{k[0]}|{int(k[1])}|{k[2]}": v for k, v in sire.items()},
               bms={f"{k[0]}|{int(k[1])}|{k[2]}": v for k, v in bms.items()},
               cut=str(CUT.date())), open(OUT, 'w', encoding='utf-8'), ensure_ascii=False)
print(f"\n保存: {OUT}")
