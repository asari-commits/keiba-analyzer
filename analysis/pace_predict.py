# -*- coding: utf-8 -*-
"""
① 想定ペースの事前推定。
賭け時に既知の情報(コース/頭数/出走馬の脚質構成/馬場/クラス)だけから、
レースの実現ペース(後傾度z)を回帰予測できるか。さらに予測ペースでも
「専門家×逆ペース」の負シグナル(oracleでは24%vs6.6%)が生き残るかを検証。

リーク防止:
  - 馬の脚質(style) = 過去のみ(4角/頭数の累積平均をshift)
  - コース基準ペース = 過去のみ(会場×芝ダ×距離帯の後傾度z 累積平均をshift)
  - 馬の適性pref  = 過去好走のみ(既存手法)
  - 学習/評価は時系列split(〜2023 train / 2024〜 test)
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

JOIN = r"C:\Users\asari\Downloads\Claude\keiba-analyzer\data\processed\lap_master_joined.parquet"
MASTER = r"C:\Users\asari\Downloads\Claude\keiba-analyzer\data\processed\master.parquet"

# ---------- master から脚質素材 ----------
m = pd.read_parquet(MASTER)
m["着順_num"] = pd.to_numeric(m["着順_num"], errors="coerce")
m["4角"] = pd.to_numeric(m["4角"], errors="coerce")
m["頭数"] = pd.to_numeric(m["頭数"], errors="coerce")
m = m[m["着順_num"].notna()].copy()
m["pos_ratio"] = (m["4角"] / m["頭数"]).clip(0, 1)   # 0=逃げ 1=追込
m = m.sort_values(["馬名", "日付_dt"]).reset_index(drop=True)
# 馬の脚質(過去のみ): pos_ratio の累積平均を shift
g = m.groupby("馬名", sort=False)
m["_cs"] = g["pos_ratio"].cumsum(); m["_cc"] = g["pos_ratio"].cumcount() + 1
m["style_prior"] = (g["_cs"].shift(1) / g["_cc"].shift(1))   # 過去平均脚質

# ---------- ラップ(実現ペース) join済みを race 単位へ ----------
j = pd.read_parquet(JOIN)
race_key = ["日付", "開催", "Ｒ"]
# style_prior を per-horse で m から j へ付与
m_key = m[race_key + ["馬名", "style_prior"]]
j = j.merge(m_key, on=race_key + ["馬名"], how="left")

# race 単位の集約
def race_agg(df):
    sp = df["style_prior"]
    return pd.Series({
        "距離": df["距離"].iloc[0], "芝・ダ": df["芝・ダ"].iloc[0],
        "頭数": df["頭数"].iloc[0], "venue": df["lap_venue"].iloc[0],
        "後半3F差": df["lap_後半3F差"].iloc[0],
        "field_style_mean": sp.mean(),
        "field_front_share": (sp < 0.33).mean(),   # 先行勢の比率
        "field_style_sd": sp.std(),
        "n_style_known": sp.notna().sum(),
        "日付_dt": df["日付_dt"].iloc[0],
    })

race = j.groupby(race_key, sort=False).apply(race_agg, include_groups=False).reset_index()
race = race.dropna(subset=["後半3F差", "距離", "頭数"]).copy()
race["distbin"] = (race["距離"] // 200 * 200).astype(int)
# 実現ペースを (芝ダ×距離帯) 内で z 標準化 = 予測ターゲット
# pace_z<0 = 後傾/瞬発(差し有利), pace_z>0 = 前傾/消耗(先行有利)
grp = race.groupby(["芝・ダ", "distbin"])["後半3F差"]
race["pace_z"] = (race["後半3F差"] - grp.transform("mean")) / grp.transform("std")
race = race.dropna(subset=["pace_z"]).sort_values("日付_dt").reset_index(drop=True)

# コース基準ペース(過去のみ): venue×芝ダ×distbin の pace_z 累積平均を shift
cg = race.groupby(["venue", "芝・ダ", "distbin"], sort=False)
race["_ccs"] = cg["pace_z"].cumsum(); race["_ccc"] = cg["pace_z"].cumcount() + 1
race["course_base_z"] = cg["_ccs"].shift(1) / cg["_ccc"].shift(1)

# ---------- 特徴量 & 時系列split ----------
race["surf"] = (race["芝・ダ"] == "芝").astype(int)
feat = ["course_base_z", "頭数", "field_style_mean", "field_front_share",
        "field_style_sd", "distbin", "surf"]
race["year"] = race["日付_dt"].dt.year
tr = race[race["year"] <= 2023].dropna(subset=feat + ["pace_z"])
te = race[race["year"] >= 2024].dropna(subset=feat + ["pace_z"])
print(f"race総数={len(race)}  train(-2023)={len(tr)}  test(2024-)={len(te)}")

model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31,
                          subsample=0.8, colsample_bytree=0.8, min_child_samples=50,
                          random_state=0, verbose=-1)
model.fit(tr[feat], tr["pace_z"])
te = te.copy(); te["pace_hat"] = model.predict(te[feat])

def report_corr(name, df):
    r = np.corrcoef(df["pace_hat"], df["pace_z"])[0, 1]
    # course_base単体の相関(ベースライン)
    rb = np.corrcoef(df["course_base_z"], df["pace_z"])[0, 1]
    print(f"  [{name}] OOS corr(pred, actual)={r:.3f}  R2={r**2:.3f}  "
          f"| baseline course_base単体 corr={rb:.3f}")
print("=" * 60)
print("① ペース予測 OOS 精度")
report_corr("test2024-", te)
imp = pd.Series(model.feature_importances_, index=feat).sort_values(ascending=False)
print("  feature importance:", imp.to_dict())

# 全期間の予測(専門家テスト用に oof 的に: train期間はtrainモデル, test期間はtestで)
race["pace_hat"] = model.predict(race[feat].fillna(race[feat].median()))

# ---------- 予測ペースで専門家シグナル再検証 ----------
print("=" * 60)
print("② 予測ペースで『専門家×逆ペース』が残るか (oracle: 24.3% vs 6.6%)")
# per-horse: 適性pref(過去好走z, 過去のみ) を foundation と同じ流儀で作る
j2 = j.merge(race[race_key + ["pace_z", "pace_hat"]], on=race_key, how="inner")
j2["good"] = (pd.to_numeric(j2["着順_num"], errors="coerce") <= 3).astype(int)
j2 = j2.sort_values(["馬名", "日付_dt"]).reset_index(drop=True)
jg = j2.groupby("馬名", sort=False)
j2["_gz"] = j2["pace_z"] * j2["good"]
# cumsum then shift within group (過去のみ)
j2["_cum_gz"] = jg["_gz"].cumsum(); j2["_cum_g"] = jg["good"].cumsum()
j2["pref_z"] = jg["_cum_gz"].shift(1) / jg["_cum_g"].shift(1)
j2["n_past_good"] = jg["_cum_g"].shift(1).fillna(0)
# 専門家: 過去好走zのブレ小 & 選好強
hstd = j2[j2["good"] == 1].groupby("馬名")["pace_z"].std()
j2 = j2.merge(hstd.rename("good_z_std"), on="馬名", how="left")
j2["人気"] = pd.to_numeric(j2["人気"], errors="coerce")

spec = j2[(j2["n_past_good"] >= 2) & (j2["good_z_std"] <= 0.6) &
          (j2["pref_z"].abs() >= 0.5)].copy()
spec["pop_band"] = pd.cut(spec["人気"], [0, 3, 6, 9, 99], labels=["1-3", "4-6", "7-9", "10+"])

for label, col in [("oracle実ペース", "pace_z"), ("①予測ペース", "pace_hat")]:
    spec["align"] = np.where(np.sign(spec[col]) == np.sign(spec["pref_z"]), "同", "逆")
    print(f"\n--- {label} で判定 (専門家 n={len(spec)}) ---")
    pv = spec.pivot_table(index="pop_band", columns="align", values="good",
                          aggfunc=["mean", "size"], observed=True)
    for b in ["1-3", "4-6", "7-9", "10+"]:
        try:
            am, an = pv[("mean", "同")][b] * 100, int(pv[("size", "同")][b])
            om, on = pv[("mean", "逆")][b] * 100, int(pv[("size", "逆")][b])
            print(f"  人気{b:>4}: 同 {am:5.1f}%(n={an:4d})  逆 {om:5.1f}%(n={on:4d})  差={am-om:+5.1f}pt")
        except Exception as e:
            print(f"  人気{b}: {e}")
