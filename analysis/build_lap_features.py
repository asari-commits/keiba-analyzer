# -*- coding: utf-8 -*-
"""
A用: 現行モデルに足す「ラップ適性フィット」特徴量を per-horse-race で生成し保存。
すべてリーク防止(過去のみ / 時系列split で学習したペースモデルで予測)。

出力: data/processed/lap_features.parquet
  キー: 日付, 開催, Ｒ, 馬名
  列:
    lap_pace_hat        … 想定ペースz (出走馬の脚質構成+コースから事前推定, ①モデル)
    lap_pref_z          … 馬の得意ペースz (過去好走レースのz平均, 過去のみ)
    lap_pref_std_past   … 得意ペースのブレ(過去好走zのsd, 過去のみ)
    lap_n_past_good     … 過去好走数
    lap_is_specialist   … 明確なペース専門家か(0/1)
    lap_pace_fit        … 適合度 = lap_pref_z * lap_pace_hat (正=得意ペース, 負=逆ペース)
    lap_pace_fit_spec   … lap_pace_fit を専門家のみ有効化(それ以外0)
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

JOIN   = r"C:\Users\asari\Downloads\Claude\keiba-analyzer\data\processed\lap_master_joined.parquet"
MASTER = r"C:\Users\asari\Downloads\Claude\keiba-analyzer\data\processed\master.parquet"
OUT    = r"C:\Users\asari\Downloads\Claude\keiba-analyzer\data\processed\lap_features.parquet"

race_key = ["日付", "開催", "Ｒ"]

# ---- master: 脚質(過去のみ) ----
m = pd.read_parquet(MASTER)
m["着順_num"] = pd.to_numeric(m["着順_num"], errors="coerce")
m["4角"] = pd.to_numeric(m["4角"], errors="coerce")
m["頭数"] = pd.to_numeric(m["頭数"], errors="coerce")
m = m[m["着順_num"].notna()].copy()
m["pos_ratio"] = (m["4角"] / m["頭数"]).clip(0, 1)
m = m.sort_values(["馬名", "日付_dt"]).reset_index(drop=True)
g = m.groupby("馬名", sort=False)
m["_cs"] = g["pos_ratio"].cumsum(); m["_cc"] = g["pos_ratio"].cumcount() + 1
m["style_prior"] = g["_cs"].shift(1) / g["_cc"].shift(1)

# ---- ラップ join → race集約 ----
j = pd.read_parquet(JOIN)
j = j.merge(m[race_key + ["馬名", "style_prior"]], on=race_key + ["馬名"], how="left")

def race_agg(df):
    sp = df["style_prior"]
    return pd.Series({
        "距離": df["距離"].iloc[0], "芝・ダ": df["芝・ダ"].iloc[0],
        "頭数": df["頭数"].iloc[0], "venue": df["lap_venue"].iloc[0],
        "後半3F差": df["lap_後半3F差"].iloc[0],
        "field_style_mean": sp.mean(), "field_front_share": (sp < 0.33).mean(),
        "field_style_sd": sp.std(), "日付_dt": df["日付_dt"].iloc[0],
    })

race = j.groupby(race_key, sort=False).apply(race_agg, include_groups=False).reset_index()
race = race.dropna(subset=["後半3F差", "距離", "頭数"]).copy()
race["distbin"] = (race["距離"] // 200 * 200).astype(int)
grp = race.groupby(["芝・ダ", "distbin"])["後半3F差"]
race["pace_z"] = (race["後半3F差"] - grp.transform("mean")) / grp.transform("std")
race = race.dropna(subset=["pace_z"]).sort_values("日付_dt").reset_index(drop=True)
cg = race.groupby(["venue", "芝・ダ", "distbin"], sort=False)
race["_ccs"] = cg["pace_z"].cumsum(); race["_ccc"] = cg["pace_z"].cumcount() + 1
race["course_base_z"] = cg["_ccs"].shift(1) / cg["_ccc"].shift(1)
race["surf"] = (race["芝・ダ"] == "芝").astype(int)

# ---- ①ペース予測モデル(≤2023学習)で全レースの pace_hat ----
feat = ["course_base_z", "頭数", "field_style_mean", "field_front_share",
        "field_style_sd", "distbin", "surf"]
race["year"] = race["日付_dt"].dt.year
tr = race[race["year"] <= 2023].dropna(subset=feat + ["pace_z"])
pace_model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31,
                               subsample=0.8, colsample_bytree=0.8, min_child_samples=50,
                               random_state=0, verbose=-1)
pace_model.fit(tr[feat], tr["pace_z"])
race["lap_pace_hat"] = pace_model.predict(race[feat].fillna(race[feat].median()))

# ---- per-horse: 得意ペース(過去好走のみ) ----
jj = j.merge(race[race_key + ["pace_z", "lap_pace_hat"]], on=race_key, how="inner")
jj["good"] = (pd.to_numeric(jj["着順_num"], errors="coerce") <= 3).astype(int)
jj = jj.sort_values(["馬名", "日付_dt"]).reset_index(drop=True)
hg = jj.groupby("馬名", sort=False)
jj["_gz"]  = jj["pace_z"] * jj["good"]
jj["_gz2"] = (jj["pace_z"] ** 2) * jj["good"]
cum_gz  = hg["_gz"].cumsum();  cum_gz2 = hg["_gz2"].cumsum(); cum_g = hg["good"].cumsum()
# 過去のみ(cumsum を shift して当該行を除外)
ps  = cum_gz.groupby(jj["馬名"]).shift(1)
ps2 = cum_gz2.groupby(jj["馬名"]).shift(1)
pc  = cum_g.groupby(jj["馬名"]).shift(1)
jj["lap_n_past_good"] = pc.fillna(0)
jj["lap_pref_z"] = ps / pc
mean2 = ps2 / pc
var = (mean2 - jj["lap_pref_z"] ** 2) * (pc / (pc - 1))
jj["lap_pref_std_past"] = np.sqrt(var.clip(lower=0))

jj["lap_is_specialist"] = ((jj["lap_n_past_good"] >= 2) &
                           (jj["lap_pref_std_past"] <= 0.6) &
                           (jj["lap_pref_z"].abs() >= 0.5)).astype(int)
jj["lap_pace_fit"] = (jj["lap_pref_z"] * jj["lap_pace_hat"]).fillna(0)
jj["lap_pace_fit_spec"] = jj["lap_pace_fit"] * jj["lap_is_specialist"]

out_cols = ["lap_pace_hat", "lap_pref_z", "lap_pref_std_past", "lap_n_past_good",
            "lap_is_specialist", "lap_pace_fit", "lap_pace_fit_spec"]
out = jj[race_key + ["馬名"] + out_cols].copy()
out.to_parquet(OUT, index=False)
print(f"保存: {OUT}")
print(f"行数={len(out):,}  専門家該当={int(out['lap_is_specialist'].sum()):,}  "
      f"pref_z有効={out['lap_pref_z'].notna().sum():,}")
print(out[out_cols].describe().round(3).to_string())
