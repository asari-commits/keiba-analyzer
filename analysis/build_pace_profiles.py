# -*- coding: utf-8 -*-
"""
解説UI用データ生成: 馬ごとのペース適性プロファイル。
各馬の過去レースを「実現ペースz(後半3F差の標準化)」軸に並べ、好走(3着内)を強調。
得意ペース = 好走レースのpace_z平均。
  z>0 = 前傾/ハイペース/消耗戦(押し切り型) が得意
  z<0 = 後傾/スロー/瞬発戦(末脚型) が得意
出力: analysis/pace_profiles.json (UIに埋め込む)
"""
import json
import numpy as np
import pandas as pd

JOIN = r"C:\Users\asari\Downloads\Claude\keiba-analyzer\data\processed\lap_master_joined.parquet"
OUT  = r"C:\Users\asari\Downloads\Claude\keiba-analyzer\analysis\pace_profiles.json"

j = pd.read_parquet(JOIN)
j["着順_num"] = pd.to_numeric(j["着順_num"], errors="coerce")
j = j[j["着順_num"].notna()].copy()
j["good"] = (j["着順_num"] <= 3).astype(int)
j["distbin"] = (j["距離"] // 200 * 200).astype(int)
g = j.groupby(["芝・ダ", "distbin"])["lap_後半3F差"]
j["pace_z"] = (j["lap_後半3F差"] - g.transform("mean")) / g.transform("std")
j = j.dropna(subset=["pace_z"]).copy()
j["year"] = j["日付_dt"].dt.year

# 対象: 6走以上 & 直近(2024+)に出走のある現役寄りの馬
stats = j.groupby("馬名").agg(n=("pace_z", "size"), last=("year", "max"),
                              ngood=("good", "sum")).reset_index()
target = stats[(stats["n"] >= 6) & (stats["last"] >= 2024) & (stats["ngood"] >= 1)]
target = target.sort_values(["last", "n"], ascending=False)
# ファイルサイズ抑制のため上位800頭
target_names = set(target["馬名"].head(800))
print(f"全馬={len(stats)}  条件該当={len(target)}  採用={len(target_names)}")

sub = j[j["馬名"].isin(target_names)].copy()

def verdict(pref, ngood, std):
    if ngood < 2 or (std is not None and std > 1.2):
        return ("傾向不明", "サンプル少/ばらつき大", "unknown")
    if pref >= 0.4:
        return ("ハイペース型", "速いペースで押し切るレースが得意（前傾・消耗戦）", "high")
    if pref <= -0.4:
        return ("瞬発型", "スローからの瞬発戦が得意（後傾・末脚）", "slow")
    return ("自在", "ペースを問わず走れる（どちらでも）", "flex")

horses = []
for name, h in sub.groupby("馬名"):
    h = h.sort_values("日付_dt")
    gd = h[h["good"] == 1]["pace_z"]
    pref = float(gd.mean()) if len(gd) else float("nan")
    pstd = float(gd.std()) if len(gd) >= 2 else None
    vt, vd, vc = verdict(pref, len(gd), pstd)
    races = [{
        "z": round(float(r.pace_z), 2),
        "g": int(r.good),
        "c": int(r.着順_num),
        "y": int(r.year),
        "s": str(r._asdict()["芝・ダ"]),
        "d": int(r.距離),
    } for r in h.itertuples(index=False)]
    horses.append({
        "name": name,
        "n": int(len(h)),
        "ngood": int(h["good"].sum()),
        "pref": None if np.isnan(pref) else round(pref, 2),
        "pstd": None if pstd is None else round(pstd, 2),
        "vt": vt, "vd": vd, "vc": vc,
        "last": int(h["year"].max()),
        "races": races,
    })

# 得意方向がはっきりしている順→名前順で安定ソート
horses.sort(key=lambda x: (-(x["pref"] is not None), -abs(x["pref"] or 0)))
with open(OUT, "w", encoding="utf-8") as f:
    json.dump({"horses": horses}, f, ensure_ascii=False)

import os
print(f"保存: {OUT}  ({os.path.getsize(OUT)/1024:.0f} KB, {len(horses)}頭)")
# 分布サマリ
vc = pd.Series([h["vc"] for h in horses]).value_counts()
print("タイプ分布:", vc.to_dict())
# 例
for h in horses[:3]:
    print(f"  {h['name']}: {h['vt']} pref={h['pref']} (n={h['n']},好走{h['ngood']})")
