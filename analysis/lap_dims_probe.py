# -*- coding: utf-8 -*-
"""
未検証3次元(実ラップの緩急/決手一貫性/区間ラップ形状)の「生死判定」。
各次元について、馬の過去好走から作る適性(pref, リークなし)と
当該レースの実現形状(oracle=上限)のsign一致(同/逆)で好走率に差が出るかを、
人気帯を揃えて確認。ペース傾斜は 1-3番人気で 同50.7% vs 逆21.1% だった(基準)。
"""
import numpy as np
import pandas as pd

JOIN = r"C:\Users\asari\Downloads\Claude\keiba-analyzer\data\processed\lap_master_joined.parquet"
j = pd.read_parquet(JOIN)
j["着順_num"] = pd.to_numeric(j["着順_num"], errors="coerce")
j = j[j["着順_num"].notna()].copy()
j["人気"] = pd.to_numeric(j["人気"], errors="coerce")
j["good"] = (j["着順_num"] <= 3).astype(int)
j["distbin"] = (j["距離"] // 200 * 200).astype(int)

# ---- 区間形状スカラーをハロン配列から算出 ----
def shape_metrics(splits):
    a = np.array(splits, dtype=float)
    if len(a) < 4:
        return (np.nan, np.nan, np.nan)
    cv = a.std() / a.mean()                 # 緩急(大=不均一/緩急激しい)
    min_pos = int(np.argmin(a)) / (len(a) - 1)   # 最速ハロン位置(0早い〜1遅い=末脚)
    final_drop = a[-1] - a.min()            # 最終失速幅(大=最後止まる/持続戦)
    return (cv, min_pos, final_drop)

sm = j["lap_splits"].map(shape_metrics)
j["m_cv"] = sm.map(lambda t: t[0])          # 緩急
j["m_spurtpos"] = sm.map(lambda t: t[1])    # 区間形状: 末脚位置
j["m_finaldrop"] = sm.map(lambda t: t[2])   # 区間形状: 最終失速

# 決手一貫性: 1着決手が先行(逃/先)決着か
front = {"逃", "先"}
j["m_kette_front"] = j["lap_1着決手"].map(lambda x: 1.0 if str(x) in front else 0.0)
j.loc[j["lap_1着決手"].isna(), "m_kette_front"] = np.nan

METRICS = {
    "緩急(cv)": "m_cv",
    "末脚位置(区間形状)": "m_spurtpos",
    "最終失速(区間形状)": "m_finaldrop",
    "決手先行決着": "m_kette_front",
}

def run_probe(name, col):
    d = j.dropna(subset=[col]).copy()
    # (芝ダ×距離帯)内 z標準化
    g = d.groupby(["芝・ダ", "distbin"])[col]
    d["z"] = (d[col] - g.transform("mean")) / g.transform("std")
    d = d.dropna(subset=["z"]).sort_values(["馬名", "日付_dt"]).reset_index(drop=True)
    hg = d.groupby("馬名", sort=False)
    d["_gz"] = d["z"] * d["good"]; d["_gz2"] = d["z"]**2 * d["good"]
    cs, cs2, cg = hg["_gz"].cumsum(), hg["_gz2"].cumsum(), hg["good"].cumsum()
    ps = cs.groupby(d["馬名"]).shift(1); ps2 = cs2.groupby(d["馬名"]).shift(1)
    pc = cg.groupby(d["馬名"]).shift(1)
    d["pref"] = ps / pc; d["npg"] = pc.fillna(0)
    var = (ps2/pc - d["pref"]**2) * (pc/(pc-1))
    d["prefstd"] = np.sqrt(var.clip(lower=0))
    spec = d[(d["npg"] >= 2) & (d["prefstd"] <= 0.6) & (d["pref"].abs() >= 0.5)].copy()
    spec["align"] = np.where(np.sign(spec["z"]) == np.sign(spec["pref"]), "同", "逆")
    spec["pb"] = pd.cut(spec["人気"], [0, 3, 6, 9, 99], labels=["1-3", "4-6", "7-9", "10+"])
    print(f"\n=== {name}  (専門家 n={len(spec)}) ===")
    pv = spec.pivot_table(index="pb", columns="align", values="good",
                          aggfunc=["mean", "size"], observed=True)
    for b in ["1-3", "4-6", "7-9", "10+"]:
        try:
            am, an = pv[("mean", "同")][b]*100, int(pv[("size", "同")][b])
            om, on = pv[("mean", "逆")][b]*100, int(pv[("size", "逆")][b])
            print(f"  人気{b:>4}: 同 {am:5.1f}%(n={an:4d})  逆 {om:5.1f}%(n={on:4d})  差={am-om:+5.1f}pt")
        except Exception:
            print(f"  人気{b:>4}: n不足")

print("基準(ペース傾斜): 1-3番 同50.7 vs 逆21.1 (+29.6pt) / 4-6番 +17.5pt")
for name, col in METRICS.items():
    run_probe(name, col)
