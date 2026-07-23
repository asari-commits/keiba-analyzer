# -*- coding: utf-8 -*-
"""
ラップ検証の土台づくり & 前提確認スクリプト。
Q1: 1ハロンごとのラップ(11.8-10.3-...)を配列として取り込めているか
Q2: コース(競馬場×距離×芝ダ)単位の平均ラップ/傾向を集計できているか
Q3: 好走馬の「適したラップ傾向/適さないラップ傾向」を馬単位で把握できているか

成果物:
  data/processed/lap_course_profile.parquet  … コース別ラップ傾向
  data/processed/lap_master_joined.parquet   … master各行にラップ形状を付与(per-horse, リーク前の生特徴)
"""
import re
import numpy as np
import pandas as pd

MASTER = r"C:\Users\asari\Downloads\Claude\keiba-analyzer\data\processed\master.parquet"
LAP = r"C:\Users\asari\OneDrive\競馬ツール用データ\ラップタイム分析用\data_raw\2016年以降全場ラップタイム分析用データ.csv"
OUT_COURSE = r"C:\Users\asari\Downloads\Claude\keiba-analyzer\data\processed\lap_course_profile.parquet"
OUT_JOIN   = r"C:\Users\asari\Downloads\Claude\keiba-analyzer\data\processed\lap_master_joined.parquet"

def norm_kaisai(s):   # '3新1'->'3新', '1新12'->'1新'
    return re.sub(r"[\dA-C]+$", "", str(s))
def venue_token(s):   # '1新1'->'新'  開催コードから場略称のみ
    return re.sub(r"\d", "", str(s))
def lap_date_to_yymmdd(s):
    try:
        y, m, d = str(s).split(".")
        return f"{int(y)%100:02d}{int(m):02d}{int(d):02d}"
    except Exception:
        return None
def parse_splits(s):
    try:
        return [float(x) for x in str(s).split("-") if x not in ("", "nan")]
    except Exception:
        return []

# ================= LAP 読み込み & Q1: ハロン配列化 =================
lap = pd.read_csv(LAP, encoding="cp932")
lap["splits"] = lap["ラップタイム"].map(parse_splits)
lap["n_splits"] = lap["splits"].map(len)
lap["距離"] = pd.to_numeric(lap["距離"], errors="coerce")
lap["expected_furlongs"] = (lap["距離"] / 200)

print("=" * 64)
print("Q1: 1ハロンごとのラップ配列化")
ok = (lap["n_splits"] == lap["expected_furlongs"]).sum()
have = (lap["n_splits"] > 0).sum()
print(f"  総レース={len(lap)}  配列取得={have}  距離/200と本数一致={ok} ({ok/len(lap)*100:.1f}%)")
print(f"  本数一致しない例(端数コース等)の距離別件数:")
mism = lap[lap["n_splits"] != lap["expected_furlongs"]]
print("   ", mism["距離"].value_counts().head(8).to_dict())
print("  サンプル:", lap["距離"].iloc[0], "m ->", lap["splits"].iloc[0], f"({lap['n_splits'].iloc[0]}本)")

# ================= Q2: コース別 平均ラップ/傾向 =================
print("=" * 64)
print("Q2: コース(競馬場×芝ダ×距離)別 平均ラップ・傾向")
lap["venue"] = lap["開催"].map(venue_token)
for c in ["前3F","後3F","前5F","後5F","RPCI","PCI3"]:
    lap[c] = pd.to_numeric(lap[c], errors="coerce")
# 後半3F差 = 後3F - 前3F。数字は秒(小さいほど速い)なので:
#   >0: 後半が遅い = 前傾ラップ(ハイペース/消耗戦/先行有利)
#   <0: 後半が速い = 後傾ラップ(スロー/瞬発戦/差し有利)
lap["後半3F差"] = lap["後3F"] - lap["前3F"]
lap["馬場状態"] = lap["馬場状態"].astype(str)

course = lap.groupby(["venue","TD","距離"]).agg(
    n=("後半3F差","size"),
    前3F平均=("前3F","mean"), 後3F平均=("後3F","mean"),
    後半3F差平均=("後半3F差","mean"), 後半3F差sd=("後半3F差","std"),
    RPCI平均=("RPCI","mean"),
).reset_index()
course = course[course["n"] >= 20].copy()   # 標本20以上のコースのみ
for c in ["前3F平均","後3F平均","後半3F差平均","後半3F差sd","RPCI平均"]:
    course[c] = course[c].round(2)
# 後半3F差平均が負ほど後半が速い=後傾/瞬発
course["pace_type"] = np.where(course["後半3F差平均"]<-0.3,"後傾(瞬発寄り)",
                        np.where(course["後半3F差平均"]>0.3,"前傾(消耗寄り)","中間"))
print(f"  集計コース数(n>=20): {len(course)}")
print("  例(芝・主要距離):")
ex = course[(course["TD"]=="芝") & (course["距離"].isin([1600,2000,2400]))].sort_values(["距離","venue"])
print(ex.head(15).to_string(index=False))
course.to_parquet(OUT_COURSE, index=False)
print(f"  -> 保存: {OUT_COURSE}")

# ================= master 結合(指紋キー) =================
print("=" * 64)
print("master結合(指紋キー) & per-horse ラップ形状付与")
lapj = lap.copy()
lapj["k_date"] = lapj["日付"].map(lap_date_to_yymmdd)
lapj["k_kai"]  = lapj["開催"].map(norm_kaisai)
for c,src in [("k_dist","距離"),("k_n","頭数"),("k_f1","1着4角"),("k_f2","2着4角"),("k_f3","3着4角")]:
    lapj[c] = pd.to_numeric(lapj[src], errors="coerce")
keyc = ["k_date","k_kai","k_dist","k_n","k_f1","k_f2","k_f3"]
lapj = lapj.dropna(subset=keyc).copy()
for c in ["k_dist","k_n","k_f1","k_f2","k_f3"]:
    lapj[c] = lapj[c].astype(int)
lapj["fp"] = lapj[keyc].astype(str).agg("|".join, axis=1)
lapj = lapj[~lapj["fp"].duplicated(keep=False)].drop_duplicates("fp")

lap_feats = lapj.set_index("fp")[["splits","n_splits","venue","前3F","後3F","前5F","後5F",
                                   "RPCI","PCI3","後半3F差","1着決手","2着決手","3着決手"]]
lap_feats = lap_feats.rename(columns=lambda c: "lap_"+c if not c.startswith("lap_") else c)

m = pd.read_parquet(MASTER)
m["着順_num"] = pd.to_numeric(m["着順_num"], errors="coerce")
m = m[m["着順_num"].notna()].copy()
race_key = ["日付","開催","Ｒ"]
ms = m.sort_values(race_key+["着順_num"])
top3 = ms.groupby(race_key, sort=False).head(3).copy()
top3["pos"] = top3.groupby(race_key, sort=False).cumcount()+1
corners = top3.pivot_table(index=race_key, columns="pos", values="4角", aggfunc="first")
corners.columns=[f"mf{c}" for c in corners.columns]
corners=corners.reset_index()
mrace = m.groupby(race_key).agg(距離=("距離","first"),頭数=("頭数","first")).reset_index().merge(corners,on=race_key)
mrace = mrace.dropna(subset=["mf1","mf2","mf3"])
mrace["k_date"]=mrace["日付"].astype(int).astype(str).str.zfill(6)
mrace["k_kai"]=mrace["開催"].map(norm_kaisai)
for c,src in [("k_dist","距離"),("k_n","頭数"),("k_f1","mf1"),("k_f2","mf2"),("k_f3","mf3")]:
    mrace[c]=mrace[src].astype(int)
mrace["fp"]=mrace[keyc].astype(str).agg("|".join,axis=1)
mrace=mrace.merge(lap_feats,left_on="fp",right_index=True,how="left")
matched=mrace["lap_RPCI"].notna().sum()
print(f"  master races={len(mrace)} matched={matched} ({matched/len(mrace)*100:.1f}%)")

joined = m.merge(mrace[race_key+list(lap_feats.columns)], on=race_key, how="left")
joined = joined[joined["lap_RPCI"].notna()].copy()
# 保存(list列splitsはparquet可)
keep = ["日付","日付_dt","開催","Ｒ","馬名","着順_num","人気","距離","芝・ダ","頭数",
        "lap_splits","lap_n_splits","lap_venue","lap_前3F","lap_後3F","lap_前5F","lap_後5F",
        "lap_RPCI","lap_PCI3","lap_後半3F差","lap_1着決手"]
joined[keep].to_parquet(OUT_JOIN, index=False)
print(f"  -> 保存: {OUT_JOIN}  ({len(joined):,}行 per-horse)")

# ================= Q3: 馬単位の適性傾向デモ =================
print("=" * 64)
print("Q3: 馬単位『適したラップ傾向/適さないラップ傾向』の把握(デモ)")
j = joined.copy()
j["good"] = (j["着順_num"]<=3).astype(int)
j["distbin"] = j["距離"]//200*200
grp = j.groupby(["芝・ダ","distbin"])["lap_後半3F差"]
# z<0=後傾/瞬発, z>0=前傾/消耗 (元指標の符号を継承)
j["ペースz"] = (j["lap_後半3F差"]-grp.transform("mean"))/grp.transform("std")
def pace_label(z):
    return "後傾/瞬発戦" if z < 0 else "前傾/消耗戦"
# 出走数の多い馬で例示
cnt = j["馬名"].value_counts()
demo_horse = cnt[cnt.between(8,20)].index[0]
h = j[j["馬名"]==demo_horse].sort_values("日付_dt")
gd = h[h["good"]==1]["ペースz"]
bd = h[h["good"]==0]["ペースz"]
print(f"  例: 『{demo_horse}』 出走{len(h)} / 好走{h['good'].sum()}")
print(f"    適したラップ傾向 = 好走時のペースz 平均 {gd.mean():+.2f} (n={len(gd)})  "
      f"→ {pace_label(gd.mean())}向き")
print(f"    適さない傾向     = 凡走時のペースz 平均 {bd.mean():+.2f} (n={len(bd)})")
print(f"    好走レースの実ラップ例:")
for _,r in h[h['good']==1].head(3).iterrows():
    print(f"      {int(r['日付'])} {r['芝・ダ']}{int(r['距離'])}m 着{int(r['着順_num'])} "
          f"ペースz={r['ペースz']:+.2f}({pace_label(r['ペースz'])}) splits={r['lap_splits']}")
print("\n  ※この『好走時z平均』を全馬・過去のみ(リークなし)で集約したものが適性スコア。")
print("  ※z<0=後傾/瞬発, z>0=前傾/消耗。前後半バランス/決手/緩急(lap_std)へ多次元化可能。")
