# -*- coding: utf-8 -*-
"""
Step1 検証: ラップCSVから作る「馬の適性ラップスコア」で好走率(複勝率)に差が出るか。
リーク防止: 馬の適性は対象レースより前の好走レースのラップ形状のみから集約。
"""
import re
import numpy as np
import pandas as pd

MASTER = r"C:\Users\asari\Downloads\Claude\keiba-analyzer\data\processed\master.parquet"
LAP = r"C:\Users\asari\OneDrive\競馬ツール用データ\ラップタイム分析用\data_raw\2016年以降全場ラップタイム分析用データ.csv"

def norm_kaisai(s):
    # '3新1'->'3新', '1新12'->'1新', '1函C'->'1函'
    return re.sub(r"[\dA-C]+$", "", str(s))

def lap_date_to_yymmdd(s):
    # '2016. 4.30' -> '160430'
    try:
        y, m, d = str(s).split(".")
        return f"{int(y)%100:02d}{int(m):02d}{int(d):02d}"
    except Exception:
        return None

# ---------- LAP ----------
lap = pd.read_csv(LAP, encoding="cp932")
lap = lap[lap["1着4角"].notna() & lap["2着4角"].notna() & lap["3着4角"].notna()].copy()
lap["k_date"] = lap["日付"].map(lap_date_to_yymmdd)
lap["k_kai"] = lap["開催"].map(norm_kaisai)
lap["k_dist"] = pd.to_numeric(lap["距離"], errors="coerce")
lap["k_n"] = pd.to_numeric(lap["頭数"], errors="coerce")
lap["k_f1"] = pd.to_numeric(lap["1着4角"], errors="coerce")
lap["k_f2"] = pd.to_numeric(lap["2着4角"], errors="coerce")
lap["k_f3"] = pd.to_numeric(lap["3着4角"], errors="coerce")
lap["前3F"] = pd.to_numeric(lap["前3F"], errors="coerce")
lap["後3F"] = pd.to_numeric(lap["後3F"], errors="coerce")
lap["前5F"] = pd.to_numeric(lap["前5F"], errors="coerce")
lap["後5F"] = pd.to_numeric(lap["後5F"], errors="coerce")
lap["RPCI"] = pd.to_numeric(lap["RPCI"], errors="coerce")

# ラップ緩急(実ラップの標準偏差)
def lap_std(s):
    try:
        vals = [float(x) for x in str(s).split("-") if x not in ("", "nan")]
        return np.std(vals) if len(vals) >= 3 else np.nan
    except Exception:
        return np.nan
lap["lap_std"] = lap["ラップタイム"].map(lap_std)

lap_keycols = ["k_date","k_kai","k_dist","k_n","k_f1","k_f2","k_f3"]
lap_k = lap.dropna(subset=lap_keycols).copy()
for c in ["k_dist","k_n","k_f1","k_f2","k_f3"]:
    lap_k[c] = lap_k[c].astype(int)
lap_k["fp"] = lap_k[lap_keycols].astype(str).agg("|".join, axis=1)
# 重複キーは落とす(一意結合のため)
dup = lap_k["fp"].duplicated(keep=False)
print(f"LAP rows total={len(lap)}, keyable={len(lap_k)}, dup_fp_rows={dup.sum()}")
lap_u = lap_k[~dup].drop_duplicates("fp").set_index("fp")

# ---------- MASTER ----------
m = pd.read_parquet(MASTER)
m["着順_num"] = pd.to_numeric(m["着順_num"], errors="coerce")
m = m[m["着順_num"].notna()].copy()
m["fs"] = m.groupby(["日付","開催","Ｒ"])["馬番"].transform("count")  # race size in master

# race-level fingerprint from master: 1/2/3着の4角 (vectorized)
race_key = ["日付","開催","Ｒ"]
ms = m.sort_values(race_key+["着順_num"])
top3 = ms.groupby(race_key, sort=False).head(3).copy()
top3["pos"] = top3.groupby(race_key, sort=False).cumcount() + 1
corners = top3.pivot_table(index=race_key, columns="pos", values="4角", aggfunc="first")
corners.columns = [f"mf{c}" for c in corners.columns]
corners = corners.reset_index()
mrace = m.groupby(race_key).agg(距離=("距離","first"),頭数=("頭数","first")).reset_index()
mrace = mrace.merge(corners, on=race_key)
for c in ["mf1","mf2","mf3"]:
    if c not in mrace.columns: mrace[c] = np.nan
mrace["k_date"] = mrace["日付"].astype(int).astype(str).str.zfill(6)
mrace["k_kai"] = mrace["開催"].map(norm_kaisai)
mrace = mrace.dropna(subset=["mf1","mf2","mf3"])
for c,src in [("k_dist","距離"),("k_n","頭数"),("k_f1","mf1"),("k_f2","mf2"),("k_f3","mf3")]:
    mrace[c] = mrace[src].astype(int)
mrace["fp"] = mrace[["k_date","k_kai","k_dist","k_n","k_f1","k_f2","k_f3"]].astype(str).agg("|".join, axis=1)

# join lap features onto master races (lap_ prefix でmaster既存列との衝突回避)
lap_u = lap_u.rename(columns={"前3F":"lap_前3F","後3F":"lap_後3F","前5F":"lap_前5F",
                              "後5F":"lap_後5F","RPCI":"lap_RPCI","lap_std":"lap_std",
                              "1着決手":"lap_決手"})
feat_cols = ["lap_前3F","lap_後3F","lap_前5F","lap_後5F","lap_RPCI","lap_std","lap_決手"]
mrace = mrace.merge(lap_u[feat_cols], left_on="fp", right_index=True, how="left")
matched = mrace["lap_RPCI"].notna().sum()
print(f"MASTER races={len(mrace)}, matched to lap={matched} ({matched/len(mrace)*100:.1f}%)")

# ---------- attach to per-horse rows ----------
mm = m.merge(mrace[race_key+feat_cols], on=race_key, how="left")
mm = mm[mm["lap_RPCI"].notna()].copy()

# ペース形状指標: 後傾度 = 後3F - 前3F  (>0 後傾/瞬発, <0 前傾/消耗)
mm["kotai"] = mm["lap_後3F"] - mm["lap_前3F"]
# 距離・芝ダ内でz標準化(距離依存を除去)
mm["distbin"] = (mm["距離"]//200*200)
grp = mm.groupby(["芝・ダ","distbin"])["kotai"]
mm["kotai_z"] = (mm["kotai"] - grp.transform("mean")) / grp.transform("std")
mm = mm[mm["kotai_z"].notna()].copy()

mm["good"] = (mm["着順_num"] <= 3).astype(int)
mm = mm.sort_values(["馬名","日付_dt"]).reset_index(drop=True)

# 馬ごとの過去好走レースの平均ペース形状(リークなし: 当該行を含めずcumsumをshift)
# good行のkotai_zだけを累積し、当該行を除外(shift)して過去好走のみを集約
g = mm.groupby("馬名", sort=False)
mm["_gz"] = mm["kotai_z"] * mm["good"]            # 好走時のみz、それ以外0
mm["_g"] = mm["good"]
mm["_cum_sum"] = g["_gz"].cumsum()
mm["_cum_cnt"] = g["_g"].cumsum()
# shiftで当該行を除外 -> 過去のみ
mm["_ps"] = g["_cum_sum"].shift(1)
mm["_pc"] = g["_cum_cnt"].shift(1)
mm["horse_pref_z"] = mm["_ps"] / mm["_pc"]
mm["n_past_good"] = mm["_pc"].fillna(0)
mm["mismatch"] = (mm["kotai_z"] - mm["horse_pref_z"]).abs()

# 検証対象: 過去好走2回以上ある馬-レース
q = mm[mm["n_past_good"] >= 2].copy()
print(f"\n検証対象 horse-races (過去好走>=2): {len(q)}")
print(f"全体好走率(base): {q['good'].mean()*100:.1f}%   平均頭数:{q['頭数'].mean():.1f}")

# mismatch を三分位でバケット
q["bucket"] = pd.qcut(q["mismatch"], 3, labels=["適合(小)","中","不適合(大)"])
tab = q.groupby("bucket", observed=True).agg(
    n=("good","size"), 好走率=("good","mean"),
    平均人気=("人気","mean"), mismatch平均=("mismatch","mean")).reset_index()
tab["好走率"] = (tab["好走率"]*100).round(1)
tab["平均人気"] = tab["平均人気"].round(2)
tab["mismatch平均"] = tab["mismatch平均"].round(3)
print("\n=== ペース適合バケット別 好走率(複勝率) ===")
print(tab.to_string(index=False))

# 人気で強さを揃えた層内比較(1-5番人気のみ)
qp = q[q["人気"].between(1,5)].copy()
qp["bucket"] = pd.qcut(qp["mismatch"], 3, labels=["適合(小)","中","不適合(大)"])
tab2 = qp.groupby("bucket", observed=True).agg(n=("good","size"), 好走率=("good","mean")).reset_index()
tab2["好走率"]=(tab2["好走率"]*100).round(1)
print("\n=== 1-5番人気に限定(強さを揃える) ペース適合別 好走率 ===")
print(tab2.to_string(index=False))

# --- 追試A: 極端デシル(最適合 vs 最不適合) ---
q["mm_decile"] = pd.qcut(q["mismatch"], 10, labels=False)
best = q[q["mm_decile"]==0]; worst = q[q["mm_decile"]==9]
print(f"\n=== 追試A 極端デシル ===")
print(f"最適合10%: n={len(best)} 好走率={best['good'].mean()*100:.1f}% 平均人気={best['人気'].mean():.2f}")
print(f"最不適合10%: n={len(worst)} 好走率={worst['good'].mean()*100:.1f}% 平均人気={worst['人気'].mean():.2f}")

# --- 追試B: 符号アライメント(強い選好を持つ馬だけ) ---
strong = q[q["horse_pref_z"].abs() >= 0.5].copy()
aligned = strong[np.sign(strong["kotai_z"])==np.sign(strong["horse_pref_z"])]
opposed = strong[np.sign(strong["kotai_z"])!=np.sign(strong["horse_pref_z"])]
print(f"\n=== 追試B 選好強(|pref_z|>=0.5)の馬: 今回ペースが選好と同方向 vs 逆方向 ===")
print(f"同方向: n={len(aligned)} 好走率={aligned['good'].mean()*100:.1f}% 平均人気={aligned['人気'].mean():.2f}")
print(f"逆方向: n={len(opposed)} 好走率={opposed['good'].mean()*100:.1f}% 平均人気={opposed['人気'].mean():.2f}")

# --- 追試C: 専門家(過去好走のペース形状がブレない馬) ---
# 注意: この標準偏差は全期間の好走から算出(specialist集合の定義にのみ使用)。
# 将来: past-onlyに変更して選択リークを除くのが厳密。within比較(同/逆)はリークなし。
horse_good_std = mm[mm["good"]==1].groupby("馬名")["kotai_z"].std()
q = q.merge(horse_good_std.rename("horse_good_z_std"), on="馬名", how="left")
spec = q[(q["horse_good_z_std"]<=0.6) & (q["horse_pref_z"].abs()>=0.5)].copy()
if len(spec):
    spec_al = spec[np.sign(spec["kotai_z"])==np.sign(spec["horse_pref_z"])]
    spec_op = spec[np.sign(spec["kotai_z"])!=np.sign(spec["horse_pref_z"])]
    print(f"\n=== 追試C 専門家(好走ペースがブレない,std<=0.6 & 選好強): 同方向 vs 逆方向 ===")
    print(f"対象n={len(spec)}")
    print(f"同方向: n={len(spec_al)} 好走率={spec_al['good'].mean()*100:.1f}% 平均人気={spec_al['人気'].mean():.2f}")
    print(f"逆方向: n={len(spec_op)} 好走率={spec_op['good'].mean()*100:.1f}% 平均人気={spec_op['人気'].mean():.2f}")

    # 人気帯を揃えた層内比較(popularity confound除去)
    spec = spec.copy()
    spec["align"] = np.where(np.sign(spec["kotai_z"])==np.sign(spec["horse_pref_z"]),"同方向","逆方向")
    spec["pop_band"] = pd.cut(spec["人気"], [0,3,6,9,99], labels=["1-3","4-6","7-9","10+"])
    print("\n=== 追試C-2 専門家: 人気帯を揃えた 同方向 vs 逆方向 好走率 ===")
    pv = spec.pivot_table(index="pop_band", columns="align", values="good",
                          aggfunc=["mean","size"], observed=True)
    for band in ["1-3","4-6","7-9","10+"]:
        try:
            am = pv[("mean","同方向")][band]*100; an = int(pv[("size","同方向")][band])
            om = pv[("mean","逆方向")][band]*100; on = int(pv[("size","逆方向")][band])
            print(f"人気{band:>4}: 同方向 {am:5.1f}%(n={an:4d})   逆方向 {om:5.1f}%(n={on:4d})   差={am-om:+.1f}pt")
        except Exception as e:
            print(f"人気{band}: 集計不可 {e}")
