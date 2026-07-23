# -*- coding: utf-8 -*-
"""
A: 隔離A/Bテスト。現行モデル(baseline)に ラップ適性フィット特徴 を足すと
   OOSの的中率・回収率が上がるかを、production を書き換えずに検証。
   - 同じ build_features / FEATURE_COLS / LGBMRanker設定 / 時系列80:20split を使用
   - baseline = FEATURE_COLS のみ
   - treatment = FEATURE_COLS + lap_* 特徴
"""
import sys
from pathlib import Path
ROOT = Path(r"C:\Users\asari\Downloads\Claude\keiba-analyzer")
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import ndcg_score
from features import build_features, FEATURE_COLS, TARGET_COL

LAP = ROOT / "data" / "processed" / "lap_features.parquet"
MASTER = ROOT / "data" / "processed" / "master.parquet"
race_key = ["日付", "開催", "Ｒ"]

LAP_COLS = ["lap_pace_hat", "lap_pref_z", "lap_pref_std_past", "lap_n_past_good",
            "lap_is_specialist", "lap_pace_fit", "lap_pace_fit_spec"]

def parse_tan(x):
    try:
        return float(str(x).replace("(", "").replace(")", "").replace(",", "").strip())
    except Exception:
        return np.nan

print("データ読み込み & 特徴量生成...")
df = pd.read_parquet(MASTER)
df["日付_dt"] = pd.to_datetime(df["日付_dt"], errors="coerce")
df = df.sort_values("日付_dt").reset_index(drop=True)
feat = build_features(df, verbose=False)

# 払戻(per-horse) を保持
feat["_tan"] = df["単勝配当"].map(parse_tan).values
feat["_fuku"] = pd.to_numeric(df["複勝配当"], errors="coerce").values
feat["_pop"] = pd.to_numeric(df["人気"], errors="coerce").values

# lap特徴を結合
lap = pd.read_parquet(LAP)
feat = feat.merge(lap, on=race_key + ["馬名"], how="left")
for c in LAP_COLS:
    feat[c] = feat[c].fillna(0)

# TARGETありに限定・時系列split(現行と同じ 80:20)
d = feat.dropna(subset=[TARGET_COL]).sort_values("日付_dt").reset_index(drop=True)
split = int(len(d) * 0.8)
tr, te = d.iloc[:split].copy(), d.iloc[split:].copy()
rk = (d["日付"].astype(str) + d["開催"].astype(str) + d["Ｒ"].astype(str))
tr_rk, te_rk = rk.iloc[:split], rk.iloc[split:]
print(f"train={len(tr)}  test={len(te)}  期間 test: {te['日付_dt'].min().date()}〜{te['日付_dt'].max().date()}")

def make_groups(rkser):
    # rankerは連続group前提。順序維持でサイズ列
    return rkser.groupby(rkser, sort=False).size().values

def fit_predict(cols):
    use = [c for c in cols if c in d.columns]
    Xtr, Xte = tr[use].astype(float), te[use].astype(float)
    ytr = tr[TARGET_COL].astype(float)
    m = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", n_estimators=500,
                       learning_rate=0.05, num_leaves=63, min_child_samples=20,
                       subsample=0.8, colsample_bytree=0.8, random_state=42,
                       n_jobs=-1, verbose=-1)
    m.fit(Xtr, ytr, group=make_groups(tr_rk))
    return m.predict(Xte), use

def evaluate(pred, tag):
    t = te.copy()
    t["pred"] = pred
    t["rk"] = te_rk.values
    win_hit=[]; fuku_top3=[]; toppick_fuku=[]; tan_roi=[]; fuku_roi=[]; ndcgs=[]
    for _, grp in t.groupby("rk"):
        if len(grp) < 3:
            continue
        ranked = grp.sort_values("pred")       # 低pred=好走予測(現行と同じ)
        top = ranked.iloc[0]
        actual = ranked["actual"].values if "actual" in ranked else ranked[TARGET_COL].values
        # NDCG: relevance = 4着以内で重み(1着最大)
        rel = np.clip(4 - grp[TARGET_COL].values, 0, None)
        order = (-grp["pred"].values)          # 高relを上位にしたい→predは低いほど良いので反転
        try:
            ndcgs.append(ndcg_score([rel], [order], k=3))
        except Exception:
            pass
        ch = top[TARGET_COL]
        win_hit.append(ch == 1.0)
        toppick_fuku.append(ch <= 3.0)
        fuku_top3.append(1.0 in ranked.head(3)[TARGET_COL].values)
        tan_roi.append(top["_tan"] if ch == 1.0 else 0.0)
        fuku_roi.append(top["_fuku"] if (ch <= 3.0 and not np.isnan(top["_fuku"])) else 0.0)
    n = len(win_hit)
    print(f"\n[{tag}] {n}レース")
    print(f"  単勝的中率(top1)   : {np.mean(win_hit)*100:5.1f}%")
    print(f"  複勝的中率(top1が3着内): {np.mean(toppick_fuku)*100:5.1f}%")
    print(f"  1着をtop3で当てた率 : {np.mean(fuku_top3)*100:5.1f}%")
    print(f"  NDCG@3             : {np.mean(ndcgs):.4f}")
    print(f"  単勝回収率(top1) : {np.nanmean(tan_roi):6.1f}%")
    print(f"  複勝回収率(top1) : {np.nanmean(fuku_roi):6.1f}%")
    return dict(win=np.mean(win_hit), fuku=np.mean(toppick_fuku),
                ndcg=np.mean(ndcgs), tan=np.nanmean(tan_roi), fuku_roi=np.nanmean(fuku_roi))

print("\n=== baseline (現行 FEATURE_COLS) 学習 ===")
pb, use_b = fit_predict(FEATURE_COLS)
print(f"使用特徴量 {len(use_b)}個")
rb = evaluate(pb, "baseline")

print("\n=== treatment (+ lap特徴) 学習 ===")
pt, use_t = fit_predict(FEATURE_COLS + LAP_COLS)
print(f"使用特徴量 {len(use_t)}個 (lap {len(LAP_COLS)}個追加)")
rt = evaluate(pt, "treatment")

print("\n=== 差分 (treatment - baseline) ===")
print(f"  単勝的中率 : {(rt['win']-rb['win'])*100:+.2f}pt")
print(f"  複勝的中率 : {(rt['fuku']-rb['fuku'])*100:+.2f}pt")
print(f"  NDCG@3     : {rt['ndcg']-rb['ndcg']:+.4f}")
print(f"  単勝回収率 : {rt['tan']-rb['tan']:+.1f}pt")
print(f"  複勝回収率 : {rt['fuku_roi']-rb['fuku_roi']:+.1f}pt")

# lap特徴の重要度
mt = lgb.LGBMRanker(objective="lambdarank", n_estimators=500, learning_rate=0.05,
                    num_leaves=63, min_child_samples=20, subsample=0.8,
                    colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
mt.fit(tr[use_t].astype(float), tr[TARGET_COL].astype(float), group=make_groups(tr_rk))
imp = pd.Series(mt.feature_importances_, index=use_t).sort_values(ascending=False)
print("\n=== lap特徴の重要度順位(全特徴中) ===")
for c in LAP_COLS:
    rank = list(imp.index).index(c) + 1
    print(f"  {c:22s} imp={imp[c]:5d}  (全{len(use_t)}中 {rank}位)")

# ========== 追加: 「危険な人気馬」フィルタとしての実用価値 ==========
print("\n" + "="*60)
print("追加検証: baseline top1 を『専門家×逆ペース(トラップ)』かで層別")
t = te.copy(); t["pred"] = pb; t["rk"] = te_rk.values
picks = []
for _, grp in t.groupby("rk"):
    if len(grp) < 3:
        continue
    top = grp.sort_values("pred").iloc[0]
    picks.append(top)
P = pd.DataFrame(picks)
P["trap"] = ((P["lap_is_specialist"] == 1) & (P["lap_pace_fit"] < 0)).astype(int)
P["win"] = (P[TARGET_COL] == 1.0)
P["fuku"] = (P[TARGET_COL] <= 3.0)
P["tan_ret"] = np.where(P["win"], P["_tan"], 0.0)
P["fuku_ret"] = np.where(P["fuku"], P["_fuku"].fillna(0), 0.0)

def show(name, sub):
    if len(sub) == 0:
        print(f"  {name}: n=0"); return
    print(f"  {name:26s}: n={len(sub):4d}  複勝率={sub['fuku'].mean()*100:5.1f}%  "
          f"単勝率={sub['win'].mean()*100:5.1f}%  "
          f"単回収={np.nanmean(sub['tan_ret']):6.1f}%  複回収={np.nanmean(sub['fuku_ret']):6.1f}%  "
          f"平均人気={sub['_pop'].mean():.2f}")

show("全top1", P)
show("トラップtop1(専門家×逆)", P[P["trap"] == 1])
show("非トラップtop1", P[P["trap"] == 0])
print(f"  → トラップは全体の {P['trap'].mean()*100:.1f}% ({int(P['trap'].sum())}レース)")
# 人気帯を揃えて(1-5番人気のtop1のみ)
pop = P[P["_pop"].between(1, 5)]
print("  [1-5番人気のtop1に限定]")
show("  トラップ", pop[pop["trap"] == 1])
show("  非トラップ", pop[pop["trap"] == 0])
