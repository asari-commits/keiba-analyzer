# -*- coding: utf-8 -*-
"""
運用実績から改良点を探す（本番モデル・データは書き換えない）。

2つのデータで本命◎ほかの成績を条件別に分解する:
  A) 事前予測  : 予測ログ保管庫（レース前に公開した予測）× master の結果
  B) 前向き検証: 各レース日の「その時点で運用していたモデル」（git履歴の lgbm_model.pkl）
                 で予測し直したもの。学習に含まれていないレースだけ＝実運用と同条件で、
                 Aより標本が大きい。

使い方: python analysis/track_review.py [--from 260801]
"""
import argparse
import io
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from features import build_features, filter_recent_years  # noqa: E402


def _pay(v):
    try:
        s = str(v).replace(",", "").strip()
        return 0.0 if s.startswith("(") else float(s)
    except Exception:
        return np.nan


def model_commits():
    out = subprocess.run(["git", "log", "--format=%h %ct", "--", "data/processed/lgbm_model.pkl"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    return [(out[i], int(out[i + 1])) for i in range(0, len(out), 2)]


def load_pkl(commit, name):
    blob = subprocess.run(["git", "show", f"{commit}:data/processed/{name}"],
                          cwd=ROOT, capture_output=True, check=True).stdout
    return pickle.loads(blob)


def class_band(r):
    n = str(r["レース名"]) + str(r.get("クラス名", ""))
    if "新馬" in n:
        return "1新馬"
    if "未勝利" in n:
        return "2未勝利"
    if "1勝" in n or "１勝" in n or "500万" in n:
        return "3_1勝"
    if "2勝" in n or "２勝" in n or "1000万" in n:
        return "4_2勝"
    if "3勝" in n or "３勝" in n or "1600万" in n:
        return "5_3勝"
    if any(k in n for k in ["G1", "G2", "G3", "Ｇ"]):
        return "7重賞"
    return "6OP/L"


def seg_table(top, col, label=None):
    g = top.groupby(col)
    t = pd.DataFrame({
        "R数": g.size(),
        "勝率": g["win"].mean() * 100,
        "複勝率": g["plc"].mean() * 100,
        "単回収": g["tan"].sum() / g.size(),
        "複回収": g["fuku"].sum() / g.size(),
    }).round(1)
    print(f"\n--- {label or col} ---")
    print(t.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="d_from", default="260801")
    a = ap.parse_args()

    m = pd.read_parquet(ROOT / "data" / "processed" / "master.parquet")
    m["日付_dt"] = pd.to_datetime(m["日付_dt"], errors="coerce")
    feat = build_features(filter_recent_years(m), verbose=False)
    feat["d6"] = feat["日付"].astype(str).str[-6:]
    H = feat[feat["d6"] >= a.d_from].copy()
    H["chk"] = pd.to_numeric(H["着順_num"], errors="coerce")
    H["pop"] = pd.to_numeric(H["人気"], errors="coerce")
    H["rk"] = H["d6"] + "_" + H["開催"].astype(str) + "_" + H["Ｒ"].astype(str)
    H = H[H.groupby("rk")["chk"].transform(lambda s: s.notna().sum()) >= 5].copy()
    H["tanp"] = H["単勝配当"].map(_pay)
    H["fukup"] = H["複勝配当"].map(_pay)
    # クラス・頭数等は master 側から（日付×開催×R×馬名で結合。index は揃っていない）
    mm = m[["日付", "開催", "Ｒ", "馬名", "レース名", "クラス名", "芝・ダ", "距離", "頭数"]].copy()
    for df_ in (mm, H):
        df_["_k"] = (df_["日付"].astype(str).str[-6:] + "_" + df_["開催"].astype(str) + "_"
                     + pd.to_numeric(df_["Ｒ"], errors="coerce").astype("Int64").astype(str)
                     + "_" + df_["馬名"].astype(str).str.strip())
    mm = mm.drop_duplicates("_k").set_index("_k")
    for c in ["レース名", "クラス名", "芝・ダ", "距離", "頭数"]:
        H[c] = H["_k"].map(mm[c])

    # 各レース日に「その前日までにコミットされていた最新モデル」を割り当てる
    commits = model_commits()                      # 新しい順
    race_ts = pd.to_datetime("20" + H["d6"], format="%Y%m%d").astype("int64") // 10**9
    H["sN"] = np.nan
    H["sA"] = np.nan
    used = {}
    for d6 in sorted(H["d6"].unique()):
        ts = int(pd.Timestamp("20" + d6).timestamp())
        c = next((h for h, t in commits if t < ts), None)
        if c is None:
            continue
        used.setdefault(c, []).append(d6)
    for c, days in used.items():
        idx = H["d6"].isin(days)
        for nm, fn in [("sN", "lgbm_model.pkl"), ("sA", "lgbm_model_anaba.pkl")]:
            try:
                mdl = load_pkl(c, fn)
            except Exception:
                continue
            X = H.loc[idx].copy()
            for f in mdl["features"]:
                if f not in X.columns:
                    X[f] = np.nan
            H.loc[idx, nm] = mdl["model"].predict(X[mdl["features"]])
    H = H[H["sN"].notna()].copy()
    print(f"前向き検証: {H['rk'].nunique()}R（{H['d6'].min()}〜{H['d6'].max()}、モデル{len(used)}世代）")

    H["rN"] = H.groupby("rk")["sN"].rank(ascending=False, method="first")
    H["rA"] = H.groupby("rk")["sA"].rank(ascending=False, method="first")
    H["win"] = (H["chk"] == 1).astype(int)
    H["plc"] = (H["chk"] <= 3).astype(int)
    H["tan"] = np.where(H["win"] == 1, H["tanp"].fillna(0), 0)
    H["fuku"] = np.where(H["plc"] == 1, H["fukup"].fillna(0), 0)
    # 1位と2位のスコア差（自信の強さ）
    s_sorted = H.sort_values(["rk", "sN"], ascending=[True, False])
    gap = s_sorted.groupby("rk")["sN"].apply(lambda s: s.iloc[0] - s.iloc[1] if len(s) > 1 else np.nan)
    H["gap"] = H["rk"].map(gap)

    top = H[H["rN"] == 1].copy()
    top["人気帯"] = pd.cut(top["pop"], [0, 1, 2, 3, 5, 99], labels=["1人気", "2人気", "3人気", "4-5人気", "6人気~"])
    top["クラス"] = top.apply(class_band, axis=1)
    top["芝ダ"] = top["芝・ダ"].astype(str).str[:1]
    top["距離帯"] = pd.cut(pd.to_numeric(top["距離"], errors="coerce"), [0, 1400, 1800, 2200, 4000],
                        labels=["~1400", "1500-1800", "1900-2200", "2300~"])
    top["頭数帯"] = pd.cut(pd.to_numeric(top["頭数"], errors="coerce"), [0, 10, 14, 18],
                        labels=["~10頭", "11-14頭", "15頭~"])
    top["自信(差)"] = pd.qcut(top["gap"], 4, labels=["差小", "差やや小", "差やや大", "差大"])
    top["穴モデル一致"] = np.where(top["rA"] == 1, "一致", "不一致")
    top["開催場"] = top["開催"].astype(str).str.replace(r"[0-9A-C]", "", regex=True)

    fav = H[H["pop"] == 1].drop_duplicates("rk")
    print("\n=== 全体（本命◎ vs 1番人気を毎回買った場合）===")
    for nm, t in [("本命◎", top), ("1番人気", fav)]:
        print(f"  {nm:5}: {len(t)}R 勝率{t['win'].mean()*100:.1f}% 複勝率{t['plc'].mean()*100:.1f}% "
              f"単回収{t['tan'].sum()/len(t):.1f}% 複回収{t['fuku'].sum()/len(t):.1f}%")
    print(f"  本命◎が1番人気だった割合: {(top['pop']==1).mean()*100:.1f}%")

    diff = top[top["pop"] != 1]
    fav_d = fav[fav["rk"].isin(diff["rk"])]
    print(f"\n  ◎が1番人気でないレース {len(diff)}R:  ◎ 勝率{diff['win'].mean()*100:.1f}% 複勝率{diff['plc'].mean()*100:.1f}% "
          f"単回収{diff['tan'].sum()/len(diff):.1f}% 複回収{diff['fuku'].sum()/len(diff):.1f}%  ／  同レースの1番人気 "
          f"勝率{fav_d['win'].mean()*100:.1f}% 複勝率{fav_d['plc'].mean()*100:.1f}% "
          f"単回収{fav_d['tan'].sum()/len(fav_d):.1f}% 複回収{fav_d['fuku'].sum()/len(fav_d):.1f}%")

    for col in ["人気帯", "クラス", "芝ダ", "距離帯", "頭数帯", "自信(差)", "穴モデル一致", "開催場"]:
        seg_table(top, col)

    # 2〜6位の印の成績
    print("\n--- 予測順位別（全馬） ---")
    rr = H[H["rN"] <= 8]
    g = rr.groupby("rN")
    print(pd.DataFrame({"勝率": g["win"].mean() * 100, "複勝率": g["plc"].mean() * 100,
                        "単回収": g["tan"].mean(), "複回収": g["fuku"].mean(),
                        "平均人気": g["pop"].mean()}).round(1).to_string())

    # 穴モデル1位（通常モデルとは別馬）× 人気帯
    an = H[(H["rA"] == 1) & (H["rN"] != 1)].copy()
    an["人気帯"] = pd.cut(an["pop"], [0, 3, 5, 9, 99], labels=["1-3", "4-5", "6-9", "10~"])
    seg_table(an, "人気帯", "穴モデル1位（◎と別馬）の人気帯別")

    # 「人気より評価が高い馬」（通常順位 − 人気の乖離）
    H["乖離"] = H["pop"] - H["rN"]
    H["乖離帯"] = pd.cut(H["乖離"], [-99, -3, 0, 2, 4, 99], labels=["人気≫評価", "≒", "+1~2", "+3~4", "+5~"])
    seg_table(H[H["rN"] <= 3], "乖離帯", "上位3頭(◎○▲)の 人気−予測順位 別")

    out = ROOT / "analysis" / "out"
    out.mkdir(exist_ok=True)
    top.to_parquet(out / "track_review_top.parquet", index=False)
    H[["rk", "d6", "開催", "馬名", "pop", "chk", "win", "plc", "tan", "fuku", "sN", "sA", "rN", "rA",
       "頭数", "レース名"]].to_parquet(out / "track_review_all.parquet", index=False)


if __name__ == "__main__":
    main()
