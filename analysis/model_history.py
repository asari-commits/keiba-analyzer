# -*- coding: utf-8 -*-
"""
週次再学習でモデルが何か変わったかを検証する。

git に残る各週のモデル（data/processed/lgbm_model.pkl）を取り出し、
  1) 同じ「どのモデルもまだ学習していない期間」のレースで本命（予測1位）の成績を比較
  2) 隣り合う週のモデル同士で、本命がどれだけ入れ替わったか
  3) 特徴量重要度の上位がどう動いたか
を出す。本番モデル・データは一切書き換えない。

使い方: python analysis/model_history.py
"""
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

# (commit, 表示名, 学習に含まれる最終日 YYMMDD)
SNAPSHOTS = [
    ("4aa9307", "7/26まで", "260726"),
    ("9b6eeaa", "8/9まで", "260809"),
    ("d49ab84", "8/30まで", "260830"),
    ("9ee4d05", "9/13まで(距離破損あり)", "260913"),
    ("f1e78ce", "9/13まで(修復後)", "260913"),
    ("975f8d9", "9/27まで", "260927"),
    ("9d1b7b4", "10/4まで", "261004"),
    ("HEAD", "10/12まで(最新)", "261012"),
]
EVAL_FROM = "260801"   # 特徴量を作る評価対象の開始日


def load_model(commit: str) -> dict:
    blob = subprocess.run(["git", "show", f"{commit}:data/processed/lgbm_model.pkl"],
                          cwd=ROOT, capture_output=True, check=True).stdout
    return pickle.loads(blob)


def _pay(v):
    try:
        return float(str(v).replace(",", "").strip())
    except Exception:
        return np.nan


def top1_metrics(H: pd.DataFrame, col: str, d_from: str, d_to: str) -> dict:
    w = H[(H["d6"] > d_from) & (H["d6"] <= d_to)]
    if w.empty:
        return {}
    idx = w.groupby("rk")[col].idxmax()
    t = w.loc[idx]
    n = len(t)
    win = (t["chk"] == 1)
    plc = (t["chk"] <= 3)
    return {
        "R数": n,
        "本命勝率": win.mean() * 100,
        "本命複勝率": plc.mean() * 100,
        "単回収": (t["tan"].where(win, 0).fillna(0).sum()) / (100 * n) * 100,
        "複回収": (t["fuku"].where(plc, 0).fillna(0).sum()) / (100 * n) * 100,
    }


def main():
    print("=== 特徴量構築（現在のmaster・評価期間のみ使用）===")
    m = pd.read_parquet(ROOT / "data" / "processed" / "master.parquet")
    m["日付_dt"] = pd.to_datetime(m["日付_dt"], errors="coerce")
    feat = build_features(filter_recent_years(m), verbose=False)
    feat["d6"] = feat["日付"].astype(str).str[-6:]
    H = feat[feat["d6"] >= EVAL_FROM].copy()
    H["chk"] = pd.to_numeric(H["着順_num"], errors="coerce")
    H["rk"] = H["d6"] + "_" + H["開催"].astype(str) + "_" + H["Ｒ"].astype(str)
    H = H[H.groupby("rk")["chk"].transform(lambda s: s.notna().sum()) >= 5].copy()
    H["tan"] = H["単勝配当"].map(_pay)
    H["fuku"] = H["複勝配当"].map(_pay)
    last = H["d6"].max()
    print(f"評価用レース: {H['rk'].nunique()}R（{EVAL_FROM}〜{last}）\n")

    models = {}
    for c, lbl, cut in SNAPSHOTS:
        mdl = load_model(c)
        X = H.copy()
        for f in mdl["features"]:
            if f not in X.columns:
                X[f] = np.nan
        H[f"s_{c}"] = mdl["model"].predict(X[mdl["features"]])
        models[c] = mdl

    # 1) 共通の未学習期間で比較
    windows = [("260913", last, "9/19〜（9/13以前のモデル同士）", lambda cut: cut <= "260913"),
               ("261004", last, "10/10〜（10/4以前のモデル同士）", lambda cut: cut <= "261004")]
    for d_from, d_to, title, ok in windows:
        print(f"=== 前向き検証: {title} ===")
        rows = []
        for c, lbl, cut in SNAPSHOTS:
            if not ok(cut):
                continue
            r = top1_metrics(H, f"s_{c}", d_from, d_to)
            if r:
                rows.append({"モデル": lbl, **r})
        print(pd.DataFrame(rows).round(1).to_string(index=False), "\n")

    # 1b) 各モデル自身の未学習期間（参考：期間が違うので横比較は不可）
    print("=== 参考: 各モデルの学習後の全期間 ===")
    rows = []
    for c, lbl, cut in SNAPSHOTS:
        r = top1_metrics(H, f"s_{c}", cut, last)
        if r:
            rows.append({"モデル": lbl, "期間": f"{cut}の翌日〜", **r})
    print(pd.DataFrame(rows).round(1).to_string(index=False), "\n")

    # 2) 本命の入れ替わり（8月以降の全レース）
    print("=== 本命の一致率（8月以降の全レースで、隣の週のモデルと比べて）===")
    top = {c: H.loc[H.groupby("rk")[f"s_{c}"].idxmax(), ["rk", "馬名"]].set_index("rk")["馬名"]
           for c, _, _ in SNAPSHOTS}
    for (c1, l1, _), (c2, l2, _) in zip(SNAPSHOTS, SNAPSHOTS[1:]):
        a = (top[c1] == top[c2]).mean() * 100
        print(f"  {l1:>18} → {l2:<18} 本命一致 {a:5.1f}%")
    a = (top[SNAPSHOTS[0][0]] == top["HEAD"]).mean() * 100
    print(f"  {SNAPSHOTS[0][1]:>18} → {'10/12まで(最新)':<18} 本命一致 {a:5.1f}%\n")

    # 3) 特徴量重要度の上位
    print("=== 特徴量重要度 上位10（gain順位）===")
    imp = {}
    for c, lbl, _ in [SNAPSHOTS[0], SNAPSHOTS[-2], SNAPSHOTS[-1]]:
        mm = models[c]
        s = pd.Series(mm["model"].feature_importance(importance_type="gain"), index=mm["features"])
        imp[lbl] = s.rank(ascending=False).astype(int)
    df = pd.DataFrame(imp)
    print(df.sort_values(df.columns[-1]).head(10).to_string())


if __name__ == "__main__":
    main()
