# -*- coding: utf-8 -*-
"""
予測ログの空白日を「その日に運用していたモデル」で埋める（区分＝再現）。

固定表示（predict_week.py）をしなかった週は事前予測が残っていない。現在のモデルで
予測し直す「補完」はそのレースを学習済みで成績が甘く出るため、ここでは各レース日の
前日までに git にコミットされていた最新の lgbm_model.pkl / lgbm_model_anaba.pkl を使う。
学習に含まれていないレースなので、事前予測と同じ条件の成績になる。

  python analysis/reconstruct_pred_log.py            # 事前予測が無い日をすべて埋める
  python analysis/reconstruct_pred_log.py --dry-run  # 対象日の確認のみ

結果は data/processed/pred_log_archive.parquet（git管理）に追記する。事前予測がある
レースは上書きしない（重複排除で 事前 > 再現 > 補完 の順に残る）。
"""
import argparse
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

import result_tracker as rt  # noqa: E402


def model_commits():
    out = subprocess.run(["git", "log", "--format=%h %ct", "--", "data/processed/lgbm_model.pkl"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    return [(out[i], int(out[i + 1])) for i in range(0, len(out), 2)]   # 新しい順


def load_pkl(commit, name):
    blob = subprocess.run(["git", "show", f"{commit}:data/processed/{name}"],
                          cwd=ROOT, capture_output=True, check=True).stdout
    return pickle.loads(blob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="d_from", default="260620", help="対象開始日(YYMMDD)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    m = pd.read_parquet(rt.MASTER_PARQUET)
    m_days = sorted(d for d in m["日付"].astype(str).unique() if d >= a.d_from)
    cur = rt.load_pred_log()
    have_pre = set()
    if not cur.empty:
        ok = cur["honmei"].astype(str).str.strip() != ""
        pre = cur["src"].astype(str).replace("", rt.SRC_PRE) == rt.SRC_PRE
        have_pre = set(cur.loc[ok & pre, "date"].astype(str).str[2:])
    commits = model_commits()
    targets = {}
    for d6 in m_days:
        if d6 in have_pre:
            continue
        ts = int(pd.Timestamp("20" + d6).timestamp())
        c = next((h for h, t in commits if t < ts), None)
        if c:
            targets.setdefault(c, []).append(d6)
    days = sorted(d for v in targets.values() for d in v)
    print(f"事前予測が無い日: {len(days)}日 {days}")
    print(f"使うモデル: {len(targets)}世代")
    if a.dry_run or not days:
        return

    from features import build_features, filter_recent_years
    m["日付_dt"] = pd.to_datetime(m["日付_dt"], errors="coerce")
    feat = build_features(filter_recent_years(m), verbose=False)
    feat["d6"] = feat["日付"].astype(str).str[-6:]
    total = 0
    for c, ds in sorted(targets.items(), key=lambda kv: kv[1][0]):
        H = feat[feat["d6"].isin(ds)].copy()
        if H.empty:
            continue
        for col, fn in (("pred_score", "lgbm_model.pkl"), ("pred_score_anaba", "lgbm_model_anaba.pkl")):
            mdl = load_pkl(c, fn)
            X = H.copy()
            for f in mdl["features"]:
                if f not in X.columns:
                    X[f] = np.nan
            H[col] = mdl["model"].predict(X[mdl["features"]])
        # レース前に分からない列（人気・結果）は渡さない
        frame = H[["d6", "開催", "Ｒ", "馬名", "pred_score", "pred_score_anaba"]
                  + (["レース名"] if "レース名" in H.columns else [])].rename(columns={"d6": "日付"})
        items = rt.build_log_items(frame)
        n = rt.save_pred_logs_bulk(items, src=rt.SRC_RECON, path=rt.ARCHIVE_PATH)
        total += n
        print(f"  モデル {c}: {','.join(ds)} → {n}R")
    print(f"再現として保管庫に追加: {total}R")


if __name__ == "__main__":
    main()
