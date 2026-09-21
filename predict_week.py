#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
当週の出馬表を予測し、last_pred.parquet を生成して commit/push する。
アプリの既定表示（起動時に last_pred.parquet を復元）を常に最新週に保つためのもの。

背景:
  アプリは起動時に last_pred.parquet を読み込んで既定表示にする（app.py）。
  このファイルが古いままだと、Cloud再起動のたびに古い日付に戻り、毎回アプリ内で
  出馬表を再アップロードする羽目になる。当週分をここで作って commit しておけば、
  アプリを開くだけで最新週が出る（アプリ内アップロード不要）。

使い方:
  python predict_week.py                     # 当週用フォルダの最新日(既定2日=週末)を自動検出
  python predict_week.py 20260926 20260927   # 日付を明示
  python predict_week.py --days 2            # 最新から何日分含めるか(既定2)
  python predict_week.py --folder D:/xxx     # 出馬表フォルダを指定
  python predict_week.py --no-push           # commit/push をスキップ
  python predict_week.py --dry-run           # 予測して中身を確認するだけ（書き込みなし）

前提:
  出馬表フォルダに '<YYYYMMDD>.csv'(出馬表) と '<YYYYMMDD>前走.csv'(前走) が揃っていること。
  ローカルWindows(Targetがある環境)で実行。Cloudへは git push 経由で反映。
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
LAST_PRED = ROOT / "data" / "processed" / "last_pred.parquet"
DEFAULT_FOLDER = r"C:\Users\asari\OneDrive\競馬ツール用データ\出走馬データ(当週用)"
CHILD_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def log(m=""):
    print(m, flush=True)


def detect_dates(folder: Path, days: int) -> list[str]:
    """出馬表(<8桁>.csv、前走を除く)のうち、前走siblingがある日付を新しい順に days 件返す。"""
    dates = []
    for p in folder.glob("*.csv"):
        m = re.fullmatch(r"(\d{8})", p.stem)
        if not m:
            continue
        d = m.group(1)
        if (folder / f"{d}前走.csv").exists():
            dates.append(d)
    dates = sorted(set(dates), reverse=True)
    return dates[:days]


def predict_dates(folder: Path, dates: list[str]) -> pd.DataFrame:
    from load_shutuba_target import (load_shutuba_target, load_maesou_target,
                                      merge_maesou_into_shutuba)
    from pipeline_target import predict_both_from_df
    parts = []
    for d in sorted(dates):
        sp = folder / f"{d}.csv"
        mp = folder / f"{d}前走.csv"
        if not sp.exists():
            log(f"  ⚠️ {d}: 出馬表CSVがありません（スキップ）")
            continue
        sh = load_shutuba_target(file_bytes=sp.read_bytes(), filename=sp.name)
        if mp.exists():
            mae = load_maesou_target(file_bytes=mp.read_bytes(), filename=mp.name)
            sh = merge_maesou_into_shutuba(sh, mae)
            _mn = sh['前走着順'].notna().sum() if '前走着順' in sh.columns else 0
            log(f"  {d}: 出馬 {len(sh)}頭・前走 {_mn}件マージ")
        else:
            log(f"  {d}: 出馬 {len(sh)}頭（前走CSVなし＝信頼度が低めに出ます）")
        parts.append(predict_both_from_df(sh.copy()))
    if not parts:
        raise SystemExit("❌ 予測対象がありません。")
    return pd.concat(parts, ignore_index=True)


def health(pred: pd.DataFrame):
    """本命の予想複勝%中央値をざっくり点検（低いとデータ不足の疑い）。"""
    try:
        from pred_utils import softmax_probs as sfx
        try:
            from fuku_calibration import calibrated_fuku_prob as cfp
        except Exception:
            cfp = None
        p = pred.copy()
        # 開催は '1中1' 固定なので日付を含めないと別日の同会場Rが衝突する
        p['rk'] = p['日付'].astype(str) + '_' + p['開催'].astype(str) + '_' + p['Ｒ'].astype(str)
        p['wp'] = p.groupby('rk')['pred_score'].transform(sfx)
        p['fp'] = (cfp(p['wp']).values if cfp is not None else p['wp'] * 3)
        p['pr'] = p.groupby('rk')['pred_score'].rank(ascending=False, method='first')
        med = float((p[p['pr'] == 1]['fp'] * 100).median())
        log(f"  本命の予想複勝% 中央値: {med:.0f}%  （通常65%前後。低い時は前走CSV同梱を確認）")
    except Exception:
        pass


def run(cmd, label):
    log(f"\n▶ {label}")
    if subprocess.run(cmd, cwd=str(ROOT), env=CHILD_ENV).returncode != 0:
        raise SystemExit(f"❌ {label} 失敗")


def main():
    ap = argparse.ArgumentParser(description="当週予測を last_pred.parquet に生成・コミット")
    ap.add_argument("dates", nargs="*", help="対象日付(YYYYMMDD)。省略時は最新を自動検出")
    ap.add_argument("--folder", default=DEFAULT_FOLDER, help="出馬表フォルダ")
    ap.add_argument("--days", type=int, default=2, help="自動検出時に含める最新日数(既定2=週末)")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    folder = Path(a.folder)
    log("=== 当週予測の生成 ===")
    log(f"フォルダ: {folder}")
    dates = a.dates or detect_dates(folder, a.days)
    if not dates:
        raise SystemExit("❌ 出馬表(<8桁>.csv＋<8桁>前走.csv)が見つかりません。")
    log(f"対象日付: {sorted(dates)}\n")

    pred = predict_dates(folder, dates)
    log(f"\n予測完了: {len(pred)}頭 / {pred.groupby(['日付','開催','Ｒ']).ngroups}レース"
        f" / 日付 {sorted(pred['日付'].astype(str).unique())}")
    health(pred)

    if a.dry_run:
        log("\n(--dry-run) last_pred.parquet は書き込みませんでした。")
        return

    LAST_PRED.parent.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(LAST_PRED, index=False)
    log(f"\n保存: {LAST_PRED}")

    if a.no_push:
        log("(--no-push) commit/push はスキップしました。")
        return
    run(["git", "add", "data/processed/last_pred.parquet"], "git add")
    if not subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=str(ROOT),
                          env=CHILD_ENV, capture_output=True, text=True).stdout.strip():
        log("\n変更なし（既に最新）。コミットしません。")
        return
    msg = (f"既定表示を更新: {'・'.join(sorted(pred['日付'].astype(str).unique()))} の当週予測\n\n"
           "predict_week.py によるlast_pred更新（アプリ起動時の既定表示を最新週に）。\n"
           "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>")
    run(["git", "commit", "-m", msg], "git commit")
    run(["git", "push", "origin", "HEAD"], "git push")
    log("\n✅ 完了！アプリを開くと最新週が既定表示になります（必要ならCloudをReboot）。")


if __name__ == "__main__":
    main()
