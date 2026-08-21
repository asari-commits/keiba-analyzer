#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
週次ワンコマンド更新スクリプト。

Target からエクスポートした結果CSV1セット
  （基本 / 基本2 / タイム / 前走 / 生産データ / レース）
を取り込み、
  master追記 → 配布データ再生成 →（任意）ラップ更新 → 再学習 → git push
までを1コマンドで自動実行する。

前提: あなたのローカルWindows（Targetがある環境）で実行する。
      Streamlit Cloud には git push 経由で反映される（今までと同じ）。

使い方:
  python weekly_update.py                    # 既定フォルダ(Downloads)内の最新セットを自動検出
  python weekly_update.py 20260815-0816      # 日付レンジ(ファイル名の接頭辞)を明示
  python weekly_update.py --folder D:/export  # 別フォルダを指定
  python weekly_update.py --lap              # ラップ(lap_master_joined / pace_profiles)も更新
  python weekly_update.py --no-train         # 再学習をスキップ（データ取り込みだけ確認）
  python weekly_update.py --no-push          # commit/push をスキップ（ローカルのみ）
  python weekly_update.py --dry-run          # 検出・検証のみ。一切書き込まない

ラップ更新(--lap)の前に、Targetのラップ出力CSVを下記へ上書き保存しておくこと:
  C:\\Users\\asari\\OneDrive\\競馬ツール用データ\\ラップタイム分析用\\data_raw\\2016年以降全場ラップタイム分析用データ.csv
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
PROC = ROOT / "data" / "processed"

# 取り込むファイル種別。基本は必須(base)。他は「あれば・行数が合えば」取り込む。
KEYWORDS = ["基本2", "基本", "タイム", "前走", "生産データ", "レース", "配当"]
# 横結合の対象（基本=base。基本2は最後＝壊れやすいので任意）
MERGE_ORDER = ["タイム", "前走", "生産データ", "レース", "基本2"]
KEEP_SEISAN = {"種牡馬", "母父馬", "種牡馬コード", "母父馬コード"}

# コミット対象（結果フロー）
DEPLOY_FILES = [
    "data/processed/master.parquet",
    "data/processed/feature_store.pkl",
    "data/processed/lgbm_model.pkl",
    "data/processed/lgbm_model_anaba.pkl",
    "data/processed/fuku_calibration.pkl",
    "data/processed/accuracy_log.parquet",
]
# コミット対象（--lap 追加分）
LAP_FILES = [
    "data/processed/lap_master_joined.parquet",
    "data/processed/lap_course_profile.parquet",
    "data/processed/lap_tendency_profile.parquet",
    "analysis/pace_profiles.json",
]

CHILD_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def log(msg=""):
    print(msg, flush=True)


def read_cp932(path: Path) -> pd.DataFrame:
    for enc in ("cp932", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, dtype=str)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"文字コードを判定できませんでした: {path}")


def detect_set(folder: Path, prefix: str | None):
    """フォルダ内の結果CSVを接頭辞(=日付レンジ)ごとにグループ化し、対象セットを返す。"""
    pat = re.compile(r"^(?P<prefix>.+?)(?P<kw>基本2|基本|タイム|前走|生産データ|レース|配当)\.csv$")
    groups: dict[str, dict[str, Path]] = {}
    for p in folder.glob("*.csv"):
        m = pat.match(p.name)
        if not m:
            continue
        groups.setdefault(m.group("prefix"), {})[m.group("kw")] = p

    if not groups:
        raise SystemExit(f"❌ 結果CSVが見つかりません（フォルダ: {folder}）")

    if prefix:
        # 部分一致で許容（'20260815-0816' でも '20260815' でも拾えるように）
        cand = {k: v for k, v in groups.items() if prefix in k}
        if not cand:
            raise SystemExit(f"❌ 接頭辞 '{prefix}' に一致するセットがありません。候補: {sorted(groups)}")
        pfx = sorted(cand, key=lambda k: len(cand[k]), reverse=True)[0]
    else:
        # 基本を含むグループのうち、接頭辞の先頭8桁(日付)が最大のもの
        def keydate(k):
            mm = re.search(r"(\d{8})", k)
            return mm.group(1) if mm else "00000000"
        eligible = {k: v for k, v in groups.items() if "基本" in v}
        if not eligible:
            raise SystemExit("❌ 『基本』CSVを含むセットが見つかりません。")
        pfx = sorted(eligible, key=lambda k: (keydate(k), len(eligible[k])))[-1]

    files = groups[pfx]
    if "基本" not in files:
        raise SystemExit(f"❌ セット '{pfx}' に『基本』CSVがありません（基本は必須）。")
    return pfx, files


def build_merged(files: dict[str, Path]) -> tuple[pd.DataFrame, list[str]]:
    """基本をbaseに、行数・馬名が整合するファイルだけ横結合する。壊れたファイルはスキップ。"""
    base = read_cp932(files["基本"])
    n = len(base)
    base_name = base["馬名"].astype(str).str.strip() if "馬名" in base.columns else None
    log(f"  基本: {n}行 × {len(base.columns)}列  日付={sorted(base['日付'].astype(str).unique())}")

    df = base.copy()
    used = ["基本"]
    skipped = []
    for kw in MERGE_ORDER:
        if kw not in files:
            skipped.append(f"{kw}(なし)")
            continue
        other = read_cp932(files[kw])
        # 行数チェック
        if len(other) != n:
            skipped.append(f"{kw}(行数{len(other)}≠{n})")
            continue
        # 馬名の位置整合チェック（名前列がある場合のみ）
        nm = "馬名" if "馬名" in other.columns else ("馬名S" if "馬名S" in other.columns else None)
        if nm is not None and base_name is not None:
            align = (other[nm].astype(str).str.strip().values == base_name.values).mean()
            if align < 0.99:
                skipped.append(f"{kw}(整合{align*100:.0f}%)")
                continue
        # 重複列を落として結合（生産は種牡馬等を残す）
        dup = set(df.columns) & set(other.columns)
        drop = (dup - KEEP_SEISAN) if kw == "生産データ" else dup
        other = other.drop(columns=list(drop), errors="ignore")
        df = pd.concat([df, other], axis=1)
        used.append(kw)

    df = df.loc[:, ~df.columns.duplicated(keep="last")]
    log(f"  → 横結合: {len(df)}行 × {len(df.columns)}列")
    log(f"  取り込み: {', '.join(used)}")
    if skipped:
        log(f"  ⚠️ スキップ: {', '.join(skipped)}")
    return df, sorted(df["日付"].astype(str).unique())


def run(cmd: list[str], label: str):
    log(f"\n▶ {label}")
    r = subprocess.run(cmd, cwd=str(ROOT), env=CHILD_ENV)
    if r.returncode != 0:
        raise SystemExit(f"❌ {label} が失敗しました（exit {r.returncode}）。中断します。")


def git_output(args: list[str]) -> str:
    return subprocess.run(["git", *args], cwd=str(ROOT), env=CHILD_ENV,
                          capture_output=True, text=True, encoding="utf-8").stdout.strip()


def main():
    ap = argparse.ArgumentParser(description="週次ワンコマンド更新")
    ap.add_argument("prefix", nargs="?", default=None,
                    help="ファイル名の接頭辞(日付レンジ)。例: 20260815-0816。省略時は最新を自動検出")
    ap.add_argument("--folder", default=str(Path.home() / "Downloads"),
                    help="結果CSVのあるフォルダ（既定: ~/Downloads）")
    ap.add_argument("--lap", action="store_true", help="ラップ(lap_master_joined / pace_profiles)も更新")
    ap.add_argument("--no-train", action="store_true", help="再学習をスキップ")
    ap.add_argument("--no-push", action="store_true", help="git commit/push をスキップ")
    ap.add_argument("--dry-run", action="store_true", help="検出・検証のみ（書き込みなし）")
    args = ap.parse_args()

    folder = Path(args.folder)
    log("=== 週次更新 ===")
    log(f"フォルダ: {folder}")

    pfx, files = detect_set(folder, args.prefix)
    log(f"対象セット: '{pfx}'  （{', '.join(sorted(files))}）\n")

    log("=== 横結合・検証 ===")
    merged, dates = build_merged(files)
    # 主要列の充足チェック
    for c in ["着順", "人気", "単勝配当", "種牡馬", "前走着順", "クラス名"]:
        if c in merged.columns:
            nn = (merged[c].fillna("").astype(str).str.strip().replace("nan", "") != "").sum()
            log(f"    {c}: {nn}/{len(merged)}")
    log(f"\n追加対象日付: {dates}")

    if args.dry_run:
        log("\n(--dry-run) ここまで。書き込みは行いません。")
        return

    # 1) master 追記（マージ済みCSVを単一ファイルモードで append_master に渡す）
    tmp_csv = PROC / f"_weekly_merged_{pfx.strip('-_ ')}.csv"
    merged.to_csv(tmp_csv, encoding="utf-8-sig", index=False)
    try:
        run([sys.executable, str(SRC / "append_master.py"), str(tmp_csv)], "master追記 (append_master.py)")
    finally:
        try:
            tmp_csv.unlink()
        except OSError:
            pass

    # 2) 配布データ再生成
    run([sys.executable, str(SRC / "make_deploy_data.py")], "配布データ再生成 (make_deploy_data.py)")

    # 3) ラップ更新（任意）
    if args.lap:
        run([sys.executable, str(ROOT / "analysis" / "lap_foundation.py")], "ラップ結合 (lap_foundation.py)")
        run([sys.executable, str(ROOT / "analysis" / "build_pace_profiles.py")], "得意ペース生成 (build_pace_profiles.py)")

    # 4) 再学習
    if not args.no_train:
        run([sys.executable, str(SRC / "train.py")], "再学習 (src/train.py)")
    else:
        log("\n(--no-train) 再学習はスキップしました。")

    # 5) git commit / push
    if args.no_push:
        log("\n(--no-push) コミット/プッシュはスキップしました。手動で確認してください。")
        return

    commit_files = list(DEPLOY_FILES)
    if args.lap:
        commit_files += LAP_FILES
    existing = [f for f in commit_files if (ROOT / f).exists()]
    run(["git", "add", *existing], "git add")

    if not git_output(["diff", "--cached", "--name-only"]):
        log("\n変更がありません（既に最新）。コミットは作成しませんでした。")
        return

    date_label = "・".join(dates)
    lap_note = "＋ラップ" if args.lap else ""
    msg = (f"master更新: {date_label} 結果データを反映して再学習{lap_note}\n\n"
           f"weekly_update.py による自動更新（セット: {pfx}）。\n"
           f"Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>")
    run(["git", "commit", "-m", msg], "git commit")
    run(["git", "push", "origin", "HEAD"], "git push")
    log("\n✅ 完了！Streamlit Cloud に反映されます（必要なら Reboot）。")


if __name__ == "__main__":
    main()
