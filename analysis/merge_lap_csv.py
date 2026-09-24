# -*- coding: utf-8 -*-
"""
Targetから追加出力した「ラップタイム分析用」CSVを、既存の全期間CSVに追記マージする。

使い方:
    python analysis/merge_lap_csv.py <追加分CSVのパス> [--dry-run]

マージ方針（重要）:
    「追加分に含まれる開催日」については、既存側の行を丸ごと落として追加分で置き換える。
    レース単位の一意キーが CSV に無いため（同一日・同一開催に「未勝利」が複数存在し、
    日付+開催+レース名 では一意にならない）、日付単位で入れ替える方式にしている。
    これなら行の取りこぼしも二重登録も起きず、何度実行しても同じ結果になる。

処理:
    1. 既存CSV(cp932)と追加分を読み、列構成が一致するか検証
    2. 追加分が持つ開催日を既存から除去 → 追加分を連結
    3. 日付順に並べ替えて既存CSVを上書き（.bak を自動で残す）
    4. 続けて analysis/lap_foundation.py 等を実行すればラップ資産が再生成される
"""
import argparse
import shutil
import stat
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
BASE = Path(r"C:\Users\asari\OneDrive\競馬ツール用データ\ラップタイム分析用\data_raw\2016年以降全場ラップタイム分析用データ.csv")


def read_cp932(p: Path) -> pd.DataFrame:
    for enc in ("cp932", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(p, encoding=enc, dtype=str)
        except UnicodeDecodeError:
            continue
    raise SystemExit(f"文字コードを判定できません: {p}")


def sort_key(df: pd.DataFrame) -> pd.Series:
    """'2026. 7.19' → 20260719 の整数キー"""
    parts = df["日付"].astype(str).str.split(".", expand=True)
    return (parts[0].str.strip().astype(int) * 10000
            + parts[1].str.strip().astype(int) * 100
            + parts[2].str.strip().astype(int))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("addition", help="Targetから出力した追加分CSV")
    ap.add_argument("--base", default=str(BASE), help="既存の全期間CSV")
    ap.add_argument("--dry-run", action="store_true", help="検証のみ。書き込まない")
    a = ap.parse_args()

    base_p, add_p = Path(a.base), Path(a.addition)
    for p, label in ((base_p, "既存CSV"), (add_p, "追加分CSV")):
        if not p.exists():
            raise SystemExit(f"{label}が見つかりません: {p}")

    base_df, add_df = read_cp932(base_p), read_cp932(add_p)
    print(f"既存  : {len(base_df):,}行  {base_df['日付'].min()} 〜 {base_df['日付'].max()}")
    print(f"追加分: {len(add_df):,}行  {add_df['日付'].min()} 〜 {add_df['日付'].max()}")

    only_base = [c for c in base_df.columns if c not in add_df.columns]
    only_add = [c for c in add_df.columns if c not in base_df.columns]
    if only_base or only_add:
        print("\n!! 列構成が一致しません。Target側の出力項目セットを既存と揃えてください。")
        if only_base: print(f"   追加分に無い列 : {only_base}")
        if only_add:  print(f"   追加分だけの列 : {only_add}")
        raise SystemExit(1)
    add_df = add_df[base_df.columns]
    print(f"列構成: 一致（{len(base_df.columns)}列）")

    add_days = set(add_df["日付"].astype(str))
    overlap = base_df["日付"].astype(str).isin(add_days)
    kept = base_df[~overlap]
    merged = pd.concat([kept, add_df], ignore_index=True)
    merged = merged.assign(_k=sort_key(merged)).sort_values("_k").drop(columns="_k")

    new_days = sorted(add_days - set(base_df["日付"].astype(str)))
    print(f"\n置き換え対象: 追加分の{len(add_days)}開催日 "
          f"（うち既存にもある{len(add_days)-len(new_days)}日は入れ替え、{len(new_days)}日は新規）")
    print(f"  既存から除去: {int(overlap.sum()):,}行 / 追加分を投入: {len(add_df):,}行")
    print(f"  行数: {len(base_df):,} → {len(merged):,}  ({len(merged)-len(base_df):+,})")
    print(f"新規に増えた開催日: {new_days}")
    print(f"マージ後の最新日付: {merged['日付'].iloc[-1]}")

    if len(merged) < len(base_df):
        print("\n!! 行数が減っています。想定外なので中止します。")
        raise SystemExit(1)

    if a.dry_run:
        print("\n--dry-run のため書き込みませんでした。")
        return

    bak = base_p.with_suffix(".csv.bak")
    base_p.chmod(base_p.stat().st_mode | stat.S_IWRITE)
    shutil.copy2(base_p, bak)
    merged.to_csv(base_p, index=False, encoding="cp932")
    print(f"\n上書き保存: {base_p}\nバックアップ: {bak}")
    print("\n次に実行:\n    python analysis/lap_foundation.py")
    print("    python analysis/build_pace_profiles.py")
    print("    python analysis/build_lap_features.py")


if __name__ == "__main__":
    main()
