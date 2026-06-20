"""
スクレイピング → 分析 → 結果保存のパイプライン。
"""

import json
from pathlib import Path

import pandas as pd

from scraper import (
    scrape_shutuba,
    scrape_result,
    scrape_race_info,
    scrape_horse_results,
)
from analyzer import build_horse_features, score_horses, add_expected_value

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def analyze_race(race_id: str, save: bool = True) -> pd.DataFrame:
    """
    指定レースIDの出走表取得 → 馬ごとの成績収集 → スコアリング → 結果返却。
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"レース分析開始: {race_id}")
    print(f"{'='*50}")

    # 1. レース情報
    race_info = scrape_race_info(race_id)
    print(f"\nレース: {race_info.get('race_name', 'N/A')}")
    print(f"  会場: {race_info.get('venue', 'N/A')}")
    print(f"  距離: {race_info.get('distance', 'N/A')}m  コース: {race_info.get('track_type', 'N/A')}")

    # 2. 出走表
    shutuba = scrape_shutuba(race_id)
    if shutuba.empty:
        print("出走表の取得に失敗しました。")
        return pd.DataFrame()

    if save:
        shutuba.to_csv(RAW_DIR / f"{race_id}_shutuba.csv", index=False, encoding="utf-8-sig")

    # 3. 各馬の過去成績
    horse_feats = {}
    for _, row in shutuba.iterrows():
        horse_id = row.get("horse_id", "")
        horse_name = row.get("horse_name", "")
        if not horse_id:
            continue
        print(f"\n馬: {horse_name} ({horse_id})")
        try:
            hr = scrape_horse_results(horse_id, max_races=20)
            if save and not hr.empty:
                hr.to_csv(RAW_DIR / f"horse_{horse_id}.csv", index=False, encoding="utf-8-sig")
            horse_feats[horse_id] = build_horse_features(hr)
        except Exception as e:
            print(f"  スキップ: {e}")

    # 4. スコアリング
    scored = score_horses(shutuba, horse_feats)
    scored = add_expected_value(scored)

    if save:
        scored.to_csv(PROCESSED_DIR / f"{race_id}_scored.csv", index=False, encoding="utf-8-sig")
        with open(PROCESSED_DIR / f"{race_id}_race_info.json", "w", encoding="utf-8") as f:
            json.dump(race_info, f, ensure_ascii=False, indent=2)

    return scored


def print_summary(scored: pd.DataFrame) -> None:
    """スコアリング結果をターミナルに整形して表示。"""
    if scored.empty:
        print("データなし")
        return

    cols = ["score_rank", "umaban", "horse_name", "jockey", "score", "win_rate",
            "fukusho_rate", "avg_chakujun", "expected_value"]
    display_cols = [c for c in cols if c in scored.columns]

    print(f"\n{'='*70}")
    print("【予想スコア順位】")
    print(f"{'='*70}")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.3f}".format)
    print(scored[display_cols].to_string(index=False))


if __name__ == "__main__":
    import sys

    # 使い方: python pipeline.py <race_id>
    # 例:    python pipeline.py 202606050811
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <race_id>")
        print("例: python pipeline.py 202606050811")
        sys.exit(1)

    race_id = sys.argv[1]
    result = analyze_race(race_id)
    print_summary(result)
