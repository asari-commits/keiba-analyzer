"""
デプロイ用データ生成スクリプト（データ更新のたびに実行）。

master.csv（全期間）から「直近5年の master.parquet」と「事前計算ストア
feature_store.pkl」を作る。クラウドはこの2ファイルを git 経由で読むだけで予測する
（gdriveダウンロード・変換・ストア構築をクラウドで行わない＝起動高速・低メモリ）。

使い方:
    1) 新しいレース結果を master.csv に追記（append_master.py など）
    2) python src/train.py            # 直近5年でモデル再学習（任意・データが増えた時）
    3) python src/make_deploy_data.py # 5年 master.parquet + feature_store.pkl を生成
    4) git add data/processed/master.parquet data/processed/feature_store.pkl \
              data/processed/lgbm_model.pkl data/processed/lgbm_model_anaba.pkl
       git commit -m "data: 最新レース反映" && git push
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import filter_recent_years, RECENT_YEARS  # noqa: E402
import feature_store as fs  # noqa: E402

MASTER_CSV     = Path(__file__).parent.parent / "data" / "processed" / "master.csv"
MASTER_PARQUET = Path(__file__).parent.parent / "data" / "processed" / "master.parquet"


def main():
    if not MASTER_CSV.exists():
        print(f"master.csv が見つかりません: {MASTER_CSV}")
        sys.exit(1)

    print("=== master.csv 読み込み ===")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    # 日付_dt: 既存の正しい列を優先、無ければ6桁(YYMMDD)としてパース
    if "日付_dt" in df.columns and pd.to_datetime(df["日付_dt"], errors="coerce").notna().mean() > 0.9:
        df["日付_dt"] = pd.to_datetime(df["日付_dt"], errors="coerce")
    else:
        df["日付_dt"] = pd.to_datetime(df["日付"].astype(str).str.zfill(6),
                                      format="%y%m%d", errors="coerce")
    print(f"  {len(df)}行 × {len(df.columns)}列")

    print(f"=== 直近{RECENT_YEARS}年に限定 ===")
    df5 = filter_recent_years(df).reset_index(drop=True)
    print(f"  {len(df)}行 → {len(df5)}行")

    MASTER_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df5.to_parquet(MASTER_PARQUET, index=False)
    _mb = MASTER_PARQUET.stat().st_size // 1024 // 1024
    print(f"  master.parquet 保存（{_mb}MB, {len(df5)}行）")

    print("=== 事前計算ストア構築 ===")
    store = fs.build_store(df5)
    fs.save_store(store)
    _kb = fs.STORE_PATH.stat().st_size // 1024
    print(f"  feature_store.pkl 保存（{_kb}KB, n_master={store['_meta']['n_master']}, "
          f"cutoff={store['_meta']['cutoff']}）")

    # 整合性: master.parquet の行数 == store n_master（予測時の鮮度判定に使う）
    import pyarrow.parquet as pq
    nrows = pq.read_metadata(str(MASTER_PARQUET)).num_rows
    ok = nrows == store["_meta"]["n_master"]
    print(f"=== 整合性: master行数={nrows} == store n_master={store['_meta']['n_master']} → {ok} ===")
    if not ok:
        print("⚠️ 不一致。予測時にストアが再構築されます（遅くなる）。")
    print("\n完了。git add → commit → push でクラウドに反映してください。")


if __name__ == "__main__":
    main()
