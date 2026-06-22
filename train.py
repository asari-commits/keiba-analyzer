"""
学習スクリプト。
使い方:
  python train.py          # 両モデルを学習
  python train.py --main   # 通常モデルのみ
  python train.py --anaba  # 穴馬モデルのみ
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from features import build_features, FEATURE_COLS
from model import train, feature_importance, MODEL_PATH, MODEL_ANABA_PATH, ANABA_EXCLUDE

MASTER_CSV     = Path(__file__).parent / 'data' / 'processed' / 'master.csv'
MASTER_PARQUET = Path(__file__).parent / 'data' / 'processed' / 'master.parquet'

args = set(sys.argv[1:])
run_main  = '--anaba' not in args  # デフォルト: 両方
run_anaba = '--main'  not in args

print("=== 競馬予想モデル 学習開始 ===\n")

print("データ読み込み中...")
# アプリが配布するのは master.parquet（CSVはDL時に変換・削除される）。
# parquet があればそれを優先し、無ければ master.csv にフォールバック。
if MASTER_PARQUET.exists():
    df = pd.read_parquet(MASTER_PARQUET)
    print(f"  読み込み元: {MASTER_PARQUET.name}")
elif MASTER_CSV.exists():
    df = pd.read_csv(MASTER_CSV, encoding='utf-8-sig', low_memory=False)
    print(f"  読み込み元: {MASTER_CSV.name}")
else:
    raise SystemExit("master.parquet / master.csv が見つかりません。アプリでmaster.csvをDLしてください。")
df['日付_dt'] = pd.to_datetime(df['日付_dt'], errors='coerce')
df = df.sort_values('日付_dt').reset_index(drop=True)
print(f"  {len(df)}行 x {len(df.columns)}列\n")

print("特徴量エンジニアリング中（脚質×コース適性を含む）...")
feat_df = build_features(df, verbose=True)

# ── 通常モデル ──────────────────────────────────────────────────────────
if run_main:
    print("\n\n" + "="*55)
    print("【通常モデル】全特徴量（人気・過去実績込み）")
    print("="*55)
    model_main = train(feat_df, save=True, model_path=MODEL_PATH, exclude_cols=set())

    print("\n=== 特徴量重要度 TOP20（通常モデル） ===")
    use_cols_main = [c for c in FEATURE_COLS if c in feat_df.columns]
    imp_main = feature_importance(model_main, use_cols_main)
    print(imp_main.head(20).to_string(index=False))

# ── 穴馬モデル ─────────────────────────────────────────────────────────
if run_anaba:
    print("\n\n" + "="*55)
    print("【穴馬モデル】人気・過去実績を除外 → コース・脚質重視")
    print(f"  除外特徴量: {sorted(ANABA_EXCLUDE)}")
    print("="*55)
    model_anaba = train(feat_df, save=True, model_path=MODEL_ANABA_PATH, exclude_cols=ANABA_EXCLUDE)

    print("\n=== 特徴量重要度 TOP20（穴馬モデル） ===")
    use_cols_anaba = [c for c in FEATURE_COLS if c in feat_df.columns and c not in ANABA_EXCLUDE]
    imp_anaba = feature_importance(model_anaba, use_cols_anaba)
    print(imp_anaba.head(20).to_string(index=False))

print("\n\n学習完了！")
if run_main:  print(f"  通常モデル  → {MODEL_PATH}")
if run_anaba: print(f"  穴馬モデル  → {MODEL_ANABA_PATH}")
