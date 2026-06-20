"""動作確認用：最初の5,000行でfeatures.pyをテスト"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from features import build_features

df = pd.read_csv(
    Path(__file__).parent / 'data' / 'processed' / 'master.csv',
    encoding='utf-8-sig',
    nrows=5000
)
df['日付_dt'] = pd.to_datetime(df['日付_dt'], errors='coerce')
df = df.sort_values('日付_dt').reset_index(drop=True)

print(f"テストデータ: {len(df)}行\n")
feat = build_features(df, verbose=True)

print(f"\nテスト完了: {feat.shape}")
print(feat.dtypes.value_counts())
print("\n先頭3行（主要特徴量）:")
from features import FEATURE_COLS
show = [c for c in FEATURE_COLS if c in feat.columns][:15]
print(feat[show].head(3).to_string())
