import pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')
df = pd.read_csv(r'C:\Users\asari\Downloads\Claude\keiba-analyzer\data\processed\master.csv',
                 encoding='utf-8-sig', low_memory=False, nrows=200)
pay_cols = ['単勝配当','複勝配当','馬連','馬単','３連複','３連単','着順_num','人気']
pay_cols = [c for c in pay_cols if c in df.columns]
print("配当列サンプル（着順1〜3の行）:")
print(df[df['着順_num'].isin([1,2,3])][pay_cols].head(12).to_string(index=False))
print("\n単勝配当の形式サンプル:", df['単勝配当'].dropna().head(10).tolist())
