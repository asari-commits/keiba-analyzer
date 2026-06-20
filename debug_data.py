import pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv(r'C:\Users\asari\Downloads\基本.csv', encoding='cp932', nrows=10)

print("=== 走破タイム サンプル ===")
print(df['走破タイム'].tolist())

print("\n=== 着順 サンプル ===")
print(df['着順'].tolist())

print("\n=== 種牡馬（基本2から）===")
df2 = pd.read_csv(r'C:\Users\asari\Downloads\基本2.csv', encoding='cp932', nrows=5)
print(df2['種牡馬'].tolist() if '種牡馬' in df2.columns else '列なし')

print("\n=== 生産データの種牡馬 ===")
ds = pd.read_csv(r'C:\Users\asari\Downloads\生産データ.csv', encoding='cp932', nrows=5)
print(ds['種牡馬'].tolist() if '種牡馬' in ds.columns else '列なし')
