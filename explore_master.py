import pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv(r'C:\Users\asari\Downloads\Claude\keiba-analyzer\data\processed\master.csv', encoding='utf-8-sig')

print(f"shape: {df.shape}")
print(f"\n全列名:\n{list(df.columns)}")

print("\n=== 主要列のサンプル値 ===")
show = ['着順_num','走破秒','距離','芝・ダ','馬場状態','上り3F','PCI','RPCI',
        'Ave-3F','前走着順','前走距離','前走馬場状態','馬体重','馬体重増減',
        '種牡馬','母父馬','斤量','人気','賞金','キャリア']
show = [c for c in show if c in df.columns]
for c in show:
    vals = df[c].dropna().unique()[:6]
    print(f"  {c}: {list(vals)}")

print("\n=== 着順_num 分布（上位10） ===")
print(df['着順_num'].value_counts().head(10))

print("\n=== 芝・ダ 分布 ===")
print(df['芝・ダ'].value_counts())

print("\n=== 馬場状態 分布 ===")
print(df['馬場状態'].value_counts())

print("\n=== コース区分 分布 ===")
if 'コース区分' in df.columns:
    print(df['コース区分'].value_counts())
