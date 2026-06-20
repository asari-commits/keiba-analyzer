import pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')

df_kihon  = pd.read_csv(r'C:\Users\asari\Downloads\基本.csv',       encoding='cp932')
df_kihon2 = pd.read_csv(r'C:\Users\asari\Downloads\基本2.csv',      encoding='cp932')
df_time   = pd.read_csv(r'C:\Users\asari\Downloads\タイム.csv',     encoding='cp932')
df_maesen = pd.read_csv(r'C:\Users\asari\Downloads\前走.csv',       encoding='cp932')
df_seisan = pd.read_csv(r'C:\Users\asari\Downloads\生産データ.csv', encoding='cp932')

dfs = {'基本': df_kihon, '基本2': df_kihon2, 'タイム': df_time, '前走': df_maesen, '生産': df_seisan}
key = ['日付', '開催', 'Ｒ']

for name, df in dfs.items():
    has = [c for c in key if c in df.columns]
    horse_col = '馬名' if '馬名' in df.columns else '馬名S'
    dup = df.duplicated(subset=has + ([horse_col] if horse_col in df.columns else [])).sum()
    print(f"{name}: キー={has}, 馬名列='{horse_col}', 重複行={dup}")

print()
# 行数が全て揃っているので index結合も可能か確認
print("全ファイル行数:", {n: len(d) for n, d in dfs.items()})

# ソート確認（同じ順序かチェック）
print()
print("先頭3行の日付+開催+馬名 確認:")
for name, df in dfs.items():
    horse_col = '馬名' if '馬名' in df.columns else '馬名S'
    r_col = 'Ｒ' if 'Ｒ' in df.columns else None
    cols = ['日付', '開催'] + ([r_col] if r_col else []) + [horse_col]
    cols = [c for c in cols if c in df.columns]
    print(f"  {name}: {df[cols].head(3).values.tolist()}")
