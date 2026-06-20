import pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')

files = {
    '基本':   r'C:\Users\asari\Downloads\基本.csv',
    '基本2':  r'C:\Users\asari\Downloads\基本2.csv',
    'タイム': r'C:\Users\asari\Downloads\タイム.csv',
    '前走':   r'C:\Users\asari\Downloads\前走.csv',
    '生産':   r'C:\Users\asari\Downloads\生産データ.csv',
}

for name, path in files.items():
    df = pd.read_csv(path, encoding='cp932')
    dates = df['日付'].dropna().astype(str)
    print(f"{name}: {len(df):,}行  日付範囲: {dates.min()} 〜 {dates.max()}")
