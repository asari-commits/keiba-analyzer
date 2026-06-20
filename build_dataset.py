"""
Targetエクスポートの複数CSVを結合し、分析用マスターデータを作成する。
output: data/processed/master.csv
"""
import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

INPUT_DIR = r'C:\Users\asari\Downloads'
OUTPUT = r'C:\Users\asari\Downloads\Claude\keiba-analyzer\data\processed\master.csv'

print("読み込み中...")
df_kihon  = pd.read_csv(f'{INPUT_DIR}/基本.csv',       encoding='cp932', low_memory=False)
df_kihon2 = pd.read_csv(f'{INPUT_DIR}/基本2.csv',      encoding='cp932', low_memory=False)
df_time   = pd.read_csv(f'{INPUT_DIR}/タイム.csv',     encoding='cp932', low_memory=False)
df_maesen = pd.read_csv(f'{INPUT_DIR}/前走.csv',       encoding='cp932', low_memory=False)
df_seisan = pd.read_csv(f'{INPUT_DIR}/生産データ.csv', encoding='cp932', low_memory=False)

# ---------------------------------------------------------------------------
# 重複列の整理（基本をベースに他から追加列のみ取り込む）
# ---------------------------------------------------------------------------
BASE_KEY_COLS = ['日付', '開催', 'Ｒ', '馬名']

# 生産データから種牡馬・母父馬を優先的に取り込むため、
# 基本/基本2のNaN列より生産データの実データを優先する
PREFER_FROM_SEISAN = {'種牡馬', '母父馬'}

def drop_overlap(base_cols, add_df, add_name):
    """add_dfからbaseと重複する列を除き、追加列だけ返す"""
    add_df = add_df.copy()
    if '馬名S' in add_df.columns and '馬名' not in add_df.columns:
        add_df = add_df.rename(columns={'馬名S': '馬名'})
    overlap = set(add_df.columns) & set(base_cols)
    # 生産データの場合は種牡馬・母父馬を重複扱いにしない（上書きしたい）
    if add_name == '生産':
        overlap -= PREFER_FROM_SEISAN
    keep = [c for c in add_df.columns if c not in overlap or c == '馬名']
    dropped = overlap - {'馬名'}
    if dropped:
        print(f"  {add_name}: 重複除外 {sorted(dropped)}")
    return add_df[keep]

print("\n結合処理中...")
master = df_kihon.copy()
master = master.rename(columns={'馬名': '馬名'})  # 念のため

for add_df, name in [
    (df_kihon2, '基本2'),
    (df_time,   'タイム'),
    (df_maesen, '前走'),
    (df_seisan, '生産'),
]:
    extra = drop_overlap(master.columns.tolist(), add_df, name)
    # indexが揃っているので横結合
    master = pd.concat([master, extra.reset_index(drop=True)], axis=1)
    print(f"  {name} 結合後: {master.shape}")

# ---------------------------------------------------------------------------
# 基本的なクリーニング
# ---------------------------------------------------------------------------
# 重複列名を解消（同名列が複数ある場合、最後のものを残す）
master = master.loc[:, ~master.columns.duplicated(keep='last')]
print(f"重複列除去後: {master.shape[1]}列")

print("\nクリーニング中...")

# 着順を数値化（全角数字対応、除外・中止などはNaN）
def parse_chakujun(v):
    try:
        return int(str(v).translate(str.maketrans('０１２３４５６７８９', '0123456789')))
    except (ValueError, TypeError):
        return None

master['着順_num'] = master['着順'].apply(parse_chakujun)

# 走破タイムを秒に変換
# Targetの形式: 1456 → 1:45.6 → 105.6秒
def time_to_sec(t):
    try:
        s = str(int(t))  # 例: "1456"
        if len(s) == 4:   # 1:45.6
            return int(s[0]) * 60 + int(s[1:3]) + int(s[3]) / 10
        elif len(s) == 5: # 1:04.5 → "10045" → 10:04.5 はありえないので別処理
            return int(s[0]) * 60 + int(s[1:3]) + int(s[3:]) / 10
    except (ValueError, TypeError):
        return None

master['走破秒'] = master['走破タイム'].apply(time_to_sec)

# 日付をdatetime変換（260614 → 2026-06-14）
def wareki_to_date(d):
    try:
        s = str(int(d))
        if len(s) == 6:
            y = 2000 + int(s[:2])
            return pd.Timestamp(year=y, month=int(s[2:4]), day=int(s[4:6]))
    except:
        return pd.NaT
    return pd.NaT

master['日付_dt'] = master['日付'].apply(wareki_to_date)

# ---------------------------------------------------------------------------
# 出力
# ---------------------------------------------------------------------------
import pathlib
pathlib.Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
master.to_csv(OUTPUT, index=False, encoding='utf-8-sig')
print(f"\n完成: {master.shape[0]}行 x {master.shape[1]}列")
print(f"保存先: {OUTPUT}")

# ---------------------------------------------------------------------------
# サマリー表示
# ---------------------------------------------------------------------------
print("\n=== データ概要 ===")
print(f"期間: {master['日付_dt'].min().date()} 〜 {master['日付_dt'].max().date()}")
print(f"レース数: {master[['日付','開催','Ｒ']].drop_duplicates().shape[0]}")
horse_col = next(c for c in master.columns if c == '馬名')
print(f"ユニーク馬数: {master[horse_col].nunique()}")
print(f"欠損率 (主要列):")
key_cols = ['着順_num', '走破秒', '上り3F', 'PCI', '前走着順', '馬体重', '種牡馬']
key_cols = [c for c in key_cols if c in master.columns]
for c in key_cols:
    rate = float(master[c].isna().mean())
    print(f"  {c}: {rate:.1%}")
