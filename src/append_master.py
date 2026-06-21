"""
Target の基本データエクスポート CSV を既存の master.csv に差分追加するスクリプト。

使い方:
    python src/append_master.py 新データ.csv [新データ2.csv ...]

例:
    python src/append_master.py "C:/Users/asari/Downloads/260620.csv" "C:/Users/asari/Downloads/260621.csv"

手順:
    1. Target → エクスポート → 基本データ で 6/20・6/21 分のみ出力
    2. 上記コマンドを実行
    3. Google Drive に master.csv を再アップロード（または master.parquet を削除）
"""
import sys
import re
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

MASTER_CSV = Path(__file__).parent.parent / "data" / "processed" / "master.csv"
MASTER_PARQUET = Path(__file__).parent.parent / "data" / "processed" / "master.parquet"

# master.csv が持つ完全列リスト（この順序を維持する）
MASTER_COLS = [
    'Ｍ', '日付', '開催', 'Ｒ', 'レース名', 'Ｃ', '性別', '年齢', '騎手', '斤量', '頭数',
    '馬番', '馬印', '馬印2', '馬印3', '馬印4', 'レース印１', '人気', '着順', '芝・ダ', '距離',
    'コース区分', '馬場状態', '賞金', '多頭出し', '所属', '調教師', '走破タイム', '着差',
    '2角', '3角', '4角', '上り3F', 'PCI', '好走', 'PCI3', 'RPCI', '上3F地点差',
    '馬体重', '馬体重増減', 'ブリンカー', '単勝配当', '複勝配当', '枠連', '馬連', '馬単',
    '３連複', '３連単', 'データ順番号', '曜日', '重量種別', '限定', 'Ave-3F', '平均速度',
    '-3F平均速度', '上り3F平均速度', '間隔', '前走開催', '前走レース名', '前芝・ダ', '前距離',
    '前走馬場状態', '前走人気', '前走着順', '前2角', '前3角', '前4角', '前走上り3F',
    '替', '前騎手', '前走斤量', '前走頭数', '前走馬番', '前走走破タイム', '前走着差タイム',
    '前走馬印', '前走馬印2', '前走馬印3', '前走馬印4', '前走レース印１', '前走日付',
    '馬名', 'キャリア', '種牡馬', '母父馬', '馬主(最新/仮想)', '生産者', '毛色', '馬記号',
    '生年月日', '市場取引価格(万/最終)', '取引市場(最終)', '産地',
    # 以下は計算列（既存 or 本スクリプトで付与）
    '着順_num', '走破秒', '日付_dt', 'クラス名', 'クラス_num', '天気',
]


def _to_num(s: pd.Series) -> pd.Series:
    tr = str.maketrans('０１２３４５６７８９', '0123456789')
    return pd.to_numeric(
        s.astype(str).str.translate(tr).str.extract(r'(\d+)')[0],
        errors='coerce'
    )


def _raw_time_to_sec(s: pd.Series) -> pd.Series:
    def _conv(v):
        try:
            t = str(int(float(v)))
            if len(t) == 4:
                return int(t[0]) * 60 + int(t[1:3]) + int(t[3]) / 10
            elif len(t) == 5:
                return int(t[0]) * 60 + int(t[1:3]) + int(t[3:]) / 10
        except Exception:
            return np.nan
        return np.nan
    return s.apply(_conv)


_CLASS_PATTERNS = [
    (9,  re.compile(r'G[Ｇ]?1|Ｇ１')),
    (8,  re.compile(r'G[Ｇ]?2|Ｇ２')),
    (7,  re.compile(r'G[Ｇ]?3|Ｇ３')),
    (6,  re.compile(r'\(L\)|（L）|（Ｌ）|Listed|リステッド')),
    (4,  re.compile(r'3勝|1600万')),
    (3,  re.compile(r'2勝|1000万')),
    (2,  re.compile(r'1勝|500万')),
    (1,  re.compile(r'未勝利')),
    (0,  re.compile(r'新馬')),
    (5,  re.compile(r'オープン|特別')),
]

def _infer_class_num(race_name: str) -> float:
    s = str(race_name)
    for num, pat in _CLASS_PATTERNS:
        if pat.search(s):
            return float(num)
    return float('nan')


def add_computed_cols(df: pd.DataFrame) -> pd.DataFrame:
    """master.csv と同じ計算列を付与する"""
    # 着順_num
    if '着順_num' not in df.columns:
        df['着順_num'] = _to_num(df['着順']) if '着順' in df.columns else np.nan

    # 走破秒
    if '走破秒' not in df.columns:
        if '走破タイム' in df.columns:
            df['走破秒'] = _raw_time_to_sec(pd.to_numeric(df['走破タイム'], errors='coerce'))
        else:
            df['走破秒'] = np.nan

    # 日付_dt
    if '日付_dt' not in df.columns:
        df['日付_dt'] = pd.to_datetime(df['日付'].astype(str).str.zfill(8),
                                        format='%Y%m%d', errors='coerce')
        # 6桁形式 (YYMMDD) の場合
        if df['日付_dt'].isna().all():
            df['日付_dt'] = pd.to_datetime(
                '20' + df['日付'].astype(str).str.zfill(6),
                format='%Y%m%d', errors='coerce'
            )

    # クラス名（Target から取得できなければ空欄）
    if 'クラス名' not in df.columns:
        df['クラス名'] = np.nan

    # クラス_num（レース名から推定）
    if 'クラス_num' not in df.columns:
        if 'レース名' in df.columns:
            df['クラス_num'] = df['レース名'].apply(_infer_class_num)
        else:
            df['クラス_num'] = np.nan

    # 天気（Target にあればそのまま、なければ空欄）
    if '天気' not in df.columns:
        df['天気'] = np.nan

    return df


def load_new_csv(path: Path) -> pd.DataFrame:
    """新しい Target エクスポート CSV を読み込む"""
    df = pd.read_csv(path, encoding='utf-8-sig', low_memory=False, dtype=str)
    print(f"  読み込み: {path.name}  {len(df)}行 × {len(df.columns)}列")
    df = add_computed_cols(df)
    return df


def main(new_csv_paths: list[str]) -> None:
    if not MASTER_CSV.exists():
        print(f"master.csv が見つかりません: {MASTER_CSV}")
        sys.exit(1)

    print(f"=== 既存 master.csv 読み込み ===")
    master = pd.read_csv(MASTER_CSV, encoding='utf-8-sig', low_memory=False, dtype=str)
    print(f"  {len(master)}行 × {len(master.columns)}列")
    print(f"  最新日付: {master['日付'].max()}")

    # 新データを読み込み
    print(f"\n=== 新データ読み込み ===")
    new_frames = []
    for p in new_csv_paths:
        path = Path(p)
        if not path.exists():
            print(f"  ⚠️ ファイルが見つかりません: {p}")
            continue
        df = load_new_csv(path)
        new_frames.append(df)

    if not new_frames:
        print("追加するデータがありません。")
        sys.exit(1)

    new_df = pd.concat(new_frames, ignore_index=True)
    new_dates = new_df['日付'].unique()
    print(f"  追加対象日付: {sorted(new_dates)}")

    # 重複排除: 既存 master から同じ日付の行を削除してから追加
    before = len(master)
    master_filtered = master[~master['日付'].astype(str).isin([str(d) for d in new_dates])]
    removed = before - len(master_filtered)
    if removed > 0:
        print(f"  ※ 既存データの重複 {removed}行 を上書き対象として除去")

    # 列を MASTER_COLS の順序に揃える（不足列は NaN 補完）
    for col in MASTER_COLS:
        if col not in new_df.columns:
            new_df[col] = np.nan
    new_df = new_df[[c for c in MASTER_COLS if c in new_df.columns]]

    for col in MASTER_COLS:
        if col not in master_filtered.columns:
            master_filtered[col] = np.nan
    master_filtered = master_filtered[[c for c in MASTER_COLS if c in master_filtered.columns]]

    # 日付順に結合
    combined = pd.concat([master_filtered, new_df], ignore_index=True)
    # 6桁日付を数値ソートできるように整数変換して並べ替え
    combined['_date_sort'] = pd.to_numeric(combined['日付'], errors='coerce')
    combined = combined.sort_values('_date_sort').drop(columns=['_date_sort'])
    combined = combined.reset_index(drop=True)

    # 保存
    print(f"\n=== 保存 ===")
    combined.to_csv(MASTER_CSV, encoding='utf-8-sig', index=False)
    print(f"  master.csv: {len(combined)}行 × {len(combined.columns)}列")
    print(f"  最新日付: {combined['日付'].max()}")
    print(f"  追加行数: {len(combined) - len(master_filtered)}行")

    # master.parquet が古くなるので削除（次回起動時に再変換される）
    if MASTER_PARQUET.exists():
        MASTER_PARQUET.unlink()
        print(f"  master.parquet を削除（次回 Streamlit 起動時に自動再変換されます）")

    print("\n✅ 完了！Google Drive に master.csv を再アップロードしてください。")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("使い方: python src/append_master.py 新データ.csv [新データ2.csv ...]")
        sys.exit(1)
    main(sys.argv[1:])
