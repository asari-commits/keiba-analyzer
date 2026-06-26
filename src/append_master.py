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
    # コース内外回り（JVトラックコードと、そこから導出する内/外/直）
    'トラックコード', '内外',
]


def _jv_inout(code) -> str:
    """JVトラックコード → 内/外/直（ダート等の内外区別なしは空）。
    芝: 10=直線, 11/17=内回り, 12/18=外回り（左右で番号が変わる）。"""
    s = str(code).strip()
    if not s or s in ('nan', 'None'):
        return ''
    try:
        c = int(float(s))
    except Exception:
        return ''
    if c == 10:
        return '直'
    if c in (12, 14, 16, 18, 20, 22):
        return '外'
    if c in (11, 13, 15, 17, 19, 21):
        return '内'
    return ''   # ダート(23+)・障害等は内外区別なし


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

    # 日付_dt: master の 日付 は6桁(YYMMDD)。%y%m%d で正しくパースする。
    # （旧実装は %Y%m%d で '260620'→0026年 と誤読し、5年フィルタで新データが脱落していた）
    df['日付_dt'] = pd.to_datetime(df['日付'].astype(str).str.strip().str.zfill(6),
                                   format='%y%m%d', errors='coerce')

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

    # トラックコード: エクスポートに「トラックコード(JV)」があれば取り込む。
    # 既存の 'トラックコード' 列(Target独自 0/8等)があってもJV版を優先。
    if 'トラックコード(JV)' in df.columns:
        df['トラックコード'] = df['トラックコード(JV)'].astype(str).str.strip()
    elif 'トラックコード' not in df.columns:
        df['トラックコード'] = ''
    df['トラックコード'] = df['トラックコード'].fillna('').astype(str).str.strip().replace('nan', '')

    # 内外（JVトラックコードから導出。無ければ空＝場×距離で一意なコースとして扱う）
    df['内外'] = df['トラックコード'].map(_jv_inout)

    return df


def _read_cp932(path: Path) -> pd.DataFrame:
    for enc in ('cp932', 'utf-8-sig', 'utf-8'):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, dtype=str)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"文字コードを判定できませんでした: {path}")


def load_5csv_merged(folder: Path) -> pd.DataFrame:
    """Target の5CSVセット（基本/基本2/タイム/前走/生産データ）を1フォルダから探して
    横結合し、master形式のDataFrameを返す。レース.csv があればクラス名も取り込む。
    全ファイル同じ行数・同じ並び（Targetの同時出力）であることが前提。"""
    folder = Path(folder)
    csvs = list(folder.glob('*.csv'))

    def find(kw, exclude=None):
        cands = [p for p in csvs if kw in p.stem and (exclude is None or exclude not in p.stem)]
        if not cands:
            return None
        # 候補が複数なら、よりファイル名が長い（日付プレフィクス付き等）ものを優先
        return sorted(cands, key=lambda p: len(p.stem), reverse=True)[0]

    files = {
        'kihon2': find('基本2'),
        'kihon':  find('基本', exclude='基本2'),
        'time':   find('タイム'),
        'maesou': find('前走'),
        'seisan': find('生産'),
        'race':   find('レース'),
    }
    missing = [k for k in ('kihon', 'kihon2', 'time', 'maesou', 'seisan') if files[k] is None]
    if missing:
        raise FileNotFoundError(f"5CSVのうち見つかりません: {missing}（フォルダ: {folder}）")

    frames = {k: _read_cp932(v) for k, v in files.items() if v is not None}
    for k, v in files.items():
        if v is not None:
            print(f"  {k}: {v.name}  {len(frames[k])}行 × {len(frames[k].columns)}列")

    n = len(frames['kihon'])
    for k, fr in frames.items():
        if len(fr) != n:
            raise ValueError(f"{k}: 行数不一致 ({len(fr)} vs 基本{n})。Targetで同時出力したか確認してください")

    _keep_seisan = {'種牡馬', '母父馬', '種牡馬コード', '母父馬コード'}
    df = frames['kihon'].copy()
    for k in ['kihon2', 'time', 'maesou', 'seisan', 'race']:
        if k not in frames:
            continue
        other = frames[k]
        dup = set(df.columns) & set(other.columns)
        drop = (dup - _keep_seisan) if k == 'seisan' else dup
        other = other.drop(columns=list(drop), errors='ignore')
        df = pd.concat([df, other], axis=1)
    df = df.loc[:, ~df.columns.duplicated(keep='last')]
    df = add_computed_cols(df)
    print(f"  → 横結合: {len(df)}行 × {len(df.columns)}列")
    return df


def load_new_csv(path: Path) -> pd.DataFrame:
    """新しい Target エクスポート CSV を読み込む。
    Target は Shift-JIS(cp932) 出力なので cp932 を優先し、UTF-8 にもフォールバックする。"""
    df = None
    for enc in ('cp932', 'utf-8-sig', 'utf-8'):
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False, dtype=str)
            print(f"  読み込み: {path.name}  {len(df)}行 × {len(df.columns)}列  (encoding={enc})")
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if df is None:
        raise ValueError(f"文字コードを判定できませんでした: {path}")
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
    if len(new_csv_paths) == 2 and new_csv_paths[0] == '--folder':
        # フォルダ内の5CSVセットを横結合（基本/基本2/タイム/前走/生産[/レース]）
        new_frames.append(load_5csv_merged(Path(new_csv_paths[1])))
    else:
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

    # 日付_dt を全行 6桁(YYMMDD)から一貫フォーマット(YYYY-MM-DD)で再生成。
    # （旧行=日付のみ / 新行=時刻付き の混在で pd.to_datetime が新行をNaTにし、
    #   5年フィルタから脱落していた問題を防ぐ）
    combined['日付_dt'] = pd.to_datetime(
        combined['日付'].astype(str).str.strip().str.zfill(6), format='%y%m%d', errors='coerce'
    ).dt.strftime('%Y-%m-%d')

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
