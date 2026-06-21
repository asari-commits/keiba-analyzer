"""
Targetソフトの出馬表エクスポートCSV（1ファイル形式）を読み込み、
predict_both_from_df() に渡せる形式に変換する。

Targetエクスポート列:
  場所, Ｒ, レース名, 芝ダ, 距離, R印1, 枠番, 馬番, B, 馬印1-4,
  馬名, Ｃ, 性別, 年齢, 騎手, 斤量, 所属, 調教師, 本賞金, 収得賞金,
  馬主, 生産者, 種牡馬, 母名, 母父名, 毛色, 騎手誕生日(歳), 調教師誕生日(歳)
"""
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


def _infer_class_num(race_name: str) -> float:
    """レース名からクラス番号を推定（master.csvと同一基準）"""
    s = str(race_name)
    if any(x in s for x in ['G1', 'Ｇ１', '(G1)', '（G1）']): return 9.0
    if any(x in s for x in ['G2', 'Ｇ２', '(G2)', '（G2）']): return 8.0
    if any(x in s for x in ['G3', 'Ｇ３', '(G3)', '（G3）']): return 7.0
    if any(x in s for x in ['(L)', '（L）', '（Ｌ）', 'Listed', 'リステッド']): return 6.0
    if any(x in s for x in ['3勝', '1600万']): return 4.0
    if any(x in s for x in ['2勝', '1000万']): return 3.0
    if any(x in s for x in ['1勝', '500万']): return 2.0
    if '未勝利' in s: return 1.0
    if '新馬' in s: return 0.0
    if any(x in s for x in ['オープン', 'OP', '特別']): return 5.0
    return float('nan')

# 場所名 → Target風開催文字略称（parse_venue が r'\d([^\d]+)\d' で読める形式に変換）
VENUE_ABBR = {
    '東京': '東', '中山': '中', '京都': '京', '阪神': '阪',
    '中京': '名', '小倉': '小', '新潟': '新', '福島': '福',
    '函館': '函', '札幌': '札',
}


def _infer_date(filepath: str | Path) -> str:
    """
    ファイル名から日付を推定。
    例: '0621.csv' → '20260621'
        '20260621.csv' → '20260621'
    """
    stem = Path(filepath).stem
    today_year = datetime.today().strftime('%Y')

    # 8桁ならそのまま
    if re.fullmatch(r'\d{8}', stem):
        return stem
    # 4桁 MMDD → 今年を付与
    if re.fullmatch(r'\d{4}', stem):
        return today_year + stem
    # 6桁 YYYYMM or MMDDXX など → 最初の8桁を使う
    m = re.search(r'(\d{8})', stem)
    if m:
        return m.group(1)
    # 4桁以上の数字があれば今年 + MMDD
    m = re.search(r'(\d{4})', stem)
    if m:
        return today_year + m.group(1)

    return today_year + '0101'  # フォールバック


def load_shutuba_target(filepath: str | Path | None = None,
                        date_str: str = None,
                        file_bytes: bytes = None,
                        filename: str = None) -> pd.DataFrame:
    """
    Targetの出馬表エクスポートCSVを読み込み、
    master.csv互換（+ 日付/開催 付き）のDataFrameを返す。

    Parameters
    ----------
    filepath  : CSVファイルパス（ローカル読み込み用）
    date_str  : 'YYYYMMDD' 形式。省略するとファイル名から推定。
    file_bytes: バイト列（Streamlitアップローダー用）
    filename  : アップロード元のファイル名（日付推定に使用）
    """
    import io

    if file_bytes is not None:
        # Streamlit アップロード経由：バイト列を直接 pandas に渡す
        src_name = filename or 'upload.csv'
        if date_str is None:
            date_str = _infer_date(src_name)
        df = pd.read_csv(io.BytesIO(file_bytes), encoding='cp932',
                         low_memory=False, dtype=str)
    else:
        filepath = Path(filepath)
        if date_str is None:
            date_str = _infer_date(filepath)
        df = pd.read_csv(filepath, encoding='cp932', low_memory=False, dtype=str)
    df = df.fillna('')

    # ── 列位置で確実にアクセス（列名の文字化けを回避）────────────────
    # Streamlit 経由では cp932 の日本語列名が文字化けする場合があるため、
    # デバッグで確認した列インデックスを使って直接取得する。
    # Target出馬表エクスポートの固定列レイアウト:
    #  0:場所  1:Ｒ  2:レース名  3:芝ダ  4:距離
    #  6:枠番  7:馬番  8:B  13:馬名  15:性別  16:年齢
    #  17:騎手  18:斤量  20:調教師  25:種牡馬  26:母名  27:母父名
    ncols = len(df.columns)

    def _col(idx, fallback=''):
        """列インデックスで Series を取得。範囲外なら空 Series"""
        if idx < ncols:
            return df.iloc[:, idx]
        return pd.Series([fallback] * len(df), dtype=str)

    out = pd.DataFrame({
        '_venue_raw': _col(0),
        'Ｒ':         _col(1),
        'レース名':   _col(2),
        '芝・ダ':     _col(3),
        '距離':       _col(4),
        '枠番':       _col(6),
        '馬番':       _col(7),
        'B':          _col(8),
        '馬名':       _col(13),
        '性別':       _col(15),
        '年齢':       _col(16),
        '騎手':       _col(17),
        '斤量':       _col(18),
        '調教師':     _col(20),
        '種牡馬':     _col(25),
        '母名':       _col(26),
        '母父馬':     _col(27),
    })
    df = out

    # ── 日付・開催コード付与 ────────────────────────────────────────
    df['日付'] = date_str

    def make_kaisai(v):
        abbr = VENUE_ABBR.get(str(v).strip(), str(v).strip()[:1])
        return f'1{abbr}1'

    df['開催'] = df['_venue_raw'].apply(make_kaisai)
    df = df.drop(columns=['_venue_raw'])

    # ── 型変換 ───────────────────────────────────────────────────────
    df['日付_dt'] = pd.to_datetime(df['日付'], format='%Y%m%d', errors='coerce')
    df['Ｒ']      = pd.to_numeric(df['Ｒ'], errors='coerce')
    df['距離']    = pd.to_numeric(df['距離'], errors='coerce')
    df['年齢']    = pd.to_numeric(df['年齢'], errors='coerce')
    df['斤量']    = pd.to_numeric(
        df['斤量'].astype(str).str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
    df['芝・ダ']  = df['芝・ダ'].astype(str).str.strip().str[:1]
    _ZEN = str.maketrans('０１２３４５６７８９', '0123456789')
    df['枠番']    = pd.to_numeric(df['枠番'].astype(str).str.translate(_ZEN), errors='coerce')
    df['馬番']    = pd.to_numeric(df['馬番'].astype(str).str.translate(_ZEN), errors='coerce')

    # クラス_num をレース名から推定（master.csvのクラス_numと同一基準）
    df['クラス_num'] = df['レース名'].apply(_infer_class_num)

    # 結果列はすべて NaN（未来レースのため）
    for col in ['着順', '走破タイム', '走破秒', '着順_num', '着差', '単勝配当', '複勝配当',
                '馬体重', '人気', '上3F地点差']:
        df[col] = np.nan

    # 馬名が空の行（ヘッダー行・区切り行）を除外
    df = df[df['馬名'].astype(str).str.strip() != ''].copy()
    df = df.reset_index(drop=True)

    return df


def load_shutuba_targets(filepaths: list, date_strs: list = None) -> pd.DataFrame:
    """
    複数の出馬表CSVを結合して返す。
    date_strs を省略するとファイル名から推定。
    """
    frames = []
    for i, fp in enumerate(filepaths):
        d = date_strs[i] if date_strs and i < len(date_strs) else None
        frames.append(load_shutuba_target(fp, date_str=d))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_maesou_target(file_bytes: bytes = None, filename: str = None,
                        filepath=None) -> pd.DataFrame:
    """
    Target 前走エクスポートCSV を読み込み、merge_maesou_into_shutuba() でマージできる形式に変換。

    戻り値列（features.py の add_previous_race_features が読む名前に揃える）:
      _merge_key, 馬名S, 間隔, 前走着順, 前走人気, 前走着差タイム,
      前走頭数, 前距離, 前芝・ダ, 前走馬場状態, 前走斤量,
      前2角, 前3角, 前4角

    列位置（デバッグで確認済み）:
      [0]  場R         venue_abbr + R_num (e.g. '函1')
      [4]  馬番
      [9]  馬名S
      [12] 間隔
      [14] 前芝ダ       → 前芝・ダ
      [15] 前距離
      [19] 前馬場状態    → 前走馬場状態
      [21] 前頭数        → 前走頭数
      [23] 前人気        → 前走人気
      [24] 前着順 (全角) → 前走着順
      [25] 前着差        → 前走着差タイム
      [27] 前斤量        → 前走斤量
      [36] 前通過2       → 前2角
      [37] 前通過3       → 前3角
      [38] 前通過4       → 前4角
    """
    import io

    if file_bytes is not None:
        df = pd.read_csv(io.BytesIO(file_bytes), encoding='cp932', dtype=str, low_memory=False)
    else:
        df = pd.read_csv(filepath, encoding='cp932', dtype=str, low_memory=False)
    df = df.fillna('')

    ncols = len(df.columns)

    def _c(idx):
        if idx < ncols:
            return df.iloc[:, idx].astype(str).str.strip()
        return pd.Series([''] * len(df), dtype=str)

    # 場R '函1' → venue_abbr='函', r_num=1
    venue_r = _c(0)
    venue_abbr = venue_r.str[:1]
    r_num = pd.to_numeric(venue_r.str[1:], errors='coerce').fillna(0).astype(int)
    umaban = pd.to_numeric(_c(4), errors='coerce').fillna(0).astype(int)

    # マージキー: '函1_3' (venue_abbr + R_num + '_' + 馬番)
    merge_key = venue_abbr + r_num.astype(str) + '_' + umaban.astype(str)

    # 前着順: '７着' → 7
    tr = str.maketrans('０１２３４５６７８９', '0123456789')
    chaku_raw = _c(24).str.translate(tr).str.extract(r'(\d+)')[0]

    out = pd.DataFrame({
        '_merge_key':     merge_key,
        '馬名S':           _c(9),
        '間隔':            pd.to_numeric(_c(12), errors='coerce'),
        '前走着順':        pd.to_numeric(chaku_raw, errors='coerce'),
        '前走人気':        pd.to_numeric(_c(23), errors='coerce'),
        '前走着差タイム':  pd.to_numeric(_c(25), errors='coerce'),
        '前走頭数':        pd.to_numeric(_c(21), errors='coerce'),
        '前距離':          pd.to_numeric(_c(15), errors='coerce'),
        '前芝・ダ':        _c(14),
        '前走馬場状態':    _c(19),
        '前走斤量':        pd.to_numeric(_c(27), errors='coerce'),
        '前2角':           pd.to_numeric(_c(36), errors='coerce'),
        '前3角':           pd.to_numeric(_c(37), errors='coerce'),
        '前4角':           pd.to_numeric(_c(38), errors='coerce'),
    })

    out = out[out['馬名S'] != ''].reset_index(drop=True)
    return out


def merge_maesou_into_shutuba(shutuba_df: pd.DataFrame,
                               maesou_df: pd.DataFrame) -> pd.DataFrame:
    """
    出馬表DataFrameに前走データをマージ。
    既存列はNaNの場合のみ前走データで補完。

    マージキー: venue_abbr + R_num + '_' + 馬番
      shutuba 側: '1函1'→'函' + R + '_' + 馬番
      maesou  側: load_maesou_target() の _merge_key 列
    """
    df = shutuba_df.copy()

    kai_abbr = df['開催'].astype(str).str.extract(r'\d([^\d]+)\d')[0].fillna('')
    r_str   = pd.to_numeric(df['Ｒ'],   errors='coerce').fillna(0).astype(int).astype(str)
    uma_str = pd.to_numeric(df['馬番'], errors='coerce').fillna(0).astype(int).astype(str)
    df['_merge_key'] = kai_abbr + r_str + '_' + uma_str

    maesou_cols = [c for c in maesou_df.columns if c not in ('馬名S',)]

    merged = df.merge(
        maesou_df[maesou_cols],
        on='_merge_key',
        how='left',
        suffixes=('', '_mz')
    )

    # 衝突列: 元がNaNの場合のみ前走値で埋める
    data_cols = [c for c in maesou_cols if c != '_merge_key']
    for col in data_cols:
        mz = col + '_mz'
        if mz in merged.columns:
            if col in merged.columns:
                merged[col] = merged[col].where(merged[col].notna(), merged[mz])
            else:
                merged[col] = merged[mz]
            merged = merged.drop(columns=[mz])

    merged = merged.drop(columns=['_merge_key'], errors='ignore')
    return merged


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    df = load_shutuba_target(r'C:\Users\asari\Downloads\0621.csv')
    print(f'読み込み: {len(df)}頭')
    print(df[['日付', '開催', 'Ｒ', '馬名', '騎手', '斤量', '芝・ダ', '距離']].head(10).to_string())
    print('\n開催値ユニーク:', df['開催'].unique())
