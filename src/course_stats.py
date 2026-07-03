"""
当該コース（場×芝ダ×距離）での 騎手 / 種牡馬 / 調教師 の勝率TOP を、
その日の出走メンバー（出走馬・騎手）に限って集計する。買い目カードの差し替え用。

種牡馬・調教師は予測DFに無い場合があるため master から馬名で補完する。
騎手は予測DF（show_df）の値を使う。芝ダ・距離は呼び出し側から渡す。
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

MASTER_PARQUET = Path(__file__).parent.parent / "data" / "processed" / "master.parquet"

_VENUE_FULL = {'札': '札幌', '函': '函館', '福': '福島', '新': '新潟', '東': '東京',
               '中': '中山', '名': '中京', '中京': '中京', '京': '京都', '阪': '阪神',
               '小': '小倉'}


@lru_cache(maxsize=1)
def _load_master_cols() -> pd.DataFrame:
    cols = ['日付', '開催', '芝・ダ', '距離', '着順_num', '騎手', '調教師', '種牡馬', '馬名']
    m = pd.read_parquet(MASTER_PARQUET, columns=cols)
    m['_v'] = m['開催'].astype(str).str.extract(r'\d+([^\d])')[0]
    m['_turf'] = m['芝・ダ'].astype(str).str.startswith('芝')
    m['_dist'] = pd.to_numeric(m['距離'], errors='coerce')
    m['_win'] = (pd.to_numeric(m['着順_num'], errors='coerce') == 1).astype(float)
    return m


@lru_cache(maxsize=1)
def _horse_meta() -> dict:
    """馬名 → (種牡馬, 調教師)。最新レコード優先。"""
    m = _load_master_cols()
    mm = m.sort_values('日付').dropna(subset=['馬名'])
    meta = {}
    for nm, sire, tr in zip(mm['馬名'].astype(str), mm['種牡馬'].astype(str), mm['調教師'].astype(str)):
        meta[nm] = (sire, tr)   # 後勝ち＝最新
    return meta


@lru_cache(maxsize=64)
def _course_rate_maps(venue: str, is_turf: bool, dist: int):
    """(場略号, 芝ダ, 距離) での 騎手/種牡馬/調教師 → (勝率, 騎乗数) 辞書。"""
    m = _load_master_cols()
    sub = m[(m['_v'] == venue) & (m['_turf'] == bool(is_turf)) & (m['_dist'] == dist)]
    maps = {}
    for col in ('騎手', '種牡馬', '調教師'):
        g = sub.groupby(col)['_win'].agg(['sum', 'size'])
        maps[col] = {str(k): (float(r['sum'] / r['size']) if r['size'] else 0.0, int(r['size']))
                     for k, r in g.iterrows() if str(k) not in ('', 'nan', 'None')}
    return maps


def course_top_performers(show_df: pd.DataFrame, venue: str, is_turf: bool,
                          dist: int, top_n: int = 3, min_n: int = 5) -> dict:
    """出走メンバー限定で当該コースの 騎手/種牡馬/調教師 勝率TOPを返す。

    返り値: {'course_label': str,
             '騎手'/'種牡馬'/'調教師': [{'name','rate','n','horses'}...]}
    """
    try:
        dist = int(dist)
        maps = _course_rate_maps(str(venue), bool(is_turf), dist)
    except Exception:
        return {}
    meta = _horse_meta()
    surf = '芝' if is_turf else 'ダ'
    out = {'course_label': f"{_VENUE_FULL.get(str(venue), str(venue))} {surf}{dist}m"}

    # 各馬の名義（騎手はshow_df、種牡馬/調教師はshow_df優先→masterで補完）
    def _name_for(row, col, meta_idx):
        v = str(row.get(col, '')).strip()
        if v and v not in ('', 'nan', 'None'):
            return v
        mt = meta.get(str(row.get('馬名', '')))
        return (mt[meta_idx] if mt else '')

    for col, midx in (('騎手', None), ('種牡馬', 0), ('調教師', 1)):
        rate_map = maps.get(col, {})
        if not rate_map:
            out[col] = []
            continue
        horses_by_name = {}
        for _, r in show_df.iterrows():
            nm = str(r.get(col, '')).strip() if col == '騎手' else _name_for(r, col, midx)
            if nm and nm not in ('', 'nan', 'None'):
                horses_by_name.setdefault(nm, []).append(str(r.get('馬名', '')))
        rows = []
        for nm, horses in horses_by_name.items():
            if nm in rate_map:
                rate, n = rate_map[nm]
                if n >= min_n:
                    rows.append({'name': nm, 'rate': rate, 'n': n, 'horses': horses})
        rows.sort(key=lambda x: (-x['rate'], -x['n']))
        out[col] = rows[:top_n]
    return out
