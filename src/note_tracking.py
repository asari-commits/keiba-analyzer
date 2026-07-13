# -*- coding: utf-8 -*-
"""馬ノートの成績トラッキング。

各メモについて、その馬の『メモ対象レース日より後の最初のレース（＝次走）』の結果を
masterから紐付け、評価/タグ/狙い度ごとに複勝率・勝率・回収率を集計する。
→ 人間の目（メモ）に本当にエッジがあるかを数字で検証する。
"""
from pathlib import Path

import numpy as np
import pandas as pd

import race_notes as rn

MASTER_PARQUET = Path(__file__).parent.parent / "data" / "processed" / "master.parquet"


def _parse_pay(v) -> float:
    """Target形式の配当（100円あたりの払戻金額）を float に。無効値は0。"""
    try:
        s = str(v).strip()
        if s in ('', 'nan', 'None', 'NaN') or s.startswith('('):
            return 0.0
        return float(s)
    except Exception:
        return 0.0


def _next_run_outcomes() -> pd.DataFrame:
    """1行=1メモ（次走が確定＝消化済みのもののみ）。列: 評価/狙い度/タグ/着/人気/複勝/勝/単配/複配。"""
    notes = rn.load_notes()
    if notes.empty:
        return pd.DataFrame()
    names = list({rn.normalize_name(n) for n in notes['馬名']})
    cols = ['馬名', '日付', '日付_dt', '着順_num', '人気', '単勝配当', '複勝配当']
    try:
        hist = pd.read_parquet(MASTER_PARQUET, columns=cols, filters=[('馬名', 'in', names)])
        if hist.empty:
            raise ValueError('empty')
    except Exception:
        hist = pd.read_parquet(MASTER_PARQUET, columns=cols)
        hist = hist[hist['馬名'].astype(str).map(rn.normalize_name).isin(set(names))]
    hist['_nm'] = hist['馬名'].astype(str).map(rn.normalize_name)
    hist['日付_dt'] = pd.to_datetime(hist['日付_dt'], errors='coerce')

    rows = []
    for _, nt in notes.iterrows():
        nm = rn.normalize_name(nt['馬名'])
        try:
            nd = pd.to_datetime(str(nt['日付']), format='%Y%m%d')
        except Exception:
            continue
        hh = hist[(hist['_nm'] == nm) & (hist['日付_dt'] > nd)].sort_values('日付_dt')
        if hh.empty:
            continue  # まだ次走を迎えていない（有効）
        nr = hh.iloc[0]
        chk = pd.to_numeric(nr['着順_num'], errors='coerce')
        if pd.isna(chk):
            continue  # 中止・除外等で着順なし
        pop = pd.to_numeric(nr['人気'], errors='coerce')
        win = bool(chk == 1)
        fuku = bool(chk <= 3)
        rows.append({
            '評価': str(nt.get('評価', '中立') or '中立'),
            '狙い度': int(pd.to_numeric(nt.get('狙い度', 2), errors='coerce') or 2),
            'タグ': str(nt.get('タグ', '') or ''),
            '着': int(chk), '人気': float(pop) if pd.notna(pop) else np.nan,
            '複勝': fuku, '勝': win,
            '単配': _parse_pay(nr['単勝配当']) if win else 0.0,
            '複配': _parse_pay(nr['複勝配当']) if fuku else 0.0,
        })
    return pd.DataFrame(rows)


def _agg(df: pd.DataFrame) -> dict:
    n = len(df)
    if n == 0:
        return {'n': 0}
    return {
        'n': n,
        '複勝率': round(df['複勝'].mean() * 100, 1),
        '勝率': round(df['勝'].mean() * 100, 1),
        '平均人気': round(df['人気'].mean(), 1),
        '単回収率': round(df['単配'].sum() / (n * 100) * 100, 1),
        '複回収率': round(df['複配'].sum() / (n * 100) * 100, 1),
    }


def evaluate() -> dict:
    """トラッキング結果を返す。overall/by_eval/by_aim/by_tag（DataFrame）と raw_n。"""
    df = _next_run_outcomes()
    if df.empty:
        return {'overall': {'n': 0}, 'by_eval': pd.DataFrame(),
                'by_aim': pd.DataFrame(), 'by_tag': pd.DataFrame(), 'raw_n': 0}

    be = [{'評価': ev, **_agg(g)} for ev, g in df.groupby('評価')]
    ba = [{'狙い度': f'★{aim}', **_agg(g)} for aim, g in df.groupby('狙い度')]

    _t = df.assign(_tag=df['タグ'].astype(str).str.split('・')).explode('_tag')
    _t['_tag'] = _t['_tag'].astype(str).str.strip()
    _t = _t[_t['_tag'] != '']
    bt = [{'タグ': tg, **_agg(g)} for tg, g in _t.groupby('_tag')]

    return {
        'overall': _agg(df),
        'by_eval': pd.DataFrame(be),
        'by_aim': pd.DataFrame(ba).sort_values('狙い度') if ba else pd.DataFrame(),
        'by_tag': pd.DataFrame(bt).sort_values('n', ascending=False) if bt else pd.DataFrame(),
        'raw_n': len(df),
    }
