# -*- coding: utf-8 -*-
"""
取得した単勝オッズのキャッシュ（サーバー側ファイル）。

session_state だけに持つと、ページ再読み込み・Cloudの接続し直し・画面切替で消え、
そのたびに「オッズ取得」を押し直す必要があった。取得結果をここに保存し、
「レース予測」「全レース一覧」の両方で共通に読み出す。

保存先は data/processed/live_odds_cache.parquet（git管理外）。
Streamlit Cloud ではアプリのコンテナが生きている間（再起動・再デプロイまで）保持される。
"""
import os
import threading
from pathlib import Path

import pandas as pd

CACHE_PATH = Path(__file__).parent.parent / "data" / "processed" / "live_odds_cache.parquet"
_COLS = ['date8', 'venue', 'r', '馬番', '単勝オッズ', '人気', 'ts']
_KEEP_DAYS = 21          # 古い日付は自動で捨てる（ファイル肥大化防止）
_lock = threading.Lock()


def _norm_date8(d) -> str:
    s = ''.join(ch for ch in str(d) if ch.isdigit())
    return '20' + s if len(s) == 6 else s


def _read() -> pd.DataFrame:
    if not CACHE_PATH.exists():
        return pd.DataFrame(columns=_COLS)
    try:
        df = pd.read_parquet(CACHE_PATH)
        for c in _COLS:
            if c not in df.columns:
                df[c] = pd.NA
        return df[_COLS]
    except Exception:
        return pd.DataFrame(columns=_COLS)


def _write(df: pd.DataFrame) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix('.tmp.parquet')
    df.to_parquet(tmp, index=False)
    os.replace(tmp, CACHE_PATH)   # 途中で落ちても壊れたファイルを残さない


def save_races(date8, venue: str, races: dict, ts: str) -> None:
    """races = {r(int): DataFrame[馬番, 単勝オッズ, 人気(任意)]} を保存（同じ日付・会場・Rは置換）。"""
    d8 = _norm_date8(date8)
    parts = []
    for r, od in races.items():
        if od is None or len(od) == 0 or '馬番' not in od.columns or '単勝オッズ' not in od.columns:
            continue
        p = pd.DataFrame({
            '馬番': pd.to_numeric(od['馬番'], errors='coerce'),
            '単勝オッズ': pd.to_numeric(od['単勝オッズ'], errors='coerce'),
            '人気': pd.to_numeric(od['人気'], errors='coerce') if '人気' in od.columns else pd.NA,
        }).dropna(subset=['馬番'])
        if p.empty:
            continue
        p['date8'], p['venue'], p['r'], p['ts'] = d8, str(venue), int(r), str(ts)
        parts.append(p[_COLS])
    if not parts:
        return
    new = pd.concat(parts, ignore_index=True)
    keys = set(zip(new['date8'], new['venue'], new['r']))
    with _lock:
        cur = _read()
        if not cur.empty:
            cur = cur[~cur.apply(lambda x: (str(x['date8']), str(x['venue']), int(x['r'])) in keys, axis=1)]
        out = pd.concat([cur, new], ignore_index=True)
        # 古い日付を捨てる
        try:
            dt = pd.to_datetime(out['date8'].astype(str), format='%Y%m%d', errors='coerce')
            out = out[dt >= dt.max() - pd.Timedelta(days=_KEEP_DAYS)]
        except Exception:
            pass
        _write(out)


def load_race(date8, venue: str, r: int):
    """(DataFrame[馬番, 単勝オッズ, 人気], 取得時刻) を返す。無ければ (None, '')。"""
    df = _read()
    if df.empty:
        return None, ''
    d8 = _norm_date8(date8)
    m = df[(df['date8'].astype(str) == d8) & (df['venue'].astype(str) == str(venue))
           & (pd.to_numeric(df['r'], errors='coerce') == int(r))]
    if m.empty:
        return None, ''
    return m[['馬番', '単勝オッズ', '人気']].reset_index(drop=True), str(m['ts'].iloc[-1])


def load_date(date8) -> pd.DataFrame:
    """その日の全キャッシュ（date8, venue, r, 馬番, 単勝オッズ, 人気, ts）。"""
    df = _read()
    if df.empty:
        return df
    return df[df['date8'].astype(str) == _norm_date8(date8)].reset_index(drop=True)
