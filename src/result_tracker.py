"""
予測ログの保存・結果の登録・回収率計算を行うモジュール。

ファイル:
  data/processed/pred_log.parquet    : 予測時の印・買い目ログ
  data/processed/result_log.parquet  : 確定後の着順・払戻ログ
"""
from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import numpy as np

PRED_LOG_PATH   = Path(__file__).parent.parent / "data" / "processed" / "pred_log.parquet"
RESULT_LOG_PATH = Path(__file__).parent.parent / "data" / "processed" / "result_log.parquet"

PRED_COLS = [
    'race_id', 'date', 'venue', 'r_num', 'race_name',
    'honmei', 'taiko', 'tansho', 'renshita', 'myomi',
    'baren_tickets', 'sanrenpuku_tickets',
]
RESULT_COLS = [
    'race_id',
    'chaku1', 'chaku2', 'chaku3',
    'tan_pay', 'fuku1_pay', 'fuku2_pay', 'fuku3_pay',
    'baren_pay', 'sanrenpuku_pay',
]

# ── 馬名正規化（空白・全半角ゆれを除去） ────────────────────────────────

def _norm(name: str) -> str:
    """馬名の前後空白を除去して比較用に正規化する。"""
    return str(name).strip()


# ── 買い目のシリアライズ（区切り文字をJSONで安全に保存） ────────────────

def _encode_tickets(tickets: list[tuple | list]) -> str:
    """馬名リストのリストをJSON文字列に変換。馬名内のハイフンに影響されない。"""
    return json.dumps([list(t) for t in tickets], ensure_ascii=False)


def _decode_tickets(s: str) -> list[list[str]]:
    """JSON文字列から馬名リストのリストに復元。"""
    if not s or str(s) in ('', 'nan', 'None'):
        return []
    try:
        return json.loads(s)
    except Exception:
        return []


# ── 予測ログ ──────────────────────────────────────────────────────────────

# 競馬場名の略称→正式名（保存時に統一し、ROIの場別集計で表記ゆれを防ぐ）
_VENUE_FULL = {'札': '札幌', '函': '函館', '福': '福島', '新': '新潟', '東': '東京',
               '中': '中山', '名': '中京', '京': '京都', '阪': '阪神', '小': '小倉'}


def normalize_venue(v) -> str:
    """略称(函)→正式名(函館)に統一。既に正式名ならそのまま。"""
    return _VENUE_FULL.get(str(v).strip(), str(v).strip())


def save_pred_log(race_id: str, date: str, venue: str, r_num: int,
                  race_name: str, show_df: pd.DataFrame,
                  buy_tickets: dict) -> None:
    """予測時の印・買い目をログに保存する（既存は上書き）。"""
    if '_mark' not in show_df.columns:
        return

    marks = show_df.set_index('馬名')['_mark'].to_dict()
    honmei   = next((n for n, m in marks.items() if m == '◎'), '')
    taiko    = next((n for n, m in marks.items() if m == '○'), '')
    tansho   = next((n for n, m in marks.items() if m == '▲'), '')
    renshita = ','.join(n for n, m in marks.items() if m == '△')
    myomi    = next((n for n, m in marks.items() if m == '★'), '')

    # 馬連: [[馬名1, 馬名2], ...] をJSONで保存
    baren_list = [
        [b['馬名1'], b['馬名2']]
        for b in buy_tickets.get('馬連', [])
    ]
    # 三連複: [[馬名A, 馬名B, 馬名C], ...] をJSONで保存
    s3_list = [
        list(combo)
        for combo in buy_tickets.get('三連複_fmtn', {}).get('組み合わせ', [])
    ]

    row = {
        'race_id':             race_id,
        'date':                date,
        'venue':               normalize_venue(venue),
        'r_num':               r_num,
        'race_name':           race_name,
        'honmei':              honmei,
        'taiko':               taiko,
        'tansho':              tansho,
        'renshita':            renshita,
        'myomi':               myomi,
        'baren_tickets':       _encode_tickets(baren_list),
        'sanrenpuku_tickets':  _encode_tickets(s3_list),
    }

    df = load_pred_log()
    df = df[df['race_id'] != race_id]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    PRED_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PRED_LOG_PATH, index=False)


def load_pred_log() -> pd.DataFrame:
    if PRED_LOG_PATH.exists():
        return pd.read_parquet(PRED_LOG_PATH)
    return pd.DataFrame(columns=PRED_COLS)


# ── 結果ログ ──────────────────────────────────────────────────────────────

def save_result(race_id: str, chaku1: str, chaku2: str, chaku3: str,
                tan_pay: int | None, fuku1_pay: int | None,
                fuku2_pay: int | None, fuku3_pay: int | None,
                baren_pay: int | None, sanrenpuku_pay: int | None) -> None:
    """レース結果・払戻を保存する（既存は上書き）。"""
    row = {
        'race_id':        race_id,
        'chaku1':         _norm(chaku1),
        'chaku2':         _norm(chaku2),
        'chaku3':         _norm(chaku3),
        'tan_pay':        tan_pay,
        'fuku1_pay':      fuku1_pay,
        'fuku2_pay':      fuku2_pay,
        'fuku3_pay':      fuku3_pay,
        'baren_pay':      baren_pay,
        'sanrenpuku_pay': sanrenpuku_pay,
    }
    df = load_result_log()
    df = df[df['race_id'] != race_id]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    RESULT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RESULT_LOG_PATH, index=False)


def load_result_log() -> pd.DataFrame:
    if RESULT_LOG_PATH.exists():
        return pd.read_parquet(RESULT_LOG_PATH)
    return pd.DataFrame(columns=RESULT_COLS)


# ── 回収率計算 ────────────────────────────────────────────────────────────

BET_UNIT = 100  # 1点あたり100円


def calc_roi(pred_log: pd.DataFrame, result_log: pd.DataFrame) -> dict:
    """
    予測ログ × 結果ログから券種別回収率を計算する。
    """
    merged = pred_log.merge(result_log, on='race_id', how='inner')

    if merged.empty:
        empty = pd.DataFrame(columns=['券種', '的中', '外れ', '投資額', '回収額', '回収率'])
        return {'summary': empty, 'detail': pd.DataFrame()}

    rows_tan   = []
    rows_fuku  = []
    rows_baren = []
    rows_3fuku = []
    detail_rows = []

    for _, r in merged.iterrows():
        race_id = r['race_id']
        d = str(r.get('date', ''))
        label = f"{d[:4]}/{d[4:6]}/{d[6:]} {r.get('venue','')} {r.get('r_num','')}R {r.get('race_name','')}"

        # 馬名を正規化して比較
        honmei = _norm(r.get('honmei', ''))
        chaku1 = _norm(r.get('chaku1', ''))
        chaku2 = _norm(r.get('chaku2', ''))
        chaku3 = _norm(r.get('chaku3', ''))
        top3   = {chaku1, chaku2, chaku3} - {''}

        tan_pay   = _int(r.get('tan_pay'))
        fuku_pays = [_int(r.get('fuku1_pay')), _int(r.get('fuku2_pay')), _int(r.get('fuku3_pay'))]
        baren_pay = _int(r.get('baren_pay'))
        s3_pay    = _int(r.get('sanrenpuku_pay'))

        # ── 単勝: ◎が1着 ──────────────────────────────────────────────
        tan_hit  = bool(honmei and honmei == chaku1)
        tan_recv = (tan_pay or 0) if tan_hit else 0
        rows_tan.append({'race_id': race_id, 'label': label,
                         'bet': BET_UNIT, 'recv': tan_recv, 'hit': tan_hit})

        # ── 複勝: ◎が3着以内 ──────────────────────────────────────────
        fuku_hit = bool(honmei and honmei in top3)
        if fuku_hit:
            chaku_list = [chaku1, chaku2, chaku3]
            idx = chaku_list.index(honmei) if honmei in chaku_list else -1
            fuku_recv = (fuku_pays[idx] or 0) if idx >= 0 else 0
        else:
            fuku_recv = 0
        rows_fuku.append({'race_id': race_id, 'label': label,
                          'bet': BET_UNIT, 'recv': fuku_recv, 'hit': fuku_hit})

        # ── 馬連: JSONで保存した各組み合わせを判定 ────────────────────
        baren_list = _decode_tickets(r.get('baren_tickets', ''))
        baren_bet  = len(baren_list) * BET_UNIT
        baren_recv = 0
        baren_hit  = False
        top2 = {chaku1, chaku2} - {''}
        for pair in baren_list:
            if len(pair) == 2:
                pair_set = {_norm(pair[0]), _norm(pair[1])}
                if pair_set == top2 and baren_pay:
                    baren_recv += baren_pay
                    baren_hit = True
        rows_baren.append({'race_id': race_id, 'label': label,
                           'bet': baren_bet or BET_UNIT, 'recv': baren_recv, 'hit': baren_hit})

        # ── 三連複: JSONで保存した各組み合わせを判定 ──────────────────
        s3_list = _decode_tickets(r.get('sanrenpuku_tickets', ''))
        s3_bet  = len(s3_list) * BET_UNIT
        s3_recv = 0
        s3_hit  = False
        for combo in s3_list:
            if len(combo) == 3:
                combo_set = {_norm(h) for h in combo}
                if combo_set == top3 and s3_pay:
                    s3_recv += s3_pay
                    s3_hit = True
        rows_3fuku.append({'race_id': race_id, 'label': label,
                           'bet': s3_bet or BET_UNIT, 'recv': s3_recv, 'hit': s3_hit})

        detail_rows.append({
            'レース':     label,
            '本命':       honmei,
            '1着':        chaku1, '2着': chaku2, '3着': chaku3,
            '単勝':       '◯' if tan_hit else '×',
            '複勝':       '◯' if fuku_hit else '×',
            '馬連':       '◯' if baren_hit else '×',
            '三連複':     '◯' if s3_hit else '×',
            '単勝払戻':   tan_pay,
            '複勝払戻':   fuku_pays[0] if fuku_hit else None,
            '馬連払戻':   baren_pay if baren_hit else None,
            '三連複払戻': s3_pay if s3_hit else None,
        })

    def _summary_row(name, rows):
        if not rows:
            return None
        df = pd.DataFrame(rows)
        total_bet  = df['bet'].sum()
        total_recv = df['recv'].sum()
        hits       = int(df['hit'].sum())
        misses     = len(df) - hits
        roi        = total_recv / total_bet * 100 if total_bet > 0 else 0
        return {
            '券種': name, '的中': hits, '外れ': misses,
            '投資額': int(total_bet), '回収額': int(total_recv),
            '回収率': round(roi, 1),
        }

    summary = pd.DataFrame([r for r in [
        _summary_row('本命◎ 単勝', rows_tan),
        _summary_row('本命◎ 複勝', rows_fuku),
        _summary_row('馬連',       rows_baren),
        _summary_row('三連複',     rows_3fuku),
    ] if r])
    detail = pd.DataFrame(detail_rows)

    return {'summary': summary, 'detail': detail}


def calc_daily_roi(pred_log: pd.DataFrame, result_log: pd.DataFrame) -> pd.DataFrame:
    """日別の回収率を返す（1行=1開催日、列=券種ごとの回収率と的中数）。"""
    if pred_log is None or pred_log.empty or result_log is None or result_log.empty:
        return pd.DataFrame()
    rids = set(result_log['race_id'].astype(str))
    out = []
    for d, g in pred_log.groupby(pred_log['date'].astype(str)):
        s = calc_roi(g, result_log)['summary']
        if s.empty:
            continue
        n_races = int(g['race_id'].astype(str).isin(rids).sum())
        row = {'日付': f"{d[:4]}/{d[4:6]}/{d[6:]}", 'レース数': n_races}
        for _, sr in s.iterrows():
            row[f"{sr['券種']} 回収率"] = sr['回収率']
            row[f"{sr['券種']} 的中"]  = sr['的中']
        out.append(row)
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).sort_values('日付').reset_index(drop=True)


def _int(v) -> int | None:
    try:
        return int(v) if v and str(v) not in ('', 'nan', 'None') else None
    except (ValueError, TypeError):
        return None
