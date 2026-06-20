"""
予測ログの保存・結果の登録・回収率計算を行うモジュール。

ファイル:
  data/processed/pred_log.parquet    : 予測時の印・買い目ログ
  data/processed/result_log.parquet  : 確定後の着順・払戻ログ
"""
from __future__ import annotations

from pathlib import Path
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


# ── 予測ログ ──────────────────────────────────────────────────────────────

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

    baren_tickets = '|'.join(
        f"{b['馬名1']}-{b['馬名2']}"
        for b in buy_tickets.get('馬連', [])
    )
    sanrenpuku_tickets = '|'.join(
        '-'.join(combo)
        for combo in buy_tickets.get('三連複_fmtn', {}).get('組み合わせ', [])
    )

    row = {
        'race_id':             race_id,
        'date':                date,
        'venue':               venue,
        'r_num':               r_num,
        'race_name':           race_name,
        'honmei':              honmei,
        'taiko':               taiko,
        'tansho':              tansho,
        'renshita':            renshita,
        'myomi':               myomi,
        'baren_tickets':       baren_tickets,
        'sanrenpuku_tickets':  sanrenpuku_tickets,
    }

    df = load_pred_log()
    df = df[df['race_id'] != race_id]  # 既存を削除
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
        'chaku1':         chaku1,
        'chaku2':         chaku2,
        'chaku3':         chaku3,
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

    Returns
    -------
    dict with keys:
      'summary'  : DataFrame (券種, 的中, 外れ, 投資, 回収, 回収率)
      'detail'   : DataFrame (レースごとの詳細)
    """
    merged = pred_log.merge(result_log, on='race_id', how='inner')

    if merged.empty:
        empty_summary = pd.DataFrame(columns=['券種', '的中', '外れ', '投資額', '回収額', '回収率'])
        return {'summary': empty_summary, 'detail': merged}

    rows_tan   = []
    rows_fuku  = []
    rows_baren = []
    rows_3fuku = []
    detail_rows = []

    for _, r in merged.iterrows():
        race_id   = r['race_id']
        label     = f"{r.get('date','')[:4]}年{r.get('date','')[4:6]}月{r.get('date','')[6:]}日 {r.get('venue','')} {r.get('r_num','')}R {r.get('race_name','')}"
        honmei    = str(r.get('honmei', ''))
        chaku1    = str(r.get('chaku1', ''))
        chaku2    = str(r.get('chaku2', ''))
        chaku3    = str(r.get('chaku3', ''))
        top3      = {chaku1, chaku2, chaku3}

        tan_pay  = _int(r.get('tan_pay'))
        fuku_pays = [_int(r.get('fuku1_pay')), _int(r.get('fuku2_pay')), _int(r.get('fuku3_pay'))]
        baren_pay = _int(r.get('baren_pay'))
        s3_pay    = _int(r.get('sanrenpuku_pay'))

        # 単勝: ◎が1着
        tan_hit  = (honmei == chaku1)
        tan_recv = tan_pay if (tan_hit and tan_pay) else 0
        rows_tan.append({'race_id': race_id, 'label': label,
                         'bet': BET_UNIT, 'recv': tan_recv, 'hit': tan_hit})

        # 複勝: ◎が3着以内
        fuku_hit = (honmei in top3)
        if fuku_hit:
            idx = [chaku1, chaku2, chaku3].index(honmei) if honmei in [chaku1, chaku2, chaku3] else -1
            fuku_recv = fuku_pays[idx] if idx >= 0 and fuku_pays[idx] else 0
        else:
            fuku_recv = 0
        rows_fuku.append({'race_id': race_id, 'label': label,
                          'bet': BET_UNIT, 'recv': fuku_recv, 'hit': fuku_hit})

        # 馬連: 各点ごとに判定
        baren_str = str(r.get('baren_tickets', ''))
        baren_bet = 0
        baren_recv = 0
        baren_hit = False
        for ticket in [t for t in baren_str.split('|') if t]:
            parts = ticket.split('-')
            if len(parts) == 2:
                baren_bet += BET_UNIT
                names = {parts[0], parts[1]}
                if names <= top3 and baren_pay:  # 2頭とも3着以内（馬連は1,2着）
                    # 厳密には1着・2着のみ
                    if parts[0] in {chaku1, chaku2} and parts[1] in {chaku1, chaku2}:
                        baren_recv += baren_pay
                        baren_hit = True
        rows_baren.append({'race_id': race_id, 'label': label,
                           'bet': baren_bet, 'recv': baren_recv, 'hit': baren_hit})

        # 三連複: 各点ごとに判定
        s3_str = str(r.get('sanrenpuku_tickets', ''))
        s3_bet = 0
        s3_recv = 0
        s3_hit = False
        for ticket in [t for t in s3_str.split('|') if t]:
            parts = set(ticket.split('-'))
            if len(parts) == 3:
                s3_bet += BET_UNIT
                if parts == top3 and s3_pay:
                    s3_recv += s3_pay
                    s3_hit = True
        rows_3fuku.append({'race_id': race_id, 'label': label,
                           'bet': s3_bet, 'recv': s3_recv, 'hit': s3_hit})

        detail_rows.append({
            'レース':     label,
            '本命':       honmei,
            '1着':        chaku1, '2着': chaku2, '3着': chaku3,
            '単勝':       '◯' if tan_hit else '×',
            '複勝':       '◯' if fuku_hit else '×',
            '馬連':       '◯' if baren_hit else '×',
            '三連複':     '◯' if s3_hit else '×',
            '単勝払戻':   tan_pay,
            '複勝払戻':   fuku_pays[0],
            '馬連払戻':   baren_pay,
            '三連複払戻': s3_pay,
        })

    def _summary_row(name, rows):
        if not rows:
            return None
        df = pd.DataFrame(rows)
        total_bet  = df['bet'].sum()
        total_recv = df['recv'].sum()
        hits       = df['hit'].sum()
        misses     = len(df) - hits
        roi        = total_recv / total_bet * 100 if total_bet > 0 else 0
        return {
            '券種': name,
            '的中': int(hits),
            '外れ': int(misses),
            '投資額': int(total_bet),
            '回収額': int(total_recv),
            '回収率': round(roi, 1),
        }

    summary_rows = [
        _summary_row('本命◎ 単勝', rows_tan),
        _summary_row('本命◎ 複勝', rows_fuku),
        _summary_row('馬連',       rows_baren),
        _summary_row('三連複',     rows_3fuku),
    ]
    summary = pd.DataFrame([r for r in summary_rows if r])
    detail  = pd.DataFrame(detail_rows)

    return {'summary': summary, 'detail': detail}


def _int(v) -> int | None:
    try:
        return int(v) if v and str(v) not in ('', 'nan', 'None') else None
    except (ValueError, TypeError):
        return None
