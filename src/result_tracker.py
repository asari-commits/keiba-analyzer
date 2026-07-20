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
    """予測時の印・買い目をログに保存する（同一race_id・同一レースは上書き）。"""
    row = _build_pred_row(race_id, date, venue, r_num, race_name, show_df, buy_tickets)
    if row is None:
        return
    df = _read_pred_raw()
    if not df.empty:
        df = df[df['race_id'].astype(str) != str(race_id)]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _write_pred_log(df)


def _coerce_pred_types(df: pd.DataFrame) -> pd.DataFrame:
    """予測ログの列の型を統一する。

    CSV復元した行は全て文字列、自動記録した行は r_num が int になるため、
    そのまま結合すると r_num が object（str と int の混在）になり、
    parquet 保存が ArrowTypeError で失敗して記録が一切残らなくなる。
    保存・読込の両方でここを通して型を揃える。
    """
    d = df.copy() if df is not None else pd.DataFrame()
    for c in PRED_COLS:
        if c not in d.columns:
            d[c] = 0 if c == 'r_num' else ''
    d['r_num'] = pd.to_numeric(d['r_num'], errors='coerce').fillna(0).astype('int64')
    for c in PRED_COLS:
        if c == 'r_num':
            continue
        d[c] = (d[c].astype(str)
                .replace({'nan': '', 'None': '', 'NaN': '', '<NA>': ''}))
    return d[PRED_COLS]


def _read_pred_raw() -> pd.DataFrame:
    if PRED_LOG_PATH.exists():
        return _coerce_pred_types(pd.read_parquet(PRED_LOG_PATH))
    return pd.DataFrame(columns=PRED_COLS)


def _dedup_pred_log(df: pd.DataFrame) -> pd.DataFrame:
    """同一レース(日付×開催×R)の重複を排除する。
    同じレースが netkeiba正規ID と フォールバックID の両方で保存され、
    1日のレース数が水増しされる問題への対策。
    優先: 本命(印)あり > 正規ID(数字10〜12桁) > 新しい行。
    ※本命が無い行は回収率集計に使えないため、IDの体裁より本命の有無を優先する。
      （旧実装は正規IDを優先していたため、本命ありの行が本命なしの行に負けて
        捨てられ、集計対象レースが大きく減っていた）"""
    if df is None or df.empty or not {'date', 'venue', 'r_num'}.issubset(df.columns):
        return df
    d = df.reset_index(drop=True).copy()
    d['_ord'] = range(len(d))
    _rid = d['race_id'].astype(str)
    d['_is_real'] = _rid.str.fullmatch(r'\d{10,12}').fillna(False).astype(int)
    if 'honmei' in d.columns:
        _hon = (d['honmei'].astype(str).str.strip()
                .replace({'nan': '', 'None': '', 'NaN': '', '<NA>': ''}))
        d['_has_mark'] = (_hon.str.len() > 0).astype(int)
    else:
        d['_has_mark'] = 0
    d['_dt'] = d['date'].astype(str)
    d['_vn'] = d['venue'].astype(str).map(normalize_venue)
    d['_rn'] = pd.to_numeric(d['r_num'], errors='coerce')
    d = d.sort_values(['_has_mark', '_is_real', '_ord'], ascending=[False, False, False])
    d = d.drop_duplicates(subset=['_dt', '_vn', '_rn'], keep='first')
    d = d.sort_values('_ord').drop(columns=['_ord', '_is_real', '_has_mark', '_dt', '_vn', '_rn'])
    return d.reset_index(drop=True)


def _write_pred_log(df: pd.DataFrame) -> None:
    # 型を揃えてから保存（str と int の混在で parquet 保存が失敗するのを防ぐ）
    df = _coerce_pred_types(_dedup_pred_log(df))
    PRED_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PRED_LOG_PATH, index=False)


def _build_pred_row(race_id, date, venue, r_num, race_name, show_df, buy_tickets):
    """save_pred_log 用の1行dictを組み立てる（_mark無しはNone）。"""
    if '_mark' not in show_df.columns:
        return None
    marks = show_df.set_index('馬名')['_mark'].to_dict()
    honmei   = next((n for n, m in marks.items() if m == '◎'), '')
    taiko    = next((n for n, m in marks.items() if m == '○'), '')
    tansho   = next((n for n, m in marks.items() if m == '▲'), '')
    renshita = ','.join(n for n, m in marks.items() if m == '△')
    myomi    = next((n for n, m in marks.items() if m == '★'), '')
    baren_list = [[b['馬名1'], b['馬名2']] for b in buy_tickets.get('馬連', [])]
    s3_list = [list(combo) for combo in buy_tickets.get('三連複_fmtn', {}).get('組み合わせ', [])]
    return {
        'race_id': race_id, 'date': date, 'venue': normalize_venue(venue),
        'r_num': r_num, 'race_name': race_name,
        'honmei': honmei, 'taiko': taiko, 'tansho': tansho,
        'renshita': renshita, 'myomi': myomi,
        'baren_tickets': _encode_tickets(baren_list),
        'sanrenpuku_tickets': _encode_tickets(s3_list),
    }


def save_pred_logs_bulk(items) -> int:
    """複数レースの予測ログを一括保存（IOは1回）。
    items = [(race_id, date, venue, r_num, race_name, show_df, buy_tickets), ...]
    戻り値: 保存件数。"""
    rows = [r for it in items if (r := _build_pred_row(*it)) is not None]
    if not rows:
        return 0
    new_ids = {str(r['race_id']) for r in rows}
    df = _read_pred_raw()
    if not df.empty:
        df = df[~df['race_id'].astype(str).isin(new_ids)]
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    _write_pred_log(df)   # 保存時に (日付×開催×R) 重複を排除
    return len(rows)


def load_pred_log() -> pd.DataFrame:
    return _dedup_pred_log(_read_pred_raw())


# ── 結果ログ ──────────────────────────────────────────────────────────────

def save_result(race_id: str, chaku1: str, chaku2: str, chaku3: str,
                tan_pay: int | None, fuku1_pay: int | None,
                fuku2_pay: int | None, fuku3_pay: int | None,
                baren_pay: int | None, sanrenpuku_pay: int | None,
                sanrentan_pay: int | None = None) -> None:
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
        'sanrentan_pay':  sanrentan_pay,
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

    from itertools import combinations
    _rows = {'単勝': [], '複勝': [], '馬連': [], '三連複': [], '三連単': []}
    detail_rows = []

    for _, r in merged.iterrows():
        d = str(r.get('date', ''))
        label = f"{d[:4]}/{d[4:6]}/{d[6:]} {r.get('venue','')} {r.get('r_num','')}R {r.get('race_name','')}"
        chaku1 = _norm(r.get('chaku1', ''))
        chaku2 = _norm(r.get('chaku2', ''))
        chaku3 = _norm(r.get('chaku3', ''))
        top2 = {chaku1, chaku2} - {''}
        top3 = {chaku1, chaku2, chaku3} - {''}

        # 印から本命◎と相手（○▲△…、★は△内の妙味＝重複除去で最大5頭。旧ログは★別頭で最大6頭）を取得
        honmei = _norm(r.get('honmei', ''))
        _aite_raw = ([_norm(r.get('taiko', '')), _norm(r.get('tansho', ''))]
                     + [_norm(x) for x in str(r.get('renshita', '') or '').split(',')]
                     + [_norm(r.get('myomi', ''))])
        aite = [x for x in dict.fromkeys(_aite_raw) if x and x != honmei]
        nA = len(aite)

        tan_pay = _int(r.get('tan_pay'))
        fuku_pays = [_int(r.get('fuku1_pay')), _int(r.get('fuku2_pay')), _int(r.get('fuku3_pay'))]
        baren_pay = _int(r.get('baren_pay'))
        s3_pay = _int(r.get('sanrenpuku_pay'))
        st_pay = _int(r.get('sanrentan_pay'))

        # 単勝(◎ 1点)
        tan_hit = bool(honmei and honmei == chaku1)
        _rows['単勝'].append({'bet': BET_UNIT, 'recv': (tan_pay or 0) if tan_hit else 0, 'hit': tan_hit})

        # 複勝(◎ 1点)
        fuku_hit = bool(honmei and honmei in top3)
        fuku_recv = 0
        if fuku_hit:
            _cl = [chaku1, chaku2, chaku3]
            _i = _cl.index(honmei) if honmei in _cl else -1
            fuku_recv = (fuku_pays[_i] or 0) if _i >= 0 else 0
        _rows['複勝'].append({'bet': BET_UNIT, 'recv': fuku_recv, 'hit': fuku_hit})

        # ── 馬連・三連複・三連単は BOX（印馬 = ◎＋相手○▲△★ の全通り）──
        box_set = ({honmei} if honmei else set()) | set(aite)
        _N = len(box_set)

        # 馬連 BOX (C(N,2)点): 1-2着の2頭が両方 box 内
        baren_bet = (_N * (_N - 1) // 2) * BET_UNIT
        baren_hit = bool(len(top2) == 2 and top2.issubset(box_set))
        _rows['馬連'].append({'bet': baren_bet,
                              'recv': (baren_pay or 0) if baren_hit else 0, 'hit': baren_hit})

        # 三連複 BOX (C(N,3)点): 1-2-3着の3頭が全て box 内
        s3_bet = (_N * (_N - 1) * (_N - 2) // 6) * BET_UNIT
        s3_hit = bool(len(top3) == 3 and top3.issubset(box_set))
        _rows['三連複'].append({'bet': s3_bet,
                                'recv': (s3_pay or 0) if s3_hit else 0, 'hit': s3_hit})

        # 三連単 BOX (P(N,3)=N*(N-1)*(N-2)点): 1-2-3着が全て box 内（順序はBOXで網羅）
        st_bet = (_N * (_N - 1) * (_N - 2)) * BET_UNIT
        st_hit = bool(len(top3) == 3 and top3.issubset(box_set))
        _rows['三連単'].append({'bet': st_bet,
                                'recv': (st_pay or 0) if st_hit else 0, 'hit': st_hit})

        detail_rows.append({
            'レース': label, '本命': honmei, '1着': chaku1, '2着': chaku2, '3着': chaku3,
            '単': '◯' if tan_hit else '×', '複': '◯' if fuku_hit else '×',
            '馬連': '◯' if baren_hit else '×', '三複': '◯' if s3_hit else '×',
            '三単': '◯' if st_hit else '×',
            '単払戻': tan_pay, '複払戻': fuku_pays[0] if fuku_hit else None,
            '馬連払戻': baren_pay if baren_hit else None,
            '三連複払戻': s3_pay if s3_hit else None,
            '三連単払戻': st_pay if st_hit else None,
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

    summary = pd.DataFrame([x for x in [
        _summary_row('本命◎ 単勝', _rows['単勝']),
        _summary_row('本命◎ 複勝', _rows['複勝']),
        _summary_row('馬連BOX', _rows['馬連']),
        _summary_row('三連複BOX', _rows['三連複']),
        _summary_row('三連単BOX', _rows['三連単']),
    ] if x])
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


# ══════════════════════════════════════════════════════════════════════
# 本命◎の単勝・複勝回収率（結果は master から取得）
#   netkeibaのスクレイピングに依存せず、毎週Targetで更新している master の
#   着順・単勝配当・複勝配当を突合するため、取りこぼしが起きない。
# ══════════════════════════════════════════════════════════════════════
import re as _re

MASTER_PARQUET = Path(__file__).parent.parent / "data" / "processed" / "master.parquet"

# 競馬場名 → master の開催コード内の略称（中京は「名」）
VENUE_AB = {'札幌': '札', '函館': '函', '福島': '福', '新潟': '新', '東京': '東',
            '中山': '中', '中京': '名', '京都': '京', '阪神': '阪', '小倉': '小'}

_MARK_RE = _re.compile(r'^[\s　$*▲△◎○☆★\.]+')
_master_cache: dict = {}


def _norm_horse(name) -> str:
    """馬名を比較用に正規化（先頭の印マーカー・前後空白を除去）。"""
    s = _MARK_RE.sub('', str(name))
    return _re.sub(r'[\s　]+$', '', s).strip()


def _pay_yen(v) -> float:
    """配当を円に変換。空・カッコ付き（＝オッズ表記）は0。"""
    s = str(v).strip()
    if s in ('', 'nan', 'None', 'NaN') or s.startswith('(') or s.startswith('（'):
        return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0


def _load_master_results() -> pd.DataFrame:
    """master から突合用の軽量テーブルを作る（ファイル更新時のみ読み直す）。"""
    if not MASTER_PARQUET.exists():
        return pd.DataFrame()
    mt = MASTER_PARQUET.stat().st_mtime
    if _master_cache.get('mtime') == mt:
        return _master_cache['df']
    m = pd.read_parquet(MASTER_PARQUET,
                        columns=['日付', '開催', 'Ｒ', '馬名', '着順_num', '単勝配当', '複勝配当'])
    m['_d'] = m['日付'].astype(str)
    m['_ab'] = m['開催'].astype(str).str.replace(r'[0-9A-C]', '', regex=True)
    m['_r'] = pd.to_numeric(m['Ｒ'], errors='coerce')
    m['_nm'] = m['馬名'].astype(str).map(_norm_horse)
    _master_cache['mtime'], _master_cache['df'] = mt, m
    return m


def calc_honmei_roi(pred_log: pd.DataFrame) -> dict:
    """本命◎に単勝100円・複勝100円ずつ賭けた場合の回収率を master の結果から算出。

    戻り値:
      summary : 券種別サマリー（本命◎単勝 / 本命◎複勝）
      detail  : 1行=1レースの明細
      stats   : 集計対象・未突合の件数
    """
    empty_sum = pd.DataFrame(columns=['券種', '的中', '外れ', '的中率', '投資額', '回収額', '回収率'])
    if pred_log is None or pred_log.empty:
        return {'summary': empty_sum, 'detail': pd.DataFrame(),
                'stats': {'n_pred': 0, 'n_honmei': 0, 'n_matched': 0, 'n_unmatched': 0}}

    m = _load_master_results()
    p = pred_log.copy()
    p['_honmei'] = p['honmei'].astype(str).map(_norm_horse)
    has = p[p['_honmei'] != '']

    rows = []
    n_unmatched = 0
    for _, r in has.iterrows():
        d8 = str(r.get('date', ''))
        d6 = d8[2:] if len(d8) == 8 else d8
        ab = VENUE_AB.get(str(r.get('venue', '')).strip())
        rn = pd.to_numeric(r.get('r_num'), errors='coerce')
        if m.empty or ab is None or pd.isna(rn):
            n_unmatched += 1
            continue
        sub = m[(m['_d'] == d6) & (m['_ab'] == ab) & (m['_r'] == rn)]
        hit = sub[sub['_nm'] == r['_honmei']]
        if hit.empty:
            n_unmatched += 1
            continue
        h = hit.iloc[0]
        chaku = pd.to_numeric(h['着順_num'], errors='coerce')
        tan_hit = bool(chaku == 1)
        fuku_hit = bool(pd.notna(chaku) and chaku <= 3)
        rows.append({
            '日付': d8, '開催': str(r.get('venue', '')), 'R': int(rn),
            'レース名': str(r.get('race_name', '') or ''),
            '本命': str(r.get('honmei', '')), '着順': int(chaku) if pd.notna(chaku) else None,
            '単勝': '的中' if tan_hit else '―',
            '単払戻': _pay_yen(h['単勝配当']) if tan_hit else 0.0,
            '複勝': '的中' if fuku_hit else '―',
            '複払戻': _pay_yen(h['複勝配当']) if fuku_hit else 0.0,
        })

    detail = pd.DataFrame(rows)
    stats = {'n_pred': len(p), 'n_honmei': len(has),
             'n_matched': len(detail), 'n_unmatched': n_unmatched}
    if detail.empty:
        return {'summary': empty_sum, 'detail': detail, 'stats': stats}

    n = len(detail)

    def _row(name, hit_col, pay_col):
        hits = int((detail[hit_col] == '的中').sum())
        bet = n * BET_UNIT
        recv = float(detail[pay_col].sum())
        return {'券種': name, '的中': hits, '外れ': n - hits,
                '的中率': round(hits / n * 100, 1),
                '投資額': int(bet), '回収額': int(recv),
                '回収率': round(recv / bet * 100, 1) if bet else 0.0}

    summary = pd.DataFrame([_row('本命◎ 単勝', '単勝', '単払戻'),
                            _row('本命◎ 複勝', '複勝', '複払戻')])
    return {'summary': summary, 'detail': detail, 'stats': stats}


def calc_honmei_daily(detail: pd.DataFrame) -> pd.DataFrame:
    """明細から日別の回収率を集計する。"""
    if detail is None or detail.empty:
        return pd.DataFrame()
    g = detail.groupby('日付')
    out = pd.DataFrame({
        'レース数': g.size(),
        '単勝的中': (detail['単勝'] == '的中').groupby(detail['日付']).sum(),
        '単勝回収率': g['単払戻'].sum() / (g.size() * BET_UNIT) * 100,
        '複勝的中': (detail['複勝'] == '的中').groupby(detail['日付']).sum(),
        '複勝回収率': g['複払戻'].sum() / (g.size() * BET_UNIT) * 100,
    }).reset_index()
    out['単勝回収率'] = out['単勝回収率'].round(1)
    out['複勝回収率'] = out['複勝回収率'].round(1)
    return out.sort_values('日付', ascending=False)


def master_race_counts() -> pd.DataFrame:
    """master にある日付ごとのレース数を返す（補完対象の判定用）。"""
    m = _load_master_results()
    if m.empty:
        return pd.DataFrame(columns=['日付', 'master_R'])
    g = (m.groupby('_d')[['_ab', '_r']].apply(lambda x: x.drop_duplicates().shape[0])
         .rename('master_R').reset_index().rename(columns={'_d': '日付'}))
    return g


def backfill_from_master(dates6, progress=None) -> dict:
    """master に結果がある日について、予測ログに無い（本命が入っていない）レースを
    リーク無しで再予測し、本命を求めて予測ログへ追加する。

    dates6 : ['260711', ...] の6桁日付リスト
    progress : 進捗表示用コールバック（任意）
    戻り値: {'added': 追加レース数, 'skipped': 既存レース数, 'errors': [..]}
    """
    import numpy as _np
    import pipeline_target as _P
    from reliability import assign_marks as _am, build_buy_tickets as _bt

    m_all = pd.read_parquet(MASTER_PARQUET)
    cur = load_pred_log()
    have = set()
    if not cur.empty:
        _ok = (cur['honmei'].astype(str).str.strip()
               .replace({'nan': '', 'None': '', 'NaN': '', '<NA>': ''}) != '')
        for _d, _v, _r in zip(cur.loc[_ok, 'date'].astype(str),
                              cur.loc[_ok, 'venue'].astype(str),
                              pd.to_numeric(cur.loc[_ok, 'r_num'], errors='coerce').fillna(0).astype(int)):
            have.add((_d, _v, _r))

    _AB2NAME = {v: k for k, v in VENUE_AB.items()}
    _RESULT_COLS = ['着順', '走破タイム', '走破秒', '着順_num', '着差',
                    '単勝配当', '複勝配当', '馬体重', '上3F地点差']
    items, skipped, errors = [], 0, []

    for d6 in dates6:
        d6 = str(d6)
        d8 = d6 if len(d6) == 8 else '20' + d6
        sub = m_all[m_all['日付'].astype(str) == d6].copy()
        if sub.empty:
            errors.append(f'{d8}: masterに該当日なし')
            continue
        if progress:
            progress(f'{d8} を再予測中…（{sub.groupby(["開催", "Ｒ"]).ngroups}R）')
        # 当該レースの結果は使わずに予測（_build_features_and_slice が予測日を
        # master履歴から除外するためリーク無し）
        keep_pop = pd.to_numeric(sub.get('人気'), errors='coerce')
        feed = sub.copy()
        feed['日付'] = d8
        for c in _RESULT_COLS:
            if c in feed.columns:
                feed[c] = _np.nan
        try:
            pred = _P.predict_both_from_df(feed)
        except Exception as e:
            errors.append(f'{d8}: 予測失敗 {type(e).__name__}: {e}')
            continue
        pred['_r'] = pd.to_numeric(pred['Ｒ'], errors='coerce').fillna(0).astype(int)
        pred['_pop_int'] = keep_pop.reindex(pred.index).fillna(99).astype(int) \
            if keep_pop is not None else 99
        for (kai, rn), g in pred.groupby(['開催', '_r']):
            if rn <= 0:
                continue
            ab = _re.sub(r'[0-9A-C]', '', str(kai))
            vname = _AB2NAME.get(ab, ab)
            if (d8, vname, int(rn)) in have:
                skipped += 1
                continue
            g = g.copy()
            g['pred_rank'] = g['pred_score'].rank(ascending=False, method='min').astype(int)
            if 'pred_score_anaba' in g.columns:
                g['pred_rank_anaba'] = g['pred_score_anaba'].rank(ascending=False, method='min').astype(int)
            g = _am(g)
            rname = str(g['レース名'].iloc[0]) if 'レース名' in g.columns else ''
            items.append((f'{d8}_{kai}_{int(rn)}', d8, vname, int(rn), rname, g, _bt(g)))

    added = save_pred_logs_bulk(items) if items else 0
    return {'added': added, 'skipped': skipped, 'errors': errors}
