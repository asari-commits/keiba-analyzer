"""
レース条件別 人気馬信頼度モジュール。

master.csv の実績データから「レース種別 × 人気帯」ごとの馬券内率を集計し、
モデルスコアに組み合わせて 本命◎/対抗○/穴△ を導出する。
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

MASTER_CSV        = Path(__file__).parent.parent / "data" / "processed" / "master.csv"
RELIABILITY_CACHE = Path(__file__).parent.parent / "data" / "processed" / "reliability_table.parquet"


# ── レース種別分類 ────────────────────────────────────────────────────────

def classify_race_type(race_row: pd.Series | dict) -> str:
    """
    1行（レース情報）からレース種別文字列を返す。

    返り値例:
        'G1_芝', 'G2_ダ', 'G3_芝', 'OP_芝', 'ハンデ_芝',
        '馬齢_ダ', '別定_芝', '定量_芝',
        '新馬_芝', '未勝利_芝', '牝限_芝' など
    """
    def _get(col, default=''):
        v = race_row.get(col, default)
        return str(v) if v is not None and str(v) != 'nan' else default

    race_name   = _get('レース名')
    weight_type = _get('重量種別')   # ハンデ / 定量 / 馬齢 / 別定
    limit_col   = _get('限定')        # 牝 など
    is_turf     = _get('is_turf', _get('芝・ダ', ''))

    # 芝 / ダート
    if str(is_turf) == '1' or str(is_turf).startswith('芝'):
        surf = '芝'
    elif str(is_turf) == '0' or str(is_turf).startswith('ダ'):
        surf = 'ダ'
    else:
        surf = '芝'  # fallback

    # 重賞グレード
    for g in ('G1', 'G2', 'G3', 'GⅠ', 'GⅡ', 'GⅢ'):
        if g in race_name:
            grade = g.replace('Ⅰ', '1').replace('Ⅱ', '2').replace('Ⅲ', '3')
            return f'{grade}_{surf}'

    # 新馬
    if '新馬' in race_name:
        return f'新馬_{surf}'

    # 未勝利
    if '未勝利' in race_name:
        return f'未勝利_{surf}'

    # 牝馬限定
    if '牝' in race_name or '牝' in limit_col:
        return f'牝限_{surf}'

    # 重量種別
    if weight_type == 'ハンデ':
        return f'ハンデ_{surf}'
    if weight_type == '定量':
        return f'定量_{surf}'
    if weight_type == '別定':
        return f'別定_{surf}'
    if weight_type == '馬齢':
        return f'馬齢_{surf}'

    return f'条件_{surf}'


def pop_band(pop: int | float) -> str:
    """人気を帯に変換"""
    p = int(pop) if pd.notna(pop) else 99
    if p == 1:   return '1番人気'
    if p <= 3:   return '2-3番人気'
    if p <= 6:   return '4-6番人気'
    return '7番人気以下'


# ── 信頼度テーブルの構築 ─────────────────────────────────────────────────

def build_reliability_table(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    master.csv から レース種別 × 人気帯 ごとの馬券内率を集計する。
    最低サンプル数 30 未満は全体平均で補完。

    返り値列: race_type, pop_band, fuku_rate, win_rate, n
    """
    df = master_df.copy()

    # 着順数値化
    if '着順_num' not in df.columns:
        tr = str.maketrans('０１２３４５６７８９', '0123456789')
        df['着順_num'] = pd.to_numeric(
            df['着順'].astype(str).str.translate(tr).str.extract(r'(\d+)')[0],
            errors='coerce'
        )

    # 人気数値化
    df['_pop'] = pd.to_numeric(df['人気'], errors='coerce')
    df['_pop_band'] = df['_pop'].apply(lambda x: pop_band(x) if pd.notna(x) else None)
    df = df[df['_pop_band'].notna() & df['着順_num'].notna()].copy()

    # レース種別分類
    df['_race_type'] = df.apply(classify_race_type, axis=1)

    # 馬券内フラグ
    df['_fuku'] = (df['着順_num'] <= 3).astype(int)
    df['_win']  = (df['着順_num'] == 1).astype(int)

    # 集計
    grp = df.groupby(['_race_type', '_pop_band']).agg(
        fuku_rate=('_fuku', 'mean'),
        win_rate=('_win', 'mean'),
        n=('_fuku', 'count'),
    ).reset_index().rename(columns={'_race_type': 'race_type', '_pop_band': 'pop_band'})

    # サンプル数が少ない行は全体平均で補完
    overall = df.groupby('_pop_band').agg(
        fuku_rate=('_fuku', 'mean'),
        win_rate=('_win', 'mean'),
    ).rename(columns={'fuku_rate': '_overall_fuku', 'win_rate': '_overall_win'})

    grp = grp.merge(overall, left_on='pop_band', right_index=True, how='left')
    mask = grp['n'] < 30
    grp.loc[mask, 'fuku_rate'] = grp.loc[mask, '_overall_fuku']
    grp.loc[mask, 'win_rate']  = grp.loc[mask, '_overall_win']
    grp = grp.drop(columns=['_overall_fuku', '_overall_win'])

    return grp


@lru_cache(maxsize=1)
def _load_reliability_table() -> pd.DataFrame:
    # キャッシュ済み parquet があればそちらを優先（高速）
    if RELIABILITY_CACHE.exists():
        return pd.read_parquet(RELIABILITY_CACHE)
    if not MASTER_CSV.exists():
        return pd.DataFrame(columns=['race_type', 'pop_band', 'fuku_rate', 'win_rate', 'n'])
    master = pd.read_csv(MASTER_CSV, encoding='utf-8-sig', low_memory=False)
    table = build_reliability_table(master)
    RELIABILITY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(RELIABILITY_CACHE, index=False)
    return table


def rebuild_reliability_cache() -> pd.DataFrame:
    """master.csv から信頼度テーブルを再構築してキャッシュを更新する"""
    if not MASTER_CSV.exists():
        return pd.DataFrame(columns=['race_type', 'pop_band', 'fuku_rate', 'win_rate', 'n'])
    master = pd.read_csv(MASTER_CSV, encoding='utf-8-sig', low_memory=False)
    table = build_reliability_table(master)
    RELIABILITY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(RELIABILITY_CACHE, index=False)
    _load_reliability_table.cache_clear()
    return table


def get_reliability(race_type: str, pop: int | float) -> dict:
    """
    race_type と 人気 から馬券内率・勝率を返す。
    テーブルにない組み合わせはデフォルト値。
    """
    table = _load_reliability_table()
    pb = pop_band(pop)
    row = table[(table['race_type'] == race_type) & (table['pop_band'] == pb)]
    if row.empty:
        # fallback: 同じ人気帯の全体平均
        row = table[table['pop_band'] == pb]
    if row.empty:
        return {'fuku_rate': 0.33, 'win_rate': 0.10, 'n': 0}
    r = row.iloc[0]
    return {'fuku_rate': float(r['fuku_rate']), 'win_rate': float(r['win_rate']), 'n': int(r['n'])}


# ── 本命スコア計算（案A：モデル順位主軸）────────────────────────────────

def calc_honmei_score(show_df: pd.DataFrame, race_row: pd.Series) -> pd.DataFrame:
    """
    【案A】モデル順位を主軸に 本命◎/対抗○/穴△ を決定する。
    人気信頼度はスコアに影響させず、信頼度表示（_confidence）のみに使う。

    追加列:
      - _fuku_rate   : レース種別×人気帯の実績馬券内率（表示用）
      - _confidence  : モデル評価と人気の一致度（高いほど信頼性が高い）
      - _honmei_label: '本命◎' / '対抗○' / '穴△' / '－'
      - _teki_badge  : '◎' / '○' / '△' / '×'

    ラベル割り当てルール:
      本命◎ = モデル1位
      対抗○ = モデル2〜3位 かつ 人気1〜5番
      穴△  = モデル1〜4位 かつ 人気6番以下
      （対抗と穴が重複する場合: モデル2〜3位で人気6以上 → 穴△を優先）
      残り  = '－'
    """
    out = show_df.copy()
    race_type = classify_race_type(race_row)
    out['race_type'] = race_type

    # 人気信頼度を取得（表示用のみ）
    fukus = []
    confidences = []
    for _, row in out.iterrows():
        pop = row.get('_pop_int', 10)
        rel = get_reliability(race_type, pop)
        fuku = rel['fuku_rate']
        fukus.append(fuku)
        # 信頼度 = モデル確率と実績馬券内率が両方高いほど高い
        prob = float(row.get('_win_prob', 0.05))
        # レース内での相対モデル確率（0〜1）
        confidences.append((prob, fuku))

    out['_fuku_rate'] = fukus

    # モデル確率をレース内で正規化して信頼度スコアを計算
    probs = out['_win_prob'].fillna(0).values
    max_prob = probs.max() if probs.max() > 0 else 1.0
    rel_probs = probs / max_prob  # 最高確率の馬を1.0とした相対値

    out['_confidence'] = [
        rel_probs[i] * 0.7 + fukus[i] * 0.3
        for i in range(len(out))
    ]

    # ── ラベル割り当て（モデル順位主軸）──────────────────────────────
    model_ranks = out['pred_rank'].fillna(99).astype(int)
    pops        = out['_pop_int'].fillna(99).astype(int)

    labels = []
    badges = []
    for i in range(len(out)):
        mr  = int(model_ranks.iloc[i])
        pop = int(pops.iloc[i])

        if mr == 1:
            # モデル1位は常に本命（人気に関わらず）
            labels.append('本命◎')
            badges.append('◎')
        elif mr <= 3 and pop >= 6:
            # モデル上位3位かつ低人気 → 穴
            labels.append('穴△')
            badges.append('△')
        elif mr <= 3:
            # モデル2〜3位かつ人気5番以内 → 対抗
            labels.append('対抗○')
            badges.append('○')
        elif mr == 4 and pop >= 6:
            # モデル4位で低人気 → 穴候補
            labels.append('穴△')
            badges.append('△')
        elif mr <= 5 and pop <= 2:
            # モデル4〜5位だが1〜2番人気 → 対抗（人気が高く侮れない）
            labels.append('対抗○')
            badges.append('○')
        else:
            labels.append('－')
            badges.append('×')

    out['_honmei_label'] = labels
    out['_teki_badge']   = badges
    return out


def honmei_summary(show_df: pd.DataFrame) -> dict:
    """
    本命◎/対抗○/妙味★ の馬名と人気、根拠をまとめた辞書を返す。
    ※ランキング表と一致させるため assign_marks の _mark 列を主軸に判定する
      （旧 _honmei_label は calc_honmei_score 由来で assign_marks と食い違うため不使用）。
    """
    mark_col = '_mark' if '_mark' in show_df.columns else None

    def _get(mark):
        if mark_col is None:
            return None
        rows = show_df[show_df[mark_col] == mark]
        if rows.empty:
            return None
        r = rows.iloc[0]
        return {
            'name':  str(r.get('馬名', '')),
            'pop':   int(r.get('_pop_int', 99)),
            'fuku':  float(r.get('_fuku_rate', 0)),
            'conf':  float(r.get('_confidence', 0)),
            'ev':    r.get('EV単勝', float('nan')),
        }

    return {
        '本命': _get('◎'),
        '対抗': _get('○'),
        '穴':   _get('★'),
        'race_type': show_df['race_type'].iloc[0] if 'race_type' in show_df.columns else '',
    }


# ── 印付けロジック ────────────────────────────────────────────────────────

MARK_COLORS = {
    '◎': '#f1c40f',   # 本命 ゴールド
    '○': '#2ecc71',   # 対抗 グリーン
    '▲': '#e67e22',   # 単穴 オレンジ
    '△': '#3498db',   # 連下 ブルー
    '★': '#9b59b6',   # 妙味 パープル
    '':  '#555',
}

# 単勝の買い判定閾値（バックテスト実証: 本命EV>=20で回収率+14.5%, >=50で+19.3%）
EV_TAN_BUY    = 20.0   # これ以上で「買い」
EV_TAN_STRONG = 50.0   # これ以上で「妙味大」


def assign_marks(show_df: pd.DataFrame) -> pd.DataFrame:
    """各馬に印を機械的に付与（_mark列）。通常モデル順位を主軸、★は妙味フラグ。

      ◎ 本命 = 通常モデル1位
      ○ 対抗 = 通常モデル2位
      ▲ 単穴 = 通常モデル3位
      △ 連下 = 通常モデル4〜6位（最大3頭）
      ★ 妙味 = 連下(通常4〜6位)のうち人気6番以下かつ穴馬モデル最上位の1頭を昇格表示
    買い目の「相手」= ○▲△…（最大5頭）。★は△の中の妙味馬を色分け表示するフラグで、
    別頭ではない（相手には元の△として含まれる）。

    ※旧仕様（★=通常7位以下×穴最上位）を廃止。検証(192R)で当該選定は同人気の市場
      複勝率を5.3pt下回る-EVと判明したため、相手を通常上位に限定し、★は市場が薄評価
      でもモデル/穴が評価する「連下内の妙味馬」を示す指標に変更した。
    """
    out = show_df.copy()
    out['_mark'] = ''
    if 'pred_rank' not in out.columns:
        return out

    mr = pd.to_numeric(out['pred_rank'], errors='coerce').fillna(99).astype(int)
    ar = (pd.to_numeric(out['pred_rank_anaba'], errors='coerce').fillna(99).astype(int)
          if 'pred_rank_anaba' in out.columns else pd.Series(99, index=out.index))
    pop = (pd.to_numeric(out['_pop_int'], errors='coerce').fillna(99).astype(int)
           if '_pop_int' in out.columns else pd.Series(99, index=out.index))

    # ◎○▲ = 通常1,2,3位（各1頭）
    for rank, mark in [(1, '◎'), (2, '○'), (3, '▲')]:
        idx = list(out.index[mr == rank])
        if idx:
            out.loc[idx[0], '_mark'] = mark
    # △ 連下 = 通常4〜6位（各1頭・最大3頭）
    for rank in (4, 5, 6):
        idx = list(out.index[mr == rank])
        if idx:
            out.loc[idx[0], '_mark'] = '△'
    # ★ 妙味 = 連下(△)のうち人気6番以下で穴馬モデル最上位の1頭を★に昇格
    cand = out[(out['_mark'] == '△') & (pop >= 6)]
    if not cand.empty:
        i = ar.reindex(cand.index).idxmin()
        out.loc[i, '_mark'] = '★'

    return out


def build_buy_tickets(show_df: pd.DataFrame) -> dict:
    """
    印に基づくテンプレート買い目を生成する。

    返り値:
        {
          '単勝':       [{'馬名': ..., 'pop': ..., 'ev': ...}],
          '馬連':       [{'馬名1': ..., '馬名2': ..., 'ev_est': ...}],
          '三連複_fmtn': {'1列': [...], '2列': [...], '3列': [...], '点数': N},
        }
    """
    marks = show_df.set_index('馬名')['_mark'].to_dict() if '_mark' in show_df.columns else {}
    pop_d = show_df.set_index('馬名')['_pop_int'].to_dict() if '_pop_int' in show_df.columns else {}
    ev_d  = show_df.set_index('馬名').get('EV単勝', pd.Series(dtype=float)).to_dict()
    prob_d = show_df.set_index('馬名').get('_win_prob', pd.Series(dtype=float)).to_dict()

    honmei  = [n for n, m in marks.items() if m == '◎']
    taiko   = [n for n, m in marks.items() if m == '○']
    tansho  = [n for n, m in marks.items() if m == '▲']
    renshita= [n for n, m in marks.items() if m == '△']
    myomi   = [n for n, m in marks.items() if m == '★']

    def _info(name):
        return {
            '馬名': name,
            'pop':  int(pop_d.get(name, 99)),
            'ev':   float(ev_d.get(name, float('nan'))),
            'prob': float(prob_d.get(name, 0)),
        }

    # 単勝: ◎ をEVで「買い/見送り」判定（本命×EVプラスが妙味の核心。実証で回収率+14〜19%）
    tan = []
    for n in honmei:
        _ti = _info(n)
        _ev = _ti['ev']
        if pd.notna(_ev) and _ev >= EV_TAN_STRONG:
            _ti['buy'], _ti['rating'] = True, '妙味大'
        elif pd.notna(_ev) and _ev >= EV_TAN_BUY:
            _ti['buy'], _ti['rating'] = True, '買い'
        else:
            _ti['buy'], _ti['rating'] = False, '見送り'
        tan.append(_ti)

    # 本命◎ + 相手（○▲△…最大5頭）の流し。★は△内の妙味馬なのでdict.fromkeysで重複除去。
    h = honmei[0] if honmei else None
    aite = [x for x in dict.fromkeys(taiko + tansho + renshita + myomi) if x and x != h]
    nA = len(aite)

    from itertools import combinations, permutations
    fuku  = [_info(h)] if h else []
    baren = [{'馬名1': h, '馬名2': x} for x in aite] if h else []                # nA点
    s3    = [sorted([h, a, b]) for a, b in combinations(aite, 2)] if h else []    # C(nA,2)点
    san   = [[h, a, b] for a, b in permutations(aite, 2)] if h else []           # nA*(nA-1)点

    return {
        '単勝': tan,
        '複勝': fuku,
        '馬連': baren,
        '相手': aite,
        '三連複_fmtn': {'軸': h, '相手': aite, '点数': len(s3),  '組み合わせ': s3},
        '三連単_fmtn': {'軸': h, '相手': aite, '点数': len(san), '組み合わせ': san},
    }


def evaluate_race_bets(race_df: pd.DataFrame, unit: int = 100) -> list:
    """結果回顧用: 1レースの印から各買い目の的中・払戻・損益を実配当で算出する。

    必須列: _mark（◎○▲△★）, 着（int 着順）, 馬番,
            tan_pay/fuku_pay（各馬の単勝/複勝配当・非該当NaN）,
            umaren/sanpuku/santan（レース単位払戻）。
    ◎単勝・◎複勝・★単勝＋テンプレ流し（馬連/三連複/三連単＝◎-相手）を評価。
    オッズ列が無い/全欠損なら空listを返す（呼び出し側で非表示に）。
    """
    if race_df.empty or '_mark' not in race_df.columns or 'tan_pay' not in race_df.columns:
        return []
    d = race_df.copy()
    d['着'] = pd.to_numeric(d['着'], errors='coerce')
    d['馬番'] = pd.to_numeric(d['馬番'], errors='coerce')
    if d['着'].notna().sum() == 0:
        return []

    def _uma(mark):
        r = d[d['_mark'] == mark]
        return None if r.empty else r.iloc[0]

    hon = _uma('◎')
    if hon is None:
        return []
    hb = hon['馬番']
    aite = d[d['_mark'].isin(['○', '▲', '△', '★'])]['馬番'].dropna().tolist()
    aite = [a for a in aite if a != hb]
    aset = set(aite)
    nA = len(aite)
    star = _uma('★')

    top = {int(k): d.loc[d['着'] == k, '馬番'].iloc[0]
           for k in (1, 2, 3) if (d['着'] == k).any()}
    w1, w2, w3 = top.get(1), top.get(2), top.get(3)
    um = pd.to_numeric(d['umaren'], errors='coerce').dropna()
    sp = pd.to_numeric(d['sanpuku'], errors='coerce').dropna()
    st = pd.to_numeric(d['santan'], errors='coerce').dropna()
    um = um.iloc[0] if len(um) else np.nan
    sp = sp.iloc[0] if len(sp) else np.nan
    st = st.iloc[0] if len(st) else np.nan

    rows = []

    def _add(name, pts, hit, pay):
        bet = pts * unit
        recv = (pay / 100.0 * unit) if (hit and pd.notna(pay)) else 0
        rows.append({'券種': name, '点数': pts, '投資': int(bet),
                     '払戻': int(recv), '的中': bool(hit and pd.notna(pay)),
                     '損益': int(recv - bet)})

    _add('◎ 単勝', 1, hb == w1, hon.get('tan_pay'))
    _add('◎ 複勝', 1, pd.notna(hon.get('fuku_pay')), hon.get('fuku_pay'))
    if star is not None:
        _add('★ 単勝', 1, star['馬番'] == w1, star.get('tan_pay'))
    if nA >= 1:
        hit = (hb in {w1, w2}) and ({w1, w2} - {hb}).issubset(aset)
        _add('馬連（◎-相手）', nA, hit, um)
    if nA >= 2:
        hit3 = (hb in {w1, w2, w3}) and ({w1, w2, w3} - {hb}).issubset(aset)
        _add('三連複（◎-相手）', nA * (nA - 1) // 2, hit3, sp)
        hitt = (hb == w1) and (w2 in aset) and (w3 in aset)
        _add('三連単（◎1着-相手）', nA * (nA - 1), hitt, st)
    return rows
