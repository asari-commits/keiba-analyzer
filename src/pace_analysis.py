"""
ペース・ラップ分析モジュール。

master.csv の PCI / RPCI / 上り3F / 上3F地点差 を使い、
「コース別ペース傾向」と「馬別ラップ適性」を算出する。

PCI（ペースチェンジインデックス）の読み方:
  < 46     : 強前傾（ハイペース・消耗戦）
  46 〜 48 : 前傾
  48 〜 52 : 平均的
  52 〜 54 : 後傾
  > 54     : 強後傾（スロー・瞬発力勝負）

このモジュールはキャッシュ付きで master.csv を一度だけ読み込む。
"""
from __future__ import annotations
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

MASTER_CSV = Path(__file__).parent.parent / "data" / "processed" / "master.csv"

VENUE_MAP = {
    '東': '東京', '中': '中山', '京': '京都', '阪': '阪神',
    '名': '中京', '小': '小倉', '新': '新潟', '福': '福島',
    '函': '函館', '札': '札幌',
}

# PCI 区分ラベル
def pci_label(pci: float) -> str:
    if pd.isna(pci):
        return '不明'
    if pci < 46:
        return '強前傾⚡'
    if pci < 48:
        return '前傾↗'
    if pci < 52:
        return '平均'
    if pci < 54:
        return '後傾↘'
    return '強後傾🐢'

def pci_color(pci: float) -> str:
    if pd.isna(pci):
        return '#888'
    if pci < 46:
        return '#e74c3c'
    if pci < 48:
        return '#e67e22'
    if pci < 52:
        return '#f1c40f'
    if pci < 54:
        return '#2ecc71'
    return '#3498db'


@lru_cache(maxsize=1)
def _load_master() -> pd.DataFrame:
    if not MASTER_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(MASTER_CSV, encoding='utf-8-sig', low_memory=False)

    # 数値変換
    for col in ['PCI', 'RPCI', 'PCI3', 'Ave-3F', '上り3F', '上3F地点差',
                '人気', '距離', '頭数', '4角', '3角', '2角']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 着順_num（全角数字・カタカナ含む文字列対応）
    _tr = str.maketrans('０１２３４５６７８９', '0123456789')
    if '着順' in df.columns:
        df['着順_num'] = pd.to_numeric(
            df['着順'].astype(str).str.translate(_tr).str.extract(r'(\d+)')[0],
            errors='coerce'
        )

    # 場略称 → 場名
    if '開催' in df.columns:
        df['_venue_name'] = (df['開催'].astype(str)
                             .str.extract(r'\d([^\d]+)\d')[0]
                             .map(VENUE_MAP))

    # 距離カテゴリ
    if '距離' in df.columns:
        df['_dist_cat'] = pd.cut(
            df['距離'], bins=[0, 1400, 1800, 2200, 9999],
            labels=['短距離', 'マイル', '中距離', '長距離'])

    # 4角コーナー位置（数値化）
    for col in ['4角', '3角', '2角']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


# ─────────────────────────────────────────────
# 1. コース別ペース傾向
# ─────────────────────────────────────────────

def course_pace_profile(venue_name: str, is_turf: bool,
                        distance: int) -> dict:
    """
    コース（競馬場×芝ダ×距離帯）の過去ペース傾向を返す。

    Returns
    -------
    dict:
      avg_pci        : 平均PCI
      pci_label      : ペースラベル
      front_win_rate : 4角1〜3番手馬の勝率
      agari_win_rate : 上り3F最速馬の勝率（追い込み有利度）
      n_races        : サンプルレース数
      dist_cat       : 距離カテゴリ
      famous_races   : 代表レース名リスト（最大3件）
    """
    df = _load_master()
    if df.empty:
        return {}

    surf = '芝' if is_turf else 'ダ'
    dist_cat = _dist_cat_label(distance)

    filt = df[
        (df['_venue_name'] == venue_name) &
        (df['芝・ダ'] == surf) &
        (df['_dist_cat'] == dist_cat)
    ].copy()

    if len(filt) < 20:
        return {'n_races': len(filt), 'avg_pci': None}

    n_races = filt.groupby(['日付', '開催', 'Ｒ']).ngroups
    avg_pci  = filt['PCI'].mean()
    avg_rpci = filt['RPCI'].mean()

    # 4角先行馬（1〜3番手）の勝率
    winners = filt[filt['着順_num'] == 1].copy()
    if '4角' in filt.columns:
        total_races = n_races
        front_wins  = (winners['4角'] <= 3).sum()
        front_win_rate = front_wins / total_races if total_races > 0 else None
    else:
        front_win_rate = None

    # 上り3F最速馬の勝率（レースごとに上り最速を特定）
    if '上り3F' in filt.columns:
        race_grp = filt.groupby(['日付', '開催', 'Ｒ'])
        agari_win_count = 0
        race_count_a = 0
        for _, grp in race_grp:
            valid = grp['上り3F'].dropna()
            if len(valid) < 3:
                continue
            best_agari_idx = grp['上り3F'].idxmin()
            race_count_a += 1
            if best_agari_idx in grp.index and grp.loc[best_agari_idx, '着順_num'] == 1:
                agari_win_count += 1
        agari_win_rate = agari_win_count / race_count_a if race_count_a >= 20 else None
    else:
        agari_win_rate = None

    # 代表レース名
    if 'レース名' in filt.columns:
        famous = (filt[filt['着順_num'] == 1]['レース名']
                  .value_counts().head(3).index.tolist())
    else:
        famous = []

    return {
        'n_races':        n_races,
        'avg_pci':        round(avg_pci, 1) if pd.notna(avg_pci) else None,
        'avg_rpci':       round(avg_rpci, 1) if pd.notna(avg_rpci) else None,
        'pci_label':      pci_label(avg_pci),
        'pci_color':      pci_color(avg_pci),
        'front_win_rate': round(front_win_rate * 100, 1) if front_win_rate is not None else None,
        'agari_win_rate': round(agari_win_rate * 100, 1) if agari_win_rate is not None else None,
        'dist_cat':       dist_cat,
        'venue':          venue_name,
        'surf':           surf,
        'famous_races':   famous,
    }


# ─────────────────────────────────────────────
# 2. 馬別ラップ適性
# ─────────────────────────────────────────────

def horse_pace_aptitude(horse_name: str) -> dict:
    """
    馬の過去成績からペース・ラップ適性を分析。

    Returns
    -------
    dict:
      pref_pace      : 得意ペース ('前傾' / '後傾' / '問わない')
      front_win_rate : 前傾ラップ(PCI<49)での勝率
      back_win_rate  : 後傾ラップ(PCI>51)での勝率
      avg_agari      : 平均上り3F
      avg_agari_rank : 上り3Fのレース内平均順位（1=最速）
      avg_corner_pos : 平均4角位置（低いほど前）
      style          : 'まくり' / '先行' / '差し' / '追込'
      win_pci_avg    : 勝ったレースの平均PCI
      n_races        : 出走数
      n_wins         : 勝利数
    """
    df = _load_master()
    if df.empty:
        return {}

    horse_df = df[df['馬名'] == horse_name].copy() if '馬名' in df.columns else pd.DataFrame()
    if len(horse_df) < 3:
        return {'n_races': len(horse_df), 'pref_pace': '不明'}

    n_races = len(horse_df)
    wins    = horse_df[horse_df['着順_num'] == 1]
    n_wins  = len(wins)

    # 平均上り3F（レース内での順位も）
    avg_agari = horse_df['上り3F'].mean() if '上り3F' in horse_df.columns else None

    # 上り3F レース内順位（低いほど速い）
    agari_ranks = []
    if '上り3F' in df.columns and '馬名' in df.columns:
        for (d, k, r), grp in df.groupby(['日付', '開催', 'Ｒ']):
            if horse_name in grp['馬名'].values:
                horse_agari = grp[grp['馬名'] == horse_name]['上り3F'].iloc[0]
                if pd.notna(horse_agari):
                    rank_in_race = (grp['上り3F'].dropna() <= horse_agari).sum()
                    agari_ranks.append(rank_in_race)
    avg_agari_rank = np.mean(agari_ranks) if agari_ranks else None

    # 平均4角位置
    avg_corner = horse_df['4角'].mean() if '4角' in horse_df.columns else None

    # 脚質判定（平均4角位置×頭数）
    if '4角' in horse_df.columns and '頭数' in horse_df.columns:
        pos_ratio = (horse_df['4角'] / horse_df['頭数']).mean()
        if pos_ratio < 0.2:
            style = '逃げ・先行'
        elif pos_ratio < 0.4:
            style = '好位差し'
        elif pos_ratio < 0.65:
            style = '差し'
        else:
            style = '追込'
    else:
        style = '不明'

    # 前傾/後傾ペース別成績
    if 'PCI' in horse_df.columns:
        front_df = horse_df[horse_df['PCI'] < 49]
        back_df  = horse_df[horse_df['PCI'] > 51]

        def _wr(sub):
            n = len(sub)
            w = (sub['着順_num'] == 1).sum() if n > 0 else 0
            return round(w / n * 100, 1) if n >= 3 else None

        front_wr = _wr(front_df)
        back_wr  = _wr(back_df)

        if front_wr is not None and back_wr is not None:
            if front_wr >= back_wr * 1.5:
                pref = '前傾ラップ得意'
            elif back_wr >= front_wr * 1.5:
                pref = '後傾ラップ得意'
            else:
                pref = 'ペース不問'
        else:
            pref = '判定データ不足'

        win_pci_avg = wins['PCI'].mean() if len(wins) > 0 else None
    else:
        front_wr = back_wr = win_pci_avg = None
        pref = '不明'

    # 上り3F地点差（後方から差してきた距離の平均）
    avg_koma = horse_df['上3F地点差'].mean() if '上3F地点差' in horse_df.columns else None

    return {
        'n_races':        n_races,
        'n_wins':         n_wins,
        'pref_pace':      pref,
        'front_win_rate': front_wr,
        'back_win_rate':  back_wr,
        'avg_agari':      round(avg_agari, 1) if avg_agari is not None and pd.notna(avg_agari) else None,
        'avg_agari_rank': round(avg_agari_rank, 1) if avg_agari_rank is not None else None,
        'avg_corner_pos': round(avg_corner, 1) if avg_corner is not None and pd.notna(avg_corner) else None,
        'avg_koma':       round(avg_koma, 1) if avg_koma is not None and pd.notna(avg_koma) else None,
        'style':          style,
        'win_pci_avg':    round(win_pci_avg, 1) if win_pci_avg is not None and pd.notna(win_pci_avg) else None,
    }


# ─────────────────────────────────────────────
# 3. レース名別傾向（有馬記念など）
# ─────────────────────────────────────────────

def race_name_profile(race_name_keyword: str) -> dict:
    """
    レース名（部分一致）の過去傾向を分析。

    Returns
    -------
    dict: avg_pci, front_win_rate, agari_win_rate, avg_agari_winner, n_races
    """
    df = _load_master()
    if df.empty or 'レース名' not in df.columns:
        return {}

    filt = df[df['レース名'].astype(str).str.contains(race_name_keyword, na=False)]
    if len(filt) < 5:
        return {'n_races': len(filt)}

    winners    = filt[filt['着順_num'] == 1]
    n_races    = filt.groupby(['日付', '開催', 'Ｒ']).ngroups
    avg_pci    = filt['PCI'].mean()
    front_wins = (winners['4角'] <= 3).sum() if '4角' in winners.columns else None
    front_wr   = front_wins / n_races * 100 if (n_races > 0 and front_wins is not None) else None
    avg_agari_winner = winners['上り3F'].mean() if '上り3F' in winners.columns else None

    return {
        'n_races':           n_races,
        'avg_pci':           round(avg_pci, 1) if pd.notna(avg_pci) else None,
        'pci_label':         pci_label(avg_pci),
        'front_win_rate':    round(front_wr, 1) if front_wr is not None else None,
        'avg_agari_winner':  round(avg_agari_winner, 1) if avg_agari_winner is not None and pd.notna(avg_agari_winner) else None,
    }


# ─────────────────────────────────────────────
# 4. ペース適合スコア（馬×コース傾向）
# ─────────────────────────────────────────────

def pace_fit_score(horse_apt: dict, course_prof: dict) -> tuple[float, str]:
    """
    馬のペース適性とコースのペース傾向の適合度を返す。

    Returns: (score 0.0〜1.0, label)
    """
    if not horse_apt or not course_prof:
        return (0.5, '判定不可')

    pref  = horse_apt.get('pref_pace', '')
    label = course_prof.get('pci_label', '')
    fwr   = horse_apt.get('front_win_rate')
    bwr   = horse_apt.get('back_win_rate')
    avg_pci = course_prof.get('avg_pci')

    if avg_pci is None or fwr is None or bwr is None:
        return (0.5, '判定不可')

    # コースが前傾（avg_pci < 49）か後傾（>51）か
    if avg_pci < 49:
        # 前傾コース → 前傾得意馬ほど適合
        score = fwr / 100.0 if fwr else 0.3
        fit_label = '前傾コース適合◎' if fwr and fwr > 20 else '前傾コース適合△'
    elif avg_pci > 51:
        # 後傾コース → 後傾得意馬ほど適合
        score = bwr / 100.0 if bwr else 0.3
        fit_label = '後傾コース適合◎' if bwr and bwr > 20 else '後傾コース適合△'
    else:
        score = 0.5
        fit_label = 'ペース不問コース'

    return (round(min(score, 1.0), 3), fit_label)


def _dist_cat_label(distance: int) -> str:
    if distance <= 1400:
        return '短距離'
    if distance <= 1800:
        return 'マイル'
    if distance <= 2200:
        return '中距離'
    return '長距離'


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    print('=== 東京芝2000m ペース傾向 ===')
    p = course_pace_profile('東京', True, 2000)
    for k, v in p.items():
        print(f'  {k}: {v}')

    print('\n=== 有馬記念 過去傾向 ===')
    r = race_name_profile('有馬記念')
    for k, v in r.items():
        print(f'  {k}: {v}')
