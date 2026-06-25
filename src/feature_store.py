"""
特徴量ストア（事前計算キャッシュ）— ⚠️ 実験中（WIP・本番未接続）。

【状態 2026-06-25】予測 0.5 秒・ストア構築 9.5 秒まで高速化を実証したが、
フル再計算との一致検証で一部の特徴量（agari_hist_avg / course_pci_mean 等）に
残差があり、1レース予測でも本命が一致しないケースがある（4R中1R）。
精度一致を詰めるまで pipeline_target からは呼び出さない（予測結果を変えないため）。
継続タスク: 残差列の原因特定 → フル再計算と本命100%一致を確認 → 接続。

予測のたびに過去 48 万行へ build_features を掛け直すと数十秒＋大量メモリ
（クラウドOOMの原因）になる。本モジュールは「馬個体に依存しない集計」
（騎手・調教師・血統・コース先行有利度・コースPCI・タイム偏差）を master から
1 回だけ集計してストア化し、予測時はそれを参照して上書きする。

馬名グループに紐づく特徴量（馬通算・コース適性・近走・脚質・条件別など）は、
予測時に「出走馬の master 履歴だけ」を抽出して build_features を掛ければ
完全一致で得られる（馬名グループ集計は他馬の行に影響されないため）。

→ pipeline_target.predict_* がこのストアを使うことで、48 万行の再計算を排除する。
"""
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from features import (add_race_context, add_previous_race_features,
                      MIN_PED_N, MIN_JOCKEY_N)

STORE_PATH = Path(__file__).parent.parent / "data" / "processed" / "feature_store.pkl"

# master から本ストア構築に必要な列（current 4角は pipeline では未使用＝pace_excuse は 0）
STORE_NEEDED_COLS = [
    '日付_dt', '着順_num', '騎手', '調教師', '種牡馬', '母父馬',
    '芝・ダ', '馬場状態', '距離', '頭数', '斤量', '開催', '馬名',
    '前4角', '前走頭数', 'PCI', '走破秒',
]


def _prep(master: pd.DataFrame) -> pd.DataFrame:
    """キー生成と style_score に必要な行ローカル列を付与する。"""
    df = master.copy()
    df = add_race_context(df)            # is_turf, baba_num, dist_num, dist_cat
    df = add_previous_race_features(df)  # prev_pos_4c, prev_horses, prev_pos_ratio
    df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
    return df


def _overall_stats(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """add_historical_stats の「新規行（=master の次の行）」値を groupby で再現。
       win_rate=Σ(着順==1)/N, fuku_rate=Σ(着順<=3)/N, avg_chaku=Σ着順/N, n=N(全行)。"""
    chaku = df['着順_num']
    g = pd.DataFrame({
        'k':   df[col],
        'win': (chaku == 1).astype(float),
        'fk':  (chaku <= 3).astype(float),
        'ch':  chaku,
    })
    agg = g.groupby('k').agg(n=('k', 'size'), win=('win', 'sum'),
                             fk=('fk', 'sum'), ch=('ch', 'sum'))
    out = pd.DataFrame(index=agg.index)
    out['win_rate']  = (agg['win'] / agg['n']).where(agg['n'] > 0)
    out['fuku_rate'] = (agg['fk']  / agg['n']).where(agg['n'] > 0)
    out['avg_chaku'] = (agg['ch']  / agg['n']).where(agg['n'] > 0)
    out['n_races']   = agg['n']
    return out


def _cond_fuku(df: pd.DataFrame, key: pd.Series, min_n: int) -> pd.Series:
    """add_pedigree/jockey_aptitude の条件別複勝率（新規行値）。
       rate = Σ(着順<=3) / Σ(着順notna) , 分母 >= min_n のみ有効。"""
    chaku = df['着順_num']
    valid = chaku.notna().astype(float)
    fuku  = (chaku <= 3).astype(float).fillna(0.0)
    t = pd.DataFrame({'k': key.values, 'v': valid.values, 'f': fuku.values})
    agg = t.groupby('k').agg(n=('v', 'sum'), f=('f', 'sum'))
    return (agg['f'] / agg['n']).where(agg['n'] >= min_n)


def build_store(master: pd.DataFrame) -> dict:
    """master から馬個体非依存の集計ストアを構築する。"""
    df = _prep(master)
    chaku = df['着順_num']

    is_turf  = df['is_turf'].fillna(0).astype(int)
    dist_cat = df['dist_cat'].fillna(-1).astype(float).astype(int)
    dist_num = df['dist_num'].fillna(-1).astype(float).astype(int)
    baba     = df['baba_num'].fillna(0)
    venue    = df['開催'].astype(str)

    store: dict = {'_meta': {'n_master': int(len(df)),
                             'max_date': str(df['日付_dt'].max())}}

    # ── 騎手・調教師 通算成績 ─────────────────────────────────
    store['jockey']  = _overall_stats(df, '騎手')
    store['trainer'] = _overall_stats(df, '調教師')

    # ── 騎手 条件別複勝率（芝ダ / 距離帯）──────────────────────
    jk = df['騎手'].astype(str)
    store['jockey_surf'] = _cond_fuku(df, jk + '_s' + is_turf.astype(str), MIN_JOCKEY_N)
    store['jockey_dist'] = _cond_fuku(df, jk + '_d' + dist_cat.astype(str), MIN_JOCKEY_N)

    # ── 血統（種牡馬・母父）条件別複勝率 ───────────────────────
    valid = chaku.notna().astype(float)
    fuku  = (chaku <= 3).astype(float).fillna(0.0)
    wet   = (baba >= 1).astype(float)
    for ped_col, tag in [('種牡馬', 'sire'), ('母父馬', 'bms')]:
        ped = df[ped_col].astype(str)
        store[f'{tag}_surf'] = _cond_fuku(df, ped + '_s' + is_turf.astype(str), MIN_PED_N)
        store[f'{tag}_dist'] = _cond_fuku(df, ped + '_d' + dist_cat.astype(str), MIN_PED_N)
        store[f'{tag}_sd']   = _cond_fuku(
            df, ped + '_s' + is_turf.astype(str) + '_d' + dist_cat.astype(str), MIN_PED_N)
        # wet（稍重以上）別: 種牡馬ごとに wet 行のみで集計
        tw = pd.DataFrame({'k': ped.values,
                           'nw': (wet * valid).values,
                           'fw': (fuku * wet).values})
        aggw = tw.groupby('k').agg(nw=('nw', 'sum'), fw=('fw', 'sum'))
        store[f'{tag}_wet'] = (aggw['fw'] / aggw['nw']).where(aggw['nw'] >= MIN_PED_N)

    # ── コース先行有利度 pace_front_ratio（会場×芝ダ×距離）──────
    style_score = (df['prev_pos_4c'] / df['prev_horses']).clip(0, 1)
    ck = venue + '_' + is_turf.astype(str) + '_' + dist_num.astype(str)
    is_front_win = ((style_score < 0.3) & (chaku == 1)).astype(float).where(chaku.notna())
    is_any_win   = (chaku == 1).astype(float).where(chaku.notna())
    tp = pd.DataFrame({'k': ck.values, 'fw': is_front_win.values,
                       'aw': is_any_win.values})
    aggp = tp.groupby('k').agg(n=('k', 'size'), fw=('fw', 'sum'), aw=('aw', 'sum'))
    store['pace_front'] = (aggp['fw'] / aggp['aw'].clip(lower=1)).where(aggp['n'] > 5)

    # ── コース平均PCI course_pci_mean（会場×芝ダ×距離、分母=全行数）──
    pci = pd.to_numeric(df['PCI'], errors='coerce') if 'PCI' in df.columns else pd.Series(np.nan, index=df.index)
    tc = pd.DataFrame({'k': ck.values, 'p': pci.values})
    aggc = tc.groupby('k').agg(n=('k', 'size'), s=('p', 'sum'))
    store['course_pci'] = (aggc['s'] / aggc['n']).where(aggc['n'] >= 5)

    # ── 馬ごと走破タイム偏差 horse_time_dev_avg ───────────────
    # course_time_avg(会場非依存・芝ダ×距離) を as-of で求め、time_dev=走破秒-avg、
    # それを馬ごとに累積平均（新規行値=Σ/有効数）。
    if '走破秒' in df.columns:
        df2 = df.sort_values('日付_dt').reset_index(drop=True)
        time_s = pd.to_numeric(df2['走破秒'], errors='coerce')
        it2 = df2['is_turf'].fillna(0).astype(int)
        dn2 = df2['dist_num'].fillna(-1).astype(float).astype(int)
        ck2 = it2.astype(str) + '_' + dn2.astype(str)
        _v = time_s.notna().astype(float)
        _t = time_s.fillna(0.0)
        ck_n   = _v.groupby(ck2).cumsum().shift(1)
        ck_sum = (_t * _v).groupby(ck2).cumsum().shift(1)
        course_time_avg = (ck_sum / ck_n).where(ck_n > 0)
        time_dev = time_s - course_time_avg
        td = pd.DataFrame({'k': df2['馬名'].values,
                           'vd': time_dev.notna().astype(float).values,
                           'td': (time_dev.fillna(0.0) * time_dev.notna()).values})
        aggt = td.groupby('k').agg(n=('vd', 'sum'), s=('td', 'sum'))
        store['horse_time_dev'] = (aggt['s'] / aggt['n']).where(aggt['n'] > 0)
    else:
        store['horse_time_dev'] = pd.Series(dtype=float)

    return store


def save_store(store: dict, path: Path = STORE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(store, f)


def load_store(path: Path = STORE_PATH) -> dict | None:
    if not Path(path).exists():
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


# 上書きする（＝馬個体非依存で部分集合build_featuresでは正しく出ない）列
OVERRIDE_COLS = [
    'jockey_win_rate', 'jockey_fuku_rate', 'jockey_avg_chaku', 'jockey_n_races',
    'trainer_win_rate', 'trainer_fuku_rate', 'trainer_avg_chaku',
    'jockey_surf_fuku', 'jockey_dist_fuku',
    'sire_surf_fuku', 'sire_dist_fuku', 'sire_sd_fuku', 'sire_wet_fuku',
    'bms_surf_fuku', 'bms_dist_fuku', 'bms_sd_fuku', 'bms_wet_fuku',
    'pace_front_ratio', 'course_pci_mean', 'horse_time_dev_avg',
    'pace_excuse_senko', 'pace_excuse_sashi', 'senko_revenge_fit', 'sashi_revenge_fit',
    'style_course_fit',
]


def apply_store(feat: pd.DataFrame, new_df: pd.DataFrame, store: dict) -> pd.DataFrame:
    """部分集合build_featuresで得た新規行 feat に、ストア由来の集計列を上書きする。
       feat は date ソートで並び替わるため、new_df のキー列を行キーで結合して整列する。"""
    out = feat.reset_index(drop=True).copy()

    # ── new_df のキー列を行キー(開催・Ｒ・馬番・馬名)で feat に整列結合 ──
    nd = new_df.copy()
    nd['馬番'] = pd.to_numeric(nd['馬番'], errors='coerce')
    out['馬番'] = pd.to_numeric(out['馬番'], errors='coerce')
    key_cols = ['開催', 'Ｒ', '馬番', '馬名']
    carry = [c for c in ['調教師', '種牡馬', '母父馬'] if c in nd.columns]
    keydf = nd[key_cols + carry].drop_duplicates(key_cols)
    out = out.merge(keydf, on=key_cols, how='left')

    # キー生成（feat 側の is_turf/dist_cat/dist_num + 整列済みキー列）
    is_turf  = out['is_turf'].fillna(0).astype(int).astype(str)
    dist_cat = out['dist_cat'].fillna(-1).astype(float).astype(int).astype(str)
    dist_num = out['dist_num'].fillna(-1).astype(float).astype(int).astype(str)
    venue    = out['開催'].astype(str)

    def _col(name):
        return out[name].astype(str) if name in out.columns else pd.Series('', index=out.index)

    def _map(series_store, keys):
        return pd.Series(keys, index=out.index).map(series_store)

    # 騎手・調教師 通算（騎手は feat の META に存在、調教師は結合済み）
    for col, pref, has_n in [('騎手', 'jockey', True), ('調教師', 'trainer', False)]:
        tbl = store[pref]
        k = _col(col)
        out[f'{pref}_win_rate']  = _map(tbl['win_rate'],  k.values)
        out[f'{pref}_fuku_rate'] = _map(tbl['fuku_rate'], k.values)
        out[f'{pref}_avg_chaku'] = _map(tbl['avg_chaku'], k.values)
        if has_n:
            out[f'{pref}_n_races'] = _map(tbl['n_races'], k.values)

    # 騎手 条件別
    jk = _col('騎手')
    out['jockey_surf_fuku'] = _map(store['jockey_surf'], (jk + '_s' + is_turf).values)
    out['jockey_dist_fuku'] = _map(store['jockey_dist'], (jk + '_d' + dist_cat).values)

    # 血統
    for ped_col, tag in [('種牡馬', 'sire'), ('母父馬', 'bms')]:
        ped = _col(ped_col)
        out[f'{tag}_surf_fuku'] = _map(store[f'{tag}_surf'], (ped + '_s' + is_turf).values)
        out[f'{tag}_dist_fuku'] = _map(store[f'{tag}_dist'], (ped + '_d' + dist_cat).values)
        out[f'{tag}_sd_fuku']   = _map(store[f'{tag}_sd'],   (ped + '_s' + is_turf + '_d' + dist_cat).values)
        out[f'{tag}_wet_fuku']  = _map(store[f'{tag}_wet'],  ped.values)

    # コース先行有利度・PCI
    ckey = (venue + '_' + is_turf + '_' + dist_num).values
    out['pace_front_ratio'] = _map(store['pace_front'], ckey)
    out['course_pci_mean']  = _map(store['course_pci'], ckey)

    # 馬タイム偏差
    out['horse_time_dev_avg'] = _map(store['horse_time_dev'], _col('馬名').values)

    # 展開不利度: pipeline は current「4角」未ロードのため常に 0（baseline と一致）
    for c in ['pace_excuse_senko', 'pace_excuse_sashi', 'senko_revenge_fit', 'sashi_revenge_fit']:
        out[c] = 0.0

    # 脚質コース適合度（style_avg3 は feat 側にある。pace_front_ratio は上書き済み）
    optimal = 1 - out['pace_front_ratio'].fillna(0.5)
    out['style_course_fit'] = 1 - (out['style_avg3'] - optimal).abs()

    return out
