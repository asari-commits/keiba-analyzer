"""
特徴量エンジニアリング。
「レース前に確定している情報のみ」を使う設計。

■ 使用OK（レース前に確定）
  - レース条件: 距離・コース・馬場・頭数・斤量
  - 前走の全結果: 着順・タイム・コーナー位置・上り・ペース
  - 馬・騎手・調教師の過去通算成績
  - 馬体重・馬齢・血統・キャリア
  - 人気（オッズ）

■ 使用NG（当該レース中/後でしか分からない）
  - 当該レースの4角通過順・上り3F・走破タイム・着差
  - koma_diff / agari_rank_in_race / time_diff_from_winner
  - PCI・RPCI・Ave-3F（当該レース分）

新データが増えたら add_feature_block() を末尾に追加するだけ。
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def _to_num(s: pd.Series) -> pd.Series:
    """全角数字→半角 + numeric変換"""
    table = str.maketrans('０１２３４５６７８９', '0123456789')
    return pd.to_numeric(
        s.astype(str).str.translate(table).str.strip(),
        errors='coerce'
    )

def _parse_kinryo(s: pd.Series) -> pd.Series:
    """'57☆' '50▲' ' 56 ' → 57.0"""
    return pd.to_numeric(
        s.astype(str).str.extract(r'(\d+\.?\d*)')[0],
        errors='coerce'
    )

def _raw_time_to_sec(s: pd.Series) -> pd.Series:
    """Target整数タイム: 1456 → 105.6秒"""
    def _conv(v):
        try:
            t = str(int(v))
            if len(t) == 4:
                return int(t[0]) * 60 + int(t[1:3]) + int(t[3]) / 10
            elif len(t) == 5:
                return int(t[0]) * 60 + int(t[1:3]) + int(t[3:]) / 10
        except Exception:
            return np.nan
    return s.apply(_conv)


# ---------------------------------------------------------------------------
# ブロック①: レース条件（当日確定情報）
# ---------------------------------------------------------------------------

def add_race_context(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out['is_turf']     = (out['芝・ダ'] == '芝').astype(int)
    out['baba_num']    = out['馬場状態'].map({'良': 0, '稍': 1, '重': 2, '不': 3})
    out['dist_num']    = pd.to_numeric(out['距離'], errors='coerce')
    out['dist_cat']    = pd.cut(
        out['dist_num'], bins=[0, 1400, 1800, 2200, 9999], labels=[0, 1, 2, 3]
    ).astype(float)
    out['horses_num']  = pd.to_numeric(out['頭数'], errors='coerce')
    out['kinryo_num']  = _parse_kinryo(out['斤量'])
    out['course_num']  = out['コース区分'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3}) \
                         if 'コース区分' in out.columns else np.nan

    return out


# ---------------------------------------------------------------------------
# ブロック②: 前走の戦術・ペース特徴量（レース前に確定）
# ---------------------------------------------------------------------------

def add_previous_race_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    前走データを使い、今走の条件変化・前走の走り方を特徴化。
    前走の4角位置・上り・PCIなどは当然レース前に確定している。
    """
    out = df.copy()

    # --- 前走結果 ---
    out['prev_chakujun']  = _to_num(out['前走着順'])
    out['prev_popular']   = _to_num(out['前走人気'])
    out['prev_agari']     = pd.to_numeric(out['前走上り3F'], errors='coerce')
    out['prev_time_sec']  = _raw_time_to_sec(out['前走走破タイム'])
    out['prev_chakusa_t'] = pd.to_numeric(out['前走着差タイム'], errors='coerce')  # 勝ち馬との差(秒)
    out['prev_umaban']    = pd.to_numeric(out['前走馬番'], errors='coerce')
    out['prev_horses']    = pd.to_numeric(out['前走頭数'], errors='coerce')

    # --- 前走の位置取り（コーナー通過順） ---
    out['prev_pos_2c'] = _to_num(out['前2角'])
    out['prev_pos_3c'] = _to_num(out['前3角'])
    out['prev_pos_4c'] = _to_num(out['前4角'])

    # 前走の追い込み度: 4角順位 - 着順（大きい＝後方から追い込んだ）
    out['prev_koma_diff'] = out['prev_pos_4c'] - out['prev_chakujun']

    # 前走の先行度: 4角順位 / 頭数（小さいほど前にいた）
    out['prev_pos_ratio'] = out['prev_pos_4c'] / out['prev_horses']

    # --- 今走との条件変化 ---
    out['dist_change']    = out['dist_num'] - pd.to_numeric(out['前距離'], errors='coerce')
    out['interval_weeks'] = pd.to_numeric(out['間隔'], errors='coerce')

    prev_baba = out['前走馬場状態'].map({'良': 0, '稍': 1, '重': 2, '不': 3})
    out['baba_change']   = out['baba_num'] - prev_baba
    out['track_changed'] = (out['芝・ダ'] != out['前芝・ダ']).astype(int) \
                           if '前芝・ダ' in out.columns else 0

    # 騎手変更フラグ
    out['jockey_changed'] = (out['替'].notna() & (out['替'].astype(str).str.strip() != '')) \
                             .astype(int) if '替' in out.columns else 0

    # 前走の斤量との差
    out['kinryo_change'] = out['kinryo_num'] - pd.to_numeric(out['前走斤量'], errors='coerce')

    return out


# ---------------------------------------------------------------------------
# ブロック③: 馬・騎手・調教師の過去通算成績（時系列リーク防止）
# ---------------------------------------------------------------------------

def add_historical_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    当該レースより「過去のレース結果のみ」から集計。
    ソート→逐次更新で厳密に未来情報を使わない。
    """
    out = df.sort_values('日付_dt').reset_index(drop=True)

    def _calc(group_col: str, prefix: str):
        """expanding shift で時系列リークなしに高速集計"""
        chaku = out['着順_num']
        key   = out[group_col]

        win  = (chaku == 1).astype(float).where(chaku.notna() & key.notna())
        fuku = (chaku <= 3).astype(float).where(chaku.notna() & key.notna())

        # groupby + cumsum/shift で「当該行より前の累積値」を取得
        g_total = key.groupby(key).cumcount()           # 同一キーの通し番号（0始まり）
        g_win   = win.groupby(key).cumsum().shift(1)
        g_fuku  = fuku.groupby(key).cumsum().shift(1)
        g_sum   = chaku.where(key.notna()).groupby(key).cumsum().shift(1)
        g_n     = key.groupby(key).cumcount()           # shift前の件数

        # 件数（shiftで1行ずらした後の累積数）
        g_n_shifted = g_n  # cumcount は 0,1,2... なので shift不要（それ自体が「今まで何件か」）

        df_stat = pd.DataFrame({
            f'{prefix}_win_rate':  (g_win  / g_n_shifted).where(g_n_shifted > 0),
            f'{prefix}_fuku_rate': (g_fuku / g_n_shifted).where(g_n_shifted > 0),
            f'{prefix}_avg_chaku': (g_sum  / g_n_shifted).where(g_n_shifted > 0),
            f'{prefix}_n_races':   g_n_shifted,
        }, index=out.index)

        return df_stat

    for col, pref in [('馬名', 'horse'), ('騎手', 'jockey'), ('調教師', 'trainer')]:
        if col in out.columns:
            print(f"  {col} 集計中...")
            out = pd.concat([out, _calc(col, pref)], axis=1)

    return out


# ---------------------------------------------------------------------------
# ブロック④: 過去2〜3走の安定性（ローリング成績）
# ---------------------------------------------------------------------------

def add_rolling_form(df: pd.DataFrame, windows: list[int] = [2, 3]) -> pd.DataFrame:
    """
    各馬の直近N走の平均上り3F・平均着差・平均着順を計算。
    groupby + shift のベクトル演算で実装（iterrows不使用）。

    データはすでに日付順ソート済みを前提とする。
    shift(1) で「1つ前の走」、shift(2) で「2つ前の走」を取得し、
    それらの平均・標準偏差を計算する。
    """
    out = df.copy()
    horse = out['馬名']

    # 馬ごとに時系列でshiftして過去走の値を取得
    # prev_agari / prev_chakujun は前走データから既に計算済み（shift=1相当）
    agari_s1  = out['prev_agari']                                            # 1走前
    chaku_s1  = out['prev_chakujun']
    chakusa_s1 = out['prev_chakusa_t']

    # 2走前: 馬グループ内でshift(2)
    agari_s2   = out.groupby(horse, sort=False)['prev_agari'].shift(1)       # 2走前
    chaku_s2   = out.groupby(horse, sort=False)['prev_chakujun'].shift(1)
    chakusa_s2 = out.groupby(horse, sort=False)['prev_chakusa_t'].shift(1)

    # 3走前: shift(3)
    agari_s3   = out.groupby(horse, sort=False)['prev_agari'].shift(2)       # 3走前
    chaku_s3   = out.groupby(horse, sort=False)['prev_chakujun'].shift(2)
    chakusa_s3 = out.groupby(horse, sort=False)['prev_chakusa_t'].shift(2)

    # 直近2走平均（1走前 + 2走前）
    out['agari_avg2']   = (agari_s1 + agari_s2) / 2
    out['agari_std2']   = pd.concat([agari_s1, agari_s2], axis=1).std(axis=1)
    out['chakusa_avg2'] = (chakusa_s1 + chakusa_s2) / 2
    out['chaku_avg2']   = (chaku_s1 + chaku_s2) / 2

    # 直近3走平均（1走前 + 2走前 + 3走前）
    out['agari_avg3']   = (agari_s1 + agari_s2 + agari_s3) / 3
    out['agari_std3']   = pd.concat([agari_s1, agari_s2, agari_s3], axis=1).std(axis=1)
    out['chakusa_avg3'] = (chakusa_s1 + chakusa_s2 + chakusa_s3) / 3
    out['chaku_avg3']   = (chaku_s1 + chaku_s2 + chaku_s3) / 3

    return out


# ---------------------------------------------------------------------------
# ブロック⑤: コース適性（芝/ダ × 距離帯 × 馬場）
# ---------------------------------------------------------------------------

def add_course_aptitude(df: pd.DataFrame) -> pd.DataFrame:
    """
    馬ごとに「同一コース条件（芝/ダ × 距離カテゴリ）での過去成績」を集計。
    groupby + cumsum + shift のベクトル演算で実装（iterrows不使用）。
    """
    out = df.copy()

    # 複合キー列を作成
    out['_course_key'] = (
        out['馬名'].astype(str) + '_'
        + out['is_turf'].astype(str) + '_'
        + out['dist_cat'].astype(str)
    )
    key    = out['_course_key']
    chaku  = out['着順_num']

    win  = (chaku == 1).astype(float).where(chaku.notna())
    fuku = (chaku <= 3).astype(float).where(chaku.notna())

    g_n    = key.groupby(key).cumcount()                    # 0,1,2,... = これまでの出走数
    g_win  = win.groupby(key).cumsum().shift(1)             # 累積勝利数（当該レース除く）
    g_fuku = fuku.groupby(key).cumsum().shift(1)
    g_sum  = chaku.where(key.notna()).groupby(key).cumsum().shift(1)

    out['course_win_rate']  = (g_win  / g_n).where(g_n > 0)
    out['course_fuku_rate'] = (g_fuku / g_n).where(g_n > 0)
    out['course_avg_chaku'] = (g_sum  / g_n).where(g_n > 0)
    out['course_n_races']   = g_n

    out = out.drop(columns=['_course_key'])
    return out


# ---------------------------------------------------------------------------
# ブロック⑥: 脚質 × コース適性
# ---------------------------------------------------------------------------

def add_style_course_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """
    馬の「脚質（先行か追い込みか）」とコース毎の「先行有利度」から
    相性スコアを計算する。

    脚質スコア（style_score）:
        前走4角位置 / 前走頭数 → 0=逃げ(先頭), 1=最後方(追い込み)
    ローリング脚質スコア（style_avg3）:
        直近3走のstyle_scoreの平均 → 馬の傾向を安定化

    コース先行有利度（pace_front_ratio）:
        過去にそのコース(会場×芝ダ×距離帯)で
        「4角1〜3番手から勝った割合」を時系列リーク防止で集計
        → 値が大きいほど先行有利なコース

    脚質コース適合度（style_course_fit）:
        先行有利コース → style_score が小さい馬（先行馬）にボーナス
        追い込み有利コース → style_score が大きい馬（追い込み馬）にボーナス
        fit = 1 - |style_avg3 - (1 - pace_front_ratio)|
        → 値が大きいほど相性良い（0〜1）
    """
    out = df.copy()

    # ── 脚質スコア（前走4角位置の比率）──────────────────────────────
    pos = out['prev_pos_4c']
    horses = out['prev_horses']
    out['style_score'] = (pos / horses).clip(0, 1)   # 0=逃げ 1=追い込み

    # 直近3走の脚質スコア平均
    horse = out['馬名']
    s1 = out['style_score']
    s2 = out.groupby(horse, sort=False)['style_score'].shift(1)
    s3 = out.groupby(horse, sort=False)['style_score'].shift(2)
    out['style_avg3'] = (s1.fillna(s2).fillna(s3) + s2.fillna(s1) + s3.fillna(s1)) / 3

    # ── コース先行有利度（時系列リーク防止）────────────────────────────
    # 勝ち馬（着順=1）が4角で上位3番手以内にいた割合を会場×芝ダ×距離帯で集計
    out['_venue'] = out['開催'].astype(str) if '開催' in out.columns else ''
    course_key = (
        out['_venue'] + '_'
        + out['is_turf'].astype(str) + '_'
        + out['dist_cat'].astype(str)
    )
    out['_ck'] = course_key

    # 4角先行で勝ったか: 着順=1 かつ 前走4角位置<=3 ではなく
    # 「このレースの着順=1馬の4角位置」が必要だが当該レース情報は4角しかない。
    # ここでは prev_pos_4c（前走の4角位置）が1〜3だった馬が1着だったケースで近似する。
    # より精度を上げるには当レース4角位置が必要だが現在はleakのためNG。
    # 代替: 「前走で先行（style_score < 0.3）だった馬の前走着順が1だった割合」
    chaku = out['着順_num']
    is_front_win = (out['style_score'] < 0.3) & (chaku == 1)
    is_any_win   = (chaku == 1)

    ck = out['_ck']
    fw_cum = is_front_win.astype(float).where(chaku.notna()).groupby(ck).cumsum().shift(1)
    aw_cum = is_any_win.astype(float).where(chaku.notna()).groupby(ck).cumsum().shift(1)
    g_n    = ck.groupby(ck).cumcount()

    out['pace_front_ratio'] = (fw_cum / aw_cum.clip(lower=1)).where(g_n > 5)

    # ── 脚質コース適合度 ────────────────────────────────────────────────
    # 先行有利コース(pace_front_ratio高い) → 先行馬(style_avg3低い)が有利
    # optimal_style = 1 - pace_front_ratio （追い込み有利なら1に近づく）
    optimal = 1 - out['pace_front_ratio'].fillna(0.5)
    out['style_course_fit'] = 1 - (out['style_avg3'] - optimal).abs()

    out = out.drop(columns=['_venue', '_ck'], errors='ignore')
    return out


# ---------------------------------------------------------------------------
# ブロック⑦: 馬体重・血統・属性
# ---------------------------------------------------------------------------

def add_horse_attributes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out['weight_num']        = pd.to_numeric(out['馬体重'],     errors='coerce')
    out['weight_change']     = pd.to_numeric(out['馬体重増減'], errors='coerce')
    out['weight_change_abs'] = out['weight_change'].abs()
    out['career_num']        = pd.to_numeric(out['キャリア'],   errors='coerce')
    out['sex_num']            = out['性別'].map({'牡': 0, '牝': 1, 'セ': 2})
    out['age_num']            = pd.to_numeric(out['年齢'],       errors='coerce')

    for col, new_col in [('種牡馬', 'sire_code'), ('母父馬', 'bms_code')]:
        if col in out.columns:
            out[new_col] = pd.Categorical(out[col]).codes

    return out


# ---------------------------------------------------------------------------
# ブロック⑧: PCIベース ペース特徴量（master.csvにPCI列がある場合のみ有効）
# ---------------------------------------------------------------------------

def add_pace_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    PCIと上り3Fから時系列リーク防止でペース系特徴量を追加。
    出馬表CSVのみ入力の場合は全列NaNになる（予測は通る）。
    """
    out = df.copy()

    # prev_pci: 前走のコースPCI（馬グループ内で1行shift）
    if 'PCI' in out.columns:
        out['_pci_tmp'] = pd.to_numeric(out['PCI'], errors='coerce')
        out['prev_pci'] = out.groupby('馬名', sort=False)['_pci_tmp'].shift(1)
        out = out.drop(columns=['_pci_tmp'])
    else:
        out['prev_pci'] = np.nan

    # agari_hist_avg: 馬ごと累積平均上り3F（当該レース除く）
    if '上り3F' in out.columns:
        out['_ag_tmp'] = pd.to_numeric(out['上り3F'], errors='coerce')
        g_n   = out.groupby('馬名', sort=False)['_ag_tmp'].cumcount()
        g_sum = out.groupby('馬名', sort=False)['_ag_tmp'].cumsum().shift(1)
        out['agari_hist_avg'] = (g_sum / g_n).where(g_n > 0)
        out = out.drop(columns=['_ag_tmp'])
    else:
        out['agari_hist_avg'] = np.nan

    # horse_front/back_pace_wr: 前傾(PCI<49)/後傾(PCI>51)別 累積勝率
    if 'PCI' in out.columns and '着順_num' in out.columns:
        pci  = pd.to_numeric(out['PCI'], errors='coerce')
        win  = (out['着順_num'] == 1).astype(float).where(out['着順_num'].notna())

        out['_fm'] = (pci < 49).astype(float).where(pci.notna())   # front mask
        out['_bm'] = (pci > 51).astype(float).where(pci.notna())   # back mask
        out['_fw'] = (win * out['_fm']).fillna(0)
        out['_bw'] = (win * out['_bm']).fillna(0)

        g_fc = out.groupby('馬名', sort=False)['_fm'].cumsum().shift(1)
        g_fw = out.groupby('馬名', sort=False)['_fw'].cumsum().shift(1)
        g_bc = out.groupby('馬名', sort=False)['_bm'].cumsum().shift(1)
        g_bw = out.groupby('馬名', sort=False)['_bw'].cumsum().shift(1)

        out['horse_front_pace_wr'] = (g_fw / g_fc.clip(lower=1)).where(g_fc >= 3)
        out['horse_back_pace_wr']  = (g_bw / g_bc.clip(lower=1)).where(g_bc >= 3)
        out = out.drop(columns=['_fm', '_bm', '_fw', '_bw'])
    else:
        out['horse_front_pace_wr'] = np.nan
        out['horse_back_pace_wr']  = np.nan

    # course_pci_mean: コース別（場×芝ダ×距離帯）過去平均PCI（leakなし）
    if 'PCI' in out.columns:
        out['_pci2'] = pd.to_numeric(out['PCI'], errors='coerce')
        _turf = out.get('is_turf', pd.Series(0, index=out.index))
        _dcat = out.get('dist_cat', pd.Series(0, index=out.index))
        out['_ck2'] = (out['開催'].astype(str) + '_' +
                       _turf.astype(str) + '_' + _dcat.astype(str))
        g_cn = out.groupby('_ck2', sort=False)['_pci2'].cumcount()
        g_cs = out.groupby('_ck2', sort=False)['_pci2'].cumsum().shift(1)
        out['course_pci_mean'] = (g_cs / g_cn).where(g_cn >= 5)
        out = out.drop(columns=['_pci2', '_ck2'])
    else:
        out['course_pci_mean'] = np.nan

    return out


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

# レース前に確定している特徴量のみ
FEATURE_COLS = [
    # レース条件
    'is_turf', 'baba_num', 'dist_num', 'dist_cat',
    'horses_num', 'kinryo_num', 'course_num',
    # 前走の結果・戦術
    'prev_chakujun', 'prev_popular', 'prev_agari',
    'prev_time_sec', 'prev_chakusa_t',
    'prev_pos_2c', 'prev_pos_3c', 'prev_pos_4c',
    'prev_koma_diff', 'prev_pos_ratio',
    # 条件変化
    'dist_change', 'interval_weeks', 'baba_change',
    'track_changed', 'jockey_changed', 'kinryo_change',
    # 馬・騎手・調教師 通算成績
    'horse_win_rate', 'horse_fuku_rate', 'horse_avg_chaku', 'horse_n_races',
    'jockey_win_rate', 'jockey_fuku_rate', 'jockey_avg_chaku', 'jockey_n_races',
    'trainer_win_rate', 'trainer_fuku_rate', 'trainer_avg_chaku',
    # 直近2走の安定性
    'agari_avg2', 'agari_std2', 'chakusa_avg2', 'chaku_avg2',
    # 直近3走の安定性
    'agari_avg3', 'agari_std3', 'chakusa_avg3', 'chaku_avg3',
    # コース適性（芝/ダ × 距離帯）
    'course_win_rate', 'course_fuku_rate', 'course_avg_chaku', 'course_n_races',
    # 脚質 × コース適性
    'style_score', 'style_avg3',
    'pace_front_ratio', 'style_course_fit',
    # 馬属性
    'weight_num', 'weight_change', 'weight_change_abs',
    'career_num', 'sex_num', 'age_num', 'sire_code', 'bms_code',
    # ペース適性（PCI系・master.csvがある場合のみ有効、なければNaN→0）
    'prev_pci', 'agari_hist_avg', 'horse_front_pace_wr', 'horse_back_pace_wr', 'course_pci_mean',
    # 人気（オッズ情報）
    '人気',
]

TARGET_COL = '着順_num'

META_COLS = ['日付_dt', '日付', '開催', 'Ｒ', '馬番', '馬名', '騎手', '着順_num', '人気']


def build_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    if verbose: print("[1/6] レース条件...")
    df = add_race_context(df)

    if verbose: print("[2/6] 前走特徴量...")
    df = add_previous_race_features(df)

    if verbose: print("[3/6] 馬・騎手・調教師 通算成績（時系列リーク防止）...")
    df = add_historical_stats(df)

    if verbose: print("[4/6] 直近2〜3走の安定性...")
    df = add_rolling_form(df, windows=[2, 3])

    if verbose: print("[5/6] コース適性（芝/ダ × 距離帯）...")
    df = add_course_aptitude(df)

    if verbose: print("[6/7] 脚質 × コース適性...")
    df = add_style_course_interaction(df)

    if verbose: print("[7/8] 馬属性・血統...")
    df = add_horse_attributes(df)

    if verbose: print("[8/8] PCIペース特徴量...")
    try:
        df = add_pace_features(df)
    except Exception as _e:
        if verbose:
            print(f"  ペース特徴量でエラー（スキップ）: {_e}")

    # FEATURE_COLS に含まれるが df にない列は全てNaNで補完する
    # （ペース特徴量や出馬表CSVで取得できない列の保証）
    for _fc in FEATURE_COLS:
        if _fc not in df.columns:
            df[_fc] = np.nan

    use_cols  = [c for c in FEATURE_COLS if c in df.columns]
    meta_cols = [c for c in META_COLS if c in df.columns]
    extra     = [c for c in use_cols if c not in meta_cols]
    result    = df[meta_cols + extra].copy()

    if verbose:
        print(f"\n特徴量数: {len(use_cols)}")
        missing = {c: float(result[c].isna().mean()) for c in use_cols if c in result.columns}
        high = {k: f"{v:.1%}" for k, v in missing.items() if v > 0.3}
        if high:
            print(f"欠損率30%超:\n" + "\n".join(f"  {k}: {v}" for k, v in high.items()))

    return result
