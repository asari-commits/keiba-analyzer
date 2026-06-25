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


def _prior_sum(s: pd.Series, key: pd.Series) -> pd.Series:
    """グループ(key)内で『当該行より前』の累積合計を返す（時系列リーク防止）。

    旧実装 `s.groupby(key).cumsum().shift(1)` は .shift(1) がグループ境界を跨いで
    前の行を参照するため、特徴量が「DataFrame全体の行順」に依存していた
    （＝出走馬だけの部分集合では値が変わり事前計算と一致しない原因）。
    本関数はグループ内のみで完結し行順に依存しない。欠損は0として
    合計に寄与させない（旧 cumsum の skipna と同じ意味）。
    """
    f = s.fillna(0.0)
    return f.groupby(key).cumsum() - f


# ---------------------------------------------------------------------------
# ブロック①: レース条件（当日確定情報）
# ---------------------------------------------------------------------------

def add_race_context(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out['is_turf']     = (out['芝・ダ'].str.startswith('芝').fillna(False)).astype(int) \
                         if '芝・ダ' in out.columns else 0
    _baba = out['馬場状態'] if '馬場状態' in out.columns else pd.Series('良', index=out.index)
    out['baba_num']    = _baba.map({'良': 0, '稍': 1, '重': 2, '不': 3}).fillna(0)
    out['dist_num']    = pd.to_numeric(out['距離'], errors='coerce') \
                         if '距離' in out.columns else np.nan
    out['dist_cat']    = pd.cut(
        out['dist_num'], bins=[0, 1400, 1800, 2200, 9999], labels=[0, 1, 2, 3]
    ).astype(float)
    out['horses_num']  = pd.to_numeric(out['頭数'], errors='coerce') \
                         if '頭数' in out.columns else np.nan
    out['kinryo_num']  = _parse_kinryo(out['斤量']) if '斤量' in out.columns else np.nan
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

    def _col(name, default=np.nan):
        return out[name] if name in out.columns else pd.Series(default, index=out.index)

    # --- 前走結果 ---
    out['prev_chakujun']  = _to_num(_col('前走着順'))
    out['prev_popular']   = _to_num(_col('前走人気'))
    out['prev_agari']     = pd.to_numeric(_col('前走上り3F'), errors='coerce')
    out['prev_time_sec']  = _raw_time_to_sec(_col('前走走破タイム'))
    out['prev_chakusa_t'] = pd.to_numeric(_col('前走着差タイム'), errors='coerce')
    out['prev_umaban']    = pd.to_numeric(_col('前走馬番'), errors='coerce')
    out['prev_horses']    = pd.to_numeric(_col('前走頭数'), errors='coerce')

    # --- 前走の位置取り（コーナー通過順） ---
    out['prev_pos_2c'] = _to_num(_col('前2角'))
    out['prev_pos_3c'] = _to_num(_col('前3角'))
    out['prev_pos_4c'] = _to_num(_col('前4角'))

    # 前走の追い込み度: 4角順位 - 着順（大きい＝後方から追い込んだ）
    out['prev_koma_diff'] = out['prev_pos_4c'] - out['prev_chakujun']

    # 前走の先行度: 4角順位 / 頭数（小さいほど前にいた）
    out['prev_pos_ratio'] = out['prev_pos_4c'] / out['prev_horses']

    # --- 今走との条件変化 ---
    out['dist_change']    = out['dist_num'] - pd.to_numeric(_col('前距離'), errors='coerce')
    out['interval_weeks'] = pd.to_numeric(_col('間隔'), errors='coerce')

    prev_baba = _col('前走馬場状態').map({'良': 0, '稍': 1, '重': 2, '不': 3})
    out['baba_change']   = out['baba_num'] - prev_baba
    out['track_changed'] = (out['芝・ダ'] != _col('前芝・ダ')).astype(int) \
                           if '前芝・ダ' in out.columns else 0

    # 騎手変更フラグ
    out['jockey_changed'] = (out['替'].notna() & (out['替'].astype(str).str.strip() != '')) \
                             .astype(int) if '替' in out.columns else 0

    # 前走の斤量との差
    out['kinryo_change'] = out['kinryo_num'] - pd.to_numeric(_col('前走斤量'), errors='coerce')

    # 前走の先頭との上り3F地点差（master: 上3F地点差 を馬グループ内 shift で取得）
    if '上3F地点差' in out.columns:
        horse = out['馬名']
        out['prev_top3f_diff'] = out.groupby(horse, sort=False)['上3F地点差'].shift(1)
    else:
        out['prev_top3f_diff'] = np.nan

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

    # 着順_num を確実にfloat64にする（apply返値がobjectになる場合の対策）
    if '着順_num' in out.columns:
        out['着順_num'] = pd.to_numeric(out['着順_num'], errors='coerce')

    def _calc(group_col: str, prefix: str):
        """expanding shift で時系列リークなしに高速集計"""
        chaku = out['着順_num']
        key   = out[group_col]

        win  = (chaku == 1).astype(float).where(chaku.notna() & key.notna())
        fuku = (chaku <= 3).astype(float).where(chaku.notna() & key.notna())

        # グループ内「当該行より前の累積値」（順序非依存・リーク防止）
        g_total = key.groupby(key).cumcount()           # 同一キーの通し番号（0始まり）
        g_win   = _prior_sum(win, key)
        g_fuku  = _prior_sum(fuku, key)
        g_sum   = _prior_sum(chaku.where(key.notna()), key)
        g_n     = key.groupby(key).cumcount()           # これまでの件数

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
    各馬の直近N走の平均上り3F・平均着差・平均着順を計算（最大5走）。
    groupby + shift のベクトル演算で実装（iterrows不使用）。

    データはすでに日付順ソート済みを前提とする。
    prev_agari / prev_chakujun 列 (shift=1相当) をベースに
    さらに馬グループ内でshiftして2〜5走前の値を取得する。
    """
    out = df.copy()
    horse = out['馬名']

    # 1走前（Target前走CSVまたはmaster内の「前走○○」列）
    agari_s1   = out['prev_agari']
    chaku_s1   = out['prev_chakujun']
    chakusa_s1 = out['prev_chakusa_t']
    pos4c_s1   = out['prev_pos_4c'] if 'prev_pos_4c' in out.columns else pd.Series(np.nan, index=out.index)

    # 2〜5走前: 馬グループ内でprev_*列をshift
    def _sh(col, n):
        return out.groupby(horse, sort=False)[col].shift(n)

    agari_s2, agari_s3, agari_s4, agari_s5 = (
        _sh('prev_agari', n) for n in [1, 2, 3, 4])
    chaku_s2, chaku_s3, chaku_s4, chaku_s5 = (
        _sh('prev_chakujun', n) for n in [1, 2, 3, 4])
    chakusa_s2, chakusa_s3, chakusa_s4, chakusa_s5 = (
        _sh('prev_chakusa_t', n) for n in [1, 2, 3, 4])
    pos4c_s2, pos4c_s3 = _sh('prev_pos_4c', 1), _sh('prev_pos_4c', 2)

    # 直近2走平均
    out['agari_avg2']   = (agari_s1 + agari_s2) / 2
    out['agari_std2']   = pd.concat([agari_s1, agari_s2], axis=1).std(axis=1)
    out['chakusa_avg2'] = (chakusa_s1 + chakusa_s2) / 2
    out['chaku_avg2']   = (chaku_s1 + chaku_s2) / 2

    # 直近3走平均
    out['agari_avg3']   = (agari_s1 + agari_s2 + agari_s3) / 3
    out['agari_std3']   = pd.concat([agari_s1, agari_s2, agari_s3], axis=1).std(axis=1)
    out['chakusa_avg3'] = (chakusa_s1 + chakusa_s2 + chakusa_s3) / 3
    out['chaku_avg3']   = (chaku_s1 + chaku_s2 + chaku_s3) / 3

    # 直近5走平均（欠損は NaN のまま mean で除外）
    _agari5   = pd.concat([agari_s1, agari_s2, agari_s3, agari_s4, agari_s5], axis=1)
    _chaku5   = pd.concat([chaku_s1, chaku_s2, chaku_s3, chaku_s4, chaku_s5], axis=1)
    _chakusa5 = pd.concat([chakusa_s1, chakusa_s2, chakusa_s3, chakusa_s4, chakusa_s5], axis=1)
    out['agari_avg5']   = _agari5.mean(axis=1)
    out['chaku_avg5']   = _chaku5.mean(axis=1)
    out['chakusa_avg5'] = _chakusa5.mean(axis=1)
    out['agari_std5']   = _agari5.std(axis=1)

    # 着順トレンド: 1走前 − 5走前（負=近走改善, 正=近走悪化）
    out['chaku_trend5'] = chaku_s1 - chaku_s5

    # 3走平均4角位置（脚質安定性）・位置変化
    out['pos4c_avg3']   = (pos4c_s1 + pos4c_s2 + pos4c_s3) / 3
    out['pos4c_trend3'] = pos4c_s1 - pos4c_s3   # 正=後方化, 負=先行化

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
    if '着順_num' in out.columns:
        out['着順_num'] = pd.to_numeric(out['着順_num'], errors='coerce')

    # 複合キー列を作成（距離は正確な値で一致、初距離はNaNとしてモデルに渡す）
    out['_course_key'] = (
        out['馬名'].astype(str) + '_'
        + out['is_turf'].astype(str) + '_'
        + out['dist_num'].fillna(-1).astype(float).astype(int).astype(str)
    )
    key    = out['_course_key']
    chaku  = out['着順_num']

    win  = (chaku == 1).astype(float).where(chaku.notna())
    fuku = (chaku <= 3).astype(float).where(chaku.notna())

    g_n    = key.groupby(key).cumcount()                    # 0,1,2,... = これまでの出走数
    g_win  = _prior_sum(win, key)                           # 累積勝利数（当該レース除く）
    g_fuku = _prior_sum(fuku, key)
    g_sum  = _prior_sum(chaku.where(key.notna()), key)

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
    if '着順_num' in out.columns:
        out['着順_num'] = pd.to_numeric(out['着順_num'], errors='coerce')

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
        + out['dist_num'].fillna(-1).astype(float).astype(int).astype(str)
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
    fw_cum = _prior_sum(is_front_win.astype(float).where(chaku.notna()), ck)
    aw_cum = _prior_sum(is_any_win.astype(float).where(chaku.notna()), ck)
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
# ブロック⑦: 条件別成績（場所別・騎手コンビ別）
# ---------------------------------------------------------------------------

def add_condition_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    馬の「条件別成績」を時系列リーク防止（groupby cumsum + shift）で集計。
    データはすでに日付順ソート済みを前提とする。

    追加特徴量:
      venue_*   : 開催場所 × 馬名 での過去勝率・複勝率・出走数
      combo_*   : 騎手 × 馬名 コンビでの過去勝率・複勝率・出走数
    """
    out = df.copy()
    if '着順_num' in out.columns:
        out['着順_num'] = pd.to_numeric(out['着順_num'], errors='coerce')

    chaku = out['着順_num']
    win   = (chaku == 1).astype(float).where(chaku.notna())
    fuku  = (chaku <= 3).astype(float).where(chaku.notna())

    def _cumstats(key_series: pd.Series, prefix: str) -> pd.DataFrame:
        k = key_series.fillna('__NA__')
        g_n = k.groupby(k).cumcount()
        g_w = _prior_sum(win, k)
        g_f = _prior_sum(fuku, k)
        return pd.DataFrame({
            f'{prefix}_win_rate':  (g_w / g_n).where(g_n > 0),
            f'{prefix}_fuku_rate': (g_f / g_n).where(g_n > 0),
            f'{prefix}_n_races':   g_n,
        }, index=out.index)

    # 場所別成績（馬名 × 開催場略称）
    _venue_code = out['開催'].astype(str).str.extract(r'\d([^\d]+)\d')[0].fillna('')
    venue_key = out['馬名'].astype(str) + '_' + _venue_code
    out = pd.concat([out, _cumstats(venue_key, 'venue')], axis=1)

    # 騎手コンビ成績（馬名 × 騎手）
    if '騎手' in out.columns:
        combo_key = out['馬名'].astype(str) + '_' + out['騎手'].astype(str).fillna('')
        out = pd.concat([out, _cumstats(combo_key, 'combo')], axis=1)
    else:
        for c in ['combo_win_rate', 'combo_fuku_rate', 'combo_n_races']:
            out[c] = np.nan

    # 馬場状態別成績（馬名 × 馬場状態番号）
    baba_key = out['馬名'].astype(str) + '_' + out['baba_num'].fillna(-1).astype(float).astype(int).astype(str)
    out = pd.concat([out, _cumstats(baba_key, 'baba')], axis=1)

    # 休養パターン別成績（馬名 × 休養カテゴリ）
    _iv = pd.to_numeric(out.get('interval_weeks', pd.Series(np.nan, index=out.index)), errors='coerce')
    _iv_cat = pd.cut(
        _iv,
        bins=[-np.inf, 1, 3, 9, 25, np.inf],
        labels=['連闘以下', '中1-2週', '中3-8週', '長期', '超長期']
    ).astype(str).replace('nan', '不明')
    rest_key = out['馬名'].astype(str) + '_' + _iv_cat
    out = pd.concat([out, _cumstats(rest_key, 'rest')], axis=1)

    # クラス別成績（馬名 × クラス_num）
    if 'クラス_num' in out.columns:
        class_key = out['馬名'].astype(str) + '_' + out['クラス_num'].fillna(-1).astype(float).astype(int).astype(str)
        out = pd.concat([out, _cumstats(class_key, 'class')], axis=1)
    else:
        for c in ['class_win_rate', 'class_fuku_rate', 'class_n_races']:
            out[c] = np.nan

    return out


# ---------------------------------------------------------------------------
# ブロック⑧: 走破タイム偏差
# ---------------------------------------------------------------------------

def add_time_deviation(df: pd.DataFrame) -> pd.DataFrame:
    """
    走破タイムのコース平均からの乖離を特徴化。
    master.csvの「走破秒」列（Target既変換値）を使用。

    - course_time_avg : コース別（芝ダ×距離）の平均走破秒（時系列リークなし）
    - horse_time_dev_avg : 馬がそのコースで出した時間偏差の累積平均（負=速い）
    """
    out = df.copy()
    if '走破秒' not in out.columns:
        out['course_time_avg'] = np.nan
        out['horse_time_dev_avg'] = np.nan
        return out

    time_s = pd.to_numeric(out['走破秒'], errors='coerce')

    is_turf = out.get('is_turf', pd.Series(0, index=out.index)).fillna(0)
    dist    = out.get('dist_num', pd.Series(-1, index=out.index)).fillna(-1).astype(float).astype(int)
    ck = is_turf.astype(str) + '_' + dist.astype(str)

    # NaN-safe cumsum: validマスクで重み付け
    _v = time_s.notna().astype(float)
    _t = time_s.fillna(0)
    ck_n   = _prior_sum(_v, ck)
    ck_sum = _prior_sum(_t * _v, ck)
    out['course_time_avg'] = (ck_sum / ck_n).where(ck_n > 0)

    # 各レースでの偏差（負=コース平均より速い）
    time_dev = time_s - out['course_time_avg']

    # 馬ごとの偏差累積平均（時系列リークなし）
    horse = out['馬名']
    _vd = time_dev.notna().astype(float)
    _td = time_dev.fillna(0)
    h_n   = _prior_sum(_vd, horse)
    h_sum = _prior_sum(_td * _vd, horse)
    out['horse_time_dev_avg'] = (h_sum / h_n).where(h_n > 0)

    return out


# ---------------------------------------------------------------------------
# ブロック⑨: 馬体重・血統・属性
# ---------------------------------------------------------------------------

def add_horse_attributes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def _c(name): return out[name] if name in out.columns else pd.Series(np.nan, index=out.index)
    out['weight_num']        = pd.to_numeric(_c('馬体重'),     errors='coerce')
    out['weight_change']     = pd.to_numeric(_c('馬体重増減'), errors='coerce')
    out['weight_change_abs'] = out['weight_change'].abs()
    out['career_num']        = pd.to_numeric(_c('キャリア'),   errors='coerce')
    out['sex_num']            = _c('性別').map({'牡': 0, '牝': 1, 'セ': 2})
    out['age_num']            = pd.to_numeric(_c('年齢'),       errors='coerce')

    for col, new_col in [('種牡馬', 'sire_code'), ('母父馬', 'bms_code')]:
        if col in out.columns:
            out[new_col] = pd.Categorical(out[col]).codes

    return out


# ---------------------------------------------------------------------------
# ブロック⑩: 血統（種牡馬・母父）適性 — 馬場/芝ダ/距離別の複勝率
# ---------------------------------------------------------------------------

MIN_PED_N = 30   # 血統別集計のサンプル数ガード（未満は NaN=除外）

def add_pedigree_aptitude(df: pd.DataFrame) -> pd.DataFrame:
    """
    種牡馬・母父ごとの「芝/ダ」「距離帯」「馬場(稍重以上)」別 複勝率を
    時系列リーク防止（groupby cumsum + shift）で集計する。

    - サンプル数が MIN_PED_N 未満の血統条件は NaN（予測時は0埋め＝効かない）。
    - 馬個体ではなく血統の傾向なので、距離は「帯(dist_cat)」で集計しサンプルを確保。

    生成列（種牡馬=sire / 母父=bms）:
      {sire,bms}_surf_fuku : 同じ芝/ダでの複勝率
      {sire,bms}_dist_fuku : 同じ距離帯での複勝率
      {sire,bms}_wet_fuku  : 稍重以上での複勝率
    """
    out = df.copy()
    if '着順_num' in out.columns:
        out['着順_num'] = pd.to_numeric(out['着順_num'], errors='coerce')

    chaku = out['着順_num'] if '着順_num' in out.columns else pd.Series(np.nan, index=out.index)
    valid = chaku.notna().astype(float)
    fuku  = (chaku <= 3).astype(float).fillna(0)   # 複勝(3着以内)。着順不明は0扱い

    is_turf  = out.get('is_turf',  pd.Series(0, index=out.index)).fillna(0).astype(int)
    dist_cat = out.get('dist_cat', pd.Series(np.nan, index=out.index)).fillna(-1).astype(float).astype(int)
    baba     = out.get('baba_num', pd.Series(0, index=out.index)).fillna(0)
    wet      = (baba >= 1).astype(float)           # 稍重以上フラグ

    def _rate(key: pd.Series) -> pd.Series:
        g_n = _prior_sum(valid, key)        # 有効な過去出走数（当該除く）
        g_f = _prior_sum(fuku, key)         # 過去の複勝数（当該除く）
        return (g_f / g_n).where(g_n >= MIN_PED_N)

    for ped_col, tag in [('種牡馬', 'sire'), ('母父馬', 'bms')]:
        if ped_col not in out.columns:
            for suf in ('surf_fuku', 'dist_fuku', 'sd_fuku', 'wet_fuku'):
                out[f'{tag}_{suf}'] = np.nan
            continue
        ped = out[ped_col].astype(str)

        out[f'{tag}_surf_fuku'] = _rate(ped + '_s' + is_turf.astype(str))
        out[f'{tag}_dist_fuku'] = _rate(ped + '_d' + dist_cat.astype(str))
        # 芝ダ × 距離帯 の複合（例: ダート短距離得意な産駒）。サンプルは減るので帯で。
        out[f'{tag}_sd_fuku']   = _rate(ped + '_s' + is_turf.astype(str) + '_d' + dist_cat.astype(str))

        # 馬場(稍重以上)別: wetレースのみで集計
        g_nw = _prior_sum(wet * valid, ped)
        g_fw = _prior_sum(fuku * wet, ped)
        out[f'{tag}_wet_fuku'] = (g_fw / g_nw).where(g_nw >= MIN_PED_N)

    return out


# ---------------------------------------------------------------------------
# ブロック⑩b: 騎手の条件別適性（芝ダ / 距離帯 別 複勝率）
# ---------------------------------------------------------------------------

MIN_JOCKEY_N = 30   # 騎手別集計のサンプル数ガード

def add_jockey_aptitude(df: pd.DataFrame) -> pd.DataFrame:
    """
    騎手 × 芝ダ / 距離帯 別の複勝率を時系列リーク防止で集計。
    既存の騎手総合成績(jockey_win_rate 等)に、条件別の得意不得意を補う。
    サンプル MIN_JOCKEY_N 未満は NaN（予測時0埋め＝効かない）。
    生成列: jockey_surf_fuku, jockey_dist_fuku
    """
    out = df.copy()
    if '着順_num' in out.columns:
        out['着順_num'] = pd.to_numeric(out['着順_num'], errors='coerce')
    chaku = out['着順_num'] if '着順_num' in out.columns else pd.Series(np.nan, index=out.index)
    valid = chaku.notna().astype(float)
    fuku  = (chaku <= 3).astype(float).fillna(0)

    is_turf  = out.get('is_turf',  pd.Series(0, index=out.index)).fillna(0).astype(int)
    dist_cat = out.get('dist_cat', pd.Series(np.nan, index=out.index)).fillna(-1).astype(float).astype(int)

    def _rate(key: pd.Series) -> pd.Series:
        g_n = _prior_sum(valid, key)
        g_f = _prior_sum(fuku, key)
        return (g_f / g_n).where(g_n >= MIN_JOCKEY_N)

    if '騎手' in out.columns:
        jk = out['騎手'].astype(str)
        out['jockey_surf_fuku'] = _rate(jk + '_s' + is_turf.astype(str))
        out['jockey_dist_fuku'] = _rate(jk + '_d' + dist_cat.astype(str))
    else:
        out['jockey_surf_fuku'] = np.nan
        out['jockey_dist_fuku'] = np.nan

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
        g_sum = _prior_sum(out['_ag_tmp'], out['馬名'])
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

        g_fc = _prior_sum(out['_fm'], out['馬名'])
        g_fw = _prior_sum(out['_fw'], out['馬名'])
        g_bc = _prior_sum(out['_bm'], out['馬名'])
        g_bw = _prior_sum(out['_bw'], out['馬名'])

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
        _dist = out.get('dist_num', pd.Series(-1, index=out.index)).fillna(-1).astype(float).astype(int)
        out['_ck2'] = (out['開催'].astype(str) + '_' +
                       _turf.astype(str) + '_' + _dist.astype(str))
        g_cn = out.groupby('_ck2', sort=False)['_pci2'].cumcount()
        g_cs = _prior_sum(out['_pci2'], out['_ck2'])
        out['course_pci_mean'] = (g_cs / g_cn).where(g_cn >= 5)
        out = out.drop(columns=['_pci2', '_ck2'])
    else:
        out['course_pci_mean'] = np.nan

    return out


# ---------------------------------------------------------------------------
# ブロック⑪: 展開不利度（前走で逆脚質が勝ったかを裏付けに使う）
# ---------------------------------------------------------------------------

def add_pace_excuse(df: pd.DataFrame) -> pd.DataFrame:
    """
    「展開不利で負けた」を、PCIだけでなく "そのレースで逆脚質が実際に勝ったか"
    で裏付ける。前走の事実のみ使用（当該レースの結果は使わない＝リークなし）。

    考え方:
      ① 前走で先行したのに『差し馬が勝った』展開で負けた → 先行有利な今走で巻き返し期待
      ② 前走で差したのに『先行馬が残った』展開で負けた   → 差し有利な今走で巻き返し期待

    生成列:
      pace_excuse_senko : 前走「先行して差し有利展開で負けた」度合い (0〜1)
      pace_excuse_sashi : 前走「差して先行有利展開で負けた」度合い (0〜1)
      senko_revenge_fit : pace_excuse_senko × 今走の先行有利度
      sashi_revenge_fit : pace_excuse_sashi × 今走の差し有利度
    """
    out = df.copy()
    if '着順_num' in out.columns:
        out['着順_num'] = pd.to_numeric(out['着順_num'], errors='coerce')
    chaku = out['着順_num'] if '着順_num' in out.columns else pd.Series(np.nan, index=out.index)

    # --- そのレースの勝ち馬の脚質（4角位置 / 頭数, 0=逃げ先行 1=差し追込）---
    p4 = pd.to_numeric(out['4角'], errors='coerce') if '4角' in out.columns else pd.Series(np.nan, index=out.index)
    hd = pd.to_numeric(out['頭数'], errors='coerce') if '頭数' in out.columns else pd.Series(np.nan, index=out.index)
    style_this = (p4 / hd).clip(0, 1)

    if all(c in out.columns for c in ('日付', '開催', 'Ｒ')):
        rk = out['日付'].astype(str) + out['開催'].astype(str) + out['Ｒ'].astype(str)
        win_style = style_this.where(chaku == 1)
        out['race_winner_style'] = win_style.groupby(rk).transform('mean')   # 勝ち馬の脚質を全行へ
    else:
        out['race_winner_style'] = np.nan

    # --- 前走の勝ち馬脚質（馬グループ内で1行shift）---
    prev_winner_style = out.groupby('馬名', sort=False)['race_winner_style'].shift(1) \
                        if '馬名' in out.columns else pd.Series(np.nan, index=out.index)

    # --- 前走の自分の脚質・着順 ---
    prev_self = out['prev_pos_ratio'] if 'prev_pos_ratio' in out.columns else pd.Series(np.nan, index=out.index)
    prev_lost = out['prev_chakujun'] >= 4 if 'prev_chakujun' in out.columns else pd.Series(False, index=out.index)

    # ① 前走: 先行(self<=0.4) かつ 差し馬が勝った(winner>=0.5) かつ 負けた
    cond_senko = (prev_self <= 0.4) & (prev_winner_style >= 0.5) & prev_lost
    out['pace_excuse_senko'] = (prev_winner_style - prev_self).clip(0, 1).where(cond_senko).fillna(0.0)

    # ② 前走: 差し(self>=0.6) かつ 先行馬が残った(winner<=0.4) かつ 負けた
    cond_sashi = (prev_self >= 0.6) & (prev_winner_style <= 0.4) & prev_lost
    out['pace_excuse_sashi'] = (prev_self - prev_winner_style).clip(0, 1).where(cond_sashi).fillna(0.0)

    # --- 今走の脚質有利度（pace_front_ratio: 高=先行有利, 低=差し有利）---
    pf = out['pace_front_ratio'].fillna(0.5) if 'pace_front_ratio' in out.columns else pd.Series(0.5, index=out.index)
    out['senko_revenge_fit'] = out['pace_excuse_senko'] * pf
    out['sashi_revenge_fit'] = out['pace_excuse_sashi'] * (1 - pf)

    out = out.drop(columns=['race_winner_style'], errors='ignore')   # リーク防止のため特徴量には残さない
    return out


# ---------------------------------------------------------------------------
# ブロック⑫: 形勢・妙味（前走大敗の度外視 / 昇級初戦の過剰人気警戒）
# ---------------------------------------------------------------------------

def add_form_value_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    ⑥ 前走大敗でも前々走までの実績を評価（人気落ちの妙味検出）
    ⑦ 前走クラス勝ち→今走昇級の初戦は過剰人気になりやすく評価を上げすぎない

    すべてレース前確定情報のみ（クラスは当日条件・前走以前の実績は既知）。
    生成列:
      past_avg_ex_prev : 前走を除く直近2〜4走の平均着順（小=実績堅実）
      flop_rebound     : 前走大敗(≧6着)だが それ以前は堅実(平均≦4着) のときの巻き返し余地
      class_up_first   : 前走勝ち上がり → 今走クラス昇級 の初戦フラグ(1/0)
    """
    out = df.copy()
    horse = out['馬名'] if '馬名' in out.columns else None

    # --- ⑥ 前走を除く直近成績 & 大敗の巻き返し余地 ---
    if 'prev_chakujun' in out.columns and horse is not None:
        s2 = out.groupby(horse, sort=False)['prev_chakujun'].shift(1)
        s3 = out.groupby(horse, sort=False)['prev_chakujun'].shift(2)
        s4 = out.groupby(horse, sort=False)['prev_chakujun'].shift(3)
        out['past_avg_ex_prev'] = pd.concat([s2, s3, s4], axis=1).mean(axis=1)
        prev_flop  = out['prev_chakujun'] >= 6      # 前走大敗
        past_solid = out['past_avg_ex_prev'] <= 4   # それ以前は堅実
        rebound = (out['prev_chakujun'] - out['past_avg_ex_prev']).clip(lower=0)
        out['flop_rebound'] = rebound.where(prev_flop & past_solid).fillna(0.0)
    else:
        out['past_avg_ex_prev'] = np.nan
        out['flop_rebound'] = 0.0

    # --- ⑦ 昇級初戦（前走勝ち上がり → 今走クラス↑）---
    if 'クラス_num' in out.columns and 'prev_chakujun' in out.columns and horse is not None:
        cls      = pd.to_numeric(out['クラス_num'], errors='coerce')
        prev_cls = pd.to_numeric(out.groupby(horse, sort=False)['クラス_num'].shift(1), errors='coerce')
        out['class_up_first'] = ((cls > prev_cls) & (out['prev_chakujun'] == 1)).astype(float)
    else:
        out['class_up_first'] = 0.0

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
    'prev_top3f_diff',
    # 条件変化
    'dist_change', 'interval_weeks', 'baba_change',
    'track_changed', 'jockey_changed', 'kinryo_change',
    # 馬・騎手・調教師 通算成績
    'horse_win_rate', 'horse_fuku_rate', 'horse_avg_chaku', 'horse_n_races',
    'jockey_win_rate', 'jockey_fuku_rate', 'jockey_avg_chaku', 'jockey_n_races',
    'trainer_win_rate', 'trainer_fuku_rate', 'trainer_avg_chaku',
    # 騎手の条件別適性（芝ダ・距離帯）
    'jockey_surf_fuku', 'jockey_dist_fuku',
    # 直近2走の安定性
    'agari_avg2', 'agari_std2', 'chakusa_avg2', 'chaku_avg2',
    # 直近3走の安定性
    'agari_avg3', 'agari_std3', 'chakusa_avg3', 'chaku_avg3',
    # 直近5走の安定性・トレンド
    'agari_avg5', 'agari_std5', 'chaku_avg5', 'chakusa_avg5',
    'chaku_trend5',
    # 位置取り（脚質）の安定性・トレンド
    'pos4c_avg3', 'pos4c_trend3',
    # 条件別成績（場所・騎手コンビ・馬場・休養・クラス）
    'venue_win_rate', 'venue_fuku_rate', 'venue_n_races',
    'combo_win_rate', 'combo_fuku_rate', 'combo_n_races',
    'baba_win_rate', 'baba_fuku_rate', 'baba_n_races',
    'rest_win_rate', 'rest_fuku_rate',
    'class_win_rate', 'class_fuku_rate', 'class_n_races',
    # 走破タイム偏差
    'horse_time_dev_avg',
    # コース適性（芝/ダ × 距離帯）
    'course_win_rate', 'course_fuku_rate', 'course_avg_chaku', 'course_n_races',
    # 脚質 × コース適性
    'style_score', 'style_avg3',
    'pace_front_ratio', 'style_course_fit',
    # 馬属性（生の sire_code/bms_code は産駒複勝率に置き換えのため廃止）
    'weight_num', 'weight_change', 'weight_change_abs',
    'career_num', 'sex_num', 'age_num',
    # 血統（産駒）適性: 種牡馬・母父 × 芝ダ/距離帯/複合/馬場 の複勝率
    'sire_surf_fuku', 'sire_dist_fuku', 'sire_sd_fuku', 'sire_wet_fuku',
    'bms_surf_fuku', 'bms_dist_fuku', 'bms_sd_fuku', 'bms_wet_fuku',
    # ペース適性（PCI系・master.csvがある場合のみ有効、なければNaN→0）
    'prev_pci', 'agari_hist_avg', 'horse_front_pace_wr', 'horse_back_pace_wr', 'course_pci_mean',
    # 展開不利度（前走の逆脚質勝ち判定 → 今走の脚質適性との相性）
    'pace_excuse_senko', 'pace_excuse_sashi',
    'senko_revenge_fit', 'sashi_revenge_fit',
    # 形勢・妙味（前走大敗の度外視・昇級初戦の過剰人気警戒）
    'past_avg_ex_prev', 'flop_rebound', 'class_up_first',
    # ※「人気」(当該レースのオッズ順)はモデル入力から除外。未来レースでは予測時に
    #   未確定で、過去のみ存在するためバックテストが過大評価＆ライブが荒れる原因になる。
    #   人気/オッズは EV(買い/見送り)判定専用に使う。METAには表示用に保持。
]

TARGET_COL = '着順_num'

META_COLS = ['日付_dt', '日付', '開催', 'Ｒ', '馬番', '馬名', '騎手', '着順_num', '人気']


def build_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    # 馬名の正規化（重要）: 出走表CSVの馬名は ' キシャール' '$マリブオレンジ' のように
    # 先頭に空白(半角/全角)や $ * 等のマーカーが付き、master(クリーン)と不一致になる。
    # masterもライブも同じ正規化を通すことで、馬名でグループ集計する特徴量
    # (馬の勝率・コース実績・近走安定・展開不利・前走大敗妙味 等)が正しく紐づく。
    if '馬名' in df.columns:
        df['馬名'] = (df['馬名'].astype(str)
                      .str.replace(r'^[\s　$*＊◇◆☆★・]+', '', regex=True)
                      .str.replace(r'[\s　]+$', '', regex=True))

    # 着順_num を確実にfloat64にする（None/apply返値のobject化対策）
    if '着順_num' in df.columns:
        df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')

    if verbose: print("[1/6] レース条件...")
    df = add_race_context(df)

    if verbose: print("[2/6] 前走特徴量...")
    df = add_previous_race_features(df)

    if verbose: print("[3/6] 馬・騎手・調教師 通算成績（時系列リーク防止）...")
    df = add_historical_stats(df)

    if verbose: print("[4/6] 直近2〜3走の安定性...")
    df = add_rolling_form(df, windows=[2, 3])

    if verbose: print("[5/7] コース適性（芝/ダ × 距離帯）...")
    df = add_course_aptitude(df)

    if verbose: print("[6/7] 脚質 × コース適性...")
    df = add_style_course_interaction(df)

    if verbose: print("[7/10] 条件別成績（場所・騎手・馬場・休養・クラス）...")
    df = add_condition_stats(df)

    if verbose: print("[8/10] 走破タイム偏差...")
    df = add_time_deviation(df)

    if verbose: print("[9/10] 馬属性・血統...")
    df = add_horse_attributes(df)

    if verbose: print("[9.5/10] 血統適性（馬場/芝ダ/距離別 複勝率）...")
    df = add_pedigree_aptitude(df)

    if verbose: print("[9.6/10] 騎手の条件別適性（芝ダ/距離別 複勝率）...")
    df = add_jockey_aptitude(df)

    if verbose: print("[10/10] PCIペース特徴量...")
    try:
        df = add_pace_features(df)
    except Exception as _e:
        if verbose:
            print(f"  ペース特徴量でエラー（スキップ）: {_e}")

    if verbose: print("[10.5/10] 展開不利度（前走の逆脚質勝ち判定）...")
    df = add_pace_excuse(df)

    if verbose: print("[10.7/10] 形勢・妙味（前走大敗度外視・昇級初戦警戒）...")
    df = add_form_value_signals(df)

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
