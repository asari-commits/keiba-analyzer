"""
予測画面のブラッシュアップ用ユーティリティ。
- EV（期待値）計算
- 自信度スコア
- 推奨馬券
- 予測根拠の生成
- 同条件過去回収率
"""
import numpy as np
import pandas as pd

# ── 人気別 実績統計（10年分から算出）────────────────────────────────
# 単勝平均配当・複勝平均配当・勝率・複勝率
POPULAR_STATS = {
    1:  {'avg_tan': 238,  'avg_fuku': 130,  'win_rate': 0.331, 'fuku_rate': 0.647},
    2:  {'avg_tan': 412,  'avg_fuku': 162,  'win_rate': 0.194, 'fuku_rate': 0.518},
    3:  {'avg_tan': 597,  'avg_fuku': 195,  'win_rate': 0.132, 'fuku_rate': 0.417},
    4:  {'avg_tan': 821,  'avg_fuku': 235,  'win_rate': 0.093, 'fuku_rate': 0.333},
    5:  {'avg_tan': 1117, 'avg_fuku': 290,  'win_rate': 0.071, 'fuku_rate': 0.266},
    6:  {'avg_tan': 1528, 'avg_fuku': 364,  'win_rate': 0.054, 'fuku_rate': 0.217},
    7:  {'avg_tan': 2041, 'avg_fuku': 458,  'win_rate': 0.039, 'fuku_rate': 0.165},
    8:  {'avg_tan': 2636, 'avg_fuku': 580,  'win_rate': 0.027, 'fuku_rate': 0.129},
    9:  {'avg_tan': 3549, 'avg_fuku': 733,  'win_rate': 0.021, 'fuku_rate': 0.099},
    10: {'avg_tan': 4524, 'avg_fuku': 892,  'win_rate': 0.017, 'fuku_rate': 0.079},
    11: {'avg_tan': 6058, 'avg_fuku': 1168, 'win_rate': 0.011, 'fuku_rate': 0.061},
    12: {'avg_tan': 7303, 'avg_fuku': 1397, 'win_rate': 0.010, 'fuku_rate': 0.049},
    13: {'avg_tan': 9860, 'avg_fuku': 1757, 'win_rate': 0.006, 'fuku_rate': 0.035},
    14: {'avg_tan':12202, 'avg_fuku': 2178, 'win_rate': 0.005, 'fuku_rate': 0.027},
    15: {'avg_tan':13304, 'avg_fuku': 2680, 'win_rate': 0.004, 'fuku_rate': 0.020},
    16: {'avg_tan':20847, 'avg_fuku': 3552, 'win_rate': 0.002, 'fuku_rate': 0.013},
    17: {'avg_tan':21987, 'avg_fuku': 3990, 'win_rate': 0.001, 'fuku_rate': 0.012},
    18: {'avg_tan':29310, 'avg_fuku': 4361, 'win_rate': 0.001, 'fuku_rate': 0.009},
}

# 特徴量 → 人間可読ラベル
FEATURE_LABELS = {
    'horse_avg_chaku':   '馬の平均着順が良い',
    'horse_win_rate':    '馬の過去勝率が高い',
    'horse_fuku_rate':   '馬の複勝率が高い',
    'course_avg_chaku':  'このコースで好成績',
    'course_win_rate':   'このコースの勝率が高い',
    'jockey_win_rate':   '騎手の勝率が高い',
    'jockey_avg_chaku':  '騎手の平均着順が良い',
    'trainer_win_rate':  '調教師の勝率が高い',
    'prev_agari':        '前走上り3Fが速い',
    'agari_avg3':        '直近3走の上り平均が速い',
    'prev_chakusa_t':    '前走着差が少ない（接戦）',
    'chakusa_avg3':      '直近3走の着差が少ない',
    'prev_time_sec':     '前走タイムが優秀',
    'interval_weeks':    '休養明けリフレッシュ',
    'dist_change':       '距離延長・短縮が合う',
    'weight_num':        '馬体重が安定',
    'style_course_fit':  '脚質がこのコースに合う',
    'pace_front_ratio':  '先行有利コースで先行脚質',
    'style_avg3':        '脚質傾向が安定',
    'prev_chakujun':     '前走着順が良い',
    'prev_koma_diff':    '前走で追い込み実績',
    'prev_pos_ratio':    '前走の位置取りが理想的',
    'course_fuku_rate':  'このコースの複勝率が高い',
    'agari_avg2':        '直近2走の上り平均が速い',
    'chaku_avg3':        '直近3走の着順が安定',
    'prev_pos_4c':       '前走4角で前目につけた',
}


def softmax_probs(scores: pd.Series) -> pd.Series:
    """
    pred_score（高いほど上位）を勝利確率に変換（Plackett-Luce 近似）。
    LightGBM lambdarank は高スコア=好走予測なのでそのまま softmax に掛ける。
    """
    s = scores.fillna(scores.min())
    exp = np.exp(s - s.max())   # 数値安定のため max を引く
    return exp / exp.sum()


def estimate_fuku_probs(win_probs: pd.Series) -> pd.Series:
    """
    勝利確率からtop3（複勝）確率を推定。
    Plackett-Luce モデルの近似: P(top3) ≈ min(3 × P(1st), 0.95)
    小頭数補正: 5頭以下は補正なし（実際は全馬が複勝対象に近い）
    """
    n = len(win_probs)
    if n <= 3:
        return pd.Series(1.0, index=win_probs.index)
    # 数学的近似: 3/n × (p_i / mean_p) = 3 × p_i
    fuku = (win_probs * 3.0).clip(upper=min(0.95, 3.0 / max(n, 1) * n))
    return fuku.clip(upper=0.95)


def calc_ev(model_prob: float, popular: int, bet_type: str = 'tan') -> float:
    """
    期待値を計算（ライブオッズなし時の市場統計ベース推定）。
    model_prob : モデルが推定する勝利 or 複勝確率 (0〜1)
    popular    : 人気順位 (1〜18)  ← avg_payout の参照に使用
    bet_type   : 'tan'=単勝, 'fuku'=複勝

    EV = model_prob × avg_payout - 100
    ・model_prob が市場確率と同じなら EV ≈ -21%（胴元控除分の損）
    ・model_prob が市場より高ければ EV がプラス = 割安馬券
    戻り値 : 期待値（%）、EV > 0 が期待値プラス
    """
    p = int(np.clip(popular, 1, 18))
    avg_pay = (POPULAR_STATS[p]['avg_tan']
               if bet_type == 'tan'
               else POPULAR_STATS[p]['avg_fuku'])

    ev_pct = model_prob * avg_pay - 100
    return float(round(max(ev_pct, -100.0), 1))


def calc_ev_live(win_prob: float, odds: float) -> float:
    """
    ライブオッズを使った正確なEV計算（単勝）。
    win_prob : モデル勝利確率
    odds     : 単勝オッズ（例: 3.5倍）
    """
    if odds <= 0 or np.isnan(odds):
        return float('nan')
    return float(round(max(win_prob * odds * 100 - 100, -100.0), 1))


def implied_odds(win_prob: float) -> float:
    """モデル確率から推定単勝倍率を計算（馬券の公正オッズ = 1/P）"""
    if win_prob <= 0:
        return float('nan')
    return round(1.0 / win_prob, 1)


def confidence_score(race_df: pd.DataFrame) -> float:
    """
    レース自信度スコア（0〜1）。
    予測1位と2位のスコア差 / 全体スコア幅
    """
    probs = softmax_probs(race_df['pred_score'])
    sorted_p = probs.sort_values(ascending=False)
    if len(sorted_p) < 2:
        return 0.0
    gap = sorted_p.iloc[0] - sorted_p.iloc[1]
    return float(np.clip(gap * 10, 0, 1))


def recommend_bet(conf: float, top3_ev: list[float]) -> str:
    """自信度とEVから推奨馬券を返す"""
    max_ev = max(top3_ev) if top3_ev else -100
    if conf >= 0.7 and max_ev >= 50:
        return "単勝・複勝"
    elif conf >= 0.5 and max_ev >= 20:
        return "複勝・馬連"
    elif conf >= 0.4:
        return "馬連・三連複"
    elif max_ev >= 100:
        return "複勝（穴狙い）"
    else:
        return "見送りも検討"


def get_reasons(horse_row: pd.Series, race_df: pd.DataFrame, top_n: int = 3) -> list[str]:
    """
    その馬がなぜ上位予測されているかの根拠を返す。
    フィールド内での相対的な優位性を評価。
    前走CSVのデータ（prev_chakujun / interval_weeks / prev_pos_4c など）も活用。
    """
    reasons = []

    # (feature, higher_is_better, threshold_ratio)
    eval_features = [
        # 過去通算成績（master.csvがある場合）
        ('horse_avg_chaku',   False, 0.8),
        ('course_avg_chaku',  False, 0.8),
        ('agari_avg3',        False, 0.85),
        ('prev_agari',        False, 0.85),
        ('horse_win_rate',    True,  1.3),
        ('course_win_rate',   True,  1.5),
        ('jockey_win_rate',   True,  1.3),
        ('trainer_win_rate',  True,  1.3),
        ('horse_fuku_rate',   True,  1.3),
        ('course_fuku_rate',  True,  1.5),
        ('chaku_avg3',        False, 0.8),
        ('chakusa_avg3',      False, 0.7),
        # 前走CSVで補完される特徴量（出馬表のみでも利用可）
        ('prev_chakusa_t',    False, 0.65),   # 前走着差タイム（小さいほど接戦）
        ('prev_chakujun',     False, 0.75),   # 前走着順（小さいほど好走）
        ('prev_pos_4c',       False, 0.6),    # 前走4角位置（小さいほど先行）
        ('style_course_fit',  True,  1.2),
        ('pace_front_ratio',  True,  1.2),
    ]

    for feat, higher_is_better, threshold in eval_features:
        if feat not in horse_row.index or feat not in race_df.columns:
            continue
        val = horse_row[feat]
        if pd.isna(val):
            continue
        field_vals = race_df[feat].dropna()
        if len(field_vals) < 2:
            continue
        field_mean = field_vals.mean()
        if pd.isna(field_mean) or field_mean == 0:
            continue

        ratio = val / field_mean
        if higher_is_better and ratio >= threshold:
            reasons.append(FEATURE_LABELS.get(feat, feat))
        elif not higher_is_better and ratio <= threshold:
            reasons.append(FEATURE_LABELS.get(feat, feat))

        if len(reasons) >= top_n:
            break

    # ── フィーチャーで判定できなかった場合の情報ベースfallback ────────
    if len(reasons) < top_n:
        # モデルスコアの相対位置
        score = horse_row.get('pred_score')
        if pd.notna(score):
            scores = race_df['pred_score'].dropna()
            if len(scores) > 1:
                pct = (scores > score).sum() / len(scores)
                if pct >= 0.7 and 'モデル総合スコアが上位' not in reasons:
                    reasons.append('モデル総合スコアが上位')

        # 前走着順（直接表示）
        prev_chaku = horse_row.get('prev_chakujun')
        if pd.notna(prev_chaku) and '前走着順が良い' not in reasons:
            chaku_int = int(prev_chaku)
            if chaku_int <= 3:
                reasons.append(f'前走{chaku_int}着（好走直後）')
            elif chaku_int <= 5:
                reasons.append(f'前走{chaku_int}着')

        # 前走着差（直接表示）
        prev_chakusa = horse_row.get('prev_chakusa_t')
        if pd.notna(prev_chakusa) and float(prev_chakusa) <= 0.3:
            reasons.append(f'前走着差{float(prev_chakusa):.1f}秒（接戦）')

        # 休養明け
        interval = horse_row.get('interval_weeks')
        if pd.notna(interval) and float(interval) >= 8:
            reasons.append(f'{int(interval)}週ぶり（リフレッシュ）')

        # 騎手実績
        jwr = horse_row.get('jockey_win_rate')
        if pd.notna(jwr) and float(jwr) > 0.15:
            reasons.append('騎手の勝率が高い')

        # 斤量有利
        kilo = pd.to_numeric(horse_row.get('kinryo_num', horse_row.get('斤量')), errors='coerce')
        if pd.notna(kilo):
            race_kilo = pd.to_numeric(
                race_df.get('kinryo_num', race_df.get('斤量', pd.Series(dtype=float))),
                errors='coerce'
            )
            field_avg = race_kilo.mean()
            if pd.notna(field_avg) and kilo < field_avg - 1:
                reasons.append('斤量が軽い')

        # ペース適性（PCI系特徴量）
        if len(reasons) < top_n:
            front_wr = horse_row.get('horse_front_pace_wr')
            back_wr  = horse_row.get('horse_back_pace_wr')
            prev_pci = horse_row.get('prev_pci')
            if pd.notna(front_wr) and float(front_wr) >= 0.25:
                reasons.append(f'前傾ペース勝率{float(front_wr):.0%}')
            elif pd.notna(back_wr) and float(back_wr) >= 0.25:
                reasons.append(f'後傾ペース勝率{float(back_wr):.0%}')
            if pd.notna(prev_pci):
                pci_v = float(prev_pci)
                if pci_v < 46:
                    reasons.append('前走が強前傾ペース経験')
                elif pci_v > 54:
                    reasons.append('前走が強後傾ペース経験')

        # コース勝率（数値直接表示）
        if len(reasons) < top_n:
            cwr = horse_row.get('course_win_rate')
            if pd.notna(cwr) and float(cwr) >= 0.20:
                reasons.append(f'このコース勝率{float(cwr):.0%}')

        # 前走4角位置（先行馬アピール）
        if len(reasons) < top_n:
            pos4 = horse_row.get('prev_pos_4c')
            if pd.notna(pos4) and float(pos4) <= 3:
                reasons.append(f'前走4角{int(pos4)}番手（先行）')

    # 最終fallback（それでも空の場合）
    if not reasons:
        score = horse_row.get('pred_score')
        rank  = horse_row.get('pred_rank')
        if pd.notna(score) and pd.notna(rank):
            reasons.append(f'モデル評価{int(rank)}位（詳細根拠集計中）')
        else:
            reasons.append('根拠データ集計中')

    return reasons[:top_n]


def course_history_roi(master_df: pd.DataFrame, venue_name: str,
                       is_turf: int, dist: int) -> dict:
    """
    同条件（競馬場×芝ダ×距離帯±100m）での過去モデル的中実績を返す。
    master_df に pred_rank が含まれている場合のみ機能する。
    """
    VENUE_MAP = {
        '東': '東京', '中': '中山', '京': '京都', '阪': '阪神',
        '名': '中京', '小': '小倉', '新': '新潟', '福': '福島',
        '函': '函館', '札': '札幌',
    }

    df = master_df.copy()
    df['_venue'] = df['開催'].astype(str).str.extract(r'\d([^\d]+)\d')[0].map(VENUE_MAP)
    df['距離_num'] = pd.to_numeric(df['距離'], errors='coerce')

    filt = df[
        (df['_venue'] == venue_name) &
        (df['芝・ダ'] == ('芝' if is_turf else 'ダ')) &
        (df['距離_num'].between(dist - 100, dist + 100))
    ]

    n = len(filt)
    if n == 0:
        return {'n': 0, 'tan_roi': None, 'fuku_roi': None}

    def parse_pay(s):
        try:
            v = str(s).strip()
            if v.startswith('(') or v in ('', 'nan', 'None', 'NaN'):
                return np.nan
            f = float(v)
            return f if (f == f and f > 0) else np.nan
        except Exception:
            return np.nan

    tan_pay = filt['単勝配当'].apply(parse_pay) if '単勝配当' in filt.columns else pd.Series(dtype=float)
    fuku_pay = filt['複勝配当'].apply(parse_pay) if '複勝配当' in filt.columns else pd.Series(dtype=float)

    chaku = pd.to_numeric(
        filt['着順'].astype(str).str.translate(str.maketrans('０１２３４５６７８９','0123456789')),
        errors='coerce'
    )
    races = filt.groupby(['日付', '開催', 'Ｒ']).ngroups

    win_pay_sum = tan_pay.where(chaku == 1).sum()
    fuku_pay_sum = fuku_pay.where(chaku <= 3).sum()
    tan_roi = (win_pay_sum - races * 100) / (races * 100) * 100 if races > 0 else None
    fuku_roi = (fuku_pay_sum - races * 100) / (races * 100) * 100 if races > 0 else None

    return {
        'n_races': races,
        'n_horses': n,
        'tan_roi': round(tan_roi, 1) if tan_roi is not None else None,
        'fuku_roi': round(fuku_roi, 1) if fuku_roi is not None else None,
    }
