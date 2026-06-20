"""
統計分析・フィルタリングモジュール。
scraperで取得したDataFrameを受け取り、分析・スコアリングを行う。
"""

import re
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 馬体重の前走比変換
# ---------------------------------------------------------------------------

def parse_bataiju(bataiju_str: str) -> tuple[float | None, float | None]:
    """
    '510(+4)' → (510.0, +4.0)
    '計不' → (None, None)
    """
    m = re.match(r"(\d+)\(([+\-]\d+)\)", str(bataiju_str))
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


# ---------------------------------------------------------------------------
# タイム → 秒数変換
# ---------------------------------------------------------------------------

def time_to_sec(t: str) -> float | None:
    """'1:34.5' → 94.5"""
    m = re.match(r"(\d+):(\d+\.\d+)", str(t))
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    return None


# ---------------------------------------------------------------------------
# 過去成績から特徴量を計算
# ---------------------------------------------------------------------------

def build_horse_features(horse_results: pd.DataFrame) -> pd.Series:
    """
    馬の過去成績DataFrameから特徴量を生成。
    horse_resultsの列名はnetkeiba日本語列名を想定。
    """
    feats = {}

    if horse_results.empty:
        return pd.Series(feats)

    df = horse_results.copy()

    # 着順を数値化 (除外・中止などは NaN)
    def parse_chakujun(v):
        try:
            return int(str(v).replace("着", ""))
        except ValueError:
            return np.nan

    chaku_col = next((c for c in df.columns if "着順" in c or "着" == c), None)
    if chaku_col:
        df["_chaku"] = df[chaku_col].apply(parse_chakujun)
        valid = df["_chaku"].dropna()
        feats["avg_chakujun"] = valid.mean() if len(valid) else np.nan
        feats["win_rate"] = (valid == 1).mean() if len(valid) else 0.0
        feats["rentai_rate"] = (valid <= 2).mean() if len(valid) else 0.0
        feats["fukusho_rate"] = (valid <= 3).mean() if len(valid) else 0.0
        feats["recent3_avg"] = valid.head(3).mean() if len(valid) >= 1 else np.nan

    # 走行タイム
    time_col = next((c for c in df.columns if "タイム" in c), None)
    if time_col:
        df["_sec"] = df[time_col].apply(time_to_sec)
        valid_sec = df["_sec"].dropna()
        feats["avg_time_sec"] = valid_sec.mean() if len(valid_sec) else np.nan

    # 人気
    popular_col = next((c for c in df.columns if "人気" in c), None)
    if popular_col:
        def parse_pop(v):
            try:
                return int(str(v))
            except ValueError:
                return np.nan
        df["_pop"] = df[popular_col].apply(parse_pop)
        valid_pop = df["_pop"].dropna()
        if len(valid_pop) and chaku_col:
            # 人気より着順が良かった率（穴馬傾向）
            feats["upset_rate"] = (df["_chaku"] < df["_pop"]).mean()

    return pd.Series(feats)


# ---------------------------------------------------------------------------
# 統計スコアリング
# ---------------------------------------------------------------------------

def score_horses(shutuba_df: pd.DataFrame, horse_feats: dict[str, pd.Series]) -> pd.DataFrame:
    """
    出走表と各馬の特徴量から総合スコアを計算。

    horse_feats: {horse_id: feature_series}
    """
    df = shutuba_df.copy()

    feat_df = pd.DataFrame(horse_feats).T  # index = horse_id
    feat_df.index.name = "horse_id"
    feat_df = feat_df.reset_index()

    df = df.merge(feat_df, on="horse_id", how="left")

    # --- スコア計算（シンプルな加重合計）---
    # 各指標を 0-1 に正規化して加重
    score = pd.Series(0.0, index=df.index)

    def norm_series(s: pd.Series, invert=False) -> pd.Series:
        """Min-max正規化。invertはTrue→低いほど良い場合に反転。"""
        s = pd.to_numeric(s, errors="coerce")
        mn, mx = s.min(), s.max()
        if mx == mn:
            return pd.Series(0.5, index=s.index)
        normed = (s - mn) / (mx - mn)
        return 1 - normed if invert else normed

    weights = {
        "win_rate": 3.0,
        "rentai_rate": 2.0,
        "fukusho_rate": 1.5,
        "avg_chakujun": -2.0,   # 低いほど良い → 負の重みで加算
        "recent3_avg": -2.5,
    }

    for col, w in weights.items():
        if col in df.columns:
            invert = w < 0
            normed = norm_series(df[col], invert=invert)
            score += normed.fillna(0) * abs(w)

    df["score"] = score
    df["score_rank"] = df["score"].rank(ascending=False, method="min").astype(int)

    return df.sort_values("score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 回収率・期待値計算
# ---------------------------------------------------------------------------

def calc_expected_value(
    odds: float,
    win_prob: float,
    tax_rate: float = 0.25,
) -> float:
    """
    単純期待値 = 予測勝率 × 払戻金額 - 1.0
    払戻金額 = odds × (1 - tax_rate) ※ 実際はJRAの控除率は約25%
    """
    payout = odds * (1 - tax_rate)
    return win_prob * payout - 1.0


def add_expected_value(df: pd.DataFrame) -> pd.DataFrame:
    """
    scoreを確率に変換してEVを付与。
    odds列が数値として存在する場合のみ計算。
    """
    result = df.copy()

    if "odds" not in result.columns:
        return result

    result["odds_num"] = pd.to_numeric(result["odds"], errors="coerce")

    # スコアをsoftmaxで確率に変換
    scores = result["score"].fillna(0)
    exp_scores = np.exp(scores - scores.max())
    result["win_prob"] = exp_scores / exp_scores.sum()

    result["expected_value"] = result.apply(
        lambda r: calc_expected_value(r["odds_num"], r["win_prob"])
        if pd.notna(r["odds_num"])
        else np.nan,
        axis=1,
    )

    return result


# ---------------------------------------------------------------------------
# 条件フィルタ
# ---------------------------------------------------------------------------

def filter_by_conditions(
    df: pd.DataFrame,
    min_win_rate: float | None = None,
    max_avg_chakujun: float | None = None,
    min_fukusho_rate: float | None = None,
) -> pd.DataFrame:
    """条件を指定してDataFrameを絞り込む。"""
    mask = pd.Series(True, index=df.index)

    if min_win_rate is not None and "win_rate" in df.columns:
        mask &= df["win_rate"] >= min_win_rate

    if max_avg_chakujun is not None and "avg_chakujun" in df.columns:
        mask &= df["avg_chakujun"] <= max_avg_chakujun

    if min_fukusho_rate is not None and "fukusho_rate" in df.columns:
        mask &= df["fukusho_rate"] >= min_fukusho_rate

    return df[mask].reset_index(drop=True)
