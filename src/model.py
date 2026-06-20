"""
LightGBMによる着順予測モデル。
学習・評価・予測・保存/ロードをまとめて管理する。
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import ndcg_score

from features import FEATURE_COLS, TARGET_COL

MODEL_PATH       = Path(__file__).parent.parent / "data" / "processed" / "lgbm_model.pkl"
MODEL_ANABA_PATH = Path(__file__).parent.parent / "data" / "processed" / "lgbm_model_anaba.pkl"

# 穴馬モデルで除外する特徴量（人気・過去実績ベースの指標）
ANABA_EXCLUDE = {'人気', 'horse_win_rate', 'horse_fuku_rate', 'horse_avg_chaku', 'horse_n_races'}


# ---------------------------------------------------------------------------
# 学習
# ---------------------------------------------------------------------------

def train(feat_df: pd.DataFrame, save: bool = True,
          model_path: Path = None, exclude_cols: set = None) -> lgb.LGBMRanker:
    """
    model_path: 保存先（Noneなら MODEL_PATH）
    exclude_cols: 除外する特徴量名のセット（穴馬モデル用）
    """
    if model_path is None:
        model_path = MODEL_PATH
    """
    LightGBM Ranker（Learning to Rank）でレース内着順を学習。
    GroupKFold で日付ベースの時系列分割を行い過学習を防ぐ。
    """
    exclude = exclude_cols or set()
    use_cols = [c for c in FEATURE_COLS if c in feat_df.columns and c not in exclude]
    df = feat_df.dropna(subset=[TARGET_COL]).copy()

    X = df[use_cols].astype(float)
    y = df[TARGET_COL].astype(float)

    # グループ = レース（同一レース内でランキング学習）
    race_key = df['日付'].astype(str) + df['開催'].astype(str) + df['Ｒ'].astype(str)
    groups = race_key.factorize()[0]

    # 各グループのサイズ（LightGBM Rankerに必要）
    group_sizes = df.groupby(race_key).size().values

    # 時系列分割: 直近20%をテストに使用
    split_idx = int(len(df) * 0.8)
    # 日付でソートされていることを前提
    train_mask = df.index < df.index[split_idx]

    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]
    g_train = df[train_mask].groupby(race_key[train_mask]).size().values
    g_test  = df[~train_mask].groupby(race_key[~train_mask]).size().values
    race_key_test = race_key[~train_mask]

    print(f"学習データ: {X_train.shape[0]}行  テストデータ: {X_test.shape[0]}行")
    print(f"使用特徴量: {len(use_cols)}個")

    model = lgb.LGBMRanker(
        objective='lambdarank',
        metric='ndcg',
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X_train, y_train,
        group=g_train,
        eval_set=[(X_test, y_test)],
        eval_group=[g_test],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(50)],
    )

    # --- 評価 ---
    print("\n=== モデル評価（テストデータ） ===")
    pred = model.predict(X_test)
    test_df = X_test.copy()
    test_df['pred'] = pred
    test_df['actual'] = y_test.values
    test_df['race_key'] = race_key_test.values

    evaluate(test_df)

    if save:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump({'model': model, 'features': use_cols}, f)
        print(f"\nモデル保存: {model_path}")

    return model


# ---------------------------------------------------------------------------
# 評価
# ---------------------------------------------------------------------------

def evaluate(test_df: pd.DataFrame) -> None:
    """1着的中率・複勝的中率・回収率シミュレーション"""

    results = []
    for race_key, grp in test_df.groupby('race_key'):
        if len(grp) < 3:
            continue
        ranked = grp.sort_values('pred')  # スコアが低い＝着順が良い（Ranker）
        pred_1st = ranked.iloc[0]['actual']
        pred_top3 = ranked.head(3)['actual'].tolist()

        results.append({
            'win_hit': pred_1st == 1.0,
            'fuku_hit': 1.0 in pred_top3,
        })

    res = pd.DataFrame(results)
    print(f"  1着的中率 : {res['win_hit'].mean():.1%}  ({res['win_hit'].sum()}/{len(res)}レース)")
    print(f"  複勝的中率: {res['fuku_hit'].mean():.1%}  ({res['fuku_hit'].sum()}/{len(res)}レース)")


# ---------------------------------------------------------------------------
# 特徴量重要度
# ---------------------------------------------------------------------------

def feature_importance(model: lgb.LGBMRanker, feature_cols: list) -> pd.DataFrame:
    imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    return imp


# ---------------------------------------------------------------------------
# 予測（新規レース用）
# ---------------------------------------------------------------------------

def load_model() -> tuple[lgb.LGBMRanker, list]:
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['features']


def predict_race(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    特徴量DataFrameを受け取り、スコア・予測順位を付与して返す。
    タイム系の特徴量（走破秒・time_diff_from_winner等）は
    レース前は不明なので0埋めで補完する。
    """
    model, use_cols = load_model()

    df = feat_df.copy()
    X = df.reindex(columns=use_cols).astype(float).fillna(0)

    df['score'] = model.predict(X)
    df['pred_rank'] = df['score'].rank(method='min').astype(int)  # 小さいほど上位

    return df.sort_values('pred_rank').reset_index(drop=True)
