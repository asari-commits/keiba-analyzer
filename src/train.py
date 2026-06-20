"""
LightGBM Ranker 学習スクリプト。
通常モデル（lgbm_model.pkl）と穴馬モデル（lgbm_model_anaba.pkl）を学習する。

実行:
    python src/train.py

データ:
    data/processed/master.csv  ← build_dataset.py で生成

出力:
    data/processed/lgbm_model.pkl
    data/processed/lgbm_model_anaba.pkl
"""
import sys
import pickle
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit

from features import build_features, FEATURE_COLS, TARGET_COL

MASTER_CSV       = Path(__file__).parent.parent / "data" / "processed" / "master.csv"
MODEL_PATH       = Path(__file__).parent.parent / "data" / "processed" / "lgbm_model.pkl"
MODEL_ANABA_PATH = Path(__file__).parent.parent / "data" / "processed" / "lgbm_model_anaba.pkl"


# ── LightGBM ハイパーパラメータ ─────────────────────────────────────────────
PARAMS_NORMAL = {
    'objective':        'lambdarank',
    'metric':           'ndcg',
    'ndcg_eval_at':     [1, 3, 5],
    'learning_rate':    0.05,
    'num_leaves':       63,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq':     5,
    'lambda_l1':        0.1,
    'lambda_l2':        0.1,
    'verbose':          -1,
    'n_jobs':           -1,
}

# 穴馬モデル: 低人気馬（6番人気以下）が上位に来るケースに特化
# label_gain の重みを高人気馬と低人気馬で差をつける
PARAMS_ANABA = {
    **PARAMS_NORMAL,
    'num_leaves':       31,
    'min_data_in_leaf': 10,
    'learning_rate':    0.03,
}

NUM_ROUNDS = 500
EARLY_STOPPING = 50
TEST_RATIO = 0.2


def make_label(chaku: pd.Series, n_horses: int = 18) -> pd.Series:
    """着順を降順ラベルに変換（1着=最大値、着外=0）"""
    chaku = pd.to_numeric(chaku, errors='coerce')
    label = (n_horses + 1 - chaku).clip(lower=0)
    label = label.where(chaku.notna(), other=0)
    return label.astype(int)


def train_model(feat_df: pd.DataFrame, use_cols: list[str],
                params: dict, model_path: Path,
                anaba_mode: bool = False) -> None:
    """
    特徴量DFからLightGBM Rankerを学習して保存する。

    anaba_mode=True のとき、1着馬を6番人気以下に絞って
    穴馬特化モデルを学習する（学習データを穴馬シナリオで増強）。
    """
    df = feat_df.copy()

    # 着順ラベル
    df['_label'] = make_label(df[TARGET_COL])

    # レースグループキー
    df['_race_key'] = (df['日付'].astype(str) +
                       df['開催'].astype(str) +
                       df['Ｒ'].astype(str))

    # 着順不明の行を除外（テストセット向けデータ）
    df = df[df['_label'] > 0].copy()

    # 時系列分割: 日付順で最後のTEST_RATIOをテストに
    dates_sorted = sorted(df['日付_dt'].dropna().unique())
    split_idx    = int(len(dates_sorted) * (1 - TEST_RATIO))
    split_date   = dates_sorted[split_idx]

    train_df = df[df['日付_dt'] < split_date].copy()
    test_df  = df[df['日付_dt'] >= split_date].copy()

    print(f"  学習: {len(train_df)}行 ({train_df['_race_key'].nunique()}R) / "
          f"検証: {len(test_df)}行 ({test_df['_race_key'].nunique()}R)")

    # 穴馬モードでは学習セットの重みを調整
    if anaba_mode:
        pop = pd.to_numeric(train_df.get('人気', pd.Series(dtype=float)), errors='coerce')
        win = (train_df['_label'] == train_df.groupby('_race_key')['_label'].transform('max'))
        # 低人気で1着の行を3倍重み付け（穴馬シナリオ強調）
        train_df['_weight'] = np.where(win & (pop >= 6), 3.0, 1.0)
    else:
        train_df['_weight'] = 1.0

    X_train = train_df[use_cols].fillna(0).astype(float)
    y_train = train_df['_label'].values
    q_train = train_df.groupby('_race_key', sort=False).size().values
    w_train = train_df['_weight'].values

    X_test = test_df[use_cols].fillna(0).astype(float)
    y_test = test_df['_label'].values
    q_test = test_df.groupby('_race_key', sort=False).size().values

    ds_train = lgb.Dataset(X_train, label=y_train, group=q_train, weight=w_train,
                           feature_name=use_cols)
    ds_test  = lgb.Dataset(X_test,  label=y_test,  group=q_test,
                           feature_name=use_cols, reference=ds_train)

    callbacks = [
        lgb.early_stopping(EARLY_STOPPING, verbose=True),
        lgb.log_evaluation(50),
    ]

    model = lgb.train(
        params,
        ds_train,
        num_boost_round=NUM_ROUNDS,
        valid_sets=[ds_train, ds_test],
        valid_names=['train', 'valid'],
        callbacks=callbacks,
    )

    # 検証セットのNDCG@1
    preds = model.predict(X_test)
    test_df['_pred'] = preds
    ndcg1_list = []
    for _, grp in test_df.groupby('_race_key', sort=False):
        if len(grp) < 2:
            continue
        best_label = grp['_label'].max()
        pred_winner_label = grp.loc[grp['_pred'].idxmax(), '_label']
        ndcg1_list.append(1.0 if pred_winner_label == best_label else 0.0)
    ndcg1 = np.mean(ndcg1_list) if ndcg1_list else float('nan')
    print(f"  検証セット 勝ち馬的中率(NDCG@1近似): {ndcg1:.3f}")

    # 特徴量重要度（上位20件）
    imp = pd.Series(model.feature_importance(importance_type='gain'),
                    index=use_cols).sort_values(ascending=False)
    print("\n  特徴量重要度（top 20）:")
    print(imp.head(20).to_string())

    # 保存
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'features': use_cols}, f)
    print(f"\n  保存: {model_path}")


def main():
    if not MASTER_CSV.exists():
        print(f"master.csv が見つかりません: {MASTER_CSV}")
        print("先に build_dataset.py を実行してください。")
        sys.exit(1)

    print("=== master.csv 読み込み ===")
    df_raw = pd.read_csv(MASTER_CSV, encoding='utf-8-sig', low_memory=False)
    print(f"  {len(df_raw)}行 × {len(df_raw.columns)}列")

    # 日付変換
    df_raw['日付_dt'] = pd.to_datetime(df_raw['日付'].astype(str), format='%Y%m%d', errors='coerce')

    # 着順数値化（全角対応）
    _tr = str.maketrans('０１２３４５６７８９', '0123456789')
    if '着順' in df_raw.columns and '着順_num' not in df_raw.columns:
        df_raw['着順_num'] = pd.to_numeric(
            df_raw['着順'].astype(str).str.translate(_tr).str.extract(r'(\d+)')[0],
            errors='coerce'
        )

    print("\n=== 特徴量構築 ===")
    feat_df = build_features(df_raw, verbose=True)

    # 使用特徴量（feat_dfに存在する列のみ）
    use_cols = [c for c in FEATURE_COLS if c in feat_df.columns]
    print(f"\n使用特徴量: {len(use_cols)}列")

    # 欠損率確認
    missing = {c: feat_df[c].isna().mean() for c in use_cols}
    high_missing = {k: f"{v:.1%}" for k, v in missing.items() if v > 0.5}
    if high_missing:
        print("欠損率50%超の特徴量:")
        for k, v in high_missing.items():
            print(f"  {k}: {v}")

    print("\n=== 通常モデル 学習 ===")
    train_model(feat_df, use_cols, PARAMS_NORMAL, MODEL_PATH, anaba_mode=False)

    print("\n=== 穴馬モデル 学習 ===")
    train_model(feat_df, use_cols, PARAMS_ANABA, MODEL_ANABA_PATH, anaba_mode=True)

    print("\n=== 完了 ===")
    print(f"  通常モデル : {MODEL_PATH}")
    print(f"  穴馬モデル : {MODEL_ANABA_PATH}")


if __name__ == '__main__':
    main()
