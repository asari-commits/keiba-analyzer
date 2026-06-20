"""
回収率バックテストシミュレーション。
学習済みモデルのテストセット（直近20%）で各買い方の回収率を計算する。
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_PATH       = Path(__file__).parent.parent / "data" / "processed" / "lgbm_model.pkl"
MODEL_ANABA_PATH = Path(__file__).parent.parent / "data" / "processed" / "lgbm_model_anaba.pkl"
MASTER_CSV       = Path(__file__).parent.parent / "data" / "processed" / "master.csv"

import sys
sys.path.insert(0, str(Path(__file__).parent))
from features import build_features, FEATURE_COLS, TARGET_COL


# ---------------------------------------------------------------------------
# 配当パース
# ---------------------------------------------------------------------------

def _parse_payout(s) -> float | None:
    """'670' → 670.0  '(12.1)' → None（非的中馬のオッズは無視）"""
    import math
    try:
        v = str(s).strip()
        if v.startswith('(') or v in ('', 'nan', 'None', 'NaN'):
            return None
        f = float(v)
        if math.isnan(f) or f <= 0:
            return None
        return f
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 予測 → レース単位でランキング
# ---------------------------------------------------------------------------

def predict_testset(feat_df: pd.DataFrame, model_path: Path = None) -> pd.DataFrame:
    """テストセット（直近20%）に対してモデル予測を実行し、ランク付きで返す"""
    if model_path is None:
        model_path = MODEL_PATH
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    model, use_cols = data['model'], data['features']

    df = feat_df.dropna(subset=[TARGET_COL]).copy()
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()

    X_test = test_df[use_cols].fillna(0).astype(float)
    test_df['pred_score'] = model.predict(X_test)

    race_key = test_df['日付'].astype(str) + test_df['開催'].astype(str) + test_df['Ｒ'].astype(str)
    test_df['pred_rank'] = test_df.groupby(race_key)['pred_score'].rank(ascending=False, method='min')

    return test_df


# ---------------------------------------------------------------------------
# 馬券シミュレーション
# ---------------------------------------------------------------------------

def simulate(test_df: pd.DataFrame, master_df: pd.DataFrame) -> dict:
    """
    各馬券種 × 買い方戦略の回収率を計算。
    master_df から配当情報を結合して使用。

    戻り値: {strategy_name: {"bets", "hits", "roi", "detail_df"}}
    """
    # 配当列をtest_dfに結合
    pay_cols = ['日付', '開催', 'Ｒ', '馬名', '着順_num',
                '単勝配当', '複勝配当', '馬連', '馬単', '３連複', '３連単', '人気']
    pay_cols = [c for c in pay_cols if c in master_df.columns]

    df = test_df.merge(
        master_df[pay_cols],
        on=['日付', '開催', 'Ｒ', '馬名'],
        how='left',
        suffixes=('', '_m')
    )

    race_key = df['日付'].astype(str) + df['開催'].astype(str) + df['Ｒ'].astype(str)
    df['race_key'] = race_key

    results = {}

    # ── 単勝（予測1位に100円） ──────────────────────────────────
    results['単勝_予測1位'] = _sim_tansho(df, bet_rank=1)

    # ── 複勝（予測1位に100円） ──────────────────────────────────
    results['複勝_予測1位'] = _sim_fukusho(df, bet_rank=1)

    # ── 複勝（予測1〜3位に各100円） ─────────────────────────────
    results['複勝_予測上位3頭'] = _sim_fukusho_multi(df, top_n=3)

    # ── 馬連（予測1・2位のBOX） ──────────────────────────────────
    results['馬連_予測1-2位'] = _sim_umaren(df)

    # ── ３連複（予測1〜3位のBOX） ────────────────────────────────
    results['３連複_予測上位3頭'] = _sim_sanrenpuku(df)

    # ── 単勝（予測1位 かつ 人気5番人気以下 = 穴狙い） ──────────
    results['単勝_穴馬狙い(6番人気以下)'] = _sim_tansho_anaba(df, min_popular=6)

    return results


def _sim_tansho(df, bet_rank=1):
    bets, hits, invested, returned = 0, 0, 0, 0
    for _, grp in df.groupby('race_key'):
        target = grp[grp['pred_rank'] == bet_rank]
        if target.empty:
            continue
        row = target.iloc[0]
        bets += 1
        invested += 100
        if row['着順_num'] == 1:
            pay = _parse_payout(row.get('単勝配当'))
            if pay:
                hits += 1
                returned += pay
    roi = (returned - invested) / invested * 100 if invested > 0 else 0
    return {'bets': bets, 'hits': hits, 'invested': invested,
            'returned': int(returned) if returned == returned else 0, 'hit_rate': hits/bets if bets else 0, 'roi': roi}


def _sim_fukusho(df, bet_rank=1):
    bets, hits, invested, returned = 0, 0, 0, 0
    for _, grp in df.groupby('race_key'):
        target = grp[grp['pred_rank'] == bet_rank]
        if target.empty:
            continue
        row = target.iloc[0]
        bets += 1
        invested += 100
        if pd.notna(row.get('着順_num')) and row['着順_num'] <= 3:
            pay = _parse_payout(row.get('複勝配当'))
            if pay:
                hits += 1
                returned += pay
    roi = (returned - invested) / invested * 100 if invested > 0 else 0
    return {'bets': bets, 'hits': hits, 'invested': invested,
            'returned': int(returned) if returned == returned else 0, 'hit_rate': hits/bets if bets else 0, 'roi': roi}


def _sim_fukusho_multi(df, top_n=3):
    bets, hits, invested, returned = 0, 0, 0, 0
    for _, grp in df.groupby('race_key'):
        targets = grp[grp['pred_rank'] <= top_n]
        for _, row in targets.iterrows():
            bets += 1
            invested += 100
            if pd.notna(row.get('着順_num')) and row['着順_num'] <= 3:
                pay = _parse_payout(row.get('複勝配当'))
                if pay:
                    hits += 1
                    returned += pay
    roi = (returned - invested) / invested * 100 if invested > 0 else 0
    return {'bets': bets, 'hits': hits, 'invested': invested,
            'returned': int(returned) if returned == returned else 0, 'hit_rate': hits/bets if bets else 0, 'roi': roi}


def _sim_umaren(df):
    bets, hits, invested, returned = 0, 0, 0, 0
    for _, grp in df.groupby('race_key'):
        top2 = grp[grp['pred_rank'] <= 2]
        if len(top2) < 2:
            continue
        bets += 1
        invested += 100
        actual_top2 = set(grp[grp['着順_num'] <= 2]['馬名'].tolist())
        pred_top2   = set(top2['馬名'].tolist())
        if pred_top2 == actual_top2:
            # 馬連配当は1着馬行に格納
            pay_row = grp[grp['着順_num'] == 1]
            if not pay_row.empty:
                pay = _parse_payout(pay_row.iloc[0].get('馬連'))
                if pay:
                    hits += 1
                    returned += pay
    roi = (returned - invested) / invested * 100 if invested > 0 else 0
    return {'bets': bets, 'hits': hits, 'invested': invested,
            'returned': int(returned) if returned == returned else 0, 'hit_rate': hits/bets if bets else 0, 'roi': roi}


def _sim_sanrenpuku(df):
    bets, hits, invested, returned = 0, 0, 0, 0
    for _, grp in df.groupby('race_key'):
        top3 = grp[grp['pred_rank'] <= 3]
        if len(top3) < 3:
            continue
        bets += 1
        invested += 100
        actual_top3 = set(grp[grp['着順_num'] <= 3]['馬名'].tolist())
        pred_top3   = set(top3['馬名'].tolist())
        if pred_top3 == actual_top3:
            pay_row = grp[grp['着順_num'] == 1]
            if not pay_row.empty:
                pay = _parse_payout(pay_row.iloc[0].get('３連複'))
                if pay:
                    hits += 1
                    returned += pay
    roi = (returned - invested) / invested * 100 if invested > 0 else 0
    return {'bets': bets, 'hits': hits, 'invested': invested,
            'returned': int(returned) if returned == returned else 0, 'hit_rate': hits/bets if bets else 0, 'roi': roi}


def _sim_tansho_anaba(df, min_popular=6):
    """予測1位 かつ 人気が低い（穴馬）のみ単勝"""
    bets, hits, invested, returned = 0, 0, 0, 0
    for _, grp in df.groupby('race_key'):
        target = grp[grp['pred_rank'] == 1]
        if target.empty:
            continue
        row = target.iloc[0]
        pop = pd.to_numeric(row.get('人気'), errors='coerce')
        if pd.isna(pop) or pop < min_popular:
            continue  # 人気馬はスキップ
        bets += 1
        invested += 100
        if row['着順_num'] == 1:
            pay = _parse_payout(row.get('単勝配当'))
            if pay:
                hits += 1
                returned += pay
    roi = (returned - invested) / invested * 100 if invested > 0 else 0
    return {'bets': bets, 'hits': hits, 'invested': invested,
            'returned': int(returned) if returned == returned else 0, 'hit_rate': hits/bets if bets else 0, 'roi': roi}


# ---------------------------------------------------------------------------
# サマリー表示
# ---------------------------------------------------------------------------

def compare_models(feat_df: pd.DataFrame, master_df: pd.DataFrame) -> dict:
    """
    通常モデルと穴馬モデルのシミュレーション結果を両方返す。
    戻り値: {'main': results_dict, 'anaba': results_dict}
    """
    out = {}
    for label, path in [('main', MODEL_PATH), ('anaba', MODEL_ANABA_PATH)]:
        if not path.exists():
            print(f"  {label} モデルが見つかりません: {path}")
            continue
        test_df = predict_testset(feat_df, model_path=path)
        out[label] = simulate(test_df, master_df)
    return out


def print_comparison(comparison: dict) -> None:
    """2モデルの回収率を並べて表示"""
    labels = {'main': '通常モデル', 'anaba': '穴馬モデル'}
    strategies = list(next(iter(comparison.values())).keys())

    w = 30
    print(f"\n{'='*85}")
    print("【モデル比較】通常モデル vs 穴馬モデル（テストセット: 直近20%のレース）")
    print(f"{'='*85}")
    header = f"{'戦略':<{w}}"
    for label in comparison:
        header += f"  {'回収率':>8}({'的中率':>6})"
    print(header)
    print('-' * 85)
    for s in strategies:
        row = f"{s:<{w}}"
        for label, results in comparison.items():
            if s in results:
                r = results[s]
                row += f"  {r['roi']:>+8.1f}%({r['hit_rate']:>5.1%})"
        print(row)
    print()
    print("  ※ 回収率がプラスほど収益、的中率は予測1位（または対象馬）が的中した割合")


def print_simulation_results(results: dict) -> None:
    print(f"\n{'='*75}")
    print("【回収率シミュレーション結果】（テストセット: 直近20%のレース）")
    print(f"{'='*75}")
    print(f"{'戦略':<28} {'購入数':>7} {'的中':>6} {'的中率':>7} {'投資額':>9} {'回収額':>9} {'回収率':>8}")
    print(f"{'-'*75}")
    for name, r in results.items():
        print(
            f"{name:<28} {r['bets']:>7,} {r['hits']:>6,} "
            f"{r['hit_rate']:>7.1%} "
            f"{r['invested']:>9,}円 {r['returned']:>9,}円 "
            f"{r['roi']:>+7.1f}%"
        )
