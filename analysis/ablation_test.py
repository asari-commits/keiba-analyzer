# -*- coding: utf-8 -*-
"""
特徴量アブレーションテスト。
「重要度は高いが好走と単独相関がほぼゼロ」の特徴量を抜いて再学習し、
OOS（時系列ホールドアウト）の 勝ち馬的中/複勝的中/単複回収率 がどう変わるか比較する。
本番モデルは一切上書きしない（メモリ上で学習・評価するだけ）。

使い方: python analysis/ablation_test.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import lightgbm as lgb

import train as T
from features import build_features, FEATURE_COLS, filter_recent_years

ABLATE = ['career_num', 'class_diff', 'race_front_ratio', 'interval_weeks', 'course_pci_mean']


def _parse_pay(v):
    try:
        return float(str(v).replace(',', '').strip())
    except Exception:
        return np.nan


def load_feat():
    df = pd.read_csv(T.MASTER_CSV, encoding='utf-8-sig', low_memory=False)
    if '日付_dt' in df.columns and pd.to_datetime(df['日付_dt'], errors='coerce').notna().mean() > 0.9:
        df['日付_dt'] = pd.to_datetime(df['日付_dt'], errors='coerce')
    else:
        df['日付_dt'] = pd.to_datetime(df['日付'].astype(str).str.zfill(6), format='%y%m%d', errors='coerce')
    df = filter_recent_years(df).reset_index(drop=True)
    tr = str.maketrans('０１２３４５６７８９', '0123456789')
    if '着順' in df.columns and '着順_num' not in df.columns:
        df['着順_num'] = pd.to_numeric(df['着順'].astype(str).str.translate(tr).str.extract(r'(\d+)')[0], errors='coerce')
    return build_features(df, verbose=False)


def split(feat_df):
    df = feat_df.copy()
    df['_label'] = T.make_label(df[T.TARGET_COL])
    df['_race_key'] = df['日付'].astype(str) + df['開催'].astype(str) + df['Ｒ'].astype(str)
    df = df[df['_label'] > 0].copy()
    ds = sorted(df['日付_dt'].dropna().unique())
    split_date = ds[int(len(ds) * (1 - T.TEST_RATIO))]
    return df[df['日付_dt'] < split_date].copy(), df[df['日付_dt'] >= split_date].copy()


def train_and_eval(train_df, test_df, use_cols, label):
    Xtr = train_df[use_cols].fillna(0).astype(float); ytr = train_df['_label'].values
    qtr = train_df.groupby('_race_key', sort=False).size().values
    Xte = test_df[use_cols].fillna(0).astype(float); yte = test_df['_label'].values
    qte = test_df.groupby('_race_key', sort=False).size().values
    dtr = lgb.Dataset(Xtr, label=ytr, group=qtr, feature_name=use_cols)
    dte = lgb.Dataset(Xte, label=yte, group=qte, feature_name=use_cols, reference=dtr)
    model = lgb.train(T.PARAMS_NORMAL, dtr, num_boost_round=T.NUM_ROUNDS,
                      valid_sets=[dte], valid_names=['valid'],
                      callbacks=[lgb.early_stopping(T.EARLY_STOPPING, verbose=False)])
    t = test_df.copy()
    t['_pred'] = model.predict(Xte)
    t['_tan'] = t['単勝配当'].map(_parse_pay) if '単勝配当' in t.columns else np.nan
    t['_fuku'] = t['複勝配当'].map(_parse_pay) if '複勝配当' in t.columns else np.nan
    t['_chaku'] = pd.to_numeric(t['着順_num'], errors='coerce')
    win = ndcg = tan_hit = fuku_hit = tan_ret = fuku_ret = 0.0
    nR = 0
    for _, g in t.groupby('_race_key', sort=False):
        if len(g) < 2:
            continue
        nR += 1
        top1 = g.loc[g['_pred'].idxmax()]
        c = top1['_chaku']
        if c == 1: tan_hit += 1
        if c <= 3: fuku_hit += 1
        # 的中時のみ配当を回収（100円賭け）
        tan_ret += (top1['_tan'] if c == 1 and pd.notna(top1['_tan']) else 0.0)
        fuku_ret += (top1['_fuku'] if c <= 3 and pd.notna(top1['_fuku']) else 0.0)
        ndcg += 1.0 if g['_pred'].idxmax() == g['_label'].idxmax() else 0.0
    inv = nR * 100.0
    return {
        'label': label, 'n_feat': len(use_cols), 'nR': nR,
        '勝ち馬的中': ndcg / nR * 100,
        '単勝的中': tan_hit / nR * 100,
        '複勝的中': fuku_hit / nR * 100,
        '単回収': tan_ret / inv * 100,
        '複回収': fuku_ret / inv * 100,
        'best_iter': model.best_iteration,
    }


def main():
    print("=== 特徴量構築 ===")
    feat = load_feat()
    tr, te = split(feat)
    print(f"学習 {len(tr)}行 / 検証 {len(te)}行 ({te['_race_key'].nunique()}R)")
    base_cols = [c for c in FEATURE_COLS if c in feat.columns]
    abl_cols = [c for c in base_cols if c not in ABLATE]
    print(f"除外特徴量: {ABLATE}")
    print(f"フル {len(base_cols)}特徴 / 除外後 {len(abl_cols)}特徴\n")
    rows = []
    for lbl, cols in [('フル(現行)', base_cols), ('5特徴量除外', abl_cols)]:
        print(f"--- {lbl} 学習中 ---")
        rows.append(train_and_eval(tr, te, cols, lbl))
    out = pd.DataFrame(rows).set_index('label')
    for c in ['勝ち馬的中', '単勝的中', '複勝的中', '単回収', '複回収']:
        out[c] = out[c].round(2)
    print("\n=== OOS比較（検証セット） ===")
    print(out[['n_feat', 'nR', '勝ち馬的中', '単勝的中', '複勝的中', '単回収', '複回収', 'best_iter']].to_string())
    d = out.loc['5特徴量除外'] - out.loc['フル(現行)']
    print("\n=== 差分（除外 − フル。プラス=除外で改善）===")
    print({k: round(float(d[k]), 2) for k in ['勝ち馬的中', '単勝的中', '複勝的中', '単回収', '複回収']})


if __name__ == '__main__':
    main()
