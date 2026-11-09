# -*- coding: utf-8 -*-
"""海外の実績をモデルに反映する案の比較（本番モデル・master は書き換えない）。

前提: analysis/overseas_fetch.py で data/overseas_cache/ に重賞出走馬の全戦績を集めてある。

比べる案
  baseline : 現行と同じ（FEATURE_COLS、国内成績だけ）
  (a) 戦績に足す : 海外のレースを master に行として加えてから build_features。
                   前走・通算成績・近走平均に海外のレースが入る。加えた行は学習・評価から外す。
  (b) 特徴量を足す : FEATURE_COLS + 海外特徴量（ov_*）。予測日より前の海外出走だけを使う（リーク防止）。
  (c) 印だけ     : モデルは baseline のまま。海外実績馬に印を出す（ここでは baseline のずれの大きさで判断）

学習は src/train.py と同じ作り方（make_label＝1着が最大、日付で80:20の時系列分割、同じ LightGBM 設定）。
ただし早期終了は学習期間の末尾10%で行い、評価期間（最後の20%）は学習に一切使わない。
"""
import json, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")
from features import build_features, FEATURE_COLS, TARGET_COL   # noqa: E402

CACHE = ROOT / "data" / "overseas_cache"
JRA = "札幌|函館|福島|新潟|東京|中山|中京|京都|阪神|小倉"
NAR = ("大井", "川崎", "船橋", "浦和", "門別", "盛岡", "水沢", "金沢", "笠松", "名古屋", "園田", "姫路", "高知", "佐賀", "帯広", "札幌(地)")
PARAMS = {"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [1, 3, 5], "learning_rate": 0.05, "num_leaves": 63,
          "min_data_in_leaf": 20, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
          "lambda_l1": 0.1, "lambda_l2": 0.1, "verbose": -1, "n_jobs": -1, "seed": 42}
OV_COLS = ["ov_n", "ov_g1_n", "ov_win_n", "ov_grade_win_n", "ov_g1_best_score", "ov_top3_rate", "ov_last_chaku", "ov_weeks_since"]
NAR_COLS = ["nar_jpn_n", "nar_jpn_top3_n", "nar_jpn_win_n", "nar_jpn1_n", "off_prev", "off_prev_chaku", "off_prev_grade"]


def make_label(ch, n=18):
    ch = pd.to_numeric(ch, errors="coerce")
    return (n + 1 - ch).clip(lower=0).where(ch.notna(), 0).astype(int)


# ------------------------------------------------------------------ 海外の出走を読む
def load_overseas():
    rows = []
    for f in (CACHE / "horse").glob("*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        for c in d["rows"]:
            ven = c[1]
            if re.match(rf"^\d*({JRA})\d*$", ven):
                continue
            kind = "nar" if any(v in ven for v in NAR) else "ov"
            ymd = c[0].split("/")
            dist = next((x for x in c if re.match(r"^[芝ダ障]\d{3,4}$", x)), "")
            rows.append(dict(馬名=d["name"], 日付=int(ymd[0][2:] + ymd[1] + ymd[2]), 日付_dt=pd.Timestamp(c[0].replace("/", "-")),
                             開催=ven, レース名=c[4], 頭数=pd.to_numeric(c[6], errors="coerce"),
                             人気=pd.to_numeric(c[10], errors="coerce"), 着順=c[11],
                             着順_num=pd.to_numeric(c[11], errors="coerce"),
                             芝ダ=dist[:1] if dist else "芝", 距離=pd.to_numeric(dist[1:], errors="coerce") if dist else np.nan,
                             kind=kind,
                             grade=("G1" if re.search(r"\((G|Jpn)(I|1)\)", c[4]) else "G2" if re.search(r"\((G|Jpn)(II|2)\)", c[4])
                                    else "G3" if re.search(r"\((G|Jpn)(III|3)\)", c[4]) else "")))
    ov = pd.DataFrame(rows).drop_duplicates(["馬名", "日付", "レース名"]).sort_values(["馬名", "日付"])
    return ov


def nar_features(df, runs, m):
    """地方の交流重賞（Jpn）の成績と、本当の前走が master の外（海外・地方）だったか。予測日より前だけを使う。"""
    out = pd.DataFrame(0.0, index=df.index, columns=NAR_COLS)
    nar = runs[(runs["kind"] == "nar") & (runs["grade"] != "")]
    by_n = {n: g for n, g in nar.groupby("馬名")}
    by_all = {n: g for n, g in runs.groupby("馬名")}
    jra_last = m.groupby("馬名")["日付"].apply(lambda x: np.sort(x.values))
    for i in df.index[df["馬名"].isin(by_all)]:
        n, d0 = df.at[i, "馬名"], df.at[i, "日付"]
        g = by_n.get(n)
        if g is not None:
            p = g[g["日付"] < d0]
            out.at[i, "nar_jpn_n"] = len(p); out.at[i, "nar_jpn_top3_n"] = (p["着順_num"] <= 3).sum()
            out.at[i, "nar_jpn_win_n"] = (p["着順_num"] == 1).sum(); out.at[i, "nar_jpn1_n"] = (p["grade"] == "G1").sum()
        a = by_all[n]; a = a[a["日付"] < d0]
        if a.empty:
            continue
        last = a.iloc[-1]
        js = jra_last.get(n, np.array([]))
        jl = js[js < d0].max() if (js < d0).any() else 0
        if last["日付"] > jl:                      # master にある最後の国内走より新しい＝本当の前走は海外か地方
            out.at[i, "off_prev"] = 1.0
            out.at[i, "off_prev_chaku"] = last["着順_num"] if pd.notna(last["着順_num"]) else 0.0
            out.at[i, "off_prev_grade"] = {"G1": 3, "G2": 2, "G3": 1}.get(last["grade"], 0)
    return out


def ov_features(df, ov):
    """各行（馬名・日付）について、その日より前の海外出走だけで特徴量を作る。"""
    out = pd.DataFrame(index=df.index, columns=OV_COLS, dtype=float).fillna(0.0)
    out["ov_last_chaku"] = 0.0; out["ov_weeks_since"] = 999.0
    by = {n: g for n, g in ov.groupby("馬名")}
    idx = df.index[df["馬名"].isin(by)]
    for i in idx:
        g = by[df.at[i, "馬名"]]
        p = g[g["日付"] < df.at[i, "日付"]]
        if p.empty:
            continue
        ch = p["着順_num"]
        out.at[i, "ov_n"] = len(p)
        out.at[i, "ov_g1_n"] = (p["grade"] == "G1").sum()
        out.at[i, "ov_win_n"] = (ch == 1).sum()
        out.at[i, "ov_grade_win_n"] = ((ch == 1) & (p["grade"] != "")).sum()
        g1 = ch[p["grade"] == "G1"].dropna()
        out.at[i, "ov_g1_best_score"] = max(0.0, (6 - g1.min()) / 5) if len(g1) else 0.0      # G1で5着以内なら加点
        out.at[i, "ov_top3_rate"] = (ch <= 3).mean()
        out.at[i, "ov_last_chaku"] = ch.iloc[-1] if pd.notna(ch.iloc[-1]) else 0.0
        out.at[i, "ov_weeks_since"] = (df.at[i, "日付_dt"] - p["日付_dt"].iloc[-1]).days / 7
    return out


# ------------------------------------------------------------------ 学習と評価
def split(feat):
    d = feat[make_label(feat[TARGET_COL]) > 0].copy()
    d["_label"] = make_label(d[TARGET_COL])
    d["_rk"] = d["日付"].astype(str) + d["開催"].astype(str) + d["Ｒ"].astype(str)
    dates = sorted(d["日付_dt"].dropna().unique())
    t_test = dates[int(len(dates) * 0.8)]
    t_val = dates[int(len(dates) * 0.72)]
    return d[d["日付_dt"] < t_val], d[(d["日付_dt"] >= t_val) & (d["日付_dt"] < t_test)], d[d["日付_dt"] >= t_test], t_test


def fit(tr, va, cols):
    def ds(x, ref=None):
        x = x.sort_values(["日付_dt", "_rk"])
        return lgb.Dataset(x[cols].fillna(0).astype(float), label=x["_label"], group=x.groupby("_rk", sort=False).size().values,
                           feature_name=cols, reference=ref), x
    dtr, _ = ds(tr); dva, _ = ds(va, dtr)
    return lgb.train(PARAMS, dtr, num_boost_round=500, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])


def softmax_in_race(s, rk):
    e = np.exp(s - s.groupby(rk).transform("max"))
    return e / e.groupby(rk).transform("sum")


def evaluate(te, pred, tag, ovmask):
    t = te.copy(); t["pred"] = pred; t["wp"] = softmax_in_race(t["pred"], t["_rk"])
    t["prank"] = t.groupby("_rk")["pred"].rank(ascending=False, method="first")
    top = t[t["prank"] == 1]
    tan = pd.to_numeric(top["単勝配当"], errors="coerce").fillna(0).where(top[TARGET_COL] == 1, 0)
    res = dict(案=tag, レース数=t["_rk"].nunique(), 本命勝率=(top[TARGET_COL] == 1).mean() * 100, 本命複勝率=(top[TARGET_COL] <= 3).mean() * 100,
               本命単回収=tan.mean())
    o = t[ovmask.loc[t.index]]
    g = o[o["_grade"]]
    for lab, x in (("海外実績馬", o), ("うち重賞", g)):
        res[f"{lab}_頭数"] = len(x)
        res[f"{lab}_予測勝率"] = x["wp"].mean() * 100
        res[f"{lab}_実際勝率"] = (x[TARGET_COL] == 1).mean() * 100
        res[f"{lab}_予測順位"] = x["prank"].mean()
        res[f"{lab}_実着順"] = x[TARGET_COL].mean()
        res[f"{lab}_本命時複勝率"] = (x[x["prank"] == 1][TARGET_COL] <= 3).mean() * 100 if (x["prank"] == 1).any() else np.nan
    print(tag, {k: round(v, 2) if isinstance(v, float) else v for k, v in res.items()}, flush=True)
    return res, t


def main():
    runs = load_overseas()
    ov = runs[runs["kind"] == "ov"]
    print(f"海外の出走 {len(ov)}行 / {ov['馬名'].nunique()}頭、地方 {len(runs) - len(ov)}行（うち交流重賞 {((runs['kind']=='nar') & (runs['grade']!='')).sum()}行）", flush=True)
    m = pd.read_parquet(ROOT / "data" / "processed" / "master.parquet")
    m["日付_dt"] = pd.to_datetime(m["日付_dt"], errors="coerce")
    m = m.sort_values("日付_dt").reset_index(drop=True)

    # baseline / (b)
    feat = build_features(m.copy(), verbose=False)
    feat = pd.concat([feat, ov_features(feat, ov), nar_features(feat, runs, m)], axis=1)
    tr, va, te, t_test = split(feat)
    print(f"学習 {len(tr)} / 早期終了 {len(va)} / 評価 {len(te)}行（{pd.Timestamp(t_test).date()}以降）", flush=True)
    te["_grade"] = te["レース名"].astype(str).str.contains("G1|G2|G3")
    ovmask = te["ov_n"] > 0
    offmask = te["off_prev"] > 0
    jpnmask = te["nar_jpn_n"] > 0
    base_cols = [c for c in FEATURE_COLS if c in feat.columns]
    mb = fit(tr, va, base_cols); rb, tb = evaluate(te, mb.predict(te[base_cols].fillna(0).astype(float)), "baseline", ovmask)
    cols_b = base_cols + OV_COLS
    mbb = fit(tr, va, cols_b); rbb, tbb = evaluate(te, mbb.predict(te[cols_b].fillna(0).astype(float)), "(b)特徴量を足す", ovmask)
    imp = pd.Series(mbb.feature_importance("gain"), index=cols_b).rank(ascending=False)
    print("ov特徴量の重要度順位:", {c: int(imp[c]) for c in OV_COLS}, flush=True)
    cols_b2 = base_cols + OV_COLS + NAR_COLS
    mb2 = fit(tr, va, cols_b2); pb2 = mb2.predict(te[cols_b2].fillna(0).astype(float))
    rb2, tb2 = evaluate(te, pb2, "(b2)海外＋地方を足す", ovmask)
    imp2 = pd.Series(mb2.feature_importance("gain"), index=cols_b2).rank(ascending=False)
    print("地方特徴量の重要度順位:", {c: int(imp2[c]) for c in NAR_COLS}, flush=True)
    # 「本当の前走が海外・地方」の馬と、交流重賞の経験馬でも見る
    extra = []
    for tag, pr in (("baseline", tb), ("(b)", tbb), ("(b2)", tb2)):
        for lab, mk in (("前走がmaster外", offmask), ("交流重賞の経験あり", jpnmask)):
            x = pr[mk.loc[pr.index]]
            extra.append(dict(案=tag, 区分=lab, 頭数=len(x), 予測勝率=round(x["wp"].mean() * 100, 2), 実際勝率=round((x[TARGET_COL] == 1).mean() * 100, 2),
                              予測順位=round(x["prank"].mean(), 2), 実着順=round(x[TARGET_COL].mean(), 2),
                              予測複勝圏=round((x["prank"] <= 3).mean() * 100, 1), 実複勝率=round((x[TARGET_COL] <= 3).mean() * 100, 1)))
    E = pd.DataFrame(extra); print(E.to_string(), flush=True)
    E.to_csv(ROOT / "analysis" / "overseas_ab_offmaster.csv", index=False, encoding="utf-8-sig")

    # (a) 海外のレースを行として加える
    add = runs[runs["日付"] >= m["日付"].min()].copy()      # 海外と地方の両方を行として足す
    add = add.rename(columns={"芝ダ": "芝・ダ"})
    add["開催"] = "9海1"; add["Ｒ"] = 11; add["_aug"] = 1
    ma = pd.concat([m.assign(_aug=0), add[[c for c in add.columns if c in m.columns or c == "_aug"]]], ignore_index=True)
    ma = ma.sort_values("日付_dt", kind="stable").reset_index(drop=True)
    try:
        fa = build_features(ma.copy(), verbose=False)
        keep = fa["_aug"].fillna(0) == 0 if "_aug" in fa.columns else ~fa["開催"].astype(str).eq("9海1")
        fa = fa[keep].copy()
        fa = fa.merge(feat[["馬名", "日付", "開催", "Ｒ", "ov_n"]], on=["馬名", "日付", "開催", "Ｒ"], how="left")
        tra, vaa, tea, _ = split(fa)
        tea["_grade"] = tea["レース名"].astype(str).str.contains("G1|G2|G3")
        mA = fit(tra, vaa, base_cols); ra, ta = evaluate(tea, mA.predict(tea[base_cols].fillna(0).astype(float)), "(a)戦績に足す", tea["ov_n"].fillna(0) > 0)
    except Exception as e:
        print("(a) は build_features が通らず:", repr(e)[:300], flush=True); ra = {"案": "(a)戦績に足す", "エラー": repr(e)[:120]}

    R = pd.DataFrame([rb, rbb, rb2, ra])
    R.to_csv(ROOT / "analysis" / "overseas_ab_result.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(R.T.to_string(), flush=True)
    # 海外実績馬の一覧（評価期間）
    o = tb[ovmask.loc[tb.index]][["日付", "レース名", "馬名", TARGET_COL, "prank", "wp"]].copy()
    o["prank_b"] = tbb.loc[o.index, "prank"]; o["wp_b"] = tbb.loc[o.index, "wp"]
    o.sort_values("日付").to_csv(ROOT / "analysis" / "overseas_ab_horses.csv", index=False, encoding="utf-8-sig")
    print(o.sort_values("日付").tail(40).round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
