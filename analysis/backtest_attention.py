# -*- coding: utf-8 -*-
"""注目馬（逆行好走・恵まれ好走）の判定方法を、次走成績で比べる。

  python analysis/backtest_attention.py

比べるもの
  新式: 位置比 = コースの例年の帯複勝率 ÷ 例年のニュートラル（attention_horses.py と同じ）
  旧式: 位置比 = その日その場路面の帯複勝率 ÷ その日のニュートラル（帯の頭数5未満は出さない）
  bias-tracker: 馬ノートで ソース=bias-tracker かつ タグに「バイアス逆行で好走」（実際の登録）
いずれも「次走」= master 上でその馬の次の出走。次走が master に無いものは除く。
ペースの z は master の RPCI をコース（場×路面×距離）平均・sd で標準化し、符号を反転（＋＝前傾）。
基準線: 次走の人気ごとの全馬平均（勝率・複勝率）。超過 = 実績 − 人気から期待される値。
※例年値は全期間で計算しているので、少しだけ先の情報を含む（帯ごとの複勝率は年によってほぼ変わらない）。
"""
import os, sys
import numpy as np, pandas as pd

sys.stdout.reconfigure(encoding='utf-8'); pd.set_option('display.width', 300)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
n_ = lambda s: pd.to_numeric(s, errors='coerce')
KEY = ['日付', '開催', 'Ｒ']
BANDS = [0, .2, .45, .7, 9]


def load():
    M = pd.read_parquet(os.path.join(ROOT, 'data', 'processed', 'master.parquet'))
    for c in ['頭数', '4角', '人気', 'RPCI', '単勝配当', '複勝配当']:
        M[c] = n_(M[c])
    M['着'] = n_(M['着順_num'])
    M = M.dropna(subset=['着', '頭数'])
    M = M[M['芝・ダ'].isin(['芝', 'ダ'])].copy()
    M['場'] = M['開催'].astype(str).str.extract(r'([中名新札函小京阪東福])')[0]
    M['距'] = n_(M['距離'].astype(str).str.extract(r'(\d+)')[0])
    M['dt'] = pd.to_datetime('20' + M['日付'].astype(str), format='%Y%m%d', errors='coerce')
    M['rel4'] = M['4角'] / M['頭数']
    M['帯'] = pd.cut(M['rel4'], BANDS, labels=['A', 'B', 'C', 'D']).astype(str)
    M['ag'] = n_(M['上り3F'])
    M['上順'] = M.groupby(KEY)['ag'].rank(method='min')
    M['f'] = (M['着'] <= 3).astype(int); M['w'] = (M['着'] == 1).astype(int)
    M['rid'] = M['日付'].astype(str) + M['開催'].astype(str) + M['Ｒ'].astype(str)
    return M


def payouts_are_per_horse(M):
    """単勝配当が勝ち馬の行にだけ入っているか（レース全行に同じ値が入っている形式かを判定）。"""
    g = M[M['単勝配当'].notna()]
    return (g['着'] == 1).mean() > 0.9


def add_ratios(M):
    races = M.drop_duplicates('rid')
    # 新式: コースの例年値（場×路面×距離が30R以上ならそれ、なければ場×路面）
    out = {}
    for keys in (['場', '芝・ダ', '距'], ['場', '芝・ダ']):
        neu = races.groupby(keys).apply(lambda g: pd.Series({'R': len(g), 'neu': (3 / g['頭数']).mean()}))
        rate = M.groupby(keys + ['帯'])['f'].mean()
        out[len(keys)] = (neu, rate)
    neu3, rate3 = out[3]; neu2, rate2 = out[2]
    k3 = list(zip(M['場'], M['芝・ダ'], M['距']))
    use3 = np.array([(k in neu3.index) and (neu3.loc[k, 'R'] >= 30) for k in k3])
    r3 = np.array([rate3.get(k + (b,), np.nan) / neu3.loc[k, 'neu'] if u else np.nan
                   for k, b, u in zip(k3, M['帯'], use3)])
    r2 = np.array([rate2.get((v, s, b), np.nan) / neu2.loc[(v, s), 'neu'] if (v, s) in neu2.index else np.nan
                   for v, s, b in zip(M['場'], M['芝・ダ'], M['帯'])])
    M['比_新'] = np.where(use3, r3, r2)
    # 旧式: その日その場路面
    M['day'] = M['日付'].astype(str) + M['場'] + M['芝・ダ']
    dneu = races.assign(day=races['日付'].astype(str) + races['場'] + races['芝・ダ']) \
                .groupby('day')['頭数'].apply(lambda s: (3 / s).mean())
    dr = M.groupby(['day', '帯'])['f'].agg(['mean', 'size'])
    M['比_旧'] = [dr.loc[(d, b), 'mean'] / dneu[d] if (d, b) in dr.index and dr.loc[(d, b), 'size'] >= 5 else np.nan
                 for d, b in zip(M['day'], M['帯'])]
    # ペース z（RPCI は大きいほど後傾なので符号を反転）
    st = races.groupby(['場', '芝・ダ', '距'])['RPCI'].agg(['mean', 'std'])
    mu = st['mean'].reindex(list(zip(M['場'], M['芝・ダ'], M['距']))).values
    sd = st['std'].reindex(list(zip(M['場'], M['芝・ダ'], M['距']))).values
    M['z'] = -(M['RPCI'] - mu) / sd
    return M


def add_next(M, per_horse):
    M = M.sort_values(['馬名', 'dt', 'Ｒ'])
    g = M.groupby('馬名')
    for c in ['着', '人気', 'f', 'w', '単勝配当', '複勝配当', 'dt']:
        M['次_' + c] = g[c].shift(-1)
    if not per_horse:  # レース単位の配当なら、着順で当たり分だけにする
        M['次_単勝配当'] = np.where(M['次_着'] == 1, M['次_単勝配当'], 0)
    M['次_単勝配当'] = M['次_単勝配当'].fillna(0)
    M['次_複勝配当'] = np.where(M['次_着'] <= 3, M['次_複勝配当'].fillna(0), 0)
    return M


def summarize(X, exp, label):
    X = X[X['次_着'].notna()]
    n = len(X)
    if n == 0:
        return dict(判定=label, n=0)
    e = X['次_人気'].map(exp['f']); ew = X['次_人気'].map(exp['w'])
    return dict(判定=label, n=n, 次走勝率=round(X['次_w'].mean() * 100, 1), 次走複勝率=round(X['次_f'].mean() * 100, 1),
                人気からの期待複勝率=round(e.mean() * 100, 1), 複勝の超過=round((X['次_f'] - e).mean() * 100, 1),
                勝率の超過=round((X['次_w'] - ew).mean() * 100, 1),
                単回収=round(X['次_単勝配当'].sum() / n, 0), 複回収=round(X['次_複勝配当'].sum() / n, 0),
                次走平均人気=round(X['次_人気'].mean(), 1))


def main():
    M = load()
    per = payouts_are_per_horse(M)
    print(f"master {M['日付'].astype(str).min()}〜{M['日付'].astype(str).max()} / 配当は{'勝ち馬の行だけ' if per else 'レースの全行'}に入っている形式")
    M = add_ratios(M)
    M = add_next(M, per)
    exp = M.groupby('人気')[['f', 'w']].mean()
    F = M[M['f'] == 1]
    sets = {
        '新式 逆行好走（全）': F[F['比_新'] < 0.6],
        '新式 逆行好走・高（z>0）': F[(F['比_新'] < 0.6) & (F['z'] > 0)],
        '新式 逆行好走・低（z<=0）': F[(F['比_新'] < 0.6) & (F['z'] <= 0)],
        '旧式 逆行好走（全）': F[F['比_旧'] < 0.6],
        '旧式 逆行好走・高（z>0）': F[(F['比_旧'] < 0.6) & (F['z'] > 0)],
        '新式 恵まれ好走（全）': F[(F['比_新'] > 1.5) & (F['人気'] >= 4)],
        '新式 恵まれ好走・強（上がり7位以下）': F[(F['比_新'] > 1.5) & (F['人気'] >= 4) & (F['上順'] >= 7)],
        '旧式 恵まれ好走（全）': F[(F['比_旧'] > 1.5) & (F['人気'] >= 4)],
        '参考: 3着以内の全馬': F,
    }
    print("\n■ 全期間（次走が master にある分）")
    print(pd.DataFrame([summarize(v, exp, k) for k, v in sets.items()]).to_string(index=False))

    # 新旧の違いが出る部分だけ（片方でだけ逆行好走になる馬）
    a = F['比_新'] < 0.6; b = F['比_旧'] < 0.6
    print("\n■ 新旧で判定が分かれる馬")
    print(pd.DataFrame([summarize(F[a & b], exp, '両方で逆行'), summarize(F[a & ~b.fillna(False)], exp, '新式だけ逆行'),
                        summarize(F[~a & b.fillna(False)], exp, '旧式だけ逆行')]).to_string(index=False))

    # 8〜9月：bias-tracker の実際の登録との直接比較
    import race_notes as rn
    N = rn.load_notes()
    N['馬名'] = N['馬名'].map(rn.normalize_name)
    bt = N[(N['ソース'] == 'bias-tracker') & N['タグ'].astype(str).str.contains('バイアス逆行')].copy()
    bt['日付'] = bt['日付'].astype(str).str[-6:]
    M['日付s'] = M['日付'].astype(str)
    B = bt.merge(M, left_on=['馬名', '日付'], right_on=['馬名', '日付s'], how='inner')
    per_ = (M['dt'] >= '2026-08-01') & (M['dt'] <= '2026-09-30')
    Fp = F[per_.reindex(F.index, fill_value=False)]
    print(f"\n■ 2026年8〜9月の直接比較（bias-tracker の登録 {len(bt)}件、master と結合 {len(B)}件）")
    print(pd.DataFrame([summarize(B, exp, 'bias-tracker 実登録（4〜5着も含む）'),
                        summarize(B[B['着'] <= 3], exp, 'bias-tracker 実登録のうち3着以内'),
                        summarize(Fp[Fp['比_新'] < 0.6], exp, '新式 逆行好走（同期間）'),
                        summarize(Fp[(Fp['比_新'] < 0.6) & (Fp['z'] > 0)], exp, '新式 逆行好走・高（同期間）')]).to_string(index=False))
    print(f"  ※次走が10/4までに済んでいるものだけ。bias-tracker 登録のうち次走済み {B['次_着'].notna().sum()}件")


if __name__ == '__main__':
    main()
