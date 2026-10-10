# -*- coding: utf-8 -*-
"""コースの構造（例年値）に逆らって好走した馬／構造に恵まれて好走した馬を抽出し、注目馬リストに追記する。

【注意】記録用。予想の材料には使わない（2026/10/10）。
  analysis/backtest_attention.py で5年分（2021/10〜2026/10）を検証した結果、逆行好走も恵まれ好走も
  次走は人気どおり（逆行好走 n=5,663: 複勝率+0.4pt・単回収77%、恵まれ好走 n=7,314: +0.1pt）で、
  判定基準をその日の傾向にしてもコースの例年値にしても予測力が無かった。
  そのため race_factors.py での表示も、馬ノートへの登録も行っていない。

  python analysis/attention_horses.py <YYYYMMDD> [<YYYYMMDD> ...]
  python analysis/attention_horses.py --rebuild      # day_results にある全日付で作り直す

入力: data/raw/day_results/<YYYYMMDD>.csv と _meta.csv（fetch_day_results.py の出力）
出力: data/processed/attention_horses.csv（指定した日付の分は作り直して差し替える）

判定式（2026/10/10 改訂）
  ・バイアスの軸は4角位置だけ。上がり順位は位置の結果なので判定には使わない
  ・位置比 = そのコースの例年の「その4角帯の複勝率」÷ 例年のニュートラル（3÷頭数の平均）
      例年値は master 全期間。場×路面×距離で30R以上あればそれを、なければ場×路面を使う
      ※旧式（〜10/10）は「その日の帯の複勝率」を使っていたが、同じ日の前半→後半で r≒0、
        翌日へも r≒0.1 と持ち越されないと検証されたため、コースの構造に切り替えた
  ・逆行好走 = 3着以内 かつ 位置比<0.6
      価値: レースが後傾(z<=0)なら「高」、前傾(z>0)なら「低」
            （前傾は前がつぶれて後方に有利な流れなので、後方からの好走はペースに助けられた側。
             10/10までは向きを逆にしていた。データ上も z<=0 が +0.7pt、z>0 が +0.1pt で差は小さい）
            同じレースから2頭以上出たら「中」（レース自体が特殊）。ペースが出せない距離は「不明」
  ・恵まれ好走 = 3着以内 かつ 位置比>1.5 かつ 4番人気以下 → 次走の割引材料
      割引: 上がり7位以下なら「強」（位置だけで残った）、それ以外は「軽」
  z = (後半3F差 − コース平均) ÷ sd（lap_course_profile.parquet、2016年以降）
"""
import glob, os, sys
import numpy as np, pandas as pd

sys.stdout.reconfigure(encoding='utf-8'); pd.set_option('display.width', 300)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAY = os.path.join(ROOT, 'data', 'raw', 'day_results')
OUT = os.path.join(ROOT, 'data', 'processed', 'attention_horses.csv')
PROF = os.path.join(ROOT, 'data', 'processed', 'lap_course_profile.parquet')
MASTER = os.path.join(ROOT, 'data', 'processed', 'master.parquet')
V1 = {'札幌': '札', '函館': '函', '福島': '福', '新潟': '新', '東京': '東',
      '中山': '中', '中京': '名', '京都': '京', '阪神': '阪', '小倉': '小'}
MIN_R = 30
BANDS = [0, .2, .45, .7, 9]


def course_baseline():
    """コース（場×路面×距離 と 場×路面）ごとの 4角帯別複勝率・ニュートラル・レース数。"""
    n_ = lambda s: pd.to_numeric(s, errors='coerce')
    M = pd.read_parquet(MASTER, columns=['日付', '開催', 'Ｒ', '着順_num', '頭数', '4角', '芝・ダ', '距離'])
    M['着'] = n_(M['着順_num']); M['頭数'] = n_(M['頭数']); M['4角'] = n_(M['4角'])
    M = M.dropna(subset=['着', '頭数', '4角'])
    M = M[M['芝・ダ'].isin(['芝', 'ダ'])].copy()
    inv = {v: k for k, v in V1.items()}
    M['場'] = M['開催'].astype(str).str.extract(r'([中名新札函小京阪東福])')[0].map(inv)
    M['距離'] = n_(M['距離'].astype(str).str.extract(r'(\d+)')[0])
    M['帯'] = pd.cut(M['4角'] / M['頭数'], BANDS, labels=['A', 'B', 'C', 'D']).astype(str)
    M['f'] = (M['着'] <= 3).astype(int)
    M['rid'] = M['日付'].astype(str) + M['開催'].astype(str) + M['Ｒ'].astype(str)
    out = {}
    for keys in (['場', '芝・ダ', '距離'], ['場', '芝・ダ']):
        for k, g in M.groupby(keys):
            races = g.drop_duplicates('rid')
            out[tuple(k)] = dict(R=len(races), neu=(3 / races['頭数']).mean(),
                                 rate=g.groupby('帯')['f'].mean().to_dict())
    return out


def ratio(base, ven, sd, dist, band):
    b, lvl = base.get((ven, sd, dist)), '場×路面×距離'
    if b is None or b['R'] < MIN_R:
        b, lvl = base.get((ven, sd)), '場×路面'
    if b is None or band not in b['rate']:
        return np.nan, np.nan, ''
    return b['rate'][band] / b['neu'], b['rate'][band], lvl


def extract(ymd, base):
    T = pd.read_csv(os.path.join(DAY, f"{ymd}.csv"), encoding='utf-8-sig')
    L = pd.read_csv(os.path.join(DAY, f"{ymd}_meta.csv"), encoding='utf-8-sig')
    T = T[T['芝ダ'].isin(['芝', 'ダ'])].copy()
    T = T[T['pos4'].notna() & T['着'].notna()]          # 競走中止などはその馬だけ除く
    T['rel4'] = T['pos4'] / T['頭数']
    T['上順'] = T.groupby(['場', 'R'])['上3F'].rank(method='min')
    T['4角帯'] = pd.cut(T['rel4'], BANDS, labels=['A', 'B', 'C', 'D']).astype(str)
    T['f'] = (T['着'] <= 3).astype(int)
    P = pd.read_parquet(PROF)

    def z(r):
        p = P[(P['venue'] == V1.get(r['場'])) & (P['TD'] == r['芝ダ']) & (P['距離'] == r['距離'])]
        if not len(p) or pd.isna(r['後半3F差']):
            return np.nan
        return (r['後半3F差'] - p['後半3F差平均'].iloc[0]) / p['後半3F差sd'].iloc[0]
    L['z'] = L.apply(z, axis=1)
    T = T.merge(L[['場', 'R', 'z']], on=['場', 'R'], how='left')

    rr = [ratio(base, v, s, d, b) for v, s, d, b in zip(T['場'], T['芝ダ'], T['距離'], T['4角帯'])]
    T['位置比'] = [x[0] for x in rr]
    T['例年帯複勝率'] = [x[1] for x in rr]
    T['基準'] = [x[2] for x in rr]

    F = T[T['f'] == 1].copy()
    rev = F[F['位置比'] < 0.6].copy()
    rev['区分'] = '逆行好走'
    n_same = rev.groupby(['場', 'R'])['馬番'].transform('size')
    rev['評価'] = np.where(n_same >= 2, '中', np.where(rev['z'].isna(), '不明',
                                                   np.where(rev['z'] <= 0, '高', '低')))
    rev['意味'] = '記録用（次走の予測力なし）'
    luck = F[(F['位置比'] > 1.5) & (F['人気'] >= 4)].copy()
    luck['区分'] = '恵まれ好走'
    luck['評価'] = np.where(luck['上順'] >= 7, '強', '軽')
    luck['意味'] = '記録用（次走の予測力なし）'
    A = pd.concat([rev, luck], ignore_index=True)
    A['日付'] = ymd
    cols = ['日付', '場', 'R', '芝ダ', '距離', '馬場', '頭数', '馬番', '馬名', '着', '人気', 'オッズ',
            'rel4', '4角帯', '上順', '例年帯複勝率', '位置比', '基準', 'z', '区分', '評価', '意味']
    A = A[cols]
    for c, d in (('rel4', 2), ('位置比', 2), ('z', 2), ('例年帯複勝率', 3)):
        A[c] = A[c].round(d)
    for c in ('着', '人気', '上順', '頭数', '距離'):
        A[c] = A[c].astype(int)
    return A


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__); sys.exit(1)
    rebuild = args == ['--rebuild']
    days = (sorted(os.path.basename(f)[:8] for f in glob.glob(os.path.join(DAY, '*_meta.csv')))
            if rebuild else args)
    base = course_baseline()
    new = pd.concat([extract(d, base) for d in days], ignore_index=True)
    if os.path.exists(OUT) and not rebuild:
        old = pd.read_csv(OUT, encoding='utf-8-sig', dtype={'日付': str})
        old = old[~old['日付'].isin(days)]      # 同じ日付は作り直す（条件から外れた馬を残さない）
        new = pd.concat([old, new], ignore_index=True)
    new = new.sort_values(['日付', '場', 'R', '着']).reset_index(drop=True)
    new.to_csv(OUT, index=False, encoding='utf-8-sig')
    show = new[new['日付'].isin(days)]
    print(f"保存 {OUT}  全{len(new)}件（今回 {len(show)}件）")
    for (k, e), g in show.groupby(['区分', '評価']):
        print(f"\n[{k}・{e}] {len(g)}頭")
        print(g[['日付', '場', 'R', '芝ダ', '距離', '馬名', '着', '人気', 'rel4', '例年帯複勝率',
                 '位置比', '上順', 'z']].to_string(index=False))


if __name__ == '__main__':
    main()
