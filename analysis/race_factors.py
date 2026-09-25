# -*- coding: utf-8 -*-
"""レース分析：4ファクター統合版

  python analysis/race_factors.py <race_id> <場(漢字1字)> <芝|ダ> <距離> [クラス名で絞る]
  例) python analysis/race_factors.py 202609040707 阪 芝 1400 1勝

出力
  ① 3角→4角の位置変化（押し上げ／下げ）  ← 4角単独より情報量が多い
  ② 血統（種牡馬・母父）のコース別成績
  ③ 隊列シミュレーション（下部に入力フォーマットの定義あり）
  ④ 馬単位PCI
  ⑤ 統合サマリー
"""
import sys, re, json, warnings, urllib.request
import pandas as pd, numpy as np

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
pd.set_option('display.width', 400); pd.set_option('display.max_rows', 700); pd.set_option('display.max_colwidth', 30)

DATA = r"C:\Users\asari\Downloads\Claude\keiba-analyzer\data\processed"
MODEL = r"C:\Users\asari\Downloads\Claude\keiba-analyzer\analysis\score_model.json"
CROSS = r"C:\Users\asari\Downloads\Claude\keiba-analyzer\analysis\cross_table.json"
UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', 'Referer': 'https://race.netkeiba.com/'}
tag = re.compile(r'<[^>]+>')
clean = lambda x: re.sub(r'\s+', ' ', tag.sub(' ', x).replace('&nbsp;', ' ')).strip()
get = lambda u: urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=30).read().decode('utf-8', 'ignore')
n_ = lambda s: pd.to_numeric(s, errors='coerce')
BAND = lambda x: None if pd.isna(x) else ('A' if x <= .2 else ('B' if x <= .45 else ('C' if x <= .7 else 'D')))
BANDL = {'A': 'A 先頭〜2割', 'B': 'B 2-4.5割', 'C': 'C 4.5-7割', 'D': 'D 7割以降'}
WB = lambda x: None if pd.isna(x) else ('内1/3' if x <= .34 else ('中1/3' if x <= .67 else '外1/3'))

# ---- ⑥ 上がり順予測の係数 ----
# 2005-2025の20.6万走で学習し、2025-06-14以降の4.97万走で検証（r=+0.382, MAE=0.227）
# relag = 上り3F順位 ÷ 頭数（0に近いほど上がり上位）
AG_COEF = dict(const=0.4003, p_relag5=0.50028, p_rel4_3=-0.03872,
               p_pci5=-0.00234, cls差=0.01712, 距差100=0.01276)
AG_POS_SLOPE = -0.060   # 普段より位置を前に取るとrelagは悪化（効果は小さい）
CLASS_MAP = [('新馬', 0), ('未勝利', 0), ('１勝', 1), ('1勝', 1), ('２勝', 2), ('2勝', 2),
             ('３勝', 3), ('3勝', 3), ('オープン', 4), ('Ｇ３', 5), ('G3', 5),
             ('Ｇ２', 6), ('G2', 6), ('Ｇ１', 8), ('G1', 8)]

# ---- 隊列シミュレーションのパラメータ（ここを触れば挙動が変わる） ----
ADJ_SHORTEN = -0.04   # 前走から200m以上の短縮 → 前に行きやすい
ADJ_EXTEND = +0.04    # 前走から200m以上の延長 → 位置を落としやすい
ADJ_WAKU = 0.02       # 枠補正の大きさ（コースの内外バイアスから符号を自動決定）
ADJ_RIVAL = 0.04      # 同型が3頭目以降にいるとき、1頭ごとに後ろへ押し出される量
NIGE_TH = 0.15        # 「逃げ候補」とみなす想定位置のしきい値
FRONT_TH = 0.35       # ここ以下の馬だけが「前を狙う集団」＝競合補正の対象


def agg(df, key, mn=1):
    g = df.groupby(key, observed=True).agg(n=('着', 'size'), 勝=('着', lambda s: (s == 1).sum()),
                                           複=('着', lambda s: (s <= 3).sum()),
                                           単R=('tan', 'mean'), 複R=('fuk', 'mean'))
    g['勝率%'] = (g.勝 / g.n * 100).round(1); g['複勝%'] = (g.複 / g.n * 100).round(1)
    g['単ROI'] = g.単R.round(0); g['複ROI'] = g.複R.round(0)
    return g[g.n >= mn][['n', '勝', '複', '勝率%', '複勝%', '単ROI', '複ROI']]


def fetch_entries(rid):
    h = get(f"https://race.netkeiba.com/race/shutuba.html?race_id={rid}")
    i = h.find('RaceData01')
    cond = clean(h[i:i + 420])[:240] if i > 0 else ''
    nm = re.search(r'RaceName[^>]*>\s*([^<]+)', h)
    rows, seen = [], set()
    seg = h[h.find('Shutuba_Table'):]
    for tr in re.findall(r'<tr[^>]*class="[^"]*HorseList[^"]*"[^>]*>(.*?)</tr>', seg, re.S):
        c = [clean(x) for x in re.findall(r'<td[^>]*>(.*?)</td>', tr, re.S | re.I)]
        if len(c) < 10:
            continue
        try:
            waku, uma = int(c[0]), int(c[1])
        except Exception:
            continue
        if uma in seen:
            continue
        seen.add(uma)
        sa = re.match(r'([牡牝セ])(\d+)', c[4])
        jk = re.findall(r'/jockey/[^"]*"[^>]*>([^<]+)</a>', tr)
        rows.append(dict(枠=waku, 馬番=uma, 馬名=c[3], 性=sa.group(1) if sa else '',
                         齢=int(sa.group(2)) if sa else None, 斤=c[5],
                         騎手=jk[0].strip() if jk else ''))
    E = pd.DataFrame(rows).sort_values('馬番').reset_index(drop=True)
    try:
        j = json.loads(get(f"https://race.netkeiba.com/api/api_get_jra_odds.html?type=1&race_id={rid}&action=init"))
        od = j.get('data', {}).get('odds', {}).get('1', {})
        E['オッズ'] = E['馬番'].map({int(k): float(v[0]) for k, v in od.items() if v[0] not in ('', '---.-')})
        E['人気'] = E['オッズ'].rank(method='min').astype('Int64')
    except Exception:
        E['オッズ'] = np.nan; E['人気'] = pd.NA
    return E, cond, (nm.group(1).strip() if nm else '')


VEN_FULL = {'中': '中山', '阪': '阪神', '東': '東京', '京': '京都', '名': '中京', '新': '新潟',
            '札': '札幌', '函': '函館', '福': '福島', '小': '小倉'}


def fetch_entries_csv(path, ven, rid):
    """TARGETの枠順CSVから出走表を作る。オッズだけは netkeiba から取りにいく（無ければ空）。"""
    for enc in ('cp932', 'utf-8-sig', 'utf-8'):
        try:
            d = pd.read_csv(path, encoding=enc, dtype=str)
            break
        except UnicodeDecodeError:
            continue
    R = int(str(rid)[-2:])
    d = d[(d['場所'] == VEN_FULL.get(ven, ven)) & (n_(d['Ｒ']) == R)].copy()
    if d.empty:
        raise SystemExit(f"CSVに {VEN_FULL.get(ven, ven)} {R}R が見つかりません: {path}")
    E = pd.DataFrame(dict(枠=n_(d['枠番']).astype(int), 馬番=n_(d['馬番']).astype(int),
                          馬名=d['馬名'].str.strip(), 性=d['性別'].str.strip(),
                          齢=n_(d['年齢']).astype('Int64'), 斤=d['斤量'].str.strip(),
                          騎手=d['騎手'].str.strip())).sort_values('馬番').reset_index(drop=True)
    cond = f"{d['レース名'].iloc[0]}  {d['芝ダ'].iloc[0]}{d['距離'].iloc[0]}m  {len(E)}頭  （枠順CSV）"
    try:
        j = json.loads(get(f"https://race.netkeiba.com/api/api_get_jra_odds.html?type=1&race_id={rid}&action=init"))
        od = j.get('data', {}).get('odds', {}).get('1', {})
        E['オッズ'] = E['馬番'].map({int(k): float(v[0]) for k, v in od.items() if v[0] not in ('', '---.-')})
        E['人気'] = E['オッズ'].rank(method='min').astype('Int64')
    except Exception:
        E['オッズ'] = np.nan; E['人気'] = pd.NA
    return E, cond, str(d['レース名'].iloc[0])


def load_master():
    M = pd.read_parquet(DATA + r"\master.parquet")
    M['着'] = n_(M['着順_num']); M = M.dropna(subset=['着'])
    M['距'] = n_(M['距離'].astype(str).str.extract(r'(\d+)')[0])
    for c in ['人気', '馬番', '頭数', '2角', '3角', '4角', '年齢', '前距離', '前走頭数', 'キャリア']:
        M[c] = n_(M[c])
    M['ag'] = n_(M['上り3F'])
    M['rel4'] = (M['4角'] / M['頭数']).round(3)
    M['rel3'] = (M['3角'] / M['頭数']).round(3)
    M['押上'] = (M['rel3'] - M['rel4']).round(3)
    M['relw'] = (M['馬番'] / M['頭数']).round(3)
    M['ba'] = M['馬場状態'].astype(str).str[0]
    M['上順'] = M.groupby(['日付', '開催', 'Ｒ'])['ag'].rank(method='min')
    M['ven'] = M['開催'].astype(str).str.extract(r'([中名新札函小京阪東福])')[0]
    M['dt'] = pd.to_datetime('20' + M['日付'].astype(str), format='%Y%m%d', errors='coerce')
    M['tan'] = n_(M['単勝配当'].astype(str).str.replace(r'[()（）,]', '', regex=True)).fillna(0)
    M['fuk'] = n_(M['複勝配当'].astype(str).str.replace(r'[()（）,]', '', regex=True)).fillna(0)
    M['pci'] = n_(M['PCI'])
    M['体重'] = n_(M['馬体重'])
    M['cls'] = n_(M['クラス_num'])
    M['relag'] = (M['上順'] / M['頭数']).round(3)   # 上り3F順位÷頭数。0に近いほど上がり上位
    return M


def main():
    av = sys.argv[1:]
    csvsrc = None
    if '--csv' in av:
        i = av.index('--csv'); csvsrc = av[i + 1]; av = av[:i] + av[i + 2:]
    rid, ven, surf, dist = av[0], av[1], av[2], int(av[3])
    cls = av[4] if len(av) > 4 else None

    E, cond, rname = fetch_entries_csv(csvsrc, ven, rid) if csvsrc else fetch_entries(rid)
    M = load_master()
    C = M[(M['ven'] == ven) & (M['芝・ダ'] == surf) & (M['距'] == dist)]
    CC = C[C['クラス名'].astype(str).str.contains(cls, na=False)] if cls else C
    NF = len(E)

    print("=" * 110)
    print(f"{rname}   {ven}{surf}{dist}   {NF}頭")
    print(cond)
    print(f"コース母数: {len(C):,}頭 / {C.groupby(['日付','開催','Ｒ']).ngroups}R" +
          (f"   （{cls}限定 {len(CC):,}頭 / {CC.groupby(['日付','開催','Ｒ']).ngroups}R）" if cls else ""))
    print("=" * 110)

    # ---------------- ① 3角→4角の位置変化 ----------------
    print("\n" + "#" * 96)
    print("#  ① 3角 → 4角 の位置変化（押し上げ／下げ）")
    print("#" * 96)
    C2 = C.dropna(subset=['rel3', 'rel4']).copy()
    cut = pd.cut(C2['押上'], [-9, -0.15, -0.05, 0.05, 0.15, 9],
                 labels=['下げた(-0.15↓)', 'やや下げ', '維持', 'やや押上', '押し上げた(+0.15↑)'])
    print(agg(C2.assign(k=cut).dropna(subset=['k']), 'k', 10).to_string())
    print("\n[4角帯 × 押し上げ] 同じ4角位置でも、押し上げてきたのか下げたのかで別物")
    C2['4帯'] = C2['rel4'].apply(BAND); C2['押帯'] = np.where(C2['押上'] >= 0.05, '押上', np.where(C2['押上'] <= -0.05, '下げ', '維持'))
    pv = C2.dropna(subset=['4帯']).pivot_table(index='4帯', columns='押帯', values='着',
                                               aggfunc=lambda s: round((s == 1).mean() * 100, 1), observed=True)
    cnt = C2.dropna(subset=['4帯']).pivot_table(index='4帯', columns='押帯', values='着', aggfunc='size', observed=True)
    print("勝率%:"); print(pv.reindex(['A', 'B', 'C', 'D']).to_string())
    print("頭数:"); print(cnt.reindex(['A', 'B', 'C', 'D']).to_string())

    # ---------------- ② 血統 ----------------
    print("\n" + "#" * 96)
    print("#  ② 血統（このコースでの種牡馬・母父）")
    print("#" * 96)
    sw = agg(C.assign(k=C['種牡馬']), 'k', 25).sort_values('勝率%', ascending=False)
    print("[種牡馬 n>=25 上位10]"); print(sw.head(10).to_string())
    print("[種牡馬 n>=25 下位6]"); print(sw.tail(6).to_string())
    bw = agg(C.assign(k=C['母父馬']), 'k', 20).sort_values('勝率%', ascending=False)
    print("[母父 n>=20 上位8]"); print(bw.head(8).to_string())

    # ---------------- ④ PCI コース基準 ----------------
    win = C[C['着'] == 1]
    pci_win, pci_all = win['pci'].mean(), C['pci'].mean()
    print("\n" + "#" * 96)
    print("#  ④ 馬単位PCI のコース基準")
    print("#" * 96)
    print(f"全体平均 {pci_all:.1f} / 勝ち馬平均 {pci_win:.1f}")
    print(agg(C.assign(k=pd.qcut(C['pci'], 5, labels=['最低20%', '20-40', '40-60', '60-80', '最高20%'])).dropna(subset=['k']), 'k', 10).to_string())

    # ---------------- 各馬プロファイル ----------------
    prof = []
    for _, e in E.iterrows():
        d = M[M['馬名'] == e['馬名']].sort_values('dt')
        r = dict(枠=e['枠'], 馬番=e['馬番'], 馬名=e['馬名'], 性齢=f"{e['性']}{e['齢']}", 斤=e['斤'],
                 騎手=e['騎手'], オッズ=e['オッズ'])
        if len(d):
            sf = d[d['芝・ダ'] == surf]
            r3 = d.tail(3)
            r.update(種牡馬=d['種牡馬'].iloc[-1], 母父=d['母父馬'].iloc[-1],
                     近3走4角=round(r3['rel4'].median(), 2) if r3['rel4'].notna().any() else np.nan,
                     近3走押上=round(r3['押上'].mean(), 2) if r3['押上'].notna().any() else np.nan,
                     押上回数=int((d['押上'] >= 0.05).sum()), 下げ回数=int((d['押上'] <= -0.05).sum()),
                     PCI5走=round(sf['pci'].tail(5).mean(), 1) if len(sf) else np.nan,
                     PCI最高=round(sf['pci'].max(), 1) if len(sf) else np.nan,
                     逃率=round((d['rel4'] <= .15).mean() * 100) if d['rel4'].notna().any() else np.nan,
                     先行率=round((d['rel4'] <= .45).mean() * 100) if d['rel4'].notna().any() else np.nan,
                     p_relag5=round(d['relag'].tail(5).mean(), 3) if d['relag'].tail(5).notna().any() else np.nan,
                     p_push5=round(d['押上'].tail(5).mean(), 3) if d['押上'].tail(5).notna().any() else np.nan,
                     p_rel4_3=round(d['rel4'].tail(3).median(), 3) if d['rel4'].tail(3).notna().any() else np.nan,
                     p_pci5=round(sf['pci'].tail(5).mean(), 1) if len(sf) and sf['pci'].tail(5).notna().any() else np.nan,
                     前走cls=d['cls'].iloc[-1],
                     位置ブレ=round(d['rel4'].tail(5).std(), 2) if d['rel4'].tail(5).notna().sum() >= 2 else np.nan,
                     出走数=int(d['rel4'].notna().sum()),
                     休養週=round((d['dt'].max() - d['dt'].iloc[-2]).days / 7, 1) if len(d) >= 2 else np.nan,
                     前走距離=int(d['距'].iloc[-1]) if pd.notna(d['距'].iloc[-1]) else None,
                     体重=int(d['体重'].iloc[-1]) if pd.notna(d['体重'].iloc[-1]) else None,
                     キャリア=int(d['キャリア'].iloc[-1]) if pd.notna(d['キャリア'].iloc[-1]) else len(d),
                     上順3走=round(r3['上順'].mean(), 1) if r3['上順'].notna().any() else np.nan)
        prof.append(r)
    P = pd.DataFrame(prof)

    # 血統の該当成績を貼る
    swd = sw['勝率%'].to_dict(); swn = sw['n'].to_dict()
    bwd = bw['勝率%'].to_dict(); bwn = bw['n'].to_dict()
    P['父勝率'] = P['種牡馬'].map(swd); P['父n'] = P['種牡馬'].map(swn)
    P['母父勝率'] = P['母父'].map(bwd); P['母父n'] = P['母父'].map(bwn)
    P['PCI差'] = (P['PCI5走'] - pci_win).round(1)

    # ---------------- ⑥ 上がり順の予測 ----------------
    today_cls = next((v for k, v in CLASS_MAP if k in (rname + ' ' + cond)), np.nan)
    P['cls差'] = today_cls - P['前走cls'] if pd.notna(today_cls) else 0.0
    P['距差100'] = (dist - P['前走距離']) / 100.0
    a = AG_COEF
    P['予測relag'] = (a['const'] + a['p_relag5'] * P['p_relag5'].fillna(0.5)
                    + a['p_rel4_3'] * P['p_rel4_3'].fillna(0.5)
                    + a['p_pci5'] * P['p_pci5'].fillna(pci_all)
                    + a['cls差'] * P['cls差'].fillna(0)
                    + a['距差100'] * P['距差100'].fillna(0)).clip(0.02, 1.0).round(3)
    P['予測上り順'] = (P['予測relag'] * NF).round(1)
    P['上り評価'] = pd.cut(P['予測relag'], [0, .40, .48, .56, .64, 1.01],
                        labels=['①最上位', '②上位', '③中位', '④下位', '⑤最下位'])

    # ---------------- ⑦ 統合スコア（5ファクターを束ねた勝率） ----------------
    P['統合スコア'] = np.nan
    try:
        with open(MODEL, encoding='utf-8') as f:
            SM = json.load(f)
        db = int(dist // 400 * 400)
        P['sire_r'] = [SM['sire'].get(f"{surf}|{db}|{s}", SM['base']) for s in P['種牡馬'].fillna('')]
        P['bms_r'] = [SM['bms'].get(f"{surf}|{db}|{s}", SM['base']) for s in P['母父'].fillna('')]
        P['pred_relag'] = P['予測relag']; P['career'] = P['キャリア']
        P['weight'] = P['体重']; P['n_field'] = NF; P['rest_w'] = P['休養週']
        X = np.column_stack([((P[c].astype(float).fillna(SM['mu'][c]) - SM['mu'][c]) / SM['sd'][c]).values
                             for c in SM['feats']])
        z = X @ np.array(SM['coef']) + SM['intercept']
        p = 1 / (1 + np.exp(-z))
        P['統合スコア'] = (p / p.sum() * 100).round(1)          # レース内で合計100%に正規化
        P['模型順'] = P['統合スコア'].rank(ascending=False, method='min').astype(int)
        P['市場順'] = P['オッズ'].rank(method='min')
        # 模型スコアをそのまま信じない。市場順位×模型順位の「実績勝率」に変換する。
        # 検証52,572走の結果、模型が情報を足せるのは1番人気だけ（36.1%〜22.8%）で、
        # 2番人気以下は模型順位で勝率がほとんど動かない。
        with open(CROSS, encoding='utf-8') as f:
            CT = json.load(f)
        pb = lambda r: '1人気' if r <= 1 else ('2人気' if r <= 2 else ('3人気' if r <= 3 else
             ('4-5' if r <= 5 else ('6-8' if r <= 8 else '9↓'))))
        mb = lambda r: '模型1位' if r <= 1 else ('模型2-3' if r <= 3 else ('模型4-6' if r <= 6 else '模型7↓'))
        P['人気帯'] = P['市場順'].apply(lambda r: pb(r) if pd.notna(r) else None)
        P['模型帯'] = P['模型順'].apply(mb)
        P['期待勝率'] = [CT.get(f"{a}|{b}", {}).get('win', np.nan) for a, b in zip(P['人気帯'], P['模型帯'])]
        P['期待複勝'] = [CT.get(f"{a}|{b}", {}).get('fuku', np.nan) for a, b in zip(P['人気帯'], P['模型帯'])]
        P['最終順'] = P['期待勝率'].rank(ascending=False, method='min')
        ok = True
    except Exception as e:
        print(f"（統合スコア: モデル未読込 {e}）"); ok = False

    # ---------------- ③ 隊列シミュレーション ----------------
    inner = agg(C.assign(k=C['relw'].apply(WB)).dropna(subset=['k']), 'k')
    in_w = inner.loc['内1/3', '勝率%'] if '内1/3' in inner.index else np.nan
    out_w = inner.loc['外1/3', '勝率%'] if '外1/3' in inner.index else np.nan
    waku_sign = -1 if (pd.notna(in_w) and pd.notna(out_w) and in_w > out_w) else 1  # 内有利なら内を前に寄せる

    P['relw'] = (P['馬番'] / NF).round(2)
    P['基準'] = P['近3走4角'].fillna(0.55)
    P['距離補正'] = np.where(P['前走距離'].notna() & (P['前走距離'] - dist >= 200), ADJ_SHORTEN,
                          np.where(P['前走距離'].notna() & (dist - P['前走距離'] >= 200), ADJ_EXTEND, 0.0))
    P['枠補正'] = np.where(P['relw'] <= 0.34, waku_sign * ADJ_WAKU,
                        np.where(P['relw'] >= 0.67, -waku_sign * ADJ_WAKU, 0.0))
    P['仮'] = P['基準'] + P['距離補正'] + P['枠補正']
    # 競合補正は「前を狙う馬どうし」だけで起きる。中団以降の馬は押し出されない。
    front = P['仮'] <= FRONT_TH
    P['同型順'] = np.nan
    P.loc[front, '同型順'] = P.loc[front, '仮'].rank(method='first')
    P['競合補正'] = np.where(front & (P['同型順'] > 2), (P['同型順'] - 2) * ADJ_RIVAL, 0.0)
    P['想定rel4'] = (P['仮'] + P['競合補正']).clip(0.02, 1.0).round(2)
    P = P.sort_values('想定rel4').reset_index(drop=True)
    P['想定帯'] = P['想定rel4'].apply(BAND)
    # 予測の信頼度：位置が安定していて出走数が多い馬ほど当たる（検証で確認済み）
    P['信頼度'] = np.where(P['出走数'].fillna(0) < 4, '低(キャリア浅)',
                        np.where(P['位置ブレ'].fillna(1) >= 0.25, '低(位置不安定)',
                                 np.where(P['位置ブレ'].fillna(1) >= 0.15, '中', '高')))

    band_win = agg(C.assign(k=C['rel4'].apply(BAND)).dropna(subset=['k']), 'k')['勝率%'].to_dict()
    P['帯勝率%'] = P['想定帯'].map(band_win)

    n_nige = int((P['想定rel4'] <= NIGE_TH).sum())
    lap = pd.read_parquet(DATA + r"\lap_master_joined.parquet")
    lap['距'] = n_(lap['距離'].astype(str).str.extract(r'(\d+)')[0])
    lap['ven'] = lap['開催'].astype(str).str.extract(r'([中名新札函小京阪東福])')[0]
    LC = lap[(lap['ven'] == ven) & (lap['芝・ダ'] == surf) & (lap['距'] == dist)].drop_duplicates(['日付', '開催', 'Ｒ'])
    base_d = LC['lap_後半3F差'].mean()
    shift = {0: -0.5, 1: 0.0, 2: +0.3}.get(n_nige, +0.3 + 0.3 * (n_nige - 2))
    print("\n" + "#" * 96)
    print("#  ③ 隊列シミュレーション")
    print("#" * 96)
    print(f"枠バイアス判定: 内1/3 {in_w}% vs 外1/3 {out_w}%  → {'内枠を前に補正' if waku_sign<0 else '外枠を前に補正'}")
    print(f"逃げ候補（想定rel4<={NIGE_TH}）: {n_nige}頭")
    est_d = base_d + shift
    print(f"想定ペース: コース基準 後半3F差 {base_d:+.2f} → 補正後 {est_d:+.2f}")
    # 「緩い＝先行有利」と決めつけず、実データの条件付き勝率をそのまま出す
    key3 = ['日付', '開催', 'Ｒ']
    lapj = LC[key3 + ['lap_後半3F差']].drop_duplicates(key3)
    CP = C.merge(lapj, on=key3, how='inner')
    if len(CP) > 200:
        CP['ペース帯'] = pd.qcut(CP['lap_後半3F差'], 4, labels=['①緩い', '②', '③', '④速い'])
        edges = CP.groupby('ペース帯', observed=True)['lap_後半3F差'].mean().round(2)
        near = (edges - est_d).abs().idxmin()
        print(f"  → 想定 {est_d:+.2f} に最も近い帯は【{near}】(平均{edges[near]:+.2f})。その帯での実績:")
        sub = CP[CP['ペース帯'] == near]
        a = agg(sub.assign(k=sub['rel4'].apply(BAND)).dropna(subset=['k']), 'k')['勝率%']
        u = agg(sub.dropna(subset=['上順']).assign(k=lambda x: x['上順'].apply(
            lambda v: '1位' if v <= 1 else ('2-3位' if v <= 3 else ('4-6位' if v <= 6 else '7位↓')))), 'k')['勝率%']
        print("     4角帯別 勝率%: " + " / ".join(f"{k}:{v}" for k, v in a.items()))
        print("     上がり順別 勝率%: " + " / ".join(f"{k}:{v}" for k, v in u.reindex(['1位', '2-3位', '4-6位', '7位↓']).dropna().items()))
        wsub = sub[sub['着'] == 1]
        print(f"     勝ち馬の平均: 4角{wsub['rel4'].mean():.2f} / 上がり順{wsub['上順'].mean():.2f}")
    # レース単位の信頼度：位置が読める馬が少なければシミュ全体を割り引く
    hi = int((P['信頼度'] == '高').sum())
    print(f"\n  ※シミュ全体の信頼度: 「高」{hi}/{len(P)}頭 → " +
          ('位置予測はある程度信用してよい' if hi >= len(P) * 0.35 else
           '**位置が読める馬が少なく、隊列予測は参考程度に留めること**'))
    cols = ['馬番', '馬名', '騎手', 'オッズ', '基準', '距離補正', '枠補正', '競合補正', '想定rel4', '想定帯', '帯勝率%', '信頼度', '位置ブレ', '出走数', '逃率', '先行率']
    print(P[cols].to_string(index=False))
    order = " - ".join("".join(f"({int(x)})" for x in g['馬番']) for _, g in P.groupby((P['想定rel4'] * 5).round()))
    print(f"\n想定隊列: {order}")

    # ---------------- ⑤ 統合サマリー ----------------
    print("\n" + "#" * 96)
    print("#  ⑤ 統合サマリー")
    print("#" * 96)
    print("[⑥ 上がり順の予測]  予測relag = 上り3F順位÷頭数 の予測値。小さいほど上位")
    print("   検証実績(4.97万走): ①最上位 上り1位率18.4%/勝率13.1%  →  ⑤最下位 上り1位率2.7%/勝率3.0%")
    print(P.sort_values('予測relag')[['馬番', '馬名', 'オッズ', '予測relag', '予測上り順', '上り評価',
                                     'p_relag5', 'p_pci5', 'cls差', '距差100', '上順3走']].to_string(index=False))
    if ok:
        print("\n" + "#" * 96)
        print("#  ⑦ 統合スコア（5ファクターを束ねた勝率／レース内合計100%）")
        print("#" * 96)
        print("期待勝率 = 「市場の人気帯 × 模型の順位帯」の実績勝率（検証52,572走）。模型スコア単体では順位付けしない。")
        Q = P.sort_values('期待勝率', ascending=False)
        print(Q[['最終順', '馬番', '馬名', '騎手', 'オッズ', '市場順', '模型順', '統合スコア',
                 '期待勝率', '期待複勝', '予測relag', '上り評価', '想定rel4', '想定帯']].to_string(index=False))
        fav = P.loc[P['オッズ'].idxmin()] if P['オッズ'].notna().any() else None
        if fav is not None:
            fr = int(fav['模型順'])
            v = ('模型も1位 → 実績36.1%。**単騎にしてよい**' if fr == 1 else
                 ('模型2-3位 → 実績32.5%。標準的' if fr <= 3 else
                  ('模型4-6位 → 実績31.5%。やや割引だが依然として最有力' if fr <= 6 else
                   '模型7位以下 → 実績22.8%まで落ちる。**1番人気の単騎は避ける**')))
            print(f"\n  1番人気 {int(fav['馬番'])}{fav['馬名']}（模型{fr}位）→ {v}")
        print("  ※模型が勝率を動かせるのは1番人気のみ（36.1%〜22.8%）。2番人気以下は模型順位で" +
              "勝率がほとんど変わらないため、人気順の評価を優先すること。")
        # 人気を裏切りそうな馬（人気5番以内 × 模型下位）
        low = P[(P['市場順'] <= 5) & (P['統合スコア'] <= P['統合スコア'].quantile(0.4))]
        if len(low):
            print("  人気上位だが模型が低評価（検証: 勝率8.7%/複勝28.0%。全体16.5%/44.5%）: " +
                  " ".join(f"{int(r['馬番'])}{r['馬名']}({r['統合スコア']}%)" for _, r in low.iterrows()))
    print("\n[各ファクターの内訳]")
    s = ['馬番', '馬名', 'オッズ', '統合スコア', '想定rel4', '想定帯', '信頼度', '予測relag', '上り評価',
         '近3走押上', '押上回数', 'PCI5走', 'PCI差', '種牡馬', '父勝率', '母父', '母父勝率', 'キャリア', '体重']
    print(P.sort_values('馬番')[[c for c in s if c in P.columns]].to_string(index=False))

    print("""
--------------------------------------------------------------------------------------------
③ 隊列シミュレーションの入力フォーマット（手で上書きしたいときはここを埋める）
--------------------------------------------------------------------------------------------
 必須（自動取得できる）           : 馬番 / 近3走の4角相対位置 / 前走距離 / 枠
 任意（人が足すと精度が上がる）   : 下記4つ
   1. 想定脚質メモ  例) 11:逃げ宣言, 5:テン速い, 8:出遅れ癖
   2. 枠順以外のペース情報 例) 前走同型不在で楽逃げ→今回は競られる
   3. 乗り替わり後の指示変化 例) 新騎手が先行策を明言
   4. 当日パドック/返し馬  例) 気配◎, 入れ込み
 パラメータ（スクリプト冒頭で変更可）
   ADJ_SHORTEN=-0.04  200m以上の短縮 → 前に行きやすい
   ADJ_EXTEND =+0.04  200m以上の延長 → 位置を落とす
   ADJ_WAKU   = 0.02  枠補正（内外どちらを前に寄せるかはコース実績から自動判定）
   ADJ_RIVAL  = 0.04  同型3頭目以降が1頭ごとに後ろへ押し出される量
   NIGE_TH    = 0.15  「逃げ候補」とみなす想定rel4のしきい値
--------------------------------------------------------------------------------------------""")


if __name__ == '__main__':
    main()
