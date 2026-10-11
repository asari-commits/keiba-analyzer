# -*- coding: utf-8 -*-
"""1日分の予想（前日）と、当日のオッズ取り直し。

  前日の夜:  python analysis/pred_day.py prepare 20261018 05:04:05
  当日の朝:  python analysis/pred_day.py watch   20261018 05:04:05 [--interval 10]
  1回だけ:   python analysis/pred_day.py once    20261018 05:04:05

  場コード:回:日次  例) 05:04:05 = 東京 4回 5日目（場コード 05東京 06中山 08京都 09阪神 ...）

prepare
  ・新馬戦と障害を除く全レースで race_factors.py を並列実行（data/raw/pred_<日付>_<場>/R??.txt）
  ・最新方針で機械的に印を付ける（下記）→ スライドPDFを OneDrive に保存
watch / once
  ・最新オッズを取り直し、⑦（人気帯×模型帯の実績勝率）と印を付け直す。模型の評価（統合スコア）は
    オッズと無関係なので前日の値をそのまま使う
  ・先頭に「変更点」のページを付けたPDFを OneDrive に上書き保存（スマホから最新版が見られる）
  ・watch は --interval 分ごとに繰り返し、最終レースの発走時刻を過ぎたら終了する

印のルール（2026/10/11 時点の方針）
  1. ⑦の期待勝率の順
  2. 「人気上位×模型低評価」（5番人気以内かつ統合スコアがレース内の下位40%）の馬は勝率8.7%に割り引き、◎○には置かない
  3. 消しは「想定D帯 × 位置の信頼度が高 × そのコースのD帯の例年複勝率10%以下」だけ
  4. 1番人気が模型7位以下で、2番手との差が5pt以内なら◎○を入れ替える（入れ替え先が上の3に当たる想定D帯なら入れ替えず2頭軸）
  5. 当日の傾向・注目馬・クッション値は使わない
  穴候補（実験枠）: 4〜6番人気で、同じ路面の直近5走relag平均が全路面の直近5走平均より0.05以上良い馬
"""
import argparse, glob, json, os, re, shutil, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np, pandas as pd

sys.stdout.reconfigure(encoding='utf-8'); pd.set_option('display.width', 250)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'analysis'))
DATA = os.path.join(ROOT, 'data', 'processed')
ONEDRIVE = os.path.join(os.path.expanduser('~'), 'OneDrive', '競馬ツール用データ')
CROSS = os.path.join(ROOT, 'analysis', 'cross_table.json')
VEN = {'01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京',
       '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉'}
V1 = {'札幌': '札', '函館': '函', '福島': '福', '新潟': '新', '東京': '東',
      '中山': '中', '中京': '名', '京都': '京', '阪神': '阪', '小倉': '小'}
WARN_RATE = 8.7
FAV_NOTE = {'模型1位': '単騎にしてよい', '模型2-3': '標準的', '模型4-6': 'やや割引だが依然として最有力',
            '模型7↓': '1番人気の単騎は避ける'}
n_ = lambda s: pd.to_numeric(s, errors='coerce')
KEY = ['日付', '開催', 'Ｒ']


def pb(r):
    return '1人気' if r <= 1 else ('2人気' if r <= 2 else ('3人気' if r <= 3 else
           ('4-5' if r <= 5 else ('6-8' if r <= 8 else '9↓'))))


def mb(r):
    return '模型1位' if r <= 1 else ('模型2-3' if r <= 3 else ('模型4-6' if r <= 6 else '模型7↓'))


def outdir(date, ven):
    d = os.path.join(ROOT, 'data', 'raw', f'pred_{date}_{ven}')
    os.makedirs(d, exist_ok=True)
    return d


def get_odds(rid):
    j = json.loads(urllib.request.urlopen(urllib.request.Request(
        f"https://race.netkeiba.com/api/api_get_jra_odds.html?type=1&race_id={rid}&action=init",
        headers={'User-Agent': 'Mozilla/5.0'}), timeout=20).read().decode('euc-jp', 'ignore'))
    od = j.get('data', {}).get('odds', {}).get('1', {})
    return {int(k): float(v[0]) for k, v in od.items() if v[0] not in ('', '---.-')}


# ---------------------------------------------------------------- prepare
def race_list(date, spec):
    import race_factors as rf
    vc, kai, nichi = spec.split(':')
    ven = VEN[vc]
    races = []
    for R in range(1, 13):
        rid = f"{date[:4]}{vc}{int(kai):02d}{int(nichi):02d}{R:02d}"
        try:
            E, cond, nm = rf.fetch_entries(rid)
        except Exception as e:
            print('出馬表NG', rid, e); continue
        c = cond.replace(' ', '')
        m = re.search(r'(芝|ダ|障)(\d{3,4})m', c)
        t = re.search(r'(\d+:\d+)発走', c)
        if not m or m.group(1) == '障' or '新馬' in c:
            continue
        cls = ('未勝利' if '未勝利' in c else '1勝' if '１勝' in c or '1勝' in c else
               '2勝' if '２勝' in c or '2勝' in c else '3勝' if '３勝' in c or '3勝' in c else None)
        key = '2歳' if '２歳' in c else (cls or 'OP')
        races.append(dict(R=R, rid=rid, レース名=nm, 芝ダ=m.group(1), 距離=int(m.group(2)), cls=cls, 区分=key,
                          発走=t.group(1) if t else '', 頭数=len(E)))
        time.sleep(0.3)
    return ven, races


def run_factors(d, ven, races):
    def one(r):
        args = [sys.executable, os.path.join(ROOT, 'analysis', 'race_factors.py'), r['rid'], V1[ven],
                r['芝ダ'], str(r['距離'])] + ([r['cls']] if r['cls'] else [])
        with open(os.path.join(d, f"R{r['R']:02d}.txt"), 'w', encoding='utf-8') as f:
            subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, timeout=1200)
        print(f"  分析完了 {r['R']}R")
    with ThreadPoolExecutor(max_workers=5) as ex:
        list(ex.map(one, races))


def load_master():
    M = pd.read_parquet(os.path.join(DATA, 'master.parquet'))
    M['着'] = n_(M['着順_num']); M = M.dropna(subset=['着'])
    for c in ['頭数', '4角', '人気', '年齢']:
        M[c] = n_(M[c])
    M['ven'] = M['開催'].astype(str).str.extract(r'([中名新札函小京阪東福])')[0]
    M['距'] = n_(M['距離'].astype(str).str.extract(r'(\d+)')[0])
    M['ag'] = n_(M['上り3F']); M['rel4'] = M['4角'] / M['頭数']
    M['上順'] = M.groupby(KEY)['ag'].rank(method='min')
    M['relag'] = M['上順'] / M['頭数']
    M['帯'] = pd.cut(M['rel4'], [0, .2, .45, .7, 9], labels=['A', 'B', 'C', 'D'])
    M['w'] = (M['着'] == 1).astype(int); M['f'] = (M['着'] <= 3).astype(int)
    M = M.merge(M.groupby(KEY)['年齢'].max().rename('mx').reset_index(), on=KEY)
    M['dt'] = pd.to_datetime('20' + M['日付'].astype(str), format='%Y%m%d', errors='coerce')
    return M


def course_row(M, ven, r):
    X = M[(M['ven'] == V1[ven]) & (M['芝・ダ'] == r['芝ダ']) & (M['距'] == r['距離'])]
    cl = X['クラス名'].astype(str)
    k = r['区分']
    if k == '2歳':
        Y = X[X['mx'] == 2]
    elif k == 'OP':
        Y = X[cl.str.contains('Ｇ|OP|オープン|ｵｰﾌﾟﾝ', na=False)]
    elif k == '未勝利':
        Y = X[cl.str.contains('未勝利', na=False)]
    else:
        Y = X[cl.str.contains(k[0] + '勝|' + '１２３'[int(k[0]) - 1] + '勝', na=False)]
    Z, lvl = (Y, k) if Y.groupby(KEY).ngroups >= 30 else (X, '全クラス')
    b = Z.groupby('帯', observed=True)['f'].mean() * 100
    W = Z[Z['w'] == 1]
    return dict(基準=lvl, レース数=Z.groupby(KEY).ngroups, A複=b.get('A', np.nan), D複=b.get('D', np.nan),
                勝馬上順=W['上順'].mean(), 人1勝=Z[Z['人気'] == 1]['w'].mean() * 100)


def section(txt, start, end):
    i = txt.find(start); j = txt.find(end, i + 1)
    return txt[i:j] if i >= 0 else ''


def parse_factors(path):
    txt = open(path, encoding='utf-8').read()
    sim = re.search(r'シミュ全体の信頼度: 「高」(\d+)/(\d+)頭', txt)
    s7 = section(txt, '⑦ 統合スコア', '[各ファクターの内訳]')
    rows = []
    for ln in s7.split('\n'):
        t = ln.split()
        if len(t) == 14 and re.match(r'^\d+\.\d$', t[0]):
            rows.append(dict(馬番=int(t[1]), 馬名=t[2], 騎手=t[3], 前日オッズ=float(t[4]), 前日人気=int(float(t[5])),
                             模型順=int(t[6]), 統合スコア=float(t[7]), 上り評価=t[11],
                             想定rel4=float(t[12]), 想定帯=t[13]))
    P = pd.DataFrame(rows)
    rel = {}
    for ln in section(txt, '[各ファクターの内訳]', '----------').split('\n'):
        t = ln.split()
        if len(t) > 7 and t[0].isdigit():
            rel[int(t[0])] = t[6]
    P['信頼度'] = P['馬番'].map(rel).fillna('')
    return P, (int(sim.group(1)) if sim else None)


def surface_gap(M, names, surf, date):
    """穴候補用: 全路面の直近5走relag平均 − 同じ路面の直近5走relag平均（正＝この路面の方が上がり上位）。"""
    A = M[['馬名', 'dt', '芝・ダ', 'relag']].copy()
    last = A['dt'].max()
    for f in sorted(glob.glob(os.path.join(ROOT, 'data', 'raw', 'day_results', '2*.csv'))):
        if not re.fullmatch(r'\d{8}\.csv', os.path.basename(f)):
            continue                                    # _meta.csv（レース単位）は除く
        d = os.path.basename(f)[:8]
        if pd.Timestamp(d) <= last or d >= date:
            continue
        X = pd.read_csv(f, encoding='utf-8-sig')
        X = X[X['芝ダ'].isin(['芝', 'ダ']) & X['着'].notna()].copy()
        X['上順'] = X.groupby(['場', 'R'])['上3F'].rank(method='min')
        A = pd.concat([A, pd.DataFrame({'馬名': X['馬名'].str.strip(), 'dt': pd.Timestamp(d),
                                        '芝・ダ': X['芝ダ'], 'relag': X['上順'] / X['頭数']})])
    A = A.dropna(subset=['dt', 'relag']).sort_values(['馬名', 'dt'])
    out = {}
    for nm in names:
        H = A[A['馬名'] == nm]
        S = H[H['芝・ダ'] == surf]
        out[nm] = (H.tail(5)['relag'].mean() - S.tail(5)['relag'].mean()) if len(S) and len(H) else np.nan
    return out


def prepare(date, spec):
    ven, races = race_list(date, spec)
    d = outdir(date, ven)
    print(f"{ven} {date}: 対象 {len(races)}R（新馬・障害を除く） → 7ファクター分析を並列実行")
    run_factors(d, ven, races)
    M = load_master()
    for r in races:
        P, sim = parse_factors(os.path.join(d, f"R{r['R']:02d}.txt"))
        gap = surface_gap(M, P['馬名'].tolist(), r['芝ダ'], date)
        P['路面ズレ'] = P['馬名'].map(gap)
        r.update(P=P, シミュ高=sim, course=course_row(M, ven, r))
    meta = dict(date=date, ven=ven, spec=spec, races=races)
    pd.to_pickle(meta, os.path.join(d, 'pred.pkl'))
    res = evaluate(meta, odds_now=None)
    path = build_pdf(meta, res, latest=False)
    print('前日版PDF:', path)


# ---------------------------------------------------------------- 印付け
def evaluate(meta, odds_now=None):
    """odds_now=None なら前日オッズ、dict(R→{馬番:オッズ}) なら最新オッズで⑦と印を付ける。"""
    CT = json.load(open(CROSS, encoding='utf-8'))
    out = []
    for r in meta['races']:
        P = r['P'].copy()
        od = (odds_now or {}).get(r['R'])
        P['オッズ'] = P['馬番'].map(od) if od else P['前日オッズ']
        P['取消'] = P['オッズ'].isna()
        Q = P[~P['取消']].copy()
        Q['人気'] = Q['オッズ'].rank(method='min').astype(int)
        Q['期待勝率'] = [CT.get(f"{pb(a)}|{mb(b)}", {}).get('win', np.nan) for a, b in zip(Q['人気'], Q['模型順'])]
        Q['警告'] = (Q['人気'] <= 5) & (Q['統合スコア'] <= Q['統合スコア'].quantile(0.4))
        fav = Q.sort_values(['人気', '模型順']).iloc[0]
        fav_note = FAV_NOTE[mb(fav['模型順'])]
        Q['実効'] = np.where(Q['警告'], Q['期待勝率'].clip(upper=WARN_RATE), Q['期待勝率'])
        dcut = r['course']['D複'] <= 10
        Q['消し'] = (Q['想定帯'] == 'D') & (Q['信頼度'] == '高') & dcut
        cand = Q[~Q['消し']].sort_values(['実効', '期待勝率', '人気'], ascending=[False, False, True]).reset_index(drop=True)
        order = list(cand[~cand['警告']].head(2).index)
        sw = ''
        if '避ける' in fav_note and len(order) == 2 and cand.loc[order[0], '人気'] == 1 \
                and cand.loc[order[0], '実効'] - cand.loc[order[1], '実効'] <= 5:
            if dcut and cand.loc[order[1], '想定帯'] == 'D':
                sw = '2頭軸'
            else:
                order = [order[1], order[0]]; sw = '入替'
        rest = [i for i in cand.index if i not in order][:3]
        Q['印'] = ''
        for s, i in zip(['◎', '○', '▲', '△', '△'], order + rest):
            Q.loc[Q['馬番'] == cand.loc[i, '馬番'], '印'] = s
        Q.loc[Q['消し'], '印'] = '消'
        Q['穴'] = (Q['路面ズレ'] >= 0.05) & Q['人気'].between(4, 6)
        out.append(dict(R=r['R'], P=Q, fav_note=fav_note, swap=sw, 取消=P.loc[P['取消'], '馬番'].tolist()))
    return out


def diff(prev, now):
    """前日版と最新版の違いを文章にする。"""
    lines = []
    a = prev['P'].set_index('馬番'); b = now['P'].set_index('馬番')
    for s in ['◎', '○', '▲']:
        pa = a.index[a['印'] == s].tolist(); pn = b.index[b['印'] == s].tolist()
        if pa != pn:
            lines.append(f"{s}: {pa[0] if pa else '-'} {a.loc[pa[0], '馬名'] if pa else ''} → {pn[0] if pn else '-'} {b.loc[pn[0], '馬名'] if pn else ''}")
    for u in b.index:
        if u in a.index and abs(int(b.loc[u, '人気']) - int(a.loc[u, '人気'])) >= 3 and (a.loc[u, '印'] or b.loc[u, '印']):
            lines.append(f"{a.loc[u, '印'] or '無'}{u} {b.loc[u, '馬名']}: {int(a.loc[u, '人気'])}番人気 → {int(b.loc[u, '人気'])}番人気（{b.loc[u, 'オッズ']}倍）")
    for u in now['取消']:
        lines.append(f"取消/オッズなし: {u}")
    pa_ana = set(a.index[a['穴']]); pn_ana = set(b.index[b['穴']])
    if pa_ana != pn_ana:
        lines.append(f"穴候補: {sorted(pa_ana) or 'なし'} → {sorted(pn_ana) or 'なし'}")
    return lines


def check(date, spec, write=True):
    vc = spec.split(':')[0]
    d = outdir(date, VEN[vc])
    meta = pd.read_pickle(os.path.join(d, 'pred.pkl'))
    now = datetime.now()
    odds, started = {}, set()
    for r in meta['races']:
        hh, mm = map(int, r['発走'].split(':')) if r['発走'] else (23, 59)
        if now >= now.replace(hour=hh, minute=mm, second=0) and now.strftime('%Y%m%d') == date:
            started.add(r['R']); continue
        try:
            odds[r['R']] = get_odds(r['rid'])
        except Exception as e:
            print('オッズNG', r['R'], e)
        time.sleep(0.3)
    prev = {x['R']: x for x in evaluate(meta, None)}
    cur = {x['R']: x for x in evaluate(meta, odds)}
    changes = {R: ([] if R in started else diff(prev[R], cur[R])) for R in prev}
    stamp = now.strftime('%H:%M')
    print(f"\n=== {meta['ven']} {date} {stamp} 時点のオッズ")
    for R in sorted(changes):
        st = '発走済み' if R in started else ('変更あり' if changes[R] else '変更なし')
        print(f"  {R}R {st}" + ''.join('\n     ' + x for x in changes[R]))
    # 前日印と最新オッズでの印を両方記録しておく（どちらが良いかを後で比べるため）
    snap = []
    for R, x in cur.items():
        if R in started:
            continue
        for _, h in x['P'].iterrows():
            snap.append(dict(時刻=now.strftime('%Y-%m-%d %H:%M'), R=R, 馬番=int(h['馬番']), 馬名=h['馬名'],
                             オッズ=h['オッズ'], 人気=int(h['人気']), 印=h['印'],
                             前日印=prev[R]['P'].set_index('馬番')['印'].get(h['馬番'], ''), 穴=bool(h['穴'])))
    if snap:
        f = os.path.join(d, 'odds_snapshots.csv')
        pd.DataFrame(snap).to_csv(f, mode='a', header=not os.path.exists(f), index=False, encoding='utf-8-sig')
    if write:
        res = [cur[R] if R not in started else prev[R] for R in sorted(cur)]
        path = build_pdf(meta, res, latest=True, changes=changes, started=started, stamp=stamp, prev=prev)
        print('最新版PDF:', path)
    return max((r['発走'] for r in meta['races'] if r['発走']), default='23:59')


def watch(date, spec, interval):
    while True:
        last = check(date, spec)
        hh, mm = map(int, last.split(':'))
        now = datetime.now()
        if now >= now.replace(hour=hh, minute=mm, second=0):
            print('最終レースの発走時刻を過ぎたので終了します'); break
        time.sleep(interval * 60)


# ---------------------------------------------------------------- PDF
def build_pdf(meta, res, latest=False, changes=None, started=(), stamp='', prev=None):
    """latest=True のとき、印の列は前日の印（本線）。最新オッズで付け直した印は「今」の列に参考として出す。
    2026/10/11 の検証で、締切オッズで全部付け直すと◎の勝ち数は同じ・▲は悪化したため、本線は前日印のままにしている。"""
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.colors import HexColor
    try:
        pdfmetrics.registerFont(TTFont('G', r'C:\Windows\Fonts\BIZ-UDGothicR.ttc', subfontIndex=0))
        pdfmetrics.registerFont(TTFont('GB', r'C:\Windows\Fonts\BIZ-UDGothicB.ttc', subfontIndex=0))
    except Exception:
        pass
    W, H = 960, 540
    INK, SUB, LINE, ACC, WARN = (HexColor('#1f2328'), HexColor('#57606a'), HexColor('#d0d7de'),
                                 HexColor('#0b5cad'), HexColor('#b35900'))
    ORDER = {'◎': 0, '○': 1, '▲': 2, '△': 3, '消': 9}
    races = {r['R']: r for r in meta['races']}
    date, ven = meta['date'], meta['ven']
    md = f"{int(date[4:6])}/{int(date[6:])}"

    def wrap(c, text, x, y, width, font='G', size=16, lead=22, color=INK):
        c.setFont(font, size); c.setFillColor(color); line = ''
        for ch in text:
            if pdfmetrics.stringWidth(line + ch, font, size) > width:
                k = len(line)
                while k > 0 and line[k - 1].isascii() and line[k - 1].isalnum() and ch.isascii():
                    k -= 1
                if 0 < k < len(line):
                    c.drawString(x, y, line[:k]); y -= lead; line = line[k:] + ch
                else:
                    c.drawString(x, y, line); y -= lead; line = ch
            else:
                line += ch
        if line:
            c.drawString(x, y, line); y -= lead
        return y

    def header(c, title, sub=''):
        c.setFillColor(INK); c.setFont('GB', 30); c.drawString(36, H - 58, title)
        if sub:
            c.setFillColor(SUB); c.setFont('G', 16); c.drawString(36, H - 86, sub)
        c.setStrokeColor(LINE); c.line(36, H - 98, W - 36, H - 98)

    suffix = '_最新オッズ' if latest else ''
    path = os.path.join(outdir(date, ven), f"{ven}_{date}_予想{suffix}.pdf")
    c = canvas.Canvas(path, pagesize=(W, H))
    c.setTitle(f'{md} {ven} 予想{suffix}')
    # 変更点（最新版のみ）
    if latest:
        header(c, f'{md} {ven}  最新オッズでの変化', f'{stamp} 時点のオッズ ／ 本線は前日の印のまま。付け直した印は各ページの「今」の列に参考表示')
        y = H - 130
        for R in sorted(changes):
            st = '発走済み' if R in started else ('変更あり' if changes[R] else '変更なし')
            c.setFillColor(WARN if changes[R] else SUB); c.setFont('GB', 17)
            c.drawString(36, y, f"{R}R  {st}")
            yy = y
            for x in changes[R][:3]:
                yy = wrap(c, x, 170, yy, W - 210, size=14, lead=18, color=WARN)
            y = min(y - 26, yy - 4)
            if y < 40:
                c.showPage(); header(c, '変更点（続き）'); y = H - 130
        c.showPage()
    # 一覧
    header(c, f'{md} {ven} 予想一覧',
           '新馬・障害は除く ／ 印は前日オッズでの⑦統合スコア順' + (f'（{stamp}時点の変化は前のページ）' if latest else ''))
    cols = [36, 86, 300, 520, 740]
    c.setFont('GB', 16); c.setFillColor(SUB)
    for x, t in zip(cols, ['R', 'レース', '◎', '○', '▲']):
        c.drawString(x, H - 128, t)
    y = H - 160
    for x in (([prev[k['R']] for k in res]) if latest else res):
        r = races[x['R']]; P = x['P']
        nm = lambda s: (lambda q: f"{int(q.iloc[0]['馬番'])} {q.iloc[0]['馬名']}" if len(q) else '')(P[P['印'] == s])
        c.setFillColor(INK); c.setFont('GB', 18); c.drawString(cols[0], y, f"{x['R']}R")
        c.setFont('G', 14); c.drawString(cols[1], y, f"{r['レース名'][:9]} {r['芝ダ']}{r['距離']}")
        c.setFont('GB', 16)
        for xx, s in zip(cols[2:], ['◎', '○', '▲']):
            c.drawString(xx, y, nm(s)[:13])
        c.setStrokeColor(LINE); c.line(36, y - 11, W - 36, y - 11)
        y -= 33
    hit = [f"{x['R']}R {int(h['馬番'])} {h['馬名']}" for x in res for _, h in x['P'][x['P']['穴']].iterrows()]
    c.setFillColor(WARN); c.setFont('GB', 15)
    c.drawString(36, 26, '穴候補（実験枠・少額）：' + ('／'.join(hit) if hit else 'なし'))
    c.showPage()
    # 各レース
    for x in res:
        r = races[x['R']]; cr = r['course']; P = x['P']
        header(c, f"{x['R']}R  {r['レース名']}", f"{r['芝ダ']}{r['距離']}・{r['頭数']}頭・{r['発走']}発走"
               + (f"　（{stamp}時点のオッズ）" if latest and x['R'] not in started else ''))
        y = H - 126
        c.setFont('G', 15); c.setFillColor(SUB)
        c.drawString(36, y, f"コース例年（{cr['基準']}）：4角A帯の複勝{cr['A複']:.0f}%・D帯{cr['D複']:.0f}% ／ 勝ち馬の上がり平均{cr['勝馬上順']:.1f}位 ／ 1番人気の勝率{cr['人1勝']:.0f}%")
        y -= 22
        sim = r['シミュ高']
        c.drawString(36, y, f"位置が読める馬 {sim}/{r['頭数']}頭" + ('（少ない → 位置では判断しない）' if sim is not None and sim < r['頭数'] / 2 else '')
                     + f"　1番人気：{x['fav_note']}")
        y -= 34
        tc = [36, 84, 136, 330, 500, 590] if not latest else [36, 76, 120, 166, 356, 500, 590]
        heads = (['印', '馬番', '馬名', '人気(倍)', '期待勝率', 'ひとこと'] if not latest else
                 ['印', '今', '馬番', '馬名', '人気 前日→今(倍)', '期待勝率', 'ひとこと'])
        c.setFont('GB', 15); c.setFillColor(SUB)
        for xx, s in zip(tc, heads):
            c.drawString(xx, y, s)
        y -= 8; c.setStrokeColor(LINE); c.line(36, y, W - 36, y); y -= 24
        if latest:
            pm = prev[x['R']]['P'].set_index('馬番')['印']
            P = P.copy(); P['今印'] = P['印']; P['印'] = P['馬番'].map(pm).fillna('')
            Q = P[(P['印'] != '') | (P['今印'] != '')].copy()
            Q['o'] = Q['印'].map(ORDER).fillna(Q['今印'].map(ORDER).fillna(9) + 10)
            tc = [tc[0]] + tc[2:]                       # 以下は前日版と同じ列位置の扱いにする
        else:
            Q = P[P['印'] != ''].copy(); Q['o'] = Q['印'].map(ORDER)
        for _, h in Q.sort_values(['o', '実効'], ascending=[True, False]).iterrows():
            col = WARN if h['印'] == '消' else INK
            c.setFillColor(ACC if h['印'] in ('◎', '○') else col); c.setFont('GB', 22); c.drawString(tc[0], y - 2, h['印'] or '－')
            if latest:
                now_m = h['今印'] or '－'
                c.setFillColor(WARN if now_m != (h['印'] or '－') else SUB); c.setFont('GB', 18); c.drawString(76, y - 1, now_m)
            c.setFillColor(col); c.setFont('GB', 20); c.drawString(tc[1] + 6, y, f"{int(h['馬番'])}")
            c.drawString(tc[2], y, str(h['馬名'])[:10] if not latest else str(h['馬名'])[:9])
            c.setFont('G', 16)
            pop = (f"{int(h['前日人気'])}→{int(h['人気'])}番人気" if latest else f"{int(h['人気'])}番人気") + f"({h['オッズ']:.1f})"
            moved = latest and abs(int(h['人気']) - int(h['前日人気'])) >= 3
            c.setFillColor(WARN if moved else col); c.drawString(tc[3], y, pop)
            c.setFillColor(col); c.drawString(tc[4], y, f"{h['期待勝率']:.1f}%" + ('→8.7' if h['警告'] and h['期待勝率'] > 8.7 else ''))
            bits = [f"模型{int(h['模型順'])}位"]
            if h['上り評価'] == '①最上位': bits.append('上がり予測◎')
            elif h['上り評価'] in ('④下位', '⑤最下位'): bits.append('上がり予測×')
            bits.append(f"想定{h['想定帯']}帯" + ('（位置の信頼度高）' if h['信頼度'] == '高' else ''))
            if h['警告']: bits.append('警告→割引')
            if h['穴']: bits.insert(0, '★穴候補（実験枠）')
            if h['印'] == '消': bits.insert(0, '消し：想定D帯×位置の読みが確か×D帯不利のコース')
            yy = wrap(c, '・'.join(bits), tc[5], y, W - 36 - tc[5], size=13, lead=16,
                      color=(WARN if (h['穴'] or h['警告']) else SUB))
            y = min(y - 34, yy - 6)
        if x['取消']:
            c.setFillColor(WARN); c.setFont('G', 14); c.drawString(36, 98, f"取消・オッズなし: {x['取消']}")
        hit = [f"{int(h['馬番'])} {h['馬名']}（{int(h['人気'])}番人気）" for _, h in P[P['穴']].iterrows()]
        c.setFillColor(WARN if hit else SUB); c.setFont('GB' if hit else 'G', 16)
        c.drawString(36, 76, '穴候補（実験枠・少額）：' + ('／'.join(hit) + '　▲相当' if hit else 'なし'))
        c.setStrokeColor(LINE); c.line(36, 64, W - 36, 64)
        fav = P.sort_values('人気').iloc[0]
        buy = ('軸：◎○。1番人気は警告で割引（▲以下）' if fav['警告'] else
               '軸：◎（単勝の軸にしてよい型）→ ○▲△へ' if '単騎' in x['fav_note'] and '避ける' not in x['fav_note'] else
               '軸：◎○の2頭（1番人気の単勝1点は避ける）' if '避ける' in x['fav_note'] else
               '軸：◎中心。◎○の2頭軸も可 → ▲△へ')
        c.setFillColor(ACC); c.setFont('GB', 18); c.drawString(36, 38, buy)
        c.showPage()
    c.save()
    if os.path.isdir(ONEDRIVE):
        shutil.copy(path, os.path.join(ONEDRIVE, os.path.basename(path)))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prepare', 'watch', 'once'])
    ap.add_argument('date'); ap.add_argument('spec')
    ap.add_argument('--interval', type=int, default=10)
    a = ap.parse_args()
    if a.cmd == 'prepare':
        prepare(a.date, a.spec)
    elif a.cmd == 'once':
        check(a.date, a.spec)
    else:
        watch(a.date, a.spec, a.interval)


if __name__ == '__main__':
    main()
