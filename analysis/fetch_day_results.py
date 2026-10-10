# -*- coding: utf-8 -*-
"""netkeiba から1日分の結果（着順・人気・上がり・4角順位）とラップを取得する。

  python analysis/fetch_day_results.py <YYYYMMDD> <場コード:回:日次> [<場コード:回:日次> ...]
  例) python analysis/fetch_day_results.py 20261004 05:04:02 08:04:02   # 4回東京2日目・4回京都2日目

場コード: 01札幌 02函館 03福島 04新潟 05東京 06中山 07中京 08京都 09阪神 10小倉
出力: data/raw/day_results/<YYYYMMDD>.csv（1行=1頭）と <YYYYMMDD>_meta.csv（1行=1レース、後半3F差つき）

※結果ページは UTF-8（euc-jp ではない）。列名は「着 順」のように空白が入るので除去してから判定する。
※未確定のレースは飛ばすので、開催中に何度実行してもよい（確定分だけで上書き）。
"""
import io, os, re, sys, time, urllib.request
import numpy as np, pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, 'data', 'raw', 'day_results')
UA = {'User-Agent': 'Mozilla/5.0'}
VEN = {'01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京',
       '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉'}


def get(url):
    for _ in range(3):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=25).read().decode('utf-8', 'ignore')
        except Exception:
            time.sleep(1.2)
    return ''


def strip_tags(h):
    h = re.sub(r'<script.*?</script>', ' ', h, flags=re.S | re.I)
    h = re.sub(r'<style.*?</style>', ' ', h, flags=re.S | re.I)
    h = re.sub(r'<br\s*/?>', '\n', h, flags=re.I)
    h = re.sub(r'</t[dh]>', '\t', h, flags=re.I)
    h = re.sub(r'</tr>', '\n', h, flags=re.I)
    return re.sub(r'<[^>]+>', '', h).replace('&nbsp;', ' ')


def corner_order(c4):
    """'(3,5)-7,1=2' のような通過順を {馬番: 順位} にする。同列は平均順位。"""
    order, pos = {}, []
    for g in re.findall(r'\((.*?)\)|(\d+)', c4.replace('*', '').replace('=', '-')):
        pos.append([int(x) for x in re.findall(r'\d+', g[0])] if g[0] else [int(g[1])])
    rk = 1
    for g in pos:
        for u in g:
            order[u] = rk + (len(g) - 1) / 2
        rk += len(g)
    return order


def fetch_race(rid, ven, R):
    h = get(f"https://race.netkeiba.com/race/result.html?race_id={rid}")
    if not h:
        return None, None
    t = strip_tags(h)
    sd = re.search(r'(芝|ダ|障)\s*(\d{3,4})m', t)
    cond = re.search(r'天候[:：]\s*(\S+?)\s*/\s*馬場[:：]\s*(\S+)', t)
    c4 = re.search(r'4コーナー\s*\t*([0-9\(\)\,\-\*=\s]+)', t)
    lap = re.findall(r'\n((?:\d+\.\d\t){4,})\n', t)
    laps = [float(x) for x in lap[-1].split('\t') if x.strip()] if lap else []
    try:
        tbls = pd.read_html(io.StringIO(h))
    except Exception:
        tbls = []
    res = None
    for tb in tbls:
        tb = tb.copy(); tb.columns = [re.sub(r'\s+', '', str(c)) for c in tb.columns]
        if '着順' in tb.columns and '馬番' in tb.columns:
            res = tb; break
    if res is None:
        return None, None
    cmap = {}
    for c in res.columns:
        if c == '着順': cmap[c] = '着'
        elif c == '馬番': cmap[c] = '馬番'
        elif '馬名' in c: cmap[c] = '馬名'
        elif c.startswith('人気'): cmap[c] = '人気'
        elif 'オッズ' in c or c == '単勝': cmap[c] = 'オッズ'
        elif '後3F' in c: cmap[c] = '上3F'
    res = res.rename(columns=cmap)
    if not all(k in res.columns for k in ['着', '馬番', '人気', '上3F']):
        return None, None
    order = corner_order(c4.group(1).strip()) if c4 else {}
    base = dict(場=ven, R=R, 芝ダ=sd.group(1) if sd else '', 距離=int(sd.group(2)) if sd else np.nan,
                天候=cond.group(1) if cond else '', 馬場=cond.group(2) if cond else '', 頭数=len(res))
    rows = []
    for _, r in res.iterrows():
        try:
            uma = int(r['馬番'])
        except Exception:
            continue
        rows.append(dict(base, 着=pd.to_numeric(r['着'], errors='coerce'), 馬番=uma,
                         馬名=str(r.get('馬名', '')).strip(), 人気=pd.to_numeric(r['人気'], errors='coerce'),
                         オッズ=pd.to_numeric(r.get('オッズ', np.nan), errors='coerce'),
                         上3F=pd.to_numeric(r['上3F'], errors='coerce'), pos4=order.get(uma, np.nan)))
    # 距離が200で割り切れないコース（ダ1300・ダ2100 等）は最初のラップが100mなので前半3Fを出さない
    odd = bool(sd) and int(sd.group(2)) % 200 != 0
    f3 = sum(laps[:3]) if len(laps) >= 6 and not odd else np.nan
    b3 = sum(laps[-3:]) if len(laps) >= 6 else np.nan
    meta = dict(base, 前3F=round(f3, 1) if f3 == f3 else np.nan, 後3F=round(b3, 1) if b3 == b3 else np.nan,
                後半3F差=round(b3 - f3, 1) if b3 == b3 else np.nan)
    return rows, meta


def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    ymd, specs = sys.argv[1], sys.argv[2:]
    os.makedirs(OUTDIR, exist_ok=True)
    rows, metas = [], []
    for sp in specs:
        vc, kai, nichi = sp.split(':')
        ven = VEN[vc]
        for R in range(1, 13):
            rid = f"{ymd[:4]}{vc}{int(kai):02d}{int(nichi):02d}{R:02d}"
            rr, mt = fetch_race(rid, ven, R)
            if rr is None:
                print(f"  未確定/取得NG {ven}{R}R"); continue
            rows += rr; metas.append(mt)
            print(f"OK {ven}{R}R {mt['芝ダ']}{mt['距離']} {mt['馬場']} {mt['頭数']}頭")
    out = os.path.join(OUTDIR, f"{ymd}.csv")
    pd.DataFrame(rows).to_csv(out, index=False, encoding='utf-8-sig')
    pd.DataFrame(metas).to_csv(out.replace('.csv', '_meta.csv'), index=False, encoding='utf-8-sig')
    print(f"\n保存 {out}  {len(rows)}頭 / {len(metas)}レース")


if __name__ == '__main__':
    main()
