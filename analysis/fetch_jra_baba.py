# -*- coding: utf-8 -*-
"""JRA公式の「過去の含水率・クッション値」PDFを全開催分ダウンロードして1つの表にする。

  python analysis/fetch_jra_baba.py [開始年]

出力: data/processed/jra_baba.parquet
  場 / 開催回 / 日次 / 日付(YYMMDD) / 曜日 / 使用コース / クッション値
  / 芝含水率_ゴール前 / 芝含水率_4角 / ダ含水率_ゴール前 / ダ含水率_4角

※含水率は2018/7/27から、クッション値は2020/9/11から公開。
※PDFのレイアウトが2025年に変わっている:
   2021-2024 = 週ブロックごとに金/土/日が横並び（使用コースの記載なし）
   2025以降  = 1日1行（使用コース・測定時刻あり）
※金曜（非開催日）の測定も含む。日次がNaNの行は金曜の事前測定か旧形式。
"""
import datetime, os, re, sys, time, urllib.request
import numpy as np, pandas as pd
import pdfplumber

sys.stdout.reconfigure(encoding='utf-8')
UA = {'User-Agent': 'Mozilla/5.0'}
ROOT = r"C:\Users\asari\Downloads\Claude\keiba-analyzer"
CACHE = os.path.join(ROOT, 'data', 'raw', 'jra_baba_pdf')
OUT = os.path.join(ROOT, 'data', 'processed', 'jra_baba.parquet')
VEN = {'sapporo': '札幌', 'hakodate': '函館', 'fukushima': '福島', 'niigata': '新潟',
       'tokyo': '東京', 'nakayama': '中山', 'chukyo': '中京', 'kyoto': '京都',
       'hanshin': '阪神', 'kokura': '小倉'}
START = int(sys.argv[1]) if len(sys.argv) > 1 else 2021
os.makedirs(CACHE, exist_ok=True)


def get(url, binary=False):
    for _ in range(3):
        try:
            d = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30).read()
            return d if binary else d.decode('utf-8', 'ignore')
        except Exception:
            time.sleep(1.5)
    return None


def pdf_links(year):
    u = ('https://www.jra.go.jp/keiba/baba/archive/' if year == 2026
         else f'https://www.jra.go.jp/keiba/baba/archive/{year}.html')
    h = get(u)
    if not h:
        return []
    return sorted(set(re.findall(r'href="(/keiba/baba/archive/\d{4}pdf/[a-z]+\d{2}\.pdf)"', h)))


def _nums(blk, pat, n, occ=0):
    """blk の中で pat に続く n 個の数値（'-'はNaN）を返す。occ で何番目のヒットか指定。"""
    ms = re.findall(pat + r'[^\d\-]*((?:\s*(?:[\d.]+|-)){1,' + str(n) + r'})', blk)
    if len(ms) <= occ:
        return [np.nan] * n
    v = re.findall(r'[\d.]+|-', ms[occ])[:n]
    out = [np.nan if x == '-' else float(x) for x in v]
    return out + [np.nan] * (n - len(out))


def parse_old(txt, ven, kai):
    """2021〜2024の旧形式。"""
    rows = []
    for blk in re.split(r'(?=第[０-９\d一二三四五六七八九十]+日)', txt):
        mr = re.search(r'[（(](\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日', blk)
        mh = re.search(r'((?:[月火水木金土日]曜日\s*){2,5})', blk)
        if not mr or not mh:
            continue
        try:
            start = datetime.date(int(mr.group(1)), int(mr.group(2)), int(mr.group(3)))
        except ValueError:
            continue
        yobis = re.findall(r'([月火水木金土日])曜日', mh.group(1))
        n = len(yobis)
        cv = _nums(blk, r'芝コースクッション値', n)
        sg = _nums(blk, r'芝コース含水率\s*ゴール前', n)
        dg = _nums(blk, r'ダートコース含水率\s*ゴール前', n)
        s4 = _nums(blk, r'４コーナー', n, occ=0)
        d4 = _nums(blk, r'４コーナー', n, occ=1)
        for i in range(n):
            dt = start + datetime.timedelta(days=i)
            rows.append(dict(場=ven, 開催回=kai, 日次=np.nan,
                             日付=f"{dt.year % 100:02d}{dt.month:02d}{dt.day:02d}",
                             曜日=yobis[i], 使用コース=None, クッション値=cv[i],
                             芝含水率_ゴール前=sg[i], 芝含水率_4角=s4[i],
                             ダ含水率_ゴール前=dg[i], ダ含水率_4角=d4[i]))
    return rows


def parse_new(txt, ven, kai, year):
    """2025以降の新形式（1日1行）。"""
    rows = []
    for line in txt.split('\n'):
        s = line.replace('　', ' ').strip()
        m = re.search(r'(?:第\s*(\d+)日\s*)?(\d{1,2})月\s*(\d{1,2})日\s*([月火水木金土日])曜日\s*([A-E])?\s*'
                      r'(?:(\d{1,2}:\d{2})\s*([\d.]+)\s*)?'
                      r'(?:(\d{1,2}:\d{2})\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+))?', s)
        if not m or not m.group(2):
            continue
        nichi, mo, da, yo, course = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4), m.group(5)
        cv = float(m.group(7)) if m.group(7) else np.nan
        sg, s4 = (float(m.group(9)), float(m.group(10))) if m.group(9) else (np.nan, np.nan)
        dg, d4 = (float(m.group(11)), float(m.group(12))) if m.group(11) else (np.nan, np.nan)
        rows.append(dict(場=ven, 開催回=kai, 日次=int(nichi) if nichi else np.nan,
                         日付=f"{year % 100:02d}{mo:02d}{da:02d}", 曜日=yo, 使用コース=course,
                         クッション値=cv, 芝含水率_ゴール前=sg, 芝含水率_4角=s4,
                         ダ含水率_ゴール前=dg, ダ含水率_4角=d4))
    return rows


def parse(path, ven, kai, year):
    with pdfplumber.open(path) as pdf:
        txt = '\n'.join((p.extract_text() or '') for p in pdf.pages)
    if ('開催日次' in txt) or ('使用コース' in txt):
        return parse_new(txt, ven, kai, year)
    return parse_old(txt, ven, kai)


all_rows = []
for year in range(START, 2027):
    links = pdf_links(year)
    if not links:
        print(f"{year}: リンク取得NG"); continue
    print(f"{year}: {len(links)}開催")
    for href in links:
        m = re.search(r'/(\d{4})pdf/([a-z]+)(\d{2})\.pdf', href)
        if not m:
            continue
        yy, vkey, kai = int(m.group(1)), m.group(2), int(m.group(3))
        ven = VEN.get(vkey, vkey)
        path = os.path.join(CACHE, f"{yy}_{vkey}{kai:02d}.pdf")
        if not os.path.exists(path):
            d = get('https://www.jra.go.jp' + href, binary=True)
            if not d:
                print('  DL NG', href); continue
            open(path, 'wb').write(d)
            time.sleep(0.4)
        try:
            r = parse(path, ven, kai, yy)
            all_rows += r
        except Exception as e:
            print(f"  {ven}{kai}回({yy}) パースNG {type(e).__name__}: {e}")

B = pd.DataFrame(all_rows).drop_duplicates(['場', '日付']).sort_values(['日付', '場'])
B.to_parquet(OUT, index=False)
print(f"\n保存 {OUT}  {len(B):,}行")
print(f"期間 {B['日付'].min()} 〜 {B['日付'].max()}")
print(f"クッション値あり {B['クッション値'].notna().sum():,} / 芝含水率あり {B['芝含水率_ゴール前'].notna().sum():,} "
      f"/ ダ含水率あり {B['ダ含水率_ゴール前'].notna().sum():,} / 使用コースあり {B['使用コース'].notna().sum():,}")
print("\n年ごと:")
print(B.groupby(B['日付'].str[:2]).agg(行数=('場', 'size'), クッション=('クッション値', 'count'),
                                     芝含水=('芝含水率_ゴール前', 'count')).to_string())
print("\n場ごと:"); print(B.groupby('場').size().to_string())
