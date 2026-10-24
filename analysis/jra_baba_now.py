# -*- coding: utf-8 -*-
"""JRA公式の当日のクッション値・含水率を直接取得する（ブラウザ不要）。

  python analysis/jra_baba_now.py                     # 各場の最新の測定値を表示
  python analysis/jra_baba_now.py --wait 10/24        # その日の測定値が全場そろうまで5分おきに確認（最大2時間）
  python analysis/jra_baba_now.py --wait 10/24 --save # そろったら data/processed/jra_baba.parquet に追記

馬場情報ページは JavaScript で描画されるが、数値は次の2ファイルから読み込まれている（Referer が必要）。
  https://www.jra.go.jp/keiba/baba/_data_cushion.html  （<div id="rcX" title="場名"> ごとに 測定時刻・クッション値）
  https://www.jra.go.jp/keiba/baba/_data_moist.html    （同じく 芝ゴール前/4角・ダートゴール前/4角）
文字コードは Shift_JIS。開催中の全場（通常3場）が1ファイルに入っている。
※追記する行の 開催回・日次 は空欄。翌週木曜以降に fetch_jra_baba.py を実行すると公式PDFの値で置き換わる。
"""
import argparse, os, re, sys, time, urllib.request
from datetime import datetime
import numpy as np, pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data', 'processed', 'jra_baba.parquet')
H = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.jra.go.jp/keiba/baba/'}
YOBI = '月火水木金土日'


def get(name):
    return urllib.request.urlopen(urllib.request.Request('https://www.jra.go.jp/keiba/baba/' + name, headers=H),
                                  timeout=20).read().decode('shift_jis', 'ignore')


def blocks(t):
    """{場名: ブロックのHTML}"""
    out = {}
    parts = re.split(r'<div id="rc\w" title="([^"]+)">', t)
    for i in range(1, len(parts), 2):
        out[parts[i]] = parts[i + 1]
    return out


def fetch():
    cu = {v: [(re.sub(r'\s+', '', a), float(b)) for a, b in
              re.findall(r'<div class="time">([^<]+)</div>\s*<div class="cushion">([\d.]+)</div>', s)]
          for v, s in blocks(get('_data_cushion.html')).items()}
    mo = {}
    for v, s in blocks(get('_data_moist.html')).items():
        mo[v] = [(re.sub(r'\s+', '', a), float(b), float(c), float(d), float(e)) for a, b, c, d, e in re.findall(
            r'<div class="time">([^<]+)</div>\s*<div class="turf">\s*<span[^>]*>([\d.]+)</span>\s*<span[^>]*>([\d.]+)</span>\s*</div>'
            r'\s*<div class="dirt">\s*<span[^>]*>([\d.]+)</span>\s*<span[^>]*>([\d.]+)</span>', s)]
    return cu, mo


def latest(cu, mo, md=None):
    """md='10月24日' を渡すとその日の値だけ。無ければ None。"""
    rows = {}
    for v in cu:
        c = [x for x in cu[v] if md is None or x[0].startswith(md)]
        m = [x for x in mo.get(v, []) if md is None or x[0].startswith(md)]
        rows[v] = (c[0] if c else None, m[0] if m else None)
    return rows


def show(rows):
    for v, (c, m) in rows.items():
        cs = f"クッション値 {c[1]}（{c[0]}）" if c else 'クッション値 未発表'
        ms = f"芝 {m[1]}%/{m[2]}%  ダ {m[3]}%/{m[4]}%（{m[0]}）" if m else '含水率 未発表'
        print(f"  {v}: {cs}  ／  {ms}")


def save(rows, md):
    B = pd.read_parquet(OUT)
    mo_, d_ = re.match(r'(\d+)月(\d+)日', md).groups()
    ymd = f"{datetime.now().year % 100:02d}{int(mo_):02d}{int(d_):02d}"
    yo = YOBI[datetime(datetime.now().year, int(mo_), int(d_)).weekday()]
    new = [dict(場=v, 開催回=np.nan, 日次=np.nan, 日付=ymd, 曜日=yo, 使用コース=None, クッション値=c[1],
                芝含水率_ゴール前=m[1], 芝含水率_4角=m[2], ダ含水率_ゴール前=m[3], ダ含水率_4角=m[4])
           for v, (c, m) in rows.items() if c and m]
    B = pd.concat([B, pd.DataFrame(new)], ignore_index=True).drop_duplicates(['場', '日付'], keep='last').sort_values(['日付', '場'])
    B.to_parquet(OUT, index=False)
    print(f"保存 {OUT}（{len(new)}場を追記・最終 {B['日付'].max()}）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wait', help='例 10/24')
    ap.add_argument('--interval', type=int, default=5)
    ap.add_argument('--timeout', type=int, default=115, help='分')
    ap.add_argument('--save', action='store_true')
    a = ap.parse_args()
    if not a.wait:
        cu, mo = fetch(); print('各場の最新の測定値'); show(latest(cu, mo)); return
    m, d = a.wait.split('/'); md = f"{int(m)}月{int(d)}日"
    t0 = time.time()
    while True:
        try:
            cu, mo = fetch(); rows = latest(cu, mo, md)
        except Exception as e:
            print('取得NG', e); rows = {}
        done = rows and all(c and m_ for c, m_ in rows.values())
        print(datetime.now().strftime('%H:%M'), md, '全場そろった' if done else
              '未発表: ' + '・'.join(v for v, (c, m_) in rows.items() if not (c and m_)), flush=True)
        if done:
            show(rows)
            if a.save:
                save(rows, md)
            return
        if time.time() - t0 > a.timeout * 60:
            print('時間切れ。発表済みの分だけ表示します'); show(rows); return
        time.sleep(a.interval * 60)


if __name__ == '__main__':
    main()
