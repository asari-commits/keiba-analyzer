# -*- coding: utf-8 -*-
"""海外（と地方）の出走歴を netkeiba から集める。master は JRA の国内成績だけで、
海外・地方のレースはモデルの特徴量に入っていない（2026/11 ビザンチンドリームの件）。

対象: master 期間の重賞に出走した全馬。
  1) 重賞の結果ページ（race.netkeiba.com/race/result.html）から馬名と netkeiba の馬IDを取る
  2) 馬の全戦績ページ（db.netkeiba.com/horse/result/<id>/）から全レースを取る
取得結果は data/overseas_cache/ に1件ずつ保存し、再実行時は取得済みを飛ばす（途中で止めても再開できる）。
アクセスは1件ごとに間隔を空ける。

出力: data/overseas_cache/horse_runs.parquet（馬名・馬ID・全レースの行）
使い方: python analysis/overseas_fetch.py
"""
import json, re, sys, time
from pathlib import Path

import pandas as pd
import requests
import truststore

truststore.inject_into_ssl()
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "overseas_cache"
(CACHE / "race").mkdir(parents=True, exist_ok=True)
(CACHE / "horse").mkdir(parents=True, exist_ok=True)
H = {"User-Agent": "Mozilla/5.0 (keiba-analyzer research)"}
WAIT = 0.8
VENUE = {"札": "01", "函": "02", "福": "03", "新": "04", "東": "05", "中": "06", "名": "07", "京": "08", "阪": "09", "小": "10"}


def nichi(code):
    t = str(code)[2:].strip().upper()
    return int(t) if t.isdigit() else 10 + ord(t) - ord("A")


def race_id(r):
    k = str(r["開催"])
    return f"20{str(r['日付'])[:2]}{VENUE[k[1]]}{int(k[0]):02d}{nichi(k):02d}{int(r['Ｒ']):02d}"


def get(url):
    for i in range(3):
        try:
            t = requests.get(url, headers=H, timeout=20)
            t.encoding = t.apparent_encoding
            if t.status_code == 200:
                return t.text
        except Exception as e:
            print("  retry", url, e, flush=True)
        time.sleep(3 * (i + 1))
    return None


def main():
    m = pd.read_parquet(ROOT / "data" / "processed" / "master.parquet")
    g = m[m["レース名"].astype(str).str.contains("G1|G2|G3")]
    races = g.drop_duplicates(["日付", "開催", "Ｒ"])
    print(f"重賞 {len(races)} レース / 出走馬 {g['馬名'].nunique()} 頭", flush=True)

    # 1) 結果ページから 馬名 → 馬ID
    ids = {}
    for k, (_, r) in enumerate(races.iterrows()):
        rid = race_id(r)
        f = CACHE / "race" / f"{rid}.json"
        if not f.exists():
            html = get(f"https://race.netkeiba.com/race/result.html?race_id={rid}")
            pairs = re.findall(r'href="https://db\.netkeiba\.com/horse/(\d+)"[^>]*title="([^"]+)"', html or "")
            f.write_text(json.dumps({"race_id": rid, "pairs": pairs}, ensure_ascii=False), encoding="utf-8")
            time.sleep(WAIT)
        for hid, nm in json.loads(f.read_text(encoding="utf-8"))["pairs"]:
            ids[nm] = hid
        if k % 50 == 0:
            print(f"  race {k}/{len(races)}  ID取得 {len(ids)}頭", flush=True)
    want = sorted(set(g["馬名"]))
    miss = [n for n in want if n not in ids]
    print(f"ID取得 {len(want) - len(miss)}/{len(want)}頭（取れなかった {len(miss)}頭: {miss[:10]}）", flush=True)

    # 2) 馬ごとの全戦績
    rows = []
    for k, nm in enumerate(want):
        hid = ids.get(nm)
        if not hid:
            continue
        f = CACHE / "horse" / f"{hid}.json"
        if not f.exists():
            html = get(f"https://db.netkeiba.com/horse/result/{hid}/") or ""
            recs = []
            for row in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S)[1:]:
                c = [re.sub(r"<[^>]+>", "", x).strip() for x in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)]
                if len(c) > 11 and re.match(r"\d{4}/\d{2}/\d{2}", c[0]):
                    recs.append(c)
            f.write_text(json.dumps({"name": nm, "id": hid, "rows": recs}, ensure_ascii=False), encoding="utf-8")
            time.sleep(WAIT)
        d = json.loads(f.read_text(encoding="utf-8"))
        for c in d["rows"]:
            rows.append(dict(馬名=nm, 馬ID=hid, 日付=c[0], 開催=c[1], 天気=c[2], R=c[3], レース名=c[4], 頭数=c[6] if len(c) > 6 else "",
                             枠=c[7] if len(c) > 7 else "", 馬番=c[8] if len(c) > 8 else "", 人気=c[10] if len(c) > 10 else "", 着順=c[11]))
        if k % 100 == 0:
            print(f"  horse {k}/{len(want)}", flush=True)
    out = pd.DataFrame(rows)
    out.to_parquet(CACHE / "horse_runs.parquet", index=False)
    print(f"保存: {CACHE / 'horse_runs.parquet'}  {len(out)}行 / {out['馬名'].nunique()}頭", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
