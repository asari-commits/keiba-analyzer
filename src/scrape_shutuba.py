"""
netkeibaから出馬表（shutuba-hyo）を取得する。
取得したデータを master.csv の特徴量パイプラインに流せる形式で返す。

使い方:
    from scrape_shutuba import get_race_list, get_shutuba
    races = get_race_list('20260621')       # 開催日のレース一覧
    df    = get_shutuba('202506050811')     # 特定レースの出馬表
"""
import re
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from bs4 import BeautifulSoup

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/124.0.0.0 Safari/537.36'
    )
}

# netkeiba 場コード → 競馬場名（フル）
VENUE_CODE_MAP = {
    '01': '札幌', '02': '函館', '03': '福島', '04': '新潟',
    '05': '東京', '06': '中山', '07': '中京', '08': '京都',
    '09': '阪神', '10': '小倉',
}

# 競馬場名 → Target風 開催コード略称（parse_venue で使う）
VENUE_ABBR = {
    '東京': '東', '中山': '中', '京都': '京', '阪神': '阪',
    '中京': '名', '小倉': '小', '新潟': '新', '福島': '福',
    '函館': '函', '札幌': '札',
}


def _get(url: str, retry: int = 3) -> BeautifulSoup:
    for i in range(retry):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            r.encoding = r.apparent_encoding
            return BeautifulSoup(r.text, 'lxml')
        except Exception as e:
            if i == retry - 1:
                raise
            time.sleep(2 ** i)


def get_race_list(date_str: str) -> list[dict]:
    """
    指定日の全レース一覧を返す。
    date_str: 'YYYYMMDD' 形式

    戻り値: [{'race_id': '...', 'venue': '東京', 'r': 1, 'race_name': '...', 'date': '...'}]
    """
    url = f'https://race.netkeiba.com/top/race_list.html?kaisai_date={date_str}'
    soup = _get(url)
    races = []
    seen = set()

    for a in soup.select('a[href]'):
        href = a.get('href', '')
        # shutuba.html または race.html のリンクからrace_idを取得
        m = re.search(r'race_id=(\d{12})', href)
        if not m:
            continue
        race_id = m.group(1)
        if race_id in seen:
            continue
        seen.add(race_id)

        venue_code = race_id[4:6]
        r_num = int(race_id[10:12])
        race_name = a.get_text(strip=True)
        venue = VENUE_CODE_MAP.get(venue_code, venue_code)

        races.append({
            'race_id':   race_id,
            'venue':     venue,
            'r':         r_num,
            'race_name': race_name or f'R{r_num}',
            'date':      date_str,
        })

    # R番号でソート
    races.sort(key=lambda x: (x['venue'], x['r']))
    return races


def get_shutuba(race_id: str) -> pd.DataFrame:
    """
    指定レースの出馬表をDataFrameで返す。
    race_id: '202506050811' 形式（12桁）

    返すDataFrameの列は master.csv と同じ名前に合わせる。
    着順・走破タイム等の結果列は NaN。
    開催列は '1東3' 相当の形式 '99{abbr}1' で格納（parse_venue が読める）。
    """
    url = f'https://race.netkeiba.com/race/shutuba.html?race_id={race_id}'
    soup = _get(url)

    # ── レース基本情報 ──────────────────────────────────────────
    date_str   = race_id[:4] + race_id[4:6] + race_id[6:8]   # YYYYMMDD
    venue_code = race_id[4:6]
    r_num      = int(race_id[10:12])
    venue_name = VENUE_CODE_MAP.get(venue_code, venue_code)
    abbr       = VENUE_ABBR.get(venue_name, venue_name[0])

    # parse_venue() が r'\d([^\d]+)\d' でマッチできる形式
    # 開催回は不明なので 1 固定、日数も 1 固定で格納
    kaisai_str = f'1{abbr}1'

    # レース名
    race_name = ''
    for sel in ['.RaceName', '.race_name', 'h1.RaceName', '.RaceMainTitle']:
        el = soup.select_one(sel)
        if el:
            race_name = el.get_text(strip=True)
            break

    # 距離・芝ダ・馬場
    turf_dist_text = ''
    for sel in ['.RaceData01', '.racedata', '.race_data']:
        el = soup.select_one(sel)
        if el:
            turf_dist_text = el.get_text(' ', strip=True)
            break

    dist_m = re.search(r'(\d{3,4})m', turf_dist_text, re.IGNORECASE)
    dist   = int(dist_m.group(1)) if dist_m else None
    is_turf = 1 if '芝' in turf_dist_text and 'ダ' not in turf_dist_text else 0
    surf   = '芝' if is_turf else 'ダ'

    baba_m = re.search(r'(良|稍重|重|不良)', turf_dist_text)
    baba   = baba_m.group(1) if baba_m else ''

    # ── 出馬表テーブル ──────────────────────────────────────────
    # 複数のセレクタを試す
    table = None
    for sel in ['table.Shutuba_Table', 'table#shutuba_table', '.RaceTable01 table',
                'table.RaceTable01', '.ShutubaTable']:
        table = soup.select_one(sel)
        if table:
            break

    rows = []
    if table:
        for tr in table.select('tr.HorseList, tr[class*="HorseList"]'):
            row = _parse_horse_row(tr)
            if row:
                rows.append(row)

    # フォールバック: tableが見つからない場合は全trを走査
    if not rows and table is None:
        for tr in soup.select('tr'):
            cells = tr.find_all('td')
            if len(cells) >= 8:
                row = _parse_horse_row(tr)
                if row and row.get('馬名'):
                    rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # 共通列を付与
    df['日付']     = date_str
    df['開催']     = kaisai_str      # '1東1' など ← parse_venue が読める
    df['Ｒ']       = r_num
    df['レース名']  = race_name
    df['芝・ダ']   = surf
    df['距離']     = dist
    df['馬場状態']  = baba

    # 結果列はNaN
    df['着順']       = None
    df['走破タイム'] = None
    df['着順_num']   = None

    # 型変換
    df['日付_dt'] = pd.to_datetime(df['日付'], format='%Y%m%d', errors='coerce')
    df['距離']    = pd.to_numeric(df['距離'], errors='coerce')
    df['斤量']    = pd.to_numeric(df['斤量'], errors='coerce')

    # 馬体重を数値化
    if '馬体重' in df.columns:
        df['馬体重_num'] = pd.to_numeric(
            df['馬体重'].astype(str).str.extract(r'(\d{3,4})')[0], errors='coerce')

    # 人気を数値化
    if '人気' in df.columns:
        df['人気'] = pd.to_numeric(df['人気'], errors='coerce')

    return df


def _parse_horse_row(tr) -> dict | None:
    """<tr> から馬情報を抽出"""
    # 馬番
    banum = ''
    for sel in ['.Umaban', 'td.Umaban', 'td:nth-child(3)']:
        el = tr.select_one(sel)
        if el:
            banum = el.get_text(strip=True)
            break

    # 馬名
    horse_name = ''
    for sel in ['.HorseName a', '.Horse_Name a', 'td a[href*="/horse/"]']:
        el = tr.select_one(sel)
        if el:
            horse_name = el.get_text(strip=True)
            break
    if not horse_name:
        return None

    # 性齢
    sex, age = '', ''
    for sel in ['.Barei', '.Sex', 'td.Barei']:
        el = tr.select_one(sel)
        if el:
            t = el.get_text(strip=True)
            sex = t[0] if t else ''
            age = t[1:] if len(t) > 1 else ''
            break

    # 斤量
    kinryo = ''
    for sel in ['.Futan', '.Kinryo', 'td.Futan']:
        el = tr.select_one(sel)
        if el:
            kinryo = el.get_text(strip=True).replace('☆', '').replace('▲', '').replace('△', '').strip()
            break

    # 騎手
    jockey = ''
    for sel in ['.Jockey a', '.Kishu a', 'td a[href*="/jockey/"]']:
        el = tr.select_one(sel)
        if el:
            jockey = el.get_text(strip=True)
            break

    # 馬体重（当日未発表時は '-' の場合あり）
    weight_text = ''
    for sel in ['.Weight', '.Taiju', 'td.Weight']:
        el = tr.select_one(sel)
        if el:
            weight_text = el.get_text(strip=True)
            break

    # 調教師
    trainer = ''
    for sel in ['.Trainer a', '.Kyusho a', 'td a[href*="/trainer/"]']:
        el = tr.select_one(sel)
        if el:
            trainer = el.get_text(strip=True)
            break

    # 人気・オッズ
    popular = None
    odds_text = ''
    for sel in ['.Popular', 'td.Popular', '.Ninki']:
        el = tr.select_one(sel)
        if el:
            t = el.get_text(strip=True)
            m = re.search(r'(\d+)', t)
            if m:
                popular = int(m.group(1))
            break
    if popular is None:
        for sel in ['.Odds span', '.OddsVal', 'span.Odds']:
            el = tr.select_one(sel)
            if el:
                odds_text = el.get_text(strip=True)
                break

    return {
        '馬番':   banum,
        '馬名':   horse_name,
        '性別':   sex,
        '年齢':   age,
        '斤量':   kinryo,
        '騎手':   jockey,
        '馬体重': weight_text,
        '調教師': trainer,
        '人気':   popular,
    }


def get_upcoming_races(days_ahead: int = 7) -> list[str]:
    """
    今日から days_ahead 日以内の開催日リストを返す（土日のみ）。
    """
    today = datetime.today()
    dates = []
    for i in range(days_ahead):
        d = today + timedelta(days=i)
        if d.weekday() in (5, 6):  # 土=5, 日=6
            dates.append(d.strftime('%Y%m%d'))
    return dates


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    upcoming = get_upcoming_races(7)
    print(f'直近の開催日: {upcoming}')
    for date in upcoming[:2]:
        races = get_race_list(date)
        print(f'\n{date}: {len(races)}レース')
        for r in races[:3]:
            print(f"  {r['venue']} R{r['r']} {r['race_name']} ({r['race_id']})")
        if races:
            print(f"\n  最初のレースの出馬表を取得...")
            df = get_shutuba(races[0]['race_id'])
            print(df[['馬名', '騎手', '斤量', '人気']].to_string())
