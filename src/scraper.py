"""
netkeiba.com scraper for race entries and horse past performance.

Race ID format: YYYYVVKKDDRRR
  YYYY = year
  VV   = venue code (05=Tokyo, 06=Nakayama, 07=Chukyo, 08=Kyoto, 09=Hanshin, ...)
  KK   = kai (meeting number at that venue for the year)
  DD   = day number within the meeting
  RR   = race number (01-12)
"""

import time
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en-US;q=0.9",
}

VENUE_CODES = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}

SLEEP_SEC = 1.5  # polite crawl delay


def _get(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.encoding = resp.apparent_encoding
    resp.raise_for_status()
    time.sleep(SLEEP_SEC)
    return BeautifulSoup(resp.text, "lxml")


# ---------------------------------------------------------------------------
# 出走表 (shutuba) scraping
# ---------------------------------------------------------------------------

def scrape_shutuba(race_id: str) -> pd.DataFrame:
    """
    出走表を取得して DataFrame で返す。
    columns: race_id, waku, umaban, horse_name, horse_id,
             sex_age, kinryo, jockey, jockey_id, trainer, trainer_id,
             bataiju, popular
    """
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    print(f"[shutuba] {url}")
    soup = _get(url)

    rows = []
    table = soup.find("table", class_=re.compile(r"Shutuba_Table|HorseList"))
    if table is None:
        # フォールバック: 最初の大きなテーブル
        tables = soup.find_all("table")
        table = max(tables, key=lambda t: len(t.find_all("tr")), default=None)

    if table is None:
        print("[shutuba] テーブルが見つかりません")
        return pd.DataFrame()

    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 7:
            continue
        try:
            waku = tds[0].get_text(strip=True)
            umaban = tds[1].get_text(strip=True)

            horse_a = tr.find("a", href=re.compile(r"/horse/\d+"))
            horse_name = horse_a.get_text(strip=True) if horse_a else ""
            horse_id = ""
            if horse_a:
                m = re.search(r"/horse/(\d+)", horse_a["href"])
                horse_id = m.group(1) if m else ""

            # 性齢・斤量・騎手の列インデックスはページにより多少ずれる
            sex_age = tds[4].get_text(strip=True) if len(tds) > 4 else ""
            kinryo = tds[5].get_text(strip=True) if len(tds) > 5 else ""

            jockey_a = tr.find("a", href=re.compile(r"/jockey/"))
            jockey = jockey_a.get_text(strip=True) if jockey_a else ""
            jockey_id = ""
            if jockey_a:
                m = re.search(r"/jockey/(\w+)", jockey_a["href"])
                jockey_id = m.group(1) if m else ""

            trainer_a = tr.find("a", href=re.compile(r"/trainer/"))
            trainer = trainer_a.get_text(strip=True) if trainer_a else ""
            trainer_id = ""
            if trainer_a:
                m = re.search(r"/trainer/(\w+)", trainer_a["href"])
                trainer_id = m.group(1) if m else ""

            # 馬体重（括弧付き増減を含む）
            bataiju_td = next(
                (td for td in tds if re.search(r"\d{3}\(", td.get_text())),
                None,
            )
            bataiju = bataiju_td.get_text(strip=True) if bataiju_td else ""

            if not umaban.isdigit():
                continue

            rows.append({
                "race_id": race_id,
                "waku": waku,
                "umaban": int(umaban),
                "horse_name": horse_name,
                "horse_id": horse_id,
                "sex_age": sex_age,
                "kinryo": kinryo,
                "jockey": jockey,
                "jockey_id": jockey_id,
                "trainer": trainer,
                "trainer_id": trainer_id,
                "bataiju": bataiju,
            })
        except Exception as e:
            print(f"  row parse error: {e}")
            continue

    df = pd.DataFrame(rows)
    print(f"  → {len(df)} 頭取得")
    return df


# ---------------------------------------------------------------------------
# レース結果 (result) scraping
# ---------------------------------------------------------------------------

def scrape_result(race_id: str) -> pd.DataFrame:
    """
    レース結果を取得。
    columns: race_id, chakujun, waku, umaban, horse_name, horse_id,
             sex_age, kinryo, jockey, time, chakusa, popular, odds
    """
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    print(f"[result] {url}")
    soup = _get(url)

    rows = []
    table = soup.find("table", id="All_Result_Table")
    if table is None:
        table = soup.find("table", class_=re.compile(r"Result"))

    if table is None:
        print("[result] テーブルが見つかりません")
        return pd.DataFrame()

    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 8:
            continue
        try:
            chakujun = tds[0].get_text(strip=True)
            if not re.match(r"^\d+$", chakujun):
                continue

            waku = tds[1].get_text(strip=True)
            umaban = tds[2].get_text(strip=True)

            horse_a = tr.find("a", href=re.compile(r"/horse/\d+"))
            horse_name = horse_a.get_text(strip=True) if horse_a else ""
            horse_id = ""
            if horse_a:
                m = re.search(r"/horse/(\d+)", horse_a["href"])
                horse_id = m.group(1) if m else ""

            sex_age = tds[4].get_text(strip=True) if len(tds) > 4 else ""
            kinryo = tds[5].get_text(strip=True) if len(tds) > 5 else ""

            jockey_a = tr.find("a", href=re.compile(r"/jockey/"))
            jockey = jockey_a.get_text(strip=True) if jockey_a else ""

            time_str = tds[7].get_text(strip=True) if len(tds) > 7 else ""
            chakusa = tds[8].get_text(strip=True) if len(tds) > 8 else ""
            popular = tds[10].get_text(strip=True) if len(tds) > 10 else ""
            odds = tds[11].get_text(strip=True) if len(tds) > 11 else ""

            rows.append({
                "race_id": race_id,
                "chakujun": int(chakujun),
                "waku": waku,
                "umaban": int(umaban) if umaban.isdigit() else umaban,
                "horse_name": horse_name,
                "horse_id": horse_id,
                "sex_age": sex_age,
                "kinryo": kinryo,
                "jockey": jockey,
                "time": time_str,
                "chakusa": chakusa,
                "popular": popular,
                "odds": odds,
            })
        except Exception as e:
            print(f"  row parse error: {e}")
            continue

    df = pd.DataFrame(rows)
    print(f"  → {len(df)} 頭取得")
    return df


# ---------------------------------------------------------------------------
# レース情報 (race info) scraping
# ---------------------------------------------------------------------------

def scrape_race_info(race_id: str) -> dict:
    """
    レース名・距離・馬場・天気などのメタ情報を取得。
    """
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    soup = _get(url)

    info = {"race_id": race_id}

    # レース名
    race_name_tag = soup.find("h1", class_=re.compile(r"RaceName|race_name", re.I))
    if race_name_tag is None:
        race_name_tag = soup.find("div", class_=re.compile(r"RaceName", re.I))
    info["race_name"] = race_name_tag.get_text(strip=True) if race_name_tag else ""

    # 距離・馬場・天気などのテキスト
    data_div = soup.find("div", class_=re.compile(r"RaceData|race_data", re.I))
    if data_div:
        text = data_div.get_text(" ", strip=True)
        info["race_data_raw"] = text

        m = re.search(r"([芝ダ障])(\d+)m", text)
        if m:
            info["track_type"] = m.group(1)
            info["distance"] = int(m.group(2))

        m = re.search(r"天候\s*[:：]\s*(\S+)", text)
        info["weather"] = m.group(1) if m else ""

        m = re.search(r"馬場\s*[:：]\s*(\S+)", text)
        info["baba"] = m.group(1) if m else ""

    venue_code = race_id[4:6]
    info["venue"] = VENUE_CODES.get(venue_code, venue_code)

    return info


# ---------------------------------------------------------------------------
# 馬の過去成績 scraping
# ---------------------------------------------------------------------------

def scrape_horse_results(horse_id: str, max_races: int = 20) -> pd.DataFrame:
    """
    馬の過去成績を取得。
    columns: date, race_name, venue, distance, track_type, baba,
             chakujun, time, kinryo, jockey, popular, odds, prize
    """
    url = f"https://db.netkeiba.com/horse/result/{horse_id}/"
    print(f"[horse] {url}")
    soup = _get(url)

    table = soup.find("table", class_=re.compile(r"db_h_race_results|nk_tb_common"))
    if table is None:
        tables = soup.find_all("table")
        table = max(tables, key=lambda t: len(t.find_all("tr")), default=None)

    if table is None:
        return pd.DataFrame()

    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    rows = []
    for tr in table.find_all("tr")[1 : max_races + 1]:
        tds = tr.find_all("td")
        if len(tds) < 5:
            continue
        row = {headers[i]: tds[i].get_text(strip=True) for i in range(min(len(headers), len(tds)))}
        row["horse_id"] = horse_id
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  → {len(df)} レース分取得")
    return df


# ---------------------------------------------------------------------------
# 騎手成績集計
# ---------------------------------------------------------------------------

def scrape_jockey_stats(jockey_id: str) -> dict:
    """騎手の通算・直近成績サマリを取得。"""
    url = f"https://db.netkeiba.com/jockey/result/recent/{jockey_id}/"
    print(f"[jockey] {url}")
    soup = _get(url)

    stats = {"jockey_id": jockey_id}
    table = soup.find("table")
    if table:
        rows = table.find_all("tr")
        if len(rows) >= 2:
            headers = [th.get_text(strip=True) for th in rows[0].find_all("th")]
            values = [td.get_text(strip=True) for td in rows[1].find_all("td")]
            stats.update(dict(zip(headers, values)))
    return stats
