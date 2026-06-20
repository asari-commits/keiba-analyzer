"""
Netkeibaからレース確定後の結果・払戻データを取得する。

取得内容:
  - 着順・馬番・馬名（1〜3着）
  - 単勝払戻・複勝払戻（1〜3着分）
  - 馬連払戻・三連複払戻

レースIDは scrape_odds.build_race_id() で生成する。
"""
import re
import logging
import requests
import pandas as pd
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/124.0.0.0 Safari/537.36'
    ),
    'Referer': 'https://race.netkeiba.com/',
}
_SESSION = requests.Session()
_SESSION.headers.update(_HEADERS)


def fetch_race_result(race_id: str) -> dict:
    """
    Netkeibaのレース結果ページから着順・払戻を取得する。

    Returns
    -------
    dict with keys:
      'race_id'        : str
      'horses'         : list of {'chaku': int, 'umaban': int, 'name': str}  (1〜3着)
      'tan'            : int | None   単勝払戻
      'fuku'           : list[int]    複勝払戻（1〜3着分）
      'baren'          : int | None   馬連払戻
      'sanrenpuku'     : int | None   三連複払戻
      'error'          : str | None   エラーメッセージ
    """
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}&rf=race_submenu"
    result = {
        'race_id': race_id,
        'horses': [],
        'tan': None,
        'fuku': [],
        'baren': None,
        'sanrenpuku': None,
        'error': None,
    }

    try:
        resp = _SESSION.get(url, timeout=15)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        result['error'] = f"通信エラー: {e}"
        return result

    soup = BeautifulSoup(resp.content, 'html.parser')

    # ── 着順テーブル ──────────────────────────────────────────────────
    result_table = soup.find('table', class_='RaceTable01')
    if result_table:
        for tr in result_table.find_all('tr')[1:4]:  # 1〜3着
            tds = tr.find_all('td')
            if len(tds) < 4:
                continue
            try:
                chaku  = int(tds[0].get_text(strip=True))
                umaban = int(tds[2].get_text(strip=True))
                name   = tds[3].get_text(strip=True)
                result['horses'].append({'chaku': chaku, 'umaban': umaban, 'name': name})
            except (ValueError, IndexError):
                continue

    # ── 払戻テーブル ──────────────────────────────────────────────────
    pay_tables = soup.find_all('table', class_='PayTable01')
    for tbl in pay_tables:
        for tr in tbl.find_all('tr'):
            tds = tr.find_all('td')
            if len(tds) < 2:
                continue
            label = tr.find('th')
            label_text = label.get_text(strip=True) if label else ''

            def _parse_yen(td):
                txt = td.get_text(strip=True).replace(',', '').replace('円', '')
                try:
                    return int(txt)
                except ValueError:
                    return None

            if '単勝' in label_text:
                result['tan'] = _parse_yen(tds[-1])
            elif '複勝' in label_text:
                for td in tds:
                    v = _parse_yen(td)
                    if v and v > 0:
                        result['fuku'].append(v)
            elif '馬連' in label_text:
                result['baren'] = _parse_yen(tds[-1])
            elif '三連複' in label_text or '3連複' in label_text:
                result['sanrenpuku'] = _parse_yen(tds[-1])

    if not result['horses'] and result['tan'] is None:
        result['error'] = "結果データが取得できませんでした（レース未確定 or ページ構造変化）"

    return result
