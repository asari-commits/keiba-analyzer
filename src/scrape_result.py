"""
Netkeibaからレース確定後の結果・払戻データを取得する。

PayTable01の列構造: [馬番/組合せ] [払戻(円)] [人気]
→ 払戻は必ず tds[1] を取る（tds[-1] は人気列で誤り）
"""
import time
import logging
import requests
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


def _parse_yen(td) -> int | None:
    """tdタグから払戻金額を取得。'1,230円' → 1230"""
    txt = td.get_text(strip=True).replace(',', '').replace('円', '').replace('\xa0', '').strip()
    try:
        v = int(txt)
        return v if v >= 100 else None  # 100円未満は馬番・人気等なので除外
    except ValueError:
        return None


def fetch_race_result(race_id: str) -> dict:
    """
    Netkeibaのレース結果ページから着順・払戻を取得する。

    PayTable01 列構造: 馬番(td[0]) | 払戻(td[1]) | 人気(td[2])
    複勝は rowspan=3 で3行に分かれる → current_type で追跡

    Returns
    -------
    dict: race_id, horses(1〜3着), tan, fuku[], baren, sanrenpuku, error
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

    # リトライ付きHTTP取得
    for attempt in range(3):
        try:
            resp = _SESSION.get(url, timeout=20)
            resp.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            if attempt == 2:
                result['error'] = f"通信エラー: {e}"
                return result
            time.sleep(2)

    soup = BeautifulSoup(resp.content, 'html.parser')

    # ── 着順テーブル ──────────────────────────────────────────────────────
    # 列: 着順(0) | 枠(1) | 馬番(2) | 馬名(3) | 性齢(4) | ...
    result_table = soup.find('table', class_='RaceTable01')
    if result_table:
        for tr in result_table.find_all('tr')[1:]:
            tds = tr.find_all('td')
            if len(tds) < 4:
                continue
            try:
                chaku = int(tds[0].get_text(strip=True))
            except ValueError:
                continue  # 除外・取消行をスキップ
            if chaku > 3:
                break
            try:
                umaban = int(tds[2].get_text(strip=True))
                name   = tds[3].get_text(strip=True)
                result['horses'].append({'chaku': chaku, 'umaban': umaban, 'name': name})
            except (ValueError, IndexError):
                continue

    # ── 払戻テーブル ──────────────────────────────────────────────────────
    # 列構造: [馬番/組合せ(td[0])] [払戻金額(td[1])] [人気(td[2])]
    # 複勝は th rowspan=3 で3行に分かれる → current_type で行をまたいで追跡
    pay_tables = soup.find_all('table', class_='PayTable01')
    current_type = ''

    for tbl in pay_tables:
        for tr in tbl.find_all('tr'):
            th  = tr.find('th')
            tds = tr.find_all('td')

            if th:
                current_type = th.get_text(strip=True)

            # 払戻列は td[1]（td[0]=馬番/組合せ、td[2]=人気）
            if len(tds) < 2:
                continue
            pay_td = tds[1]

            if '単勝' in current_type:
                result['tan'] = _parse_yen(pay_td)

            elif '複勝' in current_type:
                v = _parse_yen(pay_td)
                if v and len(result['fuku']) < 3:
                    result['fuku'].append(v)

            elif '馬連' in current_type:
                result['baren'] = _parse_yen(pay_td)

            elif '三連複' in current_type or '3連複' in current_type:
                result['sanrenpuku'] = _parse_yen(pay_td)

    if not result['horses'] and result['tan'] is None:
        result['error'] = "結果未確定またはページ構造変化"

    return result


def fetch_all_results(race_ids: list[str],
                      progress_cb=None) -> dict[str, dict]:
    """
    複数レースの結果を一括取得する。

    Parameters
    ----------
    race_ids    : race_id のリスト
    progress_cb : (done: int, total: int, race_id: str) → None

    Returns
    -------
    {race_id: fetch_race_result の戻り値}
    """
    results = {}
    total = len(race_ids)
    for i, race_id in enumerate(race_ids):
        try:
            results[race_id] = fetch_race_result(race_id)
        except Exception as e:
            results[race_id] = {
                'race_id': race_id, 'horses': [],
                'tan': None, 'fuku': [], 'baren': None,
                'sanrenpuku': None, 'error': str(e),
            }
        if progress_cb:
            progress_cb(i + 1, total, race_id)
        if i < total - 1:
            time.sleep(1.2)
    return results
