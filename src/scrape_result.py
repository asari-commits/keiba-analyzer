"""
Netkeibaからレース確定後の結果・払戻データを取得する。
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


def _parse_yen(text: str) -> int | None:
    """'1,230円' → 1230、変換できない場合は None"""
    txt = text.strip().replace(',', '').replace('円', '').replace('\xa0', '')
    try:
        v = int(txt)
        return v if v > 0 else None
    except ValueError:
        return None


def fetch_race_result(race_id: str) -> dict:
    """
    Netkeibaのレース結果ページから着順・払戻を取得する。

    Returns
    -------
    dict:
      race_id, horses (1〜3着), tan, fuku[], baren, sanrenpuku, error
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
        resp = _SESSION.get(url, timeout=20)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        result['error'] = f"通信エラー: {e}"
        return result

    soup = BeautifulSoup(resp.content, 'html.parser')

    # ── 着順テーブル ──────────────────────────────────────────────────────
    result_table = soup.find('table', class_='RaceTable01')
    if result_table:
        for tr in result_table.find_all('tr')[1:]:
            tds = tr.find_all('td')
            if len(tds) < 4:
                continue
            try:
                chaku = int(tds[0].get_text(strip=True))
                if chaku > 3:
                    break  # 4着以下は不要
                umaban = int(tds[2].get_text(strip=True))
                name   = tds[3].get_text(strip=True)
                result['horses'].append({'chaku': chaku, 'umaban': umaban, 'name': name})
            except (ValueError, IndexError):
                continue

    # ── 払戻テーブル ──────────────────────────────────────────────────────
    # Netkeibaの払戻テーブルは券種ごとにth(rowspan)があり、
    # 複勝は3行に分かれてtd=[馬番, 払戻]が続く。
    # current_type で行をまたいで券種を追跡する。
    pay_tables = soup.find_all('table', class_='PayTable01')
    current_type = ''

    for tbl in pay_tables:
        for tr in tbl.find_all('tr'):
            th = tr.find('th')
            tds = tr.find_all('td')

            if th:
                current_type = th.get_text(strip=True)

            if not tds:
                continue

            if '単勝' in current_type:
                # td[0]=馬番, td[1]=払戻
                if len(tds) >= 2:
                    result['tan'] = _parse_yen(tds[-1].get_text())

            elif '複勝' in current_type:
                # 1行に [馬番, 払戻] が入る（3行分ある）
                # 払戻は必ず100円以上なので馬番(1〜18)と区別できる
                if len(tds) >= 2:
                    v = _parse_yen(tds[-1].get_text())
                    if v and v >= 100:
                        result['fuku'].append(v)

            elif '馬連' in current_type:
                if len(tds) >= 2:
                    result['baren'] = _parse_yen(tds[-1].get_text())

            elif '三連複' in current_type or '3連複' in current_type:
                if len(tds) >= 2:
                    result['sanrenpuku'] = _parse_yen(tds[-1].get_text())

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
    progress_cb : (done: int, total: int, race_id: str) を受け取る進捗コールバック

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
            time.sleep(1.2)  # サーバー負荷軽減
    return results
