"""
Netkeibaからレース確定後の結果・払戻データを取得する。

払戻テーブルのHTML構造は2パターン存在する:
  [A] 旧ページ: <table class="PayTable01">  列=[馬番, 払戻(円), 人気]
  [B] 新ページ: <tr class="Tansho/Fukusho/Umaren/Fuku3"> の中の
               <td class="Payout"><span>XXX円YYY円</span></td>
               ※ 複勝3点が1つのspanにまとまって入る: "110円360円960円"

どちらにも対応するため両方を試みる。
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


def _split_payout(text: str) -> list[int]:
    """
    '110円360円960円' や '1,230円' から払戻金額のリストを返す。
    100円未満（馬番・人気などの数値）は除外する。
    """
    values = []
    for part in text.replace(',', '').replace('\xa0', '').split('円'):
        part = part.strip()
        if part:
            try:
                v = int(part)
                if v >= 100:
                    values.append(v)
            except ValueError:
                pass
    return values


def _payout_from_tr(soup, tr_class: str) -> list[int]:
    """
    <tr class="Tansho"> 等から Payout td の span テキストを取り出してパース。
    見つからない場合は空リストを返す。
    """
    tr = soup.find('tr', class_=tr_class)
    if not tr:
        return []
    td = tr.find('td', class_='Payout')
    if not td:
        return []
    span = td.find('span')
    if not span:
        # spanがなければtd直テキストを使う
        return _split_payout(td.get_text(strip=True))
    return _split_payout(span.get_text(strip=True))


def _payout_from_paytable(soup) -> dict:
    """
    旧ページ用: <table class="PayTable01"> から払戻を取得。
    列構造: [馬番(td[0])] [払戻(td[1])] [人気(td[2])]
    複勝は rowspan=3 なので current_type を引き継いで処理する。
    """
    out = {'tan': None, 'fuku': [], 'baren': None, 'sanrenpuku': None}
    pay_tables = soup.find_all('table', class_='PayTable01')
    if not pay_tables:
        return out

    current_type = ''
    for tbl in pay_tables:
        for tr in tbl.find_all('tr'):
            th  = tr.find('th')
            tds = tr.find_all('td')
            if th:
                current_type = th.get_text(strip=True)
            if len(tds) < 2:
                continue
            vals = _split_payout(tds[1].get_text(strip=True))
            if not vals:
                continue
            if '単勝' in current_type and out['tan'] is None:
                out['tan'] = vals[0]
            elif '複勝' in current_type and len(out['fuku']) < 3:
                out['fuku'].extend(v for v in vals if len(out['fuku']) < 3)
            elif '馬連' in current_type and out['baren'] is None:
                out['baren'] = vals[0]
            elif ('三連複' in current_type or '3連複' in current_type) \
                    and out['sanrenpuku'] is None:
                out['sanrenpuku'] = vals[0]
    return out


def fetch_race_result(race_id: str) -> dict:
    """
    Netkeibaのレース結果ページから着順・払戻を取得する。

    Returns: race_id, horses(1〜3着), tan, fuku[], baren, sanrenpuku, error
    """
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}&rf=race_submenu"
    result = {
        'race_id':    race_id,
        'horses':     [],
        'tan':        None,
        'fuku':       [],
        'baren':      None,
        'sanrenpuku': None,
        'error':      None,
    }

    try:
        resp = _SESSION.get(url, timeout=20)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        result['error'] = f"通信エラー: {e}"
        return result

    soup = BeautifulSoup(resp.content, 'html.parser')

    # ── 着順テーブル (RaceTable01) ──────────────────────────────────────
    # 列: 着順(0) | 枠(1) | 馬番(2) | 馬名(3) | ...
    result_table = soup.find('table', class_='RaceTable01')
    if result_table:
        for tr in result_table.find_all('tr')[1:4]:  # 先頭3行 = 1〜3着
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

    # ── 払戻: パターンB (新形式) ─ tr class="Tansho" 等 ───────────────
    # 先に試みる。単勝が取れればこちらで確定。
    tan_vals  = _payout_from_tr(soup, 'Tansho')
    fuku_vals = _payout_from_tr(soup, 'Fukusho')
    bar_vals  = _payout_from_tr(soup, 'Umaren')
    s3_vals   = _payout_from_tr(soup, 'Fuku3')

    if tan_vals:
        result['tan']        = tan_vals[0]
        result['fuku']       = fuku_vals[:3]
        result['baren']      = bar_vals[0] if bar_vals else None
        result['sanrenpuku'] = s3_vals[0]  if s3_vals  else None
    else:
        # ── 払戻: パターンA (旧形式) ─ PayTable01 ─────────────────────
        pay = _payout_from_paytable(soup)
        result['tan']        = pay['tan']
        result['fuku']       = pay['fuku']
        result['baren']      = pay['baren']
        result['sanrenpuku'] = pay['sanrenpuku']

    if not result['horses'] and result['tan'] is None:
        result['error'] = "結果未確定またはページ構造変化"

    return result


def fetch_all_results(race_ids: list[str],
                      progress_cb=None) -> dict[str, dict]:
    """複数レースの結果を並列一括取得する（ThreadPoolExecutor）。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = {}
    total = len(race_ids)
    completed = 0

    def _fetch_one(race_id):
        try:
            return race_id, fetch_race_result(race_id)
        except Exception as e:
            return race_id, {
                'race_id': race_id, 'horses': [],
                'tan': None, 'fuku': [], 'baren': None,
                'sanrenpuku': None, 'error': str(e),
            }

    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(_fetch_one, rid): rid for rid in race_ids}
        for fut in as_completed(futs):
            rid, res = fut.result()
            results[rid] = res
            completed += 1
            if progress_cb:
                progress_cb(completed, total, rid)

    return results
