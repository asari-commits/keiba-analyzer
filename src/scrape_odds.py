"""
Netkeibaからリアルタイム単勝オッズを取得する。

レースID形式（12桁）: YYYY VV KK DD RR
  YYYY = 年 (e.g. 2026)
  VV   = 場コード 2桁 (函館=02, 東京=05 など)
  KK   = 開催回 2桁 (第1回=01)
  DD   = 開催日 2桁 (第1日=01)
  RR   = レース番号 2桁 (1R=01)

開催コード '1函1' → venue='函館', kai=1, day=1
→ race_id = '2026' + '02' + '01' + '01' + '01' → '202602010101'
"""
import re
import time
import json
import logging
from datetime import datetime

import requests
import pandas as pd

logger = logging.getLogger(__name__)

# 場名 → Netkeiba場コード（JRA開催場）
VENUE_CODE: dict[str, str] = {
    '札幌': '01', '函館': '02', '福島': '03', '新潟': '04',
    '東京': '05', '中山': '06', '中京': '07', '京都': '08',
    '阪神': '09', '小倉': '10',
}

# 場略称 → 場名（load_shutuba_target の VENUE_ABBR 逆引き）
ABBR_TO_VENUE: dict[str, str] = {
    '札': '札幌', '函': '函館', '福': '福島', '新': '新潟',
    '東': '東京', '中': '中山', '名': '中京', '京': '京都',
    '阪': '阪神', '小': '小倉',
}

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


_race_id_cache: dict[str, dict] = {}   # date_str → {(venue_abbr, r_num): race_id}


def get_race_ids_for_date(date_str: str) -> dict[tuple, str]:
    """
    指定日の全レースIDをNetkeibaから取得。
    戻り値: {(venue_abbr, r_num): race_id}
    例: {('函', 1): '202602010101', ('東', 3): '202605020803', ...}
    """
    if date_str in _race_id_cache:
        return _race_id_cache[date_str]

    url = (
        f"https://race.netkeiba.com/top/race_list_sub.html"
        f"?kaisai_date={date_str}"
    )
    result: dict[tuple, str] = {}

    try:
        resp = _SESSION.get(url, timeout=10)
        resp.raise_for_status()

        # race_id は href に埋め込まれている: ?race_id=202602010101
        ids_found = re.findall(r'race_id=(\d{12})', resp.text)
        for rid in dict.fromkeys(ids_found):   # 重複除去・順序保持
            vcode  = rid[4:6]
            r_num  = int(rid[10:12])
            # 逆引き: vcode → venue_name → abbr
            for venue_name, vc in VENUE_CODE.items():
                if vc == vcode:
                    abbr = next((a for a, v in ABBR_TO_VENUE.items() if v == venue_name), None)
                    if abbr:
                        result[(abbr, r_num)] = rid
                    break

    except requests.exceptions.RequestException as e:
        logger.warning(f"レースリスト取得失敗 ({date_str}): {e}")

    _race_id_cache[date_str] = result
    return result


def build_race_id(date_str: str, kaisai_code: str, r_num: int) -> str | None:
    """
    開催日・開催コード・R番号からNetkeiba用レースIDを返す。

    まず当日のレース一覧（Netkeiba）から正確なIDを取得し、
    取得できない場合は開催コードから推定する（精度は低い）。

    Parameters
    ----------
    date_str    : 'YYYYMMDD'
    kaisai_code : '1函1' 形式
    r_num       : レース番号 (int)
    """
    # 場略称を取得
    m = re.match(r'^(\d+)([^\d]+)(\d+)$', str(kaisai_code).strip())
    abbr = m.group(2) if m else None

    if abbr:
        # まずNetkeibaのレース一覧から正確なIDを取得
        id_map = get_race_ids_for_date(date_str)
        race_id = id_map.get((abbr, r_num))
        if race_id:
            return race_id

    # フォールバック: 開催コードから推定
    if m is None:
        return None
    year    = str(date_str)[:4]
    kai_num = int(m.group(1))
    day_num = int(m.group(3))
    venue   = ABBR_TO_VENUE.get(abbr)
    if venue is None:
        return None
    vcode = VENUE_CODE.get(venue)
    if vcode is None:
        return None

    logger.warning(f"レース一覧取得失敗。開催コードから推定: {date_str} {kaisai_code} {r_num}R")
    return f"{year}{vcode}{kai_num:02d}{day_num:02d}{r_num:02d}"


def fetch_odds_tan(race_id: str, retries: int = 2) -> pd.DataFrame:
    """
    Netkeibaの単勝オッズJSONを取得してDataFrameで返す。

    Returns
    -------
    DataFrame columns: 馬番(int), 単勝オッズ(float), 人気(int)
    空DFはエラー時（ログ参照）
    """
    url = (
        f"https://race.netkeiba.com/api/api_get_jra_odds.html"
        f"?race_id={race_id}&type=1&action=init"
    )

    for attempt in range(retries + 1):
        try:
            resp = _SESSION.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return _parse_tan_json(data)
        except requests.exceptions.HTTPError as e:
            logger.warning(f"HTTP {e.response.status_code} for {url}")
            break
        except requests.exceptions.RequestException as e:
            logger.warning(f"リクエストエラー (attempt {attempt+1}): {e}")
            if attempt < retries:
                time.sleep(1.5)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"JSONパースエラー: {e}")
            break

    # JSONが取れなければHTMLからスクレイピングを試みる
    return _scrape_odds_html(race_id)


def _parse_tan_json(data: dict) -> pd.DataFrame:
    """
    Netkeiba APIレスポンスJSONから単勝オッズを抽出。

    確認済みの形式:
      data['data']['odds']['1']['01'] = ['179.3', '0.0', '13']
        → [単勝オッズ, 複勝下限?, 人気]
      馬番は2桁ゼロ埋め文字列 ('01', '02', ...)
      取消馬: 人気='9999', オッズ<=0
    """
    rows = []

    try:
        tan_map = data['data']['odds']['1']   # 単勝 = type '1'
        for uma_str, val in tan_map.items():
            try:
                uma    = int(uma_str)
                odds_f = float(val[0])
                pop    = int(val[2]) if len(val) >= 3 else -1

                # 取消・除外馬をスキップ
                if pop >= 999 or odds_f <= 0:
                    continue
                rows.append({'馬番': uma, '単勝オッズ': odds_f, '人気': pop})
            except (ValueError, TypeError, IndexError):
                continue
        if rows:
            return _sort_by_popular(pd.DataFrame(rows))
    except (KeyError, TypeError):
        pass

    return pd.DataFrame(columns=['馬番', '単勝オッズ', '人気'])


def _scrape_odds_html(race_id: str) -> pd.DataFrame:
    """HTMLテーブルから単勝オッズをスクレイピング（JSONが取れない場合のフォールバック）"""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.error("beautifulsoup4 が未インストールです: pip install beautifulsoup4")
        return pd.DataFrame(columns=['馬番', '単勝オッズ', '人気'])

    url = f"https://race.netkeiba.com/odds/index.html?type=b1&race_id={race_id}"
    try:
        resp = _SESSION.get(url, timeout=12)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.warning(f"HTML取得エラー: {e}")
        return pd.DataFrame(columns=['馬番', '単勝オッズ', '人気'])

    soup = BeautifulSoup(resp.content, 'html.parser')

    rows = []
    # テーブル構造候補: #OddsTansyoBody > tr
    table = soup.find('table', id='OddsTansyoBody') or soup.find('table', id='odds_tan_table')
    if table is None:
        # クラスで探す
        for t in soup.find_all('table'):
            if t.get('class') and any('Odds' in c for c in t.get('class', [])):
                table = t
                break

    if table is None:
        logger.warning("オッズテーブルが見つかりませんでした")
        return pd.DataFrame(columns=['馬番', '単勝オッズ', '人気'])

    for tr in table.find_all('tr'):
        tds = tr.find_all('td')
        if len(tds) < 3:
            continue
        try:
            uma    = int(tds[0].get_text(strip=True))
            odds_f = float(tds[-1].get_text(strip=True).replace(',', ''))
            rows.append({'馬番': uma, '単勝オッズ': odds_f, '人気': -1})
        except (ValueError, IndexError):
            continue

    if not rows:
        return pd.DataFrame(columns=['馬番', '単勝オッズ', '人気'])

    df = pd.DataFrame(rows)
    # 人気が取れていない場合はオッズ昇順で付番
    if (df['人気'] == -1).all():
        df = df.sort_values('単勝オッズ').reset_index(drop=True)
        df['人気'] = range(1, len(df) + 1)
    return _sort_by_popular(df)


def _sort_by_popular(df: pd.DataFrame) -> pd.DataFrame:
    """人気昇順でソート"""
    if '人気' in df.columns and not (df['人気'] == -1).all():
        df = df.sort_values('人気').reset_index(drop=True)
    return df.reset_index(drop=True)


def apply_odds_to_pred(pred_df: pd.DataFrame, odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    pred_df に実オッズ・人気を付与し EV を再計算して返す。

    新規/更新列:
      単勝オッズ_live  : 取得した実オッズ (float)
      人気_live        : 取得した実人気順 (int)
      EV単勝_live      : 実オッズ使用の単勝期待値 (%)
    """
    from pred_utils import softmax_probs

    if odds_df.empty:
        return pred_df

    result = pred_df.copy()

    # 馬番でマージ
    odds_df = odds_df.copy()
    odds_df['馬番'] = pd.to_numeric(odds_df['馬番'], errors='coerce')
    result['馬番'] = pd.to_numeric(result['馬番'], errors='coerce')

    result = result.merge(
        odds_df[['馬番', '単勝オッズ', '人気']].rename(
            columns={'単勝オッズ': '単勝オッズ_live', '人気': '人気_live'}
        ),
        on='馬番',
        how='left',
    )

    # 実オッズを使ったEV計算（レースごとにsoftmaxで勝利確率を計算）
    race_key = (
        result['日付'].astype(str) +
        result['開催'].astype(str) +
        result['Ｒ'].astype(str)
    )
    ev_live = []
    for key, grp in result.groupby(race_key):
        probs = softmax_probs(grp['pred_score'])
        for idx, row in grp.iterrows():
            odds_val = row.get('単勝オッズ_live')
            prob = probs.loc[idx] if idx in probs.index else None
            if pd.notna(odds_val) and prob is not None and odds_val > 0:
                ev = prob * float(odds_val) * 100 - 100
                ev = float(max(ev, -100.0))
            else:
                ev = float('nan')
            ev_live.append((idx, ev))

    ev_series = pd.Series({i: v for i, v in ev_live}, name='EV単勝_live')
    result['EV単勝_live'] = ev_series

    return result


_race_info_cache: dict[str, dict] = {}


def fetch_race_info(race_id: str) -> dict:
    """
    Netkeiba shutuba.html から発走時刻・馬場状態・天候を取得。
    戻り値: {'time': '15:30', 'baba': '良', 'tenki': '晴'}  ← 取れない項目は空文字
    """
    if race_id in _race_info_cache:
        return _race_info_cache[race_id]

    result = {'time': '', 'baba': '', 'tenki': ''}
    try:
        from bs4 import BeautifulSoup
        url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
        resp = _SESSION.get(url, timeout=8)
        resp.encoding = 'euc-jp'
        soup = BeautifulSoup(resp.text, 'html.parser')
        tag = soup.find(class_='RaceData01')
        if tag:
            text = tag.get_text(' ', strip=True)
            m_time  = re.search(r'(\d{1,2}:\d{2})', text)
            m_baba  = re.search(r'馬場[：:]\s*([良稍重不])', text)
            m_tenki = re.search(r'天候[：:]\s*(\S+?)(?:\s|/|$)', text)
            if m_time:  result['time']  = m_time.group(1)
            if m_baba:  result['baba']  = m_baba.group(1)
            if m_tenki: result['tenki'] = m_tenki.group(1)
    except Exception:
        pass

    _race_info_cache[race_id] = result
    return result


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    # テスト: 2026/06/20 函館 1R
    date   = '20260620'
    kaisai = '1函1'
    r      = 1
    rid = build_race_id(date, kaisai, r)
    print(f"race_id: {rid}")

    if rid:
        df = fetch_odds_tan(rid)
        print(df.to_string(index=False))
