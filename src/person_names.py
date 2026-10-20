"""
騎手・調教師名を master（TARGET形式）の表記にそろえる。

背景:
  騎手・調教師の過去成績（jockey_win_rate 等）は master の「騎手」「調教師」列を
  名前で集計している。TARGET の出馬表CSVは master と同じ表記（例: 松山弘平・杉山晴紀、
  4文字超は切り詰め: 大久保龍・Ｍ．デム）なのでそのまま一致するが、netkeiba の
  出馬表は略称（松山・杉山晴・手塚久・Mデムーロ）で、照合できず成績が0になっていた。
  （2026/10/20 菊花賞分析で判明。一致したのはルメール・武豊だけだった）

変換の順番:
  1. 上書き表（取り違えやすいもの。例: 堀→堀宣行。前方一致だと堀内岳志になる）
  2. 完全一致（TARGET経路はここで全員一致し、何も変わらない）
  3. 切り詰め一致（master は4文字で切れる: シュタルケ→シュタル、Mデムーロ→Ｍ．デム）
  4. 直近2年の master で「略称の文字が先頭から順に含まれる」名前を候補にし、
     調教師は所属（美浦/栗東）が合うものに絞って、前方一致→出走数の多い順で選ぶ
     （手塚久→手塚貴久 のように、略称が前方一致にならない場合もあるため）
  どれにも当たらない名前（騎手未定の ○○ など）は元のまま残す。
"""
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import pandas as pd

MASTER_PARQUET = Path(__file__).parent.parent / "data" / "processed" / "master.parquet"

# 候補を探す期間（master の最新日から遡る年数）。引退者・同名の古い候補を避ける。
RECENT_YEARS = 2

# netkeiba 略称 → master 表記。出走数の多さだけでは取り違えるものを登録する。
# 分析 2026/10/20: 堀は堀内岳志(551走)が堀宣行(405走)より多く、辻は辻野泰之が辻哲英より多い。
OVERRIDES = {
    '騎手': {},
    '調教師': {
        '堀': '堀宣行',
        '辻': '辻哲英',
        '手塚久': '手塚貴久',
        '和田郎': '和田正一',   # 和田正一郎（master は4文字で切れる）。和田勇介と取り違えやすい
    },
}

# netkeiba の所属ラベル → master の「所属」表記（調教師の絞り込み用）
AFFIL_MAP = {'美浦': '(美)', '栗東': '(栗)', '地方': '[地]', '海外': '[外]'}

_MARKS = re.compile(r'[\s\.．・▲△☆◇★◆]')


def _key(name) -> str:
    """照合用キー: 全角半角を統一し、空白・中黒・ピリオド・減量記号を除く"""
    return _MARKS.sub('', unicodedata.normalize('NFKC', str(name)))


def _is_subseq(abbr: str, full: str) -> bool:
    it = iter(full)
    return all(ch in it for ch in abbr)


@lru_cache(maxsize=2)
def _pools(master_path: str, mtime: float) -> dict:
    """master から {列: (全期間の キー→表記, 直近の 表記→出走数, 表記→主な所属)} を作る"""
    cols = ['日付_dt', '騎手', '調教師', '所属']
    m = pd.read_parquet(master_path, columns=cols)
    m['日付_dt'] = pd.to_datetime(m['日付_dt'], errors='coerce')
    recent = m[m['日付_dt'] >= m['日付_dt'].max() - pd.DateOffset(years=RECENT_YEARS)]
    out = {}
    for col in ['騎手', '調教師']:
        names = m[col].dropna().astype(str).str.strip()
        exact = {_key(n): n for n in names.unique() if n}
        counts = recent[col].dropna().astype(str).str.strip().value_counts()
        affil = (recent.dropna(subset=[col, '所属'])
                 .groupby(recent[col].astype(str).str.strip())['所属']
                 .agg(lambda s: s.value_counts().index[0]).to_dict())
        out[col] = (exact, counts, affil)
    return out


def _load_pools() -> dict | None:
    if not MASTER_PARQUET.exists():
        return None
    return _pools(str(MASTER_PARQUET), MASTER_PARQUET.stat().st_mtime)


def resolve_name(name, col: str, affil: str | None = None, pools: dict | None = None) -> str:
    """略称1件を master 表記に直す。直せなければ元の値を返す。col は '騎手' か '調教師'"""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return name
    raw = str(name).strip()
    k = _key(raw)
    if not k:
        return name
    pools = pools if pools is not None else _load_pools()
    if pools is None:
        return name
    exact, counts, affils = pools[col]

    if k in OVERRIDES.get(col, {}):
        return OVERRIDES[col][k]
    if k in exact:
        return exact[k]

    # master は4文字で切り詰められる（シュタルケ→シュタル、Ｍ．デムーロ→Ｍ．デム）
    trunc = [n for n in counts.index if len(n) == 4 and k.startswith(_key(n))]
    if trunc:
        return max(trunc, key=lambda n: counts[n])

    # 外国人騎手は master で頭文字が落ちることがある（Ｍヴェロン→ヴェロン、Ｒヘルナン→ヘルナン）
    if len(k) > 2 and k[0].isascii() and k[0].isalpha():
        r = resolve_name(k[1:], col, affil=affil, pools=pools)
        if _key(r) in exact:
            return r

    cands = [n for n in counts.index
             if (kn := _key(n)) and kn[0] == k[0] and len(kn) > len(k) and _is_subseq(k, kn)]
    if not cands:
        # 略称の末尾が master の切り詰め（4文字）より後ろにある場合（和田郎→和田正一[郎]）
        cands = [n for n in counts.index
                 if len(n) == 4 and (kn := _key(n))[0] == k[0] and _is_subseq(k[:2], kn)]
    if affil and col == '調教師':
        same = [n for n in cands if affils.get(n) == affil]
        cands = same or cands
    if not cands:
        return name
    # 前方一致を優先し、その中で直近の出走数が多い候補
    return max(cands, key=lambda n: (_key(n).startswith(k), counts[n]))


def normalize_person_names(df: pd.DataFrame) -> pd.DataFrame:
    """df の「騎手」「調教師」列を master 表記にそろえたコピーを返す。
    調教師は「所属」列（(美)/(栗) または 美浦/栗東）があれば候補の絞り込みに使う。"""
    pools = _load_pools()
    if pools is None or df.empty:
        return df
    df = df.copy()
    for col in ['騎手', '調教師']:
        if col not in df.columns:
            continue
        if col == '調教師' and '所属' in df.columns:
            aff = df['所属'].astype(str).str.strip().map(lambda a: AFFIL_MAP.get(a, a))
        else:
            aff = pd.Series([None] * len(df), index=df.index)
        cache = {}
        vals = []
        for v, a in zip(df[col], aff):
            ck = (v, a)
            if ck not in cache:
                cache[ck] = resolve_name(v, col, affil=a, pools=pools)
            vals.append(cache[ck])
        df[col] = vals
    return df
