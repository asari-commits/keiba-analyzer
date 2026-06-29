"""次走狙い（管理者メモ馬）の保存・管理。

映像分析等でデータに見えない不利・ロスがあった馬を「次走狙い」として登録し、
次に出走する際の予測表示にタグを出すための軽量ストア。
"""
import re
from pathlib import Path
import pandas as pd

WATCH_PATH = Path(__file__).parent.parent / "data" / "processed" / "watch_horses.parquet"

# フォームの選択肢（情報統一のため定型化。＋自由記述メモで補足）
REASON_OPTIONS = [
    '出遅れ', '展開不利', '枠順不利', '距離延長狙い', '距離短縮狙い',
    '直線進路なし', '道中不利', 'かかり', '発汗', '競争中止', 'その他不利',
]

_COLS = ['id', '馬名', 'メモ日', '理由', '狙い度', 'メモ', 'ソース', '登録時刻']


def normalize_name(s) -> str:
    """master と同じ馬名正規化（先頭の空白/マーカー、末尾空白を除去）。"""
    s = str(s)
    s = re.sub(r'^[\s　$*▲△◎○☆★\.]+', '', s)
    s = re.sub(r'[\s　]+$', '', s)
    return s.strip()


def load_watch() -> pd.DataFrame:
    if WATCH_PATH.exists():
        df = pd.read_parquet(WATCH_PATH)
        for c in _COLS:
            if c not in df.columns:
                df[c] = '' if c != '狙い度' else 2
        return df[_COLS]
    return pd.DataFrame(columns=_COLS)


def add_watch(馬名, 理由, 狙い度, メモ, メモ日, ソース='手動') -> None:
    df = load_watch()
    reasons = '・'.join(理由) if isinstance(理由, (list, tuple)) else str(理由 or '')
    ts = pd.Timestamp.now()
    row = {
        'id': ts.strftime('%Y%m%d%H%M%S%f'),
        '馬名': normalize_name(馬名),
        'メモ日': pd.to_datetime(メモ日).strftime('%Y%m%d'),
        '理由': reasons,
        '狙い度': int(狙い度),
        'メモ': str(メモ or ''),
        'ソース': ソース,
        '登録時刻': ts.strftime('%Y-%m-%d %H:%M'),
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    WATCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(WATCH_PATH, index=False)


def auto_register_candidates(candidates, メモ日) -> int:
    """脚余し候補を「次走期待(自動)」として一括登録（同一馬・同一メモ日の自動分は重複登録しない）。
    追加した頭数を返す。"""
    md = pd.to_datetime(メモ日).strftime('%Y%m%d')
    df = load_watch()
    existing = set()
    if not df.empty:
        existing = set(zip(df['馬名'].map(normalize_name),
                           df['メモ日'].astype(str), df['ソース'].astype(str)))
    added = 0
    for nm in candidates:
        key = (normalize_name(nm), md, '脚余し自動')
        if key in existing:
            continue
        add_watch(nm, '次走期待(自動)', 1, '', メモ日, ソース='脚余し自動')
        existing.add(key)
        added += 1
    return added


def register_candidates_bulk(pairs) -> int:
    """pairs: iterable of (馬名, メモ日)。脚余し候補を「次走期待(自動)」で一括登録（重複スキップ）。
    1回のファイル書き込みでまとめて保存。追加頭数を返す。"""
    df = load_watch()
    existing = set()
    if not df.empty:
        existing = set(zip(df['馬名'].map(normalize_name),
                           df['メモ日'].astype(str), df['ソース'].astype(str)))
    rows = []
    base = pd.Timestamp.now()
    for nm, md in pairs:
        try:
            md8 = pd.to_datetime(md).strftime('%Y%m%d')
        except Exception:
            continue
        key = (normalize_name(nm), md8, '脚余し自動')
        if key in existing:
            continue
        existing.add(key)
        rows.append({
            'id': base.strftime('%Y%m%d%H%M%S%f') + str(len(rows)),
            '馬名': normalize_name(nm), 'メモ日': md8, '理由': '次走期待(自動)',
            '狙い度': 1, 'メモ': '', 'ソース': '脚余し自動',
            '登録時刻': base.strftime('%Y-%m-%d %H:%M'),
        })
    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        WATCH_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(WATCH_PATH, index=False)
    return len(rows)


def delete_watch(watch_id) -> None:
    df = load_watch()
    df = df[df['id'].astype(str) != str(watch_id)].reset_index(drop=True)
    WATCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(WATCH_PATH, index=False)


def annotate_active(watch_df: pd.DataFrame, last_race_dates: dict) -> pd.DataFrame:
    """各メモに「有効/消化済み」を付与して返す。
    last_race_dates: {正規化馬名: 最終出走日(datetime)}。
    馬がメモ日以降に走っていれば「消化済み」（次走を迎えた）、それ以外は「有効」。"""
    df = watch_df.copy()
    if df.empty:
        df['状態'] = []
        return df

    def _state(r):
        nm = normalize_name(r['馬名'])
        lr = last_race_dates.get(nm)
        if lr is None or pd.isna(lr):
            return '有効'   # master履歴なし→未消化扱い
        try:
            memo_d = pd.to_datetime(str(r['メモ日']), format='%Y%m%d')
        except Exception:
            return '有効'
        return '消化済み' if pd.to_datetime(lr) > memo_d else '有効'

    df['状態'] = df.apply(_state, axis=1)
    return df
