"""馬ノート（回顧DB）の保存・管理。

映像・パドック等、データに見えない不利/見どころ/状態を、結果回顧でレース単位に
メモとして蓄積する独自データ層。次にその馬が出走する際、予測ヘッダにタグ表示する。

旧 watch_horses（次走狙い）を置き換える。ソース: 手動 / LLM。
"""
import re
from pathlib import Path
import pandas as pd

NOTES_PATH = Path(__file__).parent.parent / "data" / "processed" / "race_notes.parquet"

# タグ候補（カテゴリ別）。UIのグルーピングにも使う。＋自由メモで補足。
# 「不利」＝物理的な不利（接触・進路を塞がれた等）のみ。ペース・位置取りは「展開」で表す。
TAG_GROUPS = {
    '不利': ['出遅れ', '包まれる', '進路なし', '前が壁', '挟まれる', '接触',
             '外々を回された', '掛かる', '折り合い欠く'],
    '展開': ['展開不利(前残り)', '展開不利(差し届かず)', 'ハイペースで脚を使った',
             'スローで後方待機', '直線だけの競馬'],
    'ポジ': ['スムーズでこの内容', '上がり最速', '見どころ十分',
             '狙うべきメモ馬', '勝ちに等しい内容'],
    '状態': ['太め残り', '気配良好', 'パドック良', 'パドック悪', '発汗', 'チャカつき',
             '落馬', '競走中止', '除外', '落鉄'],
    # 次走の条件示唆（この馬を次に狙うべき条件）。観察ではなく人間の見立て。
    '次走': ['次走距離延長で狙い', '次走距離短縮で狙い', '次走内枠で狙い', '次走外枠で狙い',
             '次走小回りコースで狙い', '次走ワンターンで狙い', '次走直線長いコースで狙い',
             '次走広いコースで狙い', '次走平坦なコースで狙い'],
}
ALL_TAGS = [t for ts in TAG_GROUPS.values() for t in ts]

EVAL_OPTIONS = ['中立', '次走注目', '危険(過剰人気警戒)', '度外視']

_COLS = ['id', '馬名', '日付', '開催', 'Ｒ', 'レース名',
         '評価', '狙い度', 'タグ', 'メモ', 'ソース', '登録時刻']


def normalize_name(s) -> str:
    """master と同じ馬名正規化（先頭のマーカー/空白・末尾空白を除去）。"""
    s = str(s)
    s = re.sub(r'^[\s　$*▲△◎○☆★\.]+', '', s)
    s = re.sub(r'[\s　]+$', '', s)
    return s.strip()


def _read_raw() -> pd.DataFrame:
    if NOTES_PATH.exists():
        df = pd.read_parquet(NOTES_PATH)
        for c in _COLS:
            if c not in df.columns:
                df[c] = '' if c != '狙い度' else 2
        return df[_COLS]
    return pd.DataFrame(columns=_COLS)


def _write_raw(df: pd.DataFrame) -> None:
    """ローカル parquet ミラーへ保存（Sheets利用時もフォールバック用に常に書く）。"""
    NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[_COLS].to_parquet(NOTES_PATH, index=False)


def _sheets():
    """Googleスプレッドシートが利用可能ならそのモジュールを返す。未設定/失敗なら None。"""
    try:
        import sheets_store as _ss
        if _ss.available():
            return _ss
    except Exception:
        return None
    return None


def storage_status() -> dict:
    """保存先の状態を返す（UI表示用）。"""
    try:
        import sheets_store as _ss
        if _ss.configured():
            ok = _ss.available()
            return {'mode': 'sheets' if ok else 'sheets_error',
                    'ok': ok, 'error': _ss.last_error()}
    except Exception as e:
        return {'mode': 'local', 'ok': True, 'error': str(e)}
    return {'mode': 'local', 'ok': True, 'error': ''}


def load_notes() -> pd.DataFrame:
    ss = _sheets()
    if ss is not None:
        try:
            df = ss.read_df()
            _write_raw(df)   # ローカルミラーを最新化
            return df
        except Exception:
            pass
    return _read_raw()


def _tags_to_str(tags) -> str:
    if isinstance(tags, (list, tuple)):
        return '・'.join(str(t) for t in tags if str(t).strip())
    return str(tags or '')


_KEY_COLS = ['馬名', '日付', '開催', 'Ｒ']


def add_note(馬名, 日付, 評価='中立', 狙い度=2, タグ=None, メモ='',
             開催='', Ｒ='', レース名='', ソース='手動') -> None:
    """1レース×1頭のメモを保存（同一 馬名×日付×開催×R は上書き）。
    Sheets設定時はシートへ upsert、常にローカル parquet ミラーも更新する。"""
    try:
        d8 = pd.to_datetime(str(日付)).strftime('%Y%m%d')
    except Exception:
        d8 = str(日付)
    nm = normalize_name(馬名)
    ts = pd.Timestamp.now()
    row = {
        'id': ts.strftime('%Y%m%d%H%M%S%f'),
        '馬名': nm, '日付': d8, '開催': str(開催), 'Ｒ': str(Ｒ),
        'レース名': str(レース名 or ''),
        '評価': str(評価 or '中立'), '狙い度': int(狙い度),
        'タグ': _tags_to_str(タグ), 'メモ': str(メモ or ''),
        'ソース': str(ソース), '登録時刻': ts.strftime('%Y-%m-%d %H:%M'),
    }
    ss = _sheets()
    if ss is not None:
        try:
            ss.upsert(row, key_cols=_KEY_COLS)
        except Exception:
            pass
    # ローカル parquet ミラー（Sheets の有無に関わらず常に更新）
    df = _read_raw()
    key_mask = ((df['馬名'].map(normalize_name) == nm) &
                (df['日付'].astype(str) == d8) &
                (df['開催'].astype(str) == str(開催)) &
                (df['Ｒ'].astype(str) == str(Ｒ)))
    df = df[~key_mask]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _write_raw(df)


def delete_note(note_id) -> None:
    ss = _sheets()
    if ss is not None:
        try:
            ss.delete_by_id(note_id)
        except Exception:
            pass
    df = _read_raw()
    df = df[df['id'].astype(str) != str(note_id)].reset_index(drop=True)
    _write_raw(df)


def replace_all(df: pd.DataFrame) -> int:
    """CSV等からの一括置換（Sheets設定時はシートも置換）。返り値=件数。"""
    out = df.copy()
    for c in _COLS:
        if c not in out.columns:
            out[c] = '' if c != '狙い度' else 2
    out['馬名'] = out['馬名'].map(normalize_name)
    out['狙い度'] = pd.to_numeric(out['狙い度'], errors='coerce').fillna(2).astype(int)
    out = out[_COLS]
    ss = _sheets()
    if ss is not None:
        try:
            ss.overwrite(out)
        except Exception:
            pass
    _write_raw(out)
    return len(out)


def annotate_active(notes_df: pd.DataFrame, last_race_dates: dict) -> pd.DataFrame:
    """各メモに「有効/消化済み」を付与。
    last_race_dates: {正規化馬名: 最終出走日(datetime)}。
    メモ対象レース日より後にその馬が走っていれば「消化済み」（次走を迎えた）。"""
    df = notes_df.copy()
    if df.empty:
        df['状態'] = []
        return df

    def _state(r):
        nm = normalize_name(r['馬名'])
        lr = last_race_dates.get(nm)
        try:
            memo_d = pd.to_datetime(str(r['日付']), format='%Y%m%d')
        except Exception:
            return '有効'
        if lr is None or pd.isna(lr):
            return '有効'
        return '消化済み' if pd.to_datetime(lr) > memo_d else '有効'

    df['状態'] = df.apply(_state, axis=1)
    return df


def active_notes_for_horses(names, last_race_dates: dict) -> dict:
    """出走馬（names）のうち『有効』なメモを {正規化馬名: note_row(dict)} で返す。
    予測ヘッダのタグ表示用。同一馬に複数あれば最新日付を優先。"""
    df = load_notes()
    if df.empty:
        return {}
    want = {normalize_name(n) for n in names}
    df = df[df['馬名'].map(normalize_name).isin(want)]
    if df.empty:
        return {}
    df = annotate_active(df, last_race_dates)
    df = df[df['状態'] == '有効'].copy()
    if df.empty:
        return {}
    df = df.sort_values('日付')
    out = {}
    for _, r in df.iterrows():
        out[normalize_name(r['馬名'])] = r.to_dict()   # 後勝ち＝最新日付
    return out
