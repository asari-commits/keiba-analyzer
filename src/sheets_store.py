# -*- coding: utf-8 -*-
"""馬ノートの永続化バックエンド（Googleスプレッドシート）。

Streamlit Cloud はディスクが再デプロイで消えるため、馬ノートを Google スプレッドシート
に置いて永続化する。ユーザーはそのシートを直接開いて一括追加・編集・削除でき、
アプリはそれをそのまま読み書きする（＝スプレッドシート一括更新も同時に実現）。

必要な secrets（Streamlit Cloud の Secrets / ローカルは .streamlit/secrets.toml）:
    [gcp_service_account]      ← サービスアカウントの鍵JSONの中身をそのまま
    type = "service_account"
    project_id = "..."
    private_key = "-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n"
    client_email = "xxx@xxx.iam.gserviceaccount.com"
    ...
    race_notes_sheet = "https://docs.google.com/spreadsheets/d/XXXX/edit"  # またはキーだけ

未設定なら available()=False（＝従来どおりローカル parquet にフォールバック）。
"""
import re
import time

import pandas as pd

# Avast 等の HTTPS 検査ソフトが入った PC では、通信に検査ソフトの証明書が差し込まれる。
# requests は certifi の証明書リストを使うためそれを信頼できず、Google への接続が
# CERTIFICATE_VERIFY_FAILED で失敗する。OS の証明書ストア（ブラウザと同じ基準）を使わせる。
# truststore が無い環境（Streamlit Cloud 等）では何もしない。
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

WORKSHEET_TITLE = "race_notes"
_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# モジュールレベルのキャッシュ（Streamlit の再実行をまたいで保持される）。
_state = {"client": None, "ws": None, "df": None, "ts": 0.0, "err": ""}


_file_secrets = {"loaded": False, "data": {}}


def _secrets_from_file() -> dict:
    """このリポジトリ直下の .streamlit/secrets.toml を直接読む。

    st.secrets は実行時のカレントディレクトリ基準でしか secrets.toml を探さないため、
    keiba-review や keiba-bias-tracker から呼ぶと設定が見つからず、
    気づかないままローカル保存に落ちてしまう。それを防ぐための経路。
    """
    if _file_secrets["loaded"]:
        return _file_secrets["data"]
    _file_secrets["loaded"] = True
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, ".streamlit", "secrets.toml")
    if os.path.exists(path):
        try:
            import tomllib
            with open(path, "rb") as f:
                _file_secrets["data"] = tomllib.load(f)
        except Exception:
            try:
                import toml
                with open(path, encoding="utf-8") as f:
                    _file_secrets["data"] = toml.load(f)
            except Exception:
                pass
    return _file_secrets["data"]


def _get_secret(key, default=None):
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    v = _secrets_from_file().get(key)
    if v not in (None, ""):
        return v
    import os
    return os.environ.get(key, default)


def _load_sa_info() -> dict:
    """サービスアカウント情報を dict で返す。2通りの secrets 記法に対応:
    ① [gcp_service_account] テーブル形式（各項目を key = "value"）
    ② gcp_service_account_json = '''<鍵JSONの中身まるごと>'''（貼るだけで楽）"""
    try:
        import streamlit as st
        if "gcp_service_account" in st.secrets:
            return dict(st.secrets["gcp_service_account"])
        raw = st.secrets.get("gcp_service_account_json", "")
        if raw:
            import json
            return json.loads(raw)
    except Exception:
        pass
    fs = _secrets_from_file()
    if "gcp_service_account" in fs:
        return dict(fs["gcp_service_account"])
    raw = fs.get("gcp_service_account_json", "")
    if raw:
        import json
        return json.loads(raw)
    raise KeyError("gcp_service_account / gcp_service_account_json が secrets にありません。")


def configured() -> bool:
    """secrets に接続情報が入っているか（実接続はまだ試さない）。"""
    try:
        import streamlit as st
        has_sa = ("gcp_service_account" in st.secrets) or bool(st.secrets.get("gcp_service_account_json", ""))
        has_sheet = bool(st.secrets.get("race_notes_sheet", ""))
        if has_sa and has_sheet:
            return True
    except Exception:
        pass
    fs = _secrets_from_file()
    has_sa = ("gcp_service_account" in fs) or bool(fs.get("gcp_service_account_json", ""))
    return bool(has_sa and fs.get("race_notes_sheet", ""))


def _sheet_key(val: str) -> str:
    """URL でもキーでも受け取り、スプレッドシートキーを返す。"""
    s = str(val).strip()
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", s)
    return m.group(1) if m else s


def _get_ws():
    """ワークシートを取得（無ければ作成しヘッダを書く）。失敗時は例外。"""
    if _state["ws"] is not None:
        return _state["ws"]
    import gspread
    from google.oauth2.service_account import Credentials
    from race_notes import _COLS

    sa_info = _load_sa_info()
    creds = Credentials.from_service_account_info(sa_info, scopes=_SCOPES)
    client = gspread.authorize(creds)
    # st.secrets を直に引くと、カレントディレクトリが違うだけで落ちる。_get_secret 経由にする。
    sheet_ref = _get_secret("race_notes_sheet")
    if not sheet_ref:
        raise KeyError("race_notes_sheet が secrets にありません。")
    sh = client.open_by_key(_sheet_key(sheet_ref))
    try:
        ws = sh.worksheet(WORKSHEET_TITLE)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=WORKSHEET_TITLE, rows=1000, cols=len(_COLS))
        ws.update([_COLS], value_input_option="RAW")
    # ヘッダが空なら書く
    head = ws.row_values(1)
    if not head:
        ws.update([_COLS], value_input_option="RAW")
    _state["client"], _state["ws"] = client, ws
    return ws


def available() -> bool:
    """接続情報があり、実際にワークシートへ到達できるか。"""
    if not configured():
        return False
    try:
        _get_ws()
        _state["err"] = ""
        return True
    except Exception as e:
        _state["err"] = str(e)
        return False


def last_error() -> str:
    return _state.get("err", "")


def _invalidate():
    _state["df"] = None
    _state["ts"] = 0.0


def read_df(ttl: float = 30.0) -> pd.DataFrame:
    """シート全行を DataFrame で返す（_COLS 準拠）。ttl 秒はキャッシュ。"""
    from race_notes import _COLS
    now = time.time()
    if _state["df"] is not None and (now - _state["ts"]) < ttl:
        return _state["df"].copy()
    ws = _get_ws()
    records = ws.get_all_records()  # 1行目をヘッダとした dict のリスト
    df = pd.DataFrame(records)
    for c in _COLS:
        if c not in df.columns:
            df[c] = "" if c != "狙い度" else 2
    # 型を安定させる（日付/idは文字列、狙い度は数値）
    for c in _COLS:
        if c == "狙い度":
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(2).astype(int)
        else:
            df[c] = df[c].astype(str).replace({"nan": "", "None": ""})
    df = df[_COLS]
    _state["df"], _state["ts"] = df, now
    return df.copy()



def _col_letter(n: int) -> str:
    """1 -> A, 12 -> L。列数からA1記法の列名を作る。"""
    out = ""
    while n:
        n, r = divmod(n - 1, 26)
        out = chr(65 + r) + out
    return out


def _with_retry(fn, tries: int = 5):
    """Google API のクォータ超過（429）を指数バックオフで待つ。

    シート全体の書き換えを馬1頭ごとに呼ぶと、すぐ 1分あたりの書き込み上限に当たる。
    ここで黙って失敗するとデータが消えるので、必ず待って通す。
    """
    import time as _t
    delay = 2.0
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            retryable = ("429" in msg or "Quota" in msg or "quota" in msg
                         or "500" in msg or "503" in msg)
            if not retryable or i == tries - 1:
                raise
            _t.sleep(delay)
            delay *= 2


def overwrite(df: pd.DataFrame) -> None:
    """シート全体を df で置き換える（ヘッダ＋全行）。一括取込・削除に使う。"""
    from race_notes import _COLS
    ws = _get_ws()
    out = df.copy()
    for c in _COLS:
        if c not in out.columns:
            out[c] = "" if c != "狙い度" else 2
    out = out[_COLS].astype(object).where(pd.notna(out[_COLS]), "")
    values = [_COLS] + out.values.tolist()

    # clear() してから update() すると、update が失敗した瞬間にシートが空のまま残る。
    # 呼び出し側が例外を握りつぶす作りなので、気づかないまま全消えになり得る
    # （実際に 314件 → 1件 まで失われた）。
    # 先に本体を書き、成功してから余った行だけを消す順番にする。
    need = len(values)
    if ws.row_count < need:
        ws.add_rows(need - ws.row_count)
    _with_retry(lambda: ws.update(values, value_input_option="RAW"))
    if ws.row_count > need:
        last_col = _col_letter(len(_COLS))
        _with_retry(lambda: ws.batch_clear([f"A{need + 1}:{last_col}{ws.row_count}"]))
    _invalidate()


def upsert(row: dict, key_cols) -> None:
    """key_cols が一致する行を置換、無ければ追記。"""
    df = read_df(ttl=0)
    mask = pd.Series(True, index=df.index)
    for k in key_cols:
        mask &= (df[k].astype(str) == str(row.get(k, "")))
    df = df[~mask]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    overwrite(df)


def _keys_of(df, key_cols):
    """キー列を1本の文字列にまとめる。区切りは通常データに出ない文字を使う。"""
    import pandas as _pd
    if df.empty:
        return _pd.Series([], dtype=str)
    return df[list(key_cols)].astype(str).agg("\x00".join, axis=1)


def upsert_many(rows, key_cols) -> int:
    """複数行をまとめて upsert する。シートへの書き込みは最後の1回だけ。

    1頭ずつ upsert() を呼ぶとシート全体の読み書きが頭数ぶん走り、
    Google の書き込みクォータに当たる。一括プッシュは必ずこちらを使う。
    """
    from race_notes import _COLS
    rows = list(rows)
    if not rows:
        return 0
    cur = read_df(ttl=0)
    new = pd.DataFrame(rows)
    for c in _COLS:
        if c not in new.columns:
            new[c] = "" if c != "狙い度" else 2
    new = new[_COLS]
    # 同じバッチ内に同じキーが複数あれば、後に来たものを採用する
    new = new.drop_duplicates(subset=list(key_cols), keep="last")
    keep = cur[~_keys_of(cur, key_cols).isin(set(_keys_of(new, key_cols)))]
    overwrite(pd.concat([keep, new], ignore_index=True))
    return len(new)


def delete_by_id(note_id) -> None:
    df = read_df(ttl=0)
    df = df[df["id"].astype(str) != str(note_id)].reset_index(drop=True)
    overwrite(df)
