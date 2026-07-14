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

WORKSHEET_TITLE = "race_notes"
_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# モジュールレベルのキャッシュ（Streamlit の再実行をまたいで保持される）。
_state = {"client": None, "ws": None, "df": None, "ts": 0.0, "err": ""}


def _get_secret(key, default=None):
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    import os
    return os.environ.get(key, default)


def _load_sa_info() -> dict:
    """サービスアカウント情報を dict で返す。2通りの secrets 記法に対応:
    ① [gcp_service_account] テーブル形式（各項目を key = "value"）
    ② gcp_service_account_json = '''<鍵JSONの中身まるごと>'''（貼るだけで楽）"""
    import streamlit as st
    if "gcp_service_account" in st.secrets:
        return dict(st.secrets["gcp_service_account"])
    raw = st.secrets.get("gcp_service_account_json", "")
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
        return bool(has_sa and has_sheet)
    except Exception:
        return False


def _sheet_key(val: str) -> str:
    """URL でもキーでも受け取り、スプレッドシートキーを返す。"""
    s = str(val).strip()
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", s)
    return m.group(1) if m else s


def _get_ws():
    """ワークシートを取得（無ければ作成しヘッダを書く）。失敗時は例外。"""
    if _state["ws"] is not None:
        return _state["ws"]
    import streamlit as st
    import gspread
    from google.oauth2.service_account import Credentials
    from race_notes import _COLS

    sa_info = _load_sa_info()
    creds = Credentials.from_service_account_info(sa_info, scopes=_SCOPES)
    client = gspread.authorize(creds)
    sh = client.open_by_key(_sheet_key(st.secrets["race_notes_sheet"]))
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
    ws.clear()
    ws.update(values, value_input_option="RAW")
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


def delete_by_id(note_id) -> None:
    df = read_df(ttl=0)
    df = df[df["id"].astype(str) != str(note_id)].reset_index(drop=True)
    overwrite(df)
