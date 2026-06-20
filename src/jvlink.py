"""
JV-Link Python連携モジュール（64bitメインプロセス側）。
32bitブリッジ経由でJV-Linkを呼び出す。
"""
import json
import subprocess
from pathlib import Path

BRIDGE_DIR  = Path(__file__).parent.parent / "jvlink_bridge"
PY32_EXE    = BRIDGE_DIR / "python32" / "python.exe"
BRIDGE_PY   = BRIDGE_DIR / "bridge.py"
SOFT_ID_FILE = BRIDGE_DIR / "software_id.txt"


def _call(args: list) -> dict:
    """32bitブリッジを呼び出してJSONを受け取る"""
    if not PY32_EXE.exists():
        raise RuntimeError(
            "32bit Pythonが未セットアップです。\n"
            "python setup_jvlink.py を実行してください。"
        )
    result = subprocess.run(
        [str(PY32_EXE), str(BRIDGE_PY)] + args,
        capture_output=True, text=True, encoding='utf-8'
    )
    if result.returncode != 0 and not result.stdout:
        raise RuntimeError(f"ブリッジエラー: {result.stderr}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"JSON解析エラー: {result.stdout[:200]}")


def setup(software_id: str) -> dict:
    """ソフトIDを登録してJV-Linkを初期化"""
    return _call(["init", software_id])


def ping() -> dict:
    """接続確認"""
    return _call(["ping"])


def get_today_races() -> dict:
    """本日の開催レース情報を取得"""
    return _call(["get_today_races"])


def is_configured() -> bool:
    return SOFT_ID_FILE.exists() and PY32_EXE.exists()
