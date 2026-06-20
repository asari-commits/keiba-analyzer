"""
JV-Link COM接続テスト。
実行前にJVLinkAgent.exeが起動している必要があります。
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import win32com.client

print("JV-Link COM接続テスト")
print("=" * 40)

try:
    jvlink = win32com.client.Dispatch("JVDTLab.JVLink")
    print("✓ COMオブジェクト生成OK")
except Exception as e:
    print(f"✗ COMオブジェクト生成失敗: {e}")
    print("\n→ JVLinkAgent.exe を起動してから再実行してください")
    print(f"  場所: C:\\Program Files (x86)\\JRA-VAN\\Data Lab\\JVLinkAgent.exe")
    sys.exit(1)

# バージョン確認
try:
    ver = jvlink.JVGetVersionInfo()
    print(f"✓ JV-Link バージョン: {ver}")
except Exception as e:
    print(f"  バージョン取得: {e}")

# ソフトID設定（ここにJRA-VANのソフトIDを入れる）
SOFTWARE_ID = "UNKNOWN"  # ← あとで設定

if SOFTWARE_ID != "UNKNOWN":
    ret = jvlink.JVInit(SOFTWARE_ID)
    print(f"JVInit 戻り値: {ret}  (0=正常, -1=エラー, -3=未登録)")
else:
    print("\n※ ソフトIDが未設定です。")
    print("  JRA-VANの管理画面またはTargetの設定画面でソフトIDを確認してください。")
    print("  確認できたら test_jvlink.py の SOFTWARE_ID を書き換えてください。")
