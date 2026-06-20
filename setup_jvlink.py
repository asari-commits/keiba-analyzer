"""
JV-Link用 32bit Pythonブリッジのセットアップ。
実行すると:
  1. Python 3.11 32bit embeddable をダウンロード
  2. pipをセットアップ
  3. pywin32をインストール
  4. ブリッジスクリプトを配置
"""
import sys, zipfile, subprocess, urllib.request
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

BRIDGE_DIR = Path(r"C:\Users\asari\Downloads\Claude\keiba-analyzer\jvlink_bridge")
PY32_DIR   = BRIDGE_DIR / "python32"
PY32_EXE   = PY32_DIR / "python.exe"
PY32_ZIP_URL = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-win32.zip"
GET_PIP_URL  = "https://bootstrap.pypa.io/get-pip.py"

def step(msg):
    print(f"\n{'='*50}\n{msg}\n{'='*50}")

step("1/4: ディレクトリ作成")
BRIDGE_DIR.mkdir(parents=True, exist_ok=True)
PY32_DIR.mkdir(exist_ok=True)
print(f"  {BRIDGE_DIR}")

step("2/4: Python 3.11 32bit embeddable をダウンロード")
zip_path = BRIDGE_DIR / "python32.zip"
if not PY32_EXE.exists():
    print(f"  ダウンロード中: {PY32_ZIP_URL}")
    urllib.request.urlretrieve(PY32_ZIP_URL, zip_path)
    print("  展開中...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(PY32_DIR)
    zip_path.unlink()
    print(f"  → {PY32_EXE}")
else:
    print("  既にインストール済み")

# python311._pth の site-packages を有効化（pip使用に必要）
pth_file = next(PY32_DIR.glob("python3*._pth"), None)
if pth_file:
    content = pth_file.read_text()
    if "#import site" in content:
        pth_file.write_text(content.replace("#import site", "import site"))
        print("  site-packages 有効化完了")

step("3/4: pip インストール")
getpip_path = BRIDGE_DIR / "get-pip.py"
if not (PY32_DIR / "Scripts" / "pip.exe").exists():
    print("  get-pip.py ダウンロード中...")
    urllib.request.urlretrieve(GET_PIP_URL, getpip_path)
    subprocess.run([str(PY32_EXE), str(getpip_path)], check=True)
    getpip_path.unlink()
    print("  pip インストール完了")
else:
    print("  既にインストール済み")

step("4/4: pywin32 インストール")
pip_exe = PY32_DIR / "Scripts" / "pip.exe"
result = subprocess.run(
    [str(pip_exe), "install", "pywin32"],
    capture_output=True, text=True
)
print(result.stdout[-500:] if result.stdout else "")
if result.returncode != 0:
    print("エラー:", result.stderr[-300:])
else:
    print("  pywin32 インストール完了")

print("\n✓ セットアップ完了")
print(f"  32bit Python: {PY32_EXE}")
