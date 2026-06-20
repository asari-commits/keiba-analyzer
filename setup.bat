@echo off
echo === keiba-analyzer セットアップ ===
echo.

echo [1/2] パッケージインストール中...
python -m pip install -r requirements.txt

echo.
echo [2/2] 完了！
echo.
echo 起動方法:
echo   Web UI:    python -m streamlit run src/app.py
echo   CLI分析:   python src/pipeline.py ^<race_id^>
echo              例: python src/pipeline.py 202606050811
echo.
pause
