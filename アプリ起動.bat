@echo off
REM keiba-analyzer ローカル起動ランチャー（ダブルクリックで実行）
REM ブラウザで http://localhost:8501 が開き、クラウドより高速に動きます。
cd /d "%~dp0"
echo ============================================
echo  競馬予測アプリ をローカル起動します
echo  ブラウザが自動で開きます（閉じるにはこの黒い画面でCtrl+C）
echo ============================================
python -m streamlit run src/app.py
pause
