@echo off
REM keiba-analyzer local launcher (double-click to run)
REM Opens http://localhost:8501 in your browser. Faster than the cloud.
cd /d "%~dp0"
echo ============================================
echo  Starting keiba-analyzer (local)
echo  Browser opens automatically.
echo  To stop: press Ctrl+C in this window.
echo ============================================
python -m streamlit run src/app.py
pause
