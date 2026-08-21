@echo off
REM 週次ワンコマンド更新（ダブルクリック用）
REM 引数はそのまま weekly_update.py に渡る。
REM   例) weekly_update.bat 20260815-0816
REM   例) weekly_update.bat --lap
chcp 65001 >nul
cd /d "%~dp0"
python weekly_update.py %*
echo.
echo ----------------------------------------
echo 終了しました。ウィンドウを閉じてください。
pause
