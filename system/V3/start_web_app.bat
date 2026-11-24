@echo off
rem BanglaLPR V3 web app: double-click, then open http://127.0.0.1:8499 in the browser.
cd /d "%~dp0"
set PY=D:\498R\_dataset_build\venv\Scripts\python.exe
if not exist "%PY%" set PY=python
start "" http://127.0.0.1:8499
"%PY%" app\server.py --port 8499
pause
