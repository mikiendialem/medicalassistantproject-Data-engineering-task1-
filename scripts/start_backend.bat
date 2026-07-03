@echo off
setlocal
cd /d "%~dp0\.."
python -m uvicorn backend.main:app --port 8420
endlocal