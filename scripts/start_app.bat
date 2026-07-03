@echo off
setlocal
cd /d "%~dp0\.."
start "Symptom Backend" cmd /k python -m uvicorn backend.main:app --port 8420
start "" http://localhost:8420/
endlocal