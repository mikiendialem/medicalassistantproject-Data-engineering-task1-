@echo off
setlocal
cd /d "%~dp0\.."
python model\train.py
endlocal