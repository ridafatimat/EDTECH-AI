@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
python -m pip install -r frontend\requirements.txt
python -m streamlit run frontend\app.py
endlocal
