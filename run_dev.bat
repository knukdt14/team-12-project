@echo off
REM backend(FastAPI, 포트 8000)를 새 창으로 띄운 뒤 frontend(Streamlit)를 실행합니다.
REM run_dev.sh의 Windows cmd/PowerShell용 버전.
REM 끝낼 땐 이 창에서 Ctrl+C 누르고, 따로 뜬 "CarDoc Backend" 창도 닫아주세요.

start "CarDoc Backend" cmd /k "call conda activate %CONDA_DEFAULT_ENV% && cd backend && uvicorn main:app --port 8000"

timeout /t 3 /nobreak >nul

set BACKEND_BASE_URL=http://127.0.0.1:8000
streamlit run frontend/app.py
