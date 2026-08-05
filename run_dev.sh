#!/usr/bin/env bash
# estimate_api.py(FastAPI, 포트 8000)를 백그라운드로 띄운 뒤 app.py(Streamlit)를 실행.
# Ctrl+C로 종료하면 백그라운드 서버도 같이 정리됩니다.
set -e

python3 -m uvicorn estimate_api:app --port 8000 &
API_PID=$!

cleanup() {
  echo ""
  echo "estimate_api.py 종료 중... (PID $API_PID)"
  kill "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT

sleep 2
python3 -m streamlit run app.py
