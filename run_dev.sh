#!/usr/bin/env bash
# backend/(FastAPI: diagnose/estimate/repair-shops/chat, 포트 8000)를 백그라운드로 띄운 뒤
# frontend/app.py(Streamlit)를 실행.
# BACKEND_BASE_URL을 로컬 backend로 명시 — /diagnose, /chat은 이 주소로 감(YOLO/RAG는
# Render 무료 티어에 못 올라가서 로컬 전용). /estimate, /geocode, /repair-shops는
# frontend/utils/api_client.py 설계상 이 값과 무관하게 항상 Render로 가므로, 로컬
# backend를 켜고 있어도 카카오 API 키를 따로 준비할 필요가 없습니다.
# Ctrl+C로 종료하면 백그라운드 서버도 같이 정리됩니다.
set -e

(cd backend && python3 -m uvicorn main:app --port 8000) &
API_PID=$!

cleanup() {
  echo ""
  echo "backend 종료 중... (PID $API_PID)"
  kill "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT

sleep 2
BACKEND_BASE_URL=http://127.0.0.1:8000 python3 -m streamlit run frontend/app.py
