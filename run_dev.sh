#!/usr/bin/env bash
# backend/(FastAPI: diagnose/estimate/repair-shops/chat, 포트 8000)를 백그라운드로 띄운 뒤
# frontend/app.py(Streamlit)를 실행.
# BACKEND_BASE_URL을 로컬 backend로 명시해서, .env의 ESTIMATE_API_BASE_URL(Render 배포
# 주소)로 fallback되지 않고 방금 띄운 로컬 backend를 쓰도록 합니다.
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
