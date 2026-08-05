"""가벼운 배포용 진입점 (Render 무료 티어 등, RAM 제한 환경).

main.py 전체(YOLO 진단 + RAG 상담 포함)는 torch/langchain 등 무거운 의존성 때문에
512MB급 무료 호스팅에서 메모리 부족으로 죽을 가능성이 높음.
이 파일은 무거운 diagnose/chat 라우터를 빼고, 견적(estimate)과 카카오맵
(repair_shops)만 남긴 버전 — estimate_api.py(루트, KBU 브랜치 유산)를 대체.

실행:
    cd backend && uvicorn main_light:app --host 0.0.0.0 --port $PORT
필요 패키지: backend/requirements-light.txt (backend/requirements.txt 전체가 아님)
"""
from fastapi import FastAPI

from routers import estimate, repair_shops

app = FastAPI(title="CarDoc Backend (light)", version="0.1.0")

app.include_router(estimate.router, tags=["estimate"])
app.include_router(repair_shops.router, tags=["repair_shops"])


@app.get("/")
async def root():
    return {"status": "ok", "service": "CarDoc backend (light: estimate + repair_shops only)"}


@app.get("/health")
def health():
    return {"status": "ok"}
