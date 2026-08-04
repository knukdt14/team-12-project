"""FastAPI 엔트리포인트.

실행: cd backend && uvicorn main:app --reload
Swagger UI: http://localhost:8000/docs
"""
from dotenv import load_dotenv
from fastapi import FastAPI

# 인자 없이 호출하면 현재 작업 디렉토리부터 상위로 올라가며 .env를 찾는다.
# 저장소 루트의 .env(카카오 API 키 등)를 backend/ 안에서 실행해도 자동으로 찾는다.
load_dotenv()

from routers import chat, diagnose, estimate, repair_shops  # noqa: E402

app = FastAPI(title="CarDoc Backend", version="0.1.0")

app.include_router(diagnose.router, tags=["diagnose"])
app.include_router(estimate.router, tags=["estimate"])
app.include_router(repair_shops.router, tags=["repair-shops"])
app.include_router(chat.router, tags=["chat"])


@app.get("/")
async def root():
    return {"status": "ok", "service": "CarDoc backend"}
