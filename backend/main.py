"""FastAPI 엔트리포인트.

실행: cd backend && uvicorn main:app --reload
Swagger UI: http://localhost:8000/docs
"""
from fastapi import FastAPI

from routers import chat, diagnose, estimate

app = FastAPI(title="CarDoc Backend", version="0.1.0")

app.include_router(diagnose.router, tags=["diagnose"])
app.include_router(estimate.router, tags=["estimate"])
app.include_router(chat.router, tags=["chat"])


@app.get("/")
async def root():
    return {"status": "ok", "service": "CarDoc backend"}
