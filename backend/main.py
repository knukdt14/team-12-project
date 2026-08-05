"""FastAPI 엔트리포인트.

실행: cd backend && uvicorn main:app --reload
Swagger UI: http://localhost:8000/docs
"""
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 인자 없이 호출하면 현재 작업 디렉토리부터 상위로 올라가며 .env를 찾는다.
# 저장소 루트의 .env(카카오 API 키 등)를 backend/ 안에서 실행해도 자동으로 찾는다.
load_dotenv()

from routers import chat, diagnose, estimate, repair_shops  # noqa: E402

app = FastAPI(title="CarDoc Backend", version="0.2.0")

# 프론트(Streamlit)가 다른 컨테이너/오리진에서 호출하므로 CORS를 허용한다.
# 운영 배포 시에는 ALLOWED_ORIGINS 환경변수로 도메인을 좁히는 것을 권장.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(diagnose.router, tags=["diagnose"])
app.include_router(estimate.router, tags=["estimate"])
app.include_router(repair_shops.router, tags=["repair-shops"])
app.include_router(chat.router, tags=["chat"])


@app.get("/")
async def root():
    return {"status": "ok", "service": "CarDoc backend"}


@app.get("/health")
async def health():
    """docker-compose healthcheck 용.

    여기서 LLM 상태를 확인하지 않는 것은 의도적이다. 모델 다운로드(약 1.6GB)를
    기다리게 하면 backend가 unhealthy로 잡히고, depends_on 때문에 frontend까지
    안 뜬다. 진단·견적은 LLM 없이도 동작하므로 UI를 먼저 띄우는 편이 낫다.
    """
    return {"status": "ok"}


@app.get("/health/llm")
async def health_llm():
    """LLM(Ollama) 준비 상태. 프론트 사이드바 배지가 조회한다.

    첫 기동 직후에는 server_up=True, ready=False인 구간이 2~3분 있다.
    이때 /chat은 실패하지 않고 검색 결과 기반 폴백 응답을 낸다.
    """
    from services import llm_client

    return llm_client.status()
