"""POST /chat — 세션의 진단·견적 결과를 바탕으로 한 정비 상담 챗봇.

TODO: services/rag.py(FAISS 검색 top-3 청크) + services/llm_client.py(Ollama
EXAONE 호출) 연결. RAG 인덱스(ai/docs, build_vectorstore.py)가 아직 없어서
이 엔드포인트는 diagnose/estimate보다 뒤에 작업하는 걸 권장. 지금은 API 형태만
확인하기 위한 더미 응답.
"""
from fastapi import APIRouter

from schemas import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest):
    # 더미 응답 — RAG/LLM 연결 전
    return ChatResponse(
        answer=f"(더미 응답) '{payload.message}'에 대한 상담 기능은 아직 준비 중입니다."
    )
