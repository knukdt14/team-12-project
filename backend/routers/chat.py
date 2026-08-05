"""POST /chat — 정비 상담 챗봇 (RAG + LLM).

services/rag.py(FAISS 검색) + services/llm_client.py(Ollama EXAONE 호출) 연결.

TODO(세션 컨텍스트 주입): README 스펙상 "세션의 진단·견적 결과를 시스템 프롬프트에
주입"해야 하는데, 지금 /diagnose·/estimate가 session_id를 받지 않아서 결과를 세션에
저장할 방법이 아직 없다. 우선 RAG+LLM 기본 상담만 연결했고, 진단/견적 결과를 대화에
반영하는 건 별도 작업 필요 (diagnose/estimate에 session_id 추가 + 세션 저장소 설계).
"""
from fastapi import APIRouter, HTTPException

from schemas import ChatRequest, ChatResponse
from services import llm_client, rag

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest):
    chunks = rag.search(payload.message)

    try:
        answer = llm_client.generate(payload.message, context_chunks=chunks)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"상담 기능을 일시적으로 사용할 수 없습니다 (LLM 서버 연결 실패: {e})",
        )

    return ChatResponse(answer=answer)
