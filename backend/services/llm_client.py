"""Ollama REST 호출 클라이언트 (EXAONE-3.5-2.4B-Instruct, GGUF Q4).

Docker Compose 환경에서는 컨테이너 내부 네트워크로 http://llm:11434 를 쓰고,
로컬 단독 실행 시에는 http://localhost:11434 를 쓴다 — 환경변수 OLLAMA_BASE_URL로
오버라이드 가능(docker-compose.yml에서 backend 서비스에 주입 예정).

주의: 아직 Ollama가 로컬에 세팅되어 있지 않으면 generate()가 ConnectionError를
던진다 — routers/chat.py에서 이를 잡아 "상담 기능을 일시적으로 사용할 수 없습니다"
같은 안내로 폴백해야 한다.
"""
import os
from typing import List, Optional

import requests

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "exaone3.5:2.4b")
MAX_TOKENS = 300
REQUEST_TIMEOUT = 60  # 초 — 로컬 CPU 추론은 느릴 수 있어 여유있게

# 상담 프롬프트 가드레일 (README "생존 원칙"/llm_guardrails 참고):
# - 검색된 청크 밖 내용은 답변하지 않고 "모르면 정비소 방문 권유"
# - 3문장 이내로 제한
SYSTEM_PROMPT = (
    "당신은 차량 파손 진단·수리비 상담 챗봇입니다. "
    "아래 제공된 참고 자료 범위 안에서만 답변하세요. "
    "참고 자료에 없는 내용은 추측하지 말고 '정확한 정보를 위해 정비소 방문을 권장드립니다'라고 안내하세요. "
    "답변은 3문장 이내로 간결하게 작성하세요. "
    "금액 수치는 직접 생성하지 말고, 이미 계산되어 전달된 값만 인용하세요."
)


def _build_prompt(user_message: str, context_chunks: List[str], diagnose_estimate_context: Optional[str]):
    parts = []
    if diagnose_estimate_context:
        parts.append(f"[진단·견적 결과]\n{diagnose_estimate_context}")
    if context_chunks:
        joined = "\n---\n".join(context_chunks)
        parts.append(f"[참고 자료]\n{joined}")
    parts.append(f"[사용자 질문]\n{user_message}")
    return "\n\n".join(parts)


def generate(user_message: str, context_chunks: List[str] = None, diagnose_estimate_context: str = None):
    """Ollama /api/generate 호출해서 답변 문자열을 반환 (비스트리밍).

    Ollama가 안 떠 있으면 requests.exceptions.ConnectionError가 그대로 올라간다.
    """
    prompt = _build_prompt(user_message, context_chunks or [], diagnose_estimate_context)

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": MODEL_NAME,
            "system": SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": MAX_TOKENS},
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["response"].strip()
