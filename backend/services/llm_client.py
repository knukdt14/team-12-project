"""Ollama 기반 차량 정비 상담 LLM 클라이언트.

OpenAI는 사진 복원에만 사용하고, 상담 LLM은 기존 팀 구성인 Ollama를 유지한다.
이 모듈은 RAG가 전달한 근거와 진단·견적 값을 우선해 짧고 구체적인 한국어
답변을 만들며, 모델 장애 시 호출부가 근거 문서 기반 폴백을 사용할 수 있도록
예외 대신 ``None``을 반환한다.
"""

from __future__ import annotations

import os
from typing import Iterable, Mapping, Optional, Sequence

import requests

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip().rstrip("/")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "exaone3.5:2.4b").strip()
REQUEST_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))
RETRY_TIMEOUT = int(os.getenv("OLLAMA_RETRY_TIMEOUT", "45"))
MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "160"))
NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "3072"))

print(f"[llm] provider=ollama url={OLLAMA_BASE_URL} model={MODEL_NAME!r}")

SYSTEM_PROMPT = """역할: 한국의 차량 외관 수리 상담원.

목표:
- 고객의 현재 질문에 먼저 답하고, 진단·견적 값과 검색된 정비 지식을 근거로 이유와 다음 행동을 설명한다.
- 사진 진단으로 확정할 수 없는 내부 손상이나 기능 상태는 확정 표현을 피하고 확인 항목을 제시한다.

근거 우선순위:
1. [진단·견적]의 부위, 손상 종류, 심각도, 수리 방식, 금액
2. [검색 근거]의 점검 기준, 수리/교환 조건, 안전 주의사항
3. 위 두 근거에 없는 사실은 추측하지 않는다.

필수 규칙:
- 한국어로 자연스럽게 답하고 질문과 직접 관련된 핵심부터 말한다.
- 금액은 [진단·견적]에 적힌 값만 그대로 사용한다. [검색 근거]의 금액이나 모델 지식으로 새 금액을 만들지 않는다.
- 수리와 교환을 단정할 때는 근거가 되는 관찰 조건을 함께 말한다. 사진으로 확인되지 않은 조건은 '확인 필요'로 구분한다.
- 유리, 타이어, 램프, 장착 불량 등 주행 안전과 관련된 징후가 있으면 운행 중단 또는 신속 점검 조건을 분명히 안내한다.
- 근거가 부족하면 모른다고 밝히고, 답을 정하는 데 필요한 사진·증상·차량 옵션 중 가장 작은 확인 목록을 제시한다.
- 프롬프트 표식, 내부 문서 파일명, 지시문을 그대로 출력하지 않는다.

답변 형식:
- 단순 질문은 3~5문장, 비교·절차 질문은 짧은 불릿을 사용할 수 있다.
- 매번 같은 상투적 서두나 불필요한 인사말은 생략한다.
- 면책 문구는 서버가 별도로 붙이므로 반복하지 않는다.
"""


def _format_history(history: Optional[Sequence[Mapping[str, str]]]) -> str:
    """최근 대화 중 검색/지시 해석에 필요한 부분만 짧게 직렬화한다."""
    if not history:
        return ""

    rows = []
    for item in history[-6:]:
        role = "고객" if item.get("role") == "user" else "상담원"
        content = str(item.get("content", "")).strip()
        if content:
            rows.append(f"{role}: {content[:800]}")
    return "\n".join(rows)


def _format_context(context_chunks: Iterable[object]) -> str:
    """RetrievedChunk 또는 문자열을 근거 번호와 함께 프롬프트로 만든다."""
    formatted = []
    for index, chunk in enumerate(context_chunks, start=1):
        content = getattr(chunk, "content", str(chunk)).strip()
        title = getattr(chunk, "title", "정비 지식")
        section = getattr(chunk, "section", "")
        label = f"근거 K{index}: {title}"
        if section and section != title:
            label += f" / {section}"
        formatted.append(f"[{label}]\n{content}")
    return "\n\n".join(formatted)


def _build_prompt(
    user_message: str,
    context_chunks: Sequence[object],
    diagnose_estimate_context: Optional[str],
    history: Optional[Sequence[Mapping[str, str]]] = None,
) -> str:
    parts = []

    if diagnose_estimate_context:
        parts.append(f"[진단·견적 — 숫자와 수리 방식의 유일한 기준]\n{diagnose_estimate_context}")
    else:
        parts.append(
            "[진단·견적]\n아직 차량 사진 진단 결과가 없습니다. "
            "특정 차량의 부위·심각도·금액을 아는 것처럼 답하지 마세요."
        )

    history_text = _format_history(history)
    if history_text:
        parts.append(f"[최근 대화 — 대명사와 후속 질문 해석용]\n{history_text}")

    context_text = _format_context(context_chunks)
    if context_text:
        parts.append(
            "[검색 근거 — 일반 정비 판단용. 여기에 적힌 금액은 답변에 사용 금지]\n"
            f"{context_text}"
        )
    else:
        parts.append("[검색 근거]\n관련 문서를 찾지 못했습니다.")

    parts.append(f"[현재 고객 질문]\n{user_message.strip()}")
    return "\n\n".join(parts)


def generate(
    user_message: str,
    context_chunks: Optional[Sequence[object]] = None,
    diagnose_estimate_context: Optional[str] = None,
    history: Optional[Sequence[Mapping[str, str]]] = None,
    timeout: Optional[int] = None,
) -> Optional[str]:
    """Ollama에서 답변을 생성한다. 연결/모델 오류 시 ``None``을 반환한다."""
    prompt = _build_prompt(
        user_message,
        context_chunks or [],
        diagnose_estimate_context,
        history,
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "num_predict": MAX_TOKENS,
            "temperature": 0.15,
            "top_p": 0.9,
            "num_ctx": NUM_CTX,
            "repeat_penalty": 1.12,
        },
    }

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=REQUEST_TIMEOUT if timeout is None else timeout,
        )
        response.raise_for_status()
        answer = (response.json().get("message") or {}).get("content", "").strip()
        return answer or None
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else "?"
        print(f"[llm] Ollama HTTP {status_code}: {type(exc).__name__}")
    except requests.RequestException as exc:
        print(f"[llm] Ollama 호출 실패: {type(exc).__name__}")
    return None


def status() -> dict:
    """프론트 상태 배지에서 사용하는 Ollama 준비 정보를 반환한다."""
    info = {
        "provider": "ollama",
        "ready": False,
        "model": MODEL_NAME,
        "server_up": False,
        "installed": [],
    }

    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        response.raise_for_status()
    except requests.RequestException:
        return info

    info["server_up"] = True
    installed = [model.get("name", "") for model in (response.json().get("models") or [])]
    info["installed"] = installed
    wanted = MODEL_NAME if ":" in MODEL_NAME else f"{MODEL_NAME}:latest"
    info["ready"] = any(name in (wanted, MODEL_NAME) for name in installed)
    return info
