"""POST /chat — 진단·견적 결과를 바탕으로 한 정비 상담 챗봇 (RAG + LLM).

흐름:
    질문 → services/rag.py로 ai/docs 지식 검색(top-3)
         → 검색 결과 + 진단 요약으로 프롬프트 구성
         → services/llm_client.py로 Ollama 호출 (system/user 역할 분리)
         → 금액 가드레일 검사
         → 위반하거나 실패하면 검색 결과 기반 폴백 응답

세션 컨텍스트:
    /diagnose·/estimate가 session_id를 받지 않아 서버에 세션 저장소가 없다.
    대신 프론트가 ChatRequest.diagnosis_summary에 이번 세션의 진단·견적 요약을
    실어 보낸다. backend를 무상태로 유지할 수 있는 방식이다.

금액 가드레일(단가표.json llm_guardrails.rule1):
    "금액은 단가표의 min/max를 그대로 사용하고 LLM이 임의 생성·수정하지 않는다"

    프롬프트로만 지시하면 작은 모델은 이를 무시하고 금액을 지어낸다
    (실제로 "타이어 펑크는 500만원" 같은 응답이 관측됨).
    프롬프트는 요청이지 강제가 아니므로, 생성된 답변을 코드로 한 번 더 검사한다.
    진단 결과에 없는 금액이 답변에 등장하면 그 답변은 버린다.
"""
import re

from fastapi import APIRouter

from schemas import ChatRequest, ChatResponse
from services import llm_client, rag

router = APIRouter()

DISCLAIMER = "실제 견적은 차종·연식·업체에 따라 달라질 수 있습니다."

NO_ANSWER = (
    "질문에 답할 근거 자료를 찾지 못했습니다. "
    "정확한 확인은 정비소 방문이 필요합니다."
)

NO_PRICE_ANSWER = (
    "이 손상은 단가표에 등록된 조합이 아니어서 예상 금액을 산출하지 못했습니다. "
    "정확한 비용은 정비소에서 실물을 확인한 뒤 정밀 견적을 받아보셔야 합니다."
)

# "12,000원", "35만원", "3 만 원" 등 금액 표현
PRICE_PATTERN = re.compile(r"\d[\d,\.\s]*\s*만?\s*원")

# CJK 통합 한자 영역. 한국어 특화 모델로 바꾼 뒤로는 거의 안 걸리지만,
# 모델을 qwen 계열로 되돌릴 때를 대비해 가드레일은 남겨둔다.
HANZI_PATTERN = re.compile(r"[一-鿿]")

KOREAN_ONLY_REMINDER = (
    "\n\n[경고] 직전 답변에 중국어가 섞였습니다. "
    "이번에는 반드시 한글과 숫자만 사용해 한국어로 다시 답변하세요."
)

# 모델이 답변 대신 프롬프트를 그대로 옮겨 적는 경우(prompt echo)의 흔적.
ECHO_MARKERS = (
    "[이 고객의 진단 결과",
    "[참고 정비 자료",
    "[고객이 지금 물어본 것]",
    "[진단 결과]",
    "[참고 자료]",
)


def _has_chinese(text: str) -> bool:
    return bool(HANZI_PATTERN.search(text))


def _is_echo(text: str) -> bool:
    hit = [m for m in ECHO_MARKERS if m in text]
    if hit:
        print(f"[chat] 프롬프트 되읊기 감지 {hit} — 답변을 폐기합니다.")
        return True
    return False


def _numbers(text: str):
    """텍스트에서 숫자만 뽑아 비교용 집합으로 반환 (쉼표·공백 제거)."""
    return {n.replace(",", "").replace(" ", "") for n in re.findall(r"[\d,\s]*\d", text)}


def _violates_price_guardrail(answer: str, diagnosis_summary: str) -> bool:
    """답변에 진단 결과에 없는 금액이 들어 있으면 True."""
    mentioned = PRICE_PATTERN.findall(answer)
    if not mentioned:
        return False

    allowed = _numbers(diagnosis_summary)
    for token in mentioned:
        digits = re.sub(r"[^\d]", "", token)
        if digits and digits not in allowed:
            print(f"[chat] 금액 가드레일 위반 — 근거 없는 금액 '{token}' 생성됨. 답변 폐기.")
            return True
    return False


def _fallback_answer(chunks, has_price: bool) -> str:
    """LLM을 못 쓰거나 답변을 폐기했을 때, 검색된 지식 문서로 대체."""
    if not chunks:
        return NO_PRICE_ANSWER if not has_price else NO_ANSWER

    body = "\n\n".join(f"· {c.strip()}" for c in chunks[:2])
    head = (
        "예상 금액은 단가표에 해당 조합이 없어 제시할 수 없습니다. "
        "관련 정비 자료를 안내드립니다.\n\n"
        if not has_price
        else "관련 정비 자료를 안내드립니다.\n\n"
    )
    return f"{head}{body}"


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest):
    chunks = rag.search(payload.message)
    summary = payload.diagnosis_summary or ""
    has_price = bool(PRICE_PATTERN.search(summary))

    answer = llm_client.generate(
        payload.message,
        context_chunks=chunks,
        diagnose_estimate_context=summary,
    )

    # 중국어가 섞이면 경고를 붙여 한 번만 다시 생성한다.
    # (무한 재시도는 CPU 추론 시간이 배로 늘어나므로 1회로 제한)
    if answer and _has_chinese(answer):
        print("[chat] 답변에 중국어 감지 — 한국어로 재생성을 1회 시도합니다.")
        retry = llm_client.generate(
            payload.message + KOREAN_ONLY_REMINDER,
            context_chunks=chunks,
            diagnose_estimate_context=summary,
        )
        answer = retry if (retry and not _has_chinese(retry)) else None

    if answer and _is_echo(answer):
        answer = None

    if answer and _violates_price_guardrail(answer, summary):
        answer = None  # 근거 없는 금액을 만든 답변은 사용하지 않는다

    used_llm = answer is not None
    if not used_llm:
        answer = _fallback_answer(chunks, has_price)

    # 면책 고지는 LLM에게 시키지 않고 여기서 한 번만 붙인다.
    # 프롬프트로 시키면 매번 다른 표현으로 길게 늘어놓아 답변이 장황해지고,
    # 정작 물어본 내용에 쓸 문장 수를 잡아먹는다 (llm_guardrails.rule3).
    answer = f"{answer.rstrip()}\n\n_{DISCLAIMER}_"

    return ChatResponse(answer=answer, used_llm=used_llm)
