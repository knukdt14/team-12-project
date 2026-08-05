"""POST /chat — 진단·견적 결과를 바탕으로 한 정비 상담 챗봇 (RAG + LLM).

흐름:
    질문 → services/rag.py로 지식 문서 검색(top-3)
         → 검색 결과 + 진단 요약(payload.diagnosis_summary)으로 프롬프트 구성
         → services/llm_client.py로 Ollama 호출 (system/user 역할 분리)
         → 금액 가드레일 검사
         → 위반하거나 실패하면 검색 결과 기반 폴백 응답

세션 저장소 없이, 프론트가 매 요청마다 그 세션의 진단·견적 요약 문자열을
diagnosis_summary로 같이 보내는 방식으로 컨텍스트를 주입한다.

금액 가드레일 (단가표.json llm_guardrails.rule1):
    "금액은 단가표의 min/max를 그대로 사용하고 LLM이 임의 생성·수정하지 않는다"

    프롬프트로만 지시하면 작은 모델은 이를 무시하고 금액을 지어낸다
    (실제로 "타이어 펑크는 500만원" 같은 응답이 관측됨). 프롬프트는 요청이지
    강제가 아니므로, 생성된 답변을 코드로 한 번 더 검사한다. 진단 결과에
    없는 금액이 답변에 등장하면 그 답변은 버린다.
"""
import re

from fastapi import APIRouter

from schemas import ChatRequest, ChatResponse
from services import llm_client, rag

router = APIRouter()

NO_ANSWER = (
    "질문에 답할 근거 자료를 찾지 못했습니다. "
    "정확한 확인은 정비소 방문이 필요합니다."
)

DISCLAIMER = "실제 견적은 차종·연식·업체에 따라 달라질 수 있습니다."

NO_PRICE_ANSWER = (
    "이 손상은 단가표에 등록된 조합이 아니어서 예상 금액을 산출하지 못했습니다. "
    "정확한 비용은 정비소에서 실물을 확인한 뒤 정밀 견적을 받아보셔야 합니다.\n\n"
    f"{DISCLAIMER}"
)

# "12,000원", "35만원", "3 만 원" 등 금액 표현
PRICE_PATTERN = re.compile(r"\d[\d,\.\s]*\s*만?\s*원")

# CJK 통합 한자 영역. 작은 모델은 답변 중간에 중국어로 새는 일이 잦다.
# 우리 지식 문서와 진단 결과는 전부 한글이므로, 한자가 나오면 언어 이탈로 간주한다.
HANZI_PATTERN = re.compile(r"[一-鿿]")

KOREAN_ONLY_REMINDER = (
    "\n\n[경고] 직전 답변에 중국어가 섞였습니다. "
    "이번에는 반드시 한글과 숫자만 사용해 한국어로 다시 답변하세요."
)

# 모델이 답변 대신 프롬프트를 그대로 옮겨 적는 경우(prompt echo)의 흔적.
ECHO_MARKERS = ("■ 진단 결과", "■ 참고 자료", "■ 고객 질문")


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


def _fallback_answer(contexts, has_price: bool) -> str:
    """LLM을 못 쓰거나 답변을 폐기했을 때, 검색된 지식 문서로 대체."""
    if not has_price and not contexts:
        return NO_PRICE_ANSWER
    if not contexts:
        return NO_ANSWER

    body = "\n\n".join(f"· {c['text'].strip()}" for c in contexts[:2])
    head = (
        "예상 금액은 단가표에 해당 조합이 없어 제시할 수 없습니다. "
        "관련 정비 자료를 안내드립니다.\n\n"
        if not has_price
        else "관련 정비 자료를 안내드립니다.\n\n"
    )
    return f"{head}{body}\n\n{DISCLAIMER}"


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest):
    contexts = rag.search(payload.message)
    summary = payload.diagnosis_summary or ""
    has_price = bool(PRICE_PATTERN.search(summary))

    prompt = rag.build_prompt(
        question=payload.message,
        contexts=contexts,
        diagnosis_summary=summary,
    )

    answer = llm_client.generate(prompt, system=rag.SYSTEM_PROMPT)

    # 중국어가 섞이면 경고를 붙여 한 번만 다시 생성한다.
    # (무한 재시도는 CPU 추론 시간이 배로 늘어나므로 1회로 제한)
    if answer and _has_chinese(answer):
        print("[chat] 답변에 중국어 감지 — 한국어로 재생성을 1회 시도합니다.")
        retry = llm_client.generate(prompt, system=rag.SYSTEM_PROMPT + KOREAN_ONLY_REMINDER)
        if retry and not _has_chinese(retry):
            answer = retry
        else:
            print("[chat] 재생성에도 중국어가 남아 답변을 폐기합니다.")
            answer = None

    if answer and _is_echo(answer):
        answer = None

    if answer and _violates_price_guardrail(answer, summary):
        answer = None  # 근거 없는 금액을 만든 답변은 사용하지 않는다

    used_llm = answer is not None
    if not used_llm:
        answer = _fallback_answer(contexts, has_price)

    return ChatResponse(
        answer=answer,
        sources=[c["source"] for c in contexts],
        used_llm=used_llm,
    )
