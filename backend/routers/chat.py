"""진단·견적과 검색 근거를 결합한 차량 정비 상담 API."""

from __future__ import annotations

import re
from functools import partial

from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

from schemas import ChatRequest, ChatResponse, RAGSource
from services import llm_client, rag

router = APIRouter()

DISCLAIMER = "실제 견적과 수리 범위는 차종·연식·옵션·업체의 실물 점검에 따라 달라질 수 있습니다."
NO_ANSWER = (
    "현재 질문을 뒷받침할 진단 결과나 정비 근거를 찾지 못했습니다. "
    "손상 부위가 보이는 전체 사진과 근접 사진, 주행 중 나타나는 증상을 알려주세요."
)
NO_PRICE_ANSWER = (
    "이 손상은 현재 단가표에 등록된 조합이 아니어서 예상 금액을 산출하지 못했습니다. "
    "금액을 추측하지 않고, 정비소의 실물 점검 후 정밀 견적을 받는 것이 안전합니다."
)

PRICE_PATTERN = re.compile(
    r"(?<!\d)(?P<number>\d[\d,\s]*(?:\.\d+)?)\s*(?P<man>만)?\s*원"
)
PRICE_RANGE_PATTERN = re.compile(
    r"(?<!\d)(?P<first>\d[\d,\s]*(?:\.\d+)?)\s*(?P<first_man>만)?\s*(?:원)?\s*"
    r"(?:~|〜|－|-|–|—|에서)\s*"
    r"(?P<second>\d[\d,\s]*(?:\.\d+)?)\s*(?P<second_man>만)?\s*원"
)
HANZI_PATTERN = re.compile(r"[一-鿿]")
KOREAN_ONLY_REMINDER = (
    "\n\n직전 답변에 한자 또는 중국어가 섞였습니다. "
    "고유한 정비 용어를 포함해 자연스러운 한국어와 숫자만 사용해 다시 답하세요."
)
ECHO_MARKERS = (
    "[진단·견적",
    "[검색 근거",
    "[최근 대화",
    "[현재 고객 질문]",
    "근거 우선순위:",
)


def _has_chinese(text: str) -> bool:
    return bool(HANZI_PATTERN.search(text))


def _is_echo(text: str) -> bool:
    hit = [marker for marker in ECHO_MARKERS if marker in text]
    if hit:
        print(f"[chat] 프롬프트 되읊기 감지 {hit} - 답변 폐기")
        return True
    return False


def _to_won(number: str, uses_man: bool) -> int:
    normalized = number.replace(",", "").replace(" ", "")
    value = float(normalized)
    return int(round(value * 10000 if uses_man else value))


def _money_values(text: str) -> set[int]:
    """'300,000원'과 '30만원'을 같은 원 단위 값으로 정규화한다."""
    values: set[int] = set()
    range_spans = []
    for match in PRICE_RANGE_PATTERN.finditer(text):
        first_man = bool(match.group("first_man"))
        second_man = bool(match.group("second_man"))
        # '30~45만원'처럼 앞 단위가 생략되면 뒤 단위를 상속한다.
        values.add(_to_won(match.group("first"), first_man or second_man))
        values.add(_to_won(match.group("second"), second_man))
        range_spans.append(match.span())

    for match in PRICE_PATTERN.finditer(text):
        if any(start <= match.start() < end for start, end in range_spans):
            continue
        values.add(_to_won(match.group("number"), bool(match.group("man"))))
    return values


def _violates_price_guardrail(answer: str, diagnosis_summary: str) -> bool:
    mentioned = _money_values(answer)
    if not mentioned:
        return False

    allowed = _money_values(diagnosis_summary)
    unexpected = mentioned - allowed
    if unexpected:
        print(f"[chat] 금액 가드레일 위반 - 허용되지 않은 원 단위 값 {sorted(unexpected)}")
        return True
    return False


def _fallback_answer(chunks: list[rag.RetrievedChunk], has_price: bool) -> str:
    if not chunks:
        return NO_PRICE_ANSWER if not has_price else NO_ANSWER

    excerpts = []
    for chunk in chunks:
        # 폴백은 LLM 가드레일을 거치지 않으므로 RAG 표의 금액 행을 직접 노출하지
        # 않는다. 사용자 세션의 확정 룰베이스 금액은 별도 견적 화면이 담당한다.
        safe_lines = [
            line for line in chunk.content.splitlines() if not _money_values(line)
        ]
        compact = re.sub(r"\n{3,}", "\n\n", "\n".join(safe_lines).strip())
        if not compact:
            continue
        excerpts.append(f"**{chunk.title} · {chunk.section}**\n{compact[:900]}")
        if len(excerpts) >= 2:
            break
    if not excerpts:
        return NO_PRICE_ANSWER if not has_price else NO_ANSWER
    head = (
        "예상 금액은 현재 단가표에 해당 조합이 없어 제시할 수 없습니다. "
        "대신 관련 점검 기준을 안내합니다.\n\n"
        if not has_price
        else "검색된 정비 기준을 안내합니다.\n\n"
    )
    return head + "\n\n".join(excerpts)


def _source_models(chunks: list[rag.RetrievedChunk]) -> list[RAGSource]:
    sources, seen = [], set()
    for chunk in chunks:
        key = (chunk.source, chunk.section)
        if key in seen:
            continue
        seen.add(key)
        sources.append(RAGSource(**chunk.source_payload()))
    return sources


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest):
    history = [item.model_dump() for item in payload.history[-6:]]
    summary = payload.diagnosis_summary.strip()
    chunks = await run_in_threadpool(
        partial(
            rag.search,
            payload.message,
            diagnosis_context=summary,
            history=history,
        )
    )
    has_price = bool(_money_values(summary))

    answer = await run_in_threadpool(
        partial(
            llm_client.generate,
            payload.message,
            context_chunks=chunks,
            diagnose_estimate_context=summary,
            history=history,
        )
    )

    if answer and _has_chinese(answer):
        retry = await run_in_threadpool(
            partial(
                llm_client.generate,
                payload.message + KOREAN_ONLY_REMINDER,
                context_chunks=chunks,
                diagnose_estimate_context=summary,
                history=history,
            )
        )
        answer = retry if retry and not _has_chinese(retry) else None

    if answer and (_is_echo(answer) or _violates_price_guardrail(answer, summary)):
        answer = None

    used_llm = answer is not None
    if not used_llm:
        answer = _fallback_answer(chunks, has_price)

    answer = f"{answer.rstrip()}\n\n_{DISCLAIMER}_"
    return ChatResponse(
        answer=answer,
        used_llm=used_llm,
        rag_used=bool(chunks),
        sources=_source_models(chunks),
    )
