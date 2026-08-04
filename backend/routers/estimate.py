"""POST /estimate — 탐지 결과 → 수리비 견적 산출.

services/estimator.py(단가표 조회)와 연결됨.
LLM 리포트 생성(services/llm_client.py)은 아직 없어서, report는 조회 결과를
템플릿 문장으로 조립한 것 — 나중에 llm_client.py 연결되면 이 부분을 "계산된
값을 3문장으로 다듬어달라"는 프롬프트로 LLM에 넘기도록 교체할 것 (금액 자체는
LLM이 생성하지 않고 단가표 값을 그대로 사용해야 함 — README llm_guardrails 참고).
"""
from fastapi import APIRouter

from schemas import EstimateItem, EstimateRequest, EstimateResponse
from services import estimator

router = APIRouter()


@router.post("/estimate", response_model=EstimateResponse)
async def estimate(payload: EstimateRequest):
    items = []
    report_lines = []

    for det in payload.results:
        priced = estimator.lookup(det.part_en, det.damage_type_en, det.severity)

        if priced is None:
            items.append(
                EstimateItem(
                    part=det.part,
                    method=estimator.NO_PRICE_MESSAGE,
                    min_cost=0,
                    max_cost=0,
                )
            )
            report_lines.append(f"{det.part}: {estimator.NO_PRICE_MESSAGE}")
            continue

        items.append(
            EstimateItem(
                part=det.part,
                method=priced["method"],
                min_cost=priced["min_cost"],
                max_cost=priced["max_cost"],
            )
        )
        report_lines.append(
            f"{det.part}에 {det.damage_type} 손상이 확인되어 {priced['method']}이(가) 필요합니다."
        )

    total = sum(item.max_cost for item in items)
    report = " ".join(report_lines) if report_lines else "탐지된 손상이 없어 견적 항목이 없습니다."

    return EstimateResponse(items=items, total=total, report=report)
