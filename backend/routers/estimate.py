"""POST /estimate — 탐지 결과 → 수리비 견적 산출.

TODO: services/estimator.py(단가표 조회·룰베이스 계산) + services/llm_client.py
(계산된 값을 3문장 리포트로 요약) 연결. 지금은 API 형태만 확인하기 위한 더미 응답.
"""
from fastapi import APIRouter

from schemas import EstimateItem, EstimateRequest, EstimateResponse

router = APIRouter()


@router.post("/estimate", response_model=EstimateResponse)
async def estimate(payload: EstimateRequest):
    # 더미 응답 — 단가표.json 조회 로직 연결 전
    items = [
        EstimateItem(part="front-bumper", method="판금+도색", min_cost=150000, max_cost=300000)
    ]
    total = sum(item.max_cost for item in items)
    return EstimateResponse(
        items=items,
        total=total,
        report="전방 범퍼에 중간 정도의 찌그러짐이 확인되어 판금 및 도색 수리가 필요합니다.",
    )
