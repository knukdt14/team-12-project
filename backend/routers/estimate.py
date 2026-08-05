"""POST /estimate — 단일 손상 건 기준 수리비 견적 산출.

팀원(estimate_api.py) 구조와 통일: 이전 버전은 /diagnose 결과 리스트 전체를
한 번에 받았지만, 지금은 (부위, 손상 종류, 심각도) 단일 건을 받는다.
/diagnose가 여러 건을 탐지하면 프론트엔드가 건별로 이 엔드포인트를 반복 호출한다.

services/estimator.py(단가표 조회)와 연결됨.
"""
from fastapi import APIRouter

from schemas import EstimateRequest, EstimateResponse
from services import estimator

router = APIRouter()


@router.post("/estimate", response_model=EstimateResponse)
async def estimate_repair(payload: EstimateRequest):
    result = estimator.estimate(payload.part, payload.damage_type, payload.severity)
    return EstimateResponse(**result)
