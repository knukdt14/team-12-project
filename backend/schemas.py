"""Pydantic 요청/응답 스키마.

API 명세는 프로젝트 루트 README.md "4. API 명세" 참고.
지금은 스캐폴딩 단계라 값 검증 위주로만 정의하고, 실제 추론/계산 로직은
services/ 쪽에 연결되는 대로 여기 필드도 맞춰 조정합니다.
"""
from typing import List

from pydantic import BaseModel


class Detection(BaseModel):
    part: str  # 화면 표시용 한글 부위명 (예: "전방 범퍼")
    part_en: str  # 단가표.json 조회용 YOLO 원본 클래스명 (예: "front-bumper-dent")
    damage_type: str  # 화면 표시용 한글 손상 종류 (예: "찌그러짐")
    damage_type_en: str  # 단가표.json 조회용 분류기 원본 클래스명 (예: "dent")
    severity: str  # "minor" | "moderate" | "severe" (단가표.json 정의와 통일)
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]


class DiagnoseResponse(BaseModel):
    results: List[Detection]


class EstimateItem(BaseModel):
    part: str
    method: str
    min_cost: int
    max_cost: int


class EstimateRequest(BaseModel):
    results: List[Detection]


class EstimateResponse(BaseModel):
    items: List[EstimateItem]
    total: int
    report: str  # LLM이 생성한 요약 문장 (services/llm_client.py 연결 전까지는 더미 텍스트)


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str
