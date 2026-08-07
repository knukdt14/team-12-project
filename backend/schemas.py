"""Pydantic 요청/응답 스키마.

API 명세는 프로젝트 루트 README.md "4. API 명세" 참고.
지금은 스캐폴딩 단계라 값 검증 위주로만 정의하고, 실제 추론/계산 로직은
services/ 쪽에 연결되는 대로 여기 필드도 맞춰 조정합니다.
"""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


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


class EstimateRequest(BaseModel):
    """단일 손상 건 기준 견적 요청 (팀원 estimate_api.py 구조와 통일).

    part: 단가표.json items 키(예: "front_bumper") 또는 YOLO part_en(예:
    "front-bumper-dent") 둘 다 허용 — services/estimator.py에서 자동 매핑 시도.
    /diagnose가 여러 건을 탐지하면, 프론트엔드가 건별로 이 엔드포인트를 반복 호출한다.
    """

    part: str
    damage_type: str
    severity: str


class EstimateResponse(BaseModel):
    """팀원 estimate_api.py와 동일한 필드 구성. 조회 실패 시 success=False +
    message만 채워지고 나머지는 None."""

    success: bool
    part: Optional[str] = None
    part_label: Optional[str] = None
    damage_type: Optional[str] = None
    severity: Optional[str] = None
    repair_method: Optional[str] = None
    min_cost: Optional[int] = None
    max_cost: Optional[int] = None
    source: Optional[str] = None
    note: Optional[str] = None
    disclaimer: Optional[str] = None
    message: Optional[str] = None


class RepairShop(BaseModel):
    name: str
    address: str
    phone: str
    distance: str
    lat: float
    lng: float
    place_url: str


class RepairShopsResponse(BaseModel):
    success: bool
    count: Optional[int] = None
    shops: Optional[List[RepairShop]] = None
    message: Optional[str] = None


class GeocodeResponse(BaseModel):
    success: bool
    address: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    message: Optional[str] = None


class ChatHistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class RAGSource(BaseModel):
    title: str
    section: str
    source: str


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=2000)
    # 이번 세션의 진단·견적 요약. 챗봇이 금액을 지어내지 않고 이 값을 인용하도록
    # 프롬프트에 그대로 넣는다 (단가표.json llm_guardrails.rule1).
    diagnosis_summary: str = Field(default="", max_length=4000)
    # 대명사형 후속 질문("그럼 교체해야 해?")을 해석하기 위한 최근 대화.
    history: List[ChatHistoryItem] = Field(default_factory=list, max_length=8)


class ChatResponse(BaseModel):
    answer: str
    used_llm: bool = True  # False면 룰베이스 또는 검색 결과 기반 안전 응답
    answer_mode: Literal["llm", "rule_based", "rag_fallback"] = "llm"
    rag_used: bool = False
    sources: List[RAGSource] = Field(default_factory=list)
