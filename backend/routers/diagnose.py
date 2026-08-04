"""POST /diagnose — 파손 이미지 업로드 → 부위/손상 탐지 결과 반환.

TODO: services/detector.py 연결 후 app.py의 YOLO(부위 탐지) + ResNet18(손상 종류
분류) 추론 로직으로 교체. 지금은 API 형태만 확인하기 위한 더미 응답.
"""
from fastapi import APIRouter, File, UploadFile

from schemas import Detection, DiagnoseResponse

router = APIRouter()


@router.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(file: UploadFile = File(...)):
    # 더미 응답 — 업로드된 파일 내용은 아직 사용하지 않음(형태만 검증)
    dummy = [
        Detection(
            part="front-bumper",
            damage_type="dent",
            severity="medium",
            confidence=0.87,
            bbox=[120.0, 80.0, 340.0, 260.0],
        )
    ]
    return DiagnoseResponse(results=dummy)
