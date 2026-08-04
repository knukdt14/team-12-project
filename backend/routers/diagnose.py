"""POST /diagnose — 파손 이미지 업로드 → 부위/손상 탐지 결과 반환.

services/detector.py의 실제 YOLO(부위 탐지) + ResNet18(손상 종류 분류) 추론과 연결됨.
severity는 박스 크기(이미지 대비 면적 비율) 기준 low/medium/high 휴리스틱 (services/detector.py 참고).
"""
from fastapi import APIRouter, File, HTTPException, UploadFile

from schemas import Detection, DiagnoseResponse
from services import detector

router = APIRouter()


@router.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(file: UploadFile = File(...)):
    image_bytes = await file.read()
    try:
        raw_results = detector.detect(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    results = [Detection(**r) for r in raw_results]
    return DiagnoseResponse(results=results)
