"""OpenAI 기반 차량 복원 예상 이미지 API."""

from __future__ import annotations

import asyncio
import hmac
import io
import json
import os
import time
from collections import defaultdict, deque

from fastapi import APIRouter, File, Form, Header, HTTPException, Request, Response, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from schemas import Detection
from services import image_restorer

router = APIRouter()

MAX_UPLOAD_BYTES = int(os.getenv("REPAIR_MAX_UPLOAD_MB", "20")) * 1024 * 1024
MAX_IMAGE_PIXELS = int(os.getenv("REPAIR_MAX_IMAGE_PIXELS", "20000000"))
MAX_DETECTIONS = 20
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
RATE_LIMIT_PER_MINUTE = max(1, int(os.getenv("REPAIR_RATE_LIMIT_PER_MINUTE", "4")))
MAX_CONCURRENT_GENERATIONS = max(
    1, int(os.getenv("REPAIR_MAX_CONCURRENT_GENERATIONS", "1"))
)
REPAIR_API_TOKEN = os.getenv("REPAIR_API_TOKEN", "").strip()
_generation_slots = asyncio.Semaphore(MAX_CONCURRENT_GENERATIONS)
_request_times: defaultdict[str, deque[float]] = defaultdict(deque)


def _authorize(provided_token: str | None) -> None:
    """배포에서 공유 토큰을 설정한 경우 유료 엔드포인트를 내부 호출로 제한."""
    if REPAIR_API_TOKEN and not hmac.compare_digest(
        provided_token or "", REPAIR_API_TOKEN
    ):
        raise HTTPException(status_code=401, detail="복원 API 인증에 실패했습니다.")


def _check_rate_limit(client_host: str) -> None:
    now = time.monotonic()
    recent = _request_times[client_host]
    while recent and now - recent[0] >= 60:
        recent.popleft()
    if len(recent) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(
            status_code=429,
            detail="복원 요청이 너무 많습니다. 잠시 후 다시 시도해주세요.",
        )
    recent.append(now)


@router.get("/health/repair-preview")
async def repair_preview_health():
    """실제 유료 호출 없이 API 키와 선택 모델 설정 여부만 반환한다."""
    return image_restorer.status()


@router.post(
    "/repair-preview",
    response_class=Response,
    responses={
        200: {"content": {"image/png": {}}, "description": "복원 예상 PNG"},
        413: {"description": "업로드 크기 초과"},
        422: {"description": "이미지 또는 탐지 정보 오류"},
        429: {"description": "OpenAI 요청 한도 초과"},
        502: {"description": "OpenAI 응답 오류"},
        503: {"description": "OpenAI 설정 또는 권한 오류"},
        504: {"description": "OpenAI 요청 시간 초과"},
    },
)
async def repair_preview(
    request: Request,
    file: UploadFile = File(...),
    detections_json: str = Form(...),
    x_repair_token: str | None = Header(default=None),
):
    _authorize(x_repair_token)
    if file.content_type and file.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=422, detail="JPG, PNG 또는 WebP 이미지만 지원합니다.")

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(status_code=422, detail="업로드된 이미지가 비어 있습니다.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="이미지 파일 크기 제한을 초과했습니다.")

    try:
        raw_detections = json.loads(detections_json)
        if not isinstance(raw_detections, list) or not raw_detections:
            raise ValueError
        if len(raw_detections) > MAX_DETECTIONS:
            raise HTTPException(status_code=422, detail="손상 영역이 너무 많습니다.")
        detections = [Detection.model_validate(item) for item in raw_detections]
    except HTTPException:
        raise
    except (json.JSONDecodeError, TypeError, ValueError, ValidationError) as exc:
        raise HTTPException(status_code=422, detail="손상 탐지 정보 형식이 올바르지 않습니다.") from exc

    detections = [detection for detection in detections if detection.confidence >= 0.3]
    if not detections:
        raise HTTPException(status_code=422, detail="복원할 손상 영역을 찾지 못했습니다.")

    try:
        original = Image.open(io.BytesIO(content))
        if original.format not in {"JPEG", "PNG", "WEBP"}:
            raise UnidentifiedImageError("unsupported image format")
        if original.width * original.height > MAX_IMAGE_PIXELS:
            raise HTTPException(status_code=413, detail="이미지 해상도 제한을 초과했습니다.")
        # 픽셀 수를 먼저 확인한 뒤 디코딩해 압축 폭탄의 메모리 사용을 제한한다.
        original.load()
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=422, detail="이미지 파일을 읽을 수 없습니다.") from exc

    payload = [detection.model_dump() for detection in detections]
    _check_rate_limit(request.client.host if request.client else "unknown")

    try:
        await asyncio.wait_for(_generation_slots.acquire(), timeout=0.1)
    except TimeoutError as exc:
        raise HTTPException(
            status_code=429,
            detail="다른 복원 요청을 처리 중입니다. 잠시 후 다시 시도해주세요.",
        ) from exc

    try:
        repaired = await run_in_threadpool(image_restorer.restore_image, original, payload)
    except image_restorer.RestorationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.public_message) from exc
    finally:
        _generation_slots.release()

    output = io.BytesIO()
    repaired.save(output, format="PNG", optimize=True)
    return Response(
        content=output.getvalue(),
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )
