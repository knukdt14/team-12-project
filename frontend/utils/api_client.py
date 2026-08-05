"""backend(FastAPI)와의 HTTP 통신을 전담하는 얇은 클라이언트 모듈.

프론트/백엔드 분리 원칙: Streamlit(app.py)은 YOLO/ResNet18 등 모델을 직접
로드하지 않고, 이 모듈을 통해서만 backend를 호출한다. app.py는 UI/상태
관리와 (backend 연결 실패 시) 로컬 폴백 로직만 담당한다.

주소는 팀장님 main 정책을 유지해 용도별로 분리한다.
- 로컬 AI backend(main.py): 진단, 상담, LLM/RAG 상태, OpenAI 사진 복원
- Render 경량 backend(main_light.py): 견적, 주소 변환, 주변 정비소

YOLO/RAG는 Render 무료 티어 메모리 한계로 로컬에서 실행한다. 지도 라우터 자체를
로컬에서 시험할 때만 MAP_BACKEND_BASE_URL을 지정한다.
"""
import json
import mimetypes
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DIAGNOSE_BASE_URL = (
    os.environ.get("BACKEND_BASE_URL", "").strip() or "http://127.0.0.1:8000"
).rstrip("/")
MAP_BASE_URL = (
    os.environ.get("MAP_BACKEND_BASE_URL", "").strip()
    or os.environ.get("ESTIMATE_API_BASE_URL", "").strip()
    or "https://team-12-project.onrender.com"
).rstrip("/")

DIAGNOSE_URL = f"{DIAGNOSE_BASE_URL}/diagnose"
CHAT_URL = f"{DIAGNOSE_BASE_URL}/chat"
LLM_HEALTH_URL = f"{DIAGNOSE_BASE_URL}/health/llm"
REPAIR_PREVIEW_URL = f"{DIAGNOSE_BASE_URL}/repair-preview"
REPAIR_PREVIEW_HEALTH_URL = f"{DIAGNOSE_BASE_URL}/health/repair-preview"

ESTIMATE_URL = f"{MAP_BASE_URL}/estimate"
GEOCODE_URL = f"{MAP_BASE_URL}/geocode"
REPAIR_SHOPS_URL = f"{MAP_BASE_URL}/repair-shops"
REPAIR_API_TOKEN = os.environ.get("REPAIR_API_TOKEN", "").strip()


def call_diagnose_api(image_bytes, conf_threshold=0.3, filename="upload.jpg"):
    """backend의 /diagnose를 호출해 탐지 결과 리스트를 받아온다.

    성공 시 (results, None), 실패 시 (None, 에러메시지) 튜플을 반환한다.
    results의 각 원소는 {"part", "part_en", "damage_type", "damage_type_en",
    "severity", "confidence", "bbox"} 형태의 dict (schemas.Detection과 동일).
    """
    try:
        response = requests.post(
            DIAGNOSE_URL,
            files={"file": (filename, image_bytes, "image/jpeg")},
            data={"conf_threshold": conf_threshold},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["results"], None
    except requests.RequestException as e:
        return None, str(e)


def call_estimate_api(part, damage_type, severity, timeout=3):
    """backend의 /estimate를 호출해 견적 dict를 반환한다.

    연결 실패/HTTP 에러 시 requests.RequestException을 그대로 올린다 —
    호출부(app.py)에서 잡아서 단가표.json 로컬 폴백으로 넘어간다.
    """
    response = requests.post(
        ESTIMATE_URL,
        json={"part": part, "damage_type": damage_type, "severity": severity},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def call_geocode_api(address, timeout=10):
    """backend의 /geocode를 호출해 {"success", "lat", "lng", ...}를 반환한다."""
    response = requests.get(GEOCODE_URL, params={"address": address}, timeout=timeout)
    response.raise_for_status()
    return response.json()


def call_repair_shops_api(x, y, radius, query="자동차 정비소", timeout=10):
    """backend의 /repair-shops를 호출해 {"success", "shops", ...}를 반환한다."""
    response = requests.get(
        REPAIR_SHOPS_URL,
        params={"x": x, "y": y, "radius": radius, "query": query},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def call_chat_api(session_id, message, diagnosis_summary="", history=None, timeout=150):
    """backend의 /chat을 호출해 {"answer", "used_llm"}을 반환한다.

    diagnosis_summary: 이번 세션의 진단·견적 요약 문자열.
        backend에 세션 저장소가 없어서 프론트가 매 요청에 실어 보낸다.
        이 값이 없으면 LLM이 인용할 금액이 없어 "정비소에 방문하세요"만
        반복하는 무의미한 답변이 나온다.

    timeout이 긴 이유: LLM이 CPU 추론이라 첫 응답에 1~2분이 걸릴 수 있다.
    """
    response = requests.post(
        CHAT_URL,
        json={
            "session_id": session_id,
            "message": message,
            "diagnosis_summary": diagnosis_summary,
            "history": history or [],
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def call_llm_health(timeout=5):
    """backend의 /health/llm을 호출해 LLM 준비 상태를 반환. 실패하면 None.

    첫 기동 후 2~3분간은 모델(약 1.6GB) 다운로드 중이라 ready=False다.
    이 구간을 UI에 표시하지 않으면 폴백 답변을 버그로 오해하게 된다.
    """
    try:
        response = requests.get(LLM_HEALTH_URL, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def call_repair_preview_health(timeout=5):
    """OpenAI 복원 기능의 키 설정 여부와 모델 정보를 반환한다."""
    try:
        response = requests.get(REPAIR_PREVIEW_HEALTH_URL, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def call_repair_preview_api(image_bytes, detections, filename="vehicle.jpg", timeout=210):
    """backend에서 OpenAI 복원 예상 이미지를 생성해 PNG bytes로 반환한다."""
    content_type = mimetypes.guess_type(filename)[0] or "image/jpeg"
    if content_type not in {"image/jpeg", "image/png", "image/webp"}:
        content_type = "image/jpeg"
    response = requests.post(
        REPAIR_PREVIEW_URL,
        files={"file": (filename, image_bytes, content_type)},
        data={"detections_json": json.dumps(detections, ensure_ascii=False)},
        headers={"X-Repair-Token": REPAIR_API_TOKEN} if REPAIR_API_TOKEN else None,
        timeout=timeout,
    )
    if not response.ok:
        try:
            detail = response.json().get("detail")
        except (ValueError, AttributeError):
            detail = None
        raise RuntimeError(detail or "AI 복원 backend 요청에 실패했습니다.")
    if not response.content:
        raise RuntimeError("AI 복원 backend가 빈 이미지를 반환했습니다.")
    return response.content
