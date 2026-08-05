"""backend(FastAPI)와의 HTTP 통신을 전담하는 얇은 클라이언트 모듈.

프론트/백엔드 분리 원칙: Streamlit(app.py)은 YOLO/ResNet18 등 모델을 직접
로드하지 않고, 이 모듈을 통해서만 backend를 호출한다. app.py는 UI/상태
관리와 (backend 연결 실패 시) 로컬 폴백 로직만 담당한다.

로컬 개발 기준 URL이며, 배포 시 BACKEND_BASE_URL 환경변수로 덮어쓸 수 있다.
"""
import os

import requests

BASE_URL = os.environ.get(
    "BACKEND_BASE_URL", os.environ.get("ESTIMATE_API_BASE_URL", "http://127.0.0.1:8000")
)

DIAGNOSE_URL = f"{BASE_URL}/diagnose"
ESTIMATE_URL = f"{BASE_URL}/estimate"
GEOCODE_URL = f"{BASE_URL}/geocode"
REPAIR_SHOPS_URL = f"{BASE_URL}/repair-shops"
CHAT_URL = f"{BASE_URL}/chat"


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


def call_chat_api(session_id, message, timeout=30):
    """backend의 /chat을 호출해 {"answer": ...}를 반환한다.

    app.py의 "AI 상담" 탭은 아직 이 함수를 쓰지 않고 하드코딩된 답변을 쓰고
    있음 — /chat 연동은 별도 작업으로 남아있음 (RAGS.py 통합 여부 포함).
    """
    response = requests.post(
        CHAT_URL,
        json={"session_id": session_id, "message": message},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()
