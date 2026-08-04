"""
FastAPI Backend — 차량 수리비 견적 API

실행:
    uvicorn estimate_api:app --reload --port 8000

테스트:
    http://127.0.0.1:8000/docs
"""

import json
from pathlib import Path
import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent
PRICE_TABLE_PATH = BASE_DIR / "단가표.json"



app = FastAPI(
    title="CarDoc Estimate API",
    version="1.0.0",
)


with open(PRICE_TABLE_PATH, "r", encoding="utf-8") as f:
    PRICE_DATA = json.load(f)


class EstimateRequest(BaseModel):
    part: str
    damage_type: str
    severity: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/estimate")
def estimate_repair(req: EstimateRequest):
    items = PRICE_DATA.get("items", {})

    if req.part not in items:
        return {
            "success": False,
            "message": "단가표에 등록되지 않은 부위입니다.",
        }

    part_data = items[req.part]

    if req.damage_type not in part_data:
        return {
            "success": False,
            "message": "해당 부위의 손상 유형에 대한 견적 정보가 없습니다.",
        }

    damage_data = part_data[req.damage_type]

    if req.severity not in damage_data:
        return {
            "success": False,
            "message": (
                "단가표에 해당 조합이 없습니다. "
                "정밀 견적 필요 — 정비소 방문을 권장합니다."
            ),
        }

    result = damage_data[req.severity]

    return {
        "success": True,
        "part": req.part,
        "part_label": part_data["label"],
        "damage_type": req.damage_type,
        "severity": req.severity,
        "repair_method": result["method"],
        "min_cost": int(result["min"]),
        "max_cost": int(result["max"]),
        "source": result.get("source"),
        "note": result.get("note"),
        "disclaimer": PRICE_DATA["meta"]["disclaimer"],
    }

KAKAO_REST_API_KEY = os.getenv("KAKAO_REST_API_KEY")

@app.get("/repair-shops")
def get_repair_shops(
    x: float,              # 경도
    y: float,              # 위도
    radius: int = 3000,
    query: str = "자동차 정비소"
):
    if not KAKAO_REST_API_KEY:
        return {
            "success": False,
            "message": "Kakao REST API Key가 설정되지 않았습니다."
        }

    url = "https://dapi.kakao.com/v2/local/search/keyword.json"

    headers = {
        "Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"
    }

    params = {
        "query": query,
        "x": x,
        "y": y,
        "radius": radius,
        "sort": "distance"
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        shops = []

        for place in data.get("documents", []):
            shops.append({
                "name": place["place_name"],
                "address": (
                    place["road_address_name"]
                    or place["address_name"]
                ),
                "phone": place.get("phone", ""),
                "distance": place.get("distance", ""),
                "lat": float(place["y"]),
                "lng": float(place["x"]),
                "place_url": place.get("place_url", "")
            })

        return {
            "success": True,
            "count": len(shops),
            "shops": shops
        }

    except requests.RequestException as e:
        return {
            "success": False,
            "message": f"카카오 지도 API 호출 실패: {str(e)}"
        }

@app.get("/geocode")
def geocode(address: str):

    if not KAKAO_REST_API_KEY:
        return {
            "success": False,
            "message": "Kakao REST API Key가 설정되지 않았습니다."
        }

    url = "https://dapi.kakao.com/v2/local/search/address.json"

    headers = {
        "Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"
    }

    params = {
        "query": address
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=10
        )

        # 카카오에서 실제로 어떤 응답을 줬는지 확인
        print("KAKAO STATUS:", response.status_code)
        print("KAKAO BODY:", response.text)

        # 200이 아니면 카카오 응답 내용을 그대로 반환
        if response.status_code != 200:
            return {
                "success": False,
                "status_code": response.status_code,
                "message": "카카오 주소 검색 API 호출에 실패했습니다.",
                "kakao_response": response.text
            }

        data = response.json()

        documents = data.get("documents", [])

        if not documents:
            return {
                "success": False,
                "message": "입력한 주소를 찾을 수 없습니다."
            }

        item = documents[0]

        if "x" not in item or "y" not in item:
            return {
                "success": False,
                "message": "카카오 응답에 좌표 정보가 없습니다."
            }

        return {
            "success": True,
            "address": item.get("address_name", address),
            "lat": float(item["y"]),
            "lng": float(item["x"])
        }

    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "message": f"카카오 API 통신 오류: {str(e)}"
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"주소 변환 처리 중 오류: {type(e).__name__}: {str(e)}"
        }