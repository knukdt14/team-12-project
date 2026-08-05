"""GET /repair-shops, GET /geocode — 카카오 로컬 API 기반 정비소 검색.

팀원(estimate_api.py, KBU 브랜치)의 구현을 backend/ 구조로 이식.
카카오 REST API 키가 필요 (.env의 KAKAO_REST_API_KEY, main.py에서 load_dotenv()로 로드).
"""
import os

import requests
from fastapi import APIRouter

from schemas import GeocodeResponse, RepairShop, RepairShopsResponse

router = APIRouter()

KAKAO_REST_API_KEY = os.getenv("KAKAO_REST_API_KEY")


@router.get("/repair-shops", response_model=RepairShopsResponse)
def get_repair_shops(
    x: float,  # 경도
    y: float,  # 위도
    radius: int = 3000,
    query: str = "자동차 정비소",
):
    if not KAKAO_REST_API_KEY:
        return RepairShopsResponse(success=False, message="Kakao REST API Key가 설정되지 않았습니다.")

    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"}
    params = {"query": query, "x": x, "y": y, "radius": radius, "sort": "distance"}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        shops = [
            RepairShop(
                name=place["place_name"],
                address=place["road_address_name"] or place["address_name"],
                phone=place.get("phone", ""),
                distance=place.get("distance", ""),
                lat=float(place["y"]),
                lng=float(place["x"]),
                place_url=place.get("place_url", ""),
            )
            for place in data.get("documents", [])
        ]

        return RepairShopsResponse(success=True, count=len(shops), shops=shops)

    except requests.RequestException as e:
        return RepairShopsResponse(success=False, message=f"카카오 지도 API 호출 실패: {str(e)}")


@router.get("/geocode", response_model=GeocodeResponse)
def geocode(address: str):
    if not KAKAO_REST_API_KEY:
        return GeocodeResponse(success=False, message="Kakao REST API Key가 설정되지 않았습니다.")

    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"}
    params = {"query": address}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)

        if response.status_code != 200:
            return GeocodeResponse(
                success=False,
                message=f"카카오 주소 검색 API 호출에 실패했습니다. (status={response.status_code})",
            )

        data = response.json()
        documents = data.get("documents", [])
        if not documents:
            return GeocodeResponse(success=False, message="입력한 주소를 찾을 수 없습니다.")

        item = documents[0]
        if "x" not in item or "y" not in item:
            return GeocodeResponse(success=False, message="카카오 응답에 좌표 정보가 없습니다.")

        return GeocodeResponse(
            success=True,
            address=item.get("address_name", address),
            lat=float(item["y"]),
            lng=float(item["x"]),
        )

    except requests.exceptions.RequestException as e:
        return GeocodeResponse(success=False, message=f"카카오 API 통신 오류: {str(e)}")
    except Exception as e:
        return GeocodeResponse(success=False, message=f"주소 변환 처리 중 오류: {type(e).__name__}: {str(e)}")
