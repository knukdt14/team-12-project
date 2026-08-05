"""단가표 조회 + 룰베이스 견적 계산.

data/단가표.json을 로드해서 (부위, 손상 종류, 심각도) 조합으로 수리 방식·금액 범위를 조회한다.
팀원(estimate_api.py) 구조와 통일: 단일 건 조회 + 실패 단계별 메시지.

YOLO가 탐지하는 16개 부위 클래스 중 일부(루프/필러/사이드스텝/방향지시등/후면 유리)는
단가표에 대응 항목이 없다. part 인자는 단가표 키(예: "front_bumper")를 직접 줘도 되고,
YOLO part_en(예: "front-bumper-dent")을 줘도 PART_KEY_MAP으로 자동 변환된다.
"""
import json
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "단가표.json"

# YOLO part_en(부위 클래스명) -> 단가표.json items 키 매핑.
# 단가표에 대응 항목이 없는 부위(루프/필러/사이드스텝/방향지시등/후면 유리)는
# 의도적으로 매핑하지 않는다 — 이 경우 "단가표에 등록되지 않은 부위" 메시지로 처리됨.
PART_KEY_MAP = {
    "front-bumper-dent": "front_bumper",
    "rear-bumper-dent": "rear_bumper",
    "doorouter-dent": "door",
    "fender-dent": "fender",
    "bonnet-dent": "hood",
    "boot-dent": "trunk",
    "quaterpanel-dent": "quarter_panel",
    "Sidemirror-Damage": "side_mirror",
    "Front-Windscreen-Damage": "windshield",
    "Headlight-Damage": "headlamp",
    "Taillight-Damage": "taillamp",
}

_price_table = None


def _load_price_table():
    global _price_table
    if _price_table is None:
        with open(DATA_PATH, encoding="utf-8") as f:
            _price_table = json.load(f)
    return _price_table


def _resolve_part_key(part: str, items: dict):
    """part가 단가표 키 자체면 그대로, YOLO part_en이면 PART_KEY_MAP으로 변환."""
    if part in items:
        return part
    return PART_KEY_MAP.get(part)


def estimate(part: str, damage_type: str, severity: str) -> dict:
    """(부위, 손상 종류, 심각도) 조합으로 견적을 조회해서 EstimateResponse 형태의 dict를 반환.

    실패 단계에 따라 서로 다른 메시지를 준다 (팀원 estimate_api.py와 동일한 3단계):
    1. 부위 자체가 단가표에 없음
    2. 부위는 있는데 해당 손상 유형이 없음
    3. 부위·유형은 있는데 해당 심각도 조합이 없음
    """
    table = _load_price_table()
    items = table.get("items", {})

    part_key = _resolve_part_key(part, items)
    if part_key is None or part_key not in items:
        return {"success": False, "message": "단가표에 등록되지 않은 부위입니다."}

    part_data = items[part_key]

    damage_key = damage_type.replace(" ", "_")  # "glass shatter" -> "glass_shatter"
    if damage_key not in part_data:
        return {"success": False, "message": "해당 부위의 손상 유형에 대한 견적 정보가 없습니다."}

    damage_data = part_data[damage_key]

    if severity not in damage_data:
        return {
            "success": False,
            "message": "단가표에 해당 조합이 없습니다. 정밀 견적 필요 — 정비소 방문을 권장합니다.",
        }

    result = damage_data[severity]

    return {
        "success": True,
        "part": part_key,
        "part_label": part_data.get("label", part_key),
        "damage_type": damage_type,
        "severity": severity,
        "repair_method": result["method"],
        "min_cost": int(result["min"]),
        "max_cost": int(result["max"]),
        "source": result.get("source"),
        "note": result.get("note"),
        "disclaimer": table.get("meta", {}).get("disclaimer"),
    }
