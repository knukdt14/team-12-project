"""단가표 조회 + 룰베이스 견적 계산.

data/단가표.json을 로드해서 (부위, 손상 종류, 심각도) 조합으로 수리 방식·금액 범위를 조회한다.

YOLO가 탐지하는 16개 부위 클래스 중 일부(루프/필러/사이드스텝/방향지시등/후면 유리)는
단가표에 대응 항목이 없다 — 매핑되지 않은 부위나, 매핑은 되지만 해당 (유형, 심각도)
조합이 단가표에 없는 경우 모두 llm_guardrails.rule2("단가표에 없는 조합은
'정밀 견적 필요 — 정비소 방문 권장'으로 응답한다")를 그대로 따른다.
"""
import json
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "단가표.json"

# YOLO part_en(부위 클래스명) -> 단가표.json items 키 매핑.
# 단가표에 대응 항목이 없는 부위(루프/필러/사이드스텝/방향지시등/후면 유리)는
# 의도적으로 매핑하지 않는다 — lookup()에서 NO_PRICE_MESSAGE로 처리됨.
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

NO_PRICE_MESSAGE = "정밀 견적 필요 — 정비소 방문 권장"

_price_table = None


def _load_price_table():
    global _price_table
    if _price_table is None:
        with open(DATA_PATH, encoding="utf-8") as f:
            _price_table = json.load(f)
    return _price_table


def lookup(part_en: str, damage_type_en: str, severity: str):
    """(part_en, damage_type_en, severity) 조합으로 단가표를 조회.

    단가표에 없는 조합이면 None을 반환한다 (호출부에서 NO_PRICE_MESSAGE로 처리).
    반환값: {"part_label", "method", "min_cost", "max_cost"} 또는 None
    """
    table = _load_price_table()

    part_key = PART_KEY_MAP.get(part_en)
    if part_key is None:
        return None

    item = table["items"].get(part_key)
    if item is None:
        return None

    damage_key = damage_type_en.replace(" ", "_")  # "glass shatter" -> "glass_shatter"
    damage_entry = item.get(damage_key)
    if damage_entry is None:
        return None

    price = damage_entry.get(severity)
    if price is None:
        return None

    return {
        "part_label": item.get("label", part_key),
        "method": price["method"],
        "min_cost": price["min"],
        "max_cost": price["max"],
    }
