"""YOLO(부위 탐지) + ResNet18(손상 종류 분류) 추론 래퍼.

프로젝트 루트 app.py의 모델 로딩/추론 로직을 FastAPI에서 쓸 수 있게 옮긴 버전.
Streamlit의 @st.cache_resource 대신, 모듈 로드 시점에 한 번만 로딩해서
전역 변수로 캐싱한다(load_models()를 앱 시작 시 or 첫 요청 시 1회 호출).
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18
from ultralytics import YOLO

# backend/services/detector.py -> backend/ -> 프로젝트 루트
BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = BASE_DIR / "runs/detect/runs/train_20260804_1124/weights/best.pt"  # YOLO11n, 16종, test mAP50 0.887
FALLBACK = "yolo11n.pt"  # 학습 전 테스트용 기본 모델 (ultralytics가 이름만으로 자동 다운로드)

DAMAGE_TYPE_MODEL_PATH = BASE_DIR / "runs/damage_type_classifier/best.pt"  # ResNet18, test acc 0.8150
DAMAGE_TYPE_IMG_SIZE = 224
DAMAGE_TYPE_PAD_RATIO = 0.15

DAMAGE_TYPE_KOREAN = {
    "crack": "균열",
    "dent": "찌그러짐",
    "glass shatter": "유리 파손",
    "lamp broken": "램프 파손",
    "scratch": "스크래치",
    "tire flat": "타이어 펑크",
}

# 부위별로 물리적으로 가능한 손상 종류.
#
# 왜 필요한가:
#   1단계 YOLO(부위)와 2단계 ResNet(손상 종류)이 서로를 모른 채 독립적으로
#   예측하기 때문에, "전방 범퍼 + 램프 파손" 같은 불가능한 조합이 나온다.
#   이런 조합은 단가표에 없으므로 견적이 실패하고, 챗봇은 금액을 인용하지
#   못해 "정비소에 방문하세요"만 반복하는 무의미한 답변을 하게 된다.
#
# 어떻게 고치는가:
#   분류기의 출력 logit에서 해당 부위에 불가능한 클래스를 -inf로 만든 뒤
#   argmax를 취한다(제약 조건부 예측). 정확도가 떨어지는 게 아니라, 애초에
#   틀릴 수밖에 없는 선택지를 후보에서 빼는 것이라 오히려 올라간다.
#
# 값의 근거: data/단가표.json items의 각 부위가 정의한 손상 종류와 일치시킨다.
#   단가표에 없는 부위(루프/필러/사이드스텝 등)는 패널로 간주해 scratch/dent만 허용.
PART_VALID_DAMAGES = {
    # 범퍼·패널 — 긁힘/찌그러짐/균열
    "front-bumper-dent": {"scratch", "dent", "crack"},
    "rear-bumper-dent": {"scratch", "dent", "crack"},
    "doorouter-dent": {"scratch", "dent", "crack"},
    "fender-dent": {"scratch", "dent"},
    "bonnet-dent": {"scratch", "dent"},
    "boot-dent": {"scratch", "dent"},
    "quaterpanel-dent": {"scratch", "dent"},
    "roof-dent": {"scratch", "dent"},
    "pillar-dent": {"scratch", "dent"},
    "RunningBoard-Dent": {"scratch", "dent"},
    # 구버전 17종 모델 클래스. 현재 배포 모델(16종)에는 없지만 표는 맞춰둔다.
    "Bodypanel-Dent": {"scratch", "dent"},
    # 유리 — 깨짐/금
    "Front-Windscreen-Damage": {"crack", "glass shatter"},
    "Rear-windscreen-Damage": {"crack", "glass shatter"},
    # 램프 — 파손/금
    "Headlight-Damage": {"crack", "lamp broken"},
    "Taillight-Damage": {"crack", "lamp broken"},
    "Signlight-Damage": {"crack", "lamp broken"},
    # 사이드미러 — 커버 긁힘 / 미러 깨짐
    "Sidemirror-Damage": {"scratch", "crack"},
}

KOREAN_NAMES = {
    "Front-Windscreen-Damage": "전면 유리",
    "Headlight-Damage": "전조등",
    "Rear-windscreen-Damage": "후면 유리",
    "RunningBoard-Dent": "사이드스텝",
    "Sidemirror-Damage": "사이드미러",
    "Signlight-Damage": "방향지시등",
    "Taillight-Damage": "후미등",
    "bonnet-dent": "보닛",
    "boot-dent": "트렁크",
    "doorouter-dent": "도어 외판",
    "fender-dent": "펜더",
    "front-bumper-dent": "전방 범퍼",
    "pillar-dent": "필러(기둥)",
    "quaterpanel-dent": "쿼터패널",
    "rear-bumper-dent": "후방 범퍼",
    "roof-dent": "루프",
}

# severity 규칙: 박스 면적이 전체 이미지에서 차지하는 비율 기준 (임시 휴리스틱).
# 실측 데이터로 검증된 기준은 아니라서, 데모 결과 보고 임계값(SEVERITY_*_RATIO)은 조정 가능.
SEVERITY_SMALL_RATIO = 0.05  # 이 비율 미만 → minor
SEVERITY_LARGE_RATIO = 0.15  # 이 비율 이상 → severe (그 사이는 moderate)

MIN_SIZE = 100  # px — 이보다 작으면 오탐 위험이 커서 차단 (app.py와 동일 기준)

DAMAGE_TYPE_TF = transforms.Compose([
    transforms.Resize((DAMAGE_TYPE_IMG_SIZE, DAMAGE_TYPE_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_yolo_model = None
_yolo_model_path = None
_damage_type_clf = None
_damage_type_classes = None


def load_models():
    """서버 시작 시(또는 첫 요청 시) 한 번만 호출해서 모델을 전역에 캐싱한다."""
    global _yolo_model, _yolo_model_path, _damage_type_clf, _damage_type_classes

    if _yolo_model is None:
        path = str(MODEL_PATH) if MODEL_PATH.exists() else FALLBACK
        _yolo_model = YOLO(path)
        _yolo_model_path = path

    if _damage_type_clf is None and DAMAGE_TYPE_MODEL_PATH.exists():
        ckpt = torch.load(DAMAGE_TYPE_MODEL_PATH, map_location="cpu", weights_only=False)
        class_names = ckpt["class_names"]
        clf = resnet18(weights=None)
        clf.fc = torch.nn.Linear(clf.fc.in_features, len(class_names))
        clf.load_state_dict(ckpt["model_state"])
        clf.eval()
        _damage_type_clf = clf
        _damage_type_classes = class_names


def _compute_severity(bbox, img_w, img_h):
    """박스 면적이 이미지 전체에서 차지하는 비율로 심각도를 추정.

    < 5%: minor, 5~15%: moderate, >= 15%: severe.
    손상 부위가 클수록 심각하다는 단순 가정에 기반한 임시 규칙.
    라벨(minor/moderate/severe)은 data/단가표.json의 severity_def와 맞춘 것.
    """
    x1, y1, x2, y2 = bbox
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    image_area = img_w * img_h
    if image_area <= 0:
        return "unknown"

    ratio = box_area / image_area
    if ratio < SEVERITY_SMALL_RATIO:
        return "minor"
    if ratio < SEVERITY_LARGE_RATIO:
        return "moderate"
    return "severe"


def _classify_damage_type(img_bgr, box_xyxy, part_en=None):
    """박스 영역을 여백 포함해 crop한 뒤 손상 종류를 분류해 원본 영문 클래스명 반환.

    part_en이 주어지면 PART_VALID_DAMAGES로 후보를 제한한다.
    예) 전방 범퍼에서는 scratch/dent/crack 중에서만 고른다 — 램프 파손이
        1순위로 나와도 물리적으로 불가능하므로 후보에서 제외하고 2순위를 택한다.
        이 제약이 없으면 단가표 조회가 실패해 견적·챗봇 답변이 모두 무너진다.

    한글 표시는 호출부에서 DAMAGE_TYPE_KOREAN으로 변환한다 (part_en/damage_type_en을
    단가표.json 조회에 그대로 쓰기 위해 원본 영문값을 유지).
    """
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    bw, bh = x2 - x1, y2 - y1
    x1p = max(0, int(x1 - bw * DAMAGE_TYPE_PAD_RATIO))
    y1p = max(0, int(y1 - bh * DAMAGE_TYPE_PAD_RATIO))
    x2p = min(w, int(x2 + bw * DAMAGE_TYPE_PAD_RATIO))
    y2p = min(h, int(y2 + bh * DAMAGE_TYPE_PAD_RATIO))
    if x2p - x1p < 5 or y2p - y1p < 5:
        return "-"

    crop_rgb = cv2.cvtColor(img_bgr[y1p:y2p, x1p:x2p], cv2.COLOR_BGR2RGB)
    tensor = DAMAGE_TYPE_TF(Image.fromarray(crop_rgb)).unsqueeze(0)
    with torch.no_grad():
        logits = _damage_type_clf(tensor)[0]

    allowed = PART_VALID_DAMAGES.get(part_en) if part_en else None
    if allowed:
        # 허용되지 않은 클래스의 점수를 -inf로 만들어 argmax 후보에서 제외.
        # 학습된 클래스 중 허용 목록에 하나도 없으면(매핑 실수 등) 제약을 포기하고
        # 원래 예측을 그대로 쓴다 — 전부 -inf가 되어 엉뚱한 값이 나오는 것을 방지.
        mask = torch.tensor(
            [c in allowed for c in _damage_type_classes], dtype=torch.bool
        )
        if mask.any():
            logits = logits.masked_fill(~mask, float("-inf"))

    return _damage_type_classes[int(logits.argmax().item())]


def detect(image_bytes: bytes, conf_threshold: float = 0.3):
    """이미지 바이트를 받아 부위 탐지 + 손상 종류 분류 결과 리스트를 반환.

    반환 형식은 schemas.Detection과 1:1로 맞춘다:
    [{"part", "part_en", "damage_type", "damage_type_en", "severity", "confidence", "bbox"}, ...]
    유효하지 않은 이미지/너무 작은 이미지는 ValueError를 던진다 (라우터에서 400으로 변환).
    """
    load_models()

    file_bytes = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("이미지를 읽을 수 없습니다. 유효한 JPG/PNG 파일인지 확인하세요.")

    h, w = img_bgr.shape[:2]
    if h < MIN_SIZE or w < MIN_SIZE:
        raise ValueError(f"이미지가 너무 작습니다 ({w}x{h}). 최소 {MIN_SIZE}x{MIN_SIZE} 이상이어야 합니다.")

    results = _yolo_model.predict(img_bgr, conf=conf_threshold, verbose=False)
    boxes = results[0].boxes
    names = results[0].names

    detections = []
    for b in boxes:
        part_en = names[int(b.cls[0])]
        part = KOREAN_NAMES.get(part_en, part_en)
        bbox = list(map(float, b.xyxy[0]))
        confidence = float(b.conf[0])

        damage_type_en = "-"
        if _damage_type_clf is not None:
            damage_type_en = _classify_damage_type(img_bgr, tuple(bbox), part_en)
        damage_type = DAMAGE_TYPE_KOREAN.get(damage_type_en, damage_type_en)

        detections.append({
            "part": part,
            "part_en": part_en,
            "damage_type": damage_type,
            "damage_type_en": damage_type_en,
            "severity": _compute_severity(bbox, w, h),
            "confidence": confidence,
            "bbox": bbox,
        })

    return detections
