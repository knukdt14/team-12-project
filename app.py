"""
Streamlit 데모 앱 — YOLO 차체 외관 불량 검출
실행: streamlit run app.py
"""
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18
from ultralytics import YOLO
from pathlib import Path

from src.preprocessing import draw_results, resize_for_display

# 실행 시 작업 디렉토리(cwd)가 프로젝트 폴더가 아니어도 항상 이 스크립트 기준 경로를 쓰도록 함
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "runs/detect/runs/train_20260804_1124/weights/best.pt"  # YOLO11n, 16종(Bodypanel-Dent 제거) 재학습, test mAP50 0.887
FALLBACK   = "yolo11n.pt"   # 학습 전 테스트용 기본 모델 (ultralytics가 이름만으로 자동 다운로드)

# 2단계 손상 종류 분류기 (train_damage_type.py로 학습, test acc 0.8150)
DAMAGE_TYPE_MODEL_PATH = BASE_DIR / "runs/damage_type_classifier/best.pt"
DAMAGE_TYPE_IMG_SIZE = 224  # train_damage_type.py와 동일
DAMAGE_TYPE_PAD_RATIO = 0.15  # 학습 때 crop 여백과 동일하게 맞춤 (scripts/build_damage_type_crops.py)

DAMAGE_TYPE_KOREAN = {
    "crack": "균열",
    "dent": "찌그러짐",
    "glass shatter": "유리 파손",
    "lamp broken": "램프 파손",
    "scratch": "스크래치",
    "tire flat": "타이어 펑크",
}

# 부위명 영→한 표시 매핑 (DEFECT_CLASSES.md 기준). 손상 종류(찌그러짐/파손 등)는 2단계 분류기가 "손상 종류" 열로
# 별도 표시하므로, 여기서는 부위명만 남기고 중복되는 손상 상태 표현은 뺌.
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

st.set_page_config(page_title="차체 도장 불량 검출", layout="wide")
st.title("차체 도장·외관 불량 자동 검출 시스템")
st.caption("Roboflow Car Defect Dataset + YOLO11 + OpenCV")

@st.cache_resource
def load_model():
    path = MODEL_PATH if Path(MODEL_PATH).exists() else FALLBACK
    return YOLO(path), path


@st.cache_resource
def load_damage_type_model():
    """2단계 손상 종류 분류기 로드. 없으면 (None, None) 반환 — 있을 때만 종류 열 표시."""
    if not Path(DAMAGE_TYPE_MODEL_PATH).exists():
        return None, None
    ckpt = torch.load(DAMAGE_TYPE_MODEL_PATH, map_location="cpu", weights_only=False)
    class_names = ckpt["class_names"]
    clf = resnet18(weights=None)
    clf.fc = torch.nn.Linear(clf.fc.in_features, len(class_names))
    clf.load_state_dict(ckpt["model_state"])
    clf.eval()
    return clf, class_names


DAMAGE_TYPE_TF = transforms.Compose([
    transforms.Resize((DAMAGE_TYPE_IMG_SIZE, DAMAGE_TYPE_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def classify_damage_type(img_bgr, box_xyxy, clf, class_names):
    """박스 영역을 여백 포함해 crop한 뒤 손상 종류를 분류해 한글 라벨 반환"""
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
        pred = clf(tensor).argmax(1).item()
    cls_name = class_names[pred]
    return DAMAGE_TYPE_KOREAN.get(cls_name, cls_name)


model, used_path = load_model()
st.sidebar.info(f"사용 모델: `{used_path}`")

damage_type_clf, damage_type_classes = load_damage_type_model()
if damage_type_clf is not None:
    st.sidebar.info("손상 종류 분류기: 활성화 (dent/scratch/crack 등 구분)")

conf_threshold = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.3, 0.05)

uploaded = st.file_uploader("차체 이미지 업로드 (jpg / png)", type=["jpg", "jpeg", "png"])

if uploaded:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("이미지를 읽을 수 없습니다. 유효한 JPG/PNG 파일을 업로드하세요.")
        st.stop()

    MIN_SIZE = 100  # px — 이보다 작으면 오탐 위험이 커서 차단
    h, w = img_bgr.shape[:2]
    if h < MIN_SIZE or w < MIN_SIZE:
        st.error(f"이미지가 너무 작습니다 ({w}x{h}). 최소 {MIN_SIZE}x{MIN_SIZE} 이상 업로드하세요.")
        st.stop()

    results = model.predict(img_bgr, conf=conf_threshold, verbose=False)
    vis = draw_results(img_bgr, results)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("원본")
        st.image(cv2.cvtColor(resize_for_display(img_bgr), cv2.COLOR_BGR2RGB), width="stretch")
    with col2:
        st.subheader("검출 결과")
        st.image(cv2.cvtColor(resize_for_display(vis), cv2.COLOR_BGR2RGB), width="stretch")

    # 탐지 결과 요약
    boxes = results[0].boxes
    st.divider()
    if len(boxes) == 0:
        st.success("불량 미검출 — 정상 판정")
    else:
        defects = [KOREAN_NAMES.get(results[0].names[int(b.cls[0])], results[0].names[int(b.cls[0])]) for b in boxes]
        confs   = [float(b.conf[0]) for b in boxes]
        st.error(f"불량 {len(boxes)}건 검출")

        data = {"부위": defects, "신뢰도": [f"{c:.2%}" for c in confs]}
        if damage_type_clf is not None:
            data["손상 종류"] = [
                classify_damage_type(img_bgr, tuple(map(float, b.xyxy[0])), damage_type_clf, damage_type_classes)
                for b in boxes
            ]
        df = pd.DataFrame(data)
        st.dataframe(df, width="stretch")
