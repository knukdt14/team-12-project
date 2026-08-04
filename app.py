"""
Streamlit Frontend — AI 차량 파손 진단 + 예상 수리비 + 상담 UI
실행:
    streamlit run app.py

백엔드가 켜져 있으면 /estimate API를 호출하고,
꺼져 있으면 단가표.json을 직접 읽는 fallback 로직으로 동작합니다.
"""

import base64
import io
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
import torch

import folium
from streamlit_folium import st_folium

from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18
from ultralytics import YOLO

from src.preprocessing import draw_results, resize_for_display
from repair_inpaint import (
    RepairConfig,
    generate_repaired_image_full,
    load_inpaint_pipeline,
)


# ---------------------------------------------------------
# 기본 경로
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "runs/detect/runs/train_20260804_1124/weights/best.pt"  # YOLO11n, 16종(Bodypanel-Dent 제거) 재학습, test mAP50 0.887
FALLBACK = "yolo11n.pt"

DAMAGE_TYPE_MODEL_PATH = BASE_DIR / "runs/damage_type_classifier/best.pt"
DAMAGE_TYPE_IMG_SIZE = 224
DAMAGE_TYPE_PAD_RATIO = 0.15

PRICE_TABLE_PATH = BASE_DIR / "단가표.json"
LOGO_PATH = BASE_DIR / "ajin_logo.png"

# 대시보드 집계용 진단 이력 누적 로그 (견적 생성 시마다 1행씩 append)
DIAGNOSIS_LOG_PATH = BASE_DIR / "diagnosis_log.csv"

# 교수님 요청용 "복원 예상" 이미지.
# 파일이 없으면 안내 박스만 표시합니다.
REPAIRED_SAMPLE_PATH = BASE_DIR / "assets/bokgo.jpg"

# 로컬 개발 기준. Docker에서는 환경변수로 바꾸는 것을 권장.
ESTIMATE_API_URL = "http://127.0.0.1:8000/estimate"


# ---------------------------------------------------------
# 표시/내부 코드 매핑
# ---------------------------------------------------------
DAMAGE_TYPE_KOREAN = {
    "crack": "균열",
    "dent": "찌그러짐",
    "glass shatter": "유리 파손",
    "lamp broken": "램프 파손",
    "scratch": "스크래치",
    "tire flat": "타이어 펑크",
}

# 분류기 class name -> 단가표 키
DAMAGE_CODE_MAP = {
    "crack": "crack",
    "dent": "dent",
    "glass shatter": "glass_shatter",
    "lamp broken": "lamp_broken",
    "scratch": "scratch",
    "tire flat": "tire_flat",
}

KOREAN_NAMES = {
    "Bodypanel-Dent": "차체 패널",
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

# YOLO class -> 단가표.json 부위 코드
PART_CODE_MAP = {
    "Bodypanel-Dent": None,
    "Front-Windscreen-Damage": "windshield",
    "Headlight-Damage": "headlamp",
    "Rear-windscreen-Damage": None,
    "RunningBoard-Dent": None,
    "Sidemirror-Damage": "side_mirror",
    "Signlight-Damage": None,
    "Taillight-Damage": "taillamp",
    "bonnet-dent": "hood",
    "boot-dent": "trunk",
    "doorouter-dent": "door",
    "fender-dent": "fender",
    "front-bumper-dent": "front_bumper",
    "pillar-dent": None,
    "quaterpanel-dent": "quarter_panel",
    "rear-bumper-dent": "rear_bumper",
    "roof-dent": None,
}

SEVERITY_KO = {
    "minor": "경미",
    "moderate": "중간",
    "severe": "심각",
}

SEVERITY_EN = {
    "경미": "minor",
    "중간": "moderate",
    "심각": "severe",
}


# ---------------------------------------------------------
# Streamlit 설정 / CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI 차량 파손 진단",
    page_icon="🚗",
    layout="wide",
)

st.markdown(
    """
<style>
    .block-container {
        padding-top: 2.0rem;
        padding-bottom: 4rem;
        max-width: 1500px;
    }
    .small-muted {
        color:#6B7280;
        font-size:14px;
    }
    .result-box {
        border:1px solid #E5E7EB;
        border-radius:16px;
        padding:20px 22px;
        background:#FFFFFF;
        min-height:130px;
    }
    .result-label {
        color:#6B7280;
        font-size:14px;
        margin-bottom:8px;
    }
    .result-value {
        font-size:24px;
        font-weight:800;
        line-height:1.25;
    }
    .price-value {
        font-size:30px;
        font-weight:900;
        line-height:1.25;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 모델 로드
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    path = MODEL_PATH if MODEL_PATH.exists() else FALLBACK
    return YOLO(path), path


@st.cache_resource
def load_damage_type_model():
    if not DAMAGE_TYPE_MODEL_PATH.exists():
        return None, None

    ckpt = torch.load(
        DAMAGE_TYPE_MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )
    class_names = ckpt["class_names"]

    clf = resnet18(weights=None)
    clf.fc = torch.nn.Linear(clf.fc.in_features, len(class_names))
    clf.load_state_dict(ckpt["model_state"])
    clf.eval()

    return clf, class_names



@st.cache_resource(show_spinner=False)
def load_repair_pipeline():
    """복원 버튼을 처음 눌렀을 때만 생성형 인페인팅 모델을 로드.

    sequential_offload=True: VRAM을 가장 아껴 쓰는 모드(느리지만 안전).
    GPU VRAM이 16GB 이상으로 넉넉하고 속도가 더 중요하면
    sequential_offload=False 로 바꿔보세요(대신 YOLO 등 다른 모델과
    VRAM을 나눠 쓸 때 OOM 위험이 커집니다).
    """
    return load_inpaint_pipeline(low_vram=True, sequential_offload=True)


DAMAGE_TYPE_TF = transforms.Compose(
    [
        transforms.Resize((DAMAGE_TYPE_IMG_SIZE, DAMAGE_TYPE_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def classify_damage_type_raw(img_bgr, box_xyxy, clf, class_names):
    """손상 종류 분류기의 raw class name을 반환."""
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    bw, bh = x2 - x1, y2 - y1

    x1p = max(0, int(x1 - bw * DAMAGE_TYPE_PAD_RATIO))
    y1p = max(0, int(y1 - bh * DAMAGE_TYPE_PAD_RATIO))
    x2p = min(w, int(x2 + bw * DAMAGE_TYPE_PAD_RATIO))
    y2p = min(h, int(y2 + bh * DAMAGE_TYPE_PAD_RATIO))

    if x2p - x1p < 5 or y2p - y1p < 5:
        return None

    crop_rgb = cv2.cvtColor(
        img_bgr[y1p:y2p, x1p:x2p],
        cv2.COLOR_BGR2RGB,
    )

    tensor = DAMAGE_TYPE_TF(Image.fromarray(crop_rgb)).unsqueeze(0)

    with torch.no_grad():
        pred = clf(tensor).argmax(1).item()

    return class_names[pred]


# ---------------------------------------------------------
# 견적 관련
# ---------------------------------------------------------
def load_price_table_local():
    if not PRICE_TABLE_PATH.exists():
        return None
    with open(PRICE_TABLE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def estimate_local(part, damage_type, severity):
    """FastAPI가 꺼져 있어도 Streamlit 단독 시연이 가능하도록 fallback."""
    data = load_price_table_local()

    if not data:
        return {
            "success": False,
            "message": "단가표.json 파일을 찾을 수 없습니다.",
        }

    items = data.get("items", {})

    try:
        part_data = items[part]
        result = part_data[damage_type][severity]
    except (KeyError, TypeError):
        return {
            "success": False,
            "message": "해당 부위·손상 유형·심각도의 기준 단가가 없습니다. 정밀 견적이 필요합니다.",
        }

    return {
        "success": True,
        "part": part,
        "part_label": part_data["label"],
        "damage_type": damage_type,
        "severity": severity,
        "repair_method": result["method"],
        "min_cost": int(result["min"]),
        "max_cost": int(result["max"]),
        "source": result.get("source"),
        "note": result.get("note"),
        "disclaimer": data["meta"]["disclaimer"],
        "via": "local",
    }


def get_repair_estimate(part, damage_type, severity):
    """
    1) FastAPI /estimate 호출
    2) 연결 실패 시 단가표.json 직접 조회
    """
    payload = {
        "part": part,
        "damage_type": damage_type,
        "severity": severity,
    }

    try:
        response = requests.post(
            ESTIMATE_API_URL,
            json=payload,
            timeout=3,
        )
        response.raise_for_status()
        result = response.json()
        result["via"] = "api"
        return result

    except requests.RequestException:
        return estimate_local(part, damage_type, severity)


def source_names_from_ids(source_ids):
    data = load_price_table_local()
    if not data or not source_ids:
        return []

    source_map = {
        src["id"]: src["name"]
        for src in data["meta"].get("sources", [])
    }

    ids = [s.strip() for s in source_ids.split(",")]
    return [source_map.get(s, s) for s in ids]


# ---------------------------------------------------------
# 견적서 다운로드 (CSV/Excel) + 대시보드용 이력 로깅
# ---------------------------------------------------------
DIAGNOSIS_LOG_COLUMNS = [
    "timestamp",
    "part_code",
    "part_label",
    "damage_code",
    "damage_label",
    "confidence",
    "severity",
    "repair_method",
    "min_cost",
    "max_cost",
]


def log_diagnosis(primary, estimate):
    """진단+견적 1건을 로컬 CSV에 누적 저장 (대시보드 집계용)."""
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "part_code": primary.get("part_code"),
        "part_label": primary.get("part_label"),
        "damage_code": primary.get("damage_code"),
        "damage_label": primary.get("damage_label"),
        "confidence": round(float(primary.get("confidence", 0)), 4),
        "severity": SEVERITY_KO.get(
            estimate.get("severity"), estimate.get("severity")
        ),
        "repair_method": estimate.get("repair_method"),
        "min_cost": estimate.get("min_cost"),
        "max_cost": estimate.get("max_cost"),
    }

    df_row = pd.DataFrame([row], columns=DIAGNOSIS_LOG_COLUMNS)
    write_header = not DIAGNOSIS_LOG_PATH.exists()

    df_row.to_csv(
        DIAGNOSIS_LOG_PATH,
        mode="a",
        header=write_header,
        index=False,
        encoding="utf-8-sig",
    )


def maybe_log_diagnosis(primary, estimate):
    """rerun 때마다 같은 결과가 중복 저장되지 않도록 방지 후 기록."""
    log_key = (
        primary.get("part_code"),
        primary.get("damage_code"),
        estimate.get("severity"),
        estimate.get("min_cost"),
        estimate.get("max_cost"),
    )

    if st.session_state.get("last_logged_key") == log_key:
        return

    try:
        log_diagnosis(primary, estimate)
        st.session_state["last_logged_key"] = log_key
    except Exception:
        # 로깅 실패가 견적 표시 자체를 막으면 안 되므로 조용히 무시
        pass


def load_diagnosis_log():
    if not DIAGNOSIS_LOG_PATH.exists():
        return pd.DataFrame(columns=DIAGNOSIS_LOG_COLUMNS)

    try:
        return pd.read_csv(DIAGNOSIS_LOG_PATH, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame(columns=DIAGNOSIS_LOG_COLUMNS)


def build_report_dataframe(diagnosis, estimate):
    """리포트 다운로드용 표. 검출된 모든 부위 + 대표건 견적을 한 표로.

    이전에는 "" 로 컬럼을 먼저 만든 뒤 .loc으로 숫자를 나중에 넣었는데,
    pandas가 빈 문자열만 있는 컬럼을 string 전용 dtype으로 추론해버려서
    이후 int(min_cost/max_cost)를 넣을 때 TypeError가 발생했습니다.
    그래서 행(dict) 단위로 값을 다 채운 뒤 한 번에 DataFrame을 생성하도록
    바꿔서, 컬럼 dtype이 도중에 고정되는 문제 자체를 없앴습니다.
    """
    rows = diagnosis.get("rows", [])
    has_estimate = bool(estimate and estimate.get("success"))
    primary_part = diagnosis.get("primary", {}).get("part_label") if has_estimate else None

    def cost_text(value):
        return f"{int(value):,}" if isinstance(value, (int, float)) else ""

    records = []
    for r in rows:
        record = {
            "부위": r["part_label"],
            "손상종류": r["damage_label"],
            "신뢰도": f'{r["confidence"]:.2%}',
        }

        if has_estimate:
            is_primary_row = r["part_label"] == primary_part
            record["대표건_수리방식"] = estimate.get("repair_method", "") if is_primary_row else ""
            record["대표건_예상수리비_최소"] = cost_text(estimate.get("min_cost")) if is_primary_row else ""
            record["대표건_예상수리비_최대"] = cost_text(estimate.get("max_cost")) if is_primary_row else ""

        records.append(record)

    return pd.DataFrame(records)


def build_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def build_excel_bytes(df):
    """openpyxl이 없으면 None을 반환 (호출부에서 버튼을 숨김)."""
    buffer = io.BytesIO()
    try:
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="진단결과")
    except ImportError:
        return None
    return buffer.getvalue()


# ---------------------------------------------------------
# 모델 로드
# ---------------------------------------------------------
model, used_path = load_model()
damage_type_clf, damage_type_classes = load_damage_type_model()


# ---------------------------------------------------------
# Sidebar 메뉴 + 설정
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 6px 0 18px 0;">
        <div style="font-size:26px;font-weight:800;">🚘 CarDoc AI</div>
        <div style="font-size:13px;color:#6B7280;margin-top:4px;">
            AI Vehicle Care Service
        </div>
    </div>
    """, unsafe_allow_html=True)

    menu = st.radio(
        "메뉴",
        [
            "🏠 홈",
            "📷 차량 진단",
            "💰 예상 견적",
            "📍 정비소 찾기",
            "💬 AI 상담",
            "📄 진단 리포트",
            "📊 대시보드",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("### ⚙️ 진단 설정")

    conf_threshold = st.slider(
        "Confidence threshold",
        0.1, 0.9, 0.3, 0.05,
    )

    if damage_type_clf is not None:
        st.success("손상 종류 분류기 활성화")
    else:
        st.warning("손상 종류 분류기 미탑재")

    st.caption("YOLO + ResNet 기반 2단계 진단")
    st.caption("견적은 룰베이스 단가표를 사용합니다.")

    st.divider()
    if st.session_state.get("diagnosis"):
        st.success("✅ 차량 진단 완료")
    else:
        st.caption("○ 차량 진단 전")


# ---------------------------------------------------------
# 공통 Header
# 홈 / 차량 진단에서만 표시
# ---------------------------------------------------------
if menu in ["🏠 홈", "📷 차량 진단"]:
    if LOGO_PATH.exists():
        with open(LOGO_PATH, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode("utf-8")

        st.markdown(
            f"""
            <div style="margin-bottom:22px;">
                <img src="data:image/png;base64,{logo_b64}"
                     style="width:250px;height:auto;display:block;">
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <h1 style="margin:0 0 10px 0;font-size:42px;font-weight:850;">
        차체 도장·외관 불량 자동 검출 시스템
        </h1>

        <p style="margin:0 0 26px 0;color:#6B7280;font-size:17px;">
        AI 기반 차량 외관 손상 진단 · 예상 수리비 · 상담 서비스
        </p>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 차량 진단
# 홈에서도 전체 흐름의 첫 단계로 표시
# =========================================================
if menu in ["🏠 홈", "📷 차량 진단"]:
    uploaded = st.file_uploader(
        "차체 이미지 업로드 (jpg / png)",
        type=["jpg", "jpeg", "png"],
        key="vehicle_upload",
    )

    if uploaded:
        file_bytes = np.frombuffer(uploaded.getvalue(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            st.error("이미지를 읽을 수 없습니다. 유효한 JPG/PNG 파일을 업로드하세요.")
            st.stop()

        MIN_SIZE = 100
        h, w = img_bgr.shape[:2]

        if h < MIN_SIZE or w < MIN_SIZE:
            st.error(
                f"이미지가 너무 작습니다 ({w}x{h}). "
                f"최소 {MIN_SIZE}x{MIN_SIZE} 이상 업로드하세요."
            )
            st.stop()

        with st.spinner("AI가 차량 외관 손상을 분석하고 있습니다..."):
            results = model.predict(
                img_bgr,
                conf=conf_threshold,
                verbose=False,
            )

        vis = draw_results(img_bgr, results)
        boxes = results[0].boxes

        st.markdown("### 🚘 AI 차량 진단 과정")

        image_col1, image_col2, image_col3 = st.columns(3, gap="medium")

        with image_col1:
            st.markdown("#### ① 원본 차량")
            st.image(
                cv2.cvtColor(
                    resize_for_display(img_bgr),
                    cv2.COLOR_BGR2RGB,
                ),
                use_container_width=True,
            )

        with image_col2:
            st.markdown("#### ② AI 파손 검출")
            st.image(
                cv2.cvtColor(
                    resize_for_display(vis),
                    cv2.COLOR_BGR2RGB,
                ),
                use_container_width=True,
            )

        with image_col3:
            st.markdown("#### ③ 복원 예상")

            if len(boxes) == 0:
                st.info("손상이 검출되지 않아 복원 이미지를 생성하지 않습니다.")
            else:
                # 라벨 표시용으로 가장 신뢰도가 높은 박스는 계속 참조
                top_box_obj = max(
                    boxes,
                    key=lambda b: float(b.conf[0]),
                )
                top_raw_class = results[0].names[int(top_box_obj.cls[0])]
                top_part_label = KOREAN_NAMES.get(
                    top_raw_class,
                    top_raw_class,
                )

                # 복원 대상은 conf 0.3 이상인 박스 전부 (같은 패널의 여러
                # 손상 박스를 하나의 마스크로 합쳐서 inpainting하기 위함)
                damage_boxes = [
                    tuple(map(float, b.xyxy[0]))
                    for b in boxes
                    if float(b.conf[0]) >= 0.3
                ]

                # 동일 업로드 이미지에서는 생성 결과 유지
                repair_key = (
                    uploaded.name,
                    tuple(
                        tuple(round(v, 1) for v in b)
                        for b in damage_boxes
                    ),
                    top_part_label,
                )

                if (
                    st.session_state.get("repair_image_key")
                    != repair_key
                ):
                    st.session_state.pop(
                        "generated_repair_image",
                        None,
                    )

                if st.session_state.get("generated_repair_image") is not None:
                    st.image(
                        st.session_state["generated_repair_image"],
                        use_container_width=True,
                    )
                    st.caption(
                        "※ 생성형 AI 기반 복원 시뮬레이션이며 "
                        "실제 수리 결과와 차이가 있을 수 있습니다."
                    )

                else:
                    st.info(
                        f"검출 부위: {top_part_label}\n\n"
                        "아래 버튼을 누르면 해당 손상 영역을 기준으로 "
                        "AI 복원 시뮬레이션을 생성합니다."
                    )

                    if st.button(
                        "✨ AI 복원 이미지 생성",
                        key="generate_repair_image",
                        use_container_width=True,
                    ):
                        original_pil = Image.fromarray(
                            cv2.cvtColor(
                                img_bgr,
                                cv2.COLOR_BGR2RGB,
                            )
                        )

                        try:
                            with st.spinner(
                                "생성형 AI가 손상 부위를 복원하고 있습니다... "
                                "첫 실행은 모델 다운로드 때문에 오래 걸릴 수 있습니다."
                            ):
                                repair_pipe = load_repair_pipeline()

                                # 검출된 클래스명 전체를 영문 그대로 프롬프트에
                                # 전달해 "무엇이 손상됐는지"를 명확히 지시
                                detected_classes = sorted({
                                    results[0].names[int(b.cls[0])]
                                    for b in boxes
                                    if float(b.conf[0]) >= 0.3
                                })
                                damaged_parts = ", ".join(detected_classes)

                                repaired_image, _, _ = generate_repaired_image_full(
                                    pipe=repair_pipe,
                                    original_image=original_pil,
                                    damage_boxes=damage_boxes,
                                    damaged_parts=damaged_parts,
                                    config=RepairConfig(
                                            mask_blur_radius=15,
                                            target_size=1024,
                                            steps=28,
                                            guidance_scale=2.5,
                                            true_cfg_scale=1.0,
                                            low_vram=True,
                                            seed=123,
                                    ),
                                    # 박스 밖으로 이어진 파손(범퍼/파편)까지
                                    # 편집 영역에 포함하도록 넉넉히 확장
                                    side_pad_ratio=0.45,
                                    top_pad_ratio=0.20,
                                    bottom_pad_ratio=1.10,
                                    merge_boxes=True,
                                    watermark_frac=0.12,
                                )

                            st.session_state[
                                "generated_repair_image"
                            ] = repaired_image
                            st.session_state[
                                "repair_image_key"
                            ] = repair_key
                            st.rerun()

                        except Exception as e:
                            st.error(
                                "AI 복원 이미지 생성에 실패했습니다."
                            )
                            st.code(str(e))
                            st.caption(
                                "diffusers 설치 여부와 GPU 메모리를 확인하세요."
                            )

        st.divider()

        if len(boxes) == 0:
            st.success("✅ 불량 미검출 — 정상 판정")

            st.session_state["diagnosis"] = {
                "normal": True,
                "rows": [],
            }
            st.session_state.pop("estimate", None)

        else:
            rows = []

            for b in boxes:
                raw_yolo_class = results[0].names[int(b.cls[0])]
                part_label = KOREAN_NAMES.get(raw_yolo_class, raw_yolo_class)
                part_code = PART_CODE_MAP.get(raw_yolo_class)
                confidence = float(b.conf[0])

                raw_damage = None
                damage_code = None
                damage_label = "-"

                if damage_type_clf is not None:
                    raw_damage = classify_damage_type_raw(
                        img_bgr,
                        tuple(map(float, b.xyxy[0])),
                        damage_type_clf,
                        damage_type_classes,
                    )

                    damage_code = DAMAGE_CODE_MAP.get(raw_damage)
                    damage_label = DAMAGE_TYPE_KOREAN.get(
                        raw_damage,
                        raw_damage or "-"
                    )

                rows.append(
                    {
                        "raw_yolo_class": raw_yolo_class,
                        "part_code": part_code,
                        "part_label": part_label,
                        "damage_code": damage_code,
                        "damage_label": damage_label,
                        "confidence": confidence,
                    }
                )

            primary = max(rows, key=lambda x: x["confidence"])

            st.session_state["diagnosis"] = {
                "normal": False,
                "rows": rows,
                "primary": primary,
            }

            st.markdown("### 🔍 AI 진단 결과")

            metric1, metric2, metric3, metric4 = st.columns(4)

            with metric1:
                st.metric("검출 건수", f"{len(rows)}건")

            with metric2:
                st.metric("주요 파손 부위", primary["part_label"])

            with metric3:
                st.metric("손상 종류", primary["damage_label"])

            with metric4:
                st.metric("AI 신뢰도", f'{primary["confidence"]:.1%}')

            df = pd.DataFrame(
                {
                    "부위": [r["part_label"] for r in rows],
                    "손상 종류": [r["damage_label"] for r in rows],
                    "신뢰도": [f'{r["confidence"]:.2%}' for r in rows],
                }
            )

            with st.expander("상세 검출 결과 보기"):
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                )

            # 홈에서는 기존처럼 진단 아래에 견적도 바로 표시
            if menu == "🏠 홈":
                st.markdown("### 🔧 예상 수리 가이드")

                st.info(
                    "현재 모델은 파손 부위와 손상 종류를 자동 판별합니다. "
                    "수리비 산정을 위한 손상 심각도는 현재 사용자가 선택하며, "
                    "추후 별도 심각도 모델로 자동화할 수 있습니다."
                )

                severity_ko = st.segmented_control(
                    "손상 정도를 선택하세요",
                    options=["경미", "중간", "심각"],
                    default=st.session_state.get("severity_ko", "중간"),
                    key="severity_home",
                )

                st.session_state["severity_ko"] = severity_ko or "중간"
                severity = SEVERITY_EN[st.session_state["severity_ko"]]

                if not primary["part_code"]:
                    st.warning(
                        f'현재 검출 부위 "{primary["part_label"]}"는 '
                        "단가표와 아직 매핑되지 않았습니다."
                    )

                elif not primary["damage_code"]:
                    st.warning(
                        "손상 종류가 단가표 코드와 연결되지 않아 자동 견적을 만들 수 없습니다."
                    )

                else:
                    estimate = get_repair_estimate(
                        part=primary["part_code"],
                        damage_type=primary["damage_code"],
                        severity=severity,
                    )

                    if estimate.get("success"):
                        st.session_state["estimate"] = estimate
                        maybe_log_diagnosis(primary, estimate)

                        col_a, col_b, col_c = st.columns([1, 1.3, 1.5])

                        with col_a:
                            st.markdown(
                                f"""
<div class="result-box">
    <div class="result-label">파손 부위</div>
    <div class="result-value">{estimate["part_label"]}</div>
</div>
""",
                                unsafe_allow_html=True,
                            )

                        with col_b:
                            st.markdown(
                                f"""
<div class="result-box">
    <div class="result-label">예상 수리 방식</div>
    <div class="result-value">{estimate["repair_method"]}</div>
</div>
""",
                                unsafe_allow_html=True,
                            )

                        with col_c:
                            st.markdown(
                                f"""
<div class="result-box">
    <div class="result-label">예상 수리비</div>
    <div class="price-value">{estimate["min_cost"]:,} ~ {estimate["max_cost"]:,}원</div>
</div>
""",
                                unsafe_allow_html=True,
                            )

                        source_names = source_names_from_ids(
                            estimate.get("source")
                        )

                        with st.expander("📚 견적 기준 및 근거 보기"):
                            st.write(
                                f"**손상 정도:** "
                                f"{SEVERITY_KO.get(estimate['severity'], estimate['severity'])}"
                            )

                            if source_names:
                                st.write("**참고 단가 출처:**")
                                for name in source_names:
                                    st.write(f"- {name}")

                            if estimate.get("note"):
                                st.write(f"**참고:** {estimate['note']}")

                            st.caption(estimate.get("disclaimer", ""))

                        if estimate.get("via") == "api":
                            st.caption("견적 계산: FastAPI Backend 연동")
                        else:
                            st.caption(
                                "견적 계산: FastAPI 미연결 상태 → "
                                "로컬 단가표 fallback 사용"
                            )

                    else:
                        st.warning(
                            estimate.get(
                                "message",
                                "견적 정보를 찾을 수 없습니다."
                            )
                        )


# =========================================================
# 예상 견적 단독 페이지
# =========================================================
if menu == "💰 예상 견적":
    st.markdown("## 💰 예상 수리 견적")

    diagnosis = st.session_state.get("diagnosis")

    if not diagnosis:
        st.info("📷 먼저 '차량 진단' 메뉴에서 차량 사진을 분석해주세요.")

    elif diagnosis.get("normal"):
        st.success("정상 판정 차량입니다. 현재 자동 수리 견적이 필요하지 않습니다.")

    else:
        primary = diagnosis["primary"]

        d1, d2, d3 = st.columns(3)

        with d1:
            st.metric("파손 부위", primary["part_label"])

        with d2:
            st.metric("손상 종류", primary["damage_label"])

        with d3:
            st.metric("AI 신뢰도", f'{primary["confidence"]:.1%}')

        severity_ko = st.segmented_control(
            "손상 정도를 선택하세요",
            options=["경미", "중간", "심각"],
            default=st.session_state.get("severity_ko", "중간"),
            key="severity_estimate_page",
        )

        st.session_state["severity_ko"] = severity_ko or "중간"
        severity = SEVERITY_EN[st.session_state["severity_ko"]]

        if not primary["part_code"]:
            st.warning(
                f'현재 검출 부위 "{primary["part_label"]}"는 '
                "단가표와 아직 매핑되지 않았습니다."
            )

        elif not primary["damage_code"]:
            st.warning(
                "손상 종류가 단가표 코드와 연결되지 않아 자동 견적을 만들 수 없습니다."
            )

        else:
            estimate = get_repair_estimate(
                part=primary["part_code"],
                damage_type=primary["damage_code"],
                severity=severity,
            )

            if estimate.get("success"):
                st.session_state["estimate"] = estimate
                maybe_log_diagnosis(primary, estimate)

                col_a, col_b, col_c = st.columns([1, 1.3, 1.5])

                with col_a:
                    st.metric("파손 부위", estimate["part_label"])

                with col_b:
                    st.metric("예상 수리 방식", estimate["repair_method"])

                with col_c:
                    st.metric(
                        "예상 수리비",
                        f'{estimate["min_cost"]:,} ~ {estimate["max_cost"]:,}원'
                    )

                source_names = source_names_from_ids(
                    estimate.get("source")
                )

                with st.expander("📚 견적 기준 및 근거 보기"):
                    st.write(
                        f"**손상 정도:** "
                        f"{SEVERITY_KO.get(estimate['severity'], estimate['severity'])}"
                    )

                    if source_names:
                        st.write("**참고 단가 출처:**")
                        for name in source_names:
                            st.write(f"- {name}")

                    if estimate.get("note"):
                        st.write(f"**참고:** {estimate['note']}")

                    st.caption(estimate.get("disclaimer", ""))

            else:
                st.warning(
                    estimate.get(
                        "message",
                        "견적 정보를 찾을 수 없습니다."
                    )
                )


# =========================================================
# 주변 자동차 정비소
# 홈에서도 전체 흐름의 한 단계로 표시
# =========================================================
if menu in ["🏠 홈", "📍 정비소 찾기"]:
    st.divider()
    st.markdown("### 📍 주변 자동차 정비소 찾기")

    st.caption(
        "현재 위치를 입력하면 주변 자동차 정비소를 지도에서 확인할 수 있습니다."
    )

    if "repair_shops" not in st.session_state:
        st.session_state.repair_shops = []

    if "repair_latitude" not in st.session_state:
        st.session_state.repair_latitude = None

    if "repair_longitude" not in st.session_state:
        st.session_state.repair_longitude = None

    if "repair_search_done" not in st.session_state:
        st.session_state.repair_search_done = False

    if "repair_address" not in st.session_state:
        st.session_state.repair_address = ""

    address = st.text_input(
        "현재 위치",
        placeholder="예: 대구광역시 북구 대학로 80",
        key=f"repair_shop_address_input_{menu}",
    )

    radius_km = st.selectbox(
        "검색 반경",
        options=[1, 3, 5, 10],
        index=1,
        format_func=lambda x: f"{x} km",
        key=f"repair_shop_radius_{menu}",
    )

    if st.button(
        "🔍 주변 정비소 찾기",
        use_container_width=True,
        type="primary",
        key=f"repair_shop_button_{menu}",
    ):
        if not address:
            st.warning("현재 위치를 입력해주세요.")
        else:
            try:
                geo_response = requests.get(
                    "http://127.0.0.1:8000/geocode",
                    params={"address": address},
                    timeout=10,
                )

                geo_response.raise_for_status()
                geo_result = geo_response.json()

                if not geo_result.get("success"):
                    st.error(
                        geo_result.get(
                            "message",
                            "주소를 찾을 수 없습니다."
                        )
                    )

                else:
                    latitude = geo_result["lat"]
                    longitude = geo_result["lng"]

                    shop_response = requests.get(
                        "http://127.0.0.1:8000/repair-shops",
                        params={
                            "x": longitude,
                            "y": latitude,
                            "radius": radius_km * 1000,
                            "query": "자동차 정비소",
                        },
                        timeout=10,
                    )

                    shop_response.raise_for_status()
                    shop_result = shop_response.json()

                    if not shop_result.get("success"):
                        st.error(
                            shop_result.get(
                                "message",
                                "정비소 검색에 실패했습니다."
                            )
                        )

                    else:
                        st.session_state.repair_shops = shop_result["shops"]
                        st.session_state.repair_latitude = latitude
                        st.session_state.repair_longitude = longitude
                        st.session_state.repair_search_done = True
                        st.session_state.repair_address = address

            except requests.ConnectionError:
                st.error(
                    "FastAPI 서버에 연결할 수 없습니다. "
                    "estimate_api.py가 실행 중인지 확인해주세요."
                )

            except requests.RequestException as e:
                st.error(
                    f"정비소 정보를 불러오는 중 오류가 발생했습니다: {e}"
                )

    if st.session_state.repair_search_done:
        shops = st.session_state.repair_shops
        latitude = st.session_state.repair_latitude
        longitude = st.session_state.repair_longitude

        st.success(
            f"📍 {st.session_state.repair_address} 기준 "
            f"주변 정비소 {len(shops)}곳을 찾았습니다."
        )

        if len(shops) == 0:
            st.info("선택한 반경 내에서 정비소를 찾지 못했습니다.")

        else:
            repair_map = folium.Map(
                location=[latitude, longitude],
                zoom_start=14,
            )

            folium.Marker(
                location=[latitude, longitude],
                tooltip="현재 위치",
                popup="현재 위치",
                icon=folium.Icon(
                    color="blue",
                    icon="home",
                ),
            ).add_to(repair_map)

            for shop in shops:
                phone = (
                    shop["phone"]
                    if shop["phone"]
                    else "전화번호 정보 없음"
                )

                distance = (
                    f'{shop["distance"]}m'
                    if shop["distance"]
                    else "거리 정보 없음"
                )

                popup_html = f"""
                <div style="width:230px;">
                    <b>{shop["name"]}</b><br><br>
                    📍 {shop["address"]}<br>
                    ☎ {phone}<br>
                    🚗 {distance}
                </div>
                """

                folium.Marker(
                    location=[shop["lat"], shop["lng"]],
                    tooltip=shop["name"],
                    popup=folium.Popup(
                        popup_html,
                        max_width=300,
                    ),
                ).add_to(repair_map)

            map_col, list_col = st.columns(
                [1.6, 1],
                gap="large",
            )

            with map_col:
                st.markdown("#### 🗺️ 주변 정비소 지도")

                st_folium(
                    repair_map,
                    height=550,
                    use_container_width=True,
                    key=f"repair_shop_map_{menu}",
                )

            with list_col:
                st.markdown("#### 🚗 가까운 정비소")

                for i, shop in enumerate(shops, start=1):
                    with st.container(border=True):
                        st.markdown(f"### {i}. {shop['name']}")
                        st.caption(f"📍 {shop['address']}")

                        if shop["phone"]:
                            st.write(f"☎️ {shop['phone']}")

                        if shop["distance"]:
                            try:
                                distance_m = int(shop["distance"])

                                if distance_m >= 1000:
                                    distance_text = (
                                        f"{distance_m / 1000:.1f} km"
                                    )
                                else:
                                    distance_text = f"{distance_m} m"

                                st.write(
                                    f"🚗 약 {distance_text}"
                                )

                            except ValueError:
                                st.write(
                                    f'🚗 {shop["distance"]}'
                                )

                        if shop.get("place_url"):
                            st.link_button(
                                "🗺️ 카카오맵에서 보기",
                                shop["place_url"],
                                use_container_width=True,
                            )

        if st.button(
            "검색 결과 초기화",
            key=f"reset_repair_search_{menu}",
        ):
            st.session_state.repair_shops = []
            st.session_state.repair_latitude = None
            st.session_state.repair_longitude = None
            st.session_state.repair_search_done = False
            st.session_state.repair_address = ""
            st.rerun()


# =========================================================
# AI 수리 상담
# 홈에서도 표시
# =========================================================
if menu in ["🏠 홈", "💬 AI 상담"]:
    st.divider()
    st.markdown("### 💬 AI 수리 상담")

    st.caption(
        "현재는 RAG API 연결 전입니다. "
        "진단·견적 결과를 기반으로 임시 응답을 표시합니다."
    )

    diagnosis = st.session_state.get("diagnosis")
    estimate = st.session_state.get("estimate")

    if diagnosis and not diagnosis.get("normal"):
        primary = diagnosis["primary"]

        st.info(
            f'현재 진단: {primary["part_label"]} / '
            f'{primary["damage_label"]} / '
            f'신뢰도 {primary["confidence"]:.1%}'
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input(
        "예: 이 정도면 후드를 교체해야 하나요?",
        key=f"chat_input_{menu}",
    )

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        if (
            diagnosis
            and not diagnosis.get("normal")
            and estimate
        ):
            primary = diagnosis["primary"]
            severity_ko = st.session_state.get(
                "severity_ko",
                "중간",
            )

            answer = (
                f"현재 진단 결과는 **{estimate['part_label']} / "
                f"{primary['damage_label']} / {severity_ko}**입니다. "
                f"단가표 기준 예상 수리 방식은 "
                f"**{estimate['repair_method']}**, "
                f"예상 비용은 "
                f"**{estimate['min_cost']:,}~{estimate['max_cost']:,}원**입니다. "
                "정확한 교환 여부는 실제 정비소의 현물 점검이 필요합니다."
            )

        else:
            answer = (
                "현재는 RAG 상담 기능 연결 전입니다. "
                "먼저 차량 진단과 예상 견적을 진행해주세요."
            )

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )


# =========================================================
# 진단 리포트
# =========================================================
if menu == "📄 진단 리포트":
    st.markdown("## 📄 AI 차량 진단 리포트")

    diagnosis = st.session_state.get("diagnosis")
    estimate = st.session_state.get("estimate")

    if not diagnosis:
        st.info("먼저 '차량 진단' 메뉴에서 차량 사진을 분석해주세요.")

    elif diagnosis.get("normal"):
        st.success("✅ 정상 판정 차량입니다.")

    else:
        primary = diagnosis["primary"]

        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric("파손 부위", primary["part_label"])

        with r2:
            st.metric("손상 종류", primary["damage_label"])

        with r3:
            st.metric("AI 신뢰도", f'{primary["confidence"]:.1%}')

        if estimate:
            st.markdown("### 🔧 수리 견적")

            e1, e2 = st.columns(2)

            with e1:
                st.metric(
                    "예상 수리 방식",
                    estimate["repair_method"],
                )

            with e2:
                st.metric(
                    "예상 수리비",
                    f'{estimate["min_cost"]:,} ~ '
                    f'{estimate["max_cost"]:,}원',
                )

            st.caption(
                estimate.get(
                    "disclaimer",
                    "",
                )
            )

        else:
            st.info(
                "예상 견적 메뉴에서 손상 정도를 선택해 "
                "견적을 생성해주세요."
            )

        st.divider()
        st.markdown("### ⬇️ 결과 다운로드")

        report_df = build_report_dataframe(diagnosis, estimate)

        dl1, dl2 = st.columns(2)

        with dl1:
            st.download_button(
                "📄 CSV로 다운로드",
                data=build_csv_bytes(report_df),
                file_name="차량진단_견적서.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with dl2:
            excel_bytes = build_excel_bytes(report_df)

            if excel_bytes is not None:
                st.download_button(
                    "📊 Excel로 다운로드",
                    data=excel_bytes,
                    file_name="차량진단_견적서.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument"
                        ".spreadsheetml.sheet"
                    ),
                    use_container_width=True,
                )
            else:
                st.caption(
                    "Excel 다운로드를 사용하려면 `openpyxl` 설치가 필요합니다. "
                    "(`pip install openpyxl`)"
                )


# =========================================================
# 대시보드 / 통계
# =========================================================
if menu == "📊 대시보드":
    st.markdown("## 📊 진단 통계 대시보드")

    st.caption(
        "'예상 견적' 화면에서 견적이 생성될 때마다 로컬 CSV(diagnosis_log.csv)에 "
        "자동으로 누적 저장된 진단 이력을 집계해 보여줍니다."
    )

    log_df = load_diagnosis_log()

    if log_df.empty:
        st.info(
            "아직 누적된 진단 데이터가 없습니다. "
            "'차량 진단' → 견적 생성까지 진행하면 이 화면에 데이터가 쌓입니다."
        )

    else:
        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric("누적 진단 건수", f"{len(log_df)}건")

        with m2:
            avg_min = int(log_df["min_cost"].mean())
            avg_max = int(log_df["max_cost"].mean())
            st.metric("평균 예상 수리비", f"{avg_min:,} ~ {avg_max:,}원")

        with m3:
            top_part = log_df["part_label"].mode()
            st.metric(
                "최다 손상 부위",
                top_part.iloc[0] if len(top_part) else "-",
            )

        st.divider()

        chart1, chart2 = st.columns(2)

        with chart1:
            st.markdown("#### 부위별 진단 건수")
            part_counts = (
                log_df["part_label"]
                .value_counts()
                .rename_axis("부위")
                .reset_index(name="건수")
                .set_index("부위")
            )
            st.bar_chart(part_counts)

        with chart2:
            st.markdown("#### 손상 종류별 진단 건수")
            damage_counts = (
                log_df["damage_label"]
                .value_counts()
                .rename_axis("손상종류")
                .reset_index(name="건수")
                .set_index("손상종류")
            )
            st.bar_chart(damage_counts)

        st.markdown("#### 날짜별 진단 추이")

        trend_df = log_df.copy()
        trend_df["date"] = pd.to_datetime(trend_df["timestamp"]).dt.date
        daily_counts = (
            trend_df.groupby("date").size().rename("건수").to_frame()
        )
        st.line_chart(daily_counts)

        st.markdown("#### 최근 진단 이력")

        st.dataframe(
            log_df.sort_values("timestamp", ascending=False).head(20),
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

        dl1, dl2 = st.columns(2)

        with dl1:
            st.download_button(
                "📄 전체 이력 CSV 다운로드",
                data=build_csv_bytes(log_df),
                file_name="진단이력_전체.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with dl2:
            log_excel = build_excel_bytes(log_df)

            if log_excel is not None:
                st.download_button(
                    "📊 전체 이력 Excel 다운로드",
                    data=log_excel,
                    file_name="진단이력_전체.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument"
                        ".spreadsheetml.sheet"
                    ),
                    use_container_width=True,
                )
            else:
                st.caption(
                    "Excel 다운로드를 사용하려면 `openpyxl` 설치가 필요합니다. "
                    "(`pip install openpyxl`)"
                )
