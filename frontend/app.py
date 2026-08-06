"""
Streamlit Frontend — AI 차량 파손 진단 + 예상 수리비 + 상담 UI
실행 (반드시 프로젝트 루트에서):
    streamlit run frontend/app.py

백엔드가 켜져 있으면 /estimate API를 호출하고,
꺼져 있으면 단가표.json을 직접 읽는 fallback 로직으로 동작합니다.
"""

import base64
import hashlib
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st

import folium
from streamlit_folium import st_folium

from PIL import Image

# 단가표.json / 로고 / 로그 등 저장소 루트에 남아있는 자원을 참조하기 위한 경로.
ROOT_DIR = Path(__file__).resolve().parent.parent

from utils.preprocessing import draw_detections, resize_for_display
from utils.api_client import (
    call_chat_api,
    call_diagnose_api,
    call_estimate_api,
    call_geocode_api,
    call_llm_health,
    call_repair_preview_api,
    call_repair_preview_health,
    call_repair_shops_api,
)

# AI 상담도 RAGS.py(Qwen2.5-7B를 이 프로세스에 직접 로드)를 쓰지 않고
# backend /chat(Ollama 컨테이너)으로 옮겼다. 이유:
#   1. 7B 모델을 Streamlit 프로세스에 올리면 프론트 이미지가 6GB를 넘고,
#      4bit 양자화(bitsandbytes)는 CUDA가 필요해 GPU 없는 클라우드에서 실패한다.
#   2. LLM은 backend가 소유하는 게 프론트/백엔드 분리 원칙에도 맞는다.


# ---------------------------------------------------------
# 기본 경로 (단가표.json은 backend/data/ 원본 하나만 두고 frontend가 폴백용으로 같이 읽음)
# ---------------------------------------------------------
PRICE_TABLE_PATH = ROOT_DIR / "backend/data/단가표.json"
LOGO_PATH = ROOT_DIR / "docs/ajin_logo.png"

# 대시보드 집계용 진단 이력 누적 로그 (견적 생성 시마다 1행씩 append).
# 컨테이너에서는 LOG_DIR=/app/logs(named volume)로 주입해서, 컨테이너를 지워도
# 진단 이력이 남도록 합니다. 로컬 실행 시에는 기존처럼 저장소 루트에 생성됩니다.
LOG_DIR = Path(os.getenv("LOG_DIR", ROOT_DIR))
LOG_DIR.mkdir(parents=True, exist_ok=True)
DIAGNOSIS_LOG_PATH = LOG_DIR / "diagnosis_log.csv"

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
# 진단 관련: call_diagnose_api()는 utils/api_client.py로 이동함
# (backend /diagnose 호출 로직 — 프론트/백엔드 분리 원칙에 따라 HTTP 통신은
# api_client 모듈에만 두고, app.py는 UI/상태 관리에 집중)
# ---------------------------------------------------------


# ---------------------------------------------------------
# AI 상담 — backend /chat (상세 정비 지식 RAG + Ollama/EXAONE)
# ---------------------------------------------------------
def build_diagnosis_summary(diagnosis, estimate):
    """세션의 진단·견적 결과를 LLM 프롬프트에 넣을 한 문단으로 요약.

    견적(estimate)은 없을 수 있습니다 — 단가표에 없는 (부위, 손상 종류, 심각도)
    조합이면 조회가 실패하기 때문입니다. 그 경우에도 진단 결과만은 반드시
    넘겨야 합니다. 이걸 안 넘기면 챗봇이 "사진에서 뭘 찾았는지" 자체를 모른 채
    답하게 되어, 일반론만 늘어놓는 답변이 나옵니다.
    """
    if not diagnosis or diagnosis.get("normal"):
        return ""

    primary = diagnosis.get("primary") or {}
    part_label = primary.get("part_label", "미상")
    # 견적이 성공했으면 단가표 기준 부위명을 쓰는 편이 LLM에게 더 정확합니다.
    if estimate and estimate.get("part_label"):
        part_label = estimate["part_label"]

    lines = [
        f"- 부위: {part_label}",
        f"- 손상 종류: {primary.get('damage_label', '미상')}",
        f"- 심각도: {st.session_state.get('severity_ko', '중간')} (사용자 선택값)",
        f"- 탐지 신뢰도: {primary.get('confidence', 0):.1%}",
    ]

    if estimate and estimate.get("success"):
        lines += [
            f"- 수리 방식: {estimate['repair_method']}",
            f"- 예상 비용: {estimate['min_cost']:,}원 ~ {estimate['max_cost']:,}원",
        ]
    else:
        lines.append(
            "- 견적: 단가표에 이 (부위, 손상 종류, 심각도) 조합이 없어 금액을 "
            "산출하지 못했습니다. 금액을 추측하지 말고 정비소 방문을 권하세요."
        )

    return "\n".join(lines)


def render_llm_status():
    """사이드바에 상담 LLM과 RAG 준비 상태를 표시합니다."""
    status = call_llm_health()

    if status is None:
        st.warning("백엔드에 연결할 수 없습니다.")
        return

    model = status.get("model", "?")
    if status.get("ready"):
        st.success(f"LLM 준비 완료 · {model}")
    elif status.get("server_up"):
        st.warning(f"모델 다운로드 중 · {model}")
        st.caption("완료 전까지 AI 상담은 검색 결과 기반 답변을 냅니다.")
    else:
        st.error("LLM 서버(Ollama)가 아직 기동되지 않았습니다.")

    rag_status = status.get("rag") or {}
    if rag_status.get("ready"):
        mode = {
            "hybrid": "의미+키워드",
            "hybrid_pending": "의미+키워드(첫 질문 시 로드)",
        }.get(rag_status.get("mode"), "키워드 폴백")
        st.caption(
            f"RAG 준비 완료 · 문서 {rag_status.get('documents', 0)}개 · {mode} 검색"
        )
    else:
        st.warning("RAG 정비 문서를 불러오지 못했습니다.")


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
    try:
        result = call_estimate_api(part, damage_type, severity, timeout=3)
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
# Sidebar 메뉴 + 설정
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 6px 0 18px 0;">
        <div style="font-size:26px;font-weight:800;">🚘 AutoCarCare AI</div>
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

    st.caption("AI 진단: backend API 연동 (YOLO + ResNet 기반 2단계 진단)")
    st.caption("견적은 룰베이스 단가표를 사용합니다.")

    st.divider()
    st.markdown("### 🤖 AI 상담 모델")
    render_llm_status()

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
            api_results, diagnose_error = call_diagnose_api(
                uploaded.getvalue(),
                conf_threshold=conf_threshold,
                filename=uploaded.name,
            )

        if diagnose_error:
            st.error(
                "AI 진단 backend에 연결할 수 없습니다. "
                "`backend/main.py`가 실행 중인지 확인해주세요. "
                f"(오류: {diagnose_error})"
            )
            st.stop()

        vis = draw_detections(img_bgr, api_results)
        boxes = api_results  # backend가 돌려준 dict 리스트 (부위/손상종류/신뢰도/bbox 포함)

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

            repair_status = call_repair_preview_health()
            if repair_status is None:
                st.info(
                    "복원 API 상태를 확인할 수 없습니다. backend가 실행 중인지 확인해주세요."
                )
            elif not repair_status.get("configured"):
                st.info(
                    "**OpenAI 사진 복원 키가 아직 설정되지 않았습니다.**\n\n"
                    "저장소 루트 `.env` 파일에 `OPENAI_API_KEY=` 값을 입력한 뒤 "
                    "backend를 다시 시작해주세요. 실제 키는 코드나 Git에 넣지 않습니다."
                )
            elif len(boxes) == 0:
                st.info("손상이 검출되지 않아 복원 이미지를 생성하지 않습니다.")
            elif not any(box["confidence"] >= 0.3 for box in boxes):
                st.info(
                    "복원에 사용할 신뢰도 30% 이상의 손상 영역이 없습니다. "
                    "다른 각도의 사진으로 다시 진단해주세요."
                )
            else:
                # 라벨 표시용으로 가장 신뢰도가 높은 박스는 계속 참조
                top_box_obj = max(
                    boxes,
                    key=lambda b: b["confidence"],
                )
                top_raw_class = top_box_obj["part_en"]
                top_part_label = KOREAN_NAMES.get(
                    top_raw_class,
                    top_raw_class,
                )

                # backend도 같은 임계값으로 재검증한다. 여러 박스는 하나의 큰
                # 사각형으로 합치지 않고 합집합 마스크로 만들어 정상 영역을 보존한다.
                repair_detections = [b for b in boxes if b["confidence"] >= 0.3]

                # 파일명이 같아도 내용이 다르면 재생성하도록 이미지 해시를 포함한다.
                image_digest = hashlib.sha256(uploaded.getvalue()).hexdigest()[:16]
                repair_key = (
                    image_digest,
                    tuple(
                        (
                            detection["part_en"],
                            detection["damage_type_en"],
                            round(float(detection["confidence"]), 3),
                            tuple(round(float(v), 1) for v in detection["bbox"]),
                        )
                        for detection in repair_detections
                    ),
                    top_part_label,
                    repair_status.get("model"),
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
                        "※ OpenAI 생성형 이미지 편집 기반 시뮬레이션이며 실제 수리 "
                        "결과와 차이가 있을 수 있습니다."
                    )

                else:
                    st.info(
                        f"검출 부위: {top_part_label}\n\n"
                        "아래 버튼을 누르면 해당 손상 영역을 기준으로 "
                        f"OpenAI `{repair_status.get('model', 'GPT Image')}`가 "
                        "복원 시뮬레이션을 생성합니다."
                    )
                    st.caption(
                        "복원 버튼을 누르면 차량 사진이 이미지 편집을 위해 OpenAI로 "
                        "전송됩니다. 번호판 등 민감한 정보가 있으면 먼저 가려주세요."
                    )

                    if st.button(
                        "✨ AI 복원 이미지 생성",
                        key="generate_repair_image",
                        use_container_width=True,
                    ):
                        try:
                            with st.spinner(
                                "OpenAI가 손상 부위를 복원하고 있습니다... "
                                "이미지 편집은 최대 2분 정도 걸릴 수 있습니다."
                            ):
                                repaired_bytes = call_repair_preview_api(
                                    uploaded.getvalue(),
                                    repair_detections,
                                    filename=uploaded.name,
                                )
                                repaired_image = Image.open(io.BytesIO(repaired_bytes))
                                repaired_image.load()
                                repaired_image = repaired_image.convert("RGB")

                            st.session_state[
                                "generated_repair_image"
                            ] = repaired_image
                            st.session_state[
                                "repair_image_key"
                            ] = repair_key
                            st.rerun()

                        except Exception as e:
                            st.error(f"AI 복원 이미지 생성에 실패했습니다: {e}")
                            st.caption(
                                "`.env`의 OPENAI_API_KEY, API 사용 한도와 backend 로그를 "
                                "확인해주세요."
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
                # backend /diagnose가 부위·손상종류를 이미 다 계산해서 돌려주므로,
                # 여기서는 단가표 코드(front_bumper 등)로 변환만 하면 된다.
                raw_yolo_class = b["part_en"]
                part_label = b["part"]  # 이미 한글 라벨
                part_code = PART_CODE_MAP.get(raw_yolo_class)
                confidence = b["confidence"]

                raw_damage = b["damage_type_en"] if b["damage_type_en"] not in (None, "-") else None
                damage_code = DAMAGE_CODE_MAP.get(raw_damage) if raw_damage else None
                damage_label = b["damage_type"]  # 이미 한글 라벨 (없으면 "-")

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
                geo_result = call_geocode_api(address)

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

                    shop_result = call_repair_shops_api(
                        x=longitude,
                        y=latitude,
                        radius=radius_km * 1000,
                    )

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
                    "backend(uvicorn main:app)가 실행 중인지 확인해주세요."
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
                tiles="cartodbpositron",
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
                    icon=folium.Icon(
                        color="red",
                        icon="wrench",
                        prefix="fa",
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

                with st.container(height=550, border=False):
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
        "backend /chat (RAG + LLM)에 질문을 보냅니다. 정비 지식 문서를 검색해 "
        "근거와 함께 답변하며, 금액은 단가표 조회 결과를 그대로 인용합니다."
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
            if msg.get("sources"):
                source_labels = [
                    f"{source['title']} · {source['section']}"
                    for source in msg["sources"]
                ]
                st.caption("참고 문서: " + " / ".join(source_labels))

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

        with st.chat_message("assistant"):
            progress = st.empty()
            started_at = time.monotonic()
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(
                call_chat_api,
                session_id=st.session_state.get(
                    "session_id", "streamlit-session"
                ),
                message=question,
                diagnosis_summary=build_diagnosis_summary(diagnosis, estimate),
                history=[
                    {
                        "role": msg["role"],
                        "content": msg["content"][:2000],
                    }
                    for msg in st.session_state.messages[:-1][-8:]
                ],
            )
            try:
                while not future.done():
                    elapsed = int(time.monotonic() - started_at)
                    progress.caption(
                        f"답변을 생성하는 중입니다 · {elapsed}초 "
                        "(Codespaces CPU에서는 일반 질문이 약 1분 걸릴 수 있습니다)"
                    )
                    time.sleep(2)

                try:
                    result = future.result()
                    answer = result.get("answer", "답변을 받지 못했습니다.")
                    answer_sources = result.get("sources", [])
                    answer_mode = result.get("answer_mode")
                    if answer_mode == "rag_fallback" or (
                        answer_mode is None and not result.get("used_llm", True)
                    ):
                        # 모델 다운로드 중이거나 답변이 가드레일에 걸린 경우.
                        # 사용자가 품질 저하 이유를 알 수 있게 표시합니다.
                        answer += (
                            "\n\n> ⚠️ LLM 응답을 쓰지 못해 검색된 정비 자료로 "
                            "대체했습니다. 사이드바에서 LLM 상태를 확인하세요."
                        )
                except Exception as e:
                    answer_sources = []
                    answer = (
                        "상담 서버(backend)에 연결하지 못했습니다.\n\n"
                        f"({type(e).__name__}: {e})\n\n"
                        "`docker compose logs -f backend` 로 상태를 확인해주세요."
                    )
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
                progress.empty()

            st.markdown(answer)
            if answer_sources:
                source_labels = [
                    f"{source['title']} · {source['section']}"
                    for source in answer_sources
                ]
                st.caption("참고 문서: " + " / ".join(source_labels))

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": answer_sources,
            }
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
