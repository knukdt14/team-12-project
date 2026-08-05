# backend/Dockerfile — FastAPI 추론/견적/RAG API 이미지
#
#   빌드 컨텍스트는 "저장소 루트"입니다. ai/(RAG 지식 문서)가 루트에 있어서
#   backend/ 로 좁히면 COPY가 실패합니다.
#     docker build -f backend/Dockerfile -t cardoc-backend .
#   보통은 루트에서 `docker compose up --build` 로 frontend와 함께 실행합니다.
#
# ※ 앞의 명령들을 frontend/Dockerfile과 "글자 단위로 동일하게" 유지하고 있습니다.
#   Docker는 명령이 같으면 레이어를 재사용하므로 torch(약 900MB) 레이어를
#   두 이미지가 공유합니다. 디스크가 빠듯한 Codespaces에서 특히 중요합니다.

FROM python:3.12-slim

# OpenCV(cv2) 실행에 필요한 시스템 라이브러리 + healthcheck용 curl.
# ultralytics가 GUI판 opencv-python을 함께 설치하는데, 이 판은 X11/OpenGL을
# 요구하므로 slim 이미지에서는 아래 패키지가 없으면 `import cv2`가 실패합니다.
#   ImportError: libxcb.so.1: cannot open shared object file
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libxcb1 \
        libxext6 \
        libsm6 \
        libxrender1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch를 CPU 전용 인덱스로 먼저 설치합니다.
# 기본 PyPI판은 CUDA 런타임까지 끌고 와서 이미지가 약 2.5GB 커집니다.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision

# 의존성 먼저 COPY → 소스만 바뀌면 pip 레이어는 캐시 재사용 (강의자료 p.64~65)
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# RAG 지식 문서(루트 ai/) + 백엔드 소스
COPY ai/ ai/
COPY backend/ backend/

# RAG용 FAISS 인덱스를 이미지 빌드 시점에 미리 생성합니다.
# 첫 /chat 요청에서 임베딩 모델(약 450MB)을 내려받느라 수십 초 멈추는 것을 방지.
# ai/build_vectorstore.py는 backend/data/vectorstore/ 에 저장하고,
# services/rag.py도 같은 경로를 읽습니다.
# 빌드 환경에 네트워크가 없어도 빌드는 계속되도록 실패를 무시합니다.
ENV HF_HOME=/app/.cache/huggingface
RUN python ai/build_vectorstore.py \
    || echo "[build] 인덱스 사전 생성 실패 — /chat은 Markdown 키워드 RAG로 동작합니다"

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=90s --retries=6 \
    CMD curl -f http://localhost:8000/health || exit 1

# main.py가 `from routers import ...` 형태로 import하므로 실행 위치는 backend/.
# 단, services/rag.py와 detector.py는 BASE_DIR을 "backend의 부모(=/app)"로
# 계산하므로 /app/runs, /app/backend/data/vectorstore 가 그대로 맞습니다.
WORKDIR /app/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
