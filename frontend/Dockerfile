# frontend/Dockerfile — Streamlit UI 이미지
#
#   빌드 컨텍스트는 "저장소 루트"입니다. app.py가 루트의 src/와
#   단가표.json 을 참조하므로 frontend/ 로 좁히면 COPY가 실패합니다.
#     docker build -f frontend/Dockerfile -t cardoc-frontend .
#   보통은 루트에서 `docker compose up --build` 로 backend와 함께 실행합니다.
#
# ※ 앞의 두 명령(FROM / RUN apt-get)을 backend/Dockerfile과 "글자 단위로 동일하게"
#   유지하고 있습니다. Docker는 명령이 같으면 레이어를 재사용하므로,
#   디스크가 빠듯한 Codespaces에서 베이스 레이어를 두 이미지가 공유합니다.
#
# ※ torch / ultralytics / transformers 는 여기 없습니다.
#   진단(YOLO+ResNet18)은 backend /diagnose 가, AI 상담(LLM)은 backend /chat 이
#   담당하므로 프론트는 HTTP 클라이언트 역할만 합니다.
#   덕분에 이미지가 약 6GB -> 400MB대로 줄었습니다.

FROM python:3.12-slim

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

# 의존성 먼저 COPY → 소스만 바뀌면 pip 레이어는 캐시 재사용 (강의자료 p.64~65)
COPY frontend/requirements.txt frontend/requirements.txt
RUN pip install --no-cache-dir -r frontend/requirements.txt

# 루트의 공용 모듈/데이터 + frontend 코드
COPY src/ src/
COPY 단가표.json .
COPY docs/ docs/
COPY frontend/ frontend/

# 진단 이력 CSV를 쓰는 위치 (compose에서 named volume으로 마운트)
ENV LOG_DIR=/app/logs
RUN mkdir -p /app/logs

# backend 컨테이너 주소 (docker-compose에서 서비스명으로 오버라이드)
ENV BACKEND_BASE_URL=http://backend:8000

EXPOSE 8501

HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# --server.address=0.0.0.0 이 없으면 컨테이너 밖에서 접속할 수 없습니다.
# Codespaces의 포트 포워딩 프록시 뒤에서 파일 업로드가 403으로 막히지 않도록
# XSRF 보호를 끕니다.
CMD ["streamlit", "run", "frontend/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--server.enableXsrfProtection=false", \
     "--browser.gatherUsageStats=false"]
