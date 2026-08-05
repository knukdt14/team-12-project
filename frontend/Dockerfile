# frontend(Streamlit) 이미지.
# app.py가 저장소 루트의 src/, repair_inpaint.py, 단가표.json 등을 참조하므로
# 빌드 컨텍스트는 반드시 "저장소 루트"여야 함:
#   docker build -f frontend/Dockerfile -t cardoc-frontend .
FROM python:3.11-slim

WORKDIR /app

# OpenCV(opencv-python)가 필요로 하는 시스템 라이브러리
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY frontend/requirements.txt frontend/requirements.txt
RUN pip install --no-cache-dir -r frontend/requirements.txt

# 루트의 공용 모듈/데이터 + frontend 코드 복사
COPY src/ src/
COPY repair_inpaint.py .
COPY 단가표.json .
COPY frontend/ frontend/

# backend 컨테이너 주소 (docker-compose에서 서비스명으로 오버라이드 권장)
ENV BACKEND_BASE_URL=http://backend:8000

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
