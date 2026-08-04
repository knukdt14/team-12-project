"""YOLO(부위 탐지) + ResNet18(손상 종류 분류) 추론 래퍼.

TODO: app.py(프로젝트 루트)에 이미 구현된 모델 로딩/추론 로직을
      여기로 옮겨서 routers/diagnose.py에서 호출하도록 연결.
      - MODEL_PATH: runs/detect/runs/train_20260804_1124/weights/best.pt
      - DAMAGE_TYPE_MODEL_PATH: runs/damage_type_classifier/best.pt
      참고: app.py의 KOREAN_NAMES, DAMAGE_TYPE_KOREAN 매핑, src/preprocessing.py
"""
