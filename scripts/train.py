"""
YOLO 모델 학습 스크립트
사용법: python train.py

Roboflow에서 YOLO 포맷으로 다운로드하면 data.yaml이 자동 생성됨.
data.yaml 위치를 DATA_YAML 변수에 지정 후 실행.
"""
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

DATA_YAML = "data/data.yaml"  # 층화 재분할된 데이터셋 (2026-08-03부터 기본값으로 승격, 데이터 유출 없음)
MODEL     = "yolo11n.pt"       # v2 데이터셋 기준 공정 비교용 재학습 (기존엔 옛 분할로만 학습됨, MODEL_COMPARISON.md 참고)
EPOCHS    = 50
IMG_SIZE  = 640
PROJECT   = "runs"
RUN_NAME  = f"train_{datetime.now().strftime('%Y%m%d_%H%M')}"  # 자동 타임스탬프


def main():
    model = YOLO(MODEL)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        project=PROJECT,
        name=RUN_NAME,
        patience=10,       # 10 epoch 개선 없으면 조기 종료
        batch=16,
        workers=4,
        seed=42,
        exist_ok=True,
    )

    # ultralytics 버전에 따라 project/name 조합과 실제 저장 경로가 달라질 수 있어(예: runs/detect/runs/... 로 중첩),
    # 경로를 수동 조합하지 않고 트레이너가 실제로 사용한 save_dir을 사용
    # (model.train()의 반환값인 results는 DetMetrics라 save_dir이 없음 — model.trainer.save_dir을 써야 함)
    save_dir = model.trainer.save_dir

    print("\n=== 학습 완료 ===")
    print(f"최적 모델 저장 위치: {save_dir}/weights/best.pt")

    # 학습 결과 CSV → pandas 분석
    csv_path = save_dir / "results.csv"
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(df["epoch"], df["train/box_loss"], label="train box loss")
    axes[0].plot(df["epoch"], df["val/box_loss"],   label="val box loss")
    axes[0].set_title("Box Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP@50", color="green")
    axes[1].set_title("mAP@50")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_dir / "training_curve.png", dpi=150)
    print(f"학습 곡선 저장: {save_dir}/training_curve.png")


if __name__ == "__main__":
    main()
