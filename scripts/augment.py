"""
클래스 불균형 보완용 이미지 증강 스크립트
사용법: python augment.py

200개 미만 클래스의 이미지를 증강해서 최소 200개로 맞춤.
증강된 이미지/라벨은 train/images, train/labels 에 추가됨.
"""
import cv2
import numpy as np
import os
import random
from pathlib import Path

TRAIN_IMG_DIR   = "data/train/images"
TRAIN_LABEL_DIR = "data/train/labels"
TARGET_COUNT    = 200  # 클래스당 목표 수량
SEED            = 42

CLASS_NAMES = [
    'Front-Windscreen-Damage', 'Headlight-Damage',
    'Rear-windscreen-Damage', 'RunningBoard-Dent', 'Sidemirror-Damage',
    'Signlight-Damage', 'Taillight-Damage', 'bonnet-dent', 'boot-dent',
    'doorouter-dent', 'fender-dent', 'front-bumper-dent', 'pillar-dent',
    'quaterpanel-dent', 'rear-bumper-dent', 'roof-dent'
]

random.seed(SEED)
np.random.seed(SEED)


def augment_image(img: np.ndarray) -> np.ndarray:
    """랜덤 증강 적용 — 밝기/회전/노이즈 중 1개 선택 (flip은 main에서 라벨과 함께 처리)"""
    choice = random.randint(0, 2)

    if choice == 0:
        # 밝기 조절
        factor = random.uniform(0.6, 1.4)
        img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    elif choice == 1:
        # 회전 (±15도)
        h, w = img.shape[:2]
        angle = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h))

    elif choice == 2:
        # 가우시안 노이즈
        noise = np.random.normal(0, 10, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img


def flip_labels_horizontal(lines: list) -> list:
    """좌우 반전 시 바운딩박스 x 좌표 보정"""
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            cls, x, y, w, h = parts
            x = str(round(1.0 - float(x), 6))
            new_lines.append(f"{cls} {x} {y} {w} {h}")
        else:
            new_lines.append(line)
    return new_lines


def count_class_in_labels(label_dir: str) -> dict:
    counts = {i: 0 for i in range(len(CLASS_NAMES))}
    for path in Path(label_dir).glob("*.txt"):
        for line in path.read_text().strip().splitlines():
            parts = line.strip().split()
            if parts and parts[0].isdigit():
                counts[int(parts[0])] += 1
    return counts


def get_images_with_class(cls_id: int, label_dir: str, img_dir: str) -> list:
    """특정 클래스를 포함하는 이미지 경로 목록 반환"""
    result = []
    for label_path in Path(label_dir).glob("*.txt"):
        for line in label_path.read_text().strip().splitlines():
            if line.strip().startswith(str(cls_id) + " "):
                stem = label_path.stem
                for ext in [".jpg", ".jpeg", ".png"]:
                    img_path = Path(img_dir) / (stem + ext)
                    if img_path.exists():
                        result.append((img_path, label_path))
                        break
                break
    return result


def main():
    print("클래스별 현재 수량 확인 중...")
    counts = count_class_in_labels(TRAIN_LABEL_DIR)

    low_classes = {k: v for k, v in counts.items() if v < TARGET_COUNT}
    print(f"\n200개 미만 클래스: {len(low_classes)}개")
    for cls_id, cnt in sorted(low_classes.items(), key=lambda x: x[1]):
        print(f"  [{cls_id:2d}] {CLASS_NAMES[cls_id]:<30} {cnt}개 → {TARGET_COUNT}개 목표")

    total_generated = 0

    for cls_id, current_count in low_classes.items():
        if current_count == 0:
            print(f"\n[{CLASS_NAMES[cls_id]}] 원본 이미지 없음 — 건너뜀")
            continue

        need = TARGET_COUNT - current_count
        candidates = get_images_with_class(cls_id, TRAIN_LABEL_DIR, TRAIN_IMG_DIR)

        if not candidates:
            print(f"\n[{CLASS_NAMES[cls_id]}] 이미지 없음 — 건너뜀")
            continue

        print(f"\n[{CLASS_NAMES[cls_id]}] {need}개 생성 중...")
        generated = 0

        while generated < need:
            img_path, label_path = random.choice(candidates)
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            lines = label_path.read_text().strip().splitlines()

            # 증강 적용
            choice = random.randint(0, 3)
            if choice == 0:
                aug_img = cv2.flip(img, 1)
                aug_lines = flip_labels_horizontal(lines)
            else:
                aug_img = augment_image(img)
                aug_lines = lines

            # 저장
            suffix = f"_aug{cls_id}_{generated:04d}"
            new_img_name  = img_path.stem + suffix + img_path.suffix
            new_label_name = label_path.stem + suffix + ".txt"

            cv2.imwrite(str(Path(TRAIN_IMG_DIR) / new_img_name), aug_img)
            (Path(TRAIN_LABEL_DIR) / new_label_name).write_text("\n".join(aug_lines))

            generated += 1
            total_generated += 1

        print(f"  완료: {generated}개 생성")

    print(f"\n=== 증강 완료 ===")
    print(f"총 {total_generated}개 이미지 추가됨")
    print(f"이제 python train.py 로 재학습하세요.")


if __name__ == "__main__":
    main()
