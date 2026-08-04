"""
CarDD(cardd_raw) 라벨(bbox+폴리곤 혼합)에서 손상 부위를 crop하여
2단계 손상 종류 분류기(dent/scratch/crack/glass shatter/lamp broken/tire flat) 학습용
이미지 분류 데이터셋(damage_type_crops/{train,val,test}/<class>/*.jpg) 생성.

- cardd_raw/train(9,021장)을 이미지 단위로 층화 재분할해 train/val 분리 (val 없는 원본 보완)
- cardd_raw/test는 그대로 test로 사용
- 폴리곤 라벨은 min/max 좌표로 axis-aligned bbox 변환 후 crop
- crop 시 여백(PAD_RATIO) 추가

(2026-08-04: crack처럼 원래 작은 박스에 최소 절대 crop 크기(128px)를 강제하는 방식을 시도했으나,
 배경/주변 차체가 과하게 딸려 들어가 신호가 희석되어 test 정확도가 오히려 하락(0.8345→0.7811,
 특히 crack F1 0.710→0.391)함을 확인 — 원래의 비율 기반 padding으로 되돌림.
 상세 경위는 DAMAGE_TYPE_CLASSIFIER.md 참고)

사용법: python build_damage_type_crops.py
"""
import random
from collections import defaultdict
from pathlib import Path
from PIL import Image

SOURCE_DIR   = Path("cardd_raw")
OUTPUT_DIR   = Path("damage_type_crops")
CLASS_NAMES  = ['crack', 'dent', 'glass shatter', 'lamp broken', 'scratch', 'tire flat']
VAL_RATIO    = 0.1
PAD_RATIO    = 0.15   # crop 여백 비율
MIN_CROP_PX  = 5
SEED         = 42

random.seed(SEED)


def parse_box(coords: list[float], img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """bbox(4개 값: cx,cy,w,h) 또는 polygon(6개 이상 짝수 개: x1,y1,...)에서 axis-aligned bbox(px) 반환"""
    if len(coords) == 4:
        cx, cy, w, h = coords
        return (
            (cx - w / 2) * img_w, (cy - h / 2) * img_h,
            (cx + w / 2) * img_w, (cy + h / 2) * img_h,
        )
    xs = [coords[i] * img_w for i in range(0, len(coords), 2)]
    ys = [coords[i] * img_h for i in range(1, len(coords), 2)]
    return min(xs), min(ys), max(xs), max(ys)


def collect_image_instances(split: str) -> list[tuple[Path, list[tuple[int, tuple]]]]:
    """이미지별로 (클래스, bbox) 인스턴스 목록 수집"""
    img_dir = SOURCE_DIR / split / "images"
    label_dir = SOURCE_DIR / split / "labels"
    items = []
    for img_path in img_dir.glob("*.*"):
        label_path = label_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            continue

        instances = []
        for line in label_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if not parts or not parts[0].isdigit():
                continue
            cls = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            if len(coords) != 4 and (len(coords) < 6 or len(coords) % 2 != 0):
                continue
            instances.append((cls, parse_box(coords, w, h)))

        if instances:
            items.append((img_path, instances))
    return items


def split_train_val(train_items):
    """이미지에 포함된 클래스 중 가장 희귀한 클래스 기준으로 그룹화 후 val 분리 (resplit_dataset.py와 동일 원리)"""
    class_img_freq = defaultdict(int)
    for _, instances in train_items:
        for c in set(c for c, _ in instances):
            class_img_freq[c] += 1

    groups = defaultdict(list)
    for img_path, instances in train_items:
        classes = set(c for c, _ in instances)
        rarest = min(classes, key=lambda c: class_img_freq[c])
        groups[rarest].append((img_path, instances))

    train_final, val_final = [], []
    for group in groups.values():
        random.shuffle(group)
        n_val = max(1, int(len(group) * VAL_RATIO)) if len(group) >= 3 else 0
        val_final.extend(group[:n_val])
        train_final.extend(group[n_val:])
    return train_final, val_final


def save_crops(items, split_name: str) -> dict:
    for cls_id in range(len(CLASS_NAMES)):
        (OUTPUT_DIR / split_name / CLASS_NAMES[cls_id]).mkdir(parents=True, exist_ok=True)

    counter = defaultdict(int)
    for img_path, instances in items:
        try:
            im = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        w, h = im.size
        for cls, (x1, y1, x2, y2) in instances:
            bw, bh = x2 - x1, y2 - y1
            x1p = max(0, x1 - bw * PAD_RATIO)
            y1p = max(0, y1 - bh * PAD_RATIO)
            x2p = min(w, x2 + bw * PAD_RATIO)
            y2p = min(h, y2 + bh * PAD_RATIO)
            if x2p - x1p < MIN_CROP_PX or y2p - y1p < MIN_CROP_PX:
                continue
            crop = im.crop((x1p, y1p, x2p, y2p))
            counter[cls] += 1
            out_name = f"{img_path.stem}_{counter[cls]:05d}.jpg"
            crop.save(OUTPUT_DIR / split_name / CLASS_NAMES[cls] / out_name, quality=95)
        im.close()
    return counter


def main():
    print("train 라벨 스캔 중...")
    train_items = collect_image_instances("train")
    print(f"라벨 있는 train 이미지: {len(train_items)}장")

    train_final, val_final = split_train_val(train_items)
    print(f"재분할 결과 -> train: {len(train_final)}장 / val: {len(val_final)}장")

    print("test 라벨 스캔 중...")
    test_items = collect_image_instances("test")
    print(f"라벨 있는 test 이미지: {len(test_items)}장")

    print("\ncrop 생성 중 (시간 소요)...")
    train_counts = save_crops(train_final, "train")
    val_counts = save_crops(val_final, "val")
    test_counts = save_crops(test_items, "test")

    print(f"\n{'클래스':<15}{'train':>8}{'val':>8}{'test':>8}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:<15}{train_counts[i]:>8}{val_counts[i]:>8}{test_counts[i]:>8}")

    print(f"\n완료. {OUTPUT_DIR}/{{train,val,test}}/<class>/*.jpg 로 저장됨.")


if __name__ == "__main__":
    main()
