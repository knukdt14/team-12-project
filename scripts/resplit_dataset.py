"""
데이터셋 재분할 스크립트 (train/valid/test 통합 후 층화 재분할)
사용법: python resplit_dataset.py

기존 data/train, data/valid, data/test 는 건드리지 않고,
data/train_v2, data/valid_v2, data/test_v2 에 새로 생성함.
(원본 폴더와 교체하려면 재분할 결과 확인 후 수동으로 rename)

처리 방식:
1. train/valid/test 전체 이미지를 모아 "family" 단위로 그룹화.
   - augment.py가 만든 증강본(원본 stem + "_aug{cls}_{n}")은 원본과 거의 동일한 이미지이므로,
     원본과 증강본을 같은 family로 묶어 항상 같은 split에만 배정 (data leakage 방지).
2. family가 포함한 클래스 중 전역적으로 가장 희귀한 클래스를 "병목 클래스"로 판단,
   병목 클래스 기준으로 그룹화한 뒤 각 그룹 내에서 SPLIT_RATIOS 비율로 train/valid/test 배정.
   -> 희귀 클래스도 valid/test에 비례해서 들어가도록 보장.
"""
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

SOURCE_SPLITS   = ["train", "valid", "test"]
DATA_DIR        = Path("data")
OUTPUT_SUFFIX   = "_v2"
SPLIT_RATIOS    = {"train": 0.8, "valid": 0.1, "test": 0.1}
SEED            = 42

AUG_SUFFIX_RE = re.compile(r"_aug\d+_\d+$")

CLASS_NAMES = [
    'Front-Windscreen-Damage', 'Headlight-Damage',
    'Rear-windscreen-Damage', 'RunningBoard-Dent', 'Sidemirror-Damage',
    'Signlight-Damage', 'Taillight-Damage', 'bonnet-dent', 'boot-dent',
    'doorouter-dent', 'fender-dent', 'front-bumper-dent', 'pillar-dent',
    'quaterpanel-dent', 'rear-bumper-dent', 'roof-dent'
]

random.seed(SEED)


def family_key(stem: str) -> str:
    """증강 접미사를 제거해 원본과 증강본이 같은 family key를 갖도록 함"""
    return AUG_SUFFIX_RE.sub("", stem)


def collect_all_members() -> dict:
    """전체 소스 split을 훑어 family_key -> [(img_path, label_path), ...] 딕셔너리 생성"""
    families = defaultdict(list)
    for split in SOURCE_SPLITS:
        img_dir = DATA_DIR / split / "images"
        label_dir = DATA_DIR / split / "labels"
        if not img_dir.exists():
            continue
        for img_path in list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg")):
            label_path = label_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                continue
            fk = family_key(img_path.stem)
            families[fk].append((img_path, label_path))
    return families


def get_family_classes(members: list) -> set:
    """family에 속한 모든 이미지의 라벨을 합쳐 클래스 집합 반환"""
    classes = set()
    for _, label_path in members:
        for line in label_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if parts and parts[0].isdigit():
                classes.add(int(parts[0]))
    return classes


def main():
    print("전체 이미지 스캔 및 family 그룹화 중...")
    families = collect_all_members()
    print(f"총 family 수: {len(families)}개 (원본+증강본 묶음 기준)")

    family_classes = {fk: get_family_classes(members) for fk, members in families.items()}

    # 클래스별 family 빈도 계산 (병목 클래스 판단용)
    class_family_freq = defaultdict(int)
    for classes in family_classes.values():
        for c in classes:
            class_family_freq[c] += 1

    # family를 "가장 희귀한 클래스" 기준으로 그룹화
    bottleneck_groups = defaultdict(list)
    for fk, classes in family_classes.items():
        if not classes:
            bottleneck_groups[-1].append(fk)  # 라벨 없는 family (배경 이미지)
            continue
        rarest = min(classes, key=lambda c: class_family_freq[c])
        bottleneck_groups[rarest].append(fk)

    # 그룹별로 80/10/10 분할
    split_assignment = {}  # family_key -> split name
    for cls_id, fks in bottleneck_groups.items():
        fks = fks[:]
        random.shuffle(fks)
        n = len(fks)
        n_train = int(n * SPLIT_RATIOS["train"])
        n_valid = int(n * SPLIT_RATIOS["valid"])
        for i, fk in enumerate(fks):
            if i < n_train:
                split_assignment[fk] = "train"
            elif i < n_train + n_valid:
                split_assignment[fk] = "valid"
            else:
                split_assignment[fk] = "test"

    # 결과 폴더 생성 및 복사
    out_counts = defaultdict(int)
    class_counts_per_split = {s: defaultdict(int) for s in SPLIT_RATIOS}

    for split in SPLIT_RATIOS:
        (DATA_DIR / f"{split}{OUTPUT_SUFFIX}" / "images").mkdir(parents=True, exist_ok=True)
        (DATA_DIR / f"{split}{OUTPUT_SUFFIX}" / "labels").mkdir(parents=True, exist_ok=True)

    for fk, members in families.items():
        target_split = split_assignment[fk]
        out_img_dir = DATA_DIR / f"{target_split}{OUTPUT_SUFFIX}" / "images"
        out_label_dir = DATA_DIR / f"{target_split}{OUTPUT_SUFFIX}" / "labels"

        for img_path, label_path in members:
            dst_img = out_img_dir / img_path.name
            dst_label = out_label_dir / label_path.name
            if dst_img.exists():
                continue  # 동일 파일명이 이미 소스 split 중복 스캔으로 잡힌 경우 스킵
            shutil.copy2(img_path, dst_img)
            shutil.copy2(label_path, dst_label)
            out_counts[target_split] += 1

        for c in family_classes[fk]:
            class_counts_per_split[target_split][c] += 1

    print("\n=== 재분할 결과 ===")
    total = sum(out_counts.values())
    for split in SPLIT_RATIOS:
        pct = out_counts[split] / total * 100 if total else 0
        print(f"{split}: {out_counts[split]}장 ({pct:.1f}%)")

    print("\n=== 클래스별 split 분포 (family 기준 인스턴스 수) ===")
    print(f"{'클래스':<28}{'train':>8}{'valid':>8}{'test':>8}")
    for cid, name in enumerate(CLASS_NAMES):
        row = [class_counts_per_split[s][cid] for s in SPLIT_RATIOS]
        print(f"{name:<28}{row[0]:>8}{row[1]:>8}{row[2]:>8}")

    print(f"\n완료. data/train{OUTPUT_SUFFIX}, data/valid{OUTPUT_SUFFIX}, data/test{OUTPUT_SUFFIX} 생성됨.")
    print("기존 data/train, data/valid, data/test 는 변경되지 않았습니다.")


if __name__ == "__main__":
    main()
