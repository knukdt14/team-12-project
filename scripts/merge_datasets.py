"""
데이터셋 병합 스크립트
- car-dent-detection2: 우리 클래스와 15개 일치 → 전부 병합
- car-damage: 31클래스 중 우리 클래스와 일치하는 것만 추출

사용법: python merge_datasets.py
"""
import shutil
import hashlib
from pathlib import Path

# 기준 클래스 (16종). Bodypanel-Dent는 원본 데이터셋의 라벨링 오류(배경을 가리키는 1px 점)로
# 판명되어 제거됨 — 상세 경위는 docs/DEFECT_CLASSES.md, docs/REVIEW.md 참고.
OUR_CLASSES = [
    'Front-Windscreen-Damage',  # 0
    'Headlight-Damage',         # 1
    'Rear-windscreen-Damage',   # 2
    'RunningBoard-Dent',        # 3
    'Sidemirror-Damage',        # 4
    'Signlight-Damage',         # 5
    'Taillight-Damage',         # 6
    'bonnet-dent',              # 7
    'boot-dent',                # 8
    'doorouter-dent',           # 9
    'fender-dent',              # 10
    'front-bumper-dent',        # 11
    'pillar-dent',              # 12
    'quaterpanel-dent',         # 13
    'rear-bumper-dent',         # 14
    'roof-dent',                # 15
]
OUR_CLASS_MAP = {name: idx for idx, name in enumerate(OUR_CLASSES)}

# 외부 데이터셋 클래스 → 우리 클래스 매핑 (없으면 None = 제외)
DATASET_MAPS = {
    "car-dent-detection2.v2i.yolov8": {
        0:  OUR_CLASS_MAP['Front-Windscreen-Damage'],
        1:  OUR_CLASS_MAP['Headlight-Damage'],
        2:  None,   # Major-Rear-Bumper-Dent → 없음
        3:  OUR_CLASS_MAP['Rear-windscreen-Damage'],
        4:  OUR_CLASS_MAP['RunningBoard-Dent'],
        5:  OUR_CLASS_MAP['Sidemirror-Damage'],
        6:  OUR_CLASS_MAP['Signlight-Damage'],
        7:  OUR_CLASS_MAP['Taillight-Damage'],
        8:  OUR_CLASS_MAP['bonnet-dent'],
        9:  OUR_CLASS_MAP['doorouter-dent'],
        10: OUR_CLASS_MAP['fender-dent'],
        11: OUR_CLASS_MAP['front-bumper-dent'],
        12: OUR_CLASS_MAP['pillar-dent'],
        13: OUR_CLASS_MAP['quaterpanel-dent'],
        14: OUR_CLASS_MAP['rear-bumper-dent'],
        15: OUR_CLASS_MAP['roof-dent'],
    },
    "car-damage.v1i.yolov8": {
        0:  OUR_CLASS_MAP['Front-Windscreen-Damage'],
        1:  OUR_CLASS_MAP['Headlight-Damage'],
        2:  None,   # Major-Rear-Bumper-Dent
        3:  OUR_CLASS_MAP['Rear-windscreen-Damage'],
        4:  OUR_CLASS_MAP['RunningBoard-Dent'],
        5:  OUR_CLASS_MAP['Sidemirror-Damage'],
        6:  OUR_CLASS_MAP['Signlight-Damage'],
        7:  OUR_CLASS_MAP['Taillight-Damage'],
        8:  OUR_CLASS_MAP['bonnet-dent'],
        9:  None,   # damaged
        10: None,   # damaged-door
        11: None,   # damaged-front-bumper
        12: None,   # damaged-head-light
        13: None,   # damaged-hood
        14: None,   # damaged-rear-bumper
        15: None,   # damaged-rear-window
        16: None,   # damaged-tail-light
        17: None,   # damaged-trunk
        18: None,   # damaged-window
        19: None,   # damaged-windscreen
        20: None,   # dent
        21: None,   # dent-or-scratch
        22: OUR_CLASS_MAP['doorouter-dent'],
        23: OUR_CLASS_MAP['fender-dent'],
        24: OUR_CLASS_MAP['front-bumper-dent'],
        25: None,   # medium-Bodypanel-Dent → 대응 클래스 없음 (Bodypanel-Dent 제거됨)
        26: OUR_CLASS_MAP['pillar-dent'],
        27: OUR_CLASS_MAP['quaterpanel-dent'],
        28: OUR_CLASS_MAP['rear-bumper-dent'],
        29: OUR_CLASS_MAP['roof-dent'],
        30: None,   # scratch
    },
}

TARGET_IMG_DIR   = Path("data/train/images")
TARGET_LABEL_DIR = Path("data/train/labels")

def file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()

def get_existing_hashes() -> set:
    print("기존 이미지 해시 수집 중...")
    hashes = set()
    for p in TARGET_IMG_DIR.glob("*.*"):
        hashes.add(file_hash(p))
    return hashes

def merge_dataset(ds_folder: str, class_map: dict, existing_hashes: set):
    base = Path(ds_folder)
    added = skipped_dup = skipped_cls = 0

    for split in ["train", "valid", "test"]:
        img_dir   = base / split / "images"
        label_dir = base / split / "labels"
        if not img_dir.exists():
            continue

        for img_path in list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")):
            label_path = label_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                continue

            # 중복 이미지 체크
            h = file_hash(img_path)
            if h in existing_hashes:
                skipped_dup += 1
                continue

            # 라벨 변환
            new_lines = []
            for line in label_path.read_text().strip().splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                old_cls = int(parts[0])
                new_cls = class_map.get(old_cls)
                if new_cls is None:
                    continue
                new_lines.append(f"{new_cls} {' '.join(parts[1:])}")

            if not new_lines:
                skipped_cls += 1
                continue

            # 파일명 중복 방지 — 폴더 전체 경로를 해시해 접미사 충돌 방지 (split도 포함)
            ds_hash = hashlib.md5(ds_folder.encode()).hexdigest()[:8]
            new_stem = f"{img_path.stem}_{ds_hash}_{split}"
            new_img_name   = new_stem + img_path.suffix
            new_label_name = new_stem + ".txt"

            shutil.copy2(img_path, TARGET_IMG_DIR / new_img_name)
            (TARGET_LABEL_DIR / new_label_name).write_text("\n".join(new_lines))

            existing_hashes.add(h)
            added += 1

    return added, skipped_dup, skipped_cls


def main():
    existing_hashes = get_existing_hashes()
    print(f"기존 이미지 수: {len(existing_hashes)}장\n")

    total_added = 0
    for ds_folder, class_map in DATASET_MAPS.items():
        if not Path(ds_folder).exists():
            print(f"[건너뜀] {ds_folder} 폴더 없음")
            continue

        print(f"병합 중: {ds_folder}")
        added, dup, cls_skip = merge_dataset(ds_folder, class_map, existing_hashes)
        print(f"  추가: {added}장 / 중복 제외: {dup}장 / 클래스 불일치 제외: {cls_skip}장")
        total_added += added

    print(f"\n=== 완료 ===")
    print(f"총 {total_added}장 추가됨")
    print(f"이제 python train.py 로 재학습하세요.")


if __name__ == "__main__":
    main()
