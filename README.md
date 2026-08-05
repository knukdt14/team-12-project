# 차체 도장·외관 불량 자동 검출 시스템

YOLO + OpenCV + pandas를 활용한 차체 외관 불량 검출 포트폴리오 프로젝트.
1단계(부위 탐지) + 2단계(손상 종류 분류) 구조로, "어디에 + 어떤 손상"까지 알 수 있음.

## 기술 스택
- YOLO (ultralytics, 1단계 부위 탐지) — 불량 위치 탐지 + 부위 분류
- ResNet18 (torchvision, 2단계 손상 종류 분류) — 찌그러짐/스크래치/균열 등 구분 ([docs/DAMAGE_TYPE_CLASSIFIER.md](docs/DAMAGE_TYPE_CLASSIFIER.md))
- OpenCV — 결과 시각화
- pandas / matplotlib — 불량률 통계 분석
- Streamlit — 데모 UI

## 모델 성능 요약

모든 수치는 유출 없는 test셋 기준(자세한 내용은 [docs/MODEL_COMPARISON.md](docs/MODEL_COMPARISON.md), [docs/DAMAGE_TYPE_CLASSIFIER.md](docs/DAMAGE_TYPE_CLASSIFIER.md) 참고).

### 1단계 — 부위 탐지 (YOLO)

| 모델 | 경로 | 용도 | mAP50 | mAP50-95 | Precision | Recall | git 포함 |
|---|---|---|---|---|---|---|---|
| YOLO11n (16종) | `runs/detect/runs/train_20260804_1124/weights/best.pt` | **배포 중** | 0.887 | 0.750 | 0.912 | 0.824 | O |
| YOLO11n (17종, 구버전) | 체크포인트 삭제됨(수치만 기록) | 과거 기록(Bodypanel-Dent 포함) | 0.827 | 0.706 | 0.820 | 0.803 | X |
| YOLO26n (17종, 구버전) | 체크포인트 삭제됨(수치만 기록) | 비교용, 미배포(속도 1.6배 느림) | 0.830 | 0.707 | 0.835 | 0.796 | X (문서만) |

### 2단계 — 손상 종류 분류 (ResNet18)

| 모델 | 경로 | 용도 | 정확도 | Precision(macro) | Recall(macro) | F1(macro) | git 포함 |
|---|---|---|---|---|---|---|---|
| ResNet18 | `runs/damage_type_classifier/best.pt` | **배포 중** | 0.8150 | 0.815 | 0.829 | 0.819 | O |

### 지표 설명

**1단계(탐지) 지표**
- **mAP50**: IoU(겹침 정도) 0.5 기준으로 박스를 얼마나 잘 맞췄는지 나타내는 평균 정밀도. 탐지 모델의 대표 지표로 가장 널리 쓰임
- **mAP50-95**: IoU 임계값을 0.5~0.95까지 촘촘히 바꿔가며 평균낸 값. 박스 위치까지 얼마나 정확한지 더 엄격하게 평가(mAP50보다 항상 낮게 나옴)
- **Precision**: 모델이 "손상이다"라고 예측한 것 중 실제로 맞은 비율. 낮으면 오탐(False Positive)이 많다는 뜻
- **Recall**: 실제 손상 중 모델이 놓치지 않고 찾아낸 비율. 낮으면 미탐(False Negative, 놓친 손상)이 많다는 뜻 — 불량 검사에서는 이게 낮으면 특히 위험

**2단계(분류) 지표**
- **정확도(Accuracy)**: 전체 예측 중 정답을 맞춘 비율 (test 973장 중 812장 정답)
- **Precision/Recall/F1 (macro)**: 6개 클래스(dent/scratch/crack/glass shatter/lamp broken/tire flat) 각각의 지표를 단순 평균한 값. 클래스별 데이터 양 차이와 상관없이 "각 클래스를 얼마나 고르게 잘 맞히는지" 보여줌 (클래스별 상세 수치는 DAMAGE_TYPE_CLASSIFIER.md 참고)

## 데이터셋

원본은 [Roboflow — Car Dent & Scratch Detection](https://universe.roboflow.com/sindhu/car_dent_scratch_detection-1)이지만,
`scripts/merge_datasets.py`로 외부 데이터셋 2종을 추가 병합(6,140장 → 14,920장)하고,
`scripts/resplit_dataset.py`로 증강 이미지 유출을 막는 방식의 층화 재분할(train 80% / valid 10% / test 10%)까지 거친 상태입니다.

**`data/` 폴더는 용량 문제로 GitHub에 올라가 있지 않습니다(`.gitignore` 처리됨).**
아래 Google Drive 링크에서 `data.zip`을 받아 압축 해제 후, 프로젝트 루트에 생기는 `data/` 폴더를 그대로 사용하세요.

> Google Drive 링크: `https://drive.google.com/drive/folders/1la-QzMW4ebgyVjLn5z6F2hTP7OepXctb?usp=sharing`

압축을 풀면 아래 구조가 됩니다:
```
data/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

새로운 데이터셋을 추가 병합하려면 [CHANGELOG.md](docs/CHANGELOG.md)와 `scripts/merge_datasets.py` 상단의 `DATASET_MAPS`를 참고하세요.

**2단계(손상 종류 분류)용 데이터**(`damage_type_crops/`, CarDD 기반)도 마찬가지로 GitHub엔 없습니다.
`damage_type_crops.zip`은 위 `data.zip`과 같은 Google Drive 폴더에 함께 올라가 있습니다. 받아서 프로젝트 루트에 압축 해제하세요(`damage_type_crops/{train,val,test}/<class>/*.jpg` 구조로 바로 풀림).

> Google Drive 링크: `https://drive.google.com/drive/folders/1la-QzMW4ebgyVjLn5z6F2hTP7OepXctb?usp=sharing`

## 실행 순서

```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. 데모만 볼 경우 — data/ 없이 바로 실행 가능 (배포 모델: YOLO11n 16종, test mAP50 0.887)
streamlit run app.py

# 3. 재학습하려면 먼저 data.zip / damage_type_crops.zip을 받아 풀어넣은 뒤 (항상 프로젝트 루트에서 실행)
#    자세한 스크립트별 용도는 아래 "스크립트별 용도 및 재학습 방법" 참고
python scripts/train.py
python scripts/train_damage_type.py
jupyter notebook analysis.ipynb
```

## 카카오 지도 / 견적 API 실행

"정비소 찾기"(카카오맵 연동), "예상 견적" 기능은 `estimate_api.py`(FastAPI)가 필요합니다. 두 가지 방법이 있습니다.

### 방법 1 — Render에 배포된 서버 사용 (권장, 키 필요 없음)

`estimate_api.py`는 이미 Render에 배포되어 있습니다(`https://team-12-project.onrender.com`). 환경변수만 지정하면 카카오 API 키 없이 바로 사용 가능합니다.

```bash
# PowerShell
$env:ESTIMATE_API_BASE_URL="https://team-12-project.onrender.com"
streamlit run app.py

# bash
export ESTIMATE_API_BASE_URL="https://team-12-project.onrender.com"
streamlit run app.py
```

환경변수를 지정하지 않으면 기본값(`http://127.0.0.1:8000`)을 사용합니다.

> Render 무료 티어는 15분간 요청이 없으면 서버가 잠들어서, 첫 요청 응답이 30초~1분 정도 걸릴 수 있습니다.

### 방법 2 — 로컬에서 estimate_api.py 직접 실행 (개발/디버깅용)

`estimate_api.py` 코드 자체를 수정하거나 로컬에서 바로 테스트하고 싶을 때 사용합니다. 이 경우 본인 명의 Kakao REST API 키가 필요합니다(Kakao Developers에서 무료 발급, `.env.example` 참고).

```bash
chmod +x run_dev.sh   # 최초 1회
./run_dev.sh
```

`estimate_api.py`(포트 8000)를 백그라운드로 띄운 뒤 `streamlit run app.py`를 실행합니다. `Ctrl+C`로 둘 다 종료됩니다.

## 스크립트별 용도 및 재학습 방법

이 프로젝트는 **서로 독립적인 두 모델**로 구성됩니다. `app.py`가 둘 다 사용하지만, 학습 파이프라인은 완전히 분리되어 있어서 **처음부터 다시 학습하려면 두 파이프라인을 각각 실행해야 합니다** — 한쪽만 돌리면 그 모델만 갱신됩니다.

### 1단계: 부위 탐지 (YOLO)

| 스크립트 | 언제 쓰나 |
|---|---|
| `scripts/merge_datasets.py` | 외부 데이터셋을 새로 추가하고 싶을 때 (선택) — `DATASET_MAPS`에 매핑 추가 후 실행 |
| `scripts/augment.py` | 특정 클래스 데이터가 너무 적어서 증강으로 보완하고 싶을 때 (선택) |
| `scripts/resplit_dataset.py` | 데이터셋 구성이 바뀐 뒤(병합/증강 이후) train/valid/test를 유출 없이 다시 나눌 때 |
| `scripts/train.py` | **부위 탐지 모델(YOLO) 학습 — 필수.** `data/data.yaml` 기준으로 학습, `runs/detect/runs/<timestamp>/weights/best.pt` 생성 |

### 2단계: 손상 종류 분류 (ResNet18)

| 스크립트 | 언제 쓰나 |
|---|---|
| `scripts/build_damage_type_crops.py` | CarDD 원본(`cardd_raw/`)에서 손상 종류별 crop 이미지셋(`damage_type_crops/`)을 새로 만들 때(여백 비율 등 설정을 바꾸고 싶을 때) |
| `scripts/train_damage_type.py` | **손상 종류 분류기 학습 — 필수.** `damage_type_crops/`로 학습, `runs/damage_type_classifier/best.pt` 생성 |
| `scripts/eval_damage_type.py` | 학습된 분류기를 test셋으로 재검증하고 싶을 때 (선택, 클래스별 precision/recall/f1 확인) |

### 처음부터 전부 재학습하려면

```bash
# 1단계: 부위 탐지 모델
# (data.zip을 data/ 에 풀어놓은 상태에서)
python scripts/train.py

# 2단계: 손상 종류 분류기
# (damage_type_crops.zip을 damage_type_crops/ 에 풀어놓은 상태에서, 또는 cardd_raw/부터 직접 빌드하려면 build_damage_type_crops.py 먼저 실행)
python scripts/train_damage_type.py
```

두 스크립트 다 실행해야 `app.py`가 "부위 + 손상 종류"를 모두 표시합니다. `train.py`만 돌리면 부위 탐지는 갱신되지만 손상 종류 분류기는 기존 것(`runs/damage_type_classifier/best.pt`)이 그대로 쓰입니다.

## 프로젝트 구조
```
car_defect_inspection/
├── src/
│   └── preprocessing.py         # OpenCV 시각화 보조
├── scripts/                     # 학습/데이터 처리 스크립트 (항상 프로젝트 루트에서 실행)
│   ├── train.py                    # YOLO 학습 (1단계 부위 탐지)
│   ├── augment.py                  # 클래스 불균형 보완용 이미지 증강
│   ├── merge_datasets.py           # 외부 데이터셋 병합
│   ├── resplit_dataset.py          # train/valid/test 층화 재분할
│   ├── build_damage_type_crops.py  # CarDD에서 손상 종류 분류용 crop 데이터셋 생성 (2단계)
│   ├── train_damage_type.py        # 손상 종류 분류기(ResNet18) 학습 (2단계)
│   └── eval_damage_type.py         # 손상 종류 분류기 test셋 평가 (2단계)
├── docs/                         # 문서
│   ├── REVIEW.md                    # 코드 리뷰 및 수정 이력
│   ├── MODEL_COMPARISON.md          # YOLO11n vs YOLO26n 비교
│   ├── DAMAGE_TYPE_CLASSIFIER.md    # 2단계 분류기 상세 문서
│   ├── DEFECT_CLASSES.md            # 16개 탐지 클래스 정리
│   ├── CHANGELOG.md                 # 개발 변경 이력
│   └── Claude.md                    # Claude Code 작업 규칙
├── data/                         # (미포함, 별도 다운로드) 병합·재분할된 학습 데이터
├── results/                      # 결과 보관용
├── runs/                         # 학습 결과 (best.pt, 그래프) — 배포 모델만 git에 포함
├── analysis.ipynb                # 불량률 분석
├── app.py                        # Streamlit 데모 (1+2단계 통합)
└── requirements.txt
```
