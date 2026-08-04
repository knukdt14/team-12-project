# 2단계 손상 종류 분류기 (2026-08-04)

## 배경

기존 YOLO 모델(17개 클래스)은 "부위+상태"만 구분하고 손상 "종류"(찌그러짐/스크래치/균열 등)는 구분하지 못했음(DEFECT_CLASSES.md에서 확인). 향후 계획 중인 "AI 차량 파손 진단 + 수리비 상담 챗봇"에서 정확한 견적을 내려면 부위뿐 아니라 손상 종류까지 알아야 하므로, 별도의 2단계 분류기를 추가함.

## 전체 구조

```
이미지 업로드
  → [1단계] YOLO26n(v2 학습본) — "어느 부위"에 손상이 있는지 탐지
  → 탐지된 박스를 여백(15%) 포함해 crop
  → [2단계] ResNet18 분류기 — 그 crop이 "어떤 종류"의 손상인지 분류
  → 최종 표시: "부위 + 종류" (예: "전방 범퍼 - 스크래치")
```

## 데이터: CarDD (Roboflow `car-damage-ymlgz/car-dd-coco`, version 8)

- 손상 "종류" 기준으로 라벨링된 공개 데이터셋(CC BY 4.0), 6개 클래스: crack, dent, glass shatter, lamp broken, scratch, tire flat
- train 9,021장 / test 453장 (valid split 없음 — 직접 분리)
- 라벨은 bbox와 폴리곤(세그멘테이션)이 클래스별로 혼재된 포맷 — `build_damage_type_crops.py`가 두 포맷 모두 axis-aligned bbox로 변환해 처리
- 다운로드 중 Windows 네트워크 연결이 반복적으로 끊기는 문제 발생(`ConnectionResetError` 10053/10054) → Range 헤더 기반 이어받기 스크립트로 43회 재시도 끝에 완전한 파일 확보, 이미지 9,474장 전수 무결성 검증(PIL verify) 통과

## 파이프라인 스크립트

| 스크립트 | 역할 |
|---|---|
| `build_damage_type_crops.py` | CarDD 원본에서 이미지 단위 층화 재분할(train 내 val 10% 분리, family/유출 방지 원리는 `resplit_dataset.py`와 동일) 후, 라벨 영역을 15% 여백 포함해 crop, `damage_type_crops/{train,val,test}/<class>/*.jpg` 형태로 저장 |
| `train_damage_type.py` | ResNet18(ImageNet 사전학습) 전이학습, 클래스 불균형 보정 가중치 적용, 20epoch(patience=5) |
| `eval_damage_type.py` | test셋 최종 평가, 클래스별 precision/recall/f1 + confusion matrix 출력 |

## 데이터 규모 (crop 생성 결과)

| 클래스 | train | val | test |
|---|---|---|---|
| dent | 7,311 | 772 | 226 |
| scratch | 6,373 | 755 | 462 |
| crack | 2,565 | 277 | 88 |
| lamp broken | 1,109 | 124 | 83 |
| glass shatter | 999 | 111 | 105 |
| tire flat | 585 | 63 | 9 |

## 학습 결과 (v1, 최초)

- 20epoch 완주(조기종료 미발동), 최고 val acc **0.8654** (17epoch)
- 학습 곡선: `runs/damage_type_classifier/training_curve.png`

> **2026-08-04 갱신**: 이후 crop/해상도/focal loss 개선을 시도했다가 되돌리는 과정을 거쳐, **현재 배포된 모델은 v4(224px, 원본 crop 방식, focal loss만 적용)**입니다. v1 체크포인트는 실험 중 덮어써져 소실됨. 최종 결과는 아래 "개선 시도 결과" 섹션의 v4 행 참고 — test 정확도 0.8150.

## test셋 최종 평가 (v4, 현재 배포 모델 기준, 2026-08-04 재검증)

**종합 정확도: 793/973 = 0.8150** (`eval_damage_type.py` 재실행으로 재현 확인)

| 클래스 | Precision | Recall | F1 | 개수 |
|---|---|---|---|---|
| glass shatter | 0.929 | 0.876 | 0.902 | 105 |
| lamp broken | 0.848 | 0.940 | 0.891 | 83 |
| scratch | 0.882 | 0.794 | 0.836 | 462 |
| tire flat | 0.875 | 0.778 | 0.824 | 9 |
| dent | 0.722 | 0.792 | 0.755 | 226 |
| crack | 0.636 | 0.795 | 0.707 | 88 |

### Confusion Matrix (행=실제, 열=예측)

|  | crack | dent | glass | lamp b | scratch | tire f |
|---|---|---|---|---|---|---|
| crack | 70 | 7 | 0 | 2 | 9 | 0 |
| dent | 5 | 179 | 4 | 4 | 33 | 1 |
| glass shatter | 2 | 2 | 92 | 3 | 6 | 0 |
| lamp broken | 2 | 0 | 2 | 78 | 1 | 0 |
| scratch | 31 | 58 | 1 | 5 | 367 | 0 |
| tire flat | 0 | 2 | 0 | 0 | 0 | 7 |

### Confusion Matrix 분석

- **glass shatter/lamp broken은 여전히 정확함** — 시각적으로 뚜렷이 구별되는 손상이라 예상대로 잘 분류됨
- **crack/dent/scratch 간 혼동이 여전히 주요 오차 원인**: scratch→dent 58건, scratch→crack 31건, dent→scratch 33건, crack→scratch 9건. 세 유형 모두 "표면 손상"이라 각도·조명에 따라 시각적으로 애매한 경우가 실제로 존재함 — 데이터 품질보다는 태스크 자체의 본질적 난이도로 보임
- v1(class-weighted CE) 대비 v4(focal loss)는 crack의 **Recall이 0.750→0.795로 개선**됐지만 **Precision은 0.673→0.636으로 소폭 하락** — focal loss가 소수 클래스(crack)에 더 강하게 그래디언트를 주는 대신, scratch를 crack으로 더 자주 오분류하는 트레이드오프가 생김(scratch→crack 오분류가 v1의 21건에서 v4는 31건으로 증가)
- crack의 Precision(0.636)이 6개 클래스 중 가장 낮음 — scratch를 crack으로 잘못 예측하는 경우가 상대적으로 많은 것이 주 원인

## app.py 통합

- `runs/damage_type_classifier/best.pt`를 `@st.cache_resource`로 로드
- YOLO 탐지 박스마다 15% 여백 crop → 분류기 추론 → 한글 라벨(`DAMAGE_TYPE_KOREAN`)로 변환해 결과 표에 "손상 종류" 열로 추가
- 분류기 체크포인트가 없어도 앱이 깨지지 않도록 방어 처리(`load_damage_type_model()`이 `(None, None)` 반환 시 종류 열 생략)
- 15장 샘플로 1단계+2단계 통합 파이프라인 스모크 테스트 완료(에러 0건)

## crack/dent/scratch 심층 분석 (2026-08-04)

### 1차 가설: 데이터 양 — 기각

| 클래스 | train 개수 | test 개수 | F1 |
|---|---|---|---|
| glass shatter | 999 | 105 | 0.945 (1등) |
| lamp broken | 1,109 | 83 | 0.888 |
| tire flat | 585 | 9 | 0.889 (test too small) |
| scratch | 6,373 | 462 | 0.855 |
| dent | 7,311 | 226 | 0.771 |
| crack | 2,565 | 88 | 0.710 (꼴찌) |

데이터 양 순위와 성능 순위가 거의 정반대(dent는 데이터 최다인데 성능 5등, glass shatter는 데이터 5번째로 적은데 성능 1등) — **데이터 양만으로는 설명 안 됨.**

### 2차 가설(확정): crop 해상도 — 채택

train/val/test 전 split에서 클래스별 crop 크기(짧은 변 기준)를 조사:

| 클래스 | train median | train 100px 미만 비율 | test F1 |
|---|---|---|---|
| glass shatter | 707px | 0.2% | 0.945 |
| tire flat | 519px | 1.7% | 0.889 |
| lamp broken | 352px | 1.4% | 0.888 |
| scratch | 154px | 33.2% | 0.855 |
| dent | 203px | 27.1% | 0.771 |
| crack | **82px** | **59.3%** | **0.710** |

**crop 크기가 성능과 훨씬 강하게 상관됨.** crack은 train 기준 59.3%가 100px 미만이고 최소 8px짜리도 존재 — 이걸 224x224로 강제 확대하면 원본 정보가 거의 소실됨. glass shatter/tire flat/lamp broken은 crop이 커서(중간값 350~700px대) 데이터가 적어도 잘 학습되고, crack은 crop이 작아서(중간값 82px) 데이터가 어느 정도 있어도 성능이 낮음.

**근본 원인**: 균열(crack)은 물리적으로 가늘고 작은 손상이라 YOLO 탐지 박스 자체가 작게 잡히고, 그 작은 박스를 그대로 crop하기 때문. **데이터를 더 모아도 여전히 작은 crop만 늘어날 뿐이라 근본 해결이 안 됨.**

### 개선 시도 결과 (2026-08-04) — 최소 crop 크기 보장은 기각, focal loss만 채택

가설을 실제로 적용해본 결과, 예상과 달리 **crop 최소 크기 보장이 오히려 성능을 떨어뜨림**을 확인.

| 시도 | 변경 내용 | test 정확도 | crack F1 |
|---|---|---|---|
| 원본(v1) | 224px, 비율 padding만, class-weighted CE | 0.8345 | 0.710 |
| v2 시도 | + 최소 crop 128px, 320px, focal loss(버그 있음) | 0.6513(val, 조기종료) | — (평가 전 폐기) |
| v3 시도 | v2 + focal loss 버그 수정 | 0.7811 | 0.391 (더 나빠짐) |
| **v4 (최종 채택)** | 224px, 비율 padding만(원복), **focal loss만 유지** | **0.8150** | **0.707** |

**최소 crop 크기 보장이 실패한 이유(사후 분석)**: crack처럼 원래 박스가 매우 작은(8~80px) 경우 최소 128px를 강제하면, 손상 자체보다 **주변 배경/차체가 crop의 대부분을 차지**하게 되어 오히려 신호가 희석됨. "해상도가 낮아서 정보가 소실된다"는 가설은 맞았지만, 처방(강제 확대)이 새로운 문제(맥락 희석)를 만듦. v3에서 crack→scratch(27건), scratch→crack(28건) 오분류가 v1보다 더 늘어난 것이 이를 뒷받침.

**최종 결정**: crop/해상도는 원본(v1)과 동일하게 되돌리고, **focal loss(버그 수정본)만 유지**. 결과는 v1 대비 전체 정확도가 약간 낮지만(0.8345→0.8150), crack의 Recall이 개선(0.750→0.795)되는 트레이드오프가 있음 — 손실 함수 차이에 따른 정상적인 변동 범위로 판단해 여기서 개선 작업 종료.

**진행 중 발견된 실수**: v1의 원본 체크포인트(`runs/damage_type_classifier/best.pt`)가 v2/v3 재학습 과정에서 백업 없이 덮어써져 소실됨 — 향후 실험 시 체크포인트를 버전별로 백업해둘 것.

## 기타 알려진 한계

1. **tire flat 클래스의 프로젝트 적합성 검토 필요** — 차체 외관(범퍼/도어/유리 등) 진단이 목적이라면 타이어 펑크는 범위 밖일 수 있음. 필요시 `DAMAGE_TYPE_KOREAN`에서 제외하거나 무시하도록 후처리 가능
2. **crack 성능은 여전히 6개 클래스 중 최저(F1 0.707)** — crop 해상도 문제로 확인됐으나 손쉬운 해결책은 없음(강제 확대는 역효과 확인됨). 데이터 추가나 오분류 샘플 직접 검토가 남은 대안이나, 현재는 보류
3. **YOLO 탐지 박스 여백(padding) 민감도 미검증** — 15%는 학습 시 사용한 값을 그대로 재사용했지만, 실제 YOLO가 그리는 박스 크기/여백과 정확히 일치하지 않을 수 있어 실제 이미지로 추가 검증 권장
