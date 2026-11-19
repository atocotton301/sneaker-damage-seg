# sneaker-damage-seg: 운동화 손상 탐지 데이터셋·모델 파이프라인

운동화 사진에서 손상 위치를 찾는 분할(segmentation) 모델을 만들기 위한 데이터 점검부터 평가까지의 전체 흐름입니다.
대상 클래스는 **뒤꿈치 안감 손상(`heel_lining_damage`)** 과 **밑창 들뜸(`sole_separation`)** 입니다.

| 단계 | 도구 | 상태 |
|---|---|---|
| ① 데이터 정리 | `tools/audit.py` | 완료, 가짜 데이터로 검증 |
| ② 라벨 품질 관리 | `docs/라벨_기준서_초안.md`, CVAT | 기준서 초안 완료 |
| ③ 학습 데이터 구성 | `tools/split.py`, `tools/coco2yolo.py` | 완료, 가짜 데이터로 검증 |
| ④ 모델 개발 | 아래 Colab 명령 | 실제 데이터 필요 |
| ⑤ 평가 및 인계 | `tools/eval_report.py` | 완료, 가짜 예측으로 검증 |

필요한 것: Python 3.10+, `pip install pillow numpy` (학습은 Colab에서 `ultralytics`).

## 0. 가짜 데이터로 전체 흐름 먼저 보기 (GPU 없이 1분)

```bash
python tools/make_toy.py --out toy
python tools/audit.py --images toy/images --coco toy/annotations.json --out toy/reports/audit
python tools/split.py --coco toy/annotations.json --near-dups toy/reports/audit/near_duplicates.csv --out toy/splits
python tools/coco2yolo.py --splits toy/splits --images toy/images --out toy/yolo
```

가짜 데이터에는 일부러 넣은 문제 7가지가 있습니다. audit가 모두 잡아야 정상입니다.

| 넣어 둔 문제 | audit 결과 |
|---|---|
| 같은 파일 복사본 | `exact_duplicate` |
| 같은 신발을 다른 번호(S099)로 저장 | `near_duplicate_cross_group` → split에서 한 묶음으로 합침 |
| EXIF 회전 사진 | `exif_rotated` → 변환 때 바로 세움 |
| COCO에는 있는데 파일 없음 | `missing_on_disk` |
| 파일은 있는데 COCO에 없음 | `not_in_coco` |
| 클래스 이름 오타 `Sole Separation ` | `class_name_variant` |
| 신발 번호 없는 파일 이름 | `no_group` |

## 1. 실제 데이터 준비

1. `docs/촬영_가이드.md`대로 집에 있는 운동화를 찍습니다 (파일 이름 규칙 중요).
2. CVAT에서 `docs/라벨_기준서_초안.md`대로 다각형을 그리고 **COCO 1.0** 으로 내보냅니다.
3. `data/images/`와 `data/annotations.json`으로 정리한 뒤 0번 명령에서 `toy`를 `data`로 바꿔 실행합니다.
4. audit의 `error`는 모두 고치고 다음 단계로 갑니다. `warn`은 읽어 보고 판단합니다.

**공개 데이터로 시작하기:** Roboflow Universe의 [Segmentation of Shoe Defects](https://universe.roboflow.com/shoe-defect-detection/segmentation-of-shoe-defects)
(CC BY 4.0, 마모·변색·흠집 3종)를 **COCO Segmentation** 형식으로 내려받은 뒤:

```bash
python tools/import_roboflow.py --zip <내려받은 zip> --out data
```

Roboflow가 나눠 둔 train/valid/test를 하나로 합치고, 이 키트의 그룹·근접중복 기준으로 다시 나눕니다.
실제 클래스 이름은 `classes.txt`에 저장되어 모든 도구가 그 이름을 씁니다(지우면 기본 2종으로 돌아감).

## 2. Colab에서 학습 (④)

`data/yolo` 폴더를 Google Drive에 올린 뒤 Colab(T4 GPU)에서:

```python
!pip install -q ultralytics
from google.colab import drive; drive.mount('/content/drive')
%cd /content/drive/MyDrive/sneaker-damage-seg
# data.yaml의 path를 Colab 경로로 바꿔 줍니다
!sed -i "s#^path: .*#path: /content/drive/MyDrive/sneaker-damage-seg/data/yolo#" data/yolo/data.yaml

# 기준 모델
!yolo segment train model=yolov8n-seg.pt data=data/yolo/data.yaml imgsz=640 epochs=50 seed=0 name=base_640

# 시험 데이터 평가 (클래스별 P, R, mAP)
!yolo segment val model=runs/segment/base_640/weights/best.pt data=data/yolo/data.yaml split=test

# 오탐·누락 분석용 예측 저장
!yolo segment predict model=runs/segment/base_640/weights/best.pt source=data/yolo/images/test save_txt=True save_conf=True name=pred_base
!python tools/eval_report.py --labels data/yolo/labels/test --preds runs/segment/pred_base/labels --out reports/eval_base_640
```

## 3. 비교 실험 계획 (작은 손상과 밑창 오탐 줄이기)

한 번에 하나만 바꿉니다. 모델 선택은 `val`로만 하고, `test`는 마지막에 한 번만 봅니다.

| 실험 | 바꾸는 것 | 확인할 것 |
|---|---|---|
| base_640 | 기준 | 클래스별 P/R, 크기별 recall |
| size_1024 | `imgsz=1024` | 작은 안감 손상 recall이 오르는지 |
| no_aug | `mosaic=0 fliplr=0` | 증강이 밑창 오탐을 늘리는지 |
| neg_more | 정상 사진 비율 늘림 | 정상 사진 오탐(`negative_image_fp`)이 줄어드는지 |

결과는 `reports/실험기록.md`에 표로 남깁니다: 실험 이름, 바꾼 것, seed, val mAP50(M), test는 최종 1개만.

## 4. 알아 둘 점

- `eval_report.py`는 다각형의 **박스**로 매칭합니다. 밑창 들뜸처럼 가늘고 긴 손상은 박스가 몇 픽셀만 어긋나도 IoU가 크게 떨어지므로, 공식 지표는 ultralytics `val`의 mask mAP를 쓰고 이 리포트는 어떤 사진이 틀렸는지 찾는 용도로 씁니다.
- 근접 중복 기준(`--near-dup-distance 20`)은 256비트 중 다른 비트 수입니다. 배경이 같은 매장 사진이 많으면 오탐이 늘 수 있으니 `near_duplicates.csv`를 몇 쌍 직접 열어 보고 조정합니다.
- 자동 초안(GPT·SAM) 라벨은 사람 검수 전까지 test에 넣지 않습니다.
