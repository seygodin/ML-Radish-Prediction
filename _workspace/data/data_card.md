# Radish disease — data card

재현 가능한 PyTorch 데이터 파이프라인 산출 요약. 데이터 사실은 `CLAUDE.md` / `report/REPORT.md` 단일 출처를 따른다.

- seed: **42** (downsample 동결), valid는 제공 split 그대로(재분할/다운샘플 없음).
- 이미지 크기는 **JSON이 아니라 실제 파일에서** 읽음(zero-dim 라벨 43건 방어). EXIF `exif_transpose` 후 RGB. 확장자 대소문자 무시 매칭. bbox는 실제 크기 기준 픽셀 xyxy.

## Raw kept samples (스캔 후, 다운샘플 전)

| split | normal | disease_3 | disease_4 |
|-------|-------:|----------:|----------:|
| train | 11001 | 470 | 227 |
| valid | 1303 | 76 | 24 |

## Classification settings — counts (다운샘플 후)

### normal_vs_d3

| split | normal | disease_3 |
|-------|------:|------:|
| train | 470 | 470 |
| valid | 1303 | 76 |

> train normal은 disease 수에 맞춰 다운샘플(seed 고정). valid는 원 분포.

### normal_vs_d4

| split | normal | disease_4 |
|-------|------:|------:|
| train | 227 | 227 |
| valid | 1303 | 24 |

> train normal은 disease 수에 맞춰 다운샘플(seed 고정). valid는 원 분포.

### normal_d3_d4

| split | normal | disease_3 | disease_4 |
|-------|------:|------:|------:|
| train | 227 | 227 | 227 |
| valid | 1303 | 76 | 24 |

> train normal은 disease 수에 맞춰 다운샘플(seed 고정). valid는 원 분포.

## Detection — counts

torchvision detection 형식. **정상 = 음성(빈 박스)**:
- **질병(disease_3/4) = 양성**: 박스 1개(xyxy, img_size 프레임), `labels=[1]`.
- **정상(normal) = 음성**: `boxes` shape **[0,4]** 빈 텐서, `labels` shape **[0]** 빈 텐서. 이미지는 그대로 디코드·리사이즈해 네트워크를 통과하지만 GT 박스가 없어 objectness 타깃이 0이 된다.

> **왜 음성이 필요한가**: 이전 baseline은 정상에도 박스+label=1을 줘서 objectness 음성 샘플이 0이었고, 결과적으로 모든 이미지에서 objectness≈1.0("항상 질병")으로 붕괴(collapse)했다. 정상을 빈 박스 음성으로 공급해 진짜 "질병 검출"이 되도록 수정.

`include_normal=True`면 train에서 normal을 disease 수만큼 다운샘플해 음성으로 추가, valid엔 normal 전체를 음성으로 추가. (기본 `include_normal=False`는 disease만 = 음성 없음.)

| split | disease_3 (양성) | disease_4 (양성) | normal (음성, include_normal=True) |
|-------|----------:|----------:|----------:|
| train | 470 | 227 | 697 |
| valid | 76 | 24 | 1303 |

> 검증(실제 실행, img_size=512, batch_size=8, include_normal=True): train 양성 697 / 음성 697(균형), valid 양성 100 / 음성 1303(원 분포). 양성 boxes=(1,4)·labels=(1,)·label값 1, 음성 boxes=(0,4)·labels=(0,), malformed 0건. 한 배치 안에 양성/음성 혼재 확인.

## Transforms

- **classification train (default)**: RandomResizedCrop(img_size, scale 0.6–1.0) + RandomHorizontalFlip(0.5) + ColorJitter(b/c/s=0.2, h=0.02) + ToTensor + ImageNet Normalize.
- **classification train (strong, `aug="strong"`)**: RandomResizedCrop(scale 0.5–1.0) + H/VFlip(0.5) + RandomRotation(±30°) + strong ColorJitter(b/c/s=0.4, h=0.1) + TrivialAugmentWide + ToTensor + ImageNet Normalize + RandomErasing(p=0.25) — train만 영향, eval/valid는 불변.
- **classification eval**: Resize(round(img_size*256/224)) + CenterCrop(img_size) + ToTensor + ImageNet Normalize.
- **detection train**: Resize→(img_size,img_size) + box scale, HFlip(0.5, box-sync), brightness jitter, ToTensor + ImageNet Normalize, box clamp.
- **detection eval**: Resize→(img_size,img_size) + box scale, ToTensor + ImageNet Normalize.
- ImageNet mean/std = (0.485,0.456,0.406)/(0.229,0.224,0.225).

## Skipped items (조용한 누락 없음)

스킵 0건 (모든 라벨이 실제 이미지와 매칭되고 크기/박스 유효).

## Manifests

- `_workspace/data/manifest_classification.csv` — 13101 rows (전 split의 normal/d3/d4 전체; 세팅별 label과 `in_train_*` 멤버십 컬럼 포함, 좌표는 실제 크기 기준 xyxy).
- `_workspace/data/manifest_detection.csv` — 797 rows (전 split disease만).

## Public API (experiment-runner가 import)

```python
from src.data import build_classification_loaders, build_detection_loaders

def build_classification_loaders(setting, img_size=224, batch_size=32,
                                 num_workers=8, seed=42,
                                 balance_valid=False, aug="default"):
    # setting in {"normal_vs_d3","normal_vs_d4","normal_d3_d4"}
    # aug in {"default","strong"} — TRAIN augmentation strength (valid 불변)
    # -> (train_loader, valid_loader, meta)
    #   loaders yield (images FloatTensor[B,3,H,W], labels LongTensor[B])
    #   meta: {num_classes, class_names, train_counts, valid_counts, class_weights}
    #   labels: normal=0; 2-class disease=1; 3-class normal=0/d3=1/d4=2

def build_detection_loaders(img_size=512, batch_size=8, num_workers=8,
                            seed=42, include_normal=False):
    # -> (train_loader, valid_loader, meta)
    #   loaders yield (images list[FloatTensor[3,H,W]],
    #                  targets list[dict(boxes FloatTensor[N,4] xyxy, labels LongTensor[N])])
    #   disease only by default; single class (1 = disease).
    #   include_normal=True: normal images are NEGATIVES — boxes shape [0,4],
    #     labels shape [0] (empty) so objectness gets true negative targets.
```

## Known limitations

- valid disease_4 = 24장 → 해당 지표 신뢰구간 넓게 해석.
- bbox는 거친 단일 박스(면적 중앙값 ≈50%, 중앙 집중) — 병변 핀포인트 아님. 정상에도 박스 존재(분류는 박스 미사용).
- 단일 시즌(2020-10~2021-01) — 외부 일반화 한계.
- normal_d3_d4 3-class는 disease_4(train 227)에 맞춰 전 type을 동일 수로 다운샘플 → 표본 작음.
