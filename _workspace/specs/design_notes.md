# Baseline 실험 설계 노트 (ml-researcher)

목적: `baseline/`의 백본을 **as-is**로 사용해 무(radish) 질병 분류·detection의 baseline 성능을
공정하게 측정한다. 모든 사양은 `_workspace/specs/exp_*.yaml`에 동결되어 있고,
experiment-runner가 해당 yaml만 보고 단독 실행 가능하다.

## 백본 (3종)
- `convnextv2` (variant `tiny`, pooled feat dim **768**)
- `efficientnetv2` (variant `s`, pooled feat dim **1280**)
- `nextvit` (variant `small`, pooled feat dim **1024**)
- **MambaVision 제외**: `mamba_ssm` 빌드 불가(이 환경). 4종 → 3종.
- 모두 **from-scratch**(pretrained 미로딩). `baseline/*.py`의 `*ForImageClassification`
  (SAFE 미사용) 래퍼를 그대로 사용. `*_v2`(SAFE)는 "Ours"용이라 baseline에서 제외.

## 모델 빌더 (`src/models/`)
- `build_classifier(arch, num_classes, img_size=224)` → 해당 baseline 분류 래퍼.
  forward(images, labels=None) → logits[B,C] (labels 주면 (loss, logits), 내장 CE).
- `build_detector(arch, img_size=512, with_objectness=True)` → `SingleBoxDetector`.
  backbone.forward_features(x)=(B,C) pooled → neck → box_head: **xyxy in [0,1]** (sigmoid).
  with_objectness면 (pred_boxes[B,4], obj_logit[B]) 반환.
- 두 파일 모두 `__main__` 스모크 포함, 실제 실행해 통과 확인(아래 결과).

### 파라미터 수 (스모크 측정)
| arch | classifier(2-class) | detector(single-box+obj) |
|------|--------------------:|--------------------------:|
| convnextv2     | 28.26M | 28.07M |
| efficientnetv2 | 20.84M | 20.51M |
| nextvit        | 31.27M | 31.00M |

## 비교 매트릭스

### 분류 (3 백본 × 3 세팅 = 9 run)
| 세팅 | 클래스 | num_classes | primary metric |
|------|--------|:-----------:|----------------|
| normal_vs_d3 | normal / disease_3 | 2 | PR-AUC |
| normal_vs_d4 | normal / disease_4 | 2 | PR-AUC |
| normal_d3_d4 | normal / disease_3 / disease_4 | 3 | macro PR-AUC (OvR) |

각 세팅에 convnextv2 / efficientnetv2 / nextvit → 9개 yaml.

### Detection (3 백본 = 3 run)
| arch | task | head |
|------|------|------|
| convnextv2 / efficientnetv2 / nextvit | single-box 회귀 | pooled feat → box[4]∈[0,1] (+objectness) |

총 **12 run = 9 분류 + 3 detection**.

## 하이퍼파라미터 근거 (소표본·from-scratch)
- **공정 비교**: 12 run 전부 동일 seed(42), optimizer(AdamW lr=3e-4 wd=0.05),
  cosine sched + warmup 5ep, epochs 60. 백본만 변수.
- **batch**: 분류 64(img 224), detection 8(img 512, 메모리).
- **lr 3e-4 / wd 0.05**: from-scratch ViT/ConvNeXt 계열의 보편적 안전값. pretrained가
  없어 lr을 더 키우면 소표본에서 발산 위험 → 보수적으로 고정.
- **증강**: 데이터 로더가 처리(RandomResizedCrop 0.6–1.0, HFlip, ColorJitter,
  ImageNet normalize). from-scratch·소표본이라 강증강이 과적합 억제에 핵심.
- **epochs 60 + early stop(patience 15)**: 다운샘플로 train 표본이 작아(2-class d4=227×2,
  3-class=227×3) 빠르게 과적합 → early stop on val primary metric.
- **클래스 균형**: 다운샘플로 type 간 표본 수를 맞췄으므로(data_card) loss는 기본 CE.
  meta['class_weights']는 보조로만 노출(weighted CE는 ablation 여지, baseline은 plain CE).

## Detection baseline 근거 (왜 single-box 회귀인가)
- 백본의 `forward_features`는 **공간 맵이 아니라 풀링된 (B,C) 벡터**를 반환 → FPN/anchor를
  얹을 공간 정보가 없다. 백본을 as-is로 쓰는 가장 합리적 detection은 풀링 특징 → linear box 회귀.
- 데이터 사실: **이미지당 박스 정확히 1개**, 박스는 거친 중앙 영역(면적 중앙값 ≈50%).
  → 단일 박스 회귀가 라벨 구조와 정확히 일치. anchor/NMS 불필요.
- **loss = GIoU + L1 (+objectness BCE)**. box loss는 gt 박스가 있는 이미지(질병)에만 적용
  (`box_loss_on: positive_only`). objectness는 질병/정상 구분(include_normal=True로 정상 음성
  공급). GIoU(weight 2)로 겹침, L1(weight 5)로 좌표 안정화.
- 정상에도 유사 크기 박스가 있어 "박스 존재"만으로 질병을 못 가린다(REPORT §6) →
  objectness는 분류 보조 신호로만 해석.

## eval-reporter와 합의할 metric 정의
**분류**
- `pr_auc`: 질병 점수(2-class는 disease logit softmax, 3-class는 OvR per-class) 기반 PR-AUC.
  3-class primary = macro PR-AUC.
- `macro_f1`: argmax 예측의 클래스 평균 F1.
- `recall_disease` / `precision_disease`: 질병 클래스(2-class label=1, 3-class d3·d4) 재현율/정밀도.
- `confusion`: argmax 혼동행렬.
- **accuracy 단독 보고 금지**(불균형 valid: normal이 압도). valid는 원분포(다운샘플 안 함)라
  PR-AUC·recall이 핵심. valid disease_4=24장 → d4 지표 신뢰구간 넓게 해석.

**Detection**
- `iou_distribution`: 예측 박스 vs gt 박스 IoU의 분포(median/mean/분위수).
- `iou_at_0.5_presence`: 질병 이미지에서 IoU≥0.5 비율 = "거친 영역 존재 검출" 성공률. **primary**.
- `map_at_0.5`: 단일 클래스 mAP@0.5(objectness score 랭킹). 거친 박스라 관대하게 나올 수 있으니
  IoU 분포·presence와 분리 해석.

## 한계
- **from-scratch**: pretrained 미로딩 → 소표본에서 성능 상한이 낮다. baseline 절대수치보다는
  백본 간 상대 비교·Ours 대비 기준선으로 해석.
- **소표본 다운샘플**: 3-class train은 type당 227장(=681 total). valid disease_4=24장.
- **MambaVision 제외**로 백본 4종 계획이 3종.
- **거친 단일 박스**: detection은 병변 핀포인트가 아닌 작물 영역 학습에 가까움(REPORT §6).
  미세 병변 국소화가 진짜 목표면 분류+CAM 대안이 더 적합(Ours 후보).
- **단일 시즌(2020-10~2021-01)**: 외부 도메인 일반화 검증 불가.

## 검증 결과 (실제 실행)
- `src/models/classifier.py` 스모크: 3 백본 모두 logits=(2,2), CE loss 정상, params 측정 OK.
- `src/models/detector.py` 스모크: 3 백본 모두 boxes=(2,4)∈[0,1], obj=(2,) 정상, params 측정 OK.
- `from src.models import build_classifier, build_detector` 패키지 import OK(임의 cwd에서).

## 백본 4종 추가 (from-scratch, 공정 비교)
기존 3종(convnextv2/efficientnetv2/nextvit small)은 그대로 두고, 모두 pretrained 미로딩으로 추가:

| arch | 구현 | clf params | det params | clf logits | det 출력 |
|------|------|-----------|-----------|-----------|---------|
| densenet121 | timm(num_classes=0, pretrained=False) 풀링특징(1024) + 표준 헤드 | 7.48M | 7.22M | (B,2) | boxes(B,4)∈[0,1], obj(B,) |
| resnet50 | timm 풀링특징(2048) + 표준 헤드 | 24.56M | 24.04M | (B,2) | 동일 |
| nextvit20 | `NextViTForImageClassification(model_variant='base')`, depths=[3,4,20,3], 풀링 1024 | 44.32M | 44.06M | (B,2) | 동일 |
| mamba | mambapy Vision-Mamba(d_model=256, 8층), 풀링 256 | 3.88M | 4.03M | (B,2) | 동일 |

- 표준 헤드 = `LayerNorm -> Linear(feat,512) -> GELU -> Dropout(0.1) -> Linear(num_classes)` (기존 baseline 래퍼와 동일).
- detector는 기존 `SingleBoxDetector`(백본 `forward_features(B,C)` → neck → box[4]+obj) 재사용. 4종 모두 동일 풀링특징 계약.

### mamba 구현 방식과 근거 (확정: 깨끗한 Vision-Mamba)
- `mamba_ssm` 빌드 불가 → `baseline/mambavision.py`의 `MambaVisionMixer`(=`selective_scan_fn` 의존)를 못 쓴다.
- **선택: mambapy의 `Mamba`(`MambaConfig(use_cuda=False)` → pscan 병렬스캔)로 ViT 스타일 분류기 신규 구성.**
  - 구조: Conv patch-embed(16x16, stride16) → 토큰(14×14=196 @224) + 학습 위치임베딩(해상도 다르면 bilinear interp) → `Mamba(d_model=256, n_layers=8)` → 최종 LayerNorm → 토큰 평균풀 → 표준 헤드.
  - 파일: `src/models/mamba_vision.py`.
- **MambaVision 아키텍처 복원(옵션 A)을 포기한 이유:** MambaVisionMixer는 inner 채널을 절반으로 쪼개고 dt/B/C 투영 레이아웃·z 게이트 분리 conv·스캔 후 concat 등 커스텀 규약을 쓴다. 이를 pscan의 (B,L,ED,N) 규약 위에 정확히 재현하는 것은 원본 CUDA 커널 레퍼런스 없이는 검증 불가능하고 오류 위험이 크다. 깨끗한 옵션 B는 from-scratch·수치 안정(유한 grad 확인)·동일 forward 인터페이스를 모두 만족한다.

### 새 spec 16개
`exp_<arch>_<setting>.yaml`, arch ∈ {densenet121, resnet50, nextvit20, mamba} × setting ∈ {normal_vs_d3, normal_vs_d4, normal_d3_d4, detection_singlebox}.
- 하이퍼는 기존과 동일(분류 img224/batch64, det img512/batch8 include_normal:true, AdamW lr3e-4 wd0.05 cosine warmup5 epochs60, seed42).
- arch 필드만 다름. **mamba·nextvit20는 학습 불안정 대비 `optim.amp: false`** 설정. densenet121/resnet50는 amp 미설정(기본).
- packages에 `mambapy` 추가.

## 검증 결과 — 백본 4종 추가 (실제 cuda 실행)
- classifier(2-class, img224): nextvit20/densenet121/resnet50/mamba 모두 logits=(2,2), CE loss 정상, NaN 없음.
- detector(img512): 4종 모두 boxes=(2,4)∈[0,1], obj=(2,), NaN 없음.
- mamba backward: clf/det 모두 grad-sum 유한(pscan grad 경로 정상) — 학습 가능.
- 16 spec 전부 YAML 파싱 OK(전체 28개), mamba/nextvit20 amp:false 확인.

---

# "Ours" 모델 — DINOv3 ViT-S/16 (frozen) + 2-layer MLP head

baseline는 전부 **from-scratch**(pretrained 미로딩)라 소표본·불균형·단일시즌 데이터에서 절대 성능 상한이 낮다(EXPERIMENTS §6-1). "Ours"는 **표현 학습** 방향으로 이 상한을 깬다: **자기지도(DINOv3) ImageNet 사전학습 백본을 frozen**으로 쓰고, 그 위에 작은 분류 헤드만 학습한다.

## 가설 → 메커니즘 → 예상 효과 → 측정/ablation
- **가설**: 자기지도 사전학습 DINOv3 frozen feature는 from-scratch 백본보다 무 질병 분류에 훨씬 분별력 있는 표현을 제공한다 → 동일 split·metric에서 baseline 최고치 대비 분류 성능이 크게 오른다.
- **메커니즘**: DINOv3는 대규모 이미지로 self-supervised 사전학습된 ViT-S/16. 그 pooled feature(384-d)는 잎/작물 텍스처·색·구조를 이미 잘 인코딩. **백본을 freeze**(requires_grad=False, no-grad/eval forward)하면 사전학습 지식이 그대로 보존되고(=catastrophic forgetting 없음), 소표본에 노출되는 학습 파라미터는 헤드(~0.099M)뿐이라 **과적합이 구조적으로 억제**된다. 이것이 transfer/linear-probe의 정석.
- **예상 효과**: 가장 어려운 **3-class(normal/d3/d4)에서 PR-AUC·F1-macro·disease recall 대폭 상승**. 2-class는 baseline이 이미 천장(아래 비교 기준 참조)이라 향상 여지가 적다.
- **측정/ablation**: 동일 valid split·seed42·동일 7-메트릭으로 baseline과 비교(eval-reporter, 원분포 §1 + 균형 valid §1B 모두). ablation 후보(한 번에 한 변수): ① 헤드 hidden {128/256/512}, ② 부분 unfreeze(마지막 1~2 블록만 저-LR finetune) vs full-freeze, ③ img_size 256 vs 224.

## 구조 (확정)
- 백본: `timm.create_model("vit_small_patch16_dinov3", pretrained=True, num_classes=0, img_size=256)` → pooled feat **384-d**, 21.587M params, **frozen**. pretrained_cfg: 256×256 입력, ImageNet mean/std(데이터 로더가 이미 지원 → **데이터 변경 불필요**).
- 헤드(trainable): `Linear(384,256) → GELU → Dropout(0.1) → Linear(256, num_classes)`.
- forward(images, labels=None) → logits[B,C] (labels 시 (loss,logits), 내장 CE) — baseline 래퍼와 동일 시그니처.
- 파일 `src/models/dinov3.py`, `build_classifier`의 arch 매핑에 **'dinov3'** 추가(pretrained=True·frozen=True 기본). `train()` 오버라이드로 frozen 백본은 항상 eval 유지. `trainable_parameters()` 헬퍼 제공(기존 runner의 `model.parameters()` optimizer도 동일하게 헤드만 갱신 — frozen param은 grad가 없어 미갱신).

### 파라미터 수 (실측 cuda 스모크)
| 구성 | params |
|------|-------:|
| total | 21.686M |
| backbone (frozen, requires_grad=False) | 21.587M |
| **head (trainable)** | **0.0993M (99,074)** = 전체의 0.46% |

스모크 출력: feat=(2,384), logits=(2,3), CE loss 정상, backbone any-requires_grad=False, backward 후 grad 받는 param 수 = 헤드 trainable과 정확히 일치(99,074), `.train()` 후에도 backbone.training=False.

## 하이퍼파라미터 (frozen probe 근거)
- baseline와 동일 seed(42)·동일 valid split·동일 ImageNet norm. **변수는 백본(자기지도 pretrained frozen)과 그에 맞춘 헤드 학습 레시피뿐.**
- **lr 1e-3**(baseline 3e-4보다 큼): 학습 대상이 작은 헤드뿐이고 frozen feature는 안정적이라 더 큰 lr로 빠르게 수렴.
- **epochs 30**(baseline 60보다 적음) + early stop(patience 10): frozen feature 위 probe는 빠르게 수렴하고 과적합 여지가 작다. AdamW wd 0.05, cosine + warmup 3, label_smoothing 0.
- img_size 256(DINOv3 pretrained_cfg 기준), batch 64. amp 기본(헤드만 학습이라 수치 안정).

## "+20%" 비교 기준 (명시)
- **비교 대상**: 동일 setting의 from-scratch baseline **최고치**(EXPERIMENTS §1 원분포 / §1B 균형 valid, 주지표 **PR-AUC**, 보조 **F1-macro**). seed42·동일 valid split·동일 metric 계산식(eval-reporter 합의)으로 공정 비교.
- **어디서 20% 여지가 큰가**: 2-class는 baseline이 이미 천장(원분포 d3 PR-AUC 최고 0.965=ResNet50, d4 0.867; 균형 valid d3 F1 0.980·AUROC≈1.0)이라 **상대 +20% 여지가 거의 없다**(상한 1.0). 따라서 **주 비교 셀 = 3-class(normal_d3_d4)**, 여기가 baseline의 약점:
  - **원분포 §1-3**: 3-class PR-AUC 최고 = **DenseNet121 0.570**, F1-macro 최고 = DenseNet121 0.709. → +20% 목표 = **PR-AUC ≥ 0.684**, **F1-macro ≥ 0.851**.
  - **균형 valid §1B-3**: 3-class 최고 = DenseNet121(acc 0.847·F1 0.850·AUROC 0.954·PR-AUC 0.860). → +20% 목표 = **F1-macro ≥ 1.02(상한 근접)**, **PR-AUC ≥ 1.0** — 균형 valid 3-class는 DenseNet121이 이미 높아 상대 20%는 상한에 막힘. 따라서 **원분포 3-class PR-AUC/F1-macro를 1차 판정 지표**로 삼고, 균형·2-class는 보조로 함께 보고.
  - **disease recall(3-class, §1-4)**: baseline 최고 DenseNet121 d3 0.80/d4 0.83 → 추가 향상 시 임상적으로 의미.
- **판정**: 위 1차 지표(원분포 3-class PR-AUC 및/또는 F1-macro)에서 baseline 최고 대비 상대 +20% 이상이면 목표 달성. 단, valid d4=24장 소표본이라 단일값 우열은 CI와 함께 해석(eval-reporter).

## 한계
- DINOv3 ViT-S는 **자연영상 self-supervised**라 작물 도메인과 갭이 있을 수 있음(frozen이라 도메인 적응 없음). 갭이 크면 ablation②(부분 unfreeze 저-LR)로 완화 여지.
- 입력 256×256·patch16(=16×16 토큰)이라 **미세 병변**은 패치 해상도 한계로 흐려질 수 있음(고해상 멀티스케일은 후속 ablation).
- valid disease_4=24장 소표본 한계는 baseline과 동일 — 단일값 우열 판단 금지.
- pretrained 가중치는 HF 다운로드 의존(실행 환경 네트워크 필요). 스모크에서 다운로드·로드 정상 확인.

## 검증 결과 (실제 cuda 실행)
- `src/models/dinov3.py` 스모크: pretrained 로드 + cuda forward, feat=(2,384)/logits=(2,3)/CE loss 정상, backbone frozen(requires_grad=False), trainable=head=99,074(0.46%).
- `build_classifier('dinov3', nc, img256)` 경로: logits shape 정상, backward 후 헤드만 grad 수령(99,074), `.train()`에도 backbone eval 유지 → 기존 `train.py`(model.parameters() optimizer)로도 헤드만 학습됨.
- 3 spec(`exp_dinov3_{normal_vs_d3,normal_vs_d4,normal_d3_d4}.yaml`) YAML 파싱 OK.

---

# "Ours+" 모델 — DINOv3 ViT-B/16 (frozen) @512 + 2-layer MLP head

기존 Ours(DINOv3 ViT-S/16 @256, 384-d)가 **3-class 원분포 PR-AUC에서 baseline 대비 +20%(0.689 vs 0.570)를 달성**했으나, **F1-macro는 0.698로 baseline(0.709)과 동급에 머물렀다**(EXPERIMENTS §6.3). 한계 진단(§6.4)은 두 가지: ① 256×256·patch16(=16×16 토큰)의 **저해상도라 미세 병변 표현이 흐려짐**, ② ViT-S의 **표현 용량 한계**. Ours+는 **frozen·no-forgetting 원칙은 그대로 유지**한 채 백본·입력·헤드를 키워 이 두 한계를 직접 공략한다.

## 가설 → 메커니즘 → 예상 효과 → 측정/ablation
- **가설**: 더 큰 자기지도 백본(ViT-S→**ViT-B**, feat 384→**768**)과 고해상도 입력(256→**512**, 토큰 16²→**32²=1024개**)은 미세 병변의 표현력을 높여, 특히 가장 어려운 **3-class에서 F1-macro·disease recall을 개선**한다(small에서 PR-AUC는 이미 목표 달성, F1은 미달이었음).
- **메커니즘**: DINOv3 ViT-B/16은 ViT-S 대비 임베딩 차원·헤드·깊이가 커 더 풍부한 자기지도 표현을 제공한다. 입력을 512로 키우면 patch16 기준 토큰이 4배(32×32)로 늘어 **잎 표면의 미세 병변·국소 텍스처가 더 잘 보존**된다. 백본은 여전히 **완전 동결**(requires_grad=False, no-grad/eval forward) → 사전학습 지식 보존, 학습 파라미터는 헤드(~0.40M)뿐이라 소표본 과적합은 구조적으로 억제. 헤드 hidden을 small(256)보다 키운 **512**로 둬 768-d feature의 분별 정보를 더 잘 활용.
- **예상 효과**: 3-class 원분포에서 F1-macro·disease(d3/d4) recall 상승, PR-AUC는 small 대비 동급 이상. 2-class는 small/baseline 모두 이미 천장(PR-AUC 0.99 근처)이라 향상 여지 작음(보조로만 보고).
- **측정/ablation**: 동일 valid split·seed42·동일 7-메트릭으로 (a) **small Ours**(§6)와 (b) **baseline 최고**(§1 원분포/§1B 균형)에 동시 비교(eval-reporter). 1차 판정 지표는 **3-class 원분포 PR-AUC·F1-macro**(small과 동일 기준). ablation 후보(한 번에 한 변수): ① img_size 512 vs 256(해상도 기여 분리), ② 헤드 hidden {256 vs 512}, ③ 부분 unfreeze(마지막 1~2 블록 저-LR).

## small 대비 변경점 (요약)
| 항목 | Ours (small) | Ours+ (base) |
|------|---|---|
| 백본 | vit_small_patch16_dinov3 (21.6M) | **vit_base_patch16_dinov3 (85.6M)** |
| pooled feat | 384-d | **768-d** |
| 입력 | 256×256 (16² 토큰) | **512×512 (32² 토큰)** |
| 헤드 hidden | 256 | **512** |
| trainable(헤드) | 99,074 (0.46%) | **395,267 (0.46% of 86.0M)** |
| epochs | 30 | **40** (wider head, 약간 더) |
| 학습 대상 | 헤드만(frozen backbone) | **헤드만(frozen backbone)** — 동일 |

- 공통 불변: arch 매핑 `dinov3_base` 추가(`build_classifier`), forward 시그니처·freeze·`train()` 오버라이드·`trainable_parameters()` 모두 small과 동일. ImageNet mean/std, 데이터 로더 변경 불필요(img_size만 512). 기존 `dinov3`(small) arch는 회귀 없이 그대로 동작.

## 구조 (확정)
- 백본: `timm.create_model("vit_base_patch16_dinov3", pretrained=True, num_classes=0, img_size=512)` → pooled feat **768-d**, 85.641M params, **frozen**. ImageNet mean/std.
- 헤드(trainable): `Linear(768,512) → GELU → Dropout(0.1) → Linear(512, num_classes)`.
- forward(images, labels=None) → logits[B,C] (labels 시 (loss,logits), 내장 CE) — baseline 래퍼·small과 동일 시그니처.
- 파일 `src/models/dinov3.py`(variant 'small'/'base' + img_size 파라미터로 일반화), `build_classifier`의 arch 매핑에 **'dinov3_base'** 추가(variant='base', hidden 512, pretrained·frozen=True 기본; spec의 `head_hidden`은 `build_classifier(head_hidden=...)`로도 전달 가능).

### 파라미터 수 (실측 cuda 스모크 @512)
| 구성 | params |
|------|-------:|
| total | 86.036M |
| backbone (frozen, requires_grad=False) | 85.641M |
| **head (trainable)** | **0.3953M (395,267, 3-class)** = 전체의 0.46% |

## 하이퍼파라미터 (frozen probe 근거)
- baseline·small과 동일 seed(42)·동일 valid split·동일 ImageNet norm. **변수는 백본 크기(S→B)·입력(256→512)·헤드 폭(256→512)·epochs(30→40)뿐.**
- **lr 1e-3 / wd 0.05 / cosine + warmup 3**: small과 동일(학습 대상이 작은 헤드뿐이라 큰 lr로 빠른 수렴). **label_smoothing 0**.
- **batch 64 @512**: 검증된 사실로 frozen base@512 batch4 forward GPU mem ≈505MB(여유 큼) → batch64도 안전, 필요 시 runner가 조정. **amp 기본**(헤드만 학습이라 수치 안정).
- **epochs 40**(small 30보다 약간 많음): 헤드가 더 넓어져(256→512) 수렴에 약간 더 여유. early stop(monitor val_pr_auc, patience 10).

## 비교 기준 (small과 동일)
- **비교 대상**: ① **small Ours**(EXPERIMENTS §6), ② 동일 setting의 from-scratch **baseline 최고**(§1 원분포 / §1B 균형, 주지표 PR-AUC, 보조 F1-macro). seed42·동일 valid split·동일 metric 계산식(eval-reporter 합의).
- **1차 판정 셀 = 3-class(normal_d3_d4) 원분포**(small과 동일). design-notes 기준 목표: **3-class 원분포 PR-AUC ≥ 0.684, F1-macro ≥ 0.851**(baseline 최고 DenseNet121 PR-AUC 0.570·F1 0.709 대비 +20%).
- **특히 주목**: small이 미달했던 **F1-macro(0.698<0.851)·disease recall**의 개선 여부가 Ours+의 핵심 평가 포인트. PR-AUC는 small이 이미 목표 달성(0.689)했으므로 **동급 이상 유지 + F1 개선**이면 Ours+의 가치 입증.
- 2-class·균형 valid는 small/baseline 모두 천장 효과(PR-AUC≈0.99~1.0)라 향상 여지 작음 → 보조 보고.

## 한계
- **연산 비용 증가**: 백본 85.6M·입력 512로 small(21.6M·256) 대비 forward 연산이 크게 늘어난다(학습은 헤드만이라 backward는 가볍지만 frozen forward 비용↑). 다만 frozen이라 feature 캐싱으로 완화 가능(후속).
- **도메인 갭 여전**: DINOv3는 자연영상 self-supervised → 작물 도메인과 갭은 small과 동일하게 존재. **frozen이라 도메인 적응이 없다**(갭이 크면 ablation③ 부분 unfreeze 저-LR로 완화 여지).
- **해상도↑가 만능은 아님**: 512로 토큰이 늘어도 GT·라벨이 거친 수준이고 valid disease_4=24장 소표본이라 F1·recall 단일값 우열은 CI와 함께 해석(eval-reporter).
- pretrained 가중치 HF 다운로드 의존(스모크에서 정상 확인). valid d4 소표본 한계는 small·baseline과 동일.

## 검증 결과 (실제 cuda 실행)
- `src/models/dinov3.py` 스모크(@512 base pretrained): 다운로드·로드 정상, feat=(2,**768**)/logits=(2,3)/CE loss 정상, backbone any-requires_grad=**False**(frozen), total **86.036M** / backbone **85.641M** / **trainable=head=0.3953M(395,267, 0.46%)**, trainable==head True.
- **small arch 회귀**: 동일 모듈에서 `variant='small'` 빌드 정상(feat=(1,384), head 0.0993M) — 회귀 없음.
- `build_classifier('dinov3_base', nc, img512)` 경로: logits=(2,3), variant='base'·img 512, head==trainable=395,267. `build_classifier('dinov3', nc, img256)`(small)도 logits 정상·variant='small' — 기존 경로 보존.
- 3 spec(`exp_dinov3_base_{normal_vs_d3,normal_vs_d4,normal_d3_d4}.yaml`) YAML 파싱 OK(arch dinov3_base, img 512, head_hidden 512, bs 64, ep 40, primary pr_auc, seed 42).

---

# "Ours+ focal+aug" — DINOv3 ViT-B/16 (frozen) @512 + 강한 증강 + focal loss

Ours+(CE, default aug)는 3-class 원분포 PR-AUC에서 baseline 대비 강했으나 **소수클래스 d4의 precision/F1이 약점**이고, 단일시즌·소표본이라 과적합 여지가 남는다. 본 변형은 **frozen·no-forgetting·동일 백본/입력/헤드/optimizer를 그대로 유지**한 채 **두 변수만** 바꾼다: ① 데이터 증강을 `strong`으로(data-engineer가 `build_classification_loaders(..., aug="strong")`로 제공), ② classification loss를 CE → **focal loss(gamma=2.0, class_weights=from_meta)**. 비교 대상인 기존 `exp_dinov3_base_*`(CE, default aug)는 **보존**한다.

## 가설 → 메커니즘 → 예상 효과 → 측정/ablation
- **가설**: (a) 강한 증강은 단일시즌·소표본의 **과적합을 줄이고 일반화를 높인다**. (b) focal loss는 다운샘플로 거의 균등해진 분포에서도 **easy한 정상(과 쉬운 d3) 샘플의 그래디언트를 down-weight**하고 hard한 d3/d4에 집중시켜, 특히 **3-class에서 소수·난해 클래스(d4)의 precision/F1·recall 균형을 개선**한다.
- **메커니즘**:
  - 증강: RandomResizedCrop·flip·color/geometry 강화(strong)로 입력 다양성↑ → frozen feature 위 작은 헤드가 표면적 단서(촬영 조건·색조)에 과적합하는 것을 억제. 백본은 frozen이라 증강은 **헤드 학습 신호의 분산만 키워** 일반화에 기여(표현 자체는 불변).
  - focal: `FL = (1-p_t)^gamma * CE` (gamma=2.0). 정상·쉬운 d3은 p_t가 빠르게 높아져 `(1-p_t)^2`가 작아짐 → 그래디언트 기여 감소. 반대로 **자주 틀리는 d4(낮은 p_t)는 modulation factor가 1에 가까워 학습 신호가 유지** → d4 결정경계가 더 선명해진다. `class_weights=from_meta`(alpha 역할)로 d4 가중을 추가 부여해 3-class에서 d4 강조.
- **예상 효과**: **3-class 원분포에서 F1-macro 상승**(특히 d4 per-class precision/F1), d3/d4 **recall과 precision의 균형 개선**, PR-AUC는 Ours+(CE) 대비 **동급 이상 유지**. 2-class(특히 d3)는 이미 천장에 가까워 향상 여지 작음(보조 보고). 증강이 과해 신호를 지나치게 흐리면 PR-AUC가 소폭 하락할 수 있어 CE 버전과 동급 유지 여부를 함께 본다.
- **측정/ablation**(한 번에 한 변수):
  - **1차 비교 기준 = 기존 `exp_dinov3_base_normal_d3_d4`(CE, default aug)** 대비 **3-class F1-macro 및 d4 per-class precision/recall** 개선. **원분포(§1) + 균형 valid(§1B) 모두** 보고(eval-reporter, seed42·동일 valid split·동일 7-메트릭).
  - PR-AUC(macro OvR)는 CE 대비 동급 이상 유지가 판정 조건. valid d4=24장 소표본 → 단일값 우열은 CI와 함께 해석.
  - **ablation 후보**: ① **gamma {1.0, 2.0, 3.0}**(focal 강도; 2.0 기본, 3.0은 과집중→소표본 분산↑ 위험), ② aug only (focal 없이 strong 증강만) vs focal only (default aug + focal) 로 두 변수의 기여 분리, ③ class_weights from_meta vs none(다운샘플로 거의 균등하므로 alpha 효과 분리), ④ label_smoothing 0 vs 0.05.

## small 대비 변경점 (Ours+ CE → Ours+ focal+aug)
| 항목 | Ours+ (CE, default aug) | Ours+ focal+aug |
|------|---|---|
| 백본/입력/헤드/optim | dinov3_base @512, hidden 512, AdamW lr1e-3 ep40 | **동일** |
| 증강 | default(RRC 0.6–1.0, HFlip, ColorJitter) | **strong** (`aug: strong`) |
| loss | cross_entropy + class_weights(from_meta) + ls0 | **focal(gamma=2.0) + class_weights(from_meta) + ls0** |
| 학습 대상 | 헤드만(frozen backbone) | **헤드만(frozen backbone)** — 동일 |
| 비교 대상 | baseline 최고 / small Ours | **기존 Ours+ CE**(동일 arch, loss·aug만 변경) |

- 변수는 **loss와 aug 둘**(혼합이지만 둘 다 "불균형·과적합 대응"이라는 한 방향). 둑 기여 분리는 위 ablation②로 별도 run에서 수행.

## train.py loss 규약 (experiment-runner 전달용 — train.py 실제 수정은 experiment-runner 담당)
현재 `run_classification`은 `F.cross_entropy(logits, labels, weight=class_weights, label_smoothing=...)`를 train/valid 양쪽에서 호출한다. focal 지원을 위해 **spec `loss.type` 분기**만 추가하면 된다:

- **spec 필드 규약** (focal 사양에 동결됨):
  - `loss.type`: `cross_entropy`(기본) | `focal`.
  - `loss.gamma`: float, focal 전용(기본 2.0). cross_entropy면 무시.
  - `loss.class_weights`: `from_meta` → `meta['class_weights']` 사용(alpha 역할), 그 외 → None.
  - `loss.label_smoothing`: float(기본 0.0), torch CE와 동일 의미.
- **wiring 방법** (둘 중 하나):
  - (권장) loss 모듈을 한 번 만들어 train/valid에 동일 적용:
    ```python
    from src.losses import build_loss  # focal만 생성; cross_entropy는 inline 유지
    if str(loss_cfg.get("type", "cross_entropy")) == "focal":
        criterion = build_loss(loss_cfg, meta["class_weights"].to(device))
        # train/valid: loss = criterion(logits, labels)
    else:
        # 기존 F.cross_entropy(logits, labels, weight=class_weights, label_smoothing=...) 유지
    ```
  - `build_loss`는 `type=='focal'`일 때만 `FocalLoss`를 반환하고, `cross_entropy`는 기존 inline 경로를 그대로 두면 된다(회귀 0).
- **API 보장**: `FocalLoss(gamma, weight=None, label_smoothing=0.0).forward(logits[B,C], targets[B]) -> scalar`. 시그니처가 `F.cross_entropy(logits, labels, ...)` 호출부와 동일 위치 인자(logits, targets)라 **호출부 한 줄 교체**로 충분. `weight`는 buffer로 등록되어 `.to(device)`로 함께 이동하나, 본 사양은 `build_loss`에 이미 `meta['class_weights'].to(device)`를 넘기므로 추가 처리 불필요.
- **snapshot 로깅 권고**: `snapshot["loss"]`에 `{"type":"focal","gamma":...,"class_weights_used":...,"label_smoothing":...}` 기록(기존 CE 분기와 대칭).

## 한계
- **두 변수 동시 변경**: focal과 strong aug를 같이 켜 단일 변수 ablation이 아니다(둘 다 과적합·불균형 대응 방향이라 묶음). 기여 분리는 ablation②(aug-only / focal-only)로 후속.
- **균형 분포에서 focal 이득 제한**: 다운샘플로 train이 거의 균등 → focal의 클래스 불균형 교정 이득은 제한적이고, 주로 **난이도 기반(easy/hard) 재가중** 효과에 의존. 효과가 작으면 CE와 동급에 머물 수 있음(그 경우 가치는 d4 precision/F1 미세 개선·과적합 억제로 한정).
- **gamma 과집중 위험**: gamma가 크면 소표본(d4=24 valid)에서 그래디언트 분산↑·불안정 → 2.0 보수적 기본, 3.0은 ablation에서만.
- **증강 과강 위험**: strong aug가 미세 병변 단서를 지우면 PR-AUC 하락 가능 → CE 버전 대비 PR-AUC 동급 유지를 판정 가드로 둠.
- valid d4=24장 소표본 한계는 Ours+·baseline과 동일 — 단일값 우열 판단 금지(CI 병기).

## 검증 결과 (실제 실행)
- `src/losses.py` FocalLoss forward/backward 실행: 2-class·3-class × gamma{0,2.0} × weighted{F,T} × ls{0,0.1} 전 조합에서 loss·grad **유한**, backward 정상.
- **gamma=0 == cross_entropy sanity**: 2-class·3-class × weighted{F,T} × ls{0,0.1} 전 조합에서 `FocalLoss(gamma=0)` 값이 `F.cross_entropy(weight=..., label_smoothing=...)`와 **diff ≤ 1.8e-7로 일치**(weighted+label_smoothing 케이스 포함; torch의 가중 label-smoothing 분해를 정확히 재현).
- 3 spec(`exp_dinov3_base_focal_{normal_vs_d3,normal_vs_d4,normal_d3_d4}.yaml`) YAML 파싱 OK: aug=strong, loss{type:focal, gamma:2.0, class_weights:from_meta, label_smoothing:0.0}, 나머지(arch dinov3_base, img512, bs64, lr1e-3, ep40, 7메트릭, primary pr_auc, seed42, smoke true) 기존과 동일.
- `build_loss(spec['loss'], class_weights)` → `FocalLoss`(gamma 2.0, weight set) 반환, 2/3-class forward·backward 정상 — train.py 호출부 한 줄 교체로 동작.

---

# Ablation (dinov3_base 3-class) — focal+aug 기여 분리

Ours+ focal+aug(`dinov3_base_focal_normal_d3_d4`)는 **focal loss와 strong aug를 동시에** 켰기 때문에, 3-class(normal_d3_d4)에서 관측된 개선이 **(a) aug 때문인지, (b) focal 때문인지, (c) 둘의 상호작용인지** 단일 run으로는 분리할 수 없다. 본 ablation은 **3-class(`normal_d3_d4`)에만** 집중(핵심 개선 지점)하여, 두 기존 사양과 **arch/입력/헤드/optimizer/seed/메트릭을 100% 동일하게 두고 aug와 loss만** 바꿔 기여를 분리한다. 새 학습 run은 **4개**(2×2의 빈 두 셀 + gamma 스윕 2개); 2×2의 두 대각 셀(CE/default, focal/strong)은 **기존 사양 재사용**(재학습 불필요).

## 2×2 설계 (aug × loss)

| | **loss = CE** | **loss = focal(g2)** |
|---|---|---|
| **aug = default (✗)** | `dinov3_base_normal_d3_d4` *(기존)* | `dinov3_base_focalonly_normal_d3_d4` **(신규 ②)** |
| **aug = strong (✓)** | `dinov3_base_augonly_normal_d3_d4` **(신규 ①)** | `dinov3_base_focal_normal_d3_d4` *(기존)* |

- 좌상(base): aug✗ loss✗ — 기준점(reference).
- 우상(focal-only ②): aug✗ loss✓ — focal만 추가.
- 좌하(aug-only ①): aug✓ loss✗ — strong aug만 추가.
- 우하(focal+aug): aug✓ loss✓ — 둘 다(기존 Ours+ focal+aug).

네 셀 모두 동일 valid split·seed42·동일 7-메트릭. 모든 신규 셀의 단일-변수 delta가 base 또는 focal+aug 기존 사양에 대해 정확히 한 축만 다르도록 구성(아래 「불변 보장」).

## gamma 스윕 {1.0, 2.0, 3.0} (strong aug 고정)

| gamma | run | 상태 |
|---|---|---|
| 1.0 | `dinov3_base_focalg1_normal_d3_d4` | **신규 ③** |
| 2.0 | `dinov3_base_focal_normal_d3_d4` | 기존(focal+aug) |
| 3.0 | `dinov3_base_focalg3_normal_d3_d4` | **신규 ④** |

세 run 모두 `aug: strong` + `class_weights: from_meta` + `label_smoothing: 0.0` 고정, **gamma만** 변경. gamma=2.0 run은 2×2 우하 셀과 동일 사양이므로 재사용 → 스윕 신규는 g1·g3 두 개.

## 기여 분리 해석 방법 (read-out)

기준 지표는 Ours+와 동일하게 **3-class 원분포 PR-AUC(macro OvR) 및 d4 per-class precision/recall/F1**(eval-reporter, 원분포 §1 + 균형 valid §1B 모두, CI 병기). 비교는 항상 동일 valid·seed42·동일 메트릭.

- **focal 효과** = `focalonly(②)` − `base(기존)`  (aug 고정=default, loss만 CE→focal). 양이면 focal 단독이 3-class를 개선.
- **aug 효과** = `augonly(①)` − `base(기존)`  (loss 고정=CE, aug만 default→strong). 양이면 strong aug 단독이 개선.
- **둘 다(합산 관측치)** = `focal+aug(기존)` − `base(기존)`.
- **상호작용(시너지/중복)** = `[focal+aug − base]` − `[(focalonly − base) + (augonly − base)]` = `focal+aug + base − focalonly − augonly`. >0이면 시너지(둘이 보완), <0이면 중복/상쇄(둘이 같은 과적합·불균형 축을 건드려 합산보다 이득이 작음 — §한계에서 예고한 "묶음" 가설과 부합).
- **gamma 민감도**: g{1,2,3}의 PR-AUC·d4 F1 곡선으로 focal 강도의 최적·강건 구간 판단. 단조 증가/감소/내부 최적 여부로 g2 기본 선택의 타당성 검증.

## 예상

- **aug 효과**: 단일시즌·소표본 과적합 억제로 PR-AUC·일반화에 **소폭(+)** 기여 예상. 단, strong aug가 미세 병변 단서를 지우면 PR-AUC 동급/소폭 하락 위험(가드: base 대비 PR-AUC 하락 시 aug 단독 기여 음수로 기록).
- **focal 효과**: train이 다운샘플로 거의 균등하므로 불균형 교정 이득은 제한적, 주로 **난이도 기반 재가중**으로 **d4 precision/F1 미세 개선**에 기여 예상(PR-AUC는 동급 가능).
- **상호작용**: 두 변수가 모두 "과적합·불균형 대응" 한 방향이라 **시너지보다 부분 중복(상호작용 ≤ 0)** 가능성이 높다고 본다 — 즉 focal+aug 합산 이득이 각 단독 이득의 단순 합보다 작을 것으로 예상. 이 경우 "한 축이 주효과(예: aug), 다른 축은 보조"라는 결론으로 귀결.
- **gamma 스윕**: g2 부근이 최적, g3은 소표본(d4 valid=24) 그래디언트 분산↑로 g2 동급 또는 소폭 하락, g1은 focal 효과 약화로 CE에 근접 — 내부 최적(g2) 또는 완만한 plateau 예상.

## 불변 보장 (4개 신규 사양)

네 사양 모두 두 기존 사양과 **arch(dinov3_base)·num_classes 3·img_size 512·source timm·pretrained·frozen_backbone·head_hidden 512·setting normal_d3_d4·batch 64·num_workers 8·AdamW lr1e-3 wd0.05 cosine warmup3 ep40·label_smoothing 0·primary pr_auc·7-메트릭·early_stop(val_pr_auc/max/patience10)·class_names·seed 42·packages·smoke true** 가 완전히 동일. 셀별로 바꾼 필드만:

| 사양 | data.aug | loss.type | loss.gamma | class_weights |
|---|---|---|---|---|
| `exp_dinov3_base_augonly_normal_d3_d4.yaml` | strong | cross_entropy | (n/a) | from_meta |
| `exp_dinov3_base_focalonly_normal_d3_d4.yaml` | default | focal | 2.0 | from_meta |
| `exp_dinov3_base_focalg1_normal_d3_d4.yaml` | strong | focal | 1.0 | from_meta |
| `exp_dinov3_base_focalg3_normal_d3_d4.yaml` | strong | focal | 3.0 | from_meta |

## 검증 결과 (실제 실행)
- 4개 신규 YAML 파싱 OK, loss/aug 필드 의도대로 확인(아래 보고). 기존 2개 사양(`exp_dinov3_base_normal_d3_d4.yaml`, `exp_dinov3_base_focal_normal_d3_d4.yaml`) **미변경**(내용 동일).

---

# "Ours" detection — DINOv3 ViT-B/16 (frozen) @512 + single-box(+objectness) head

분류 "Ours/Ours+"의 frozen-backbone 전이 레시피를 **detection으로 확장**한다. from-scratch detection baseline(convnextv2/nextvit20/… 풀링-특징 single-box 회귀, §61)은 소표본·단일시즌에서 표현 상한이 낮다. 여기서는 **자기지도(DINOv3) ImageNet 사전학습 ViT-B/16(768-d)을 frozen**으로 쓰고, 기존 `SingleBoxDetector`(풀링 특징 → box[4]∈[0,1] xyxy + objectness) 헤드만 학습한다. backbone은 `DINOv3ForImageClassification(frozen_backbone=True)`를 그대로 재사용해 `forward_features(x)->(B,768)` 풀링 계약을 만족시키고(requires_grad=False + no_grad forward + eval), 미사용 분류 헤드는 Identity로 치환해 trainable에서 제외한다.

**가설**: ImageNet 자기지도 특징은 이미 작물/배경을 강하게 분리하므로 (a) **이미지단위 질병 검출**(objectness vs 정상/질병, 주지표 det_pr_auc)에서 from-scratch baseline보다 절대 상한이 높고, (b) bbox가 병변 핀포인트가 아닌 **거친 작물 영역**(REPORT §6)이라 768-d 전역 특징만으로도 **거친 박스 국소화**(positive-only IoU·mAP@0.5)에 충분한 신호를 준다. **frozen이므로 사전학습 지식의 catastrophic forgetting이 없고**, 헤드(~0.20M, 전체의 0.23%)만 학습해 소표본 과적합 위험도 낮다.

**한계**: 풀링 전역 특징은 위치 정보를 직접 담지 않아 정밀 국소화에는 본질적 한계가 있고(미세 병변 핀포인팅 부적합), DINO 특징이 "질병"이 아닌 일반 객체성에 정렬돼 있어 detection이 작물 영역 검출로 수렴할 수 있다(분류+CAM 대안 병기 권장). frozen이라 도메인 적응 여지도 제한적(필요 시 마지막 1~2 블록 저-LR unfreeze ablation).

## 사양 (`exp_dinov3_base_detection_singlebox.yaml`)
- model {builder src.models.build_detector, arch dinov3_base, img_size 512, with_objectness true, frozen_backbone true, pretrained true}, data {build_detection_loaders, img_size 512, batch 16, include_normal true}, optim AdamW lr1e-3 wd0.05 ep40 cosine warmup3 **amp false**(ViT 안정), loss giou(2.0)+l1(5.0)+obj(1.0) box_loss_on positive_only, primary **det_pr_auc**, seed42, smoke true. 기존 detection 사양(`exp_*_detection_singlebox.yaml`)과 동일 metric·loss·train.py 경로.

## 검증 결과 (실제 cuda 스모크)
- `build_detector("dinov3_base", img_size=512)` @512 랜덤 forward → boxes (2,4)∈[0,1], objectness (2,). DINOv3 ViT-B/16 pretrained 로드 성공.
- frozen 확인: backbone `requires_grad=False`(전부), trainable==head, **total 85.84M / trainable 0.1997M (0.23%)**. 기존 arch(convnextv2 등 7종) 모두 여전히 빌드·forward OK.

---

# Stability (train ratio sweep, Ours 3-class)

Ours 헤드라인(`dinov3_base_focal_normal_d3_d4` = DINOv3 ViT-B/16 frozen + strong aug + focal γ2, 3-class normal/d3/d4)의 **데이터량 대비 성능 안정성**을 본다. 헤드라인 사양을 **그대로 클론**하고 `data.train_ratio`만 바꿔, train 표본 크기를 줄여가며 PR-AUC·F1-macro의 평균 수준과 분산을 관찰한다.

## 설계
- **스윕 지점**: `data.train_ratio ∈ {0.1, 0.3, 0.5, 0.7, 0.9}` (신규 5개 사양) + **1.0**(기존 `dinov3_base_focal_normal_d3_d4` run이 참조점, 추가 사양 없음). 총 6점.
- **축소 방식**: 로더가 지원하는 **클래스 stratified 축소** — train_ratio r에서 각 클래스 r×(227/227/227) 표본을 균등 비율로 뽑는다(검증됨). 즉 클래스 균형(다운샘플된 1:1:1 구조)을 유지한 채 절대 표본 수만 r배로 줄인다. 클래스 불균형 변화가 아니라 **순수 데이터량 효과**만 분리.
- **고정**: valid는 **동일 split으로 불변**(모든 점이 같은 검증셋으로 평가) → 직접 비교 가능. model/optim/loss(focal γ2 from_meta)/aug(strong)/eval/early_stop/seed 42 등 **train_ratio 외 전부 동일**.
- **불변 보장**: 5개 신규 사양은 헤드라인과 **arch dinov3_base·num_classes 3·img_size 512·source timm·pretrained·frozen_backbone·head_hidden 512·setting normal_d3_d4·batch 64·num_workers 8·aug strong·AdamW lr1e-3 wd0.05 cosine warmup3 ep40·focal γ2 from_meta·label_smoothing0·primary pr_auc·7-메트릭·early_stop(val_pr_auc/max/patience10)·class_names·seed 42·packages·smoke true** 완전 동일. 셀별로 바꾼 필드는 `name`과 `data.train_ratio` 둘뿐.

| 사양 | name | data.train_ratio |
|---|---|---|
| `exp_dinov3_base_focal_r10_normal_d3_d4.yaml` | dinov3_base_focal_r10_normal_d3_d4 | 0.1 |
| `exp_dinov3_base_focal_r30_normal_d3_d4.yaml` | dinov3_base_focal_r30_normal_d3_d4 | 0.3 |
| `exp_dinov3_base_focal_r50_normal_d3_d4.yaml` | dinov3_base_focal_r50_normal_d3_d4 | 0.5 |
| `exp_dinov3_base_focal_r70_normal_d3_d4.yaml` | dinov3_base_focal_r70_normal_d3_d4 | 0.7 |
| `exp_dinov3_base_focal_r90_normal_d3_d4.yaml` | dinov3_base_focal_r90_normal_d3_d4 | 0.9 |
| (기존) `exp_dinov3_base_focal_normal_d3_d4.yaml` | dinov3_base_focal_normal_d3_d4 | 1.0 (참조점) |

## 비교 지표
- **주지표 PR-AUC**(macro one-vs-rest, 헤드라인 primary와 동일)와 **F1-macro**를 train_ratio에 대해 그린다. seed 단일이므로 분산은 (a) ratio 축 곡선의 거칠기/비단조성, (b) per-class F1의 흔들림으로 정성 평가(다중 seed는 후속 ablation 여지).

## 예상
- **데이터 적을수록 성능↓**: train_ratio가 작아질수록 PR-AUC·F1-macro 평균이 하락. 단 backbone이 **frozen 사전학습**이라 학습 대상이 ~0.40M 헤드뿐이므로, from-scratch 백본보다 저데이터 강건성이 높아 **r=0.5 부근까지는 완만한 하락**, r≤0.3에서 가팔라질 것으로 예상.
- **분산↑(저데이터)**: 표본이 작을수록(특히 d4 train 표본 r×227) 결정경계가 불안정해져 점간 변동·per-class F1 흔들림이 커질 것. d4 precision 병목(valid N=24)이 저-ratio에서 더 두드러질 가능성.
- **포화**: r=0.7~1.0 구간은 헤드 probe가 이미 충분한 특징을 받아 **PR-AUC 포화(plateau)** 예상 — 이 경우 "Ours는 헤드라인 데이터량의 일부만으로도 동급 성능"이라는 데이터 효율 결론으로 귀결.

## 검증 결과 (실제 실행)
- 5개 신규 YAML 파싱 OK, `data.train_ratio` = {0.1,0.3,0.5,0.7,0.9} 의도대로 확인(아래 보고). 헤드라인 사양(`exp_dinov3_base_focal_normal_d3_d4.yaml`) 대비 diff는 `name`·`data.train_ratio` 둘뿐(나머지 100% 동일). 기존 사양 **미변경**.
