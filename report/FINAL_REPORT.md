# 무(Radish) 작물 질병 진단 — 멀티에이전트 실험 종합 보고서

작성: eval-reporter(QA). 본 보고서는 from-scratch baseline(백본 6종 × 분류 3세팅 + detection)과 제안 모델(Ours: DINOv3 frozen + 경량 head)의 분류·검출 성능을 동일 split·동일 지표·동일 seed(42)에서 비교·평가하고, 강한 증강 × focal loss의 기여를 ablation으로 분리한 결과를 종합한다. 모든 수치는 1차 산출물(`report/EXPERIMENTS.md`, `report/REPORT.md`, `report/stats.json`, `_workspace/eval/*.json`, `_workspace/specs/design_notes.md`)에서 인용했으며, 표·그림 캡션에 출처 섹션을 병기한다. 소표본(특히 valid disease_4 = 24장) 구간에서는 신뢰구간(CI)을 함께 해석하며 과대해석을 피한다.

---

## 1. Introduction

### 1.1 연구 배경과 중요성

무(radish)는 국내 주요 노지 작물로, 잎·뿌리 질병의 조기 진단은 수확 손실 저감과 방제 시기 결정에 직결된다. 본 연구는 AI-Hub 무 질병 이미지 데이터셋(정상/질병 무 작물 이미지 + 라벨 JSON)을 사용해 (1) 질병 종류별 분류와 (2) 질병 케이스의 단일 박스 object detection을 다룬다.

이 데이터의 가장 본질적 난점은 **극심한 클래스 불균형**이다(출처: `REPORT.md` §1, `stats.json`).

| split | normal | disease | 합계 | 불균형(정상:질병) |
|-------|-------:|--------:|-----:|:---------------:|
| train | 11,001 | 697 | 11,698 | 15.8 : 1 |
| valid | 1,303 | 100 | 1,403 | 13.0 : 1 |
| **전체** | **12,304** | **797** | **13,101** | — |

질병은 코드 3·4 두 종류뿐이며, 내부에서도 불균형하다: **disease_3 = 546장(train 470 / valid 76), disease_4 = 251장(train 227 / valid 24)** 로 d3가 d4의 약 2.2배다(출처: `REPORT.md` §2, `stats.json`). 특히 **valid disease_4는 24장뿐**이라 해당 클래스의 검증 지표는 통계적으로 매우 불안정하다.

데이터 특성상 추가로 유의할 점(출처: `REPORT.md` §4~§7):
- **거친 단일 bbox**: 이미지당 박스는 정확히 1개(`boxes_per_image_max=1`, multi_box=0)이며, 병변 핀포인트가 아니라 잎/작물 영역을 통째로 감싸는 거친 박스다(질병 박스 상대 면적 중앙값 ≈0.50, 정상도 ≈0.45로 박스 크기만으로 정상/질병 구분 불가). 박스 중심은 이미지 중앙(≈0.5, 0.5)에 집중.
- **단일 시즌**: 촬영 기간 2020-10-06 ~ 2021-01-14의 가을~겨울 단일 시즌, region은 전부 null → 도메인 다양성이 낮아 외부 일반화 검증에 한계.
- **데이터 품질**: 질병 라벨 43건의 JSON `width/height`가 0×0으로 기록됨 → 이미지 크기는 실제 파일에서 읽어 정규화(파이프라인에서 처리).

> **함의**: 단순 accuracy는 무의미하고(전부 정상으로 찍어도 ≈93~94%), 평가 주지표는 **PR-AUC·F1-macro·질병 recall**이어야 하며, 학습에는 다운샘플/증강/focal 등 불균형 대응이 필요하다.

### 1.2 우리의 기여

1. **재현 가능한 멀티에이전트 실험 하네스 + 불균형 인지 baseline**: from-scratch 백본 6종(ConvNeXtV2-tiny / EfficientNetV2-s / NeXtViT-small / DenseNet121 / ResNet50 / Vision-Mamba) × 분류 3세팅(normal_vs_d3 / normal_vs_d4 / normal_d3_d4)과 detection을, **원분포(불균형)·균형(1:1, 1:1:1) 이중 평가**로 측정. 모든 run은 seed=42, 동일 split·지표로 고정.
2. **Ours = DINOv3(frozen) + 경량 2-layer head 전이**: 자기지도 사전학습 ViT를 완전 동결(no forgetting)하고 소수 파라미터(head)만 학습. 가장 어려운 3-class 원분포 PR-AUC에서 baseline 최고 대비 **+34%**(0.570 → 0.765).
3. **detection objectness collapse 발견·수정**: 초기 detection은 정상을 음성으로 다루지 않아 objectness가 전 이미지에서 ~1.0으로 붕괴 → 데이터 로더·지표 처리를 수정해 정상=음성으로 재학습, 분리도를 정량 확인.
4. **강한 증강 × focal ablation**: focal+aug 개선을 2×2 단변수 ablation으로 분해해 기여를 정직하게 분리(강한 증강이 주 동력, focal은 증강과 결합 시에만 이득).
5. **Ours 견고성·데이터 효율 분석**: 입력 노이즈 sensitivity(§3.7)와 train-ratio stability(§3.8)로 frozen 전이 모델의 강건성·데이터 스케일링을 정량화.
6. **다중 파이프라인 동시 비교 + 전체 재현 절차**: baseline ↔ Ours를 한 표/그림에서 비교하고, 정합성 교차검증·재현 스크립트(+ 단일 노트북·FastAPI 데모/VQA)를 산출물로 제공.

---

## 2. Method

### 2.1 문제 정의

- **입력**: RGB 무 작물 이미지(해상도·방향 혼재 → 리사이즈·정규화·EXIF 보정). **출력**:
  - 분류 3세팅 — `normal_vs_d3`(2-class), `normal_vs_d4`(2-class), `normal_d3_d4`(3-class).
  - detection — 질병 단일 bbox(xyxy ∈ [0,1]) + 이미지단위 objectness(질병 유무).
- **Split·표본·불균형**: valid는 **원분포(다운샘플 없음)** 로 normal 1303 + d3 76 + d4 24 = 1403(detection은 질병 100 = d3 76 + d4 24, 정상 1303 음성). 불균형이 심해 accuracy·AUROC는 단독 해석 금지.
- **클래스 균형 처리**: train은 type 간 표본 수를 **다운샘플로 균형**(2-class d4 = 227×2, 3-class = type당 227). valid는 두 방식으로 평가 — 원분포, 그리고 **균형 다운샘플**(2-class 1:1, 3-class 1:1:1; seed=42 고정). 균형 valid 표본: normal_vs_d3 = 76/76(N=152), normal_vs_d4 = 24/24(N=48), normal_d3_d4 = 24/24/24(N=72).

### 2.2 평가 프로토콜·메트릭 정의

(출처: `EXPERIMENTS.md` "메트릭 정의")

**분류 7-메트릭**: accuracy, train_loss, val_loss(best epoch), recall(macro), precision(macro), **f1-macro**, AUROC + **PR-AUC(주지표)**.
- 불균형 원분포에서 **accuracy·AUROC는 포화(AUROC 0.94~1.0)** 되어 백본 변별에 둔감하므로 단독 해석 금지. 우열은 **PR-AUC·F1-macro·precision**으로 본다. PR-AUC 랜덤 하한(prevalence) = 양성 비율(원분포 3-class disease ≈0.071).
- AUROC: 2-class = P(disease) ROC-AUC / 3-class = macro OvR. PR-AUC: 2-class / 3-class = disease(d3,d4) OvR macro.

**Detection 메트릭**: det PR-AUC(주지표; objectness로 질병/정상 **이미지** 구분), det ROC-AUC, presence_recall@0.5(질병 검출 민감도), fp_rate@0.5(정상 오경보율), IoU(median; 양성 이미지에서만 국소화), mAP@0.5(검출+국소화 결합). "있는지(검출)"와 "어디인지(국소화)"를 분리 해석.

### 2.3 실험 세팅 (옵션·하이퍼파라미터)

- **환경**: Python 3.10, torch 2.11+cu128, `uv`/`.venv` 환경, 단일 GPU(cuda).
- **입력 해상도**: baseline 분류 224 / detection 512; Ours(small) 256; Ours+(base)·focal+aug·Ours-detection 512.
- **전처리·증강**: default = RandomResizedCrop(0.6–1.0) + HFlip + ColorJitter + ImageNet normalize. **strong** = 추가로 VFlip + rotation + 강한 ColorJitter + TrivialAugmentWide + RandomErasing.
- **Optimizer**: 전 run AdamW, wd=0.05, cosine sched + warmup, seed=42, label_smoothing=0.0(공통). baseline lr=3e-4·epochs=60·warmup=5; Ours lr=1e-3·warmup=3, epochs small=30 / base=40. early-stop은 val primary(PR-AUC / det_pr_auc) 기준.
- **Loss**: 분류 CE 또는 focal(γ). **AMP**: 기본 사용하되, NeXtViT·VisionMamba detection은 fp16에서 NaN 발산 → fp32(AMP off).

**부록 A 하이퍼파라미터 요약(출처: `EXPERIMENTS.md` 부록 A)**

| 그룹 | arch | img | batch | lr | epochs | warmup | aug | loss | frozen | trainable/total |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| baseline 분류(6백본×3세팅, 18 run) | 6종 | 224 | 64 | 3e-4 | 60 | 5 | default | CE | ✗(전체 학습) | =total |
| baseline detection(6백본, 6 run) | 6종 single-box+obj | 512 | 8 | 3e-4 | 60 | 5 | default | giou(2)+l1(5)+obj(1), box_loss=positive_only | ✗ | =total |
| Ours(small, 3세팅) | dinov3 ViT-S/16 | 256 | 64 | 1e-3 | 30 | 3 | default | CE | ✓ | ~99k / 21.69M |
| Ours+(base, 3세팅) | dinov3_base ViT-B/16 | 512 | 64 | 1e-3 | 40 | 3 | default | CE | ✓ | 395k / 86.04M |
| Ours+ focal+aug(3세팅) | dinov3_base ViT-B/16 | 512 | 64 | 1e-3 | 40 | 3 | **strong** | **focal γ2** | ✓ | 395k / 86.04M |
| Ours detection | dinov3_base ViT-B/16 | 512 | 16 | 1e-3 | 40 | 3 | — | giou(2)+l1(5)+obj(1) | ✓ | 0.200M / 85.84M |

> 전 run 공통: AdamW, wd=0.05, cosine, seed=42, ls=0.0, class_weights=from_meta(균형 train 로더에서 전부 1.0으로 해소 → focal은 alpha 재가중이 아니라 γ(hard-example 집중)로 작동).

### 2.4 Baseline 구성

(출처: `design_notes.md`, `EXPERIMENTS.md` §A-1)

- 백본 6종을 **as-is·from-scratch(pretrained 미로딩)** 로 분류기 래핑: 풀링 특징 → 공통 표준 헤드(`LayerNorm → Linear(feat,512) → GELU → Dropout(0.1) → Linear(num_classes)`). 백본 파라미터(M): ConvNeXtV2 28.26 / EfficientNetV2 20.84 / NeXtViT-s 31.27 / DenseNet121 7.48 / ResNet50 24.56 / VisionMamba 3.88.
- **detection**: 백본의 `forward_features`가 공간 맵이 아닌 풀링된 (B,C) 벡터를 반환하므로 FPN/anchor 대신 풀링 특징 → 단일 박스 회귀(+objectness)가 라벨 구조(이미지당 박스 1개)와 정확히 일치. loss = GIoU(2) + L1(5) + objectness BCE(1), box loss는 양성(질병)에만 적용.
- **각주**: MambaVision은 `mamba_ssm` CUDA 빌드 불가 → mambapy(pscan) Vision-Mamba로 대체. NeXtViT-base(`nextvit20`)는 사용자 요청으로 리포트 표·그림에서 제외(산출물은 보존). 따라서 카운트는 **6 백본 / 24 baseline run** 기준.

### 2.5 Ours 기법 (구조·원리)

(출처: `design_notes.md`, `EXPERIMENTS.md` §6)

핵심 아이디어: 대규모 자기지도(DINOv3) ImageNet 사전학습 ViT를 **완전 동결**(`requires_grad=False`, no-grad/eval forward)하고, 그 위 **2-layer MLP head만 학습**한다. 동결이 사전학습 표현을 그대로 보존(catastrophic forgetting 없음)하고, 소표본에 노출되는 학습 파라미터를 head로 제한해 **과적합을 구조적으로 억제**한다(transfer/linear-probe 정석).

- **Ours (small)**: `vit_small_patch16_dinov3` @256, pooled 384-d, head `Linear(384,256)→GELU→Dropout→Linear`. total **21.69M / trainable(head) ~99k(0.46%)**.
- **Ours+ (base)**: `vit_base_patch16_dinov3` @512, pooled 768-d, head hidden 512. total **86.04M / trainable ~395k(0.46%)**. 해상도(256→512, 토큰 16²→32²)·백본(S→B)을 키워 미세 병변·d3↔d4 구분 표현력을 확장.
- **Ours+ focal+aug**: Ours+(base)와 backbone·head·해상도·trainable 동일, 학습만 strong aug + focal(γ2)로 변경.
- **Ours detection**: 동일 DINOv3-B frozen backbone + 기존 단일박스+objectness head만 학습. total **85.84M / trainable 0.200M(0.23%)**.
- **성능 최대화 경로**: S → B → @512 → strong aug + focal.

관련 그림: `![](figures/exp_ours_dinov3.png)`(small/base/baseline 3-way, +20% 목표선), `![](figures/exp_ours_focal.png)`(baseline/small/base-CE/focal+aug 4-way) — 출처 `EXPERIMENTS.md` §6.

### 2.6 Ours 알고리즘 (수도코드)

핵심은 **동결된 DINOv3 백본 + 경량 head만 학습**이다. 학습 시 backbone은 `requires_grad=False`·eval 모드라 gradient가 흐르지 않아 사전학습 표현이 보존되고(no forgetting), optimizer에는 head 파라미터만 전달된다.

```
# ---------- 구성 ----------
Backbone B  ← DINOv3 ViT (S/16@256 또는 B/16@512), self-supervised pretrained
            freeze(B): 모든 파라미터 requires_grad = False, B.eval()  # 통계·가중치 고정
Head H      ← Linear(feat_dim → hidden) → GELU → Dropout(0.1) → Linear(hidden → C)
              # 분류: C = num_classes(2 또는 3)
optimizer   ← AdamW( params = [p for p in H.parameters()],  lr=1e-3, wd=0.05 )  # head만
loss_fn     ← FocalLoss(γ=2, α=class_weights)   # 또는 CrossEntropy
augment     ← strong( RandomResizedCrop, H/V-flip, rotation, ColorJitter,
                       TrivialAugmentWide, RandomErasing )   # train만, focal+aug 변형

# ---------- 학습 (head-probe) ----------
for epoch in 1..E:                          # E=30(S)/40(B), cosine+warmup
  for (img, y) in train_loader:             # train은 클래스 균형 다운샘플
    x      = augment(img); x = normalize(x)
    with no_grad():   feat = B.forward_features(x)   # (Batch, feat_dim) pooled, 동결
    logits = H(feat)
    loss   = loss_fn(logits, y)
    loss.backward();  optimizer.step();  optimizer.zero_grad()   # H만 갱신
  evaluate(valid);  keep best by PR-AUC(분류)/det_pr_auc(검출)   # early-stop

# ---------- 추론 ----------
feat = B.forward_features(normalize(img));  logits = H(feat)
pred = softmax(logits)                       # 분류: 클래스 확률

# ---------- 검출 변형(동일 동결 백본) ----------
Head_det ← Linear(feat_dim → 4) (box xyxy∈[0,1]) ⊕ Linear(feat_dim → 1) (objectness)
loss_det ← GIoU(2)+L1(5) (양성 박스만) + BCE(1) (objectness, 정상=음성)
# 추론: pred_box, p_obj = Head_det(B.forward_features(x)); is_disease = p_obj > 0.5
```

> 동결+head-only이므로 학습 파라미터는 분류 head ~99k(S)/395k(B), 검출 head ~0.2M으로 전체의 0.2~0.5%에 불과하다. 이 구조가 소표본·단일시즌 데이터에서 from-scratch 대비 큰 이득(3-class PR-AUC +34%)과 입력 노이즈 강건성(§3.7), 데이터량에 대한 안정적 스케일링(§3.8)의 근거다.

---

## 3. Experiments

### 3.1 수행한 실험

- **baseline 24 run** = 분류 18(6백본 × 3세팅) + detection 6.
- **Ours** = 분류 9(small/base-CE/focal+aug × 3세팅) + detection 1.
- **Ablation 4 run**(augonly / focalonly / focalg1 / focalg3) + 참조 2개(base, focal+aug; Ours에 포함).
- 전부 seed=42, 동일 valid split, 원분포 + 균형 이중 평가.

**정합성 교차검증(§0 요약, 출처 `EXPERIMENTS.md` §0)**: baseline 24 run + Ours 4 run + ablation 6 run 전부 predictions의 valid 표본 수·라벨 분포가 manifest와 **정확히 일치(dist_match=PASS)**, 누수·오집계·split 위반 없음. 재계산(PR-AUC/F1/recall/precision/accuracy/AUROC/confusion)이 보고치와 허용오차 내 일치, 기존 3 백본의 누락 지표는 predictions에서 sklearn로 독립 재계산·교차검증. detection objectness collapse는 수정·재학습으로 해소(질병 objectness median ≈0.993–0.999 vs 정상 ≈0.000–0.004).

### 3.2 분류 결과 — 원분포 valid (7-메트릭, 주지표 PR-AUC)

(출처: `EXPERIMENTS.md` §1-1/1-2/1-3. baseline 6종은 from-scratch, **Ours 3행은 DINOv3 frozen pretrained → 동일 조건 비교 아님**. 굵게 = 신규 백본 또는 Ours.)

**normal_vs_d3 (normal 1303 / d3 76)**

| 백본 | acc | f1(M) | AUROC | **PR-AUC** |
|------|:---:|:---:|:---:|:---:|
| ConvNeXtV2 | 0.541 | 0.436 | 0.963 | 0.787 |
| EfficientNetV2 | 0.937 | 0.799 | 0.995 | 0.949 |
| NeXtViT-s | 0.938 | 0.801 | 0.995 | 0.929 |
| **DenseNet121** | 0.962 | 0.860 | 0.997 | 0.957 |
| **ResNet50** | 0.957 | 0.849 | 0.998 | **0.965** |
| **VisionMamba** | 0.898 | 0.729 | 0.989 | 0.878 |
| **Ours: DINOv3-S@256(CE)** | 0.982 | 0.924 | 1.000 | 0.996 |
| **Ours+: DINOv3-B@512(CE)** | 0.996 | 0.983 | 1.000 | 1.000 |
| **Ours+: focal+aug** | 0.998 | **0.990** | 1.000 | **1.000** |

**normal_vs_d4 (normal 1303 / d4 24 — 소표본, CI 넓음)**

| 백본 | f1(M) | **PR-AUC** |
|------|:---:|:---:|
| ConvNeXtV2 | 0.587 | 0.759 |
| EfficientNetV2 | 0.679 | 0.862 |
| NeXtViT-s | 0.631 | 0.822 |
| **DenseNet121** | 0.665 | 0.857 |
| **ResNet50** | 0.668 | **0.867** |
| **VisionMamba** | 0.549 | 0.629 |
| **Ours+: focal+aug** | **0.919** | **1.000** |

**normal_d3_d4 (3-class; normal 1303 / d3 76 / d4 24) — 가장 어려운 세팅**

| 백본 | acc | recall(M) | f1(M) | AUROC | **PR-AUC** |
|------|:---:|:---:|:---:|:---:|:---:|
| ConvNeXtV2 | 0.874 | 0.640 | 0.523 | 0.943 | 0.343 |
| EfficientNetV2 | 0.920 | 0.676 | 0.591 | 0.977 | 0.509 |
| NeXtViT-s | 0.952 | 0.617 | 0.624 | 0.969 | 0.487 |
| **DenseNet121** | 0.940 | 0.862 | **0.709** | 0.984 | **0.570** |
| **ResNet50** | 0.949 | 0.762 | 0.685 | 0.983 | 0.530 |
| **VisionMamba** | 0.875 | 0.634 | 0.524 | 0.947 | 0.359 |
| **Ours: DINOv3-S@256(CE)** | 0.961 | 0.753 | 0.698 | 0.991 | 0.689 |
| **Ours+: DINOv3-B@512(CE)** | 0.967 | 0.840 | 0.750 | 0.993 | 0.745 |
| **Ours+: focal+aug** | 0.973 | 0.850 | **0.774** | 0.995 | **0.765** |

baseline 종합 우열(PR-AUC 주지표): **DenseNet121 ≈ ResNet50 ≳ EfficientNetV2 > NeXtViT-s > VisionMamba > ConvNeXtV2**. AUROC는 6×3 전부 0.94–1.0으로 포화돼 백본을 거의 구분하지 못한다(출처 §2).

### 3.3 분류 결과 — 균형 valid (착시 제거)

(출처: `EXPERIMENTS.md` §1B-3) 균형 3-class(N=72)에서 변화가 가장 극적이다: accuracy가 일제히 급락(EfficientNetV2 0.920→0.653, NeXtViT 0.952→0.639)하고 AUROC 포화가 풀려(0.94–0.98 → 0.82–0.95) 분리력이 생긴다. 즉 원분포의 높은 acc/AUROC는 상당 부분 정상을 잘 맞힌 부풀림이었다. 균형 3-class에서 **DenseNet121이 baseline 1위(acc 0.847·F1 0.850·AUROC 0.954·PR-AUC 0.860)**, **Ours+(B,CE)는 PR 0.885·F1 0.875로 baseline 최고를 상회**(focal+aug PR 0.856·F1 0.875). 2-class 균형은 양쪽 모두 천장 근접.

관련 그림: `![](figures/exp_metrics_table.png)`(원분포 AUROC/F1/acc 막대), `![](figures/exp_metrics_balanced.png)`(균형), `![](figures/exp_cls_bars.png)`, `![](figures/exp_pr_curves.png)`, `![](figures/exp_confusion.png)` — 출처 `EXPERIMENTS.md` §1/§1B.

### 3.4 Detection 결과

(출처: `EXPERIMENTS.md` §3-1/§3-2/§3B, `ours_detection.json`)

**이미지 단위 질병 검출(주지표, 원분포: 양성 100 / 음성 1303)**

| 백본 | det PR-AUC | presence_recall@0.5 | fp_rate@0.5 |
|------|:---:|:---:|:---:|
| ConvNeXtV2 | 0.9977 | 0.990 (99/100) | 0.0031 (4/1303) |
| EfficientNetV2 | 0.9990 | 1.000 | 0.0054 (7/1303) |
| NeXtViT-s | 0.9990 | 1.000 | 0.0061 (8/1303) |
| **DenseNet121** | **0.9997** | 1.000 | 0.0031 (4/1303) |
| **ResNet50** | 0.9985 | 0.980 (98/100) | **0.0015** (2/1303) |
| **VisionMamba** | 0.9975 | 0.990 (99/100) | 0.0046 (6/1303) |
| **Ours: DINOv3-B@512(frozen, head-only)** | **1.0000** | 1.000 (100/100) | 0.0008 (1/1303) |

**국소화(양성 100장 IoU)**: IoU median 0.57–0.67(ConvNeXtV2 0.667 최고), mAP@0.5 0.46–0.60. Ours IoU median 0.565†. † Ours best epoch = ep0(det_pr_auc가 ep0부터 1.0 포화 → early-stop이 ep0 선택)이라 저장 predictions의 국소화 IoU가 전 epoch 중 최저(0.565); 후반 ep≥28에서 IoU median ≈0.64(최고 0.643)까지 상승. 검출 지표는 ep0부터 포화라 epoch 선택과 무관(출처 `ours_detection.json` `best_epoch_iou_caveat`).

**해석**: 질병 "유무" 검출은 6 백본 + Ours 전부 사실상 포화(det PR-AUC 0.997–1.000), objectness 분리 깨끗(질병 median ≈0.955–0.999 vs 정상 ≈0.017). 그러나 mAP는 거친 박스의 국소화 hit-rate에 상한이 묶여 0.46–0.60 — "있는지"는 거의 완벽, "어디인지"는 거친 crop 영역 회귀 수준. 균형 valid(§3B)에서도 검출은 전부 포화(det PR-AUC ≥0.9998).

관련 그림: `![](figures/exp_detection.png)`, `![](figures/exp_detection_balanced.png)`, `![](figures/training_curves.png)` — 출처 `EXPERIMENTS.md` §3/§4.

### 3.5 +20% 목표 판정 (정직)

(출처: `EXPERIMENTS.md` §6.3, `design_notes.md`, `ours_dinov3.json` `goal_verdict_3class_focal`) 목표 기준 = 3-class 원분포 baseline 최고(PR-AUC 0.570 DenseNet121, F1 0.709) 대비 +20% → **PR-AUC ≥ 0.684, F1-macro ≥ 0.851**.

| 지표(3-class 원분포) | baseline 최고 | Ours(S) | Ours+(B,CE) | **focal+aug** | focal 상대% | 목표 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **PR-AUC** | 0.570 | 0.689 | 0.745 | **0.765** | **+34.1%** | ≥0.684 → **달성** |
| **F1-macro** | 0.709 | 0.698 | 0.750 | **0.774** | **+9.2%** | ≥0.851 → **미달** |

- **PR-AUC: 세 변형 모두 목표선(0.684)을 상회**, focal+aug가 +34.1%(0.765)로 가장 견고.
- **F1-macro: 단조 개선(0.698→0.750→0.774)이나 절대 목표 0.851 미달** — d4 소표본 precision 병목(아래 §4) 때문. 과대포장하지 않는다: "+20% 달성"은 **3-class·원분포·PR-AUC** 에서 성립.

### 3.6 Ablation — aug × focal 기여 분리

(출처: `EXPERIMENTS.md` §6B, `ablation_dinov3.json`. backbone·head·해상도·optimizer·trainable(395k)·seed 전부 고정, loss/aug만 변수, 3-class 원분포 N=1403.)

**2×2 + 요인 분해**

| run | loss / aug | PR-AUC | F1-macro | d4 precision |
|---|---|:---:|:---:|:---:|
| base(참조) | CE / default | 0.745 | 0.750 | 0.339 |
| augonly | CE / strong | 0.756 | 0.768 | 0.389 |
| focalonly | focal γ2 / default | 0.736 | 0.735 | 0.299 |
| focal+aug(참조) | focal γ2 / strong | **0.765** | **0.774** | **0.396** |

| 지표 | aug 효과 | focal 효과 | 상호작용 | joint |
|---|:---:|:---:|:---:|:---:|
| PR-AUC | **+0.0105** | −0.0088 | **+0.0178** | +0.0195 |
| F1-macro | **+0.0177** | −0.0153 | **+0.0216** | +0.0240 |
| d4 precision | **+0.050** | −0.040 | +0.048 | +0.057 |

**결론(정직)**: (1) **강한 증강이 주 동력**(aug-only 단독으로 PR +0.0105·F1 +0.0177·d4P +0.050). (2) **focal 단독은 소폭 하락**(focalonly가 base보다 PR −0.0088·F1 −0.0153). (3) **조합 시너지**(상호작용 PR +0.0178·F1 +0.0216이 두 주효과 합보다 큼) → focal은 strong aug와 결합 시에만 이득. 단독 focal은 쓰지 말고 strong aug와 결합해야 한다는 실용 결론.

**gamma 스윕(strong aug 고정)**: PR-AUC·AUROC는 γ↑ 단조 개선(γ3 PR 0.769 최고), F1·d4 precision은 γ↓ 유리(γ1 F1 0.782·d4P 0.412 최고) — 두 곡선이 교차하며 γ2는 절충점. 단 γ들 간 차이(PR 0.011 폭, F1 0.015 폭)는 d4 N=24 단일 시드 CI 안이라 통계적 단정 불가, 추세로만 해석.

관련 그림: `![](figures/exp_ablation_dinov3.png)`(2×2 막대 + 기여분해 + gamma 곡선) — 출처 `EXPERIMENTS.md` §6B.

### 3.7 Sensitivity — 입력 노이즈 강건성 (Ours, 3-class)

(출처: `EXPERIMENTS.md` §7, `sensitivity_dinov3.json`) 학습된 Ours(focal+aug, 3-class) `best.pt`를 **재학습 없이** 로드해, 정규화된 모델 입력 텐서에 노이즈 `x ← x + torch.rand_like(x)·N_ratio`(rand ~ U[0,1), seed 고정)를 가하고 원분포 valid(N=1403)에서 재평가했다.

| N_ratio | PR-AUC | F1-macro | accuracy | AUROC | ΔPR-AUC |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.0 (clean) | 0.765 | 0.772 | 0.972 | 0.995 | — |
| 0.1 | 0.762 | 0.781 | 0.973 | 0.995 | −0.4% |
| 0.2 | 0.748 | 0.773 | 0.969 | 0.994 | −2.3% |
| 0.3 | 0.747 | 0.754 | 0.964 | 0.994 | −2.4% |
| 0.4 | 0.755 | 0.752 | 0.964 | 0.994 | −1.3% |
| 0.5 | 0.752 | 0.752 | 0.963 | 0.994 | −1.7% |

- **매우 강건**: N_ratio 0.5까지 PR-AUC 저하 최대 −2.4%, AUROC는 0.995→0.994로 거의 불변. clean(0.0)은 §3.2 Ours focal+aug(PR-AUC 0.765)와 일치(검증됨).
- 강건성의 출처는 **frozen DINOv3 백본 + head-only** 구조 — 입력 텐서 가산 노이즈가 깊은 사전학습 표현을 크게 교란하지 못한다. 저하가 단조가 아닌 것(0.3 저점 후 0.4 회복)은 d4 N=24 소표본 변동 스케일이라 N_ratio 간 미세 우열은 단정 불가. (정규화 입력 텐서에 가산한 노이즈로, 실제 픽셀/촬영 노이즈와 분포가 다름은 명시한다.)

관련 그림: `![](figures/exp_sensitivity_dinov3.png)`.

### 3.8 Stability — train ratio sweep (Ours, 3-class)

(출처: `EXPERIMENTS.md` §8, `stability_dinov3.json`) train 데이터 비율 train_ratio ∈ {0.1,…,0.9}로 Ours(focal+aug, 3-class)를 재학습(클래스 stratified 축소, 균형 유지·seed 고정)하고 동일 원분포 valid로 평가했다. train_ratio=1.0은 §3.2의 기존 run이 참조점.

| train_ratio | train 표본/클래스 | PR-AUC | F1-macro | accuracy | AUROC |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.1 | 23 | 0.489 | 0.316 | 0.701 | 0.956 |
| 0.3 | 68 | 0.640 | 0.661 | 0.947 | 0.989 |
| 0.5 | 114 | 0.680 | 0.698 | 0.955 | 0.992 |
| 0.7 | 159 | 0.748 | 0.559 | 0.939 | 0.995 |
| 0.9 | 204 | 0.751 | 0.734 | 0.963 | 0.995 |
| 1.0(참조) | 227 | **0.765** | **0.774** | 0.973 | 0.995 |

- **PR-AUC는 데이터량에 거의 단조 증가하고 90%↑에서 포화**(0.489→0.765, r0.7~1.0 +0.017로 수확체감) — 안정적 스케일링.
- **F1-macro는 비단조**(r0.7에서 0.559 급락): argmax(0.5) 의존 + d4 N=24 소표본 민감 + PR-AUC 기반 early-stop과의 어긋남 때문. 소수 클래스 절대 표본 수가 안정성의 핵심 제약임을 다시 보여준다.

관련 그림: `![](figures/exp_stability_dinov3.png)`.

### 3.9 핵심 수치 정합성 대조 (QA)

본 보고서 작성 시 핵심 수치 3개를 소스 JSON과 직접 대조해 **완전 일치**를 확인했다(`ours_dinov3.json`, `ablation_dinov3.json`, `ours_detection.json` 재조회):
- **Ours+ focal+aug 3-class 원분포 PR-AUC = 0.7647**, F1-macro = 0.7743, baseline 대비 PR 상대 **+34.09%** / F1 +9.23%, 목표 판정 `pr_auc_met=true, f1_macro_met=false` — `ours_dinov3.json`과 일치.
- **Ablation 요인 분해 PR-AUC: aug +0.01055 / focal −0.00885 / 상호작용 +0.01779** — `ablation_dinov3.json` `decomposition_2x2`와 일치.
- **Ours detection 원분포 det PR-AUC = 1.0, IoU median = 0.5649** — `ours_detection.json` `orig_full`과 일치.

---

## 4. Discussion

- **불균형에서 지표 해석**: 원분포 valid는 정상이 ≈93%라 accuracy·AUROC가 포화(AUROC 6×3 전부 0.94–1.0)되어 백본을 거의 구분하지 못한다. 우열은 **PR-AUC·F1-macro·precision**으로 갈리며(0.34–0.97로 크게 벌어짐), 균형 valid 평가가 이 착시 폭을 정량적으로 드러낸다(3-class acc 일제 급락, AUROC 0.82–0.95로 하강).
- **공정성 한계**: baseline 6종은 from-scratch(전 파라미터 trainable), Ours는 DINOv3 자기지도 pretrained backbone frozen + head-only다. **사전학습 사용 여부가 달라 동일 조건 비교가 아니며**, Ours의 우위는 상당 부분 대규모 자기지도 사전학습에서 온다. 이는 표·그림에 일관되게 병기했다.
- **성능 동력 분해**: 가장 큰 동력은 **사전학습 전이**(3-class 원분포 PR-AUC baseline 0.570 → Ours+ 0.745, +30.7%; small@256→base@512에서 PR +0.056·F1 +0.052 추가). 그 위에 **강한 증강**이 frozen feature head의 정칙화·d4 결정경계 강화로 기여(주 동력), **focal은 강한 증강과 결합 시에만 이득**(조합 시너지). 즉 전이 ≫ 강한 증강 > focal(단독 음수, 조합 시 양수).
- **d4 소표본 병목(F1 절대 목표·우열 판정의 본질적 한계)**: valid disease_4 = 24장. base-CE 3-class confusion `[[1286,1,16],[1,50,25],[0,3,21]]`에서 d4 recall은 0.875로 양호하나 정상·d3가 d4로 새는 **d4 precision 0.339**가 macro-F1을 0.750에 묶는다. focal+aug가 정상→d4 오분류를 16→9장으로 줄여 d4 precision 0.339→0.396로 완화했으나, 95% CI가 0.276–0.531로 여전히 넓고 낮아 **F1 병목은 완화될 뿐 해소되지 않는다**. 이것이 PR-AUC로는 목표 초과·argmax-F1으로는 미달인 직접 원인이며, 모든 d4 관련 단일값 우열을 CI와 함께 봐야 하는 이유다.
- **거친 bbox·objectness 의미**: GT가 작물 영역 통째(중앙·≈프레임 절반)라 detection은 미세 병변 핀포인트가 아닌 작물 영역 학습에 가깝다. 따라서 **검출(있는지)은 강하고**(det PR-AUC 포화), **국소화(어디인지)는 상한**(IoU 0.57–0.67, mAP 0.46–0.60)이 본질적이다. mAP는 국소화 hit-rate에 묶여 det_pr_auc와 분리 해석해야 한다.
- **detection best-epoch 선택 이슈**: Ours detection은 det_pr_auc가 ep0부터 1.0 포화라 best=ep0이 선택돼 저장 predictions의 국소화 IoU가 전 epoch 최저(0.565)다. 검출 지표는 무관하나, 국소화 보고에는 이 한계를 명시했다(후반 ep ≈0.64).
- **견고성·데이터 효율(§3.7~§3.8)**: frozen 전이 구조 덕에 입력 노이즈에 강건하고(N_ratio 0.5까지 PR-AUC 저하 ≤2.4%), PR-AUC가 train 데이터량에 단조 증가·90%↑ 포화로 견조하게 스케일한다. 다만 F1-macro의 안정성은 다시 **소수 클래스(d4 N=24) 절대 표본 수**에 제약된다(train_ratio가 늘어도 출렁임).
- **한계·향후 과제**: (1) baseline pretrained 미로딩으로 절대 상한이 낮음(공정 비교 시 pretrained 병행 필요), (2) **부분 unfreeze 저-LR**로 도메인 적응 여지, (3) **d4 데이터 확충/오버샘플**·미세 병변 라벨 보강으로 F1 절대 목표·국소화·안정성 개선, (4) 다시즌 데이터로 외부 일반화 검증.

---

## 5. Conclusion

- **DINOv3(frozen) + 경량 2-layer head 전이**는 from-scratch baseline 대비, 가장 어려운 3-class 원분포에서 **PR-AUC 0.570 → 0.765(+34%)** 로 주지표 목표(+20%)를 견고히 달성했다. 사전학습 표현을 동결로 보존(no forgetting)하고 소수 파라미터(head 99k/395k)만 학습해 효율적이고 재현성이 높다.
- **검출(질병 유무)** 은 baseline·Ours 모두 사실상 포화(det PR-AUC 0.997–1.000, 정상 오경보 0.08–0.6%)했으나, **국소화는 거친 GT 특성상 상한**(IoU 0.57–0.67)이 분명하다.
- **개선의 정직한 귀속**: 전이가 주 동력, 강한 증강이 다음, focal은 증강과 결합 시에만 이득(조합 시너지). **3-class F1 절대 목표(0.851)는 d4 valid 24장 precision 병목으로 미달**이며, 이는 데이터 한계지 방법 한계가 아니다.
- **견고성·데이터 효율**: Ours는 입력 노이즈에 강건하고(PR-AUC 저하 ≤2.4%, §3.7) train 데이터량에 PR-AUC가 단조 증가·90%↑ 포화(§3.8)해, 적은 데이터·잡음 환경에서도 안정적이다.
- **실용 권고**: **질병 유무 스크리닝**에는 DINOv3 frozen + head 전이가 효율적·강력하므로 우선 권장한다. **3-class 세분화와 정밀 국소화**는 d4 데이터 확충·미세 병변 라벨 보강(필요 시 분류 + CAM 보조, 부분 unfreeze)이 선행되어야 한다.

---

### 부록: 산출물·재현

- 정합성 검증: `_workspace/eval/verify_<name>.md`(24 baseline + Ours/ablation 별도). 평가 데이터: `_workspace/eval/ours_dinov3.json`, `ablation_dinov3.json`, `ours_detection.json`, `balanced_valid.json`, `balanced_valid_detection.json`, `sensitivity_dinov3.json`, `stability_dinov3.json`, `summary.json`. 추가 분석 스크립트: `run_sensitivity_eval.py`, `run_stability_eval.py`.
- 그림: `report/figures/`(분류 `exp_metrics_table.png`/`exp_metrics_balanced.png`/`exp_cls_bars.png`/`exp_pr_curves.png`/`exp_confusion.png`, detection `exp_detection.png`/`exp_detection_balanced.png`, Ours `exp_ours_dinov3.png`/`exp_ours_focal.png`, ablation `exp_ablation_dinov3.png`, sensitivity `exp_sensitivity_dinov3.png`, stability `exp_stability_dinov3.png`, 곡선 `training_curves.png` 및 run별 `curves_*.png`).
- 상세 표·해석·변경 이력: `report/EXPERIMENTS.md`(메트릭 정의·§0 정합성·§1/§1B 분류·§3/§3B detection·§6 Ours·§6B ablation·부록 A). 데이터 분석: `report/REPORT.md`, `report/stats.json`. 모델 설계: `_workspace/specs/design_notes.md`.
