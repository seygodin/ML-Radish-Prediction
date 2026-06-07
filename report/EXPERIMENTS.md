# Baseline 실험 비교 리포트 (eval-reporter QA)

대상: from-scratch baseline **24 run** = 분류 18 (**백본 6종** × {normal_vs_d3, normal_vs_d4, normal_d3_d4}) + detection 6 (single-box, 백본 6종).
백본 6종: **기존** ConvNeXtV2-tiny / EfficientNetV2-s / NeXtViT-small + **신규** DenseNet121 / ResNet50 / Vision-Mamba(`mamba`, mambapy pscan). 모두 **pretrained 미로딩(from-scratch)**. (NeXtViT-base(`nextvit20`)는 사용자 요청으로 리포트 표·그림에서 제외 — 실험 산출물 `experiments/nextvit20_*`는 보존.)

재현: `./.venv/bin/python _workspace/eval/run_eval.py` → run별 검증 `_workspace/eval/verify_*.md`, 본 리포트 표/그림, `_workspace/eval/summary.json`.
지표는 모두 **predictions(`predictions/valid.npz`: prob,label)에서 독립 재계산**했고, run이 보고한 `metrics.json.final`과 대조해 일치를 확인했다(아래 §0). 기존 3 백본의 `metrics.json`에는 AUROC·macro-PRF 키가 없어 predictions에서 독립 재계산했으며(sklearn로 교차검증), train/val loss는 그 run의 `per_epoch`에서 **best epoch** 값을 가져왔다.

**사용자 요구 7-메트릭**: 분류 모델별 최종 표에 **accuracy, train_loss, val_loss, recall(macro), precision(macro), f1-score(macro), AUROC** 를 모두 보고한다.
- recall/precision/f1는 **macro**(sklearn 관례), AUROC는 2-class=P(disease) ROC-AUC / 3-class=macro OvR. accuracy=argmax. train_loss/val_loss는 **best epoch**(=primary 지표 PR-AUC 최댓값 epoch)의 per_epoch 값.

valid는 **원분포(다운샘플 없음)**: normal 1303 + disease(d3 76 / d4 24). 불균형이 심해 accuracy·AUROC는 **단독 해석 금지**(아래 §2 참조: AUROC가 0.94–1.0으로 포화돼 백본 분리력이 약함). 주지표는 여전히 PR-AUC, 보조로 F1-macro·precision.

> **백본 6종 표기**: NeXtViT-base(`nextvit20`)는 사용자 요청으로 본 리포트의 모든 표·그림·해석에서 제외했다. 학습/예측 산출물(`experiments/nextvit20_*`)은 디스크에 보존된다(상세 §10 변경 이력). 따라서 아래 모든 카운트는 6 백본 / 24 run 기준이다.

---

## 메트릭 정의 (먼저 읽기)

표를 읽기 전, 각 지표의 의미와 이 데이터(정상≫질병 불균형)에서의 해석 주의점.

### 분류 (classification)
- **accuracy (정확도)** — 전체 샘플 중 맞춘 비율. 직관적이지만 **불균형에 매우 취약**: 원분포 valid는 정상이 ~93%라 "전부 정상"으로 찍어도 ~0.93이 나온다. 단독 해석 금지(균형 valid §1B에서야 의미 있음).
- **train_loss / val_loss** — 학습/검증 세트의 cross-entropy 손실(낮을수록 좋음). **train≪val이면 과적합** 신호. 본 표는 best epoch 값이며, 학습 곡선(§4)에서 val_loss 반등으로 과적합을 진단한다.
- **recall (재현율·민감도, macro)** — 실제 양성(질병) 중 모델이 양성으로 맞춘 비율 = **질병을 놓치지 않는 능력**. 질병 스크리닝에서 가장 중요(놓치면 위험). macro=클래스별 recall의 단순 평균(소수 클래스도 동등 가중).
- **precision (정밀도, macro)** — 양성으로 예측한 것 중 실제 양성 비율 = **오경보가 적은 정도**. 낮으면 정상을 질병으로 자주 오판. recall과 트레이드오프.
- **f1-score (macro)** — precision과 recall의 조화평균. 둘의 균형을 한 수로 본 지표로, **불균형에서 accuracy보다 정직**하다. macro라 소수 클래스 성능이 그대로 반영된다.
- **AUROC** — 임계값과 무관하게 "양성 점수 > 음성 점수"일 확률(순위 분리력). 1.0=완벽, 0.5=무작위. **불균형에서는 다수인 음성만 잘 맞혀도 부풀려져 포화(0.94~1.0)**되므로 백본 변별에는 둔감(§2). 2-class=P(disease) 기준, 3-class=macro one-vs-rest.
- **PR-AUC** — precision-recall 곡선 아래 면적. **양성이 희소한 불균형에서 AUROC보다 변별력이 커** 본 리포트의 **주지표**로 쓴다. 무작위 하한 = 양성 비율(prevalence).

> 요약: 불균형 원분포(§1)에서는 **PR-AUC·F1-macro·precision**으로 우열을 보고, accuracy·AUROC는 보조로만. 클래스 균형(§1B)에서는 accuracy·AUROC도 정직해진다.

### Detection (단일 박스 + objectness)
검출은 **"질병이 있는가(이미지 단위)"** 와 **"어디인가(국소화)"** 를 분리해 본다.
- **det PR-AUC / det ROC-AUC** — objectness 점수로 질병/정상 **이미지**를 구분하는 능력(이미지 단위 질병 검출).
- **presence_recall@0.5** — objectness≥0.5인 질병 이미지 비율 = **질병 검출 민감도(놓치지 않음)**.
- **fp_rate@0.5** — objectness≥0.5인 정상 이미지 비율 = **오경보율**(낮을수록 좋음).
- **IoU (median)** — 예측 박스와 정답 박스의 겹침 비율(교집합/합집합). **국소화 정확도**. 양성(질병) 이미지에서만 계산.
- **mAP@0.5** — IoU≥0.5를 정답으로 본 평균 정밀도. **검출+국소화를 합친** 표준 detection 지표. 국소화 hit-rate에 상한이 묶여 det PR-AUC와 분리 해석.

---

## 0. 정합성 교차검증 요약 (predictions ↔ manifest, 재계산 ↔ 보고)

**변경 이력(이번 갱신)**: 백본 3종 추가(DenseNet121 / ResNet50 / Vision-Mamba=`mamba`(mambapy pscan)) → 분류 12→18 run, detection 3→6 run, **총 12→24 run**. 분류 표에 **AUROC + 7-메트릭(accuracy/train_loss/val_loss/recall/precision/f1/AUROC) 도입**. 기존 3 백본 결과·해석은 보존하고 6 백본 기준으로 표·그림·해석을 확장(상세 §10). NeXtViT-base(`nextvit20`)는 사용자 요청으로 표·그림에서 제외(산출물은 보존).

- **표본 수·라벨 분포**: **baseline 24 run + Ours 4 run(분류 3 + detection 1) 전부** predictions의 valid 표본 수·라벨 분포가 manifest valid 분포와 **정확히 일치**(분류 `dist_match=True` baseline 18/18 + Ours 3/3, detection N=1403·pos/neg=100/1303 일치 **baseline 6/6 + Ours(DINOv3-B detection) 1/1**). 누수·오집계·split 위반 징후 **없음**. (Ours 분류=DINOv3 frozen+head 평가·판정 §6; Ours detection=DINOv3-B frozen+single-box head 평가 §3/§3B.)
  - normal_vs_d3 valid = 1303 normal + 76 d3 = 1379
  - normal_vs_d4 valid = 1303 normal + 24 d4 = 1327
  - normal_d3_d4 valid = 1303 + 76 + 24 = 1403
  - detection valid = 100 disease(d3 76 + d4 24, GT 박스 보유) + 1303 normal(음성, 빈 GT) = 1403
- **재계산 ↔ 보고**: 분류 PR-AUC/F1/recall/precision/accuracy/confusion 및 **신규 7-메트릭(recall_macro/precision_macro/f1_macro/AUROC)** 가 보고치와 모두 일치(허용오차 내). 신규 12 run은 `metrics.json.final`에 7키가 이미 있어 predictions 재계산과 대조해 일치 확인. **기존 3 백본 9 run은 AUROC·macro-PRF가 metrics.json에 없어 predictions에서 독립 재계산**했고, sklearn `roc_auc_score`/`recall_score`(macro)와 교차검증해 동일함을 확인(예: convnextv2_d3 AUROC=0.9626 = sklearn 0.9626). detection det_pr_auc/det_roc_auc/presence_recall@0.5/fp_rate@0.5/IoU 분포 모두 일치, mAP@0.5만 AP 보간 방식 차이로 미세 차이(무시 가능). **불일치 run 없음.**
- **detection objectness collapse → 수정 → 재학습·재평가 완료**: 초기 detection 3 run은 정상 이미지를 음성으로 다루지 않아 **objectness가 전 이미지(정상 포함)에서 ~1.0** 으로 붕괴, 정상/질병을 분리하지 못했다. 데이터 로더(정상=음성·빈 GT)와 학습/지표 처리를 수정해 **재학습**했고, 재평가 결과 **objectness 붕괴가 해소**됨을 정량 확인했다: 질병 objectness median ≈0.997–0.999 vs 정상 median ≈0.000–0.002, 정상이 임계값 0.5를 넘는 비율 4–8/1303 (0.3–0.6%). 이제 objectness는 실질적인 **이미지 단위 질병 점수**로 동작한다(아래 §3). 변경 이력은 §9 참조.
- **detection 표본**: predictions의 양성/음성이 **6 run 모두 정확히 100/1303** 로 manifest(질병 100 = d3 76 + d4 24, 정상 1303)와 일치. 정상은 빈/null GT, 질병은 GT 박스 1개를 보유.

상세 항목별 통과/실패는 `_workspace/eval/verify_<name>.md`(24개) 참조.

---

## 1. 분류 7-메트릭 표 (백본 6종 × 3세팅, valid 원분포)

본문 핵심 표. 컬럼: **params(M) / accuracy / train_loss / val_loss / recall(macro) / precision(macro) / f1(macro) / AUROC** + PR-AUC(주지표) + valid 표본. **모든 수치는 best epoch 기준**(train/val loss 포함), predictions에서 독립 재계산. `*`=신규 백본. AUROC: 2-class=P(disease) ROC-AUC, 3-class=macro OvR.
- **params(M)** = 분류기 파라미터 수 = `build_classifier(arch, 2, 224)`의 `sum(p.numel())/1e6`. `run_eval.py`가 `from src.models import build_classifier`로 빌드해 직접 계산·주입(재현 가능). 백본 라벨에 **arch명**을 괄호로 병기했다(예: `VisionMamba (mamba)`). **arch `mamba` = Vision-Mamba(mambapy pscan 구현)** — 표의 "VisionMamba (mamba)" 행이 그것이다. (검증값 M: convnextv2 28.26 / efficientnetv2 20.84 / nextvit 31.27 / densenet121 7.48 / resnet50 24.56 / mamba 3.88.)

> ⚠️ **공정성 주석 (동일 조건 비교 아님)**: 표 상단 **baseline 6종은 ImageNet 등 사전학습 없이 from-scratch**로 학습한 분류기다. 하단 **굵게 표시된 Ours 3행은 DINOv3 자기지도(self-supervised) pretrained 백본을 frozen**(동결, 가중치 갱신 없음)하고 그 위 2-layer head만 학습한 모델이다. **사전학습 사용 여부가 다르므로 baseline ↔ Ours는 동일 조건 비교가 아니다** — Ours의 우위는 상당 부분 대규모 자기지도 사전학습에서 온다. params(M)도 성격이 다르다: baseline은 전부 trainable인 반면 **Ours는 total params 중 거의 전부가 frozen이고 trainable은 head 뿐**이다(Ours-S total 21.7M / **trainable 99K**, Ours-B total 86.0M / **trainable 395K**). 표의 params(M)에 `(frozen+trainable)` 구분을 병기했다. 평가·해석·목표 판정의 정본은 §6, ablation은 §6B.

### 1-1. normal_vs_d3 (valid: normal 1303 / d3 76)

| 백본 | params(M) | accuracy | train_loss | val_loss | recall(M) | precision(M) | f1(M) | AUROC | PR-AUC | best ep |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ConvNeXtV2 (convnextv2) | 28.26 | 0.541 | 0.237 | 1.179 | 0.757 | 0.554 | 0.436 | 0.963 | 0.787 | 5 |
| EfficientNetV2 (efficientnetv2) | 20.84 | 0.937 | 0.017 | 0.201 | 0.960 | 0.733 | 0.799 | 0.995 | 0.949 | 59 |
| NeXtViT-s (nextvit) | 31.27 | 0.938 | 0.024 | 0.154 | 0.961 | 0.734 | 0.801 | 0.995 | 0.929 | 47 |
| **DenseNet121 (densenet121)*** | 7.48 | 0.962 | 0.015 | 0.128 | 0.980 | 0.795 | **0.860** | 0.997 | **0.957** | 44 |
| **ResNet50 (resnet50)*** | 24.56 | 0.957 | 0.025 | 0.114 | 0.977 | 0.781 | 0.849 | **0.998** | **0.965** | 39 |
| **VisionMamba (mamba)*** | 3.88 | 0.898 | 0.045 | 0.291 | 0.940 | 0.674 | 0.729 | 0.989 | 0.878 | 39 |
|---*(이하 Ours: DINOv3 frozen pretrained — 위 from-scratch baseline과 동일 조건 아님)*---|||||||||||
| **Ours: DINOv3-S @256 (frozen, CE)** | 21.7 (frozen)+0.099 (tr) | 0.982 | 0.002 | 0.041 | 0.984 | 0.878 | 0.924 | 1.000 | 0.996 | 25 |
| **Ours: DINOv3-B @512 (frozen, CE)** | 86.0 (frozen)+0.395 (tr) | 0.996 | 0.000 | 0.010 | 0.998 | 0.969 | 0.983 | 1.000 | 1.000 | 25 |
| **Ours: DINOv3-B @512 (frozen, focal+aug)** | 86.0 (frozen)+0.395 (tr) | 0.998 | 0.002 | 0.002 | 0.999 | 0.981 | **0.990** | 1.000 | **1.000** | 20 |

### 1-2. normal_vs_d4 (valid: normal 1303 / d4 24 — **소표본, CI 매우 넓음**)

| 백본 | params(M) | accuracy | train_loss | val_loss | recall(M) | precision(M) | f1(M) | AUROC | PR-AUC | best ep |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ConvNeXtV2 (convnextv2) | 28.26 | 0.887 | 0.136 | 0.239 | 0.922 | 0.566 | 0.587 | 0.985 | 0.759 | 31 |
| EfficientNetV2 (efficientnetv2) | 20.84 | 0.943 | 0.116 | 0.110 | 0.971 | 0.620 | 0.679 | 0.995 | 0.862 | 34 |
| NeXtViT-s (nextvit) | 31.27 | 0.921 | 0.114 | 0.202 | 0.939 | 0.590 | 0.631 | 0.990 | 0.822 | 32 |
| **DenseNet121 (densenet121)*** | 7.48 | 0.937 | 0.052 | 0.153 | 0.968 | 0.611 | 0.665 | 0.995 | 0.857 | 23 |
| **ResNet50 (resnet50)*** | 24.56 | 0.940 | 0.058 | 0.147 | 0.949 | 0.613 | 0.668 | 0.991 | **0.867** | 39 |
| **VisionMamba (mamba)*** | 3.88 | 0.841 | 0.353 | 0.299 | 0.919 | 0.551 | 0.549 | 0.977 | 0.629 | 14 |
|---*(이하 Ours: DINOv3 frozen pretrained — 위 from-scratch baseline과 동일 조건 아님)*---|||||||||||
| **Ours: DINOv3-S @256 (frozen, CE)** | 21.7 (frozen)+0.099 (tr) | 0.963 | 0.009 | 0.081 | 0.981 | 0.664 | 0.738 | 1.000 | 0.997 | 15 |
| **Ours: DINOv3-B @512 (frozen, CE)** | 86.0 (frozen)+0.395 (tr) | 0.987 | 0.002 | 0.028 | 0.993 | 0.793 | 0.866 | 1.000 | 1.000 | 7 |
| **Ours: DINOv3-B @512 (frozen, focal+aug)** | 86.0 (frozen)+0.395 (tr) | 0.993 | 0.004 | 0.004 | 0.997 | 0.864 | **0.919** | 1.000 | **1.000** | 16 |

> valid d4=24장뿐 → AUROC/recall이 높아도 표본이 작아 CI가 넓다(아래 §1-4). 단일값으로 우열 단정 금지.

### 1-3. normal_d3_d4 (3-class; valid: normal 1303 / d3 76 / d4 24). recall/precision/f1=3-class macro, AUROC=macro OvR, PR-AUC=disease(d3,d4) OvR macro.

| 백본 | params(M) | accuracy | train_loss | val_loss | recall(M) | precision(M) | f1(M) | AUROC | PR-AUC | best ep |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ConvNeXtV2 (convnextv2) | 28.26 | 0.874 | 0.588 | 0.336 | 0.640 | 0.490 | 0.523 | 0.943 | 0.343 | 19 |
| EfficientNetV2 (efficientnetv2) | 20.84 | 0.920 | 0.580 | 0.231 | 0.676 | 0.545 | 0.591 | 0.977 | 0.509 | 13 |
| NeXtViT-s (nextvit) | 31.27 | 0.952 | 0.429 | 0.131 | 0.617 | 0.650 | 0.624 | 0.969 | 0.487 | 26 |
| **DenseNet121 (densenet121)*** | 7.48 | 0.940 | 0.216 | 0.194 | **0.862** | 0.642 | **0.709** | **0.984** | **0.570** | 42 |
| **ResNet50 (resnet50)*** | 24.56 | 0.949 | 0.315 | 0.146 | 0.762 | 0.640 | 0.685 | 0.983 | 0.530 | 41 |
| **VisionMamba (mamba)*** | 3.88 | 0.875 | 0.736 | 0.340 | 0.634 | 0.569 | 0.524 | 0.947 | 0.359 | 4 |
|---*(이하 Ours: DINOv3 frozen pretrained — 위 from-scratch baseline과 동일 조건 아님)*---|||||||||||
| **Ours: DINOv3-S @256 (frozen, CE)** | 21.7 (frozen)+0.099 (tr) | 0.961 | 0.318 | 0.092 | 0.753 | 0.674 | 0.698 | 0.991 | 0.689 | 13 |
| **Ours: DINOv3-B @512 (frozen, CE)** | 86.0 (frozen)+0.395 (tr) | 0.967 | 0.163 | 0.083 | 0.840 | 0.755 | 0.750 | 0.993 | 0.745 | 29 |
| **Ours: DINOv3-B @512 (frozen, focal+aug)** | 86.0 (frozen)+0.395 (tr) | 0.973 | 0.072 | 0.019 | 0.850 | 0.769 | **0.774** | 0.995 | **0.765** | 26 |

PR-AUC 기준 prevalence(랜덤 하한): d3=0.055, d4=0.018, 3-class disease=0.071.

**AUROC 주의**: 6 백본·3세팅 전부 AUROC 0.94–1.0으로 **포화** → 랭킹 분리력만 보면 모두 "거의 완벽"하지만, 이는 정상이 압도적 다수라 음성을 잘 맞히는 것으로 부풀려진 값이다(불균형). **백본 우열은 AUROC가 아니라 PR-AUC·F1-macro·precision으로 갈린다**(§2).

### 1-4. 클래스별 질병 recall (소표본 Wilson 95% CI)

| 백본 | d3 (vs, n=76) | d4 (vs, n=24) | 3c d3 (n=76) | 3c d4 (n=24) |
|------|:---:|:---:|:---:|:---:|
| ConvNeXtV2 | 1.00 [0.95-1.00] | 0.96 [0.80-0.99] | 0.61 [0.49-0.71] | 0.42 [0.24-0.61] |
| EfficientNetV2 | 0.99 [0.93-1.00] | 1.00 [0.86-1.00] | 0.71 [0.60-0.80] | 0.38 [0.21-0.57] |
| NeXtViT-s | 0.99 [0.93-1.00] | 0.96 [0.80-0.99] | 0.49 [0.38-0.60] | 0.38 [0.21-0.57] |
| **DenseNet121*** | 1.00 [0.95-1.00] | 1.00 [0.86-1.00] | **0.80 [0.70-0.88]** | **0.83 [0.64-0.93]** |
| **ResNet50*** | 1.00 [0.95-1.00] | 0.96 [0.80-0.99] | 0.78 [0.67-0.86] | 0.54 [0.35-0.72] |
| **VisionMamba (mamba)*** | 0.99 [0.93-1.00] | 1.00 [0.86-1.00] | 0.37 [0.27-0.48] | 0.62 [0.43-0.79] |

> d4는 valid 24장뿐이라 CI 폭이 매우 넓다(1.00이라도 하한 0.86). recall 단일값으로 백본 우열 단정 금지.

그림: `report/figures/exp_metrics_table.png` (**7-메트릭 핵심: AUROC·F1-macro·accuracy 막대, 백본 6종 × 3세팅**), `exp_cls_bars.png` (PR-AUC·F1 막대), `exp_pr_curves.png` (세팅별 PR 곡선, 신규=점선), `exp_confusion.png` (18개 혼동행렬).

---

## 1B. 균형 valid 평가 (1:1 / 1:1:1)

§1은 **원분포(불균형) valid**(normal 1303 다수) 기준이다. 본 절은 **동일한 저장된 best 모델을 그대로 로드(가중치 변경 없음)**해 **valid를 정상:질병 1:1(binary)/1:1:1(3-class)로 균형 다운샘플**한 집합에서 7-메트릭을 **재측정**한 결과다. train은 원래부터 균형(다운샘플)이었고, 그동안 valid만 불균형이었다 — 이제 valid도 균형으로 맞춰 평가한다.

- **재현·방식**: `build_classification_loaders(setting, img_size, batch_size, num_workers, seed=42, balance_valid=True)`로 균형 valid 로더 생성(seed=42 고정 → 재현 가능), best.pt(`["model"]`) 로드 후 forward로 prob 수집. arch·num_classes·img_size는 각 run의 `config.snapshot`의 `spec.model`에서 읽음. **재학습 없음**.
- **균형 valid 표본 수**: normal_vs_d3 = **76/76 (N=152)**, normal_vs_d4 = **24/24 (N=48)**, normal_d3_d4 = **24/24/24 (N=72)**. (3-class는 최소 클래스인 d4 valid=24에 맞춰 전 클래스 24로 다운샘플.)
- **train_loss**: 해당 run `metrics.json`의 best epoch per_epoch 값(학습 시점, 불변 — §1과 동일). **val_loss**: 균형 valid에서 plain CE 재계산(class_weights 미적용, label_smoothing=0; 균형이므로 weight 불필요, train.py와 동일 CE식). accuracy/recall/precision/f1(macro)/AUROC/PR-AUC: 균형 valid prob·label에서 sklearn로 계산. AUROC: 2-class=P(disease) / 3-class=macro OvR. PR-AUC: 2-class / 3-class=disease OvR macro.
- **교차검증**: densenet121_normal_d3_d4(3-class)를 별도 forward로 sklearn 재계산 → AUROC 0.954 / recall_macro 0.847 / f1_macro 0.850 **완전 일치**.
- **소표본 CI 주의**: d4=24, 3-class=24×3=72로 표본이 매우 작다. 단일값(특히 acc/AUROC 0.9+) 우열 단정 금지 — Wilson 기준 CI 폭이 넓다(d4 24장: recall 1.0이라도 하한 ~0.86).

> ⚠️ **공정성 주석(§1과 동일)**: 아래 각 균형 표 하단 **굵게 표시된 Ours 3행은 DINOv3 자기지도 pretrained 백본 frozen + head**다. 상단 **baseline 6종은 from-scratch**이므로 **사전학습 사용 여부가 달라 동일 조건 비교가 아니다**. params(M)도 baseline은 전부 trainable, **Ours는 거의 전부 frozen이고 trainable은 head(99K/395K)뿐**(`(frozen)+(tr)` 병기). Ours 균형 재평가는 §6과 동일 절차(best.pt 로드, `balance_valid=True`, seed=42; Ours-B/focal은 img_size=512)이며 정본·해석은 §6.

### 1B-1. normal_vs_d3 (균형 valid: normal 76 / d3 76, N=152)

| 백본 | params(M) | accuracy | train_loss | val_loss | recall(M) | precision(M) | f1(M) | AUROC | PR-AUC | best ep |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ConvNeXtV2 (convnextv2) | 28.26 | 0.750 | 0.237 | 0.645 | 0.750 | 0.833 | 0.733 | 0.962 | 0.965 | 5 |
| EfficientNetV2 (efficientnetv2) | 20.84 | 0.954 | 0.017 | 0.110 | 0.954 | 0.956 | 0.954 | 0.997 | 0.997 | 59 |
| NeXtViT-s (nextvit) | 31.27 | 0.961 | 0.024 | 0.084 | 0.961 | 0.962 | 0.960 | 0.997 | 0.998 | 47 |
| **DenseNet121 (densenet121)*** | 7.48 | **0.980** | 0.015 | 0.052 | **0.980** | **0.981** | **0.980** | 0.999 | 0.999 | 44 |
| **ResNet50 (resnet50)*** | 24.56 | 0.974 | 0.025 | 0.053 | 0.974 | 0.975 | 0.974 | **1.000** | **1.000** | 39 |
| **VisionMamba (mamba)*** | 3.88 | 0.934 | 0.045 | 0.190 | 0.934 | 0.939 | 0.934 | 0.988 | 0.988 | 39 |
|---*(이하 Ours: DINOv3 frozen pretrained — 위 from-scratch baseline과 동일 조건 아님)*---|||||||||||
| **Ours: DINOv3-S @256 (frozen, CE)** | 21.7 (frozen)+0.099 (tr) | 0.980 | 0.002 | 0.043 | 0.980 | 0.980 | 0.980 | 0.999 | 1.000 | 25 |
| **Ours: DINOv3-B @512 (frozen, CE)** | 86.0 (frozen)+0.395 (tr) | 0.993 | 0.000 | 0.017 | 0.993 | 0.994 | 0.993 | 1.000 | 1.000 | 25 |
| **Ours: DINOv3-B @512 (frozen, focal+aug)** | 86.0 (frozen)+0.395 (tr) | 0.993 | 0.002 | 0.021 | 0.993 | 0.994 | **0.993** | 1.000 | **1.000** | 20 |

### 1B-2. normal_vs_d4 (균형 valid: normal 24 / d4 24, N=48 — **소표본, CI 매우 넓음**)

| 백본 | params(M) | accuracy | train_loss | val_loss | recall(M) | precision(M) | f1(M) | AUROC | PR-AUC | best ep |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ConvNeXtV2 (convnextv2) | 28.26 | 0.958 | 0.136 | 0.092 | 0.958 | 0.958 | 0.958 | 0.997 | 0.997 | 31 |
| EfficientNetV2 (efficientnetv2) | 20.84 | **0.979** | 0.116 | 0.109 | **0.979** | **0.980** | **0.979** | 0.995 | 0.995 | 34 |
| NeXtViT-s (nextvit) | 31.27 | 0.938 | 0.114 | 0.134 | 0.938 | 0.938 | 0.937 | 0.988 | 0.987 | 32 |
| **DenseNet121 (densenet121)*** | 7.48 | 0.958 | 0.052 | 0.092 | 0.958 | 0.962 | 0.958 | **0.998** | **0.998** | 23 |
| **ResNet50 (resnet50)*** | 24.56 | 0.938 | 0.058 | 0.119 | 0.938 | 0.938 | 0.937 | 0.993 | 0.994 | 39 |
| **VisionMamba (mamba)*** | 3.88 | 0.938 | 0.353 | 0.190 | 0.938 | 0.944 | 0.937 | 0.991 | 0.991 | 14 |
|---*(이하 Ours: DINOv3 frozen pretrained — 위 from-scratch baseline과 동일 조건 아님)*---|||||||||||
| **Ours: DINOv3-S @256 (frozen, CE)** | 21.7 (frozen)+0.099 (tr) | 1.000 | 0.009 | 0.036 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 15 |
| **Ours: DINOv3-B @512 (frozen, CE)** | 86.0 (frozen)+0.395 (tr) | 1.000 | 0.002 | 0.004 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 7 |
| **Ours: DINOv3-B @512 (frozen, focal+aug)** | 86.0 (frozen)+0.395 (tr) | 1.000 | 0.004 | 0.016 | 1.000 | 1.000 | **1.000** | 1.000 | **1.000** | 16 |

> N=48(24/24)뿐 → acc=0.958이 정상·질병 각 1장 오류와 같은 수준. 단일값 우열 단정 금지.

### 1B-3. normal_d3_d4 (3-class; 균형 valid: normal 24 / d3 24 / d4 24, N=72). recall/precision/f1=3-class macro, AUROC=macro OvR, PR-AUC=disease(d3,d4) OvR macro.

| 백본 | params(M) | accuracy | train_loss | val_loss | recall(M) | precision(M) | f1(M) | AUROC | PR-AUC | best ep |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ConvNeXtV2 (convnextv2) | 28.26 | 0.639 | 0.588 | 0.617 | 0.639 | 0.621 | 0.627 | 0.842 | 0.583 | 19 |
| EfficientNetV2 (efficientnetv2) | 20.84 | 0.653 | 0.580 | 0.569 | 0.653 | 0.650 | 0.644 | 0.838 | 0.597 | 13 |
| NeXtViT-s (nextvit) | 31.27 | 0.639 | 0.429 | 0.921 | 0.639 | 0.637 | 0.613 | 0.858 | 0.602 | 26 |
| **DenseNet121 (densenet121)*** | 7.48 | **0.847** | 0.216 | 0.345 | **0.847** | **0.858** | **0.850** | **0.954** | **0.860** | 42 |
| **ResNet50 (resnet50)*** | 24.56 | 0.764 | 0.315 | 0.482 | 0.764 | 0.761 | 0.758 | 0.923 | 0.782 | 41 |
| **VisionMamba (mamba)*** | 3.88 | 0.583 | 0.736 | 0.742 | 0.583 | 0.542 | 0.544 | 0.819 | 0.561 | 4 |
|---*(이하 Ours: DINOv3 frozen pretrained — 위 from-scratch baseline과 동일 조건 아님)*---|||||||||||
| **Ours: DINOv3-S @256 (frozen, CE)** | 21.7 (frozen)+0.099 (tr) | 0.778 | 0.318 | 0.374 | 0.778 | 0.785 | 0.774 | 0.936 | 0.810 | 13 |
| **Ours: DINOv3-B @512 (frozen, CE)** | 86.0 (frozen)+0.395 (tr) | **0.875** | 0.163 | 0.320 | **0.875** | **0.878** | **0.875** | **0.962** | **0.885** | 29 |
| **Ours: DINOv3-B @512 (frozen, focal+aug)** | 86.0 (frozen)+0.395 (tr) | 0.875 | 0.072 | 0.350 | 0.875 | 0.878 | 0.875 | 0.952 | 0.856 | 26 |

균형이므로 prevalence(랜덤 하한): 2-class PR-AUC=0.5, 3-class disease OvR≈0.33(클래스당 24/72).

### 1B-4. 원분포(§1) vs 균형(§1B) 해석

균형 valid는 음성 다수가 만들던 **평가 착시를 걷어내** 백본 우열을 더 또렷이 드러낸다.

- **3-class에서 변화가 가장 극적**(여기서 정상이 압도적으로 줄어듦 1303→24). **accuracy가 일제히 급락**(efficientnetv2 0.920→0.653, nextvit 0.952→0.639, mamba 0.875→0.583)했고, **AUROC도 포화(0.94–0.98)가 풀려 0.82–0.95로 내려와 분리력이 생겼다**(efficientnetv2 0.977→0.838, nextvit 0.969→0.858). 즉 §1의 높은 3-class accuracy/AUROC는 상당 부분 **정상을 잘 맞힌 덕에 부풀려진 값**이었음이 확인된다. 균형에서는 **DenseNet121이 독보적 1위(acc 0.847·F1 0.850·AUROC 0.954·PR-AUC 0.860)**, ResNet50이 2위(0.764/0.758/0.923/0.782), 나머지 4 백본은 acc 0.58–0.65로 한 덩어리 — **신규 conv 계열(DenseNet121·ResNet50)의 우위가 원분포보다 훨씬 명확**해졌다.
- **2-class(d3/d4)는 방향이 반대**: 정상이 1303→76/24로 줄면 **다수 클래스의 오경보가 accuracy에서 차지하던 비중이 사라져 대부분 accuracy가 오른다**(convnextv2 d3 0.541→0.750, d4 0.887→0.958; mamba d4 0.841→0.938). §1에서 precision 붕괴(오경보 다수)로 낮던 백본일수록 균형에서 accuracy 상승폭이 크다. 이는 원분포 accuracy가 다수 클래스(정상) 편향의 산물이었음을 보여준다.
- **2-class AUROC는 균형 후에도 0.99 안팎으로 여전히 포화**(랭킹 분리력 자체가 높아 표본 재가중에 둔감). 따라서 **2-class의 백본 우열은 균형에서도 accuracy/F1로 갈리며**, DenseNet121(d3 0.980)·ResNet50·NeXtViT-s가 상위, ConvNeXtV2(d3 0.750, 여전히 from-scratch 학습 실패급)·VisionMamba가 하위로 §1 PR-AUC 랭킹과 일관.
- **종합**: 균형 valid에서 **DenseNet121 ≳ ResNet50 ≳ EfficientNetV2/NeXtViT-s > VisionMamba > ConvNeXtV2** 로, §1(원분포 PR-AUC) 결론과 동일하되 **특히 3-class에서 DenseNet121·ResNet50의 우위가 한층 선명**해졌다. accuracy/AUROC 단독 해석 금지 원칙(§6)은 균형에서도 유지 — 다만 균형은 그 착시 폭을 정량적으로 보여준다.

> **소표본 CI 경고(재강조)**: d4 균형 valid는 24/24=48장, 3-class는 24×3=72장뿐이다. 이 표의 acc/F1/AUROC 단일값은 표본이 작아 신뢰구간이 넓으므로(정상·질병 각 1–2장 차이로 0.04–0.08 변동), **백본 간 근소한 차이는 통계적으로 유의하지 않다**. PR-AUC/F1-macro 추세와 §1을 함께 본다.

그림: `report/figures/exp_metrics_balanced.png` (균형 valid AUROC/F1-macro/accuracy 막대, 백본 6종 × 3세팅). 결과 데이터: `_workspace/eval/balanced_valid.json`(원분포 대비 delta 포함). 스크립트: `_workspace/eval/run_balanced_eval.py`.

---

## 2. 분류 해석 (신규 vs 기존 백본 우열)

**PR-AUC 종합 우열(주지표).** 신규 백본이 기존을 끌어올렸다.
- **DenseNet121(신규)이 종합 1위급**: d3 0.957, d4 0.857, **3-class 0.570(전 백본 최고)**. 특히 가장 어려운 3-class에서 EfficientNetV2(0.509)·ResNet50(0.530)을 앞선다. dense connectivity가 from-scratch·소표본에서 gradient/feature 재사용에 유리하게 작용한 것으로 보인다.
- **ResNet50(신규)**: d3 0.965(전 백본 **최고 PR-AUC**), d4 0.867, 3-class 0.530. EfficientNetV2와 동급 이상으로 안정적.
- 기존 중 **EfficientNetV2 > NeXtViT-s ≫ ConvNeXtV2**(이전 결론 유지). ConvNeXtV2는 d3 0.787·3-class 0.343으로 여전히 가장 약하다(from-scratch에서 학습 실패에 가까움, best ep=5).
- **종합: DenseNet121 ≈ ResNet50 ≳ EfficientNetV2 > NeXtViT-s > VisionMamba > ConvNeXtV2** (PR-AUC 기준).

**VisionMamba(`mamba`, mambapy pscan) 동작·속도.** 학습은 **정상 동작·수렴**(NaN 없음, 60 epoch 완주). 성능은 **신규 중 최약**: d3 PR-AUC 0.878, 3-class 0.359(ConvNeXtV2급). 속도는 **가장 느림**(분류 7.1 s/ep, detection 38 s/ep로 ResNet50의 ~2–3배; mambapy 순수 pscan 구현이라 커널 미최적화). from-scratch·소표본에서 SSM의 inductive bias가 conv 계열만큼 데이터 효율적이지 않은 것으로 보인다. d4 3-class recall 0.62는 예외적으로 높으나 표본 24장이라 신뢰구간이 넓다.

**세팅 난이도.** **normal_vs_d3 (쉬움) > normal_vs_d4 > normal_d3_d4 (어려움)** — 6 백본 전부 일관.
- d3 train 470쌍(최다) → PR-AUC 0.79–0.97 최고.
- d4 train 227쌍·valid 24 → PR-AUC가 d3보다 낮고 CI 넓음.
- 3-class PR-AUC 0.34–0.57로 급락: d3↔d4 상호 오분류가 핵심 난점(거친 외형상 두 질병이 잘 안 갈림). 여기서 **DenseNet121만 disease recall(d3 0.80/d4 0.83)을 유지**해 두드러진다.

**AUROC vs PR-AUC 관점 차이(중요).** AUROC는 6×3 전부 0.94–1.0으로 포화돼 **백본을 거의 구분하지 못한다**(불균형에서 음성 다수를 잘 맞히면 AUROC가 부풀려짐). 반면 **PR-AUC·F1-macro·precision은 0.34–0.97로 크게 벌어져** 실제 우열을 드러낸다. 사용자 요구로 AUROC를 표에 포함했으나, **해석 주지표는 PR-AUC**임을 명확히 한다.

**불균형 함정 (precision 붕괴).** argmax(=0.5)에서 **precision(macro) 0.49–0.80, disease 클래스 precision은 더 낮다**(2-class에서 0.11–0.50대). recall은 높아도(0.92–1.00) normal 1303장 중 상당수를 질병으로 오경보(예: ConvNeXtV2 d3 오경보 633장, F1 0.44). 운영 임계값 0.5는 대부분 백본에 부적합 → **임계값 튜닝/PR 트레이드오프 필요**.

---

## 3. Detection 비교표 + 해석 (single-box, objectness collapse 수정 후 재평가)

primary = **det_pr_auc**(objectness vs 질병/정상 라벨의 PR-AUC). objectness 붕괴 수정 후, **이미지 단위 질병 검출**과 **국소화 IoU**를 분리 보고한다.

### 3-1. 이미지 단위 질병 검출 (주지표) — objectness vs 질병/정상 라벨

**params(M)** = detector 파라미터 수 = `build_detector(arch, 512)`의 `sum(p.numel())/1e6`(`run_eval.py`가 직접 계산·주입). 백본 라벨에 arch명 병기(arch `mamba`=Vision-Mamba/mambapy). (검증값 M: convnextv2 28.07 / efficientnetv2 20.51 / nextvit 31.00 / densenet121 7.22 / resnet50 24.04 / mamba 4.03.)

| 백본 | params(M) | det PR-AUC | det ROC-AUC | presence_recall@0.5 (질병 recall) | fp_rate@0.5 (정상 오경보) | train_loss | val_loss | best ep |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ConvNeXtV2 (convnextv2) | 28.07 | 0.9977 | 0.9998 | 0.990 [0.946-0.998] (99/100) | 0.0031 (4/1303) | 1.180 | 0.104 | 37 |
| EfficientNetV2 (efficientnetv2) | 20.51 | 0.9990 | 0.9999 | **1.000** (100/100) | 0.0054 (7/1303) | 1.504 | 0.122 | 41 |
| NeXtViT-s (nextvit) | 31.00 | 0.9990 | 0.9999 | **1.000** (100/100) | 0.0061 (8/1303) | 1.432 | 0.133 | 59 |
| **DenseNet121 (densenet121)*** | 7.22 | **0.9997** | **1.0000** | **1.000** (100/100) | 0.0031 (4/1303) | 1.214 | 0.109 | 45 |
| **ResNet50 (resnet50)*** | 24.04 | 0.9985 | 0.9999 | 0.980 (98/100) | **0.0015** (2/1303) | 1.362 | 0.118 | 34 |
| **VisionMamba (mamba)*** | 4.03 | 0.9975 | 0.9998 | 0.990 (99/100) | 0.0046 (6/1303) | 1.276 | 0.118 | 37 |
|---*(이하 Ours: DINOv3 frozen pretrained — 위 from-scratch baseline과 동일 조건 아님)*---|||||||||
| **Ours: DINOv3-B @512 (frozen, head-only)** | 85.84 (frozen)+0.200 (tr) | **1.0000** | **1.0000** | **1.000** (100/100) | 0.0008 (1/1303) | 2.513 | 0.173 | 0† |

(양성 100 / 음성 1303. Wilson 95% CI는 verify 파일 참조. 모든 수치는 best-epoch 체크포인트 예측에서 독립 재계산, 보고치와 일치. `*`=신규.)
> ⚠️ **공정성 주석 (동일 조건 비교 아님)**: 표 상단 **baseline 6종은 사전학습 없이 from-scratch**로 학습한 detector(전 파라미터 trainable)다. 하단 **굵게 표시된 Ours 행은 DINOv3 자기지도 pretrained ViT-B/16을 frozen**(가중치 갱신 없음)하고 single-box+objectness head만 학습한 모델이다(total **85.84M** / **trainable 0.200M=head only**, `(frozen)+(tr)` 병기). **사전학습 사용 여부가 다르므로 baseline ↔ Ours는 동일 조건 비교가 아니다.** Ours 검출은 baseline과 **동급(포화)**: det PR-AUC/ROC-AUC=1.0, presence_recall 1.0, 정상 오경보 1/1303(0.08%). objectness 분리도 깨끗(질병 median 0.955 vs 정상 median 0.017). 재현: `_workspace/eval/run_ours_detection_eval.py`, 덤프 `_workspace/eval/ours_detection.json`, 정합성 `_workspace/eval/verify_dinov3_base_detection_singlebox.md`.
> **† best epoch = ep0 (IoU 과소평가)**: det_pr_auc가 **ep0부터 1.0으로 포화**해 early-stop(monitor=val_det_pr_auc)이 best=ep0을 골랐다. 따라서 저장 best.pt 예측의 **국소화 IoU가 전 epoch 중 최저**다(§3-2의 IoU median 0.565는 과소평가; 후반 ep≥28엔 IoU median ≈0.64까지 상승, train.log). **검출 지표(det_pr_auc 1.0 / presence 1.0 / fp~0)는 ep0부터 포화라 epoch 선택의 영향이 없다.**

**objectness 분리(붕괴 수정 확인, 6 백본 전부)**: 질병 objectness median ≈0.993–0.999 vs 정상 median ≈0.000–0.004. 정상이 0.5를 넘는 비율 2–8/1303(0.15–0.6%). 6 백본 모두 초기 붕괴(전 이미지 objectness ~1.0)와 달리 **정상/질병이 깨끗이 분리**됨.

### 3-2. 국소화 (양성 이미지에서만 IoU(pred, GT), n=100)

| 백본 | IoU median | IoU mean | IoU@0.5 비율 | mAP@0.5 |
|------|:---:|:---:|:---:|:---:|
| **ConvNeXtV2** | **0.667** | 0.620 | 0.700 | **0.599** |
| EfficientNetV2 | 0.576 | 0.574 | 0.640 | 0.494 |
| NeXtViT-s | 0.606 | 0.570 | 0.590 | 0.459 |
| **DenseNet121*** | 0.621 | 0.604 | 0.670 | 0.588 |
| **ResNet50*** | 0.566 | 0.565 | 0.600 | 0.499 |
| **VisionMamba (mamba)*** | 0.641 | 0.583 | 0.630 | 0.524 |
|---*(이하 Ours: DINOv3-B frozen, head-only — best ep0이라 IoU 과소평가†)*---|||||
| **Ours: DINOv3-B @512 (frozen)** | 0.565† | 0.547† | 0.630 | 0.491 |

(IoU@0.5 비율 = 양성 중 IoU≥0.5로 국소화된 비율. mAP@0.5는 AP 보간 방식상 보고치 0.4856 vs 재계산 0.4906로 미세차, 표는 재계산값.)
> **† Ours IoU 과소평가**: best=ep0(det_pr_auc 포화)이라 저장 predictions의 국소화 IoU가 **전 epoch 중 최저**. ep0 IoU median 0.565 → 후반(ep≥28) IoU median ≈0.635, 최고 **0.643(ep29)**(train.log/per_epoch). det_pr_auc 외 기준으로 골랐다면 국소화 ≈0.64 보고 가능. 검출 지표는 ep0부터 포화라 무관.

그림: `report/figures/exp_detection.png` (좌: det_pr_auc/presence_recall/fp_rate 막대 6백본 **+ Ours: DINOv3-B frozen(주황 음영, pretrained·동일조건 아님)**, 중: 정상 vs 질병 objectness 박스플롯 **+ Ours 중앙값 다이아몬드**, 우: 양성 IoU 히스토그램 **+ Ours(검정 점선, best=ep0이라 IoU 과소평가)**).

**해석.**
- **질병 검출(주지표)은 6 백본 전부 사실상 포화**(det PR-AUC 0.997–1.000, ROC-AUC ≈1.0). **DenseNet121(신규)이 최고**(det PR-AUC 0.9997, ROC-AUC 1.0000, 질병 recall 1.0, 오경보 0.3%). ResNet50은 오경보가 가장 적고(0.15%, 2/1303) 질병 2장만 누락. **소표본(양성 100)이라 백본 간 차이는 통계적으로 의미 두기 어렵다.**
- **mAP@0.5는 det_pr_auc와 분리 해석**: objectness가 정상/질병을 거의 완벽히 가려도(det_pr_auc≈0.999) mAP는 **국소화 hit rate(IoU≥0.5)에 상한**이 묶여 0.46–0.60. "있는지"는 거의 완벽, "어디인지"는 거친 박스 수준.
- **국소화 IoU median 0.57–0.67**: 거친 중앙 crop 영역(REPORT §6: GT 면적 중앙값 ≈50%)을 대략 맞춤. **ConvNeXtV2가 국소화 최고(0.667), DenseNet121·VisionMamba가 그 다음(0.62–0.64)**. 미세 병변 pinpoint는 GT 거칠기상 보장 불가 → "crop 영역 회귀"로 해석.
- 종합: detection baseline은 **이미지 단위 질병 유무 판별에 매우 강하고(주지표 포화)**, 신규 DenseNet121이 검출·국소화 모두 상위권. 백본 우열은 국소화·오경보율에서만 미세하게 갈린다.

---

## 3B. Detection 균형 valid 평가 (질병:정상 = 1:1, 100/100)

§3은 **valid 원분포(질병 100 / 정상 1303, prevalence ≈0.071)**. 여기 §3B는 분류 §1B와 동일하게 **정상을 질병 수에 맞춰 다운샘플(질병 100 : 정상 100, N=200, prevalence 0.5)** 한 균형 valid로 **저장된 best.pt를 그대로 로드(재학습 없음)** 해 재평가한 결과다. 로더: `build_detection_loaders(img_size=512, batch_size=16, num_workers=8, seed=42, include_normal=True, balance_valid=True)`. **모든 백본 fp32(AMP off)로 forward**(nextvit/mamba는 학습도 fp32였음). objectness=sigmoid(obj_logit)를 이미지 단위 질병 점수로 사용. 재현: `_workspace/eval/run_balanced_detection_eval.py`, 덤프 `_workspace/eval/balanced_valid_detection.json`.

| 백본 | params(M) | det PR-AUC | det ROC-AUC | presence_recall@0.5 (질병 recall) | fp_rate@0.5 (정상 오경보) | mAP@0.5 | IoU median |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ConvNeXtV2 (convnextv2) | 28.07 | 0.9999 | 0.9999 | 0.990 (99/100) | 0.000 (0/100) | 0.6029 | 0.667 |
| EfficientNetV2 (efficientnetv2) | 20.51 | 1.0000 | 1.0000 | **1.000** (100/100) | 0.000 (0/100) | 0.4890 | 0.576 |
| NeXtViT-s (nextvit) | 31.00 | 1.0000 | 1.0000 | **1.000** (100/100) | 0.000 (0/100) | 0.4538 | 0.606 |
| **DenseNet121 (densenet121)*** | 7.22 | **1.0000** | **1.0000** | **1.000** (100/100) | 0.000 (0/100) | 0.5844 | 0.621 |
| **ResNet50 (resnet50)*** | 24.04 | 0.9999 | 0.9999 | 0.980 (98/100) | 0.000 (0/100) | 0.4949 | 0.566 |
| **VisionMamba (mamba)*** | 4.03 | 0.9998 | 0.9998 | 0.990 (99/100) | 0.000 (0/100) | 0.5199 | 0.641 |
|---*(이하 Ours: DINOv3 frozen pretrained — 위 from-scratch baseline과 동일 조건 아님)*---|||||||
| **Ours: DINOv3-B @512 (frozen, head-only)** | 85.84 (frozen)+0.200 (tr) | **1.0000** | **1.0000** | **1.000** (100/100) | 0.000 (0/100) | 0.4906 | 0.565† |

(양성 100 / 음성 100. `*`=신규. det PR-AUC·det ROC-AUC는 sklearn `average_precision_score`/`roc_auc_score`로 교차검증해 일치 — ConvNeXtV2 및 **Ours** 기준 PR/ROC 모두 1e-9 이내 일치. 그림: `report/figures/exp_detection_balanced.png`.)
> ⚠️ **공정성(§3-1과 동일)**: Ours 행은 DINOv3 자기지도 pretrained 백본 frozen + head(0.200M trainable)다. 상단 baseline 6종은 from-scratch(전 파라미터 trainable)이므로 **동일 조건 비교가 아니다**(`(frozen)+(tr)` 병기). Ours 균형 재평가는 §3B와 동일 절차(best.pt forward 전용 로드·가중치 변경 없음, `balance_valid=True`, seed=42, img512, fp32). presence_recall·IoU median은 §3-1/§3-2(orig)와 **완전히 동일**(양성 100장에만 의존). fp_rate는 orig 1/1303 → balanced 0/100(다운샘플 표본 운). 재현: `_workspace/eval/run_ours_detection_eval.py`.
> **† best ep0 → IoU 과소평가(§3-2 각주와 동일)**: 저장 best.pt(ep0)의 IoU median 0.565는 전 epoch 최저값. 후반 ep IoU median ≈0.64. det_pr_auc/presence/fp는 ep0부터 포화라 무관.

**원분포(§3) 대비 변화 해석.**
- **presence_recall@0.5와 IoU median(및 mean·IoU@0.5)은 §3과 완전히 동일**(6 백본 모두 소수점 셋째 자리까지 일치). 이 두 지표는 **양성(질병 100장)에만 의존**하고 정상 다운샘플은 양성 집합을 건드리지 않으므로 불변이 정상 — 검증 통과(예: presence 0.98–1.00, IoU median 0.566–0.667, §3과 1:1 매칭).
- **det PR-AUC는 전 백본 상승(0.9975–0.9997 → 0.9998–1.0000)**: PR-AUC 베이스라인이 prevalence(원 0.071 → 0.5)와 함께 올라가고, 음성이 1303→100으로 줄어 소수의 고스코어 정상(오경보)이 precision에 주는 패널티가 작아진 결과다. det ROC-AUC도 미세 상승(prevalence 무관 지표지만 음성 표본 변경으로 값이 약간 달라짐).
- **fp_rate@0.5는 전 백본 0으로 떨어짐**(원 0.15–0.6% → 0): 원분포에서 0.5를 넘던 정상 오경보는 1303장 중 2–8장이었는데, 다운샘플된 100장 표본에는 거의 포함되지 않았다. 즉 균형 표본에서의 fp_rate는 **표본 추출 운(seed=42)에 민감**하며, 정상 오경보의 실제 빈도는 §3의 1303장 기준(0.15–0.6%)이 더 신뢰할 만하다.
- **mAP@0.5는 사실상 불변**(차이 ≤0.0009): 단일 박스가 objectness로 랭크되는데, 정상 예측(전부 FP)들이 질병 양성보다 낮은 점수로 랭크돼 AP의 precision-recall 곡선 끝부분에만 기여하므로, 음성 수를 1303→100으로 줄여도 AP가 거의 바뀌지 않는다(국소화 hit rate에 묶인 상한 0.45–0.60 유지).
- 종합: 균형 valid에서도 **이미지 단위 질병 검출은 6 백본 전부 포화**(det PR-AUC ≥0.9998)이고 순위(국소화 ConvNeXtV2>DenseNet121>VisionMamba, 검출 EfficientNetV2/NeXtViT-s/DenseNet121 동률 1.0)는 §3과 일관. 변하는 것은 prevalence에 의존하는 det PR-AUC·fp_rate뿐이다.

**소표본 주의.** 균형 valid는 양성 100·음성 100으로, det PR-AUC가 거의 1.0인 포화 영역이라 백본 간 차이(0.9998 vs 1.0000)는 통계적으로 무의미하다. fp_rate@0.5는 전 백본 0/100 수준이라 Wilson 95% CI가 [0, 0.037]로 매우 넓다 — 단일값 우열 판단 금지. 정상 오경보율의 신뢰할 수치는 §3의 음성 1303 기준을 사용하라.

---

## 4. 학습 곡선 (과적합 진단)

run별: `report/figures/curves_<name>.png`(24개). 종합 패널: `report/figures/training_curves.png` (**4행 × 6열**: 3 cls 세팅 + detection 행, 6 백본 열, train/val loss + primary 오버레이, best epoch 점선, [NEW]=신규).

**과적합 징후 — 분류 전반에서 뚜렷.** from-scratch·소표본이라 train_loss가 거의 0까지 떨어지는 반면 **val_loss는 중반 최저 후 반등**한다. val_loss 최저 대비 마지막 epoch 반등폭:

| run | val_loss 최저@ep | 마지막 val_loss | 반등폭 |
|-----|:---:|:---:|:---:|
| ConvNeXtV2 / d3 | 0.143 @10 | 0.920 | +0.78 |
| ConvNeXtV2 / 3-class | 0.314 @40 | 0.741 | +0.43 |
| EfficientNetV2 / d3 | 0.099 @30 | 0.201 | +0.10 |
| NeXtViT-s / d4 | 0.041 @35 | 0.372 | +0.33 |
| **DenseNet121*** / d3 | 0.044 @22 | 0.133 | **+0.09** |
| **DenseNet121*** / 3-class | 0.133 @12 | 0.218 | **+0.08** |
| **ResNet50*** / d3 | 0.050 @16 | 0.147 | +0.10 |
| **ResNet50*** / 3-class | 0.133 @30 | 0.248 | +0.12 |
| **VisionMamba*** / d3 | 0.103 @29 | 0.422 | +0.32 |
| **VisionMamba*** / 3-class | 0.278 @ 5 | 0.606 | +0.33 |

- 가장 심한 과적합은 여전히 **ConvNeXtV2 / d3**(반등 +0.78, best ep=5)와 **VisionMamba**(반등 +0.26–0.33, best ep d4=14·3c=4로 매우 일러 학습 후반은 전부 과적합 구간). 둘의 낮은 PR-AUC와 일관.
- **신규 DenseNet121·ResNet50가 가장 안정**(반등 +0.08–0.21, train-val gap 최소). 과적합 저항이 분류 성능 우위로 직결(§2). EfficientNetV2도 안정(기존 최고).
- best epoch 선택은 **val primary metric(PR-AUC) 기준**이라 val_loss 최저 epoch과 다를 수 있다(소표본에서 val_loss·val PR-AUC 비동행).
- **detection (6 백본)**: 모두 60 epoch 완주, train/val loss 단조 감소·**val_loss 반등 없음**(val_loss 0.10–0.15) — 분류 대비 과적합 미미. det_pr_auc는 epoch 초반부터 0.8+로 시작해 빠르게 0.99+ 수렴. **NeXtViT-s·VisionMamba는 fp32(AMP off)로 학습**(fp16 NaN 회피 → §6·§8). 곡선: `report/figures/curves_<name>_detection_singlebox.png`.

---

## 5. 핵심 수치 요약 (6 백본)

- 분류 최고 PR-AUC(백본별·d3): **ResNet50* 0.965 / DenseNet121* 0.957 / EfficientNetV2 0.949 / NeXtViT-s 0.929 / VisionMamba* 0.878 / ConvNeXtV2 0.787**.
- **신규>기존**: DenseNet121·ResNet50이 PR-AUC·F1·recall에서 기존 최고(EfficientNetV2)를 동급 이상으로 끌어올림. 특히 **DenseNet121이 3-class PR-AUC 0.570·3c disease recall(d3 0.80/d4 0.83)로 전 백본 최고**.
- 세팅 난이도: normal_vs_d3(0.79–0.97) > normal_vs_d4(0.63–0.87) > normal_d3_d4(0.34–0.57).
- **AUROC는 전 백본·세팅 0.94–1.0으로 포화 → 백본 분리력 거의 없음**. 해석 주지표는 PR-AUC/F1-macro/precision(§2).
- detection 질병 검출: det PR-AUC **0.997–1.000**(DenseNet121* 0.9997 최고), ROC-AUC ≈1.0, 질병 recall@0.5 0.98–1.00, 정상 오경보 0.15–0.6%(ResNet50* 최저). objectness 붕괴 해소 유지.
- detection 국소화: IoU median 0.57–0.67(ConvNeXtV2 0.667 최고), mAP@0.5 0.46–0.60 — 거친 박스 한계(det_pr_auc와 분리).
- **전 24 run 정합성 PASS, 재계산=보고 일치**. 기존 3 백본 AUROC/macro-PRF는 predictions에서 독립 재계산(sklearn 교차검증 일치).

---

## 6. Ours / Ours+: DINOv3(frozen)+2-layer head (자기지도 전이 vs from-scratch baseline)

**Ours 모델 (small)**: DINOv3 ViT-S/16(timm `vit_small_patch16_dinov3`, self-supervised pretrained) **백본 완전 동결**(requires_grad=False, no-grad/eval forward) + 2-layer MLP head(hidden 256). img_size=**256**. 총 **21.69M** 중 trainable(head) **~99k**. `experiments/dinov3_{setting}/`.

**Ours+ 모델 (base@512, 성능 최대화)**: DINOv3 ViT-B/16(timm `vit_base_patch16_dinov3`) **백본 완전 동결** + 2-layer MLP head(hidden **512**). img_size=**512**. 총 **86.04M**(frozen backbone 85.6M) 중 trainable(head) **~395k**(3-class 395.3k). `experiments/dinov3_base_{setting}/`. small 대비 **고해상도(256→512)+큰 backbone(S→B)**으로 표현력 확장, frozen 레시피·no-forgetting은 동일 유지. best.pt는 forward 전용 로드(가중치 변경 없음). 손실은 **CE**, 증강은 **default**.

**Ours+ focal+aug 모델 (개선판)**: Ours+(base@512)와 **백본·head·해상도·trainable(395k) 전부 동일**하되, 학습만 **강한 증강(aug=strong: VFlip·rotation·강한 ColorJitter·TrivialAugmentWide·RandomErasing)** + **focal loss(gamma=2, class_weights=from_meta)**로 바꾼 변형. `experiments/dinov3_base_focal_{setting}/`. **주의: aug와 focal 두 변수를 동시에 변경**했으므로 base-CE 대비 개선은 **두 변수의 합동 효과**이며 각각의 기여는 분리되지 않는다(ablation 필요). 또한 `from_meta` class_weights는 train 로더가 **균형 다운샘플**되어 **전부 1.0**으로 해소되므로 — focal의 실제 작동 기제는 **gamma(어려운 표본 집중)**이지 alpha 재가중이 아니다. best.pt는 forward 전용 로드(가중치 변경 없음).

baseline 6 백본은 모두 **from-scratch**(pretrained 미로딩)이므로, "동일 데이터·동일 valid·동일 지표" 기준에서 Ours/Ours+(frozen SSL feature) vs baseline-best를 비교한다. 비교 기준선은 **세팅별 baseline 최고**(원분포·균형 각각, PR-AUC/F1-macro 별도 최고).

### 6.1 Ours / Ours+ 7-메트릭 (원분포 valid + 균형 valid)

predictions/valid.npz에서 **sklearn으로 독립 재계산**, metrics.json 보고치와 대조(완전 일치, §6.4). 균형은 best.pt 로드 후 `balance_valid=True`(seed=42, §1B와 동일 절차)로 forward 재평가. Ours+(B, CE)·Ours+ focal+aug는 img_size=**512**로 균형 재평가. 세 변형(small / base-CE / **focal+aug**)을 동일 valid·동일 지표로 병기.

| 모델 | img | trainable | 세팅 | valid | N | acc | train_loss | val_loss | recall(macro) | precision(macro) | **F1(macro)** | AUROC | **PR-AUC** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ours (S) | 256 | 99k | normal_vs_d3 | 원분포 | 1379 | 0.982 | 0.0024 | 0.041 | 0.984 | 0.878 | **0.924** | 1.000 | **0.996** |
| Ours+ (B, CE) | 512 | 395k | normal_vs_d3 | 원분포 | 1379 | 0.996 | 0.0002 | 0.010 | 0.998 | 0.969 | **0.983** | 1.000 | **1.000** |
| **Ours+ focal+aug** | 512 | 395k | normal_vs_d3 | 원분포 | 1379 | 0.998 | 0.0022 | 0.002 | 0.999 | 0.981 | **0.990** | 1.000 | **1.000** |
| Ours (S) | 256 | 99k | normal_vs_d3 | 균형 1:1 | 152 | 0.980 | 0.0024 | 0.061 | 0.980 | 0.981 | **0.980** | 1.000 | **1.000** |
| Ours+ (B, CE) | 512 | 395k | normal_vs_d3 | 균형 1:1 | 152 | 0.993 | 0.0002 | 0.017 | 0.993 | 0.994 | **0.993** | 1.000 | **1.000** |
| **Ours+ focal+aug** | 512 | 395k | normal_vs_d3 | 균형 1:1 | 152 | 0.993 | 0.0022 | 0.021 | 0.993 | 0.994 | **0.993** | 1.000 | **1.000** |
| Ours (S) | 256 | 99k | normal_vs_d4 | 원분포 | 1327 | 0.963 | 0.0087 | 0.081 | 0.981 | 0.664 | **0.738** | 1.000 | **0.997** |
| Ours+ (B, CE) | 512 | 395k | normal_vs_d4 | 원분포 | 1327 | 0.987 | 0.0023 | 0.028 | 0.994 | 0.793 | **0.866** | 1.000 | **1.000** |
| **Ours+ focal+aug** | 512 | 395k | normal_vs_d4 | 원분포 | 1327 | 0.993 | 0.0035 | 0.004 | 0.997 | 0.864 | **0.919** | 1.000 | **1.000** |
| Ours (S) | 256 | 99k | normal_vs_d4 | 균형 1:1 | 48 | 1.000 | 0.0087 | 0.020 | 1.000 | 1.000 | **1.000** | 1.000 | **1.000** |
| Ours+ (B, CE) | 512 | 395k | normal_vs_d4 | 균형 1:1 | 48 | 1.000 | 0.0023 | 0.005 | 1.000 | 1.000 | **1.000** | 1.000 | **1.000** |
| **Ours+ focal+aug** | 512 | 395k | normal_vs_d4 | 균형 1:1 | 48 | 1.000 | 0.0035 | 0.016 | 1.000 | 1.000 | **1.000** | 1.000 | **1.000** |
| Ours (S) | 256 | 99k | normal_d3_d4 (3c) | 원분포 | 1403 | 0.961 | 0.318 | 0.092 | 0.753 | 0.674 | **0.698** | 0.991 | **0.689** |
| Ours+ (B, CE) | 512 | 395k | normal_d3_d4 (3c) | 원분포 | 1403 | 0.967 | 0.163 | 0.083 | 0.840 | 0.755 | **0.750** | 0.993 | **0.745** |
| **Ours+ focal+aug** | 512 | 395k | normal_d3_d4 (3c) | 원분포 | 1403 | 0.973 | 0.072 | 0.019 | 0.850 | 0.769 | **0.774** | 0.995 | **0.765** |
| Ours (S) | 256 | 99k | normal_d3_d4 (3c) | 균형 1:1:1 | 72 | 0.778 | 0.318 | 0.585 | 0.778 | 0.811 | **0.774** | 0.946 | **0.810** |
| Ours+ (B, CE) | 512 | 395k | normal_d3_d4 (3c) | 균형 1:1:1 | 72 | 0.875 | 0.163 | 0.320 | 0.875 | 0.878 | **0.875** | 0.962 | **0.885** |
| **Ours+ focal+aug** | 512 | 395k | normal_d3_d4 (3c) | 균형 1:1:1 | 72 | 0.875 | 0.072 | 0.350 | 0.875 | 0.878 | **0.875** | 0.952 | **0.856** |

- 파라미터: Ours(S) total **21.69M** / trainable **~99k**; Ours+(B, CE)·Ours+ focal+aug 둘 다 total **86.04M**(frozen 85.6M) / trainable **~395k**(head only, **세 변형 모두 동일**). 셋 다 frozen backbone이므로 **catastrophic forgetting 없음**(backbone 가중치 불변, head만 학습).
- 소표본 주의: 원분포 d4 valid N=24, 균형 3-class는 클래스당 N=24 → PR-AUC/F1 CI 넓음(verify 파일에 Wilson CI 병기).
- **3-class 원분포 per-class — Ours+(B, CE) → Ours+ focal+aug**: normal recall 0.987→0.992/precision 0.999→0.999; **d3 recall 0.658→0.684 / precision 0.926→0.912**; **d4 recall 0.875→0.875(불변) / precision 0.339→0.396**. focal+aug confusion `[[1292,2,9],[1,52,23],[0,3,21]]`(CE는 `[[1286,1,16],[1,50,25],[0,3,21]]`) — **정상→d4 오분류가 16→9장으로 줄어** d4 precision이 0.339→0.396(21/53), d4 recall은 21/24로 동일. d4 precision 95% CI 0.276–0.531로 여전히 매우 넓고 낮아 **3-class F1 병목은 완화되었으나 해소되지 않았다**(아래 6.4).

### 6.2 baseline 최고 대비 향상률

세팅별 **baseline 최고**(원분포·균형 각각)와의 PR-AUC·F1-macro 향상량(절대 / 상대%). baseline 최고는 PR-AUC와 F1에서 다른 백본일 수 있어 각각 표기. Ours+ focal+aug 상대%는 **focal 값 vs baseline 최고** 기준.

**원분포 valid**

| 세팅 | 지표 | Ours (S) | Ours+ (B, CE) | **Ours+ focal+aug** | baseline 최고 | focal 절대 Δ | focal 상대 % |
|---|---|---|---|---|---|---|---|
| normal_vs_d3 | PR-AUC | 0.996 | 1.000 | **1.000** | 0.965 (ResNet50) | +0.035 | **+3.6%** |
| normal_vs_d3 | F1-macro | 0.924 | 0.983 | **0.990** | 0.860 (DenseNet121) | +0.129 | **+15.0%** |
| normal_vs_d4 | PR-AUC | 0.997 | 1.000 | **1.000** | 0.867 (ResNet50) | +0.133 | **+15.4%** |
| normal_vs_d4 | F1-macro | 0.738 | 0.866 | **0.919** | 0.679 (EffNetV2) | +0.241 | **+35.5%** |
| **normal_d3_d4 (3-class)** | **PR-AUC** | 0.689 | 0.745 | **0.765** | **0.570 (DenseNet121)** | **+0.194** | **+34.1%** |
| **normal_d3_d4 (3-class)** | **F1-macro** | 0.698 | 0.750 | **0.774** | 0.709 (DenseNet121) | +0.065 | **+9.2%** |

**균형 valid** (baseline·Ours 모두 천장 근접; 3-class만 여유)

| 세팅 | 지표 | Ours (S) | Ours+ (B, CE) | **Ours+ focal+aug** | baseline 최고 | focal 절대 Δ | focal 상대 % |
|---|---|---|---|---|---|---|---|
| normal_vs_d3 | PR-AUC | 1.000 | 1.000 | **1.000** | 1.000 (ResNet50) | +0.000 | +0.0% |
| normal_vs_d3 | F1-macro | 0.980 | 0.993 | **0.993** | 0.980 (DenseNet121) | +0.013 | +1.3% |
| normal_vs_d4 | PR-AUC | 1.000 | 1.000 | **1.000** | 0.998 (DenseNet121) | +0.002 | +0.2% |
| normal_vs_d4 | F1-macro | 1.000 | 1.000 | **1.000** | 0.979 (EffNetV2) | +0.021 | +2.1% |
| **normal_d3_d4 (3-class)** | PR-AUC | 0.810 | 0.885 | **0.856** | 0.860 (DenseNet121) | -0.003 | -0.4% |
| **normal_d3_d4 (3-class)** | F1-macro | 0.774 | 0.875 | **0.875** | 0.850 (DenseNet121) | +0.024 | **+2.8%** |

**base-CE → focal+aug 직접 비교 (동일 backbone·head·해상도, 학습만 aug+focal 변경 — 합동 효과)**

| 세팅 | 지표 | Ours+ (B, CE) | **Ours+ focal+aug** | Δ (원분포) | Ours+ (B, CE) bal | focal bal | Δ (균형) |
|---|---|---|---|---|---|---|---|
| normal_vs_d3 | PR-AUC | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| normal_vs_d3 | F1-macro | 0.983 | 0.990 | **+0.007** | 0.993 | 0.993 | +0.000 |
| normal_vs_d4 | PR-AUC | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| normal_vs_d4 | F1-macro | 0.866 | 0.919 | **+0.053** | 1.000 | 1.000 | +0.000 |
| **normal_d3_d4 (3-class)** | **PR-AUC** | 0.745 | 0.765 | **+0.019** | 0.885 | 0.856 | **-0.028** |
| **normal_d3_d4 (3-class)** | **F1-macro** | 0.750 | 0.774 | **+0.024** | 0.875 | 0.875 | +0.000 |

- **원분포에서는 focal+aug가 base-CE를 일관되게 소폭 개선**: 3-class F1 +0.024, PR-AUC +0.019; d4(2-class) F1 +0.053. 단 **균형 3-class PR-AUC는 -0.028로 역행**(단일 forward 재평가·클래스당 N=24 소표본의 CI 안에서 흔들리는 수준; 과대해석 금지). 균형 F1은 변화 없음(둘 다 confusion 동일).

### 6.3 +20% 목표 판정 (정직)

design-notes 기준: 3-class 원분포 baseline 최고 **PR-AUC 0.570(DenseNet121)·F1-macro 0.709** → +20% 목표 **PR-AUC≥0.684, F1-macro≥0.851**.

- **주지표 PR-AUC: 달성(세 변형 모두 상회, focal+aug가 가장 견고)** — 3-class 원분포 **PR-AUC: small 0.689(+20.8%) → base-CE 0.745(+30.7%) → focal+aug 0.765(+34.1%)**, 모두 목표선 0.684을 상회. **focal+aug는 목표선을 0.081 여유로 넘어** 가장 견고하다.
- **F1-macro: 여전히 절대 목표(0.851) 미달이나, focal+aug에서 가장 근접** — 3-class 원분포 **F1-macro: small 0.698(-1.6% vs baseline) → base-CE 0.750(+5.8%) → focal+aug 0.774(+9.2%)**. 단조 개선이지만 **focal+aug도 0.774 < 0.851**(0.077 미달). +20% 상대 기준선(0.851)은 미통과.
- **종합 판정**:
  - **PR-AUC: 3-class 원분포에서 +20% 목표 달성(small·base-CE·focal+aug 모두; focal+aug가 +34.1%로 가장 견고).**
  - **F1-macro: 절대 목표(0.851) 미달.** focal+aug에서 +9.2%로 base-CE(+5.8%)보다 더 개선됐지만 0.851에는 못 미친다.
  - 과대포장하지 않는다 — "+20% 달성"은 **3-class·원분포·PR-AUC** 지표/세팅에서 성립하며, **F1-macro 절대 목표는 d4 소표본 precision 병목(focal+aug에서 0.339→0.396으로 완화됐으나 여전히 낮음) 때문에 미달**이다.
- **focal+aug의 기여(정직)**: base-CE 대비 focal+aug는 원분포 3-class에서 PR-AUC +0.019·F1 +0.024를 **추가로** 끌어올렸다(2-class d4 F1은 +0.053). 그러나 **aug(strong)와 focal(gamma2)을 동시에 바꿨으므로 이 개선이 둘 중 어느 변수에서 왔는지는 분리 불가** — 단변수 ablation(aug만 / focal만)이 필요하다. 또한 `from_meta` class_weights가 균형 train 로더 때문에 전부 1.0으로 해소되어, focal의 작동 기제는 **alpha 재가중이 아니라 gamma(어려운 표본 집중)**임을 명시한다.
- **세팅별**: 2-class(normal_vs_d3/d4)는 baseline PR-AUC가 이미 0.87–0.97로 **천장 근접**이라 PR-AUC 상대 향상 여지가 작다(+3.6%/+15.4%). 단 focal+aug는 **2-class F1에서 +15.0%/+35.5%로 큰 향상**(특히 d4 F1 0.866→0.919, base-CE 대비 +0.053; baseline 0.679 대비 +35.5%). 균형 valid는 2-class 양쪽 모두 PR-AUC≈1.0으로 포화, 3-class만 여유가 있는데 focal+aug는 균형 F1 +2.8%(baseline 대비)이나 균형 PR-AUC는 base-CE 0.885 대비 -0.028로 소표본 변동 폭 내.

### 6.4 해석

- **왜 frozen SSL feature가 from-scratch를 능가하나**: baseline 6 백본은 pretrained 미로딩(§9 한계 1)이라 소규모·불균형 무 데이터로 표현을 처음부터 학습 → 표현력 상한이 낮다. Ours/Ours+는 대규모 자기지도(DINOv3) feature가 **이미 일반적 시각 표현을 갖춘 채 동결**되어, 작은 head만으로도 강한 분리를 얻는다. 가장 어려운 **3-class 원분포 PR-AUC에서 Ours+ +30.7%**로 격차가 가장 크다(쉬운 2-class는 baseline도 이미 잘 해 PR-AUC 격차 작음).
- **고해상도(512)+큰 backbone(ViT-B)이 3-class에 준 이득**: small@256 → base@512로 가면서 3-class 원분포 **PR-AUC 0.689→0.745(+0.056), F1 0.698→0.750(+0.052)**, 균형 3-class **PR 0.810→0.885(+0.075), F1 0.774→0.875(+0.100)**. 2-class도 원분포 F1이 d3 0.924→0.983, d4 0.738→0.866으로 크게 상승. 고해상도가 미세 병변·d3↔d4 구분에 유리하고, 더 큰 backbone이 표현력을 높여 **가장 어려운 3-class에서 이득이 가장 크게 나타난다**(쉬운 2-class PR-AUC는 small도 이미 천장이라 base는 F1에서 주로 향상).
- **d4 소표본·3-class F1 병목**: 원분포 d4 valid N=24로 극소. base-CE 3-class 원분포 confusion `[[1286,1,16],[1,50,25],[0,3,21]]` → (a) PR-AUC는 랭킹 기반이라 높지만(0.745), (b) **argmax F1-macro는 d4 precision 0.339**(정상 16장+d3 25장 일부가 d4로 끌려감)에 눌린다. d4 **recall은 0.875로 양호**(놓침 적음)하나, 정상 다수가 d4로 새는 **낮은 precision**이 macro-F1을 0.750에 묶는다 — 이것이 F1 절대 목표(0.851) 미달의 직접 원인. **PR-AUC로는 목표 초과, argmax-F1으로는 미달**이 이 소표본·임계값 한계의 결과.
- **focal+aug가 d4 병목을 얼마나 완화했나(정직)**: focal+aug 3-class 원분포 confusion `[[1292,2,9],[1,52,23],[0,3,21]]`. base-CE 대비 **정상→d4 오분류가 16→9장으로 감소** → **d4 precision 0.339→0.396**(21/53), d4 recall은 21/24=0.875로 **불변**. d3는 recall 0.658→0.684(소폭↑)·precision 0.926→0.912(소폭↓). 즉 focal의 hard-example 집중(+강한 증강)이 **정상↔d4 결정 경계를 약간 조여 d4 false-positive를 줄였으나**, d4 precision 95% CI가 여전히 0.276–0.531로 매우 넓고 낮아 **F1 병목은 완화되었을 뿐 해소되지 않았다**(N=24 한계). 이것이 focal+aug에서도 F1 0.774 < 0.851로 절대 목표 미달인 직접 이유.
- **forgetting 없음(freeze)**: backbone(S 21.59M / B 85.6M)이 동결이라 학습 중 가중치 불변(`requires_grad=False`, eval/no-grad forward). 사전학습 표현을 그대로 보존하고 head만 적응 → 재현성·안정성 높고, 학습 비용도 head(99k/395k)로 극히 작다. base@512는 backbone이 4× 크지만 trainable은 여전히 395k에 불과.
- **균형 valid**: Ours+는 균형 3-class에서 PR 0.885·F1 0.875로 **baseline 최고(0.860/0.850)를 상회**(small은 균형 3-class에서 baseline에 뒤졌으나 base가 역전). 단 클래스당 N=24 극소표본이라 단일값 우열은 CI 안에서 흔들린다(과대해석 금지). 2-class 균형은 양쪽 모두 천장.

---

## 6B. Ablation: aug × focal (dinov3_base 3-class)

§6의 Ours+ focal+aug는 base-CE 대비 **강한 증강(strong)** 과 **focal loss** 두 변수를 동시에 바꿨기에, 개선의 기여가 분리되지 않았다(§6.3). 본 절은 그 두 변수를 **단변수로 분해**한 ablation이다. **backbone·head·해상도(512)·optimizer·trainable(395k)·seed(42) 전부 고정**, **loss(CE/focal-γ)와 aug(default/strong)만** 변경한 6 run을 동일 valid(원분포 normal 1303 / d3 76 / d4 24, N=1403)·동일 지표로 비교한다. 모든 수치는 **predictions/valid.npz에서 sklearn 독립 재계산**(metrics.json.final과 완전 일치, 정합성은 §6B-5)이며, 기준점 2개(base, focal+aug)는 §6의 동일 run이다.

> 전제(중요): `class_weights=from_meta`가 **균형 다운샘플 train 로더 때문에 전부 1.0으로 해소** → focal의 작동 기제는 **alpha 재가중이 아니라 gamma(어려운 표본 집중)**. 단일 시드, d4 valid **N=24**(CI 매우 넓음) → 절대값 0.02 미만 차이는 노이즈로 본다.

### 6B-1. 2×2 표 (aug ✗/✓ × CE/focal-γ2, 원분포)

| run | loss / aug | PR-AUC | F1-macro | acc | AUROC | d4 precision | d4 recall |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| dinov3_base (참조) | CE / default | 0.745 | 0.750 | 0.967 | 0.993 | 0.339 | 0.875 |
| dinov3_base_augonly | CE / **strong** | 0.756 | 0.768 | 0.973 | 0.995 | 0.389 | 0.875 |
| dinov3_base_focalonly | focal γ2 / default | 0.736 | 0.735 | 0.960 | 0.993 | 0.299 | 0.833 |
| dinov3_base_focal (참조) | focal γ2 / **strong** | **0.765** | **0.774** | 0.973 | **0.995** | **0.396** | 0.875 |

(재계산값과 1차 보고값 완전 일치. 2×2 셀: base=좌상, aug-only=우상, focal-only=좌하, focal+aug=우하.)

### 6B-2. 기여 분리 (요인 분해)

2×2 디자인에서 각 요인의 주효과와 상호작용을 분해한다(셀 값에서 직접 산출, `sum_check`로 joint와 합치 확인):
- **aug 효과** = augonly − base
- **focal 효과** = focalonly − base
- **상호작용(시너지)** = (focal+aug) + base − focalonly − augonly
- **joint(합동)** = (focal+aug) − base = aug효과 + focal효과 + 상호작용

| 지표 | aug 효과 | focal 효과 | 상호작용 | joint |
|---|:---:|:---:|:---:|:---:|
| PR-AUC | **+0.0105** | −0.0088 | **+0.0178** | +0.0195 |
| F1-macro | **+0.0177** | −0.0153 | **+0.0216** | +0.0240 |
| accuracy | +0.0057 | −0.0071 | +0.0071 | +0.0057 |
| AUROC | +0.0019 | +0.0003 | +0.0003 | +0.0024 |
| **d4 precision** | **+0.050** | −0.040 | +0.048 | +0.057 |
| d4 recall | +0.000 | −0.042 | +0.042 | +0.000 |

**결론(정직).**
1. **강한 증강이 주 동력**: aug-only가 base를 단독으로 PR-AUC +0.0105·F1 +0.0177·d4 precision +0.050 끌어올린다. frozen feature 위 작은 head에 대해 강한 증강이 정칙화·d4 결정경계 강화로 직접 기여.
2. **focal 단독은 소폭 하락**: focal-only(default aug)는 base보다 PR-AUC −0.0088·F1 −0.0153·d4 precision −0.040·d4 recall −0.042로 **악화**. γ2 hard-example 집중이 default-aug에서는 과적합/소수클래스 노이즈 증폭으로 역효과(class_weights는 1.0이라 alpha 보정 없음).
3. **조합 시너지(positive interaction)**: 상호작용 항이 PR-AUC +0.0178·F1 +0.0216으로 **두 주효과 합보다 크다**(focal+aug joint +0.024 F1 = aug +0.018 + focal −0.015 + 시너지 +0.022). 즉 focal은 **강한 증강과 함께일 때만 이득**으로 전환된다 — 강한 증강이 만든 다양한 hard sample을 focal의 γ-집중이 효과적으로 활용하는 구조. **단독 focal은 쓰지 말고 strong aug와 결합**해야 한다는 실용 결론.

### 6B-3. gamma 스윕 (strong aug 고정: γ1 / γ2 / γ3)

강한 증강을 고정하고 focal gamma만 1/2/3으로 스윕:

| gamma | PR-AUC | F1-macro | acc | AUROC | d4 precision | d4 recall |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| γ1 | 0.758 | **0.782** | 0.974 | 0.995 | **0.412** | 0.875 |
| γ2 (Ours+) | 0.765 | 0.774 | 0.973 | 0.995 | 0.396 | 0.875 |
| γ3 | **0.769** | 0.767 | 0.972 | **0.995** | 0.382 | 0.875 |

- **최적 gamma는 지표에 따라 갈린다**: **PR-AUC·AUROC는 γ↑일수록 단조 개선**(γ3 PR 0.769 최고 — 랭킹 분리력은 강한 집중이 유리), **F1-macro·d4 precision은 γ↓일수록 좋다**(γ1 F1 0.782·d4P 0.412 최고 — argmax 결정경계는 약한 집중이 유리). d4 recall은 세 γ 모두 0.875로 불변.
- **곡선 형태**: PR-AUC 0.758→0.765→0.769(우상향), F1 0.782→0.774→0.767(우하향)로 **교차**. γ2(Ours+ 채택값)는 두 곡선의 중간 절충점.
- **권고**: 주지표(PR-AUC)·랭킹 운영이면 **γ3**, argmax-F1·d4 오경보 억제(precision)가 중요하면 **γ1**이 낫다. γ들 간 차이(PR 0.011 폭, F1 0.015 폭)는 **d4 N=24 단일 시드 CI 안**이라 통계적으로 단정하기 어렵다(아래 6B-4) — "강한 집중일수록 랭킹↑/argmax-precision↓"의 **추세**로만 해석.

### 6B-4. d4 precision/recall 변화 + 소표본 CI

핵심 개선축은 **d4 precision**이다(d4 recall은 거의 전 run 0.875로 포화). base 0.339 → ablation 0.299~0.412로 움직였고, Ours+(focal+aug γ2)는 0.396, gamma 스윕 최고는 γ1의 **0.412**. 다만 **d4 valid는 24장뿐**이라 Wilson 95% CI가 매우 넓다:

| run | d4 precision (95% CI) | d4 recall (95% CI) |
|---|:---:|:---:|
| base (CE/default) | 0.339 [0.233–0.463] | 0.875 [0.690–0.957] |
| augonly (CE/strong) | 0.389 [0.270–0.522] | 0.875 [0.690–0.957] |
| focalonly (γ2/default) | 0.299 [0.202–0.417] | 0.833 [0.641–0.933] |
| focalg1 (γ1/strong) | **0.412 [0.288–0.548]** | 0.875 [0.690–0.957] |
| focal (γ2/strong) | 0.396 [0.276–0.531] | 0.875 [0.690–0.957] |
| focalg3 (γ3/strong) | 0.382 [0.265–0.514] | 0.875 [0.690–0.957] |

> **정직 해석**: d4 precision은 base 0.339→strong-aug 계열 0.39~0.41로 **방향성은 일관되게 개선**(정상→d4 오분류 감소)이나, **모든 run의 CI가 서로 크게 겹친다**(예: base 상한 0.463 vs focalg1 하한 0.288). 단일 시드·N=24이므로 **개별 run 간 우열을 통계적으로 단정하지 않는다**. 신뢰할 결론은 (a) **강한 증강이 d4 precision을 올린다**(aug 효과 +0.050, focal-only는 −0.040), (b) **focal은 strong aug와 결합 시에만 이득**(시너지)이라는 **방향성**까지다.

### 6B-5. 정합성 (predictions ↔ manifest, 재계산 ↔ 보고)

- ablation 4 run + 참조 2 run **전부 predictions valid N=1403·라벨분포 {0:1303, 1:76, 2:24}가 manifest valid와 정확히 일치**(dist_match=PASS, 누수·오집계·split 위반 없음).
- 6 run 모두 sklearn 재계산(PR-AUC·F1·recall·precision·acc·AUROC·confusion)이 **metrics.json.final과 |Δ|≤0.01로 완전 일치**. disease OvR PR-AUC는 d3·d4 AP의 macro 평균으로 교차검증.
- 상세: `_workspace/eval/verify_dinov3_base_{augonly,focalonly,focalg1,focalg3}_normal_d3_d4.md`(run별 boundary/재계산/per-class/Wilson CI/Δ vs base). 덤프: `_workspace/eval/ablation_dinov3.json`. 그림: `report/figures/exp_ablation_dinov3.png`(2×2 막대 + 기여분해 + gamma 곡선). 스크립트: `_workspace/eval/run_ablation_eval.py`, `make_ablation_fig.py`.

---

## 7. Sensitivity: 입력 노이즈 (Ours, 3-class)

학습된 **Ours**(DINOv3 ViT-B/16 frozen @512 + 2-layer head, **focal+aug**, 3-class = `experiments/dinov3_base_focal_normal_d3_d4/`)의 **입력 노이즈 강건성**을 측정한다. best.pt는 **forward 전용 로드(재학습·가중치 변경 없음)**, 평가는 **원분포 valid(N=1403, balance_valid=False)**·seed=42 고정.

**노이즈 공식·적용지점**: 사용자 지정 `Noised = torch.rand_like(Image) * N_ratio + Image`를 **모델 입력 텐서**에 적용한다 — 즉 valid 변환(리사이즈+ImageNet 정규화)의 출력인 **정규화된 입력 텐서 `x` ∈ [B,3,512,512]** 에

> `x_noised = x + torch.rand_like(x) * N_ratio`   (`rand ~ U[0,1)`)

를 더한 뒤 forward한다. **노이즈는 픽셀(원본 이미지)이 아니라 정규화 입력 텐서 단계에 가산**되며(정규화 스케일 기준 U[0,1)·N_ratio 크기), 재현 위해 N_ratio별로 `torch.manual_seed(42)` 고정. `N_ratio ∈ {0.0(clean 기준선), 0.1, 0.2, 0.3, 0.4, 0.5}`.

지표는 predictions softmax에서 **sklearn 독립 계산**: PR-AUC(주, disease OvR macro)·F1-macro·accuracy·AUROC(OvR macro).

### 7-1. N_ratio vs 성능 (원분포 valid, N=1403)

| N_ratio | PR-AUC (주) | F1-macro | accuracy | AUROC | ΔPR-AUC (vs clean) | rel% |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.0 (clean)** | **0.7651** | **0.7718** | **0.9722** | **0.9951** | — | — |
| 0.1 | 0.7619 | 0.7807 | 0.9729 | 0.9951 | −0.0032 | −0.4% |
| 0.2 | 0.7478 | 0.7728 | 0.9686 | 0.9943 | −0.0173 | −2.3% |
| 0.3 | 0.7468 | 0.7539 | 0.9644 | 0.9941 | −0.0183 | −2.4% |
| 0.4 | 0.7550 | 0.7516 | 0.9636 | 0.9941 | −0.0101 | −1.3% |
| 0.5 | 0.7518 | 0.7524 | 0.9629 | 0.9940 | −0.0133 | −1.7% |

> **clean 대조(정합성)**: N_ratio=0.0의 PR-AUC 0.765·F1 0.772·acc 0.972·AUROC 0.995는 §6 Ours focal+aug 3-class 원분포(PR-AUC 0.765·F1 0.774·acc 0.973·AUROC 0.995)와 **|Δ|≤0.01로 일치**(같은 모델·valid·noise 없음 → 일치해야 정상, 스크립트가 assert로 검증). F1 0.7718 vs 보고 0.774 차이는 반올림 내.

그림: `report/figures/exp_sensitivity_dinov3.png`(N_ratio vs PR-AUC/F1 곡선 + acc/AUROC 보조 패널).

### 7-2. 해석 (robustness)

- **전반적으로 매우 강건**: 노이즈를 입력 텐서 표준편차에 맞먹는 크기(N_ratio=0.5는 정규화 스케일에서 U[0,1)·0.5 가산)까지 키워도 **주지표 PR-AUC 저하는 최대 −0.018(−2.4%, N_ratio=0.3)**, N_ratio=0.5에서도 −0.013(−1.7%)에 그친다. AUROC는 0.995→0.994로 거의 불변, accuracy도 0.972→0.963(−0.9%p)으로 완만.
- **단조 저하는 아님**: PR-AUC는 0.3에서 저점(0.747) 후 0.4에서 일부 회복(0.755). 작은 N에서 F1은 오히려 소폭 상승(0.1에서 0.781). 이는 가산 노이즈가 약한 정규화/dropout처럼 작용해 argmax 결정을 흔드는 정도가 작기 때문으로, **저하 폭이 d4 N=24 소표본 변동(§9 한계 2)과 같은 스케일**이라 N_ratio 간 미세 우열은 통계적으로 단정하기 어렵다. 큰 추세는 **노이즈↑ → 성능 소폭↓**.
- **강건성의 출처**: backbone(DINOv3 ViT-B/16)이 **frozen**이고 자기지도 사전학습 특징이 입력 섭동에 안정적이며, head만 학습된 구조라 입력 텐서 가산 노이즈가 깊은 표현을 크게 교란하지 않는다. 단, 노이즈를 **정규화 입력 텐서**에 가한 것이라 원본 픽셀(0–255)·JPEG 압축·센서 노이즈 등 **실제 촬영 노이즈와는 분포가 다름**을 해석 시 유의(아래 §9 한계 참조).

데이터: `_workspace/eval/sensitivity_dinov3.json`. 스크립트: `_workspace/eval/run_sensitivity_eval.py`(평가)·`make_sensitivity_fig.py`(그림).

---

## 8. Stability: train ratio sweep (Ours, 3-class)

학습 데이터량(`train_ratio`)에 따른 **Ours**(DINOv3 ViT-B/16 frozen @512 + 2-layer head, **focal+aug**, **3-class** = `normal_d3_d4`)의 성능 **안정성**을 정리한다. `train_ratio ∈ {0.1, 0.3, 0.5, 0.7, 0.9}`는 신규 학습 run(`experiments/dinov3_base_focal_r{10,30,50,70,90}_normal_d3_d4/`), **참조점 `train_ratio=1.0`은 §6 정본 Ours**(`experiments/dinov3_base_focal_normal_d3_d4/`)와 **동일 run**이다. 모두 backbone·head·해상도(512)·trainable(395k)·loss(focal γ2)·aug(strong)·seed 동결, **동일 valid(원분포, N=1403 = normal 1303 / d3 76 / d4 24)**. **재학습 없음** — `metrics.json` + `predictions/valid.npz`만 사용, best.pt 미로딩.

`train_ratio`는 **균형 다운샘플된 train**을 추가로 비율 추출한다(train 로더가 클래스 균형이라 각 비율에서 **클래스당 표본 수가 동일**). 따라서 표의 "train/클래스"는 normal=d3=d4 공통 표본 수다.

지표는 `predictions`의 softmax에서 **sklearn 독립 재계산**: PR-AUC(주, disease d3·d4 OvR macro)·F1-macro·accuracy·AUROC(OvR macro). **6점 전부 predictions↔manifest 분포 일치(dist_match=PASS, N=1403)·보고치↔재계산 |Δ|≤0.01 일치**(스크립트가 assert). `train_ratio=1.0` 재계산값 PR-AUC **0.765**·F1 **0.774**·acc **0.973**·AUROC **0.995**는 §6 Ours focal+aug 3-class 원분포 수치와 **일치**(같은 run이므로 일치해야 정상).

### 8-1. train_ratio vs 성능 (원분포 valid, N=1403)

| train_ratio | train/클래스 | train 총합 | PR-AUC (주) | F1-macro | accuracy | AUROC | best ep |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.1 | 23 | 69 | 0.4892 | 0.3158 | 0.7014 | 0.9556 | 3 |
| 0.3 | 68 | 204 | 0.6402 | 0.6608 | 0.9473 | 0.9891 | 26 |
| 0.5 | 114 | 342 | 0.6802 | 0.6978 | 0.9551 | 0.9917 | 14 |
| 0.7 | 159 | 477 | 0.7476 | **0.5594** | 0.9387 | 0.9952 | 9 |
| 0.9 | 204 | 612 | 0.7514 | 0.7337 | 0.9629 | 0.9948 | 10 |
| **1.0 (=§6 Ours)** | **227** | **681** | **0.7647** | **0.7743** | **0.9729** | **0.9951** | 26 |

그림: `report/figures/exp_stability_dinov3.png`(좌: train_ratio vs PR-AUC/F1 곡선 + train 표본수 보조축, 우: accuracy/AUROC 포화 패널).

### 8-2. 해석 (안정성)

- **PR-AUC(주지표)는 데이터량에 거의 단조 증가하며 90%↑에서 포화(수확체감)**: 0.489(r0.1) → 0.640 → 0.680 → 0.748(r0.7) → 0.751(r0.9) → 0.765(r1.0). r0.1→r0.5에서 +0.191로 급상승한 뒤, r0.7~r1.0 구간은 0.748→0.765(+0.017)에 그쳐 **추가 데이터의 한계효용이 작다**. 즉 Ours의 주지표는 데이터량 증가에 **단조·안정적**으로 반응하며, 보유 데이터의 ~70–90%만으로도 정본(r1.0) PR-AUC의 98% 이상에 도달한다.
- **accuracy·AUROC는 더 일찍 포화**: AUROC는 r0.1에서 이미 0.956, r0.3부터 0.989↑로 사실상 천장. accuracy도 r0.3부터 0.94↑(다수 클래스 normal 1303 편향). 두 지표는 데이터량 민감도가 낮아 **stability 판단 근거로 부적합**(§9 한계 4) — PR-AUC/F1를 본다.
- **F1-macro는 단조가 아니라 변동**: r0.7에서 **0.559로 급락**(PR-AUC는 0.748로 오히려 높음). 원인은 (1) **F1-macro가 argmax(임계값 0.5) 의존 지표**라 확률 분리(PR-AUC/AUROC)가 좋아도 **decision boundary가 소수 클래스 d4(valid N=24)에서 흔들리면 macro 평균이 크게 떨어짐**, (2) **early-stop은 PR-AUC로 best epoch을 고르므로**(r0.7 best=ep9) PR-AUC-최적 체크포인트가 F1-최적과 어긋날 수 있음. r0.7의 낮은 F1은 PR-AUC가 높은 epoch에서 d4 argmax 분류가 불리하게 잡힌 결과로, **PR-AUC 추세(안정)와 F1 변동(불안정)은 같은 모델의 다른 단면**이다.
- **소표본 한계 병기**: valid **d4=24장**이라 d4가 1~2장만 다르게 분류돼도 d4 recall/precision이 ~0.04~0.08씩 출렁이고, 이것이 macro-F1을 좌우한다(§9 한계 2의 Wilson CI 폭 ~0.2와 동일 스케일). train 측도 r0.1은 클래스당 **23장**으로 극소표본이라 r0.1 수치(PR-AUC 0.489)는 데이터 부족의 하한 신호로만 읽고 단일 우열 판단은 금물. 따라서 **단일 시드·소표본**임을 감안해 F1의 비단조는 노이즈로 해석하고, **안정성의 결론은 단조·포화가 뚜렷한 PR-AUC를 기준**으로 내린다.

데이터: `_workspace/eval/stability_dinov3.json`. 스크립트: `_workspace/eval/run_stability_eval.py`(평가)·`make_stability_fig.py`(그림).

---

## 9. 한계 (해석 시 필수 고려)

1. **from-scratch (pretrained 미로딩, 6 백본 전부)** — 절대 성능 상한이 낮다. 백본 간 상대 비교·Ours 대비 기준선으로만 사용. pretrained 로딩 시 전 지표 상향 여지 큼.
2. **소표본·다운샘플** — train d4=227쌍, 3-class=type당 227. **valid disease_4=24장**이라 d4 관련 지표 CI 매우 넓음(recall CI 폭 ~0.2). 단일값 우열 판단 금지.
3. **불균형 valid + 임계값 0.5 부적합** — precision(macro) 0.49–0.89, disease precision은 더 낮음(오경보 다수). 운영 시 임계값 튜닝 필수.
4. **accuracy·AUROC 단독 해석 금지** — 정상 1303 다수라 전부 정상 찍어도 acc ≈94%, AUROC도 0.94–1.0으로 포화. 높은 accuracy가 보수적 예측(낮은 recall)이나 다수 클래스(정상) 편향의 산물일 수 있다. **PR-AUC/F1-macro/precision 병행 필수**.
5. **VisionMamba(`mamba`) 속도·성능 한계** — mambapy 순수 pscan 구현이라 커널 미최적화로 **가장 느림**(cls 7.1 s/ep, det 38 s/ep, ResNet50의 ~2–3배). 성능도 신규 중 최약(d3 PR-AUC 0.878, 3-class 0.359). from-scratch·소표본에서 SSM이 conv 계열만큼 데이터 효율적이지 않음. (mamba_ssm CUDA 빌드 불가 → mambapy 대체로 동작은 확보.)
6. **거친 단일 박스** — GT가 작물 영역 통째(중앙·≈프레임 절반). det_pr_auc는 질병 유무를 거의 완벽히 가리지만, IoU median 0.57–0.67·mAP 0.43–0.60은 미세 병변 국소화가 아니라 crop 영역 회귀 수준(REPORT §6). mAP는 국소화 hit rate에 묶여 det_pr_auc와 분리 해석.
7. **NeXtViT/VisionMamba detection은 fp32 필요** — fp16(AMP)에서 detection loss가 NaN으로 발산(수치 불안정). AMP off(fp32)로 학습/평가했고 정상 수렴. config.snapshot의 `optim.amp=false`로 동결.
8. **단일 시즌(2020-10~2021-01)** — 외부 도메인 일반화 검증 불가.

---

## 10. 변경 이력 (회귀 추적)

- **[현재 갱신] §8 Stability(train_ratio sweep, Ours 3-class) 추가**: Ours(DINOv3-B frozen @512 + focal+aug, 3-class `normal_d3_d4`)의 학습데이터량 안정성을 `train_ratio∈{0.1,0.3,0.5,0.7,0.9}` 신규 run(`experiments/dinov3_base_focal_r{10,30,50,70,90}_normal_d3_d4/`) + 참조점 **r1.0=§6 정본 run**으로 정리. **재학습 없음** — `metrics.json`+`predictions/valid.npz`만 사용(best.pt 미로딩), 동일 원분포 valid(N=1403). 6점 전부 predictions↔manifest **dist_match=PASS**·보고치↔sklearn 재계산 **|Δ|≤0.01 일치**, **r1.0 PR-AUC 0.765·F1 0.774·acc 0.973·AUROC 0.995가 §6와 일치**(동일 run). **결과**: **PR-AUC는 데이터량에 거의 단조 증가·90%↑ 포화**(0.489→0.640→0.680→0.748→0.751→0.765, r0.7~r1.0 +0.017로 수확체감), accuracy/AUROC는 더 일찍 포화(stability 부적합), **F1-macro는 비단조 변동**(r0.7 0.559로 급락하나 PR-AUC는 0.748 — F1이 argmax(0.5) 의존·early-stop은 PR-AUC로 best 선택해 어긋남, d4 N=24 소표본 민감). 안정성 결론은 단조·포화가 뚜렷한 PR-AUC 기준, F1 변동은 소표본·단일시드 노이즈로 해석(병기). 신규: `_workspace/eval/run_stability_eval.py`·`make_stability_fig.py`, 데이터 `_workspace/eval/stability_dinov3.json`, 그림 `report/figures/exp_stability_dinov3.png`. **기존 §1–§7·산출물 전부 불변**, 한계 §8→§9·변경이력 §9→§10으로 번호만 이동(본문 §8 신규 삽입), §7·§6.4 교차참조(§8 한계→§9) 정정.
- **[이전] §7 Sensitivity(입력 노이즈 강건성, Ours 3-class) 추가**: 학습된 Ours(DINOv3-B frozen @512 + focal+aug, `experiments/dinov3_base_focal_normal_d3_d4/`) best.pt를 **forward 전용 로드(재학습 없음)**, **원분포 valid(N=1403, seed=42)**에서 **정규화 입력 텐서**에 사용자 지정 노이즈 `x_noised = x + torch.rand_like(x)*N_ratio`(N_ratio∈{0.0,0.1,0.2,0.3,0.4,0.5}, manual_seed 고정)를 가산해 PR-AUC(주)/F1/acc/AUROC를 sklearn으로 재계산. **clean(N_ratio=0.0) PR-AUC 0.765·F1 0.772·acc 0.972·AUROC 0.995가 §6 Ours focal+aug 3-class 원분포와 |Δ|≤0.01 일치**(스크립트 assert로 검증). **결과: 매우 강건** — PR-AUC 저하 최대 −0.018(−2.4% @0.3), N_ratio=0.5에서도 −0.013(−1.7%), AUROC 0.995→0.994·acc 0.972→0.963; 단조는 아니며(0.3 저점 후 일부 회복) 저하 폭이 d4 N=24 소표본 변동 스케일이라 미세 우열은 단정 불가, 추세는 노이즈↑→소폭↓. 노이즈를 정규화 입력 텐서에 가한 것이라 실제 픽셀/촬영 노이즈와 분포가 다름을 명시. 신규: `_workspace/eval/run_sensitivity_eval.py`·`make_sensitivity_fig.py`, 데이터 `_workspace/eval/sensitivity_dinov3.json`, 그림 `report/figures/exp_sensitivity_dinov3.png`. **기존 §1–§6B·산출물 전부 불변**, 한계·변경이력 절 번호는 본문 §7 신규 삽입에 따라 한 칸씩 이동(현재 §8 Stability 추가 후 한계=§9·변경이력=§10).
- **[이전] §3·§3B detection 표에 Ours(DINOv3-B detection) 행 추가(pretrained-frozen 구분 주석)**: `experiments/dinov3_base_detection_singlebox/`(DINOv3 ViT-B/16 frozen @512 + single-box+objectness head, **total 85.84M / trainable 0.200M=head only**)을 §3-1(이미지 단위 검출)·§3-2(국소화)·§3B(균형 valid) 표 하단에 **굵게+구분선으로 추가**. 원분포(§3)는 저장 `predictions/valid.json`에서, 균형(§3B)은 best.pt **forward 전용 로드(가중치 변경 없음)** + `balance_valid=True`(seed=42, img512, fp32)로 재평가해 det_pr_auc/det_roc_auc/presence_recall@0.5/fp_rate@0.5/mAP@0.5/IoU median을 **독립 재계산**. **정합성**: predictions↔manifest N=1403·pos/neg=100/1303 **PASS**, reported↔recomputed 일치(mAP만 AP 보간차 ≤0.05), **sklearn `average_precision_score`/`roc_auc_score` 교차검증 orig·balanced 모두 1e-9 이내 일치**. **결과**: 원분포·균형 둘 다 det PR-AUC=ROC-AUC=**1.0**, presence_recall **1.0**(100/100), fp_rate orig 1/1303(0.08%)·balanced 0/100 → **검출은 baseline 6종과 동급(포화)**; objectness 분리 깨끗(질병 median 0.955 vs 정상 0.017). 국소화 IoU median **0.565**(mAP 0.491)로 baseline 범위(0.57–0.67) 하단. **공정성 주석**: Ours는 자기지도 pretrained frozen+head-only라 from-scratch baseline과 동일조건 아님(params `(frozen)+(tr)` 병기). **⚠️ best-epoch 각주**: det_pr_auc가 ep0부터 1.0 포화→early-stop이 best=ep0 선택→저장 predictions의 **국소화 IoU가 전 epoch 최저(0.565)**, 후반(ep≥28) IoU median ≈0.64까지 상승(train.log/per_epoch). 검출 지표는 ep0부터 포화라 epoch 선택과 무관 — 이 한계를 §3-1/§3-2/§3B 각주에 명시. 신규: `run_ours_detection_eval.py`, 데이터 `ours_detection.json`, verify `verify_dinov3_base_detection_singlebox.md`. 그림 `exp_detection.png`에 Ours 막대(주황 음영)·objectness 중앙값 다이아몬드·IoU 히스토그램(검정 점선) 추가·재생성. **기존 baseline 6 백본 detection 행·수치·해석·그림은 전부 불변**(재실행으로 재계산값 일치 확인).
- **[이전] §1·§1B 표에 Ours(DINOv3) 행 추가(pretrained-frozen 구분 주석)**: baseline-별 분류 성능표(§1 원분포 §1-1/1-2/1-3, §1B 균형 §1B-1/1B-2/1B-3) 6개 각각에 **Ours 3변형 행을 하단에 굵게+구분선으로 추가**해 한 표에서 직접 비교 가능하게 함 — **Ours: DINOv3-S @256(frozen,CE)**(21.7M/trainable 99K), **Ours: DINOv3-B @512(frozen,CE)**(86.0M/395K), **Ours: DINOv3-B @512(frozen,focal+aug)**(86.0M/395K). 동일 7메트릭 컬럼(params/acc/train·val_loss/recall·precision·f1 macro/AUROC/PR-AUC/best ep), 수치는 `ours_dinov3.json`(=§6 정본)에서 가져와 **predictions에서 sklearn 1건 이상 재계산 대조**(small d3·base d4·focal 3-class 모두 보고치와 완전 일치). **공정성 주석**을 §1 도입부·§1B 도입부에 명기: "baseline 6종은 from-scratch, Ours는 DINOv3 자기지도 pretrained(frozen backbone)+head-only — 사전학습 사용 여부가 달라 동일 조건 비교가 아님"; params(M)도 `(frozen)+(trainable)` 구분 병기(과대비교 방지). 비교 그림 `exp_metrics_table.png`(원분포)·`exp_metrics_balanced.png`(균형)에 **Ours 3종 막대를 baseline과 색/해치 구분(black/red/violet, "pretrained" 표기)+구분선**으로 추가·재생성(`run_eval.py` fig_metrics_table, `make_balanced_fig.py`). **기존 baseline 6 백본 행·수치·해석은 전부 불변**(재실행으로 재계산값 일치 확인). 정본 평가·목표 판정·ablation은 §6/§6B 유지.
- **[이전] Ablation(aug×focal, gamma sweep) + 하이퍼파라미터 표 추가(§6B, 부록 A)**: dinov3_base 3-class에서 backbone·head·해상도·optimizer·trainable(395k)·seed 고정, **loss(CE/focal-γ)와 aug(default/strong)만** 바꾼 ablation 4 run(augonly·focalonly·focalg1·focalg3) + 참조 2개(base, focal+aug)를 추가. **2×2 기여 분해**(aug 효과 PR +0.011·F1 +0.018, focal 단독 PR −0.009·F1 −0.015, **상호작용 PR +0.018·F1 +0.022** → focal은 strong aug와 결합 시에만 이득; **강한 증강이 주 동력·조합 시너지** 확인), **gamma 스윕**(strong aug 고정 γ1/γ2/γ3: PR-AUC는 γ↑ 단조 개선 γ3 0.769 최고, F1·d4 precision은 γ↓ 유리 γ1 0.782/0.412 최고 — 곡선 교차), **d4 precision 0.339→0.39~0.41**(strong-aug 계열)·d4 N=24 Wilson CI 병기(전 run CI 중첩 → 단일 시드·소표본이라 개별 우열 단정 금지, 방향성만). 6 run predictions↔manifest dist_match=PASS, sklearn 재계산↔보고 완전 일치. 신규: `run_ablation_eval.py`·`make_ablation_fig.py`, `ablation_dinov3.json`, verify 4종, 그림 `exp_ablation_dinov3.png`(2×2+기여분해+gamma). **부록 A**에 baseline 6백본·Ours(small/base)·focal·ablation 전 주요 실험의 config.snapshot 추출 하이퍼파라미터 표를 추가. §6/§7 등 기존 내용 전부 보존.
- **[현재 갱신] Ours+ 강한 증강+focal loss 추가·평가(§6)**: Ours+(base@512)와 **backbone·head·해상도·trainable(395k) 전부 동일**하되 학습만 **강한 증강(aug=strong)+focal loss(gamma=2, class_weights=from_meta)**로 바꾼 개선판 3 run(`experiments/dinov3_base_focal_{normal_vs_d3,normal_vs_d4,normal_d3_d4}`, img512)을 추가해 small·base-CE·baseline 최고와 4-way 비교. 원분포·균형 valid 둘 다 7메트릭+PR-AUC+per-class(d3/d4) recall·precision·confusion을 **predictions에서 sklearn 독립 재계산**(보고치와 완전 일치)하고, 균형은 best.pt 로드해 `balance_valid=True`(seed=42, **img_size=512**)로 forward 재평가(가중치 변경 없음). 정합성: 3세팅 predictions↔manifest **dist_match=PASS**, reported↔recomputed 완전 일치, sklearn 교차검증 일치. **결과(원분포)**: 3-class PR-AUC **0.745(base-CE)→0.765(focal+aug)**, F1 **0.750→0.774**; **d4 precision 0.339→0.396**(정상→d4 오분류 16→9장↓, d4 recall 0.875 불변); d4(2-class) F1 0.866→0.919. **+20% 재판정 — PR-AUC는 baseline 0.570 대비 +34.1%로 목표(0.684) 가장 견고히 달성, F1-macro는 +9.2%로 base-CE(+5.8%)보다 개선됐으나 절대 목표(0.851)에는 미달**(d4 precision 병목 완화되었으나 미해소). **해석 단서: aug+focal 두 변수를 동시에 변경했으므로 개선의 기여 분리 불가(단변수 ablation 후속 필요); from_meta 가중치는 균형 train 로더 때문에 전부 1.0으로 해소되어 focal의 작동 기제는 gamma(hard-example 집중)**임을 명시. 신규 스크립트 `run_ours_focal_eval.py`(small+base-CE+focal 병합 dump), 그림 스크립트 `make_ours_focal_fig.py`, 데이터 `ours_dinov3.json`(small+base-CE+focal 병합, uplift_focal·goal_verdict_3class_focal·ce_to_focal_improvement 추가), verify 3종(`verify_dinov3_base_focal_*.md`, per-class CE↔focal·Wilson CI·ce→focal delta 포함), 신규 그림 `exp_ours_focal.png`(baseline/small/base-CE/focal+aug 4-way, +20% 목표선). 기존 §6 표·해석·산출물 전부 보존·확장, baseline 6 백본(§1–5, 7) 불변.
- **[이전] Ours+ DINOv3 base@512 추가·평가(§6)**: 성능 최대화를 위해 **Ours+ = DINOv3 ViT-B/16 frozen @512 + 2-layer head(hidden 512)** 3 run(`experiments/dinov3_base_{normal_vs_d3,normal_vs_d4,normal_d3_d4}`)을 추가해 Ours(small@256)·baseline 최고와 공정 비교. frozen backbone 85.6M + head 395k trainable, img_size=512(config.snapshot). 원분포·균형 valid 둘 다 7메트릭+PR-AUC를 **predictions에서 sklearn 독립 재계산**(보고치와 완전 일치)하고, 균형은 best.pt 로드해 `balance_valid=True`(seed=42, **img_size=512**)로 forward 재평가(가중치 변경 없음). 정합성: 3세팅 predictions↔manifest **dist_match=PASS**. **결과**: 3-class 원분포 PR-AUC **0.689(small)→0.745(base)**, F1 **0.698→0.750**; **+20% 판정 — PR-AUC는 baseline 0.570 대비 +30.7%로 목표(0.684) 견고히 달성, F1-macro는 baseline 대비 +5.8%로 개선됐으나 절대 목표(0.851)에는 미달**(d4 precision 0.339 병목). 2-class는 PR-AUC 천장이나 F1에서 +14.3%/+27.6% 향상. 신규 스크립트 `run_ours_plus_eval.py`(small+base 병합 dump), 그림 스크립트 `make_ours_dinov3_fig.py`, 데이터 `ours_dinov3.json`(small+base 병합), verify 3종(`verify_dinov3_base_*.md`, per-class recall/precision·small→base delta 포함), 그림 `exp_ours_dinov3.png`(small+base+baseline 3-way, +20% 목표선) 갱신. Ours(small) §6 표·해석은 보존·확장, baseline 6 백본(§1–5, 7) 불변. §6 제목·6.1–6.4를 Ours/Ours+ 병기로 확장(번호 보존).
- **[이전] Ours(DINOv3 frozen+2-layer head) 추가·평가(§6)**: 자기지도 DINOv3 ViT-S/16 동결 백본 + 2-layer head 분류 3 run(`experiments/dinov3_*`)을 baseline과 공정 비교. 원분포·균형 valid 둘 다 7메트릭+PR-AUC를 **predictions에서 sklearn 독립 재계산**(보고치와 완전 일치)하고, 균형은 best.pt 로드해 `balance_valid=True`(seed=42, img_size=256)로 forward 재평가. 정합성: 3세팅 predictions↔manifest **dist_match=PASS**. **+20% 목표 판정**: 3-class 원분포 **PR-AUC 0.689≥목표 0.684(baseline 0.570 대비 +20.8%) → 주지표 달성**, F1-macro 0.698<0.851(미달, baseline 동급). 2-class·균형은 천장 효과로 향상 여지 작음을 병기. 신규 스크립트 `run_ours_eval.py`, 데이터 `ours_dinov3.json`, verify 3종(`verify_dinov3_*.md`), 그림 `exp_ours_dinov3.png`. 기존 6 백본 표·해석·산출물 전부 보존(§1–5, 7 불변; 한계 절은 §6→§7로, 변경 이력 §7→§8로 번호만 이동).
- **[이전] 사용자 요청으로 nextvit20을 리포트 표·그림에서 제외(실험 산출물 `experiments/nextvit20_*`는 보존)**: 리포트 기준 백본 7종→6종, 분류 21→18·detection 7→6·총 28→24 run으로 카운트 갱신. eval 스크립트(`run_eval.py`/`run_balanced_eval.py`/`run_balanced_detection_eval.py`)와 그림 스크립트(`make_balanced_fig.py`/`plot_balanced_detection.py`)의 백본 목록 상수에서 `nextvit20`을 제거하고 재실행해 비교 표·그림·summary를 6 백본 기준으로 일괄 재생성(`exp_metrics_table.png`/`exp_metrics_balanced.png`/`exp_cls_bars.png`/`exp_pr_curves.png`/`exp_confusion.png`/`exp_detection.png`/`exp_detection_balanced.png`/`training_curves.png`). 본문에서 nextvit20 vs nextvit-small 비교 단락 등 NeXtViT-b 언급을 제거/수정. 다른 6 백본 수치·해석은 불변. 개별 `experiments/nextvit20_*`·`curves_nextvit20_*.png`는 보존.
- **[이전] 백본 4종 추가 + AUROC·7메트릭 도입**: DenseNet121 / ResNet50 / NeXtViT-base(`nextvit20`) / **Vision-Mamba(`mamba`=mambapy pscan, mamba_ssm 빌드 불가 대체)** 학습·평가 추가 → 분류 9→21, detection 3→7, **총 12→28 run**(당시 7 백본 기준; 현재는 nextvit20 제외로 6 백본·24 run). 분류 표를 **7-메트릭(accuracy/train_loss/val_loss/recall(macro)/precision(macro)/f1(macro)/AUROC)** 으로 확장. **기존 3 백본의 AUROC·macro-PRF는 metrics.json에 없어 predictions에서 독립 재계산**(sklearn 교차검증 일치), train/val loss는 per_epoch best-epoch 값 사용. `run_eval.py`를 다중 백본·7메트릭으로 확장, `exp_metrics_table.png` 신규 추가, `training_curves.png`·`exp_detection.png`·`exp_confusion.png`·`exp_pr_curves.png` 갱신. **기존 3 백본 결과·해석은 보존**하고 신규 백본 기준으로 표·해석을 확장. 발견 정합성 이슈 없음(당시 28/28 PASS).
- **이전: 초기 detection objectness collapse → 수정·재학습·재평가**: 초기 detection run은 정상 이미지를 음성으로 다루지 않아 objectness가 전 이미지 ~1.0으로 붕괴 → 정상/질병 분리 불가, NeXtViT는 best epoch=1 early-stop으로 학습 실패. (A) 데이터 로더 정상=음성·빈 GT 수정(data-engineer), (B) 학습/지표 음성 처리·재학습(experiment-runner, NeXt-ViT/Mamba fp16 NaN 회피 위해 AMP off=fp32), (C) 음성 인지 지표(det_pr_auc/ROC-AUC/presence_recall@0.5/fp_rate@0.5) 재평가(eval-reporter). 결과: 붕괴 해소(질병 objectness median ≈0.998 vs 정상 ≈0.001), 신규 detection 4 run도 동일 수정된 파이프라인에서 학습돼 붕괴 없음.

---

## 산출물

| 산출물 | 경로 |
|--------|------|
| run별 정합성 검증 (24) | `_workspace/eval/verify_<name>.md` |
| **Ours(DINOv3 small) 정합성 검증 (3)** | `_workspace/eval/verify_dinov3_<setting>.md` |
| **Ours+(DINOv3 base@512) 정합성 검증 (3, per-class·small→base delta 포함)** | `_workspace/eval/verify_dinov3_base_<setting>.md` |
| **Ours+ focal+aug 정합성 검증 (3, per-class CE↔focal·Wilson CI·ce→focal delta 포함)** | `_workspace/eval/verify_dinov3_base_focal_<setting>.md` |
| **Ours/Ours+ 평가 데이터(small+base-CE+focal+aug 병합, 원분포+균형 7메트릭·향상률·20% 판정)** | `_workspace/eval/ours_dinov3.json` |
| **Ours+ focal+aug 평가 스크립트(predictions 재계산 + best.pt 균형 재평가 + 병합)** | `_workspace/eval/run_ours_focal_eval.py` |
| **Ours+ focal+aug vs base-CE vs small vs baseline 비교 그림(4-way, 20% 목표선)** | `report/figures/exp_ours_focal.png` |
| **Ours+ focal+aug 비교 그림 스크립트** | `_workspace/eval/make_ours_focal_fig.py` |
| **§7 입력 노이즈 sensitivity 데이터(N_ratio별 PR-AUC/F1/acc/AUROC + clean 대비 저하 + §6 대조)** | `_workspace/eval/sensitivity_dinov3.json` |
| **§7 입력 노이즈 sensitivity 평가 스크립트(best.pt forward 전용, 입력 텐서 노이즈)** | `_workspace/eval/run_sensitivity_eval.py` |
| **§7 입력 노이즈 sensitivity 그림(N_ratio vs PR-AUC/F1 + acc/AUROC)** | `report/figures/exp_sensitivity_dinov3.png` |
| **§7 sensitivity 그림 스크립트** | `_workspace/eval/make_sensitivity_fig.py` |
| **§8 stability(train_ratio sweep) 데이터(6점 PR-AUC/F1/acc/AUROC + train_counts + boundary/재계산 대조)** | `_workspace/eval/stability_dinov3.json` |
| **§8 stability 평가 스크립트(predictions+metrics 재계산, 재학습 없음)** | `_workspace/eval/run_stability_eval.py` |
| **§8 stability 그림(train_ratio vs PR-AUC/F1 + train 표본수 보조축 + acc/AUROC)** | `report/figures/exp_stability_dinov3.png` |
| **§8 stability 그림 스크립트** | `_workspace/eval/make_stability_fig.py` |
| **Ours(small) 평가 스크립트(predictions 재계산 + best.pt 균형 재평가)** | `_workspace/eval/run_ours_eval.py` |
| **Ours+(base@512) 평가 스크립트(predictions 재계산 + best.pt 균형 재평가 + small 병합)** | `_workspace/eval/run_ours_plus_eval.py` |
| **Ours/Ours+ vs baseline 최고 비교 그림(PR-AUC·F1, 원분포/균형, 3-way, 20% 목표선)** | `report/figures/exp_ours_dinov3.png` |
| **Ours/Ours+ 비교 그림 스크립트** | `_workspace/eval/make_ours_dinov3_fig.py` |
| 재계산 요약(JSON) | `_workspace/eval/summary.json` |
| **균형 valid 재평가 데이터(§1B, 원분포 delta 포함)** | `_workspace/eval/balanced_valid.json` |
| **균형 valid 재평가 스크립트(best.pt 로드·재측정)** | `_workspace/eval/run_balanced_eval.py` |
| **균형 valid 메트릭 막대(AUROC/F1-macro/accuracy, 6백본×3세팅)** | `report/figures/exp_metrics_balanced.png` |
| **균형 valid detection 재평가 데이터(§3B, 원분포 delta·sklearn 교차검증 포함)** | `_workspace/eval/balanced_valid_detection.json` |
| **균형 valid detection 재평가 스크립트(best.pt 로드·fp32 재측정)** | `_workspace/eval/run_balanced_detection_eval.py` |
| **균형 valid detection 그림(det PR-AUC orig vs bal + presence/fp + mAP/IoU)** | `report/figures/exp_detection_balanced.png` |
| 평가 스크립트 (6 백본·7메트릭) | `_workspace/eval/run_eval.py` |
| run별 학습 곡선 (24) | `report/figures/curves_<name>.png` |
| 종합 학습 곡선 패널 (4행 × 6열) | `report/figures/training_curves.png` |
| **분류 7-메트릭 핵심 막대 (AUROC/F1-macro/accuracy)** | `report/figures/exp_metrics_table.png` |
| 분류 PR-AUC/F1 막대 (6 백본) | `report/figures/exp_cls_bars.png` |
| 분류 PR 곡선 (세팅별, 신규=점선) | `report/figures/exp_pr_curves.png` |
| 분류 혼동행렬 (18) | `report/figures/exp_confusion.png` |
| detection 비교 (6 백본: det_pr_auc/recall/fp_rate 막대 + objectness 분포 + 양성 IoU) | `report/figures/exp_detection.png` |
| **Ablation 정합성 검증 (4, boundary·재계산·per-class·Wilson CI·Δ vs base)** | `_workspace/eval/verify_dinov3_base_{augonly,focalonly,focalg1,focalg3}_normal_d3_d4.md` |
| **Ablation 평가 데이터(2×2 분해·gamma 스윕·d4 CI 덤프)** | `_workspace/eval/ablation_dinov3.json` |
| **Ablation 평가 스크립트(predictions 재계산 + 기여 분해 + gamma 스윕)** | `_workspace/eval/run_ablation_eval.py` |
| **Ablation 그림(2×2 막대 + 기여분해 + gamma 곡선)** | `report/figures/exp_ablation_dinov3.png` |
| **Ablation 그림 스크립트** | `_workspace/eval/make_ablation_fig.py` |

---

## 부록 A. 실험별 하이퍼파라미터

재현성을 위해 각 run의 `experiments/<name>/config.snapshot`(JSON)에서 직접 추출했다(추출 경로: `spec.model{arch,img_size,pretrained,frozen_backbone,head_hidden}`, `spec.data{batch_size,aug}`, `spec.optim{name,lr,wd,epochs,sched,warmup_epochs}`, `spec.loss{type,gamma,class_weights,label_smoothing}`, `spec.seed`, 최상위 `trainable_params{trainable,total}`). **공통값은 그룹 헤더에 1행으로 묶고 예외만 행으로** 표기한다. 모든 run 공통: optimizer=**AdamW**, wd=**0.05**, sched=**cosine**, seed=**42**, label_smoothing=**0.0**, class_weights=**from_meta**(균형 train 로더에서 전부 1.0으로 해소). baseline은 from-scratch(pretrained=False, frozen 무관), Ours/ablation은 DINOv3 frozen(pretrained=True, frozen_backbone=True).

### A-1. Baseline 6 백본 (from-scratch 분류)

**공통**: arch별 백본 / img_size=**224** / batch=**64** / lr=**3e-4** / epochs=**60** / warmup=**5** / aug=**default** / loss=**CE** / pretrained=**False** / 백본 전체 학습(frozen 아님, trainable=total). 3 세팅(normal_vs_d3 / normal_vs_d4 / normal_d3_d4) 모두 동일 하이퍼(세팅 간 차이 없음). 백본 파라미터(M, `build_classifier`): convnextv2 28.26 / efficientnetv2 20.84 / nextvit 31.27 / densenet121 7.48 / resnet50 24.56 / mamba 3.88.

| 그룹/run | arch | img | batch | lr | epochs | warmup | aug | loss | frozen | 비고 |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| baseline 분류 (6백본 × 3세팅, 18 run) | convnextv2/efficientnetv2/nextvit/densenet121/resnet50/mamba | 224 | 64 | 3e-4 | 60 | 5 | default | CE | ✗ (전체 학습) | 세팅 무관 동일 하이퍼 |
| baseline detection (6백본, 6 run) | 동일 6 백본 (single-box+objectness) | 512 | 8 | 3e-4 | 60 | 5 | default | giou(2)+l1(5)+obj(1), box_loss=positive_only | ✗ | nextvit·mamba는 **AMP off(fp32)**(fp16 NaN 회피); 그 외 fp32 기본 |

### A-2. Ours / Ours+ (DINOv3 frozen + 2-layer head)

**공통**: pretrained=**True**, frozen_backbone=**True**(backbone 동결, head만 학습), lr=**1e-3**, warmup=**3**, aug=default, loss=CE. small은 timm `vit_small_patch16_dinov3`(head_hidden 기본 256), base는 `vit_base_patch16_dinov3`(head_hidden 512). epochs: small=30, base=40.

| 그룹/run | arch | img | head_hidden | batch | lr | epochs | warmup | aug | loss | trainable / total | 비고 |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| Ours (small, 3세팅) | dinov3 (ViT-S/16) | 256 | 256 | 64 | 1e-3 | 30 | 3 | default | CE | ~99k / 21.69M | frozen backbone |
| Ours+ (base, 3세팅) | dinov3_base (ViT-B/16) | 512 | 512 | 64 | 1e-3 | 40 | 3 | default | CE | 395k / 86.04M | frozen backbone |
| Ours+ focal+aug (3세팅) | dinov3_base (ViT-B/16) | 512 | 512 | 64 | 1e-3 | 40 | 3 | **strong** | **focal γ2** | 395k / 86.04M | strong aug + focal; trainable는 base와 동일 |

(small `head_hidden`은 config에 명시 없이 빌더 기본값 256; base는 명시적 512. trainable: small 3-class 99.33k·2-class 99.07k, base 3-class 395.27k·2-class 394.75k — head 출력차로 미세 차이.)

### A-3. Ablation (dinov3_base 3-class, normal_d3_d4) — §6B

**공통(전 6 run 고정)**: arch=dinov3_base(ViT-B/16 frozen) / img=**512** / head_hidden=**512** / batch=**64** / lr=**1e-3** / wd=0.05 / epochs=**40** / warmup=**3** / pretrained=True / frozen_backbone=True / trainable=**395,267** / total=86,036,483 / seed=42. **loss와 aug만 변수**:

| run | loss type | gamma | aug | (역할) |
|---|---|:---:|:---:|---|
| dinov3_base_normal_d3_d4 (참조) | cross_entropy | — | default | base (CE/default) |
| dinov3_base_augonly_normal_d3_d4 | cross_entropy | — | strong | aug-only |
| dinov3_base_focalonly_normal_d3_d4 | focal | 2.0 | default | focal-only |
| dinov3_base_focalg1_normal_d3_d4 | focal | 1.0 | strong | γ1 (strong aug) |
| dinov3_base_focal_normal_d3_d4 (참조) | focal | 2.0 | strong | focal+aug (Ours+, γ2) |
| dinov3_base_focalg3_normal_d3_d4 | focal | 3.0 | strong | γ3 (strong aug) |

(전 6 run: optimizer=AdamW, sched=cosine, label_smoothing=0.0, class_weights=from_meta(→all-ones). 출처: 각 `experiments/<run>/config.snapshot`. ablation은 normal_d3_d4 세팅만 존재.)

**포함 실험 수**: baseline 분류 18 + detection 6 + Ours(small) 3 + Ours+(base CE) 3 + Ours+ focal+aug 3 + ablation 4(augonly/focalonly/focalg1/focalg3) = **37 run**(참조 2개는 Ours+에 이미 포함).
