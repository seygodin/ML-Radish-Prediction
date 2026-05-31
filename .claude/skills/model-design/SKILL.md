---
name: model-design
description: 무 질병 분류·detection 실험을 설계할 때 반드시 사용. baseline/의 백본(ConvNeXtV2/EfficientNetV2/NeXtViT/MambaVision)을 분류기로 래핑하고, detection baseline을 정의하며, "Ours" 모델을 가설 기반으로 설계하고, loss·하이퍼파라미터·평가 프로토콜·ablation을 실험 사양(yaml)으로 동결한다. "모델 설계", "baseline 구성", "Ours 모델", "실험 사양", "loss 정하기", "ablation 계획", "백본 래핑" 요청 시 트리거. 학습 실행은 experiment-runner, 지표 계산/비교는 eval-and-report 소관.
---

# model-design

실험을 코드보다 먼저 **사양으로 동결**한다. 목적은 baseline을 공정하게 측정하고, "Ours"가 무엇을 왜 개선하는지 검증 가능하게 만드는 것이다.

## 데이터 현실에서 출발 (설계 제약)
`report/REPORT.md`의 함의가 설계를 지배한다:
- 정상:질병 13–16:1, 질병 2종(d3:d4 ≈ 2.2:1), valid disease_4=24장(지표 불안정), 중증도 극단 편중.
- bbox는 **거친 영역 박스**(면적 ≈50%, 중앙). detection은 미세 병변보다 "작물/잎 영역" 학습에 가깝다.
→ 따라서:
- 분류 loss는 **class-weighted CE 또는 focal loss** 기본. accuracy 단독 평가 금지.
- detection은 mAP와 함께 "이미지 단위 존재 검출 + IoU 분포"를 병행 측정. 미세 병변 국소화가 진짜 목표면 **분류+CAM/Grad-CAM** 대안을 설계안에 병기.

## Baseline 설계
**확정 원칙: 분류·detection 모두 `baseline/`의 백본 4종을 그대로(as-is) 사용한다.** 외부 검출기(torchvision detection, YOLO 등)나 다른 백본을 끌어오지 않는다. `baseline/`의 `convnextv2.py`, `efficientnetv2.py`, `nextvit.py`, `mambavision.py`는 timm 기반 아키텍처 구현이며, 이를 feature extractor로 공유한다.

### 분류 — 확정된 3가지 세팅
4개 백본 각각을 아래 3세팅으로 학습·평가(총 4×3=12 run). 공통 분류기 래퍼: `backbone → global pool → dropout → linear(num_classes)`, 가능하면 ImageNet pretrained 로드. 동일 입력 크기·증강·optimizer로 공정 비교.
1. **normal vs disease_3** (2-class)
2. **normal vs disease_4** (2-class)
3. **normal vs disease_3 vs disease_4** (3-class)

**클래스 균형(확정):** normal을 **다운샘플링**하여 type 간 표본 수를 맞춘다(불균형 13–16:1 제거).
- 2-class: `n_normal = n_disease_k`.
- 3-class: 세 type을 동일 수로 — 가장 적은 type(보통 disease_4) 기준으로 normal·disease_3도 다운샘플(`n = min(n_d3, n_d4, ...)`). 다운샘플은 고정 seed로 manifest에 동결(data-engineer 담당).
- 다운샘플로 균형을 맞췄으므로 class-weighted/focal은 보조 옵션. 단, accuracy 단독 평가는 여전히 금지(소표본이므로 PR-AUC/F1/recall 병행).

### Detection
- **backbone은 `baseline/`의 4종을 그대로 feature extractor로 사용**하고, 위에 간단한 detection 헤드(단일 bbox 회귀 + 객체성/클래스)를 얹는다. FPN이 필요하면 백본의 stage feature를 활용.
- 대상은 abnormal(질병) 이미지의 단일 거친 bbox. 클래스는 단일("질병") 또는 disease_code별 중 사양에 고정.
- 의존성: `torch`, `torchvision`(IO/ops), `timm`(백본 의존). 미설치면 사양 packages에 명시.

## "Ours" 설계 원칙 (가설 주도)
모든 Ours 변경은 **(가설 → 메커니즘 → 예상 효과 → 측정/ablation)** 4요소를 design_notes에 적는다. 후보 방향:
- 불균형 대응: focal/LDAM/class-balanced loss, 가중 샘플링, 증강 강도 차등.
- 표현 학습: 사전학습 전략(self-supervised/추가 작물 데이터), 멀티스케일 입력(고해상 병변 대응).
- task 결합: 분류 + CAM 기반 약지도 국소화로 거친 bbox 한계 우회.
- 견고성: 단일 시즌 데이터 → 강한 증강/test-time augmentation.
한 번에 하나의 변수만 바꿔 ablation 가능하게 한다(혼합 변경 금지).

## 공정 비교 프로토콜
- 동일 split(데이터셋 제공 valid), 동일 입력 크기, 동일 metric, **seed 고정**, 동일 epoch 예산.
- baseline과 Ours가 같은 `src/data/` 로더를 쓰도록 강제.

## 산출물 (계약)
1. **`_workspace/specs/exp_{name}.yaml`** — 자기완결 실험 사양. 권장 필드:
   ```yaml
   name: convnextv2_normal_vs_d3
   task: classification        # or detection
   model: {arch: convnextv2, source: baseline/convnextv2.py, pretrained: true, num_classes: 2}
   data: {builder: build_classification_loaders, disease_code: 3, img_size: 320, batch_size: 32}
   loss: {type: focal, gamma: 2.0, class_weights: from_manifest}
   optim: {name: adamw, lr: 3.0e-4, wd: 0.05, epochs: 30, sched: cosine}
   eval: {metrics: [pr_auc, f1, recall_disease, confusion], primary: pr_auc}
   seed: 42
   packages: [torch, torchvision, timm]
   smoke: true                 # experiment-runner가 먼저 소규모로 검증
   ```
2. **`src/models/`** — 백본 래퍼/Ours 구현(필요 시). `build_model(spec) -> nn.Module` 형태 권장.
3. **`_workspace/specs/design_notes.md`** — 비교 매트릭스, Ours 가설표, ablation 계획.

## 작업 절차
1. `data_card_*.md`로 데이터 인터페이스 확인 → 입력 크기/정규화/가중치 요구를 data-engineer에 전달.
2. baseline 비교 매트릭스 확정 → 각 셀에 대한 exp yaml 생성.
3. baseline 결과가 이미 있으면(`experiments/*/metrics.json`) 그 수치로 Ours 가설을 조정.
4. metric 정의를 eval-and-report와 일치시킨다(같은 계산식·같은 임계값 정책).

## 검증 체크리스트
- [ ] 각 exp yaml이 experiment-runner만으로 실행 가능한 자기완결 사양인가
- [ ] baseline과 Ours가 동일 split·metric·seed를 쓰는가
- [ ] Ours 변경마다 가설·ablation이 design_notes에 있는가
- [ ] 불균형 대응 loss와 불균형 인지 metric이 사양에 반영됐는가
