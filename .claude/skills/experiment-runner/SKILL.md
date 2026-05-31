---
name: experiment-runner
description: 무 질병 모델 학습·평가를 재현 가능하게 실행할 때 반드시 사용. 사양 yaml대로 .venv/uv 환경을 갖추고, 학습 루프를 돌리고, metric을 로깅하고, 체크포인트·예측·로그를 experiments/<name>/에 표준 구조로 저장한다. "학습 돌려", "모델 train", "실험 실행", "체크포인트", "재현", "GPU로 학습", "스모크 테스트 후 본학습" 요청 시 트리거. 데이터 로더는 radish-data-pipeline, 사양 정의는 model-design, 지표 비교는 eval-and-report 소관.
---

# experiment-runner

사양 yaml을 **실제 학습/평가**로 옮기고 결과를 표준 구조로 남긴다. 핵심 가치는 **재현성**과 **장애 격리**다.

## 환경 (이 프로젝트의 필수 제약)
- 시스템에 **pip/ensurepip가 없다.** 항상 프로젝트 venv를 쓴다:
  - 실행: `./.venv/bin/python ...`
  - 설치: `uv pip install --python .venv/bin/python <pkgs>` (사양 `packages` 목록대로)
- 학습 시작 전 GPU 확인: `./.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"`.
- seed를 torch/numpy/random에 모두 고정하고 `cudnn.benchmark`/`deterministic` 정책을 로그에 남긴다.

## 표준 출력 레이아웃 (계약)
모든 run은 정확히 이 구조로 남긴다 — eval-reporter가 이 경로를 읽는다.
```
experiments/<name>/
├── metrics.json        # {status, primary, per_epoch:[...], final:{...}}  ← 평가 계약 파일
│                        #   per_epoch의 각 항목은 train_loss와 val_loss를 모두 포함해야 한다
│                        #   (학습 곡선 시각화의 근거 — 누락 금지)
├── config.snapshot     # 사양 yaml + 패키지 버전 + 명령 + seed + GPU
├── train.log
├── checkpoints/        # best.pt, last.pt
└── predictions/        # valid 예측 덤프(분류: logits/prob+label; detection: boxes+scores)
```
`metrics.json`은 항상 `status`(`ok`/`failed`) 필드를 갖는다. 실패해도 파일은 남긴다.

## 작업 절차
1. `_workspace/specs/exp_{name}.yaml` 로드 → `packages` 설치 → 환경/ GPU 스냅샷 기록.
2. **스모크 우선**: 사양에 `smoke: true`면 소수 step/epoch로 먼저 돌려 데이터→모델→loss→backward 무결성과 한 번의 eval 경로를 확인한다. 깨지면 본 학습 전에 보고.
3. 본 학습은 **백그라운드로 실행**하고 `train.log`를 주기적으로 tail로 점검(장시간 작업). 주기적 검증 + best 체크포인트 저장.
4. 학습 후 valid 평가 → `predictions/` 덤프 + `metrics.json` 작성. run 요약(최종 지표·소요·경고)을 stdout 보고.

## 견고성 규칙
- **OOM/자원 부족**: 배치↓ 또는 AMP 활성 또는 해상도↓를 사양 허용 범위에서 시도하고 조정 내역을 `config.snapshot`에 기록. 임의로 사양 의미를 바꾸지 말 것.
- **재시도**: run 실패 시 1회 재시도. 재실패면 `metrics.json`에 `status: failed`+에러 요약을 남기고 다음 run으로 진행(전체 파이프라인 중단 금지).
- **resume**: 중단된 학습은 `checkpoints/last.pt`에서 재개. 같은 `<name>` 재실행 정책은 사양을 따르되 기본은 기존 결과 보존(`<name>_v2`).
- data-engineer 모듈의 공개 API를 **그대로** 호출한다. 시그니처 불일치는 고치지 말고 오케스트레이터에 보고.

## 분류 / detection 실행 메모
- 분류: class_weights/sampler는 데이터 로더가 제공(`build_classification_loaders` 반환값)하는 것을 사용. logits와 정답을 predictions에 저장해 eval-reporter가 임계값을 바꿔가며 재계산할 수 있게 한다.
- detection: torchvision 검출기는 `(images, targets)` 형식. predictions에 `boxes,scores,labels`와 GT를 함께 저장.

## 검증 체크리스트
- [ ] `.venv`/uv로만 실행했는가(시스템 python 미사용)
- [ ] smoke 통과 후 본 학습을 돌렸는가
- [ ] metrics.json/config.snapshot/predictions가 표준 경로에 있는가
- [ ] seed·패키지 버전·실제 명령이 config.snapshot에 동결됐는가
- [ ] 실패 run도 status:failed로 기록되어 파이프라인이 멈추지 않았는가
