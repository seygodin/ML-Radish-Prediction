---
name: eval-and-report
description: 무 질병 실험 결과를 검증·평가하고 baseline vs Ours 비교 리포트를 만들 때 반드시 사용. 불균형 인지 지표(PR-AUC/F1/recall/precision/confusion, detection mAP·IoU 분포)를 독립 재계산하고, 데이터 누수·라벨 정합·split 위반을 교차검증하며, 비교 표·그림을 생성한다. "결과 평가", "메트릭 계산", "baseline 비교", "Ours 평가", "정합성 검증", "실험 리포트", "혼동행렬", "mAP" 요청 시 트리거. 학습 실행은 experiment-runner, 데이터 로더는 radish-data-pipeline 소관.
---

# eval-and-report (QA)

실험 결과를 **독립 검증**하고 **공정하게 비교·보고**한다. "파일이 있다"가 아니라 **경계면 교차검증**이 본질이다.

## 핵심: 경계면 교차검증 (왜 중요한가)
모델 지표는 데이터·예측·라벨이 어긋나도 그럴듯한 숫자를 낸다. 따라서 출처가 다른 산출물을 맞대어 본다:
- `experiments/*/predictions/` 의 라벨/표본 ↔ `_workspace/data/manifest_*.csv` 의 라벨/split → 라벨 인코딩·표본 수·split이 일치하는가.
- run이 보고한 `metrics.json` ↔ predictions로 **직접 재계산한 지표** → 불일치면 둘 다 표기.
- **데이터 누수 의심**: valid 이미지가 train에 섞이지 않았는가, 정상 박스를 질병으로 오집계하지 않았는가.

## 지표 표준 (불균형 인지)
`report/REPORT.md`의 함의와 일치시킨다.
- **분류**: accuracy 단독 금지. **PR-AUC(주지표), F1, 질병 recall·precision, confusion matrix**, 임계값 민감도 곡선. valid disease_4=24장 등 **소표본 지표에는 표본 수와 신뢰구간(예: Wilson)을 병기**.
- **detection**: mAP뿐 아니라 **(1) 이미지 단위 존재 검출**(질병 유무 분류로 환산)과 **(2) IoU 분포**를 분리 보고. 박스가 거칠다는 특성상 높은 mAP가 미세 병변 국소화를 뜻하지 않음을 해석에 명시.

## 점진적 QA (incremental)
전체 종료 후 1회가 아니라 **각 run 완료 직후** 검증한다. 이유: 데이터 누수·라벨 버그는 초반 run에서 잡아야 이후 실험 낭비를 막는다. 문제 발견 시 오케스트레이터에 즉시 보고해 책임 에이전트로 라우팅한다.

## 산출물 (계약)
1. **`_workspace/eval/verify_{name}.md`** — run별 검증: 통과/실패 항목, 재계산 지표 vs 보고 지표, 발견한 정합성 문제.
2. **`report/EXPERIMENTS.md`** — 종합 비교 리포트:
   - baseline 백본 4종 비교표(정상 vs d3, 정상 vs d4), baseline vs Ours, detection 결과.
   - 각 수치에 표본 수 병기, 한계·주의 명시. 과거 결과 보존(회귀 추적).
3. **`report/figures/exp_*.png`** — 비교 시각화(막대: 모델별 PR-AUC/F1; PR 곡선; confusion matrix; detection IoU 히스토그램). matplotlib Agg, dpi≥120, **영문 라벨**(폰트 안전), 기존 `data/analyze.py` 그림 스타일과 일관.
4. **학습 과정 곡선 (필수)** — 각 run마다 `metrics.json`의 per_epoch로 **train_loss vs val_loss를 한 그래프**에, 그리고 primary 지표(분류 PR-AUC/F1, detection presence@0.5/IoU)의 epoch별 곡선을 그린다. 저장: `report/figures/curves_<name>.png` (run별) + 백본·세팅별로 묶은 종합 패널 `report/figures/training_curves.png`. 과적합 징후(val_loss 반등, train-val 격차)를 리포트 본문에서 해석한다. val_loss가 per_epoch에 없으면 experiment-runner로 되돌려 보강한다(누락 시 곡선 불가).

## 작업 절차
1. 새 `experiments/<name>/`를 감지 → predictions·metrics·config.snapshot 로드.
2. manifest와 교차검증(라벨/표본/split) → 누수·오집계 검사.
3. predictions에서 지표 **직접 재계산**(임계값 0.5 기본 + PR-AUC는 임계값 무관) → 보고치와 대조.
4. `verify_{name}.md` 작성. 문제 있으면 즉시 보고.
5. 모든(또는 요청된) run이 검증되면 `EXPERIMENTS.md`와 비교 그림 갱신.

## 재계산 메모
- 분류: predictions의 확률·정답으로 sklearn 없이도 PR-AUC/F1 계산 가능(직접 구현 또는 venv에 설치). 임계값 스윕으로 recall-precision 트레이드오프 제시.
- detection: 예측 박스 vs GT의 IoU 계산 → IoU≥0.5 매칭으로 존재 검출 TP/FP/FN, IoU 분포 히스토그램.

## 검증 체크리스트
- [ ] predictions↔manifest 라벨/표본/split이 일치하는가
- [ ] 보고 지표를 독립 재계산해 대조했는가(불일치 시 둘 다 표기)
- [ ] 소표본 지표에 표본 수/신뢰구간을 병기했는가
- [ ] detection에서 존재 검출과 국소화 IoU를 분리 보고했는가
- [ ] EXPERIMENTS.md가 baseline·Ours를 동일 기준으로 비교하고 과거 결과를 보존하는가
