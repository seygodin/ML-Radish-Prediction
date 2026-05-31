---
name: radish-ml-orchestrator
description: 무(radish) 질병 분류·detection 프로젝트의 ML 워크플로우 전체를 조율할 때 반드시 사용. 데이터 파이프라인 → 모델 설계 → 학습 실행 → 평가·비교를 data-engineer/ml-researcher/experiment-runner/eval-reporter 서브에이전트로 분배·통합한다. "baseline 측정", "baseline 학습/평가", "Ours 모델 개발", "분류/detection 실험", "모델 비교", "전체 파이프라인 돌려", 그리고 후속 요청 "다시 실행", "재실행", "업데이트", "수정", "보완", "이전 결과 기반으로", "ablation 추가", "Ours 개선", "특정 단계만 다시" 시에도 트리거. 단순 데이터 질문이나 단일 단계 작업은 해당 전문 스킬을 직접 써도 된다.
---

# radish-ml-orchestrator

무 질병 프로젝트의 ML 실험을 **파이프라인 + 생성-검증(점진적 QA)** 으로 조율한다.
누가(에이전트) 언제 어떤 순서로 협업하는지를 정의한다. 각 단계의 "어떻게"는 전문 스킬이 담는다.

## 실행 모드: 서브에이전트 + 파일 기반 핸드오프 (하이브리드)
이 환경에는 팀 프리미티브(TeamCreate/SendMessage)가 없다. 따라서 **`Agent` 도구로 서브에이전트를 호출**하고, 단계 간 데이터는 **파일(`_workspace/`, `experiments/`, `report/`)로 전달**한다. 진행/의존 관리는 `TaskCreate`/`TaskUpdate`로 추적한다.

- 모든 Agent 호출은 **`model: "opus"`** 와 정의된 `subagent_type`(`data-engineer`/`ml-researcher`/`experiment-runner`/`eval-reporter`)을 명시한다.
- 독립적인 baseline run들은 `run_in_background: true`로 병렬화하되, 자원(GPU) 한계를 고려해 동시 실행 수를 제한한다.

## Phase 0: 컨텍스트 확인 (초기/후속/부분 재실행 판별)
시작 시 기존 산출물을 보고 실행 모드를 정한다.
- `_workspace/`·`experiments/` 없음 → **초기 실행** (Phase 1부터).
- 존재 + 사용자가 부분 수정 지목(예: "loss만", "convnextv2만 다시") → **부분 재실행** (해당 에이전트/단계만 호출).
- 존재 + 새 입력/대규모 변경 → **새 실행** (기존 `_workspace/`를 `_workspace_prev/`로 이동 후 진행, `experiments/`는 보존).
판별 결과와 실행 계획을 사용자에게 1줄로 요약 보고하고 진행한다.

## 파이프라인 (단계별 모드 명시)

### 단계 1 — 데이터 파이프라인  **(서브: data-engineer)**
- 입력: 목표 task(분류/detection), 입력 크기.
- 호출: `Agent(subagent_type="data-engineer", model="opus", ...)` → `src/data/`, `_workspace/data/manifest_*`, `data_card_*` 생성.
- 게이트: data_card에 공개 API와 스모크 테스트 통과가 있어야 다음 단계로.

### 단계 2 — 실험 설계  **(서브: ml-researcher)**
- 입력: `data_card_*`, 목표(baseline 측정 / Ours 개발 / ablation).
- 호출: ml-researcher → `_workspace/specs/exp_*.yaml`, `src/models/`, `design_notes.md`.
- 게이트: 각 exp yaml이 자기완결인지(runner가 단독 실행 가능) 확인.

### 단계 3 — 실험 실행  **(서브: experiment-runner, 병렬 가능)**
- 입력: `_workspace/specs/exp_*.yaml`.
- 호출: 각 사양마다 experiment-runner. 독립 run은 `run_in_background: true`로 병렬(GPU 수만큼). 각 run이 `experiments/<name>/`를 표준 구조로 생성.
- **smoke 우선**: 사양 `smoke: true`면 runner가 소규모 검증 후 본 학습.

### 단계 4 — 평가·검증·비교  **(서브: eval-reporter, 점진적)**
- **점진적 QA**: 각 run이 끝나는 즉시 eval-reporter를 호출해 `verify_{name}.md` 생성(누수·정합·재계산). 문제 발견 시 책임 에이전트(데이터→data-engineer, 학습→experiment-runner, 설계→ml-researcher)로 라우팅하여 재실행.
- 모든(또는 요청) run 검증 후 eval-reporter가 `report/EXPERIMENTS.md` + 비교 그림 갱신.

> baseline 측정 흐름: 1→2→3(4 백본 × 2 분류 + detection)→4.
> Ours 개발 흐름: (baseline 결과 존재 가정) 2(Ours 사양)→3→4(baseline 대비 비교). 데이터 변경이 필요하면 1부터.

## 데이터 전달 프로토콜
- **파일 기반(주)**: 단계 산출물은 약속된 경로에 쓴다. 중간물은 `_workspace/{data,specs,eval}/`, 실험은 `experiments/<name>/`, 최종 리포트는 `report/`.
- **반환값 기반(보조)**: 각 서브에이전트는 완료 시 핵심 요약(생성 경로, 주요 수치, 경고)을 반환 → 오케스트레이터가 다음 단계 입력으로 연결.
- **태스크 기반**: `TaskCreate`로 단계·run을 작업으로 등록하고 의존 관계(`addBlockedBy`)를 건다.
- 파일명 규칙: `_workspace/{phase}_{agent}_{artifact}` 또는 위의 표준 경로. 최종만 `report/`에 출력, 중간물은 감사용으로 보존.

## 에러 핸들링
- 서브에이전트/Run 실패 → **1회 재시도**. 재실패면 해당 산출물 없이 진행하고 **누락을 리포트에 명시**(`metrics.json status:failed`, EXPERIMENTS.md 주석).
- 상충 수치(보고 지표 vs 재계산) → 삭제하지 말고 **둘 다 출처와 함께 병기**.
- 게이트(스모크/자기완결/정합) 미통과 → 다음 단계로 넘어가지 않고 책임 에이전트로 되돌린다.
- 환경 문제(패키지 미설치) → experiment-runner가 uv로 설치, 그래도 실패면 사용자에게 보고.

## 팀 크기 / 동시성
- 핵심 에이전트 4종. 한 흐름에서 동시 활성은 보통 1~2 에이전트(파이프라인). 단계 3의 baseline run만 병렬.
- GPU 1장 가정 시 학습 run은 순차, 여러 장이면 장 수만큼 병렬.

## 후속 작업 지원
- 부분 재실행: "convnextv2만 다시" → 단계 3의 해당 run + 단계 4만. "loss 바꿔" → 단계 2(해당 yaml)→3→4.
- 새 입력: 기존 `_workspace/`를 `_workspace_prev/`로 이동 후 단계 1부터, `experiments/`는 회귀 비교용으로 보존.
- 모든 변경은 `CLAUDE.md`의 **하네스 변경 이력** 테이블에 기록한다.

## 테스트 시나리오
**정상 흐름 (baseline 측정):**
사용자: "convnextv2랑 efficientnetv2로 정상 vs 질병 baseline 측정해."
1. Phase 0 → 초기 실행 판별. 2. data-engineer로 분류 로더 생성(스모크 통과). 3. ml-researcher가 백본 2종 × 대상 사양 yaml 생성. 4. experiment-runner가 smoke 후 본 학습(병렬). 5. 각 run 직후 eval-reporter가 정합 검증 → 이상 없으면 EXPERIMENTS.md에 PR-AUC/F1 비교표·그림 생성. 6. 사용자에게 결과 요약 + 개선 피드백 요청.

**에러 흐름 (정합성 문제):**
단계 4에서 eval-reporter가 "valid 표본 수가 manifest와 불일치(누수 의심)"를 발견 → 오케스트레이터가 해당 run을 중지, data-engineer에 split 재검토를 라우팅, 수정 후 단계 3 해당 run만 재실행 → 재검증. 최종 리포트에 "초기 누수 발견·수정" 이력을 남긴다.

## 완료 후 (하네스 진화)
실행 완료 시 사용자에게 "결과/워크플로우에 바꿀 점이 있나요?"를 묻고, 피드백을 유형별로 반영(품질→스킬, 역할→에이전트, 순서→이 오케스트레이터, 트리거→description)한 뒤 CLAUDE.md 변경 이력을 갱신한다.
