# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 하네스: 무 질병 ML 실험

**목표:** 데이터 파이프라인 → 모델 설계 → 학습 실행 → 평가·비교를 전문 에이전트 팀으로 조율해, baseline 측정과 "Ours" 모델 개발을 재현 가능하게 수행한다.

**트리거:** baseline 측정/학습/평가, Ours 모델 개발, 분류·detection 실험, 모델 비교, 전체 파이프라인 실행, 그리고 그 후속(재실행/수정/ablation/부분 재실행) 요청 시 `radish-ml-orchestrator` 스킬을 사용하라. 단순 데이터 질문이나 단일 단계 작업은 해당 전문 스킬을 직접 써도 된다.

**변경 이력:**
| 날짜 | 변경 내용 | 대상 | 사유 |
|------|----------|------|------|
| 2026-05-30 | 초기 구성 (에이전트 4 + 오케스트레이터, 서브에이전트/파일 기반 모드) | 전체 | - |
| 2026-05-30 | baseline 프로토콜 확정: 분류 3세팅(정상vsd3 / 정상vsd4 / 3-class), normal 다운샘플로 type 균등화, 분류·detection 모두 baseline/ 백본 그대로 사용 | model-design, radish-data-pipeline | 사용자 baseline 사양 확정 |
| 2026-05-30 | 학습 과정 시각화 정식화: metrics.json per_epoch에 train_loss+val_loss 필수, eval 리포트에 train/val loss·지표 학습 곡선 산출물 추가 | experiment-runner, eval-and-report | "학습 과정을 볼 수 있는 결과물 필요" 피드백 |
| 2026-05-30 | 데이터 로더 RAM 캐시(디코드 1회) 도입 — epoch 104s→~1.8s, GPU util 0%→55–87% | src/data, radish-data-pipeline | GPU util 0 지속 피드백(디코드 병목) |
| 2026-05-30 | detection objectness collapse 수정: 정상=음성(빈 박스), 지표를 이미지단위 질병검출(PR-AUC/ROC/presence/fp)+양성 IoU로 재정의, 3 run 재학습·재평가 (nextvit fp32) | src/data, src/train, src/metrics, specs | 정상 objectness≈1.0 붕괴 발견(eval QA) |
| 2026-05-30 | 백본 4종 추가(densenet121·resnet50=timm, nextvit20=NeXtViT base, mamba=mambapy Vision-Mamba; from-scratch), AUROC+7메트릭(acc/train_loss/val_loss/recall/precision/f1/auroc) 도입, 28 run 재평가·리포트 갱신, 데모 28 파이프라인 | src/models, src/train, src/metrics, specs, eval, demo | 사용자 백본·메트릭 요구 |
| 2026-05-30 | FastAPI 데모 추가: 업로드/valid선택→다중 파이프라인 동시 비교(분류+detection) | demo/, src/inference.py | 사용자 데모 요청 |
| 2026-05-30 | 메트릭 정의 섹션·모델 params 컬럼 추가, mamba 라벨 명확화 | report/EXPERIMENTS.md, eval | 사용자 요청 |
| 2026-05-30 | 균형 valid 평가 추가: 분류 1:1/1:1:1(§1B), detection 1:1(§3B) — best 모델 로드 재측정, balance_valid 옵션 | src/data, eval, report | 사용자 "valid도 균형 평가" 요청 |
| 2026-05-30 | 리포트에서 nextvit20 제외(산출물 보존), 6백본·24run으로 카운트/표/그림 재생성 | report, eval | 사용자 요청 |
| 2026-05-30 | Ours 모델 추가: DINOv3 ViT-S/16 frozen + 2-layer head(헤드만 학습=forgetting 없음). 3-class 원분포 PR-AUC +20.8%(목표 달성), F1-macro는 미달 | src/models/dinov3.py, src/train.py, specs, eval, report §6 | 사용자 "DINOv3 기반 +20% 모델" 요청 |
| 2026-05-30 | Ours+ 강화: DINOv3 ViT-B/16 frozen @512(dinov3_base). 3-class 원분포 PR-AUC 0.745=baseline 대비 +30.7%, F1 +5.8%(절대목표 0.851은 d4 precision 병목으로 미달). small 전 지표 상회 | src/models/dinov3.py, specs, eval, report §6 | 사용자 "base+512로 성능 최대화" 요청 |
| 2026-05-31 | 분류 강한 증강(aug=strong) + focal loss(gamma2) 추가. dinov3_base_focal: 3-class PR-AUC 0.765(baseline 대비 +34.1%)·F1 0.774(+9.2%), d4 precision 0.339→0.396 개선(절대 F1목표는 d4 N=24로 미달) | src/data/transforms.py, src/losses.py, src/train.py, specs, eval, report §6 | 사용자 "증강+focal로 개선" 요청 |
| 2026-05-31 | 실험별 하이퍼파라미터 표(부록A, 37 run) + aug×focal ablation·gamma sweep(§6B) 기록. 결론: 강한 증강이 주동력, focal 단독은 소폭↓, 조합 시너지 | report §6B·부록A, eval, specs | 사용자 "hyperparameter 저장 + ablation" 요청 |
| 2026-05-31 | §1·§1B 분류표에 Ours(dinov3 small/base/focal) 행 추가(pretrained-frozen 구분 주석). DINOv3-B frozen detection 추가·학습→§3/§3B Ours 행. 데모 재기동(38 파이프라인, dinov3 10개 반영, ablation 제외) | report §1/§1B/§3/§3B, src/models/detector.py, src/inference.py, demo | 사용자 "Ours를 detection 표·데모에 반영" 요청 |
| 2026-06-02 | 데모에 VQA 추가: whisper-base STT(ffmpeg 없이 WAV→배열) + SmolVLM-500M VQA. POST /api/vqa(이미지+오디오/텍스트), 프론트 VQA 패널(브라우저 WAV 녹음). 라이브 검증 HTTP 200 | src/vqa.py, demo/app.py, demo/static/index.html, requirements | 사용자 VQA 요청 |
| 2026-06-02 | Sensitivity 분석(§7): Ours(dinov3_base_focal 3-class) best.pt forward 전용, 정규화 입력텐서에 노이즈 x+rand_like(x)*N_ratio(0~0.5). 매우 강건 — PR-AUC 저하 최대 −2.4%, clean(0.0)은 §6와 일치 | report §7, _workspace/eval/run_sensitivity_eval.py | 사용자 "노이즈 sensitivity 분석" 요청 |
| 2026-06-07 | Streamlit 데모 추가(demo/streamlit_app.py): src.inference·src.vqa 재사용, 다중 파이프라인 비교+박스 오버레이+VQA(st.audio_input). 로직 검증 완료, 서버 부팅은 로컬에서 | demo/streamlit_app.py, requirements, README | 사용자 "streamlit 버전 데모" 요청 |
| 2026-06-07 | report/PAPER.md 작성: 외부 공개용 단일 자립 논문(goal~reference 9섹션+부록A/B, 그림 8개 base64 내장, 내부경로 참조 0). 서론 깔때기화 + Related Work 농업AI 실논문 인용([22]~[29]) | report/PAPER.md | 사용자 "paper final report / 공개용 정리 / 농업AI related work" 요청 |
| 2026-06-07 | §6.8 Explainability 추가: 예시 normal+abnormal(d3·d4) Ours detection 시각화(forward-only) + VQA 영어 질병특성 QA 예시. det objectness 0.026/0.978/0.950 정확분리, VQA는 보조설명 한정 | report/PAPER.md, _workspace/eval/run_explainability.py, report/figures/exp_explainability_detection.png | 사용자 "설명가능성 실험(detection 시각화+VQA QA)" 요청 |
| 2026-06-07 | §6.9 XAI 추가: Ours(dinov3_base_focal 3-class) Grad-CAM(frozen 백본→input requires_grad로 grad 흘림, blocks[-1].norm1 32×32 reshape)+LIME+SHAP. 세 기법 모두 무 잎·병변에 근거 집중(DINO 표현 정렬 교차검증). XAI 의존성 추가 | report/PAPER.md, _workspace/eval/run_xai.py, report/figures/exp_xai_dinov3.png, requirements | 사용자 "Grad-CAM/LIME/SHAP로 DINO 이해도 확인" 요청 |
| 2026-06-07 | PAPER.md 그림을 base64 내장→외부 figures/*.png 상대참조로 전환(GitHub은 data:URI 미렌더·대용량 md raw표시 → 결과 안 보임). §6.9에 Grad-CAM/LIME/SHAP 결과 정성표 추가. 공개 시 PAPER.md+report/figures/ 동반 | report/PAPER.md, CLAUDE.md | 사용자 "PAPER.md에 XAI 결과가 안 보임, 반영" 요청(그림 외부참조 선택) |
| 2026-06-07 | Related Work를 4하위섹션 재구성(§4.1 농업AI 7 / §4.2 AI기술 7 / §4.3 설명가능AI 7 / §4.4 차별성). XAI 참고문헌 7개 신규([30]Grad-CAM~[36]DINO), §6.9에 [30][31][32] 인용. 36개 참조 전부 인용·정의 일치 검증 | report/PAPER.md, CLAUDE.md | 사용자 "related work 3그룹×7+차별성" 요청 |
| 2026-06-07 | §5.0 전체 파이프라인 개요 추가: 사용자 제공 radish_overview.png(DINOv3 사전학습→전이학습 Linear Probing/Fine-tuning→head→예측→평가) 삽입 + 흐름 설명, Ours는 (A)Linear Probing 채택 명시 | report/PAPER.md(§5.0), report/figures/radish_overview.png | 사용자 "전체 플로우 그림+설명 추가" 요청 |

## Project goal

Train models to distinguish **normal (정상)** vs **diseased (질병)** radish (무) images on an
AI-Hub-style crop-disease dataset. Two ML tasks share the same data:

1. **Binary classification per disease type** — normal vs each disease class.
2. **Object detection on abnormal (diseased) cases** — localize the diseased region via bounding box.

Planned deliverables: data analysis & visualization → **baseline** model performance measurement →
**Ours** (proposed) model development. When writing code, keep the data-pipeline, baseline, and "Ours"
work separable so baselines and the proposed model can be compared on identical splits/metrics.

## Repository state

The repo holds the dataset (`data/`), a few analysis scripts under `data/`, and a generated
data-analysis report under `report/`. No build system or tests yet, and no git repository.

### Environment

System Python has **no pip/ensurepip**; use the project venv created with `uv`:
`./.venv/bin/python` (matplotlib, numpy, pandas, pillow installed). Recreate with
`uv venv .venv && uv pip install --python .venv/bin/python matplotlib numpy pandas pillow`.

### Existing scripts & artifacts (all idempotent, run from repo root)

- `data/verify_pairs.py` — confirms every label JSON pairs 1:1 with a source image.
- `data/split_by_disease.py` — relative-symlinks diseased data into
  `data/by_disease/<split>/disease_{3,4}/{images,labels}/`.
- `data/draw_bbox.py` — writes bbox-annotated copies into `.../disease_*/image_w_bbox/`.
- `data/analyze.py` → `report/REPORT.md` + `report/figures/*.png` + `report/stats.json` +
  `report/metadata.csv`. See `report/REPORT.md` for the dataset analysis and modeling implications.

### Data-quality facts that affect code (verified)

- **43 diseased labels have `width=height=0` in their JSON `description`.** Do NOT trust JSON
  dimensions — read real size from the image file (and normalize bboxes against it), or you get
  divide-by-zero / wrong bbox scaling.
- **Exactly one bbox per image** (single-object detection), and boxes are **coarse**: median ≈50%
  of image area, centered. Normal images also carry a (whole-radish) box, so box presence does not
  separate normal vs disease.
- Resolutions and orientations are **mixed** (720×960 up to 6000×4000) — resize/normalize and apply
  EXIF-orientation handling before training.

## Dataset layout (`data/`)

Split into `train/` and `valid/`. Each split has paired **source-image** and **label** directories,
distributed as `.zip` files (most large image zips are **not yet extracted** — see below):

| Kind | Train dir | Valid dir | Contents |
|------|-----------|-----------|----------|
| Source images | `[원천]무_0.정상`, `[원천]무_1.질병` | `[원천]무_0.정상`, `[원천]무_1.질병` | `.jpg` / `.JPG` |
| Labels | `[라벨]무_0.정상`, `[라벨]무_1.질병` | `[라벨]무_0.정상`, `[라벨]무_1.질병` | `.json` (one per image) |

- `[원천]` = source images, `[라벨]` = labels. `0.정상` = normal, `1.질병` = diseased.
- **Label↔image pairing:** a label file is named `<image_filename>.json`, e.g.
  `V006_..._S01_1.jpg.json` → image `V006_..._S01_1.jpg`. Strip the trailing `.json` to get the image name.
  Pairing is verified **complete and 1:1** (no orphans) by `data/verify_pairs.py` — run it after re-extracting.
- Exact counts (label = image, all matched): train **11,001 normal / 697 diseased**,
  valid **1,303 normal / 100 diseased** — **heavily imbalanced** toward normal.
  Account for this in sampling/loss/metrics.
- Image filenames have case-varying extensions (`.jpg` and `.JPG`); match case-insensitively.

### Label JSON schema

```jsonc
{
  "description": {
    "image": "<filename>.jpg",
    "date": "YYYY/MM/DD",
    "height": 960, "width": 720,   // image dims — varies (e.g. 720x960 normal, 3024x3024 diseased)
    "task": 79, "type": 0,         // type: 0 = normal, 1 = diseased
    "region": null
  },
  "annotations": {
    "disease": 0,                  // 0 = normal; nonzero = disease class id (radish: 3 and 4 observed)
    "crop": 2,                     // 2 = radish (무)
    "area": 3, "grow": 11,         // area / growth-stage codes
    "risk": 0,                     // severity: 0 = none, 1/2/3 = increasing
    "points": [ { "xtl": ., "ytl": ., "xbr": ., "ybr": . } ]   // bounding box(es), top-left/bottom-right
  }
}
```

- For **classification**, the target is `annotations.disease` (0 = normal, else disease-class label).
- For **detection**, use `annotations.points` (a list of `xtl,ytl,xbr,ybr` boxes). Boxes exist for both
  normal (whole-radish crop) and diseased (lesion region) images, so filter by `disease`/`type` when
  building the detection set of abnormal cases.
- Disease classes observed for radish so far: `3` and `4` (binary-classification models are built
  per disease type, i.e. normal-vs-class-3, normal-vs-class-4).

## Working with the data

Image data lives mostly in the `.zip` files; the large `[원천]` image zips are gigabytes
(train normal source ≈ 15 GB). Extract what you need rather than all of it, e.g.:

```bash
# extract a zip into a dir named after the zip stem (zips are flat, no internal folder)
unzip -nq "data/train/[원천]무_1.질병.zip" -d "data/train/[원천]무_1.질병"
unzip -nq "data/train/[라벨]무_1.질병.zip" -d "data/train/[라벨]무_1.질병"
```

The 8 `.zip` files are the source of truth; extracted dirs can be regenerated. The big `[원천]` image
zips are gigabytes (train normal ≈ 15 GB) — extract in the background. After re-extracting, run
`python3 data/verify_pairs.py` to confirm every JSON still pairs with an image.

Korean directory names contain brackets and non-ASCII — always quote paths, and in Python use
`glob.escape()` (the `[...]` brackets are otherwise parsed as glob character classes).
