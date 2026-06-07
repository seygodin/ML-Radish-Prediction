# 무(Radish) 질병 이미지 — 분류 & Detection

AI-Hub 스타일 무(radish) 작물 이미지에서 **정상(normal) vs 질병(disease)** 을 다루는 두 가지 비전 태스크의 baseline 측정 + 자체 모델("Ours") 개발 프로젝트.

- **분류(classification)**: 질병 종류별 — `normal vs disease_3`, `normal vs disease_4`, 그리고 3-class `normal/disease_3/disease_4`
- **Detection**: abnormal(질병) 이미지의 단일 거친 bounding box + objectness(이미지 단위 질병 검출)

이 README는 **프로젝트를 처음 받은 사람이 우리 실험 결과를 재현(reproduce)** 할 수 있도록 데이터 준비 → 학습 → 평가 → 데모까지 전 과정을 안내한다.

> 데이터셋·학습 체크포인트·가상환경은 용량 때문에 저장소에 포함되지 않는다(아래 [데이터](#-데이터-준비) 참조). 저장소에는 **코드·실험 사양(spec)·메트릭(metrics.json)·설정 스냅샷(config.snapshot)·리포트·그림**이 포함되어 그대로 재현·검증할 수 있다.

---

## 핵심 결과

전체 표·해석·하이퍼파라미터는 **[`report/EXPERIMENTS.md`](report/EXPERIMENTS.md)** (데이터 분석은 [`report/REPORT.md`](report/REPORT.md)) 참조.

**분류 (3-class `normal/d3/d4`, valid 원분포, 주지표 PR-AUC)** — 백본별 from-scratch baseline vs Ours(DINOv3 전이):

| 모델 | 학습 방식 | PR-AUC | F1-macro |
|------|----------|:------:|:--------:|
| ConvNeXtV2 / EfficientNetV2 / NeXtViT / DenseNet121 / ResNet50 / Vision-Mamba | from-scratch (최고: DenseNet121) | 0.570 | 0.709 |
| **Ours: DINOv3-S @256** (frozen + 2-layer head) | pretrained, head만 학습 | 0.689 | 0.698 |
| **Ours: DINOv3-B @512** (frozen + 2-layer head) | pretrained, head만 학습 | 0.745 | 0.750 |
| **Ours: DINOv3-B @512 + strong aug + focal** | pretrained, head만 학습 | **0.765** | **0.774** |

→ 가장 어려운 3-class에서 **Ours가 baseline 최고 대비 PR-AUC +34%**. 2-class는 baseline·Ours 모두 PR-AUC 0.95~1.0으로 포화. **Detection**은 이미지 단위 질병 검출이 전 모델 det PR-AUC ≈ 1.0(objectness가 정상≈0.0 / 질병≈0.95로 분리), 국소화 IoU median 0.57~0.67.

**Ours 추가 분석**(3-class):
- **Sensitivity (입력 노이즈 강건성)**: 정규화 입력에 `x + rand_like(x)·N_ratio`(N_ratio 0.1~0.5) 노이즈를 가해도 PR-AUC 저하 **최대 −2.4%** — frozen 백본+head 구조 덕에 매우 강건.
- **Stability (데이터량 스케일링)**: train_ratio 0.1→1.0에서 PR-AUC가 0.49→0.765로 **단조 증가하고 90%↑에서 포화**(수확체감). F1-macro는 소수 d4(24장) argmax 민감으로 일부 출렁.

> ⚠️ **해석 주의**: valid는 정상:질병 ≈ 13:1로 불균형이라 **accuracy·AUROC는 포화**되어 변별력이 약하다 — 우열은 **PR-AUC·F1-macro·precision**으로 판단한다. baseline은 from-scratch, Ours는 DINOv3 자기지도 pretrained(frozen)이라 **동일 조건 비교가 아님**에 유의. 소수 클래스 disease_4는 valid 24장으로 신뢰구간이 넓다.

![분류 7-메트릭 비교](report/figures/exp_metrics_table.png)
![Ours 개선 추이](report/figures/exp_ours_focal.png)
![노이즈 sensitivity](report/figures/exp_sensitivity_dinov3.png)
![train-ratio stability](report/figures/exp_stability_dinov3.png)

---

## 저장소 구조

```
.
├── baseline/              from-scratch 백본 구현 (ConvNeXtV2/EfficientNetV2/NeXtViT/MambaVision)
├── models/module.py       SAFEModule (백본 import용 placeholder)
├── src/
│   ├── data/              데이터 파이프라인: core/loaders/transforms/build_manifest (RAM 캐시·균형/train_ratio/aug 옵션)
│   ├── models/            분류기·detector 빌더 (build_classifier/build_detector), dinov3·timm·mamba 래퍼
│   ├── losses.py          FocalLoss
│   ├── metrics.py         불균형 인지 분류/detection 메트릭
│   ├── train.py           spec(yaml) 구동 학습/평가 진입점
│   ├── inference.py       데모용 추론 레지스트리
│   └── vqa.py             데모 VQA: whisper-base STT + SmolVLM (지연 로딩)
├── demo/                  FastAPI 데모 (app.py + static/index.html) — 다중 파이프라인 비교 + VQA
├── radish_demo.ipynb      한 노트북으로 환경설치→데이터압축해제·분석→학습→평가→데모 (self-contained)
├── data/                  *.py 스크립트만 포함 — 데이터 본체는 별도 준비(§데이터)
├── _workspace/
│   ├── specs/             실험 사양 (exp_*.yaml) — 재현의 단일 출처
│   ├── eval/              평가·리포트·그림 생성 스크립트 (run_*.py / make_*.py; sensitivity·stability 포함)
│   ├── data/              manifest_*.csv, data_card.md
│   └── launch_*.sh        다중 run 병렬 학습 런처
├── experiments/<name>/    학습 산출물 (metrics.json·config.snapshot만 커밋; 체크포인트는 .gitignore)
├── report/                EXPERIMENTS.md, FINAL_REPORT.md, REPORT.md, figures/, metadata.csv, stats.json
├── requirements.txt
└── CLAUDE.md              프로젝트 가이드 + 변경 이력
```

---

## 환경 설정

**요구사항**: Linux, NVIDIA GPU(CUDA), Python 3.10. 학습은 RTX PRO 6000 Blackwell(sm_120) + CUDA 12.8에서 검증됨. 패키지는 [`uv`](https://docs.astral.sh/uv/)로 격리한다.

```bash
# 1) uv 설치 (https://docs.astral.sh/uv/ 참고) 후 가상환경 생성
uv venv .venv

# 2) torch/torchvision — CUDA 12.8 wheel (다른 GPU면 pytorch.org에서 맞는 빌드 선택)
uv pip install --python .venv/bin/python \
    torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# 3) 나머지 의존성
uv pip install --python .venv/bin/python -r requirements.txt

# 4) 확인
./.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

> 이후 모든 명령은 **`./.venv/bin/python`** 으로 실행한다(시스템 Python 사용 금지). DINOv3 가중치는 timm을 통해 최초 사용 시 Hugging Face에서 자동 다운로드된다(인터넷 필요).

---

## 데이터 준비

데이터셋(AI-Hub 무 질병 이미지, 약 37 GB)은 저작권/용량 때문에 저장소에 없다. **8개 zip**을 아래 경로에 배치한 뒤 압축 해제한다:

```
data/train/[원천]무_0.정상.zip   data/train/[라벨]무_0.정상.zip
data/train/[원천]무_1.질병.zip   data/train/[라벨]무_1.질병.zip
data/valid/[원천]무_0.정상.zip   data/valid/[라벨]무_0.정상.zip
data/valid/[원천]무_1.질병.zip   data/valid/[라벨]무_1.질병.zip
```

```bash
# 압축 해제 (각 zip을 zip 이름의 디렉터리로)
cd data
for z in train/*.zip valid/*.zip; do unzip -nq "$z" -d "${z%.zip}"; done
cd ..

# 라벨↔이미지 1:1 매칭 검증 (전부 OK여야 함)
./.venv/bin/python data/verify_pairs.py

# 질병 종류별 분리(by_disease 심링크 생성)
./.venv/bin/python data/split_by_disease.py

# (선택) bbox 시각화, 데이터 분석 리포트
./.venv/bin/python data/draw_bbox.py
./.venv/bin/python data/analyze.py          # report/REPORT.md, report/figures, stats.json 재생성
```

**라벨 스키마 / 검증된 데이터 함정**(코드가 이미 방어함)은 `CLAUDE.md`·`report/REPORT.md` 참조 — 핵심: JSON의 width/height가 0인 라벨 43건(실제 이미지에서 크기 읽음), EXIF 회전, 확장자 대소문자 혼재, 정상:질병 13:1 불균형(train은 다운샘플로 균형).

**Manifest**: 저장소에 `_workspace/data/manifest_*.csv`가 포함되어 있어, 위처럼 이미지를 추출하면 경로가 맞아 바로 학습/평가가 된다. 처음부터 재생성하려면(전체 스캔, 수 분 소요):

```bash
./.venv/bin/python -m src.data.build_manifest
```

데이터 파이프라인 스모크 테스트: `./.venv/bin/python -m src.data.loaders`

---

## 빠른 시작: 노트북 (가장 쉬움, self-contained)

**`radish_demo.ipynb`** 한 파일이 **환경 설치(`%pip`) → 데이터 압축해제·정리·분석(EDA) → 학습 → 평가 → 데모**를 모두 수행한다(무거운 작업은 내부에서 `sys.executable` 서브프로세스로). 별도의 `.venv`가 없어도 되고, **데이터 경로(`DATA_DIR`)만 지정하면 zip을 직접 풀고 by_disease·manifest까지 정리**한다.

```bash
# (jupyter가 없으면) 커널만 준비
uv pip install --python .venv/bin/python jupyterlab ipykernel
./.venv/bin/python -m ipykernel install --user --name python3 --display-name "Python 3 (.venv)"
./.venv/bin/jupyter lab    # radish_demo.ipynb 열고 위에서부터 실행
```

- §0가 의존성을 현재 커널에 설치, §1이 `DATA_DIR`의 zip을 압축 해제·정리·EDA, §2에 **전체 42개 spec이 그룹별로 나열**(불필요한 줄 `#` 주석처리), §3~6이 평가·그림·데모.
- 모든 단계 idempotent(이미 된 건 건너뜀). 새 데이터셋이면 §1의 `REBUILD_MANIFEST=True`.

아래는 노트북 없이 CLI로 단계별 실행하는 방법이다.

## 재현: 학습

학습은 **사양 yaml 하나**로 완결된다(`_workspace/specs/exp_*.yaml`에 모델·데이터·loss·optimizer·metric·seed가 동결). 예:

```bash
# 단일 run
./.venv/bin/python -m src.train --spec _workspace/specs/exp_dinov3_base_focal_normal_d3_d4.yaml --device cuda:0

# 빠른 스모크(2 epoch, subset) — 파이프라인 무결성 확인
./.venv/bin/python -m src.train --spec _workspace/specs/<name>.yaml --smoke --device cuda:0
```

산출물은 `experiments/<spec.name>/`에 표준 구조로 저장된다: `metrics.json`(에폭별·최종 지표), `config.snapshot`(사양+패키지버전+명령+seed), `checkpoints/best.pt`, `predictions/valid.*`. 모든 run은 **seed=42 고정**.

**전체 재현**(42개 spec). GPU 여러 장이면 병렬 런처를 쓰거나, 단순 루프:

```bash
# 예: 모든 spec을 GPU 0에서 순차 실행
for s in _workspace/specs/exp_*.yaml; do
  ./.venv/bin/python -m src.train --spec "$s" --device cuda:0
done
# GPU 8장 병렬 예시는 _workspace/launch_all.sh / launch_new16.sh 참고(GPU 배정 수정해 사용)
```

실험 그룹:
- **분류 baseline (from-scratch)** — `exp_{convnextv2,efficientnetv2,nextvit,nextvit20,densenet121,resnet50,mamba}_{normal_vs_d3,normal_vs_d4,normal_d3_d4}.yaml`
- **Detection baseline** — `exp_<backbone>_detection_singlebox.yaml`
- **Ours** — `exp_dinov3_{normal_*}.yaml`(S@256), `exp_dinov3_base_{...}.yaml`(B@512), `exp_dinov3_base_focal_{...}.yaml`(+증강+focal), `exp_dinov3_base_detection_singlebox.yaml`
- **Ablation** — `exp_dinov3_base_{augonly,focalonly,focalg1,focalg3}_normal_d3_d4.yaml`

> 학습 속도: 데이터 로더가 디코드+리사이즈를 1회만 하고 RAM 캐시(워커 fork copy-on-write)하므로, 캐시 빌드(run당 ~30–60s) 후 epoch는 수 초다. DINOv3 frozen 모델은 head(수십만 파라미터)만 학습돼 매우 빠르다.

---

## 재현: 평가 & 리포트

평가 스크립트는 `experiments/*/predictions/`(또는 best.pt 재로드)에서 **지표를 독립 재계산**하고 `experiments/*/metrics.json`과 대조한 뒤 `report/EXPERIMENTS.md`의 표·그림을 재생성한다.

```bash
# baseline 분류·detection (원분포 valid) → §1, §3, 비교 그림
./.venv/bin/python _workspace/eval/run_eval.py

# 균형(1:1 / 1:1:1) valid 재평가 → §1B, §3B
./.venv/bin/python _workspace/eval/run_balanced_eval.py
./.venv/bin/python _workspace/eval/run_balanced_detection_eval.py

# Ours(DINOv3) 분류·detection 비교 + 20% 목표 판정 → §6
./.venv/bin/python _workspace/eval/run_ours_plus_eval.py
./.venv/bin/python _workspace/eval/run_ours_focal_eval.py
./.venv/bin/python _workspace/eval/run_ours_detection_eval.py

# Ablation (증강 × focal, gamma sweep) → §6B
./.venv/bin/python _workspace/eval/run_ablation_eval.py

# Ours 추가 분석 (재학습 없이 best.pt 사용) → §7 Sensitivity, §8 Stability
./.venv/bin/python _workspace/eval/run_sensitivity_eval.py   # 입력 노이즈 N_ratio 0~0.5
./.venv/bin/python _workspace/eval/run_stability_eval.py     # train_ratio 0.1~1.0
```

> Stability는 비율별 재학습이 필요하다: `exp_dinov3_base_focal_r{10,30,50,70,90}_normal_d3_d4.yaml`을 먼저 학습한 뒤 `run_stability_eval.py`로 곡선을 만든다. Sensitivity는 기존 Ours best.pt에 노이즈만 주입(재학습 불필요).

검증 결과(누수·정합·재계산 대조)는 `_workspace/eval/verify_*.md`에 run별로 남는다.

**평가 프로토콜**(요약): 분류는 PR-AUC(주)·F1-macro·recall/precision(macro)·AUROC·accuracy + train/val loss, detection은 det PR-AUC·presence_recall@0.5·fp_rate@0.5·mAP@0.5·IoU. valid는 **원분포**와 **클래스 균형(1:1 / 1:1:1)** 두 가지로 모두 평가한다. 지표 정의는 `report/EXPERIMENTS.md` 상단 "메트릭 정의" 절 참조.

---

## 데모 (FastAPI)

이미지를 업로드하거나 valid 셋에서 골라, **여러 파이프라인(백본·세팅)으로 동시에 분류+detection 결과를 비교**하는 웹 데모.

```bash
./.venv/bin/python -m uvicorn demo.app:app --host 0.0.0.0 --port 8000
# 브라우저: http://<host>:8000
```

시작 시 `experiments/`의 모든 run(체크포인트 보유)을 파이프라인으로 로드한다(순수 ablation run은 제외). REST API: `GET /api/pipelines`, `GET /api/valid-images`, `POST /api/predict`(파일 업로드 또는 `valid_image_id` + `pipelines`). 데모를 쓰려면 학습 체크포인트(`experiments/*/checkpoints/best.pt`)가 있어야 한다(위 학습 단계 수행).

**VQA (음성/텍스트 질의응답)**: 패널에서 **마이크 녹음·오디오 파일 업로드(wav/mp3/ogg/flac)·텍스트** 중 하나로 질문하면 **whisper-base**가 STT(ffmpeg 없이 soundfile로 디코딩)하고 **SmolVLM-500M**이 선택 이미지에 답한다. `POST /api/vqa`(이미지 + `audio` 또는 `question`). whisper/SmolVLM은 첫 호출 시 지연 로딩(HF 다운로드). 범용 VLM이라 무 질병 특화는 아니며 보조 설명용이다.

**Streamlit 버전**: 동일 기능(다중 파이프라인 비교 + 검출 박스 오버레이 + VQA)을 Streamlit으로도 제공한다(`src.inference`·`src.vqa` 재사용).

```bash
./.venv/bin/streamlit run demo/streamlit_app.py --server.port 8502
# 브라우저: http://<host>:8502
```

마이크 녹음은 `st.audio_input`, 오디오 파일 업로드/텍스트 질문 모두 지원. FastAPI 버전과 동일하게 학습 체크포인트가 필요하다.

---

## 방법 요약

- **Baseline**: `baseline/`의 백본 6종을 공통 분류 헤드로 래핑해 **from-scratch** 학습(공정 비교). Detection은 동일 백본의 풀링 특징 → 단일 박스 회귀 + objectness 헤드.
- **클래스 균형**: train은 정상을 다운샘플해 type 간 1:1(2-class)·1:1:1(3-class). valid는 원분포 + 균형 둘 다로 평가.
- **Ours**: **DINOv3 ViT(frozen) + 2-layer MLP 헤드**. 백본을 freeze해 사전학습 지식을 보존(catastrophic forgetting 없음)하고 헤드(수십만 파라미터)만 학습 → 작고 빠르며, 소표본·단일시즌 데이터에서 from-scratch를 크게 능가. ViT-S@256 → ViT-B@512로 키우고 **강한 증강 + focal loss(γ=2)** 를 더해 성능을 최대화.
- **Ablation 결론**(3-class): 강한 증강이 주 동력, focal 단독(기본 증강)은 소폭 하락하지만 **증강+focal 조합에서 양의 시너지**.
- **Sensitivity/Stability**(Ours): 입력 노이즈에 강건(PR-AUC 저하 ≤2.4%), PR-AUC가 train 데이터량에 단조 증가·90%↑ 포화.
- **알려진 한계**: 거친 단일 bbox(병변 핀포인트 아님)·정상에도 박스 존재, 단일 시즌 데이터, valid disease_4 24장 소표본, 불균형으로 인한 accuracy/AUROC 포화 → PR-AUC/F1 중심 해석.

**Ours 알고리즘 수도코드**는 `report/FINAL_REPORT.md` §2.6, 전체 하이퍼파라미터 표는 `report/EXPERIMENTS.md` **부록 A**, 설계 노트는 `_workspace/specs/design_notes.md` 참조. 종합 보고서는 `report/FINAL_REPORT.md`(Introduction/Method/Experiments/Discussion/Conclusion).

---

## 데이터 출처 / 라이선스

데이터셋은 AI-Hub 무(radish) 작물 질병 이미지로, 별도 약관에 따라 제공처에서 직접 취득해야 하며 본 저장소에 포함되지 않는다. 코드 사용 시 해당 데이터 라이선스를 준수할 것. (코드 라이선스는 저장소 정책에 따름.)
