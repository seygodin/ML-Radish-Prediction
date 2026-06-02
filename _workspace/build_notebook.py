#!/usr/bin/env python3
"""Generate radish_demo.ipynb — self-contained: env install + EDA + train + eval + demo.
Assumes ONLY the dataset is prepared (images/labels extracted under data/)."""
import json

cells = []
def md(t): cells.append({"cell_type":"markdown","id":f"c{len(cells):02d}","metadata":{},"source":t.strip("\n").splitlines(keepends=True)})
def code(t): cells.append({"cell_type":"code","id":f"c{len(cells):02d}","metadata":{},"execution_count":None,"outputs":[],"source":t.strip("\n").splitlines(keepends=True)})

md("""# 무(Radish) 질병 — 환경설정 · 데이터분석 · 학습 · 평가 · 데모 (self-contained)

이 노트북은 **데이터 경로(`DATA_DIR`)만 주면** 나머지를 모두 자동 수행합니다: **환경 설치 → (zip 자동 압축해제·정리) 데이터 준비 & 분석(EDA) → 학습 → 평가 → 데모**.

- §0에서 의존성을 **현재 커널 환경에 직접 설치**(`%pip`)하므로 별도의 `.venv` 준비가 필요 없습니다. 무거운 작업은 `sys.executable`(이 커널의 파이썬) **서브프로세스**로 실행합니다.
- §1에서 `DATA_DIR`의 zip을 **직접 압축 해제하고 by_disease·manifest까지 정리**합니다(이미 돼 있으면 건너뜀).
- 학습 §2에는 **전체 실험(42개)이 그룹별로 모두 나열**되어 있습니다 — **불필요한 줄은 `#`로 주석처리**해 원하는 것만 학습하세요.

**전제**: `DATA_DIR`에 AI-Hub 무 데이터(8개 zip 또는 추출된 `train/`·`valid/`)가 있음. GPU(CUDA) 권장.""")

md("""## 0. 환경 설정 (이 셀이 모든 의존성을 설치)

`%pip`로 현재 커널에 설치합니다. **torch는 CUDA 12.8 빌드**(RTX PRO 6000 Blackwell에서 검증)이며, 다른 GPU/CUDA면 [pytorch.org](https://pytorch.org)에서 맞는 빌드의 `--index-url`로 바꾸세요. (최초 1회 수 분 소요, 이미 설치돼 있으면 빠르게 통과.)""")

code('''
# §0a. 의존성 설치 (현재 커널 환경)
# torch/torchvision — CUDA 12.8 wheel index (GPU에 맞게 index-url 변경 가능)
%pip install -q torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
# 나머지 의존성 (한 줄로 — 매직은 줄바꿈 연속 불가)
%pip install -q "timm==1.0.27" einops mambapy numpy pandas pillow scikit-learn pyyaml matplotlib fastapi "uvicorn[standard]" python-multipart
print("의존성 설치 완료")
''')

code('''
# §0b. 공통 설정 — repo 루트 탐색, 파이썬(현재 커널) 지정, 헬퍼, 버전 점검
import os, sys, subprocess, json, glob, time
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / "src" / "train.py").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
assert (ROOT / "src" / "train.py").exists(), "repo 루트를 찾지 못했습니다. 노트북을 repo 안에서 여세요."
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
PY = sys.executable                     # 이 노트북 커널의 파이썬 (= 위에서 설치한 환경)
print("repo root :", ROOT)
print("python    :", PY)

def run(cmd, **kw):
    "서브프로세스 실행 + 실시간 출력. cmd는 리스트."
    print("$", " ".join(str(c) for c in cmd)); sys.stdout.flush()
    return subprocess.run(cmd, **kw)

import torch, timm
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "| device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
      "| timm", timm.__version__)
''')

md("""## 1. 데이터 준비 & 분석 (EDA)

**`DATA_DIR`만 지정**하면 노트북이 알아서 처리합니다: (zip만 있으면) **압축 해제 → 1:1 매칭 검증 → 질병종류별 분리(by_disease) → manifest 생성 → 데이터 분석 리포트(EDA)**.

- `DATA_DIR`은 `train/`·`valid/`(각각 `[원천]·[라벨]무_*` zip 또는 추출된 폴더 포함)를 담은 데이터 루트입니다. repo의 `data/`와 다르면 자동으로 심링크해 파이프라인이 그대로 동작합니다.
- 모든 단계는 **idempotent**(이미 추출/생성됐으면 건너뜀). 새 데이터셋이면 `REBUILD_MANIFEST=True`로 manifest를 다시 만드세요.""")

code('''
# §1-0. 데이터 경로 설정 — DATA_DIR이 repo의 data/와 다르면 심링크로 연결
DATA_DIR = str(ROOT / "data")     # ← train/ valid/ 를 담은 데이터 루트. 다른 경로면 여기를 바꾸세요.

src_dir = Path(DATA_DIR).expanduser().resolve()
assert src_dir.exists(), f"DATA_DIR 없음: {src_dir}"
target = ROOT / "data"
if src_dir != target.resolve():
    if target.is_symlink():
        target.unlink(); target.symlink_to(src_dir)
    elif not target.exists():
        target.symlink_to(src_dir)
    elif not any(target.iterdir()):          # 비어있는 실제 폴더면 교체
        target.rmdir(); target.symlink_to(src_dir)
    else:
        raise RuntimeError(f"{target} 가 비어있지 않은 실제 폴더라 심링크 불가. DATA_DIR을 data/로 두거나 data/를 비우세요.")
    print(f"심링크 연결: {target} -> {src_dir}")
else:
    print("DATA_DIR = repo data/ (그대로 사용)")
for sp in ("train", "valid"):
    d = f"data/{sp}"
    print(f"  {d}:", sorted(os.listdir(d))[:8] if os.path.isdir(d) else "(없음)")
''')

code('''
# §1a. zip 자동 압축 해제 (idempotent) — data/{train,valid}/*.zip -> 같은 이름 폴더
import zipfile
zips = sorted(glob.glob("data/train/*.zip") + glob.glob("data/valid/*.zip"))
print(f"발견된 zip: {len(zips)}개")
for z in zips:
    outdir = z[:-4]                                   # '<name>.zip' -> '<name>/'
    if os.path.isdir(outdir) and any(os.scandir(outdir)):
        print(f"  [skip] {os.path.basename(z)} (이미 추출됨)"); continue
    os.makedirs(outdir, exist_ok=True)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(outdir)
    print(f"  [extract] {os.path.basename(z)} -> {outdir} ({len(os.listdir(outdir))} files)")
if not zips:
    print("zip 없음 — 이미 추출돼 있다고 가정")
''')

code('''
# §1b. 라벨↔이미지 1:1 매칭 검증
run([PY, "data/verify_pairs.py"])
''')

code('''
# §1c. 질병 종류별 분리(by_disease) + manifest 생성/재생성
run([PY, "data/split_by_disease.py"])

REBUILD_MANIFEST = False     # 새(다른) 데이터셋이면 True — 전체 스캔(수 분)으로 manifest 재생성
need = REBUILD_MANIFEST or not glob.glob("_workspace/data/manifest_classification.csv")
if need:
    run([PY, "-m", "src.data.build_manifest"])
else:
    print("manifest 존재 → 재생성 생략 (새 데이터면 REBUILD_MANIFEST=True)")
''')

code('''
# §1d. 데이터 분석 리포트 생성 (report/REPORT.md, report/figures/01~11, stats.json, metadata.csv)
run([PY, "data/analyze.py"])
''')

code('''
# §1e. EDA 핵심 통계 + 그림 표시
from IPython.display import Image, Markdown, display
if os.path.exists("report/stats.json"):
    s = json.load(open("report/stats.json"))
    keys = ["total_samples","by_split_class","class_imbalance_normal_to_disease",
            "disease_code_counts","risk_distribution_disease","labels_with_zero_dims_in_json","date_range"]
    display(Markdown("**데이터 요약 (report/stats.json)**"))
    for k in keys:
        if k in s: print(f"  {k}: {s[k]}")
for title, p in [("클래스 분포(정상 vs 질병)","report/figures/01_class_counts.png"),
                 ("질병 종류 분포","report/figures/02_disease_type_dist.png"),
                 ("bbox 상대 면적","report/figures/05_bbox_rel_area.png"),
                 ("질병 샘플 + bbox (disease_3)","report/figures/10_samples_disease3.png")]:
    if os.path.exists(p): display(Markdown(f"**{title}** — `{p}`")); display(Image(filename=p))
''')

md("""## 2. 학습

아래 `SPECS`에 **전체 실험 42개가 그룹별로 모두 나열**되어 있습니다. **실행하지 않을 줄은 `#`로 주석처리**하세요.

- `SKIP_IF_DONE=True`: 이미 `metrics.json`이 있으면 학습을 건너뜁니다(재학습 없이 평가/표만 볼 때 유용). 데모용 fresh 학습이면 `False`.
- `QUICK_SMOKE=True`: `--smoke`(2 epoch, subset) 빠른 검증.
- ⚠️ 전체 42개를 단일 GPU로 다 돌리면 수 시간 걸립니다. 필요한 줄만 남기거나 GPU 여러 장이면 `_workspace/launch_all.sh`·`launch_new16.sh`(GPU 배정 수정) 사용.""")

code('''
# §2. 학습할 spec 목록 — 전체 42개를 그룹별로 나열. 불필요한 줄은 `#`로 주석처리하세요.
SPECS = [
    # ── 분류 baseline (from-scratch): 7 백본 × 3 세팅 ──────────────────────────
    "convnextv2_normal_vs_d3",     "convnextv2_normal_vs_d4",     "convnextv2_normal_d3_d4",
    "efficientnetv2_normal_vs_d3", "efficientnetv2_normal_vs_d4", "efficientnetv2_normal_d3_d4",
    "nextvit_normal_vs_d3",        "nextvit_normal_vs_d4",        "nextvit_normal_d3_d4",
    "nextvit20_normal_vs_d3",      "nextvit20_normal_vs_d4",      "nextvit20_normal_d3_d4",
    "densenet121_normal_vs_d3",    "densenet121_normal_vs_d4",    "densenet121_normal_d3_d4",
    "resnet50_normal_vs_d3",       "resnet50_normal_vs_d4",       "resnet50_normal_d3_d4",
    "mamba_normal_vs_d3",          "mamba_normal_vs_d4",          "mamba_normal_d3_d4",
    # ── detection baseline (from-scratch): single-box + objectness ────────────
    "convnextv2_detection_singlebox",   "efficientnetv2_detection_singlebox",
    "nextvit_detection_singlebox",      "nextvit20_detection_singlebox",
    "densenet121_detection_singlebox",  "resnet50_detection_singlebox",
    "mamba_detection_singlebox",
    # ── Ours: DINOv3-S @256 (frozen backbone + 2-layer head) ──────────────────
    "dinov3_normal_vs_d3",         "dinov3_normal_vs_d4",         "dinov3_normal_d3_d4",
    # ── Ours: DINOv3-B @512 (frozen) ──────────────────────────────────────────
    "dinov3_base_normal_vs_d3",    "dinov3_base_normal_vs_d4",    "dinov3_base_normal_d3_d4",
    # ── Ours+: DINOv3-B @512 + strong aug + focal loss ────────────────────────
    "dinov3_base_focal_normal_vs_d3", "dinov3_base_focal_normal_vs_d4", "dinov3_base_focal_normal_d3_d4",
    # ── Ours: DINOv3-B detection ──────────────────────────────────────────────
    "dinov3_base_detection_singlebox",
    # ── Ablation (DINOv3-B, 3-class): 증강 × focal, gamma sweep ────────────────
    "dinov3_base_augonly_normal_d3_d4",  "dinov3_base_focalonly_normal_d3_d4",
    "dinov3_base_focalg1_normal_d3_d4",  "dinov3_base_focalg3_normal_d3_d4",
]
SKIP_IF_DONE = True      # metrics.json 있으면 학습 생략 (fresh 학습이면 False)
QUICK_SMOKE  = False     # True면 --smoke 로 빠른 검증

assert all(os.path.exists(f"_workspace/specs/exp_{n}.yaml") for n in SPECS), "존재하지 않는 spec 이름이 있습니다."
print(f"학습 대상 {len(SPECS)}개 (SKIP_IF_DONE={SKIP_IF_DONE}, QUICK_SMOKE={QUICK_SMOKE})")
for name in SPECS:
    spec = f"_workspace/specs/exp_{name}.yaml"
    done = os.path.exists(f"experiments/{name}/metrics.json")
    if SKIP_IF_DONE and done and not QUICK_SMOKE:
        print(f"[skip] {name} (metrics.json 존재)"); continue
    cmd = [PY, "-m", "src.train", "--spec", spec, "--device", "cuda:0"]
    if QUICK_SMOKE: cmd.append("--smoke")
    print("="*80, f"\\nTRAIN {name}\\n", "="*80)
    run(cmd)
print("\\n학습 단계 완료.")
''')

md("""## 3. 평가 — 결과 지표

`SPECS`에 남은 run들의 `metrics.json`(best epoch 최종 지표)을 표로 봅니다.""")

code('''
# §3. 학습된 run들의 최종 지표 표
import pandas as pd
rows = []
for name in SPECS:
    p = f"experiments/{name}/metrics.json"
    if not os.path.exists(p): continue
    m = json.load(open(p)); f = m.get("final", {})
    rows.append({
        "run": name, "task": m.get("task"), "status": m.get("status"), "primary": m.get("primary"),
        "PR-AUC": f.get("pr_auc") if m.get("task")=="classification" else f.get("det_pr_auc"),
        "F1-macro": f.get("f1"), "accuracy": f.get("accuracy"), "AUROC": f.get("auroc"),
        "presence@0.5": f.get("presence_recall_at_0.5") or f.get("iou_at_0.5_presence"),
        "IoU_median": (f.get("iou_distribution") or {}).get("median") if m.get("task")=="detection" else None,
        "best_ep": f.get("epoch"),
    })
df = pd.DataFrame(rows)
pd.set_option("display.max_columns", None, "display.width", 220, "display.max_rows", 100)
df
''')

md("""## 4. 평가 — 전체 리포트 표/그림 재생성 (선택)

모든 run이 학습돼 있을 때 eval 스크립트로 `report/EXPERIMENTS.md`의 표·그림을 재생성합니다.""")

code('''
# §4. (선택) 전체 리포트 재생성 — 모든 run이 있을 때만 의미 있음
RUN_FULL_EVAL = False
if RUN_FULL_EVAL:
    for s in ["run_eval.py", "run_balanced_eval.py", "run_balanced_detection_eval.py",
              "run_ours_plus_eval.py", "run_ours_focal_eval.py", "run_ours_detection_eval.py",
              "run_ablation_eval.py"]:
        run([PY, f"_workspace/eval/{s}"])
else:
    print("RUN_FULL_EVAL=False — 저장소에 포함된 report/EXPERIMENTS.md·그림을 그대로 사용합니다.")
''')

md("""## 5. 결과 그림 (baseline vs Ours)""")

code('''
# §5. 핵심 비교 그림 인라인 표시
for title, path in [
    ("분류 7-메트릭 (baseline + Ours)", "report/figures/exp_metrics_table.png"),
    ("Ours 개선 추이 (small→base→focal vs baseline, +20% 목표선)", "report/figures/exp_ours_focal.png"),
    ("Detection (검출/objectness/IoU)", "report/figures/exp_detection.png"),
    ("Ablation: 증강 × focal", "report/figures/exp_ablation_dinov3.png"),
]:
    if os.path.exists(path):
        display(Markdown(f"**{title}** — `{path}`")); display(Image(filename=path))
    else:
        print("missing:", path)
''')

md("""## 6. 데모 (FastAPI)

학습된 체크포인트(`experiments/*/checkpoints/best.pt`)를 로드해 다중 파이프라인 동시 비교 데모를 띄웁니다. **체크포인트가 있어야** 동작하므로, fresh면 §2 학습을 먼저 수행하세요.""")

code('''
# §6. 데모 서버 백그라운드 기동
import urllib.request
DEMO_PORT = 8000
demo_log = open("_workspace/demo_notebook.log", "w")
demo_proc = subprocess.Popen([PY, "-m", "uvicorn", "demo.app:app", "--host", "0.0.0.0",
                              "--port", str(DEMO_PORT)], stdout=demo_log, stderr=subprocess.STDOUT)
print("데모 기동 중... (모델 로드까지 수십 초)")
url = f"http://localhost:{DEMO_PORT}"; ok = False
for _ in range(60):
    time.sleep(3)
    try:
        n = len(json.load(urllib.request.urlopen(url + "/api/pipelines", timeout=3)))
        print(f"OK — {n} pipelines loaded. 브라우저에서 열기: {url}"); ok = True; break
    except Exception:
        if demo_proc.poll() is not None:
            print("서버 종료됨 — 로그:"); print(open('_workspace/demo_notebook.log').read()[-2000:]); break
if not ok and demo_proc.poll() is None:
    print("아직 로딩 중일 수 있음. 잠시 후 다시 /api/pipelines 확인.")
''')

code('''
# §6b. 예시 추론 — valid 질병 이미지 1장에 여러 파이프라인 동시 적용
import urllib.parse
try:
    items = json.load(urllib.request.urlopen(url + "/api/valid-images?klass=disease_3&limit=1"))["items"]
    data = urllib.parse.urlencode({"valid_image_id": items[0]["id"], "pipelines": "all"}).encode()
    res = json.load(urllib.request.urlopen(url + "/api/predict", data=data, timeout=120))
    print("GT:", res["input"].get("ground_truth"))
    print("\\n-- classification (일부) --")
    for c in res["classification"][:6]:
        print(f'  {c["arch"]:16} {c.get("setting",""):14} -> {c["pred_class"]}')
    print("\\n-- detection --")
    for d in res["detection"][:4]:
        print(f'  {d["arch"]:24} objectness={d["objectness"]:.3f} is_disease={d["is_disease"]}')
except Exception as e:
    print("예시 추론 실패(서버 미기동/체크포인트 없음일 수 있음):", e)
''')

code('''
# §6c. 데모 서버 종료
try:
    demo_proc.terminate(); demo_proc.wait(timeout=10); print("데모 종료")
except Exception as e:
    print("종료 처리:", e)
''')

md("""## 7. 참고

- §2 `SPECS`에 전체 42개가 나열돼 있습니다. 원하는 줄만 남기고 나머지는 `#`로 주석처리하세요.
- GPU 여러 장이면 `_workspace/launch_all.sh` / `launch_new16.sh`(GPU 배정 수정)로 병렬 학습이 빠릅니다.
- 자세한 내용: `README.md`, `report/FINAL_REPORT.md`, `report/EXPERIMENTS.md`.""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.10"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("radish_demo.ipynb", "w"), ensure_ascii=False, indent=1)
print("wrote radish_demo.ipynb —", len(cells), "cells")
