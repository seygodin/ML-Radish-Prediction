#!/usr/bin/env python3
"""Generate radish_demo.ipynb (train + evaluate + demo in one notebook)."""
import json, os

cells = []
def md(text): cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})
def code(text): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})

md("""# 무(Radish) 질병 — 학습 · 평가 · 데모 (one notebook)

이 노트북은 **데이터가 이미 준비되어 있다고 가정**하고(아래 §0 확인), 대표 모델의 **학습 → 평가 → 데모**를 한 번에 실행합니다.

- 무거운 작업(학습/평가)은 프로젝트 가상환경 `./.venv/bin/python` **서브프로세스**로 실행하므로, 이 노트북 커널이 무엇이든 동작합니다.
- 결과 지표 표·그림은 인라인으로 표시하고, FastAPI 데모는 백그라운드로 띄웁니다.
- 전체 42개 실험을 다 돌리는 대신 **대표 셋**(강한 baseline + Ours + Ours-detection)만 기본 실행합니다. 전체 재현은 마지막 셀의 안내 참고.

**전제**: ① `data/`에 AI-Hub zip 배치·압축해제 완료(README §데이터), ② `uv`로 `.venv` 구성·의존성 설치 완료(README §환경 설정).
""")

code("""
# §0. 설정 — repo 루트로 이동, venv 파이썬 지정, 헬퍼
import os, sys, subprocess, json, glob, time
from pathlib import Path

# 이 노트북이 repo 어디서 열려도 루트를 찾도록
ROOT = Path.cwd()
while not (ROOT / "src" / "train.py").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
os.chdir(ROOT)
PY = str(ROOT / ".venv" / "bin" / "python")
assert Path(PY).exists(), f"venv python not found at {PY} — README의 환경 설정을 먼저 수행하세요."
print("repo root :", ROOT)
print("venv py   :", PY)

def run(cmd, **kw):
    \"\"\"서브프로세스 실행 + 실시간 출력. cmd는 리스트.\"\"\"
    print("$", " ".join(cmd)); sys.stdout.flush()
    return subprocess.run(cmd, **kw)
""")

code("""
# §0b. 환경 점검 (torch / CUDA / 주요 패키지 버전)
run([PY, "-c",
     "import torch,timm;print('torch',torch.__version__,'| cuda',torch.cuda.is_available(),"
     "'| device',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU','| timm',timm.__version__)"])
""")

md("""## 1. 데이터 확인

데이터가 준비됐는지 가볍게 확인합니다(라벨↔이미지 1:1 매칭, manifest 존재). 매칭 검증은 `data/verify_pairs.py`.
""")

code("""
# §1. 라벨↔이미지 매칭 검증 + manifest 확인
run([PY, "data/verify_pairs.py"])

man = list(glob.glob("_workspace/data/manifest_*.csv"))
print("\\nmanifests:", man if man else "(없음 — 아래 셀로 생성)")
""")

code("""
# §1b. (선택) manifest 재생성 — 저장소에 이미 있으면 건너뜀. 전체 스캔이라 수 분 소요.
if not glob.glob("_workspace/data/manifest_classification.csv"):
    run([PY, "-m", "src.data.build_manifest"])
else:
    print("manifest 존재 → 재생성 생략 (다시 만들려면 위 조건 무시하고 실행)")
""")

md("""## 2. 학습

대표 실험만 기본 실행합니다(전체 42개는 마지막 셀 참고). 각 run은 `experiments/<name>/`에 표준 산출물(metrics.json·config.snapshot·checkpoints·predictions)을 만듭니다.

- `SKIP_IF_DONE=True`: 이미 `metrics.json`이 있으면 학습을 건너뜁니다(저장소엔 metrics가 포함, 재학습 없이 평가/표만 보고 싶을 때 유용). 데모를 쓰려면 체크포인트가 필요하므로 fresh clone에서는 `False`로 두고 학습하세요.
- `QUICK_SMOKE=True`: `--smoke`(2 epoch, subset)로 빠른 동작 확인. 실제 결과 재현은 `False`.
""")

code("""
# §2. 학습할 대표 spec (이름은 _workspace/specs/exp_<name>.yaml)
SPECS = [
    "resnet50_normal_d3_d4",            # 강한 from-scratch baseline (가장 어려운 3-class)
    "dinov3_base_focal_normal_d3_d4",   # Ours+ (DINOv3-B frozen + 강한 증강 + focal), 3-class 헤드라인
    "dinov3_base_detection_singlebox",  # Ours detection (DINOv3-B frozen + single-box+objectness)
]
SKIP_IF_DONE = True      # metrics.json 있으면 학습 생략 (데모용 fresh clone이면 False 권장)
QUICK_SMOKE  = False     # True면 --smoke 로 빠른 검증

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
""")

md("""## 3. 평가 — 결과 지표

학습된 run들의 `metrics.json`(best epoch 기준 최종 지표)을 표로 봅니다. 저장소에 metrics.json이 포함되어 있어 **재학습 없이도** 우리 결과를 그대로 확인할 수 있습니다.
""")

code("""
# §3. 학습된 run들의 최종 지표 표
import pandas as pd
rows = []
for name in SPECS:
    p = f"experiments/{name}/metrics.json"
    if not os.path.exists(p): continue
    m = json.load(open(p)); f = m.get("final", {})
    rows.append({
        "run": name, "task": m.get("task"), "status": m.get("status"),
        "primary": m.get("primary"),
        "PR-AUC": f.get("pr_auc") if m.get("task")=="classification" else f.get("det_pr_auc"),
        "F1-macro": f.get("f1"),
        "accuracy": f.get("accuracy"),
        "AUROC": f.get("auroc"),
        "presence@0.5": f.get("presence_recall_at_0.5") or f.get("iou_at_0.5_presence"),
        "IoU_median": (f.get("iou_distribution") or {}).get("median") if m.get("task")=="detection" else None,
        "best_ep": f.get("epoch"),
    })
df = pd.DataFrame(rows)
pd.set_option("display.max_columns", None, "display.width", 200)
df
""")

md("""## 4. 평가 — 리포트 표/그림 재생성 (선택)

전체 실험이 모두 학습되어 있을 때, eval 스크립트로 `report/EXPERIMENTS.md`의 표·그림을 재생성합니다. (대표 셋만 학습했다면 일부 스크립트는 다른 run을 필요로 할 수 있으니 선택 실행)
""")

code("""
# §4. (선택) 전체 리포트 재생성 — 모든 run이 있을 때만 의미 있음
RUN_FULL_EVAL = False
if RUN_FULL_EVAL:
    for s in ["run_eval.py", "run_balanced_eval.py", "run_balanced_detection_eval.py",
              "run_ours_plus_eval.py", "run_ours_focal_eval.py", "run_ours_detection_eval.py",
              "run_ablation_eval.py"]:
        run([PY, f"_workspace/eval/{s}"])
else:
    print("RUN_FULL_EVAL=False — 저장소에 포함된 report/EXPERIMENTS.md·그림을 그대로 사용합니다.")
""")

md("""## 5. 결과 그림

저장소에 포함된 핵심 비교 그림을 표시합니다(위 §4를 돌렸다면 갱신된 그림).
""")

code("""
# §5. 핵심 그림 인라인 표시
from IPython.display import Image, Markdown, display
figs = [
    ("분류 7-메트릭 (baseline + Ours)", "report/figures/exp_metrics_table.png"),
    ("Ours 개선 추이 (small→base→focal vs baseline, +20% 목표선)", "report/figures/exp_ours_focal.png"),
    ("Detection (검출/objectness/IoU)", "report/figures/exp_detection.png"),
    ("Ablation: 증강 × focal", "report/figures/exp_ablation_dinov3.png"),
]
for title, path in figs:
    if os.path.exists(path):
        display(Markdown(f"**{title}** — `{path}`")); display(Image(filename=path))
    else:
        print("missing:", path)
""")

md("""## 6. 데모 (FastAPI)

학습된 체크포인트(`experiments/*/checkpoints/best.pt`)를 로드해 다중 파이프라인 동시 비교 데모를 띄웁니다. **체크포인트가 있어야** 동작하므로, fresh clone이면 §2 학습을 먼저 수행하세요.
""")

code("""
# §6. 데모 서버 백그라운드 기동
import urllib.request
DEMO_PORT = 8000
demo_log = open("_workspace/demo_notebook.log", "w")
demo_proc = subprocess.Popen([PY, "-m", "uvicorn", "demo.app:app", "--host", "0.0.0.0",
                              "--port", str(DEMO_PORT)], stdout=demo_log, stderr=subprocess.STDOUT)
print("데모 기동 중... (모델 로드까지 수십 초)")
url = f"http://localhost:{DEMO_PORT}"
ok = False
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
""")

code("""
# §6b. 예시 추론 — valid 질병 이미지 1장에 여러 파이프라인 동시 적용
import urllib.parse
try:
    items = json.load(urllib.request.urlopen(url + "/api/valid-images?klass=disease_3&limit=1"))["items"]
    vid = items[0]["id"]
    data = urllib.parse.urlencode({"valid_image_id": vid, "pipelines": "all"}).encode()
    res = json.load(urllib.request.urlopen(url + "/api/predict", data=data, timeout=120))
    print("GT:", res["input"].get("ground_truth"))
    print("\\n-- classification (일부) --")
    for c in res["classification"][:6]:
        print(f'  {c["arch"]:16} {c.get("setting",""):14} -> {c["pred_class"]}')
    print("\\n-- detection --")
    for d in res["detection"][:4]:
        print(f'  {d["arch"]:22} objectness={d["objectness"]:.3f} is_disease={d["is_disease"]}')
except Exception as e:
    print("예시 추론 실패(서버 미기동/체크포인트 없음일 수 있음):", e)
""")

code("""
# §6c. 데모 서버 종료
try:
    demo_proc.terminate(); demo_proc.wait(timeout=10); print("데모 종료")
except Exception as e:
    print("종료 처리:", e)
""")

md("""## 7. 전체 재현 안내

위는 대표 셋입니다. **전체 42개 실험**을 재현하려면 §2의 `SPECS`를 전체 목록으로 바꾸세요:

```python
SPECS = [os.path.basename(p)[4:-5] for p in glob.glob("_workspace/specs/exp_*.yaml")]  # exp_<name>.yaml -> <name>
```

GPU 여러 장이면 `_workspace/launch_all.sh` / `launch_new16.sh`(GPU 배정 수정)로 병렬 학습이 빠릅니다. 학습 후 §4의 `RUN_FULL_EVAL=True`로 `report/EXPERIMENTS.md`(표·그림)와 `report/FINAL_REPORT.md` 근거를 갱신할 수 있습니다. 자세한 내용은 `README.md`·`report/FINAL_REPORT.md` 참고.
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (.venv)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
out = "radish_demo.ipynb"
json.dump(nb, open(out, "w"), ensure_ascii=False, indent=1)
print("wrote", out, "—", len(cells), "cells")
