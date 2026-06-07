"""Explainability 실험 산출물 생성 — detection 시각화 + VQA QA 예시.

PAPER.md의 설명가능성(Explainability) 섹션을 위해:
  (1) 예시 valid 이미지(normal / disease_3 / disease_4)에 대해 Ours(DINOv3-B frozen)
      detection 결과(예측 박스 + objectness)와 GT 박스를 한 그림에 시각화 →
      report/figures/exp_explainability_detection.png
  (2) 각 이미지에 대해 "질병 특성"을 묻는 영어 질문들을 VQA(SmolVLM)로 질의하고
      응답을 모아 _workspace/eval/explainability_vqa.json 으로 저장.

모두 학습된 체크포인트를 forward-only로 사용한다(재학습 없음). 결정적 선택을 위해
각 klass의 정렬된 valid 목록 중앙 이미지를 사용한다(run_explainability와 동일 규칙).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import inference  # noqa: E402

MANIFEST = ROOT / "_workspace" / "data" / "manifest_classification.csv"
FIG_OUT = ROOT / "report" / "figures" / "exp_explainability_detection.png"
VQA_OUT = ROOT / "_workspace" / "eval" / "explainability_vqa.json"
DET_PIPE = "dinov3_base_detection_singlebox"  # Ours detection

# 질병 특성을 묻는 영어 질문(설명가능성용). 일반/특성/위치를 고르게 커버.
QUESTIONS = [
    "Is this radish leaf healthy or diseased?",
    "What disease symptoms are visible on this radish plant?",
    "Describe the color and texture of any affected or damaged areas.",
    "Which part of the image shows the most severe damage?",
]


def pick_examples() -> list[dict]:
    """각 klass의 정렬된 valid 목록 중앙 이미지를 결정적으로 선택."""
    rows = [r for r in csv.DictReader(open(MANIFEST, encoding="utf-8")) if r["split"] == "valid"]
    out = []
    for k in ["normal", "disease_3", "disease_4"]:
        sub = sorted([r for r in rows if r["klass"] == k], key=lambda r: r["image_path"])
        r = sub[len(sub) // 2]
        gt = None
        if k != "normal":
            try:
                gt = [float(r["x0"]), float(r["y0"]), float(r["x1"]), float(r["y1"])]
            except (KeyError, ValueError):
                gt = None
        out.append({"klass": k, "path": str(ROOT / r["image_path"]), "gt_box": gt,
                    "risk": r.get("risk")})
    return out


def load_image(path: str) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def main() -> None:
    inference.load_registry()
    examples = pick_examples()

    # ---- (1) detection 시각화 (3 패널) ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    det_records = []
    for ax, ex in zip(axes, examples):
        pil = load_image(ex["path"])
        res = inference.predict_image(pil, [DET_PIPE])
        det = res["detection"][0]
        ax.imshow(pil)
        ax.set_xticks([]); ax.set_yticks([])
        # GT 박스(노랑 점선)
        if ex["gt_box"]:
            x0, y0, x1, y1 = ex["gt_box"]
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   edgecolor="#ffd400", lw=2.5, ls="--"))
            ax.text(x0 + 6, y0 + 36, "GT", color="#ffd400", fontsize=11, weight="bold")
        # 예측 박스(빨강) — objectness 표기
        bx = det["box_xyxy"]
        ax.add_patch(Rectangle((bx[0], bx[1]), bx[2] - bx[0], bx[3] - bx[1], fill=False,
                               edgecolor="#e6194B", lw=2.5))
        verdict = "DISEASE" if det["is_disease"] else "NORMAL"
        ax.set_title(f'{ex["klass"]}  →  pred: {verdict}\nobjectness={det["objectness"]:.3f}',
                     fontsize=11)
        det_records.append({"klass": ex["klass"], "objectness": det["objectness"],
                            "is_disease": det["is_disease"], "pred_box": bx,
                            "gt_box": ex["gt_box"]})
    fig.suptitle("Ours (DINOv3-B frozen) detection — red = predicted box, yellow dashed = GT",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("saved figure:", FIG_OUT)

    # ---- (2) VQA QA 예시 ----
    from src import vqa as vqa_mod
    qa = []
    for ex in examples:
        pil = load_image(ex["path"])
        answers = []
        for q in QUESTIONS:
            a = vqa_mod.vqa(pil, q)
            answers.append({"question": q, "answer": a})
            print(f'[{ex["klass"]}] Q: {q}\n   A: {a}')
        qa.append({"klass": ex["klass"], "risk": ex["risk"], "qa": answers})

    VQA_OUT.write_text(json.dumps(
        {"detection": det_records, "vqa": qa, "questions": QUESTIONS,
         "det_pipeline": DET_PIPE, "examples": [e["path"] for e in examples]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved vqa json:", VQA_OUT)


if __name__ == "__main__":
    main()
