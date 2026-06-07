#!/usr/bin/env python
"""eval-reporter (QA): Sensitivity analysis -- input-noise robustness of Ours
(DINOv3 ViT-B/16 frozen @512 + 2-layer head, focal+aug, 3-class).

Loads the TRAINED Ours model
(experiments/dinov3_base_focal_normal_d3_d4/checkpoints/best.pt) for FORWARD ONLY
(never retrained / weights never modified), then evaluates on the ORIGINAL-dist
valid set (setting=normal_d3_d4, img_size=512) while adding input noise.

Noise (user-specified) is applied to the NORMALIZED MODEL-INPUT TENSOR
x in [B,3,512,512] (the output of the valid transform, including ImageNet
normalization):

    x_noised = x + torch.rand_like(x) * N_ratio          # rand ~ U[0,1)

for N_ratio in {0.0(clean baseline), 0.1, 0.2, 0.3, 0.4, 0.5}. (The user's
formula `Noised = rand_like(Image)*N_ratio + Image` applied at the input tensor.)
torch.manual_seed is fixed per (N_ratio) sweep for reproducibility.

For each N_ratio we collect softmax probs over the full valid set and compute
PR-AUC (primary, macro one-vs-rest over disease classes), F1-macro, accuracy and
AUROC (macro one-vs-rest), plus absolute/relative degradation vs the clean
(N_ratio=0.0) baseline.

The N_ratio=0.0 result MUST match the §6 Ours (dinov3_base_focal 3-class,
original-dist) numbers (same model, same valid, no noise) -- this is asserted
as a cross-check (tolerance 0.01).

Output: _workspace/eval/sensitivity_dinov3.json

Run: ./.venv/bin/python _workspace/eval/run_sensitivity_eval.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXP = ROOT / "experiments"
EVAL = ROOT / "_workspace" / "eval"
EVAL.mkdir(parents=True, exist_ok=True)

RUN_NAME = "dinov3_base_focal_normal_d3_d4"
SETTING = "normal_d3_d4"
SEED = 42
N_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

# §6 reference (Ours focal+aug, 3-class, original dist) for clean cross-check
REF_CLEAN = {"pr_auc": 0.765, "f1_macro": 0.774, "accuracy": 0.973, "auroc": 0.995}

from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from src.data import build_classification_loaders  # noqa: E402
from src.models import build_classifier  # noqa: E402


def metrics_from_probs(probs, labels, num_classes):
    """확률·정답에서 분류 지표(PR-AUC/F1/recall/precision/accuracy/AUROC)를 재계산."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    preds = probs.argmax(axis=1)
    cls = list(range(num_classes))
    acc = float((preds == labels).mean())
    f1m = float(f1_score(labels, preds, labels=cls, average="macro", zero_division=0))
    if num_classes == 2:
        auroc = float(roc_auc_score((labels == 1).astype(int), probs[:, 1]))
        pr_auc = float(average_precision_score((labels == 1).astype(int), probs[:, 1]))
    else:
        aucs, aps = [], []
        for c in range(num_classes):
            y_c = (labels == c).astype(np.int64)
            if 0 < y_c.sum() < len(y_c):
                aucs.append(float(roc_auc_score(y_c, probs[:, c])))
        auroc = float(np.mean(aucs)) if aucs else float("nan")
        for c in range(1, num_classes):  # disease classes only
            y_c = (labels == c).astype(np.int64)
            if 0 < y_c.sum() < len(y_c):
                aps.append(float(average_precision_score(y_c, probs[:, c])))
        pr_auc = float(np.mean(aps)) if aps else float("nan")
    cm = confusion_matrix(labels, preds, labels=cls).tolist()
    return {"accuracy": acc, "f1_macro": f1m, "auroc": auroc, "pr_auc": pr_auc,
            "confusion": cm}


@torch.no_grad()
def forward_collect_noised(model, loader, device, num_classes, n_ratio):
    """Forward the full valid set with additive uniform noise on the normalized
    input tensor: x = x + rand_like(x) * n_ratio. Seed fixed per sweep."""
    model.eval()
    torch.manual_seed(SEED)  # reproducible noise per N_ratio sweep
    use_amp = device.type == "cuda"
    all_probs, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        if n_ratio > 0.0:
            images = images + torch.rand_like(images) * n_ratio
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
        all_probs.append(F.softmax(logits.float(), dim=1).cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def main():
    """스크립트 진입점: 예측/체크포인트 로드 → 지표 재계산 → JSON·그림·리포트 산출."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_dir = EXP / RUN_NAME
    snap = json.loads((run_dir / "config.snapshot").read_text())
    m_model = snap["spec"]["model"]
    arch = m_model["arch"]
    num_classes = int(m_model["num_classes"])
    img_size = int(m_model["img_size"])
    head_hidden = int(m_model.get("head_hidden", 512))
    assert arch == "dinov3_base", arch
    assert num_classes == 3 and img_size == 512, (num_classes, img_size)

    # ORIGINAL-dist valid loader (balance_valid=False), deterministic valid transform
    _, valid_loader, meta = build_classification_loaders(
        setting=SETTING, img_size=img_size, batch_size=32,
        num_workers=8, seed=SEED, balance_valid=False)
    class_names = meta["class_names"]

    model = build_classifier(arch=arch, num_classes=num_classes,
                             img_size=img_size, head_hidden=head_hidden).to(device)
    ckpt = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])  # forward only; weights never modified

    per_ratio = {}
    clean = None
    for nr in N_RATIOS:
        probs, labels = forward_collect_noised(model, valid_loader, device,
                                               num_classes, nr)
        m = metrics_from_probs(probs, labels, num_classes)
        per_ratio[f"{nr:.1f}"] = {"n_ratio": nr, **m,
                                  "n_valid": int(len(labels))}
        if nr == 0.0:
            clean = m
        print(f"N_ratio={nr:.1f}: pr_auc={m['pr_auc']:.4f} f1={m['f1_macro']:.4f} "
              f"acc={m['accuracy']:.4f} auroc={m['auroc']:.4f} N={len(labels)}",
              flush=True)

    # degradation vs clean baseline
    for nr in N_RATIOS:
        k = f"{nr:.1f}"
        d = per_ratio[k]
        deg = {}
        for met in ("pr_auc", "f1_macro", "accuracy", "auroc"):
            base = clean[met]
            deg[met] = {
                "abs": d[met] - base,
                "rel_pct": (100.0 * (d[met] - base) / base) if base else float("nan"),
            }
        per_ratio[k]["degradation_vs_clean"] = deg

    # clean cross-check vs §6 reference
    crosscheck = {}
    for met, ref in REF_CLEAN.items():
        got = clean[met]
        crosscheck[met] = {"clean": got, "reference_section6": ref,
                           "match": abs(got - ref) <= 0.01}
    all_match = all(v["match"] for v in crosscheck.values())

    dump = {
        "task": "sensitivity_input_noise",
        "model": "Ours focal+aug = DINOv3 ViT-B/16 frozen @512 + 2-layer head "
                 "(hidden512) + strong aug + focal gamma2",
        "run": RUN_NAME,
        "setting": SETTING,
        "valid_distribution": "original (balance_valid=False)",
        "img_size": img_size,
        "num_classes": num_classes,
        "class_names": class_names,
        "checkpoint": str(run_dir / "checkpoints" / "best.pt"),
        "weights_modified": False,
        "seed": SEED,
        "noise_formula": "x_noised = x + torch.rand_like(x) * N_ratio   "
                         "(rand ~ U[0,1))",
        "noise_application_point": "normalized model-input tensor [B,3,512,512] "
                                   "(output of valid transform incl. ImageNet "
                                   "normalization), per-batch, seed-fixed",
        "n_ratios": N_RATIOS,
        "primary_metric": "pr_auc (macro one-vs-rest over disease classes)",
        "per_ratio": per_ratio,
        "clean_crosscheck_vs_section6": crosscheck,
        "clean_crosscheck_all_match": all_match,
    }
    (EVAL / "sensitivity_dinov3.json").write_text(
        json.dumps(dump, indent=2, default=float))
    print("\nwrote _workspace/eval/sensitivity_dinov3.json")
    print("clean cross-check vs §6:", json.dumps(crosscheck, default=float))
    print("all clean metrics match §6 (tol 0.01):", all_match)

    if not all_match:
        print("WARNING: clean (N_ratio=0.0) does NOT match §6 reference -- "
              "investigate before trusting the sweep.", file=sys.stderr)


if __name__ == "__main__":
    main()
