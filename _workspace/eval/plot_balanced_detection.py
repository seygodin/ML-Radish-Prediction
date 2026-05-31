"""Figure for balanced-valid detection eval -> report/figures/exp_detection_balanced.png."""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
d = json.loads((ROOT / "_workspace/eval/balanced_valid_detection.json").read_text())["results"]

# nextvit20 (NeXtViT-base) excluded from report per user request (artifacts preserved on disk).
archs = ["convnextv2", "efficientnetv2", "nextvit",
         "densenet121", "resnet50", "mamba"]
labels = ["ConvNeXtV2", "EfficientNetV2", "NeXtViT-s",
          "DenseNet121", "ResNet50", "VisionMamba"]
runs = [f"{a}_detection_singlebox" for a in archs]

det_pr_bal = [d[r]["det_pr_auc"] for r in runs]
det_pr_orig = [d[r]["orig_full"]["det_pr_auc"] for r in runs]
mAP = [d[r]["map_at_0.5"] for r in runs]
iou_med = [d[r]["iou_median"] for r in runs]
pres = [d[r]["presence_recall_at_0.5"] for r in runs]
fp = [d[r]["fp_rate_at_0.5"] for r in runs]

x = np.arange(len(archs))
fig, ax = plt.subplots(1, 3, figsize=(18, 5.2), dpi=130)

# (1) det PR-AUC balanced vs original
w = 0.38
ax[0].bar(x - w / 2, det_pr_orig, w, label="orig (prev~0.071)", color="#aac")
ax[0].bar(x + w / 2, det_pr_bal, w, label="balanced (1:1)", color="#2a6")
ax[0].axhline(0.5, ls="--", c="gray", lw=1, label="bal baseline (prev=0.5)")
ax[0].set_ylim(0.4, 1.005)
ax[0].set_title("det PR-AUC: original vs balanced valid")
ax[0].set_ylabel("det PR-AUC")
ax[0].legend(fontsize=8)

# (2) presence_recall & fp_rate (balanced)
ax[1].bar(x - w / 2, pres, w, label="presence_recall@0.5", color="#36c")
ax[1].bar(x + w / 2, fp, w, label="fp_rate@0.5 (normal)", color="#c44")
ax[1].set_ylim(0, 1.05)
ax[1].set_title("Presence recall vs FP rate @0.5 (balanced, 100/100)")
ax[1].legend(fontsize=8)

# (3) localization: mAP@0.5 & IoU median (invariant)
ax[2].bar(x - w / 2, mAP, w, label="mAP@0.5", color="#84a")
ax[2].bar(x + w / 2, iou_med, w, label="IoU median (pos-only, invariant)", color="#e90")
ax[2].set_ylim(0, 0.75)
ax[2].set_title("Localization: mAP@0.5 & IoU median")
ax[2].legend(fontsize=8)

for a in ax:
    a.set_xticks(x)
    a.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    a.grid(axis="y", ls=":", alpha=0.4)

fig.suptitle("Detection balanced valid (disease 100 : normal 100, seed=42) "
             "— best.pt loaded as-is, fp32", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = ROOT / "report/figures/exp_detection_balanced.png"
fig.savefig(out)
print("wrote", out)
