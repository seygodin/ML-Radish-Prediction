"""Balanced-valid metric bars: AUROC / F1-macro / accuracy, 6 backbones x 3 settings.

nextvit20 (NeXtViT-base) excluded from report per user request (artifacts preserved on disk).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
res = json.loads((ROOT / "_workspace/eval/balanced_valid.json").read_text())["results"]
# Ours (DINOv3 frozen pretrained) balanced metrics — appended per subplot, distinct color.
ours_res = json.loads((ROOT / "_workspace/eval/ours_dinov3.json").read_text())["results"]

ARCHS = ["convnextv2", "efficientnetv2", "nextvit",
         "densenet121", "resnet50", "mamba"]
LABELS = ["ConvNeXtV2", "EfficientNetV2", "NeXtViT-s",
          "DenseNet121", "ResNet50", "VisionMamba"]
# Ours: (json prefix, x-tick label, bar color, hatch). Pretrained, head-only — NOT same condition.
OURS = [("dinov3", "Ours-S\n(pre,CE)", "#000000", "//"),
        ("dinov3_base", "Ours-B\n(pre,CE)", "#8B0000", "xx"),
        ("dinov3_base_focal", "Ours-B\n(pre,focal)", "#9400D3", "..")]
SETTINGS = [("normal_vs_d3", "normal vs d3 (76/76)"),
            ("normal_vs_d4", "normal vs d4 (24/24)"),
            ("normal_d3_d4", "3-class (24/24/24)")]
METRICS = [("auroc", "AUROC"), ("f1_macro", "F1-macro"), ("accuracy", "Accuracy")]
COLORS = ["#4C72B0", "#DD8452", "#55A868"]

all_labels = LABELS + [o[1] for o in OURS]
fig, axes = plt.subplots(3, 3, figsize=(18, 11), sharey=True)
x = np.arange(len(ARCHS) + len(OURS))
w = 0.6
for r, (sk, st) in enumerate(SETTINGS):
    for cc, (mk, mlab) in enumerate(METRICS):
        ax = axes[r][cc]
        base_vals = [res[f"{a}_{sk}"][mk] for a in ARCHS]
        ours_vals = [ours_res[f"{pfx}_{sk}"]["bal"][mk] for pfx, _, _, _ in OURS]
        # baseline bars (from-scratch)
        bars = ax.bar(x[:len(ARCHS)], base_vals, w, color=COLORS[cc], edgecolor="black", linewidth=0.4)
        # Ours bars (pretrained, distinct color+hatch)
        for oi, (pfx, _, ocol, ohatch) in enumerate(OURS):
            ax.bar(x[len(ARCHS) + oi], ours_vals[oi], w, color=ocol, hatch=ohatch,
                   edgecolor="white", linewidth=0.4)
        for xi, v in zip(x, base_vals + ours_vals):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
        ax.axvline(len(ARCHS) - 0.5, ls="--", c="gray", lw=1)  # baseline | Ours divider
        ax.set_ylim(0, 1.08)
        ax.set_xticks(x)
        ax.set_xticklabels(all_labels, rotation=40, ha="right", fontsize=6.5)
        if cc == 0:
            ax.set_ylabel(st, fontsize=9)
        if r == 0:
            ax.set_title(mlab, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
fig.suptitle("Balanced-valid re-evaluation (1:1 / 1:1:1), best.pt reloaded — 6 from-scratch backbones (left of dashed line) "
             "vs Ours DINOv3 frozen-pretrained head-only (right; black/red/violet hatched). NOT same-condition comparison.",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.98])
out = ROOT / "report/figures/exp_metrics_balanced.png"
fig.savefig(out, dpi=130)
print("wrote", out)
