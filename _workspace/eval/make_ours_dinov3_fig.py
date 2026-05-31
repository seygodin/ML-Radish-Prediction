"""Ours vs Ours+ vs baseline-best comparison (PR-AUC + F1-macro, orig + balanced).

3-way grouped bars per setting: baseline-best (from-scratch), Ours (DINOv3 small
@256 frozen), Ours+ (DINOv3 base @512 frozen). +20% goal line drawn on the 3-class
panels (design-notes targets: PR-AUC>=0.684, F1-macro>=0.851 on original dist).

Reads _workspace/eval/ours_dinov3.json (merged small+base). English labels, Agg.

Run: ./.venv/bin/python _workspace/eval/make_ours_dinov3_fig.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
D = json.loads((ROOT / "_workspace/eval/ours_dinov3.json").read_text())
R = D["results"]
GOAL = D["goal_targets_3class_orig"]  # {"pr_auc":0.684,"f1_macro":0.851}

SETTINGS = [("normal_vs_d3", "normal vs d3"),
            ("normal_vs_d4", "normal vs d4"),
            ("normal_d3_d4", "3-class (normal/d3/d4)")]
SPLITS = [("orig", "original-dist valid"), ("bal", "balanced valid")]
METRICS = [("pr_auc", "PR-AUC"), ("f1_macro", "F1-macro")]

# model groups: (label, color, accessor) — baseline best is per-metric/per-split
GROUP_COLORS = {"baseline": "#999999", "small": "#4C72B0", "base": "#C44E52"}
GROUP_LABELS = {"baseline": "baseline best (from-scratch)",
                "small": "Ours (DINOv3-S @256, frozen)",
                "base": "Ours+ (DINOv3-B @512, frozen)"}

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Ours+ (DINOv3-B@512) vs Ours (DINOv3-S@256) vs baseline-best  "
             "| PR-AUC & F1-macro | +20% goal on 3-class original",
             fontsize=13, fontweight="bold")

x = np.arange(len(SETTINGS))
w = 0.26

for ri, (mk, mlab) in enumerate(METRICS):
    for ci, (sk, slab) in enumerate(SPLITS):
        ax = axes[ri][ci]
        base_vals, small_vals, plus_vals = [], [], []
        base_archs = []
        for setting, _ in SETTINGS:
            bb = R[f"dinov3_base_{setting}"]["baseline_best"][sk][mk]
            base_archs.append(bb[0])
            base_vals.append(bb[1])
            small_vals.append(R[f"dinov3_{setting}"][sk][mk])
            plus_vals.append(R[f"dinov3_base_{setting}"][sk][mk])

        b1 = ax.bar(x - w, base_vals, w, color=GROUP_COLORS["baseline"],
                    edgecolor="black", linewidth=0.4, label=GROUP_LABELS["baseline"])
        b2 = ax.bar(x, small_vals, w, color=GROUP_COLORS["small"],
                    edgecolor="black", linewidth=0.4, label=GROUP_LABELS["small"])
        b3 = ax.bar(x + w, plus_vals, w, color=GROUP_COLORS["base"],
                    edgecolor="black", linewidth=0.4, label=GROUP_LABELS["base"])
        for bars, vals in [(b1, base_vals), (b2, small_vals), (b3, plus_vals)]:
            for bb_, v in zip(bars, vals):
                ax.text(bb_.get_x() + bb_.get_width() / 2, v + 0.008,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)
        # baseline arch annotation under each baseline bar
        for xi, a in zip(x, base_archs):
            ax.text(xi - w, 0.02, a, ha="center", va="bottom", fontsize=5.5,
                    rotation=90, color="white")

        # +20% goal line on the 3-class group (original dist only — design-notes target)
        if sk == "orig":
            tgt = GOAL[mk]
            ax.plot([x[-1] - 1.5 * w, x[-1] + 1.5 * w], [tgt, tgt],
                    color="red", linestyle="--", linewidth=1.6)
            ax.text(x[-1] + 1.6 * w, tgt, f"+20% target {tgt:.3f}", color="red",
                    fontsize=7, va="center", ha="left")

        ax.set_ylim(0, 1.12)
        ax.set_xticks(x)
        ax.set_xticklabels([s for _, s in SETTINGS], fontsize=8)
        ax.set_ylabel(mlab, fontsize=10)
        ax.set_title(f"{mlab} — {slab}", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        if ri == 0 and ci == 0:
            ax.legend(fontsize=7, loc="lower left")

fig.tight_layout(rect=[0, 0, 1, 0.96])
out = ROOT / "report/figures/exp_ours_dinov3.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"wrote {out}")
