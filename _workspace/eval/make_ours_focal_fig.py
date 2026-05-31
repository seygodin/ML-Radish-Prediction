"""Ours+ focal+aug vs base-CE vs Ours(small) vs baseline-best comparison.

4-way grouped bars per setting (PR-AUC + F1-macro, orig + balanced):
baseline-best (from-scratch), Ours (DINOv3-S @256 frozen, CE), Ours+ (DINOv3-B
@512 frozen, CE default-aug), Ours+ focal+aug (DINOv3-B @512 frozen + strong aug
+ focal gamma2). +20% goal line on the 3-class original-dist panels (design-notes
targets: PR-AUC>=0.684, F1-macro>=0.851).

Reads _workspace/eval/ours_dinov3.json (small + base-CE + base-focal merged).
English labels, Agg backend.

Run: ./.venv/bin/python _workspace/eval/make_ours_focal_fig.py
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

GROUPS = [
    ("baseline", "#999999", "baseline best (from-scratch)"),
    ("small", "#4C72B0", "Ours (DINOv3-S @256, frozen, CE)"),
    ("base", "#DD8452", "Ours+ (DINOv3-B @512, frozen, CE)"),
    ("focal", "#C44E52", "Ours+ focal+aug (DINOv3-B @512, +strong aug +focal g2)"),
]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Ours+ focal+aug vs base-CE vs Ours(small) vs baseline-best  "
             "| PR-AUC & F1-macro | +20% goal on 3-class original",
             fontsize=13, fontweight="bold")

x = np.arange(len(SETTINGS))
w = 0.20

for ri, (mk, mlab) in enumerate(METRICS):
    for ci, (sk, slab) in enumerate(SPLITS):
        ax = axes[ri][ci]
        vals = {g: [] for g, _, _ in GROUPS}
        base_archs = []
        for setting, _ in SETTINGS:
            bb = R[f"dinov3_base_{setting}"]["baseline_best"][sk][mk]
            base_archs.append(bb[0])
            vals["baseline"].append(bb[1])
            vals["small"].append(R[f"dinov3_{setting}"][sk][mk])
            vals["base"].append(R[f"dinov3_base_{setting}"][sk][mk])
            vals["focal"].append(R[f"dinov3_base_focal_{setting}"][sk][mk])

        offsets = [-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w]
        for (g, color, glab), off in zip(GROUPS, offsets):
            bars = ax.bar(x + off, vals[g], w, color=color, edgecolor="black",
                          linewidth=0.4, label=glab)
            for bb_, v in zip(bars, vals[g]):
                ax.text(bb_.get_x() + bb_.get_width() / 2, v + 0.008,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=6)
        # baseline arch annotation
        for xi, a in zip(x, base_archs):
            ax.text(xi - 1.5 * w, 0.02, a, ha="center", va="bottom", fontsize=5.5,
                    rotation=90, color="white")

        if sk == "orig":
            tgt = GOAL[mk]
            ax.plot([x[-1] - 2 * w, x[-1] + 2 * w], [tgt, tgt],
                    color="red", linestyle="--", linewidth=1.6)
            ax.text(x[-1] + 2.1 * w, tgt, f"+20% target {tgt:.3f}", color="red",
                    fontsize=7, va="center", ha="left")

        ax.set_ylim(0, 1.12)
        ax.set_xticks(x)
        ax.set_xticklabels([s for _, s in SETTINGS], fontsize=8)
        ax.set_ylabel(mlab, fontsize=10)
        ax.set_title(f"{mlab} — {slab}", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        if ri == 0 and ci == 0:
            ax.legend(fontsize=6.5, loc="lower left")

fig.tight_layout(rect=[0, 0, 1, 0.96])
out = ROOT / "report/figures/exp_ours_focal.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"wrote {out}")
