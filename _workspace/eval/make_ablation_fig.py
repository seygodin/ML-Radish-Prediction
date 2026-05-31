"""Ablation: aug x focal (2x2) + gamma sweep, dinov3_base 3-class (orig-dist).

Left:  2x2 grouped bars (PR-AUC, F1-macro, d4 precision) for the four cells
       base(CE/def), aug-only(CE/strong), focal-only(focal-g2/def),
       focal+aug(focal-g2/strong) -- shows aug as main driver, focal-only dip,
       and the synergy of the combination.
Mid:   contribution decomposition bars (aug effect / focal effect / interaction)
       for PR-AUC and F1-macro.
Right: gamma sweep under strong aug (g1/g2/g3): PR-AUC, F1-macro, d4 precision.

Reads _workspace/eval/ablation_dinov3.json. English labels, Agg backend, dpi150.
Run: ./.venv/bin/python _workspace/eval/make_ablation_fig.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
D = json.loads((ROOT / "_workspace/eval/ablation_dinov3.json").read_text())
R = D["results"]
DEC = D["decomposition_2x2"]
D4 = D["d4_decomposition"]
GS = D["gamma_sweep"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

# ---- panel 1: 2x2 cells grouped bars ----
cells = [
    ("base\n(CE/default)", "dinov3_base_normal_d3_d4", "#999999"),
    ("aug-only\n(CE/strong)", "dinov3_base_augonly_normal_d3_d4", "#55A868"),
    ("focal-only\n(g2/default)", "dinov3_base_focalonly_normal_d3_d4", "#DD8452"),
    ("focal+aug\n(g2/strong)", "dinov3_base_focal_normal_d3_d4", "#C44E52"),
]
mets = [("pr_auc", "PR-AUC"), ("f1_macro", "F1-macro")]
ax = axes[0]
x = np.arange(len(cells)); w = 0.34
for i, (mk, ml) in enumerate(mets):
    vals = [R[c[1]]["metrics"][mk] for c in cells]
    bars = ax.bar(x + (i - 0.5) * w, vals, w, label=ml,
                  color=["#4C72B0", "#C44E52"][i], alpha=0.9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
# d4 precision as line on twin axis
ax2 = ax.twinx()
d4p = [R[c[1]]["per_class"]["2"]["precision"] for c in cells]
ax2.plot(x, d4p, "o--", color="#8172B3", lw=2, ms=7, label="d4 precision (orig, N=24)")
for xi, v in zip(x, d4p):
    ax2.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
             fontsize=8, color="#8172B3")
ax2.set_ylabel("d4 precision", color="#8172B3")
ax2.set_ylim(0, 0.6); ax2.tick_params(axis="y", labelcolor="#8172B3")
ax.set_xticks(x); ax.set_xticklabels([c[0] for c in cells], fontsize=9)
ax.set_ylabel("PR-AUC / F1-macro"); ax.set_ylim(0.70, 0.80)
ax.set_title("2x2 ablation: aug x focal (dinov3_base 3-class, orig-dist)")
ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)
ax.grid(axis="y", alpha=0.3)

# ---- panel 2: contribution decomposition ----
ax = axes[1]
comps = [("aug_effect", "aug effect\n(augonly-base)"),
         ("focal_effect", "focal effect\n(focalonly-base)"),
         ("interaction", "interaction\n(synergy)"),
         ("joint_effect", "joint\n(focal+aug-base)")]
x = np.arange(len(comps)); w = 0.38
for i, (mk, ml) in enumerate(mets):
    vals = [DEC[mk][c[0]] for c in comps]
    bars = ax.bar(x + (i - 0.5) * w, vals, w, label=ml,
                  color=["#4C72B0", "#C44E52"][i], alpha=0.9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2,
                v + (0.0008 if v >= 0 else -0.0008), f"{v:+.3f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([c[1] for c in comps], fontsize=8.5)
ax.set_ylabel("Δ vs base (CE/default)")
ax.set_title("Contribution decomposition (aug main driver, focal-only dip, +synergy)")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

# ---- panel 3: gamma sweep ----
ax = axes[2]
gkeys = sorted(GS, key=lambda k: float(k))
gammas = [float(k) for k in gkeys]
pr = [GS[k]["pr_auc"] for k in gkeys]
f1 = [GS[k]["f1_macro"] for k in gkeys]
d4 = [GS[k]["d4_precision"] for k in gkeys]
ax.plot(gammas, pr, "o-", color="#4C72B0", lw=2, ms=8, label="PR-AUC")
ax.plot(gammas, f1, "s-", color="#C44E52", lw=2, ms=8, label="F1-macro")
ax.plot(gammas, d4, "^--", color="#8172B3", lw=2, ms=8, label="d4 precision")
for g, v in zip(gammas, pr):
    ax.text(g, v + 0.003, f"{v:.3f}", ha="center", fontsize=8, color="#4C72B0")
for g, v in zip(gammas, f1):
    ax.text(g, v - 0.006, f"{v:.3f}", ha="center", fontsize=8, color="#C44E52")
for g, v in zip(gammas, d4):
    ax.text(g, v - 0.010, f"{v:.3f}", ha="center", fontsize=8, color="#8172B3")
ax.set_xticks(gammas); ax.set_xlabel("focal gamma (strong aug fixed)")
ax.set_ylabel("metric"); ax.set_ylim(0.36, 0.80)
ax.set_title("Gamma sweep (strong aug): F1 peaks g1, PR-AUC/d4 stable g2-g3")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

fig.suptitle("Ablation: strong augmentation x focal loss + gamma sweep "
             "(dinov3_base frozen @512, 3-class normal/d3/d4, orig-dist, seed=42, d4 N=24)",
             fontsize=12, y=1.02)
fig.tight_layout()
out = ROOT / "report/figures/exp_ablation_dinov3.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
