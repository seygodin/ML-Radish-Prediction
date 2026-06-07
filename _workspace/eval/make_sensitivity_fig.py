#!/usr/bin/env python
"""eval-reporter (QA): figure for input-noise sensitivity of Ours (3-class).

Reads _workspace/eval/sensitivity_dinov3.json and draws N_ratio vs
PR-AUC / F1-macro (primary panel) + accuracy / AUROC (secondary panel).

Output: report/figures/exp_sensitivity_dinov3.png

Run: ./.venv/bin/python _workspace/eval/make_sensitivity_fig.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "_workspace" / "eval"
FIGS = ROOT / "report" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

d = json.load(open(EVAL / "sensitivity_dinov3.json"))
pr = d["per_ratio"]
xs = d["n_ratios"]
keys = [f"{x:.1f}" for x in xs]
pr_auc = [pr[k]["pr_auc"] for k in keys]
f1 = [pr[k]["f1_macro"] for k in keys]
acc = [pr[k]["accuracy"] for k in keys]
auroc = [pr[k]["auroc"] for k in keys]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

ax1.plot(xs, pr_auc, "o-", color="#1f77b4", lw=2, ms=7, label="PR-AUC (primary)")
ax1.plot(xs, f1, "s--", color="#d62728", lw=2, ms=7, label="F1-macro")
ax1.axhline(pr_auc[0], color="#1f77b4", ls=":", alpha=0.4)
ax1.axhline(f1[0], color="#d62728", ls=":", alpha=0.4)
for x, y in zip(xs, pr_auc):
    ax1.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                 xytext=(0, 8), ha="center", fontsize=8, color="#1f77b4")
ax1.set_xlabel("N_ratio (input-tensor noise)")
ax1.set_ylabel("score")
ax1.set_title("Ours (DINOv3-B frozen, focal+aug, 3-class)\nPR-AUC / F1 vs input noise")
ax1.set_ylim(0.70, 0.82)
ax1.set_xticks(xs)
ax1.grid(True, alpha=0.3)
ax1.legend(loc="lower left")

ax2.plot(xs, acc, "^-", color="#2ca02c", lw=2, ms=7, label="accuracy")
ax2.plot(xs, auroc, "D--", color="#9467bd", lw=2, ms=7, label="AUROC")
for x, y in zip(xs, acc):
    ax2.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                 xytext=(0, -12), ha="center", fontsize=8, color="#2ca02c")
ax2.set_xlabel("N_ratio (input-tensor noise)")
ax2.set_ylabel("score")
ax2.set_title("accuracy / AUROC vs input noise")
ax2.set_ylim(0.95, 1.0)
ax2.set_xticks(xs)
ax2.grid(True, alpha=0.3)
ax2.legend(loc="lower left")

fig.suptitle(
    "Sensitivity: additive uniform noise on normalized input "
    "(x' = x + rand_like(x)*N_ratio), original-dist valid N=1403",
    fontsize=10, y=1.02)
fig.tight_layout()
out = FIGS / "exp_sensitivity_dinov3.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"wrote {out}")
