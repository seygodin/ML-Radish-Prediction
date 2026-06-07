#!/usr/bin/env python
"""eval-reporter (QA): figure for train-ratio stability sweep of Ours (3-class).

Reads _workspace/eval/stability_dinov3.json and draws:
  x = train_ratio, y = PR-AUC(primary) + F1-macro curves (points+line),
  with per-class train sample count as a secondary axis / annotation.

Output: report/figures/exp_stability_dinov3.png

Run: ./.venv/bin/python _workspace/eval/make_stability_fig.py
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

d = json.load(open(EVAL / "stability_dinov3.json"))
xs = d["train_ratios"]
keys = [f"{x:.1f}" for x in xs]
pr_auc = [d["per_ratio"][k]["pr_auc"] for k in keys]
f1 = [d["per_ratio"][k]["f1_macro"] for k in keys]
acc = [d["per_ratio"][k]["accuracy"] for k in keys]
auroc = [d["per_ratio"][k]["auroc"] for k in keys]
n_per_cls = [d["per_ratio"][k]["train_per_class"] for k in keys]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))

# --- primary panel: PR-AUC / F1 vs train_ratio + sample-count secondary axis ---
l1, = ax1.plot(xs, pr_auc, "o-", color="#1f77b4", lw=2.2, ms=8, label="PR-AUC (primary)")
l2, = ax1.plot(xs, f1, "s--", color="#d62728", lw=2.2, ms=8, label="F1-macro")
for x, y in zip(xs, pr_auc):
    ax1.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                 xytext=(0, 9), ha="center", fontsize=8, color="#1f77b4")
for x, y in zip(xs, f1):
    ax1.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                 xytext=(0, -14), ha="center", fontsize=8, color="#d62728")
ax1.axhline(pr_auc[-1], color="#1f77b4", ls=":", alpha=0.4)
ax1.set_xlabel("train_ratio (fraction of balanced train set)")
ax1.set_ylabel("score")
ax1.set_title("Ours (DINOv3-B frozen, focal+aug, 3-class)\nPR-AUC / F1-macro vs train ratio")
ax1.set_ylim(0.28, 0.86)
ax1.set_xticks(xs)
ax1.grid(True, alpha=0.3)

# secondary axis: per-class train sample count (bars, annotated)
axb = ax1.twinx()
axb.bar(xs, n_per_cls, width=0.05, color="#7f7f7f", alpha=0.18, zorder=0)
for x, n in zip(xs, n_per_cls):
    axb.annotate(f"N/cls={n}", (x, n), textcoords="offset points",
                 xytext=(0, 3), ha="center", fontsize=7, color="#555555")
axb.set_ylabel("train samples per class (balanced)", color="#555555")
axb.set_ylim(0, max(n_per_cls) * 1.55)
axb.tick_params(axis="y", labelcolor="#555555")

ax1.legend([l1, l2], ["PR-AUC (primary)", "F1-macro"], loc="lower right")

# --- secondary panel: accuracy / AUROC vs train_ratio ---
ax2.plot(xs, acc, "^-", color="#2ca02c", lw=2, ms=7, label="accuracy")
ax2.plot(xs, auroc, "D-", color="#9467bd", lw=2, ms=7, label="AUROC (OvR macro)")
for x, y in zip(xs, acc):
    ax2.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                 xytext=(0, 8), ha="center", fontsize=8, color="#2ca02c")
ax2.set_xlabel("train_ratio")
ax2.set_ylabel("score")
ax2.set_title("Saturating accuracy / AUROC\n(majority-class & ranking metrics saturate early)")
ax2.set_ylim(0.68, 1.005)
ax2.set_xticks(xs)
ax2.grid(True, alpha=0.3)
ax2.legend(loc="lower right")

fig.suptitle("Stability: train-ratio sweep (Ours, 3-class, original-dist valid N=1403)",
             fontsize=12, y=1.02)
fig.tight_layout()
out = FIGS / "exp_stability_dinov3.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
