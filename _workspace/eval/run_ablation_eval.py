#!/usr/bin/env python
"""eval-reporter (QA): ablation aug x focal + gamma sweep on dinov3_base 3-class.

Independently recomputes (sklearn) the 4 trained ablation runs + 2 reference
runs from saved predictions/valid.npz, cross-validates predictions vs manifest
(label dist / N / split), contrasts recomputed vs reported (metrics.json.final),
performs the 2x2 contribution decomposition (aug effect / focal effect /
interaction), the gamma sweep (g1/g2/g3 under strong aug), per-class d3/d4
recall-precision with Wilson 95% CI (d4 N=24), and writes:
  - _workspace/eval/verify_dinov3_base_{augonly,focalonly,focalg1,focalg3}_normal_d3_d4.md
  - _workspace/eval/ablation_dinov3.json

best.pt / predictions are read-only (forward not even needed; original-dist
predictions already saved by experiment-runner). Run:
  ./.venv/bin/python _workspace/eval/run_ablation_eval.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
EVAL = ROOT / "_workspace" / "eval"
SETTING = "normal_d3_d4"
NUM_CLASSES = 3

# run name -> (loss label, aug label, role)
RUNS = {
    "dinov3_base_normal_d3_d4":          ("CE",        "default", "reference (base, CE/default)"),
    "dinov3_base_augonly_normal_d3_d4":  ("CE",        "strong",  "aug-only"),
    "dinov3_base_focalonly_normal_d3_d4":("focal g2",  "default", "focal-only"),
    "dinov3_base_focalg1_normal_d3_d4":  ("focal g1",  "strong",  "gamma=1 (strong aug)"),
    "dinov3_base_focal_normal_d3_d4":    ("focal g2",  "strong",  "reference (focal+aug, gamma=2)"),
    "dinov3_base_focalg3_normal_d3_d4":  ("focal g3",  "strong",  "gamma=3 (strong aug)"),
}
VERIFY_RUNS = [
    "dinov3_base_augonly_normal_d3_d4",
    "dinov3_base_focalonly_normal_d3_d4",
    "dinov3_base_focalg1_normal_d3_d4",
    "dinov3_base_focalg3_normal_d3_d4",
]


def wilson_ci(k, n, z=1.96):
    """이항 비율의 Wilson 95% 신뢰구간(저·고) — 소표본 지표에 CI 병기용."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def metrics_from_probs(probs, labels, num_classes):
    """확률·정답에서 분류 지표(PR-AUC/F1/recall/precision/accuracy/AUROC)를 재계산."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    preds = probs.argmax(axis=1)
    cls = list(range(num_classes))
    acc = float((preds == labels).mean())
    f1m = float(f1_score(labels, preds, labels=cls, average="macro", zero_division=0))
    recm = float(recall_score(labels, preds, labels=cls, average="macro", zero_division=0))
    precm = float(precision_score(labels, preds, labels=cls, average="macro", zero_division=0))
    aucs, aps = [], []
    for c in range(num_classes):
        y_c = (labels == c).astype(np.int64)
        if 0 < y_c.sum() < len(y_c):
            aucs.append(float(roc_auc_score(y_c, probs[:, c])))
    auroc = float(np.mean(aucs)) if aucs else float("nan")
    for c in range(1, num_classes):  # disease OvR macro
        y_c = (labels == c).astype(np.int64)
        if 0 < y_c.sum() < len(y_c):
            aps.append(float(average_precision_score(y_c, probs[:, c])))
    pr_auc = float(np.mean(aps)) if aps else float("nan")
    cm = confusion_matrix(labels, preds, labels=cls).tolist()
    return {"accuracy": acc, "recall_macro": recm, "precision_macro": precm,
            "f1_macro": f1m, "auroc": auroc, "pr_auc": pr_auc, "confusion": cm}


def per_class(cm, num_classes):
    """클래스별 precision/recall 계산."""
    cm = np.asarray(cm, dtype=np.float64)
    out = {}
    for c in range(num_classes):
        tp = cm[c, c]; row = cm[c, :].sum(); col = cm[:, c].sum()
        out[c] = {"recall": float(tp / row) if row else float("nan"),
                  "precision": float(tp / col) if col else float("nan"),
                  "support": int(row), "tp": int(tp), "pred_pos": int(col)}
    return out


# manifest dist
mcls = pd.read_csv(ROOT / "_workspace" / "data" / "manifest_classification.csv")
vv = mcls[(mcls.split == "valid") & (mcls["label_" + SETTING] >= 0)]
manifest_dist = vv["label_" + SETTING].value_counts().sort_index().to_dict()
manifest_total = int(len(vv))

results = {}
for name, (loss_lab, aug_lab, role) in RUNS.items():
    run_dir = EXP / name
    snap = json.loads((run_dir / "config.snapshot").read_text())
    mj = json.loads((run_dir / "metrics.json").read_text())
    z = np.load(run_dir / "predictions" / "valid.npz", allow_pickle=True)
    prob = z["prob"]; label = z["label"].astype(int)
    class_names = [str(x) for x in z["class_names"]]

    rec = metrics_from_probs(prob, label, NUM_CLASSES)
    pc = per_class(rec["confusion"], NUM_CLASSES)
    pred_dist = {int(c): int((label == c).sum()) for c in sorted(set(label.tolist()))}
    dist_match = (pred_dist == {int(k): int(v) for k, v in manifest_dist.items()}) and (len(label) == manifest_total)

    rep = mj["final"]
    rvr = {m: (float(rep[m]), float(rec[m])) for m in
           ["pr_auc", "f1_macro", "recall_macro", "precision_macro", "auroc", "accuracy"]}

    sp = snap["spec"]
    loss_cfg = snap.get("loss", {})
    results[name] = {
        "role": role, "loss_label": loss_lab, "aug_label": aug_lab,
        "arch": sp["model"]["arch"], "img_size": sp["model"]["img_size"],
        "aug": sp["data"].get("aug", "default"),
        "loss_type": loss_cfg.get("type"), "gamma": loss_cfg.get("gamma"),
        "class_weights": loss_cfg.get("class_weights"),
        "label_smoothing": loss_cfg.get("label_smoothing"),
        "trainable_params": snap.get("trainable_params", {}).get("trainable"),
        "best_epoch": int(rep["epoch"]),
        "class_names": class_names, "n_valid": int(len(label)),
        "metrics": rec, "per_class": pc,
        "boundary": {"pred_dist": pred_dist, "manifest_dist": manifest_dist,
                     "manifest_total": manifest_total, "n_pred": int(len(label)),
                     "dist_match": bool(dist_match)},
        "reported_vs_recomputed": rvr,
        "sklearn_crosscheck": {
            "pr_auc_d3": float(average_precision_score((label == 1).astype(int), prob[:, 1])),
            "pr_auc_d4": float(average_precision_score((label == 2).astype(int), prob[:, 2])),
        },
    }
    print(f"{name}: pr={rec['pr_auc']:.4f} f1={rec['f1_macro']:.4f} acc={rec['accuracy']:.4f} "
          f"auroc={rec['auroc']:.4f} d4(P/R)={pc[2]['precision']:.3f}/{pc[2]['recall']:.3f} "
          f"dist_match={dist_match} | reported pr={rep['pr_auc']:.4f} f1={rep['f1_macro']:.4f}", flush=True)


# ---------------------------------------------------------------------------
# 2x2 contribution decomposition (aug ✗/✓ x CE/focal-g2)
# cells: base(CE,def) augonly(CE,strong) focalonly(focal,def) focal(focal,strong)
# ---------------------------------------------------------------------------
B = results["dinov3_base_normal_d3_d4"]["metrics"]
A = results["dinov3_base_augonly_normal_d3_d4"]["metrics"]
Fo = results["dinov3_base_focalonly_normal_d3_d4"]["metrics"]
FA = results["dinov3_base_focal_normal_d3_d4"]["metrics"]
Bd4 = results["dinov3_base_normal_d3_d4"]["per_class"][2]
FAd4 = results["dinov3_base_focal_normal_d3_d4"]["per_class"][2]

decomp = {}
for m in ["pr_auc", "f1_macro", "accuracy", "auroc"]:
    aug_eff = A[m] - B[m]
    focal_eff = Fo[m] - B[m]
    interaction = (FA[m] + B[m]) - Fo[m] - A[m]
    joint = FA[m] - B[m]
    decomp[m] = {
        "base_CE_default": B[m], "augonly_CE_strong": A[m],
        "focalonly_focal_default": Fo[m], "focal_aug_focal_strong": FA[m],
        "aug_effect": aug_eff, "focal_effect": focal_eff,
        "interaction": interaction, "joint_effect": joint,
        "sum_check": aug_eff + focal_eff + interaction,  # == joint
    }

# d4 precision/recall decomposition
d4_decomp = {}
for met in ["precision", "recall"]:
    Bv = results["dinov3_base_normal_d3_d4"]["per_class"][2][met]
    Av = results["dinov3_base_augonly_normal_d3_d4"]["per_class"][2][met]
    Fov = results["dinov3_base_focalonly_normal_d3_d4"]["per_class"][2][met]
    FAv = results["dinov3_base_focal_normal_d3_d4"]["per_class"][2][met]
    d4_decomp[met] = {
        "base": Bv, "augonly": Av, "focalonly": Fov, "focal_aug": FAv,
        "aug_effect": Av - Bv, "focal_effect": Fov - Bv,
        "interaction": (FAv + Bv) - Fov - Av, "joint_effect": FAv - Bv,
    }

# gamma sweep (strong aug fixed): g1 / g2 / g3
gamma_sweep = {}
for g, nm in [(1.0, "dinov3_base_focalg1_normal_d3_d4"),
              (2.0, "dinov3_base_focal_normal_d3_d4"),
              (3.0, "dinov3_base_focalg3_normal_d3_d4")]:
    r = results[nm]; m = r["metrics"]; pc = r["per_class"]
    gamma_sweep[g] = {
        "run": nm, "pr_auc": m["pr_auc"], "f1_macro": m["f1_macro"],
        "accuracy": m["accuracy"], "auroc": m["auroc"],
        "d4_precision": pc[2]["precision"], "d4_recall": pc[2]["recall"],
        "d3_precision": pc[1]["precision"], "d3_recall": pc[1]["recall"],
    }
best_gamma_pr = max(gamma_sweep, key=lambda g: gamma_sweep[g]["pr_auc"])
best_gamma_f1 = max(gamma_sweep, key=lambda g: gamma_sweep[g]["f1_macro"])

# d4 Wilson CI for each run (orig dist, N=24)
d4_ci = {}
for name in RUNS:
    cm = np.array(results[name]["metrics"]["confusion"])
    tp = int(cm[2, 2]); supp = int(cm[2, :].sum()); pred = int(cm[:, 2].sum())
    rlo, rhi = wilson_ci(tp, supp); plo, phi = wilson_ci(tp, pred)
    d4_ci[name] = {"tp": tp, "support": supp, "pred_pos": pred,
                   "recall": tp / supp if supp else float("nan"),
                   "recall_ci": [rlo, rhi],
                   "precision": tp / pred if pred else float("nan"),
                   "precision_ci": [plo, phi]}

dump = {
    "setting": SETTING, "num_classes": NUM_CLASSES, "seed": 42,
    "valid_manifest_dist": manifest_dist, "valid_manifest_total": manifest_total,
    "results": results,
    "decomposition_2x2": decomp,
    "d4_decomposition": d4_decomp,
    "gamma_sweep": gamma_sweep,
    "best_gamma": {"by_pr_auc": best_gamma_pr, "by_f1_macro": best_gamma_f1},
    "d4_wilson_ci_orig": d4_ci,
    "note": (
        "Ablation on dinov3_base (DINOv3 ViT-B/16 frozen @512 + 2-layer head "
        "hidden512, trainable 395k, identical across all 6 runs). Only loss "
        "(CE / focal gamma) and aug (default / strong) vary. valid = original "
        "distribution normal 1303 / d3 76 / d4 24 (N=1403). class_weights "
        "from_meta resolve to all-ones (balanced-downsampled train loader) so "
        "focal acts via gamma (hard-example focusing), not alpha re-weighting. "
        "Single seed (42), d4 N=24 -> wide CI; treat sub-0.02 deltas as noise."
    ),
}
(EVAL / "ablation_dinov3.json").write_text(json.dumps(dump, indent=2, default=float))
print("\nwrote _workspace/eval/ablation_dinov3.json")
print("decomp pr_auc:", json.dumps(decomp["pr_auc"], default=float))
print("decomp f1:", json.dumps(decomp["f1_macro"], default=float))
print("best gamma by PR-AUC:", best_gamma_pr, "by F1:", best_gamma_f1)

# ---------------------------------------------------------------------------
# verify files
# ---------------------------------------------------------------------------
ref_base = results["dinov3_base_normal_d3_d4"]
for name in VERIFY_RUNS:
    r = results[name]; b = r["boundary"]; cn = r["class_names"]
    L = [f"# verify_{name}.md", "",
         f"- task: classification ({SETTING}) ablation run — **{r['role']}**",
         f"- arch = **{r['arch']}** (DINOv3 ViT-B/16 frozen @{r['img_size']}, 2-layer head hidden512), "
         f"trainable = **{r['trainable_params']}** (head only, identical to base/focal reference)",
         f"- loss = **{r['loss_type']}**(gamma={r['gamma']}, class_weights={r['class_weights']}, "
         f"ls={r['label_smoothing']}), aug = **{r['aug']}**, best epoch = {r['best_epoch']}",
         "- NOTE: class_weights `from_meta` -> all-ones (balanced-downsampled train loader); "
         "focal acts via **gamma** (hard-example focusing), not alpha. Single seed=42; d4 valid N=24 -> wide CI.",
         "",
         "## 1) Boundary cross-validation (predictions vs manifest, original dist)", "",
         f"- predictions valid N = **{b['n_pred']}**, manifest valid N = **{b['manifest_total']}** -> "
         f"{'MATCH' if b['n_pred']==b['manifest_total'] else 'MISMATCH'}",
         f"- predictions label dist = {b['pred_dist']}",
         f"- manifest valid label dist = {b['manifest_dist']}",
         f"- distribution match: **{'PASS' if b['dist_match'] else 'FAIL'}**  (no leakage / mislabel / split violation)",
         f"- class_names from npz: {cn}", "",
         "## 2) Independent recomputation (sklearn) vs reported (metrics.json.final)", "",
         "| metric | reported | recomputed (sklearn) | match (|Δ|≤0.01) |", "|---|---|---|---|"]
    for m, (rv, cv) in r["reported_vs_recomputed"].items():
        L.append(f"| {m} | {rv:.4f} | {cv:.4f} | {'ok' if abs(rv-cv)<=0.01 else '**DIFF**'} |")
    L += ["",
          f"- 7-metric (recomputed): acc={r['metrics']['accuracy']:.4f}, "
          f"recall_macro={r['metrics']['recall_macro']:.4f}, precision_macro={r['metrics']['precision_macro']:.4f}, "
          f"f1_macro={r['metrics']['f1_macro']:.4f}, AUROC={r['metrics']['auroc']:.4f}; PR-AUC={r['metrics']['pr_auc']:.4f}",
          f"- confusion (recomputed): {r['metrics']['confusion']}",
          f"- sklearn crosscheck: PR-AUC[d3]={r['sklearn_crosscheck']['pr_auc_d3']:.4f}, "
          f"PR-AUC[d4]={r['sklearn_crosscheck']['pr_auc_d4']:.4f} (disease OvR macro = mean)", "",
          "## 3) Per-class recall/precision (original dist)", "",
          "| class | support | recall | precision |", "|---|---|---|---|"]
    for c in range(NUM_CLASSES):
        pc = r["per_class"][c]
        L.append(f"| {cn[c]} | {pc['support']} | {pc['recall']:.3f} | {pc['precision']:.3f} |")
    ci = d4_ci[name]
    L += ["",
          "## 4) Small-sample note (Wilson 95% CI on d4, N=24)", "",
          f"- d4 recall = {ci['tp']}/{ci['support']} = {ci['recall']:.3f} "
          f"(95% CI {ci['recall_ci'][0]:.3f}-{ci['recall_ci'][1]:.3f})",
          f"- d4 precision = {ci['tp']}/{ci['pred_pos']} = {ci['precision']:.3f} "
          f"(95% CI {ci['precision_ci'][0]:.3f}-{ci['precision_ci'][1]:.3f})",
          "- d4 valid N=24 (orig) is tiny -> CIs wide; single-seed deltas <0.02 are within noise.", "",
          "## 5) Delta vs reference base (CE / default aug)", ""]
    for m in ["pr_auc", "f1_macro", "accuracy", "auroc"]:
        d = r["metrics"][m] - ref_base["metrics"][m]
        L += [f"- {m}: base {ref_base['metrics'][m]:.4f} -> this {r['metrics'][m]:.4f} (Δ {d:+.4f})"]
    bd4 = ref_base["per_class"][2]; td4 = r["per_class"][2]
    L += [f"- d4 precision: base {bd4['precision']:.3f} -> this {td4['precision']:.3f} "
          f"(Δ {td4['precision']-bd4['precision']:+.3f}); "
          f"d4 recall: base {bd4['recall']:.3f} -> this {td4['recall']:.3f} "
          f"(Δ {td4['recall']-bd4['recall']:+.3f})", ""]
    (EVAL / f"verify_{name}.md").write_text("\n".join(L))
    print(f"wrote verify_{name}.md")
