#!/usr/bin/env python
"""eval-reporter (QA): Stability of Ours vs train-data ratio (train_ratio sweep).

Ours = DINOv3 ViT-B/16 frozen @512 + 2-layer head + strong aug + focal(gamma2),
3-class (normal_d3_d4). Trained runs at train_ratio in {0.1,0.3,0.5,0.7,0.9}
(experiments/dinov3_base_focal_r{10,30,50,70,90}_normal_d3_d4/) plus the
reference point train_ratio=1.0 = experiments/dinov3_base_focal_normal_d3_d4/
(the existing Ours, identical setting). NO retraining — uses saved
metrics.json + predictions/valid.npz only. best.pt never loaded.

Does, per ratio:
 1) boundary cross-validation: predictions(valid.npz) vs manifest valid dist
    (original distribution = normal 1303 / d3 76 / d4 24, N=1403).
 2) independent sklearn recomputation of PR-AUC / F1-macro / accuracy / AUROC
    cross-checked vs metrics.json final (reported).
 3) collect PR-AUC(primary)/F1-macro/accuracy/AUROC + train_counts per class.

Dumps _workspace/eval/stability_dinov3.json.

Run: ./.venv/bin/python _workspace/eval/run_stability_eval.py
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

from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
import pandas as pd  # noqa: E402

SETTING = "normal_d3_d4"
NUM_CLASSES = 3
# (train_ratio, experiment dir name)
RUNS = [
    (0.1, "dinov3_base_focal_r10_normal_d3_d4"),
    (0.3, "dinov3_base_focal_r30_normal_d3_d4"),
    (0.5, "dinov3_base_focal_r50_normal_d3_d4"),
    (0.7, "dinov3_base_focal_r70_normal_d3_d4"),
    (0.9, "dinov3_base_focal_r90_normal_d3_d4"),
    (1.0, "dinov3_base_focal_normal_d3_d4"),  # reference = existing Ours (§6)
]


def metrics_from_probs(probs, labels, num_classes):
    """확률·정답에서 분류 지표(PR-AUC/F1/recall/precision/accuracy/AUROC)를 재계산."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    preds = probs.argmax(axis=1)
    cls = list(range(num_classes))
    acc = float((preds == labels).mean())
    f1m = float(f1_score(labels, preds, labels=cls, average="macro", zero_division=0))
    aucs, aps = [], []
    for c in range(num_classes):
        y_c = (labels == c).astype(np.int64)
        if 0 < y_c.sum() < len(y_c):
            aucs.append(float(roc_auc_score(y_c, probs[:, c])))
    auroc = float(np.mean(aucs)) if aucs else float("nan")
    for c in range(1, num_classes):  # disease classes only (d3,d4)
        y_c = (labels == c).astype(np.int64)
        if 0 < y_c.sum() < len(y_c):
            aps.append(float(average_precision_score(y_c, probs[:, c])))
    pr_auc = float(np.mean(aps)) if aps else float("nan")
    cm = confusion_matrix(labels, preds, labels=cls).tolist()
    return {"accuracy": acc, "f1_macro": f1m, "auroc": auroc, "pr_auc": pr_auc,
            "confusion": cm}


# manifest valid distribution (original)
mcls = pd.read_csv(ROOT / "_workspace" / "data" / "manifest_classification.csv")
mcls_valid = mcls[mcls.split == "valid"]
lab = "label_" + SETTING
mvalid = mcls_valid[mcls_valid[lab] >= 0]
manifest_dist = {int(k): int(v) for k, v in
                 mvalid[lab].value_counts().sort_index().to_dict().items()}
manifest_total = int(len(mvalid))

per_ratio = {}
crosscheck = {}
for ratio, name in RUNS:
    run_dir = EXP / name
    snap = json.loads((run_dir / "config.snapshot").read_text())
    mj = json.loads((run_dir / "metrics.json").read_text())
    m_model = snap["spec"]["model"]
    m_data = snap["spec"]["data"]
    assert m_model["arch"] == "dinov3_base", (name, m_model["arch"])
    assert str(m_data.get("aug")) == "strong", (name, m_data.get("aug"))
    assert snap.get("loss", {}).get("type") == "focal", (name, snap.get("loss"))
    spec_ratio = m_data.get("train_ratio", 1.0)
    spec_ratio = 1.0 if spec_ratio is None else float(spec_ratio)
    assert abs(spec_ratio - ratio) < 1e-6, (name, spec_ratio, ratio)

    train_counts = mj["meta"]["train_counts"]
    valid_counts = mj["meta"]["valid_counts"]
    class_names = [str(x) for x in mj["meta"]["class_names"]]

    # recompute from predictions
    z = np.load(run_dir / "predictions" / "valid.npz", allow_pickle=True)
    prob = z["prob"]
    label = z["label"].astype(int)
    rec = metrics_from_probs(prob, label, NUM_CLASSES)

    pred_dist = {int(c): int((label == c).sum()) for c in sorted(set(label.tolist()))}
    dist_match = (pred_dist == manifest_dist) and (len(label) == manifest_total)

    rep = mj["final"]
    reported = {
        "pr_auc": float(rep["pr_auc"]),
        "f1_macro": float(rep["f1_macro"]),
        "accuracy": float(rep["accuracy"]),
        "auroc": float(rep["auroc"]),
    }
    rep_recomp_match = all(
        abs(reported[k] - rec[k]) <= 0.01 for k in reported
    )

    # extra sklearn cross-check on the disease-OvR PR-AUC at ratio reference
    if abs(ratio - 1.0) < 1e-6 or abs(ratio - 0.5) < 1e-6:
        aps = []
        for c in (1, 2):
            y_c = (label == c).astype(int)
            aps.append(float(average_precision_score(y_c, prob[:, c])))
        crosscheck[f"{ratio:.1f}"] = {
            "pr_auc_sklearn_ovr_macro": float(np.mean(aps)),
            "reported_pr_auc": reported["pr_auc"],
        }

    per_class_train = {cn: int(train_counts[cn]) for cn in class_names}
    train_per_class = int(min(per_class_train.values()))  # balanced downsample => equal
    train_total = int(sum(per_class_train.values()))

    per_ratio[f"{ratio:.1f}"] = {
        "train_ratio": ratio,
        "run": name,
        "best_epoch": int(rep["epoch"]),
        "train_counts": per_class_train,
        "train_per_class": train_per_class,
        "train_total": train_total,
        "valid_counts": {k: int(v) for k, v in valid_counts.items()},
        "n_valid": int(len(label)),
        "pr_auc": rec["pr_auc"],
        "f1_macro": rec["f1_macro"],
        "accuracy": rec["accuracy"],
        "auroc": rec["auroc"],
        "confusion": rec["confusion"],
        "reported": reported,
        "reported_vs_recomputed_match": bool(rep_recomp_match),
        "boundary": {
            "pred_dist": pred_dist,
            "manifest_dist": manifest_dist,
            "manifest_total": manifest_total,
            "n_pred": int(len(label)),
            "dist_match": bool(dist_match),
        },
    }
    print(f"r={ratio:.1f} ({name}): train/cls={train_per_class} (tot {train_total}) | "
          f"PR-AUC={rec['pr_auc']:.4f} F1={rec['f1_macro']:.4f} "
          f"acc={rec['accuracy']:.4f} AUROC={rec['auroc']:.4f} | "
          f"best_ep={int(rep['epoch'])} | dist_match={dist_match} "
          f"recomp_match={rep_recomp_match}", flush=True)

dump = {
    "task": "stability_train_ratio_sweep",
    "model": ("Ours = DINOv3 ViT-B/16 frozen @512 + 2-layer head(hidden512) + "
              "strong aug + focal(gamma=2), 3-class (normal_d3_d4)"),
    "setting": SETTING,
    "num_classes": NUM_CLASSES,
    "valid": "original distribution (balance_valid=False), N=1403 (normal 1303 / d3 76 / d4 24)",
    "primary_metric": "PR-AUC (disease d3,d4 OvR macro)",
    "train_ratios": [r for r, _ in RUNS],
    "note": (
        "train_ratio sub-samples the (already balanced-downsampled) train set; "
        "the train loader balances per class so train_counts are equal across "
        "classes at each ratio. r=1.0 is the existing Ours focal+aug 3-class run "
        "(report section 6 canonical). No retraining: metrics.json + "
        "predictions/valid.npz used; best.pt never loaded. PR-AUC monitored for "
        "early-stop; F1-macro is argmax-threshold(0.5)-dependent and sensitive to "
        "the d4 N=24 minority class."
    ),
    "per_ratio": per_ratio,
    "sklearn_crosscheck": crosscheck,
}
(EVAL / "stability_dinov3.json").write_text(json.dumps(dump, indent=2, default=float))
print("\nwrote _workspace/eval/stability_dinov3.json")
