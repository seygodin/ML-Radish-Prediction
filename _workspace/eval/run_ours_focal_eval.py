#!/usr/bin/env python
"""eval-reporter (QA): Ours+ improved = DINOv3 ViT-B/16 (frozen) @512 + 2-layer
head(hidden512) + STRONG augmentation + FOCAL loss (gamma=2, class_weights=from_meta).

Evaluates the focal+aug "Ours+ improved" runs
(experiments/dinov3_base_focal_{setting}) on BOTH the original-distribution valid
set and the balanced valid set (1:1 / 1:1:1, seed=42), with independent sklearn
metric recomputation, boundary cross-validation (predictions vs manifest),
params/trainable counts, +20% goal adjudication, per-class recall/precision
(d3/d4), and per-setting verify files.

It then MERGES the focal results with the existing small (@256) and base-CE
(@512, default aug) results already in _workspace/eval/ours_dinov3.json so that
the JSON carries ALL THREE variants for the comparison report/figure, plus a
base-CE -> base-focal delta block (the variable changed is aug+focal *together*,
so the delta attributes to the joint change, NOT to either factor alone).

best.pt weights are loaded for forward only (never modified). img_size=512 (from
config.snapshot) is honored for the balanced re-evaluation.

Run: ./.venv/bin/python _workspace/eval/run_ours_focal_eval.py
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

SETTINGS = ["normal_vs_d3", "normal_vs_d4", "normal_d3_d4"]
SEED = 42

from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


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
    disease_idx = list(range(1, num_classes))
    if num_classes == 2:
        score = probs[:, 1]
        auroc = float(roc_auc_score((labels == 1).astype(int), score))
        pr_auc = float(average_precision_score((labels == 1).astype(int), score))
    else:
        aucs, aps = [], []
        for c in range(num_classes):
            y_c = (labels == c).astype(np.int64)
            if 0 < y_c.sum() < len(y_c):
                aucs.append(float(roc_auc_score(y_c, probs[:, c])))
        auroc = float(np.mean(aucs)) if aucs else float("nan")
        for c in disease_idx:
            y_c = (labels == c).astype(np.int64)
            if 0 < y_c.sum() < len(y_c):
                aps.append(float(average_precision_score(y_c, probs[:, c])))
        pr_auc = float(np.mean(aps)) if aps else float("nan")
    cm = confusion_matrix(labels, preds, labels=cls).tolist()
    return {"accuracy": acc, "recall_macro": recm, "precision_macro": precm,
            "f1_macro": f1m, "auroc": auroc, "pr_auc": pr_auc, "confusion": cm}


def per_class_recall_precision(cm, num_classes):
    """클래스별 recall·precision 계산."""
    cm = np.asarray(cm, dtype=np.float64)
    out = {}
    for c in range(num_classes):
        tp = cm[c, c]
        row = cm[c, :].sum()   # actual class c
        col = cm[:, c].sum()   # predicted class c
        rec = tp / row if row else float("nan")
        prec = tp / col if col else float("nan")
        out[c] = {"recall": float(rec), "precision": float(prec),
                  "support": int(row), "tp": int(tp)}
    return out


# ---------------------------------------------------------------------------
# baseline best (from existing eval dumps) — original-dist + balanced
# ---------------------------------------------------------------------------
BASELINE_ARCHS = ["convnextv2", "efficientnetv2", "nextvit",
                  "densenet121", "resnet50", "mamba"]
orig_summary = json.load(open(EVAL / "summary.json"))["classification"]
bal_summary = json.load(open(EVAL / "balanced_valid.json"))["results"]


def baseline_best(setting):
    """세팅별 baseline 최고 지표를 추출(비교 기준)."""
    out = {}
    pr_rows = [(a, orig_summary[f"{a}_{setting}"]["prauc"]) for a in BASELINE_ARCHS]
    f1_rows = [(a, orig_summary[f"{a}_{setting}"]["f1_macro_recomp"]) for a in BASELINE_ARCHS]
    bp = max(pr_rows, key=lambda r: r[1]); bf = max(f1_rows, key=lambda r: r[1])
    out["orig"] = {"pr_auc": (bp[0], bp[1]), "f1_macro": (bf[0], bf[1])}
    pr_rows = [(a, bal_summary[f"{a}_{setting}"]["pr_auc"]) for a in BASELINE_ARCHS]
    f1_rows = [(a, bal_summary[f"{a}_{setting}"]["f1_macro"]) for a in BASELINE_ARCHS]
    bp = max(pr_rows, key=lambda r: r[1]); bf = max(f1_rows, key=lambda r: r[1])
    out["bal"] = {"pr_auc": (bp[0], bp[1]), "f1_macro": (bf[0], bf[1])}
    return out


# ---------------------------------------------------------------------------
# manifest cross-check
# ---------------------------------------------------------------------------
import pandas as pd  # noqa: E402
mcls = pd.read_csv(ROOT / "_workspace" / "data" / "manifest_classification.csv")
mcls_valid = mcls[mcls.split == "valid"]


def manifest_valid_dist(setting):
    """manifest의 valid 라벨 분포 반환(정합성 교차검증용)."""
    lab = "label_" + setting
    vv = mcls_valid[mcls_valid[lab] >= 0]
    return vv[lab].value_counts().sort_index().to_dict(), len(vv)


# ---------------------------------------------------------------------------
# forward (balanced valid) with best.pt
# ---------------------------------------------------------------------------
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from src.data import build_classification_loaders  # noqa: E402
from src.models import build_classifier  # noqa: E402


@torch.no_grad()
def forward_collect(model, loader, device, num_classes):
    """모델로 valid를 forward해 확률/정답을 수집."""
    model.eval()
    all_probs, all_labels, loss_sum, n = [], [], 0.0, 0
    use_amp = device.type == "cuda"
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels_d = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
        vloss = F.cross_entropy(logits.float(), labels_d)
        loss_sum += float(vloss) * images.size(0)
        n += images.size(0)
        all_probs.append(F.softmax(logits.float(), dim=1).cpu().numpy())
        all_labels.append(labels.numpy())
    return (np.concatenate(all_probs), np.concatenate(all_labels),
            loss_sum / max(1, n))


def count_params(model):
    """모델의 total/trainable 파라미터 수 반환."""
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, train


# ---------------------------------------------------------------------------
# main: evaluate Ours+ focal+aug (dinov3_base_focal @512)
# ---------------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
focal_results = {}
sk_crosscheck_focal = {}

for setting in SETTINGS:
    name = f"dinov3_base_focal_{setting}"
    run_dir = EXP / name
    snap = json.loads((run_dir / "config.snapshot").read_text())
    mj = json.loads((run_dir / "metrics.json").read_text())
    m_model = snap["spec"]["model"]
    num_classes = int(m_model["num_classes"])
    img_size = int(m_model["img_size"])
    head_hidden = int(m_model.get("head_hidden", 512))
    aug = str(snap["spec"]["data"].get("aug", "default"))
    loss_cfg = snap.get("loss", {})
    assert m_model["arch"] == "dinov3_base", m_model["arch"]
    assert aug == "strong", aug
    assert loss_cfg.get("type") == "focal", loss_cfg

    best_ep = int(mj["final"]["epoch"])
    pe_by_ep = {e["epoch"]: e for e in mj["per_epoch"]}
    train_loss = float(pe_by_ep[best_ep]["train_loss"])
    orig_val_loss = float(pe_by_ep[best_ep]["val_loss"])

    # ---- (1) ORIGINAL-DIST: recompute from saved predictions ----
    z = np.load(run_dir / "predictions" / "valid.npz", allow_pickle=True)
    prob = z["prob"]; label = z["label"].astype(int)
    class_names = [str(x) for x in z["class_names"]]
    orig = metrics_from_probs(prob, label, num_classes)
    orig_pc = per_class_recall_precision(orig["confusion"], num_classes)

    mdist, mtot = manifest_valid_dist(setting)
    pred_dist = {int(c): int((label == c).sum()) for c in sorted(set(label.tolist()))}
    dist_match = (pred_dist == {int(k_): int(v_) for k_, v_ in mdist.items()}) and (len(label) == mtot)

    rep = mj["final"]
    rep_recomp = {
        "pr_auc": (rep["pr_auc"], orig["pr_auc"]),
        "f1_macro": (rep["f1_macro"], orig["f1_macro"]),
        "recall_macro": (rep["recall_macro"], orig["recall_macro"]),
        "precision_macro": (rep["precision_macro"], orig["precision_macro"]),
        "auroc": (rep["auroc"], orig["auroc"]),
        "accuracy": (rep["accuracy"], orig["accuracy"]),
    }

    # ---- (2) BALANCED: load best.pt, forward (img_size=512) ----
    _, valid_loader, meta = build_classification_loaders(
        setting=setting, img_size=img_size, batch_size=32,
        num_workers=8, seed=SEED, balance_valid=True)
    model = build_classifier(arch="dinov3_base", num_classes=num_classes,
                             img_size=img_size, head_hidden=head_hidden).to(device)
    total_p, train_p = count_params(model)
    ckpt = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    b_prob, b_label, b_val_loss = forward_collect(model, valid_loader, device, num_classes)
    bal = metrics_from_probs(b_prob, b_label, num_classes)
    bal_pc = per_class_recall_precision(bal["confusion"], num_classes)
    bal_counts = meta["valid_counts"]

    if setting == "normal_vs_d3":
        sk_crosscheck_focal = {
            "setting": setting,
            "pr_auc_sklearn": float(average_precision_score((label == 1).astype(int), prob[:, 1])),
            "auroc_sklearn": float(roc_auc_score((label == 1).astype(int), prob[:, 1])),
            "reported_pr_auc": rep["pr_auc"], "reported_auroc": rep["auroc"],
        }

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    focal_results[name] = {
        "arch": "dinov3_base",
        "variant": "Ours+ focal+aug (DINOv3 ViT-B/16 frozen @512 + strong aug + focal gamma2)",
        "aug": aug, "loss": loss_cfg,
        "setting": setting, "num_classes": num_classes,
        "img_size": img_size, "head_hidden": head_hidden, "best_epoch": best_ep,
        "params_total_M": round(total_p / 1e6, 4),
        "params_trainable_M": round(train_p / 1e6, 4),
        "params_trainable_k": round(train_p / 1e3, 2),
        "class_names": class_names,
        "train_loss": train_loss,
        "orig": {**orig, "val_loss": orig_val_loss, "n_valid": int(len(label)),
                 "label_dist": pred_dist, "per_class": orig_pc},
        "bal": {**bal, "val_loss": b_val_loss, "n_valid": int(sum(bal_counts.values())),
                "counts": bal_counts, "per_class": bal_pc},
        "boundary": {"pred_dist": pred_dist, "manifest_dist": mdist,
                     "manifest_total": mtot, "n_pred": int(len(label)),
                     "dist_match": bool(dist_match)},
        "reported_vs_recomputed_orig": rep_recomp,
        "baseline_best": baseline_best(setting),
    }
    print(f"{name}: ORIG pr={orig['pr_auc']:.4f} f1={orig['f1_macro']:.4f} "
          f"auroc={orig['auroc']:.4f} acc={orig['accuracy']:.4f} | "
          f"BAL pr={bal['pr_auc']:.4f} f1={bal['f1_macro']:.4f} N={focal_results[name]['bal']['n_valid']} | "
          f"params total={focal_results[name]['params_total_M']}M train={focal_results[name]['params_trainable_k']}k | "
          f"dist_match={dist_match}", flush=True)

# ---------------------------------------------------------------------------
# load existing merged JSON (small + base-CE) and compute deltas
# ---------------------------------------------------------------------------
prev = json.load(open(EVAL / "ours_dinov3.json"))
GOAL = prev.get("goal_targets_3class_orig", {"pr_auc": 0.684, "f1_macro": 0.851})
prev_results = prev["results"]
small_results = {k: v for k, v in prev_results.items()
                 if k.startswith("dinov3_") and not k.startswith("dinov3_base")}
baseCE_results = {k: v for k, v in prev_results.items()
                  if k.startswith("dinov3_base_") and "focal" not in k}

# uplift vs baseline best (focal)
uplift_focal = {}
for setting in SETTINGS:
    r = focal_results[f"dinov3_base_focal_{setting}"]
    bb = r["baseline_best"]
    u = {}
    for split in ["orig", "bal"]:
        ours_pr = r[split]["pr_auc"]; ours_f1 = r[split]["f1_macro"]
        bpr_a, bpr_v = bb[split]["pr_auc"]; bf1_a, bf1_v = bb[split]["f1_macro"]
        u[split] = {
            "ours_pr_auc": ours_pr, "baseline_pr_auc": bpr_v, "baseline_pr_arch": bpr_a,
            "pr_abs": ours_pr - bpr_v, "pr_rel_pct": 100.0 * (ours_pr - bpr_v) / bpr_v,
            "ours_f1": ours_f1, "baseline_f1": bf1_v, "baseline_f1_arch": bf1_a,
            "f1_abs": ours_f1 - bf1_v, "f1_rel_pct": 100.0 * (ours_f1 - bf1_v) / bf1_v,
        }
    uplift_focal[setting] = u

# +20% verdict (focal, 3-class)
r3 = focal_results["dinov3_base_focal_normal_d3_d4"]
goal_verdict_focal = {
    "orig": {
        "pr_auc_value": r3["orig"]["pr_auc"], "pr_auc_target": GOAL["pr_auc"],
        "pr_auc_met": r3["orig"]["pr_auc"] >= GOAL["pr_auc"],
        "f1_macro_value": r3["orig"]["f1_macro"], "f1_macro_target": GOAL["f1_macro"],
        "f1_macro_met": r3["orig"]["f1_macro"] >= GOAL["f1_macro"],
    }
}

# base-CE -> base-focal delta (JOINT aug+focal change; NOT attributable to either alone)
ce_to_focal = {}
for setting in SETTINGS:
    ce = baseCE_results[f"dinov3_base_{setting}"]
    fo = focal_results[f"dinov3_base_focal_{setting}"]
    d = {}
    for split in ["orig", "bal"]:
        d[split] = {
            "pr_auc_ce": ce[split]["pr_auc"], "pr_auc_focal": fo[split]["pr_auc"],
            "pr_auc_delta": fo[split]["pr_auc"] - ce[split]["pr_auc"],
            "f1_ce": ce[split]["f1_macro"], "f1_focal": fo[split]["f1_macro"],
            "f1_delta": fo[split]["f1_macro"] - ce[split]["f1_macro"],
        }
    # per-class d3/d4 precision/recall (orig) CE vs focal
    pc = {}
    for c in range(1, fo["num_classes"]):
        pc[c] = {
            "ce_precision": ce["orig"]["per_class"][str(c)]["precision"],
            "focal_precision": fo["orig"]["per_class"][c]["precision"],
            "ce_recall": ce["orig"]["per_class"][str(c)]["recall"],
            "focal_recall": fo["orig"]["per_class"][c]["recall"],
            "support": fo["orig"]["per_class"][c]["support"],
        }
    d["per_class_orig"] = pc
    ce_to_focal[setting] = d

# ---------------------------------------------------------------------------
# merge: write small + base-CE + base-focal into ours_dinov3.json
# ---------------------------------------------------------------------------
merged_results = {**small_results, **baseCE_results, **focal_results}
dump = dict(prev)  # preserve everything else
dump["results"] = merged_results
dump["uplift_focal"] = uplift_focal
dump["goal_verdict_3class_focal"] = goal_verdict_focal
dump["ce_to_focal_improvement"] = ce_to_focal
dump["sklearn_crosscheck_focal"] = sk_crosscheck_focal
dump["focal_note"] = (
    "Ours+ focal+aug = DINOv3 ViT-B/16 frozen @512 + 2-layer head(hidden512) + "
    "strong augmentation + focal loss (gamma=2, class_weights=from_meta). "
    "TWO variables (aug strong + focal) changed simultaneously vs base-CE "
    "(default aug + CE); the ce->focal delta is the JOINT effect and is NOT "
    "attributable to either factor alone (ablation required to separate). "
    "Note: from_meta class_weights resolved to all-ones because the train loader "
    "is balanced-downsampled, so focal's gamma (hard-example focusing) is the "
    "active mechanism, not alpha re-weighting."
)
(EVAL / "ours_dinov3.json").write_text(json.dumps(dump, indent=2, default=float))
print("\nwrote _workspace/eval/ours_dinov3.json (small + base-CE + base-focal merged)")

# ---------------------------------------------------------------------------
# verify files (one per focal setting)
# ---------------------------------------------------------------------------
for setting in SETTINGS:
    r = focal_results[f"dinov3_base_focal_{setting}"]
    ce = baseCE_results[f"dinov3_base_{setting}"]
    b = r["boundary"]; k = r["num_classes"]
    L = [f"# verify_dinov3_base_focal_{setting}.md", "",
         f"- task: classification ({setting}), backbone: **dinov3_base focal+aug "
         f"(Ours+ improved = DINOv3 ViT-B/16 frozen @512 + 2-layer head hidden=512 "
         f"+ strong aug + focal loss gamma=2, class_weights=from_meta)**",
         f"- img_size = **{r['img_size']}** (from config.snapshot), num_classes = {k}, "
         f"best epoch = {r['best_epoch']}, aug = **{r['aug']}**, "
         f"loss = **focal(gamma={r['loss'].get('gamma')}, class_weights={r['loss'].get('class_weights')})**",
         f"- params: total = **{r['params_total_M']}M** (frozen ViT-B/16 backbone), "
         f"trainable = **{r['params_trainable_k']}k** (head only; identical to base-CE)",
         "- NOTE: `from_meta` class_weights resolved to all-ones (train loader is "
         "balanced-downsampled) -> focal's **gamma hard-example focusing** is the "
         "active mechanism, not alpha re-weighting. **aug+focal changed together** "
         "vs base-CE -> the delta below is the JOINT effect (ablation needed to split).",
         ""]
    L += ["## 1) Boundary cross-validation (predictions vs manifest, ORIGINAL dist)", "",
          f"- predictions valid N = **{b['n_pred']}**, manifest valid N = **{b['manifest_total']}** -> "
          f"{'MATCH' if b['n_pred']==b['manifest_total'] else 'MISMATCH'}",
          f"- predictions label dist = {b['pred_dist']}",
          f"- manifest valid label dist = {b['manifest_dist']}",
          f"- distribution match: **{'PASS' if b['dist_match'] else 'FAIL'}**",
          f"- class_names from npz: {r['class_names']}", ""]
    L += ["## 2) Independent metric recomputation vs reported (ORIGINAL dist)", "",
          "Recomputed with sklearn from predictions/valid.npz; cross-checked vs metrics.json final.", "",
          "| metric | reported | recomputed (sklearn) | match |", "|---|---|---|---|"]
    for m, (rv, cv) in r["reported_vs_recomputed_orig"].items():
        ok = abs(rv - cv) <= 0.01
        L.append(f"| {m} | {rv:.4f} | {cv:.4f} | {'ok' if ok else '**DIFF**'} |")
    L += ["",
          f"- ORIGINAL 7-metric: acc={r['orig']['accuracy']:.4f}, train_loss={r['train_loss']:.4f} (focal), "
          f"val_loss={r['orig']['val_loss']:.4f} (focal), recall_macro={r['orig']['recall_macro']:.4f}, "
          f"precision_macro={r['orig']['precision_macro']:.4f}, f1_macro={r['orig']['f1_macro']:.4f}, "
          f"AUROC={r['orig']['auroc']:.4f}; PR-AUC={r['orig']['pr_auc']:.4f}",
          f"- confusion (orig, recomputed): {r['orig']['confusion']}", ""]
    L += ["### per-class recall/precision (ORIGINAL dist) -- F1 bottleneck diagnosis", "",
          "| class | support | recall | precision |", "|---|---|---|---|"]
    for c in range(k):
        pc = r["orig"]["per_class"][c]
        L.append(f"| {r['class_names'][c]} | {pc['support']} | {pc['recall']:.3f} | {pc['precision']:.3f} |")
    L += [""]
    # CE vs focal per-class (orig)
    L += ["### CE (base, default aug) -> focal+aug per-class change (ORIGINAL dist)", "",
          "| class | support | recall CE | recall focal | precision CE | precision focal |",
          "|---|---|---|---|---|---|"]
    for c in range(1, k):
        d = ce_to_focal[setting]["per_class_orig"][c]
        L.append(f"| {r['class_names'][c]} | {d['support']} | {d['ce_recall']:.3f} | "
                 f"{d['focal_recall']:.3f} | {d['ce_precision']:.3f} | {d['focal_precision']:.3f} |")
    L += [""]
    L += ["## 3) Balanced valid re-evaluation (best.pt loaded, weights unchanged; img_size=512)", "",
          f"- balanced valid counts = {r['bal']['counts']} (N={r['bal']['n_valid']}, seed={SEED})",
          f"- BALANCED 7-metric: acc={r['bal']['accuracy']:.4f}, "
          f"val_loss={r['bal']['val_loss']:.4f} (plain CE on balanced), "
          f"recall_macro={r['bal']['recall_macro']:.4f}, precision_macro={r['bal']['precision_macro']:.4f}, "
          f"f1_macro={r['bal']['f1_macro']:.4f}, AUROC={r['bal']['auroc']:.4f}; PR-AUC={r['bal']['pr_auc']:.4f}",
          f"- confusion (balanced): {r['bal']['confusion']}", ""]
    cm = np.array(r["orig"]["confusion"])
    L += ["## 4) Small-sample note (Wilson 95% CI on disease recall/precision, ORIGINAL dist)", ""]
    for c in (range(1, k) if k > 2 else [1]):
        nt = int(cm[c, :].sum()); ntp = int(cm[c, c]); ncol = int(cm[:, c].sum())
        lo, hi = wilson_ci(ntp, nt)
        plo, phi = wilson_ci(ntp, ncol)
        L.append(f"- recall[{r['class_names'][c]}] = {ntp}/{nt} = "
                 f"{(ntp/nt if nt else float('nan')):.3f} (95% CI {lo:.3f}-{hi:.3f}); "
                 f"precision[{r['class_names'][c]}] = {ntp}/{ncol} = "
                 f"{(ntp/ncol if ncol else float('nan')):.3f} (95% CI {plo:.3f}-{phi:.3f})")
    L += ["", "- NOTE: disease_4 valid N=24 (orig) is small -> CI wide; balanced 3-class N=72 -> per-class N=24.", ""]
    u = uplift_focal[setting]
    L += ["## 5) Uplift vs baseline best", ""]
    for split, lab in [("orig", "ORIGINAL dist"), ("bal", "BALANCED")]:
        uu = u[split]
        L += [f"### {lab}",
              f"- PR-AUC: focal+aug {uu['ours_pr_auc']:.4f} vs baseline best {uu['baseline_pr_auc']:.4f} "
              f"({uu['baseline_pr_arch']}) -> abs {uu['pr_abs']:+.4f}, rel {uu['pr_rel_pct']:+.1f}%",
              f"- F1-macro: focal+aug {uu['ours_f1']:.4f} vs baseline best {uu['baseline_f1']:.4f} "
              f"({uu['baseline_f1_arch']}) -> abs {uu['f1_abs']:+.4f}, rel {uu['f1_rel_pct']:+.1f}%", ""]
    # ce->focal
    c2f = ce_to_focal[setting]
    L += ["## 6) base-CE (default aug) -> base-focal+aug improvement (JOINT aug+focal effect)", ""]
    for split, lab in [("orig", "ORIGINAL dist"), ("bal", "BALANCED")]:
        d = c2f[split]
        L += [f"### {lab}",
              f"- PR-AUC: CE {d['pr_auc_ce']:.4f} -> focal+aug {d['pr_auc_focal']:.4f} (Δ {d['pr_auc_delta']:+.4f})",
              f"- F1-macro: CE {d['f1_ce']:.4f} -> focal+aug {d['f1_focal']:.4f} (Δ {d['f1_delta']:+.4f})", ""]
    (EVAL / f"verify_dinov3_base_focal_{setting}.md").write_text("\n".join(L))
    print(f"wrote verify_dinov3_base_focal_{setting}.md")

print("\nsklearn crosscheck focal (normal_vs_d3 orig):", json.dumps(sk_crosscheck_focal, default=float))
print("3-class goal verdict focal (orig):", json.dumps(goal_verdict_focal, default=float))
print("3-class CE->focal (orig):", json.dumps(ce_to_focal["normal_d3_d4"]["orig"], default=float))
print("3-class CE->focal per-class (orig):", json.dumps(ce_to_focal["normal_d3_d4"]["per_class_orig"], default=float))
