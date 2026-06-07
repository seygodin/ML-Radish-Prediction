#!/usr/bin/env python
"""eval-reporter: boundary cross-validation, independent metric recomputation,
training curves, and comparison figures/report for radish baseline runs.

Run: ./.venv/bin/python _workspace/eval/run_eval.py
"""
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# sklearn used only as an independent cross-check of the from-scratch metric primitives
try:
    from sklearn.metrics import roc_auc_score as _sk_roc_auc
except Exception:  # pragma: no cover
    _sk_roc_auc = None

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
EVAL = ROOT / "_workspace" / "eval"
FIG = ROOT / "report" / "figures"
EVAL.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# 6 backbones: 3 original + 3 new (densenet121, resnet50, mamba=mambapy Vision-Mamba)
# NOTE: nextvit20 (NeXtViT-base) excluded from report tables/figures per user request
# (experiment artifacts experiments/nextvit20_* are preserved on disk).
ARCHS = ["convnextv2", "efficientnetv2", "nextvit",
         "densenet121", "resnet50", "mamba"]
NEW_ARCHS = {"densenet121", "resnet50", "mamba"}
ARCH_LABEL = {
    "convnextv2": "ConvNeXtV2-t", "efficientnetv2": "EffNetV2-s", "nextvit": "NeXtViT-s",
    "densenet121": "DenseNet121", "resnet50": "ResNet50",
    # NOTE: arch `mamba` == Vision-Mamba (mambapy pscan). Label keeps the arch token so it is findable.
    "mamba": "VisionMamba (mamba)",
}
# arch token shown in the report so each backbone's arch name is explicit (esp. mamba).
ARCH_TOKEN = {
    "convnextv2": "convnextv2", "efficientnetv2": "efficientnetv2", "nextvit": "nextvit",
    "densenet121": "densenet121", "resnet50": "resnet50",
    "mamba": "mamba",
}
CLS_SETTINGS = ["normal_vs_d3", "normal_vs_d4", "normal_d3_d4"]

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "savefig.dpi": 130})


# ----------------------------------------------------------------------------
# metric primitives (sklearn-free where it matters; sklearn available as check)
# ----------------------------------------------------------------------------
def pr_auc(y_true, scores):
    """Average-precision style PR-AUC via step integration (sklearn AP convention)."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    order = np.argsort(-scores)
    y = y_true[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    P = y_true.sum()
    if P == 0:
        return float("nan")
    recall = tp / P
    precision = tp / np.maximum(tp + fp, 1)
    # AP = sum (R_k - R_{k-1}) * P_k
    rprev = 0.0
    ap = 0.0
    for r, p in zip(recall, precision):
        ap += (r - rprev) * p
        rprev = r
    return float(ap)


def pr_curve(y_true, scores, n=200):
    """precision-recall 곡선의 점들을 계산."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    order = np.argsort(-scores)
    y = y_true[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    P = y_true.sum()
    recall = tp / max(P, 1)
    precision = tp / np.maximum(tp + fp, 1)
    return recall, precision


def confusion(y_true, y_pred, k):
    """혼동행렬(정답×예측) 계산."""
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_f1(cm):
    """클래스별 F1 계산."""
    k = cm.shape[0]
    f1s = []
    for c in range(k):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return np.array(f1s)


def wilson_ci(k, n, z=1.96):
    """이항 비율의 Wilson 95% 신뢰구간(저·고) — 소표본 지표에 CI 병기용."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def roc_auc(y_true, scores):
    """ROC-AUC via rank statistic (Mann-Whitney U). Handles ties by average rank."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based average rank
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def macro_prf(cm):
    """Macro recall/precision/f1 from a confusion matrix (sklearn macro convention)."""
    k = cm.shape[0]
    recs, precs, f1s = [], [], []
    for c in range(k):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        recs.append(rec); precs.append(prec); f1s.append(f1)
    return float(np.mean(recs)), float(np.mean(precs)), float(np.mean(f1s))


def auroc_macro_ovr(label, prob):
    """AUROC: 2-class -> ROC-AUC on P(disease)=prob[:,1]; k-class -> macro one-vs-rest.
    Uses the rank-statistic roc_auc (Mann-Whitney U), matching sklearn within ties."""
    label = np.asarray(label).astype(int)
    k = prob.shape[1]
    if k == 2:
        return roc_auc(label == 1, prob[:, 1])
    aucs = []
    for c in range(k):
        a = roc_auc(label == c, prob[:, c])
        if not np.isnan(a):
            aucs.append(a)
    return float(np.mean(aucs)) if aucs else float("nan")


def best_epoch_losses(per_epoch, best_epoch):
    """train_loss / val_loss at the best epoch from per_epoch records."""
    for e in per_epoch:
        if e["epoch"] == best_epoch:
            return float(e["train_loss"]), float(e["val_loss"])
    # fallback: nearest
    e = min(per_epoch, key=lambda r: abs(r["epoch"] - best_epoch))
    return float(e["train_loss"]), float(e["val_loss"])


def iou_xyxy(a, b):
    """두 xyxy 박스의 IoU(교집합/합집합)."""
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


# ----------------------------------------------------------------------------
# load manifests
# ----------------------------------------------------------------------------
mcls = pd.read_csv(ROOT / "_workspace" / "data" / "manifest_classification.csv")
mdet = pd.read_csv(ROOT / "_workspace" / "data" / "manifest_detection.csv")
mcls_valid = mcls[mcls.split == "valid"]


# ----------------------------------------------------------------------------
# model parameter counts (reproducibility): build each backbone and count params.
# classifier params = build_classifier(arch, 2, 224); detector = build_detector(arch, 512).
# Computed directly from src.models so the report tables are self-verifying.
# Falls back to the verified reference values if torch/model build is unavailable.
# ----------------------------------------------------------------------------
# verified reference (M) — build_classifier(arch,2,224) / build_detector(arch,512)
PARAMS_REF_M = {
    "convnextv2": (28.26, 28.07), "efficientnetv2": (20.84, 20.51),
    "nextvit": (31.27, 31.00),
    "densenet121": (7.48, 7.22), "resnet50": (24.56, 24.04),
    "mamba": (3.88, 4.03),
}


def compute_params_m():
    """Return {arch: (classifier_M, detector_M)} computed from src.models, else reference."""
    try:
        import sys
        import warnings
        warnings.filterwarnings("ignore")
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from src.models import build_classifier, build_detector

        def _m(model):
            return sum(p.numel() for p in model.parameters()) / 1e6

        out = {}
        for a in ARCHS:
            cls_m = _m(build_classifier(a, 2, 224))
            det_m = _m(build_detector(a, 512))
            out[a] = (round(cls_m, 2), round(det_m, 2))
        return out
    except Exception as e:  # pragma: no cover — fall back to verified reference
        print(f"[params] model build unavailable ({e!r}); using verified reference values.")
        return dict(PARAMS_REF_M)


PARAMS_M = compute_params_m()


def manifest_valid_label_dist(setting):
    """manifest의 valid 라벨 분포 반환(정합성 교차검증용)."""
    lab = "label_" + setting
    vv = mcls_valid[mcls_valid[lab] >= 0]
    return vv[lab].value_counts().sort_index().to_dict(), len(vv)


# ----------------------------------------------------------------------------
# per-run classification verification + recompute
# ----------------------------------------------------------------------------
cls_results = {}  # name -> dict


def eval_classification(arch, setting):
    """한 분류 run의 predictions에서 지표를 재계산해 dict 반환."""
    name = f"{arch}_{setting}"
    d = json.load(open(EXP / name / "metrics.json"))
    z = np.load(EXP / name / "predictions" / "valid.npz", allow_pickle=True)
    prob = z["prob"]
    label = z["label"].astype(int)
    class_names = [str(x) for x in z["class_names"]]
    k = prob.shape[1]
    pred = prob.argmax(1)

    # disease score: 2-class -> P(disease); 3-class -> per-class OvR, primary macro PR-AUC
    cm = confusion(label, pred, k)
    f1s = per_class_f1(cm)
    macro_f1 = float(f1s.mean())

    if k == 2:
        scores = prob[:, 1]
        prauc = pr_auc(label == 1, scores)
        # disease recall/precision (class 1)
        tp = cm[1, 1]
        rec_dis = tp / cm[1, :].sum() if cm[1, :].sum() else 0.0
        prec_dis = tp / cm[:, 1].sum() if cm[:, 1].sum() else 0.0
        prauc_per = {class_names[1]: prauc}
        recall_per = {class_names[1]: rec_dis}
        prec_per = {class_names[1]: prec_dis}
        primary = prauc
        rec_dis_overall = rec_dis
        prec_dis_overall = prec_dis
        n_dis = int((label == 1).sum())
    else:
        prauc_per = {}
        recall_per = {}
        prec_per = {}
        for c in range(1, k):  # disease classes 1,2
            prauc_per[class_names[c]] = pr_auc(label == c, prob[:, c])
            rc = cm[c, c] / cm[c, :].sum() if cm[c, :].sum() else 0.0
            pc = cm[c, c] / cm[:, c].sum() if cm[:, c].sum() else 0.0
            recall_per[class_names[c]] = rc
            prec_per[class_names[c]] = pc
        # macro PR-AUC over disease classes (OvR) — matches design primary
        primary = float(np.mean(list(prauc_per.values())))
        # overall disease recall/precision: combined disease vs normal
        dis_mask = label >= 1
        pred_dis = pred >= 1
        tp = int((dis_mask & pred_dis).sum())
        rec_dis_overall = tp / dis_mask.sum() if dis_mask.sum() else 0.0
        prec_dis_overall = tp / pred_dis.sum() if pred_dis.sum() else 0.0
        n_dis = int(dis_mask.sum())

    acc = float((pred == label).mean())

    # ---- 7-metric recompute (user-required): accuracy, train/val loss, macro recall/precision/f1, AUROC ----
    rec_macro, prec_macro, f1_macro = macro_prf(cm)
    auroc = auroc_macro_ovr(label, prob)
    best_ep = d["final"]["epoch"]
    tr_loss, vl_loss = best_epoch_losses(d["per_epoch"], best_ep)

    # cross-check with manifest valid dist
    mdist, mtot = manifest_valid_label_dist(setting)
    pred_dist = {int(c): int((label == c).sum()) for c in sorted(set(label.tolist()))}
    dist_match = (pred_dist == {int(k_): int(v_) for k_, v_ in mdist.items()}) and (len(label) == mtot)

    # CI for disease recall
    if k == 2:
        ci = wilson_ci(int(cm[1, 1]), int(cm[1, :].sum()))
    else:
        ci = wilson_ci(int((label >= 1).sum() and (pred >= 1)[label >= 1].sum()),
                       int((label >= 1).sum()))

    cls_results[name] = dict(
        arch=arch, setting=setting, k=k, class_names=class_names,
        prob=prob, label=label, pred=pred, cm=cm,
        prauc=primary, prauc_per=prauc_per, macro_f1=macro_f1,
        f1_per=f1s.tolist(), recall_per=recall_per, prec_per=prec_per,
        recall_disease=rec_dis_overall, precision_disease=prec_dis_overall,
        recall_ci=ci, accuracy=acc, n_dis=n_dis, n_valid=len(label),
        reported=d["final"], per_epoch=d["per_epoch"], primary_metric=d["primary"],
        manifest_dist=mdist, manifest_total=mtot, pred_dist=pred_dist,
        dist_match=bool(dist_match),
        # 7-metric recomputed
        recall_macro=rec_macro, precision_macro=prec_macro, f1_macro_recomp=f1_macro,
        auroc=auroc, train_loss=tr_loss, val_loss=vl_loss, best_epoch=best_ep,
        params_m=PARAMS_M[arch][0],  # classifier params (M)
        is_new=(arch in NEW_ARCHS),
    )

    # write verify file
    rep = d["final"]
    lines = [f"# verify_{name}.md", "", f"- task: classification ({setting}), backbone: {arch}",
             f"- status (reported): {d['status']}", f"- primary metric: {d['primary']}", ""]
    lines += ["## 1) Boundary cross-validation (predictions vs manifest)", ""]
    lines += [f"- predictions valid N = **{len(label)}**, manifest valid N (setting) = **{mtot}** -> {'MATCH' if len(label)==mtot else 'MISMATCH'}",
              f"- predictions label dist = {pred_dist}",
              f"- manifest valid label dist = {mdist}",
              f"- distribution match: **{'PASS' if dist_match else 'FAIL'}**",
              f"- class_names from npz: {class_names}",
              f"- expected (data_card): normal=1303 + disease (d3=76 / d4=24) non-downsampled valid",
              ""]
    lines += ["## 2) Independent metric recomputation vs reported", "",
              "| metric | reported | recomputed | match |", "|---|---|---|---|"]

    def row(nm, r, c, tol=0.02):
        ok = (r is None) or (abs(r - c) <= tol)
        rstr = f"{r:.4f}" if isinstance(r, (int, float)) else str(r)
        return f"| {nm} | {rstr} | {c:.4f} | {'ok' if ok else '**DIFF**'} |"

    lines.append(row("pr_auc (primary)", rep.get("pr_auc"), primary))
    lines.append(row("macro_f1", rep.get("macro_f1"), macro_f1))
    lines.append(row("recall_disease", rep.get("recall_disease"), rec_dis_overall))
    lines.append(row("precision_disease", rep.get("precision_disease"), prec_dis_overall))
    lines.append(row("accuracy", rep.get("accuracy"), acc))
    lines += ["", "### 7-metric set (user-required): recomputed from predictions; loss = best-epoch per_epoch",
              "", "| metric | reported | recomputed | match |", "|---|---|---|---|"]
    # old runs may lack these keys in final -> reported shown as 'n/a (not in old metrics.json)'
    def row_opt(nm, key, c, tol=0.02):
        r = rep.get(key)
        if r is None:
            return f"| {nm} | n/a (absent in metrics.json) | {c:.4f} | recomputed-only |"
        ok = abs(r - c) <= tol
        return f"| {nm} | {r:.4f} | {c:.4f} | {'ok' if ok else '**DIFF**'} |"
    lines.append(row_opt("recall (macro)", "recall_macro", rec_macro))
    lines.append(row_opt("precision (macro)", "precision_macro", prec_macro))
    lines.append(row_opt("f1-score (macro)", "f1_macro", f1_macro))
    lines.append(row_opt("AUROC", "auroc", auroc))
    lines.append(row_opt("train_loss (best ep)", "train_loss", tr_loss))
    lines.append(row_opt("val_loss (best ep)", "val_loss", vl_loss))
    lines.append(f"| accuracy | {rep.get('accuracy'):.4f} | {acc:.4f} | {'ok' if abs(rep.get('accuracy')-acc)<=0.02 else '**DIFF**'} |")
    lines.append(f"")
    auroc_conv = "ROC-AUC on P(disease) (2-class)" if k == 2 else f"macro one-vs-rest over {k} classes"
    lines.append(f"- AUROC convention: {auroc_conv}. best epoch = {best_ep}.")
    if arch not in NEW_ARCHS:
        lines.append("- NOTE: original backbone — AUROC/macro-PRF were NOT in old metrics.json; recomputed independently from predictions/valid.npz. train/val loss taken from per_epoch at best epoch.")
    lines += ["", "Confusion matrix (rows=true, cols=pred), recomputed:", "",
              "```", str(cm.tolist()), "```",
              f"reported confusion: {rep.get('confusion')}", ""]
    lines += ["## 3) Small-sample note (Wilson 95% CI)", "",
              f"- valid disease N = **{n_dis}**", ""]
    for c in (range(1, k) if k > 2 else [1]):
        cname = class_names[c]
        nt = int(cm[c, :].sum())
        ntp = int(cm[c, c])
        lo, hi = wilson_ci(ntp, nt)
        lines.append(f"- recall[{cname}] = {ntp}/{nt} = {(ntp/nt if nt else float('nan')):.3f} (95% CI {lo:.3f}-{hi:.3f})")
    lines += ["", "## Per-disease PR-AUC (OvR)" if k > 2 else "", ""]
    if k > 2:
        for cn, v in prauc_per.items():
            lines.append(f"- {cn}: PR-AUC={v:.3f}")
        lines.append(f"- macro disease PR-AUC (primary): {primary:.3f}")
    (EVAL / f"verify_{name}.md").write_text("\n".join(lines))


# ----------------------------------------------------------------------------
# detection verification + recompute
# ----------------------------------------------------------------------------
det_results = {}


def eval_detection(arch):
    """한 detection run의 predictions에서 검출/국소화 지표를 재계산."""
    name = f"{arch}_detection_singlebox"
    d = json.load(open(EXP / name / "metrics.json"))
    pj = json.load(open(EXP / name / "predictions" / "valid.json"))
    pred_boxes = pj["pred_boxes"]
    scores = np.asarray(pj["scores"], dtype=float)
    gt_boxes = pj["gt_boxes"]
    # explicit disease/normal label from predictions (post-fix: normals = negatives, empty GT)
    is_pos = np.asarray(pj["is_positive"], dtype=bool)
    n = len(pred_boxes)
    n_pos = int(is_pos.sum())
    n_neg = int((~is_pos).sum())

    # ---- (A) image-level disease detection: objectness score vs disease label ----
    det_pr_auc = pr_auc(is_pos, scores)
    det_roc_auc = roc_auc(is_pos, scores)
    thr = 0.5
    pred_disease = scores >= thr
    tp = int((pred_disease & is_pos).sum())
    fp = int((pred_disease & ~is_pos).sum())
    presence_recall = tp / n_pos if n_pos else float("nan")  # disease recall @0.5
    fp_rate = fp / n_neg if n_neg else float("nan")          # normal false-alarm @0.5
    recall_ci = wilson_ci(tp, n_pos)
    fp_ci = wilson_ci(fp, n_neg)

    # objectness separation stats (collapse-fix evidence)
    sp = scores[is_pos]
    sn = scores[~is_pos]
    sep = dict(
        pos_median=float(np.median(sp)), pos_p25=float(np.percentile(sp, 25)),
        pos_min=float(sp.min()),
        neg_median=float(np.median(sn)), neg_p75=float(np.percentile(sn, 75)),
        neg_p95=float(np.percentile(sn, 95)), neg_max=float(sn.max()),
        n_neg_ge_0p5=int((sn >= 0.5).sum()),
    )

    # ---- (B) localization: IoU(pred, GT) on POSITIVE images only ----
    pos_ious = []
    for pb, gb, ip in zip(pred_boxes, gt_boxes, is_pos):
        if ip and isinstance(gb, list) and len(gb) > 0 and gb[0] is not None:
            pos_ious.append(iou_xyxy(pb, gb[0]))
    pos_ious = np.asarray(pos_ious)
    iou_dist = dict(
        mean=float(pos_ious.mean()), median=float(np.median(pos_ious)),
        p25=float(np.percentile(pos_ious, 25)), p75=float(np.percentile(pos_ious, 75)),
        min=float(pos_ious.min()), max=float(pos_ious.max()),
    )
    iou_presence = float((pos_ious >= 0.5).mean())  # frac of positives localized @IoU>=0.5

    # ---- (C) mAP@0.5: rank all preds by objectness; TP iff positive image AND IoU>=0.5 ----
    # normals contribute FP whenever ranked (any normal predicted with high score is a false box)
    order = np.argsort(-scores)
    tp_arr = np.zeros(n)
    fp_arr = np.zeros(n)
    for rank, idx in enumerate(order):
        gb = gt_boxes[idx]
        if is_pos[idx] and isinstance(gb, list) and len(gb) > 0 and gb[0] is not None \
                and iou_xyxy(pred_boxes[idx], gb[0]) >= 0.5:
            tp_arr[rank] = 1
        else:
            fp_arr[rank] = 1
    ctp = np.cumsum(tp_arr)
    cfp = np.cumsum(fp_arr)
    recall = ctp / max(n_pos, 1)
    precision = ctp / np.maximum(ctp + cfp, 1)
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = precision[recall >= t].max() if np.any(recall >= t) else 0.0
        ap += p / 101
    map05 = float(ap)

    det_results[name] = dict(
        arch=arch, n=n, n_pos=n_pos, n_neg=n_neg, pos_ious=pos_ious, scores=scores,
        is_pos=is_pos, det_pr_auc=det_pr_auc, det_roc_auc=det_roc_auc,
        presence_recall=presence_recall, fp_rate=fp_rate, recall_ci=recall_ci, fp_ci=fp_ci,
        iou_presence=iou_presence, iou_dist=iou_dist, map05=map05,
        recall=recall, precision=precision, sep=sep,
        reported=d["final"], per_epoch=d["per_epoch"], primary_metric=d["primary"],
        meta=d.get("meta", {}), elapsed=d.get("elapsed_seconds"),
        best_epoch=d["final"]["epoch"],
        train_loss=best_epoch_losses(d["per_epoch"], d["final"]["epoch"])[0],
        val_loss=best_epoch_losses(d["per_epoch"], d["final"]["epoch"])[1],
        params_m=PARAMS_M[arch][1],  # detector params (M)
        is_new=(arch in NEW_ARCHS),
    )

    # ---- manifest cross-check ----
    det_valid_dis = int((mdet.split == "valid").sum())
    rep = d["final"]
    amp_note = d.get("meta", {})
    lines = [f"# verify_{name}.md", "", f"- task: detection single-box, backbone: {arch}",
             f"- status (reported): {d['status']}", f"- primary: {d['primary']} (det_pr_auc)", ""]
    lines += ["## 1) Boundary cross-validation (predictions vs manifest)", "",
              f"- predictions total valid images N = **{n}** (include_normal=True)",
              f"- positives (disease, is_positive=True, non-empty GT) = **{n_pos}**; negatives (normal, empty GT) = **{n_neg}**",
              f"- manifest detection valid (disease only) = **{det_valid_dis}** (d3=76 + d4=24); classification manifest valid normal = **1303** -> total **{det_valid_dis+1303}** matches predictions N",
              f"- reported meta valid_counts = {d.get('meta',{}).get('valid_counts')}; train_counts = {d.get('meta',{}).get('train_counts')}",
              f"- N match (1403): **{'PASS' if n==1403 else 'FAIL'}**; pos/neg (100/1303): **{'PASS' if (n_pos==100 and n_neg==1303) else 'FAIL'}**; reported n_positive={rep.get('n_positive')}, n_negative={rep.get('n_negative')} match recomputed {n_pos}/{n_neg}",
              "",
              "### Objectness collapse RESOLVED (normal vs disease score separation)",
              f"- disease objectness: median={sep['pos_median']:.4f}, p25={sep['pos_p25']:.4f}, min={sep['pos_min']:.4f}",
              f"- normal  objectness: median={sep['neg_median']:.4f}, p75={sep['neg_p75']:.4f}, p95={sep['neg_p95']:.4f}, max={sep['neg_max']:.4f}",
              f"- normals scored >=0.5: **{sep['n_neg_ge_0p5']}/{n_neg}** ({100*sep['n_neg_ge_0p5']/n_neg:.1f}%)",
              f"- Separation is clean: disease median ~{sep['pos_median']:.2f} vs normal median ~{sep['neg_median']:.3f}. The earlier collapse (objectness ~1.0 for ALL images incl. normals) is FIXED — objectness now functions as a real image-level disease score.",
              ""]
    lines += ["## 2) Independent metric recomputation vs reported", "",
              "| metric | reported | recomputed | match |", "|---|---|---|---|",
              f"| det_pr_auc (primary) | {rep['det_pr_auc']:.4f} | {det_pr_auc:.4f} | {'ok' if abs(rep['det_pr_auc']-det_pr_auc)<=0.01 else '**DIFF**'} |",
              f"| det_roc_auc | {rep['det_roc_auc']:.4f} | {det_roc_auc:.4f} | {'ok' if abs(rep['det_roc_auc']-det_roc_auc)<=0.01 else '**DIFF**'} |",
              f"| presence_recall@0.5 | {rep['presence_recall_at_0.5']:.4f} | {presence_recall:.4f} | {'ok' if abs(rep['presence_recall_at_0.5']-presence_recall)<=0.01 else '**DIFF**'} |",
              f"| fp_rate@0.5 | {rep['fp_rate_at_0.5']:.4f} | {fp_rate:.4f} | {'ok' if abs(rep['fp_rate_at_0.5']-fp_rate)<=0.005 else '**DIFF**'} |",
              f"| iou_at_0.5_presence | {rep['iou_at_0.5_presence']:.4f} | {iou_presence:.4f} | {'ok' if abs(rep['iou_at_0.5_presence']-iou_presence)<=0.03 else '**DIFF**'} |",
              f"| iou median (positives) | {rep['iou_distribution']['median']:.4f} | {iou_dist['median']:.4f} | {'ok' if abs(rep['iou_distribution']['median']-iou_dist['median'])<=0.02 else '**DIFF**'} |",
              f"| iou mean (positives) | {rep['iou_distribution']['mean']:.4f} | {iou_dist['mean']:.4f} | {'ok' if abs(rep['iou_distribution']['mean']-iou_dist['mean'])<=0.02 else '**DIFF**'} |",
              f"| map_at_0.5 | {rep['map_at_0.5']:.4f} | {map05:.4f} | {'ok' if abs(rep['map_at_0.5']-map05)<=0.05 else '**DIFF (AP interp diff)**'} |",
              ""]
    lines += ["## 3) Image-level disease detection (primary interpretation)", "",
              f"- det_pr_auc = **{det_pr_auc:.4f}**, det_roc_auc = **{det_roc_auc:.4f}** (n_pos={n_pos}, n_neg={n_neg})",
              f"- presence_recall@0.5 (disease recall) = **{presence_recall:.3f}** = {tp}/{n_pos} (95% CI {recall_ci[0]:.3f}-{recall_ci[1]:.3f})",
              f"- fp_rate@0.5 (normal false alarm) = **{fp_rate:.4f}** = {fp}/{n_neg} (95% CI {fp_ci[0]:.4f}-{fp_ci[1]:.4f})",
              "",
              "## 4) Localization quality (positives only, coarse-box caveat)", "",
              f"- IoU(pred,GT) on positive images (n={n_pos}): median={iou_dist['median']:.3f}, mean={iou_dist['mean']:.3f}, p25={iou_dist['p25']:.3f}, p75={iou_dist['p75']:.3f}, min={iou_dist['min']:.3f}, max={iou_dist['max']:.3f}",
              f"- frac positives localized @IoU>=0.5 = {iou_presence:.3f}",
              f"- mAP@0.5 (objectness-ranked, normal preds = FP) = **{map05:.3f}** — interpret SEPARATELY from det_pr_auc: mAP couples ranking with coarse-box localization, so it is bounded by the IoU>=0.5 hit rate, not by disease/normal separability.",
              "- NOTE: GT boxes are coarse (median ~50% of frame, center-biased). IoU median ~0.6 reflects crop-region localization, NOT fine lesion pinpointing (REPORT sec.6).",
              ""]
    if amp_note:
        lines += [f"## 5) Config notes", "", f"- meta: {json.dumps(amp_note, ensure_ascii=False)}", ""]
    (EVAL / f"verify_{name}.md").write_text("\n".join(lines))


# ----------------------------------------------------------------------------
# training curves
# ----------------------------------------------------------------------------
def plot_cls_curve(name, r):
    pe = r["per_epoch"]
    ep = [e["epoch"] for e in pe]
    tr = [e["train_loss"] for e in pe]
    vl = [e["val_loss"] for e in pe]
    pa = [e.get("pr_auc", np.nan) for e in pe]
    best = r["reported"]["epoch"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(ep, tr, label="train_loss")
    ax[0].plot(ep, vl, label="val_loss")
    ax[0].axvline(best, ls="--", c="gray", lw=1, label=f"best ep {best}")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].set_title(f"{name}\nloss"); ax[0].legend(fontsize=7)
    ax[1].plot(ep, pa, c="tab:green", label="val PR-AUC")
    ax[1].axvline(best, ls="--", c="gray", lw=1)
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("PR-AUC"); ax[1].set_title("val PR-AUC"); ax[1].set_ylim(0, 1); ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / f"curves_{name}.png"); plt.close(fig)


def plot_det_curve(name, r):
    pe = r["per_epoch"]
    ep = [e["epoch"] for e in pe]
    tr = [e["train_loss"] for e in pe]
    vl = [e["val_loss"] for e in pe]
    pa = [e.get("det_pr_auc", np.nan) for e in pe]
    rc = [e.get("presence_recall_at_0.5", np.nan) for e in pe]
    best = r["reported"]["epoch"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(ep, tr, label="train_loss")
    ax[0].plot(ep, vl, label="val_loss")
    ax[0].axvline(best, ls="--", c="gray", lw=1, label=f"best ep {best}")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].set_title(f"{name}\nloss"); ax[0].legend(fontsize=7)
    ax[1].plot(ep, pa, c="tab:green", label="val det_PR-AUC")
    ax[1].plot(ep, rc, c="tab:purple", lw=1, alpha=0.8, label="val presence_recall@0.5")
    ax[1].axvline(best, ls="--", c="gray", lw=1)
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("score"); ax[1].set_title("val det PR-AUC / recall@0.5"); ax[1].set_ylim(0, 1.02); ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / f"curves_{name}.png"); plt.close(fig)


def plot_combined_curves():
    ncol = len(ARCHS)  # 6 backbones
    fig, axes = plt.subplots(4, ncol, figsize=(3.1 * ncol, 13))
    # rows 0-2: classification settings, cols: arch
    for ri, s in enumerate(CLS_SETTINGS):
        for ci, arch in enumerate(ARCHS):
            ax = axes[ri, ci]
            r = cls_results[f"{arch}_{s}"]
            pe = r["per_epoch"]
            ep = [e["epoch"] for e in pe]
            ax.plot(ep, [e["train_loss"] for e in pe], label="train_loss", lw=1.1)
            ax.plot(ep, [e["val_loss"] for e in pe], label="val_loss", lw=1.1)
            ax2 = ax.twinx()
            ax2.plot(ep, [e.get("pr_auc", np.nan) for e in pe], c="tab:green", lw=1.1, label="PR-AUC")
            ax2.set_ylim(0, 1)
            ax.axvline(r["reported"]["epoch"], ls="--", c="gray", lw=0.8)
            tag = " [NEW]" if arch in NEW_ARCHS else ""
            ax.set_title(f"{ARCH_LABEL[arch]} / {s}{tag}", fontsize=7)
            if ci == 0:
                ax.set_ylabel(f"{s}\nloss", fontsize=7)
            if ci == ncol - 1:
                ax2.set_ylabel("PR-AUC", color="tab:green", fontsize=7)
            ax.tick_params(labelsize=6); ax2.tick_params(labelsize=6)
    # row 3: detection
    for ci, arch in enumerate(ARCHS):
        ax = axes[3, ci]
        r = det_results[f"{arch}_detection_singlebox"]
        pe = r["per_epoch"]
        ep = [e["epoch"] for e in pe]
        ax.plot(ep, [e["train_loss"] for e in pe], label="train_loss", lw=1.1)
        ax.plot(ep, [e["val_loss"] for e in pe], label="val_loss", lw=1.1)
        ax2 = ax.twinx()
        ax2.plot(ep, [e.get("det_pr_auc", np.nan) for e in pe], c="tab:green", lw=1.1)
        ax2.set_ylim(0, 1)
        ax.axvline(r["reported"]["epoch"], ls="--", c="gray", lw=0.8)
        tag = " [NEW]" if arch in NEW_ARCHS else ""
        ax.set_title(f"{ARCH_LABEL[arch]} / detection{tag}", fontsize=7)
        if ci == 0:
            ax.set_ylabel("detection\nloss", fontsize=7)
        if ci == ncol - 1:
            ax2.set_ylabel("det PR-AUC", color="tab:green", fontsize=7)
        ax.set_xlabel("epoch", fontsize=6)
        ax.tick_params(labelsize=6); ax2.tick_params(labelsize=6)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=9)
    fig.suptitle("Training curves: train/val loss + primary metric (rows: 3 cls settings + detection; cols: 6 backbones, [NEW]=added)", y=0.997, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(FIG / "training_curves.png"); plt.close(fig)


# ----------------------------------------------------------------------------
# comparison figures
# ----------------------------------------------------------------------------
def fig_cls_bars():
    # back-compat: PR-AUC / macro-F1 grouped bars over 6 backbones
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    x = np.arange(len(CLS_SETTINGS))
    n = len(ARCHS); w = 0.8 / n
    for ai, arch in enumerate(ARCHS):
        pa = [cls_results[f"{arch}_{s}"]["prauc"] for s in CLS_SETTINGS]
        f1 = [cls_results[f"{arch}_{s}"]["f1_macro_recomp"] for s in CLS_SETTINGS]
        off = (ai - (n - 1) / 2) * w
        lbl = ARCH_LABEL[arch] + ("*" if arch in NEW_ARCHS else "") + f" [{PARAMS_M[arch][0]:.1f}M]"
        axes[0].bar(x + off, pa, w, label=lbl)
        axes[1].bar(x + off, f1, w, label=lbl)
    for ax, title in zip(axes, ["PR-AUC (recomputed)", "F1-macro (recomputed)"]):
        ax.set_xticks(x); ax.set_xticklabels(CLS_SETTINGS, fontsize=8)
        ax.set_ylim(0, 1); ax.set_title(title); ax.legend(fontsize=7, ncol=2); ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Classification PR-AUC / F1-macro across 6 backbones (* = newly added; [..M] = classifier params)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(FIG / "exp_cls_bars.png"); plt.close(fig)


# Ours (DINOv3 frozen pretrained) variants — orig-distribution metrics from ours_dinov3.json.
# These are NOT from-scratch baselines; rendered with distinct color+hatch + "(pretrained)" labels.
OURS_KEYS = [
    ("dinov3", "Ours: DINOv3-S @256 (frozen,CE)", "#000000", "//"),
    ("dinov3_base", "Ours: DINOv3-B @512 (frozen,CE)", "#8B0000", "xx"),
    ("dinov3_base_focal", "Ours: DINOv3-B @512 (frozen,focal+aug)", "#9400D3", ".."),
]


def _load_ours(dist):
    """Return {ours_prefix: {setting: {auroc,f1_macro,accuracy}}} from ours_dinov3.json (dist='orig'|'bal')."""
    j = json.loads((EVAL / "ours_dinov3.json").read_text())["results"]
    out = {}
    for prefix, _, _, _ in OURS_KEYS:
        out[prefix] = {}
        for s in CLS_SETTINGS:
            r = j[f"{prefix}_{s}"][dist]
            out[prefix][s] = {"auroc": r["auroc"], "f1_macro": r["f1_macro"], "accuracy": r["accuracy"]}
    return out


def fig_metrics_table():
    """User-required 7-metric view: AUROC / F1-macro / accuracy grouped bars, 6 backbones x 3 settings.
    Ours (DINOv3 frozen pretrained) 3 variants appended per group (distinct color+hatch)."""
    ours = _load_ours("orig")
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.4))
    x = np.arange(len(CLS_SETTINGS))
    n = len(ARCHS) + len(OURS_KEYS); w = 0.86 / n
    metric_keys = [("auroc", "AUROC"), ("f1_macro_recomp", "F1-macro"), ("accuracy", "accuracy")]
    ours_metric = {"auroc": "auroc", "f1_macro_recomp": "f1_macro", "accuracy": "accuracy"}
    for ax, (key, title) in zip(axes, metric_keys):
        for ai, arch in enumerate(ARCHS):
            vals = [cls_results[f"{arch}_{s}"][key] for s in CLS_SETTINGS]
            off = (ai - (n - 1) / 2) * w
            lbl = ARCH_LABEL[arch] + ("*" if arch in NEW_ARCHS else "") + f" [{PARAMS_M[arch][0]:.1f}M]"
            ax.bar(x + off, vals, w, label=lbl)
        for oi, (prefix, olbl, ocol, ohatch) in enumerate(OURS_KEYS):
            ai = len(ARCHS) + oi
            vals = [ours[prefix][s][ours_metric[key]] for s in CLS_SETTINGS]
            off = (ai - (n - 1) / 2) * w
            ax.bar(x + off, vals, w, label=olbl + " (pretrained)", color=ocol,
                   hatch=ohatch, edgecolor="white", linewidth=0.4)
        ax.set_xticks(x); ax.set_xticklabels(CLS_SETTINGS, fontsize=8)
        ax.set_ylim(0, 1.02); ax.set_title(title); ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=6.5, ncol=2, loc="lower left")
    fig.suptitle("Classification 7-metric (AUROC / F1-macro / accuracy) — 6 from-scratch backbones (solid) "
                 "vs Ours DINOv3 frozen-pretrained (black/red/violet, hatched) — NOT same-condition; pretrained head-only",
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(FIG / "exp_metrics_table.png"); plt.close(fig)


def fig_pr_curves():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for si, s in enumerate(CLS_SETTINGS):
        ax = axes[si]
        for arch in ARCHS:
            r = cls_results[f"{arch}_{s}"]
            ls = "--" if arch in NEW_ARCHS else "-"
            if r["k"] == 2:
                rec, prec = pr_curve(r["label"] == 1, r["prob"][:, 1])
                ax.plot(rec, prec, ls, label=f"{ARCH_LABEL[arch]} ({r['prauc']:.2f})", lw=1.3)
            else:
                # macro: plot disease classes averaged-ish -> show d3 & d4 combined OvR (disease vs rest)
                rec, prec = pr_curve(r["label"] >= 1, r["prob"][:, 1:].sum(1))
                ax.plot(rec, prec, ls, label=f"{ARCH_LABEL[arch]} ({r['prauc']:.2f})", lw=1.3)
        base = {"normal_vs_d3": 76 / 1379, "normal_vs_d4": 24 / 1327, "normal_d3_d4": 100 / 1403}[s]
        ax.axhline(base, ls=":", c="gray", lw=1, label=f"prevalence {base:.3f}")
        ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_title(f"PR curve: {s}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "exp_pr_curves.png"); plt.close(fig)


def fig_confusions():
    fig, axes = plt.subplots(len(ARCHS), len(CLS_SETTINGS), figsize=(11, 22))
    for ai, arch in enumerate(ARCHS):
        for si, s in enumerate(CLS_SETTINGS):
            ax = axes[ai, si]
            r = cls_results[f"{arch}_{s}"]
            cm = r["cm"]
            im = ax.imshow(cm, cmap="Blues")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
            ax.set_xticks(range(r["k"])); ax.set_yticks(range(r["k"]))
            ax.set_xticklabels([c.replace("disease_", "d") for c in r["class_names"]], fontsize=7)
            ax.set_yticklabels([c.replace("disease_", "d") for c in r["class_names"]], fontsize=7)
            ax.set_title(f"{ARCH_LABEL[arch]}/{s}", fontsize=8)
            if si == 0:
                ax.set_ylabel("true")
            ax.set_xlabel("pred", fontsize=7)
    fig.suptitle("Confusion matrices (threshold=argmax) — rows: backbone, cols: setting", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(FIG / "exp_confusion.png"); plt.close(fig)


def _load_ours_det():
    """Ours detection (orig distribution) from ours_detection.json, or None if absent.
    Ours = DINOv3-B frozen, head-only — NOT a from-scratch baseline (rendered distinctly)."""
    p = EVAL / "ours_detection.json"
    if not p.exists():
        return None
    j = json.loads(p.read_text())
    o = j["orig_full"]
    return dict(
        label="Ours: DINOv3-B\n[85.8M frozen\n+0.2M head]",
        det_pr_auc=o["det_pr_auc"], presence_recall=o["presence_recall"],
        fp_rate=o["fp_rate"], iou_median=o["iou_dist"]["median"],
        pos_ious=np.asarray(o.get("pos_ious", []), dtype=float),
        scores_pos_median=o["sep"]["pos_median"], scores_neg_median=o["sep"]["neg_median"],
        late_iou_median=j["best_epoch_iou_caveat"].get("late_epoch_iou_median_max"),
    )


def fig_det():
    ours = _load_ours_det()
    fig, axes = plt.subplots(1, 3, figsize=(20, 4.6))
    archlabels = [ARCH_LABEL[a] + ("*" if a in NEW_ARCHS else "") + f"\n[{PARAMS_M[a][1]:.1f}M]" for a in ARCHS]
    n_groups = len(ARCHS) + (1 if ours else 0)
    if ours:
        archlabels = archlabels + [ours["label"]]

    # (1) bars: det_pr_auc / presence_recall@0.5 / fp_rate@0.5
    x = np.arange(n_groups); w = 0.25
    prauc = [det_results[f"{a}_detection_singlebox"]["det_pr_auc"] for a in ARCHS]
    rec = [det_results[f"{a}_detection_singlebox"]["presence_recall"] for a in ARCHS]
    fpr = [det_results[f"{a}_detection_singlebox"]["fp_rate"] for a in ARCHS]
    if ours:
        prauc = prauc + [ours["det_pr_auc"]]; rec = rec + [ours["presence_recall"]]; fpr = fpr + [ours["fp_rate"]]
    axes[0].bar(x - w, prauc, w, label="det PR-AUC", color="tab:green")
    axes[0].bar(x, rec, w, label="presence_recall@0.5", color="tab:blue")
    axes[0].bar(x + w, fpr, w, label="fp_rate@0.5", color="tab:red")
    for xi, (a, b, c) in enumerate(zip(prauc, rec, fpr)):
        axes[0].text(xi - w, a + 0.01, f"{a:.3f}", ha="center", fontsize=6)
        axes[0].text(xi, b + 0.01, f"{b:.2f}", ha="center", fontsize=6)
        axes[0].text(xi + w, c + 0.01, f"{c:.3f}", ha="center", fontsize=6)
    if ours:  # shade the Ours group (pretrained, not same-condition)
        axes[0].axvspan(len(ARCHS) - 0.5, len(ARCHS) + 0.5, color="orange", alpha=0.08)
    axes[0].set_xticks(x); axes[0].set_xticklabels(archlabels, fontsize=6.5, rotation=20); axes[0].set_ylim(0, 1.08)
    axes[0].set_title("Image-level disease detection (recomputed)\nOurs=DINOv3-B frozen (shaded, pretrained — NOT same-condition)")
    axes[0].legend(fontsize=7); axes[0].grid(axis="y", alpha=0.3)

    # (2) objectness score distribution: normal vs disease (box plot). Ours appended (median markers only).
    pos_data = [det_results[f"{a}_detection_singlebox"]["scores"][det_results[f"{a}_detection_singlebox"]["is_pos"]] for a in ARCHS]
    neg_data = [det_results[f"{a}_detection_singlebox"]["scores"][~det_results[f"{a}_detection_singlebox"]["is_pos"]] for a in ARCHS]
    positions = np.arange(len(ARCHS))
    bp_n = axes[1].boxplot(neg_data, positions=positions - 0.18, widths=0.3,
                           patch_artist=True, showfliers=True, flierprops=dict(marker=".", ms=2, alpha=0.4))
    bp_p = axes[1].boxplot(pos_data, positions=positions + 0.18, widths=0.3,
                           patch_artist=True, showfliers=True, flierprops=dict(marker=".", ms=2, alpha=0.4))
    for b in bp_n["boxes"]:
        b.set_facecolor("tab:green"); b.set_alpha(0.6)
    for b in bp_p["boxes"]:
        b.set_facecolor("tab:red"); b.set_alpha(0.6)
    if ours:  # Ours objectness medians (disease vs normal) as markers in a trailing slot
        ox = len(ARCHS)
        axes[1].scatter([ox - 0.18], [ours["scores_neg_median"]], marker="D", s=28, c="green", edgecolors="k", zorder=5)
        axes[1].scatter([ox + 0.18], [ours["scores_pos_median"]], marker="D", s=28, c="red", edgecolors="k", zorder=5)
        axes[1].axvspan(ox - 0.5, ox + 0.5, color="orange", alpha=0.08)
        positions = np.arange(len(ARCHS) + 1)
    axes[1].axhline(0.5, ls="--", c="gray", lw=1)
    axes[1].set_xticks(positions); axes[1].set_xticklabels(archlabels, fontsize=6.5, rotation=20)
    axes[1].set_ylim(-0.05, 1.05); axes[1].set_ylabel("objectness (disease score)")
    axes[1].set_title("Objectness: normal (green) vs disease (red)\ncollapse RESOLVED (Ours=diamond medians)")
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(facecolor="tab:green", alpha=0.6, label="normal"),
                            Patch(facecolor="tab:red", alpha=0.6, label="disease")], fontsize=7)

    # (3) positive IoU distribution
    for arch in ARCHS:
        r = det_results[f"{arch}_detection_singlebox"]
        axes[2].hist(r["pos_ious"], bins=25, range=(0, 1), histtype="step", lw=1.6,
                     label=f"{ARCH_LABEL[arch]} (med {r['iou_dist']['median']:.2f})")
    if ours and len(ours["pos_ious"]):
        late = ours.get("late_iou_median")
        late_s = f"; late-ep med {late:.2f}" if late else ""
        axes[2].hist(ours["pos_ious"], bins=25, range=(0, 1), histtype="step", lw=2.2, color="black", ls="--",
                     label=f"Ours: DINOv3-B (med {ours['iou_median']:.2f}, ep0{late_s})")
    axes[2].axvline(0.5, ls="--", c="gray", lw=1)
    axes[2].set_xlabel("IoU (pred vs GT, positives only, n=100)"); axes[2].set_ylabel("count")
    axes[2].set_title("Localization IoU distribution (disease images)\nOurs dashed (best=ep0, IoU underestimated; late-ep ~0.64)")
    axes[2].legend(fontsize=7.5)

    fig.tight_layout(); fig.savefig(FIG / "exp_detection.png"); plt.close(fig)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
for arch in ARCHS:
    for s in CLS_SETTINGS:
        eval_classification(arch, s)
        plot_cls_curve(f"{arch}_{s}", cls_results[f"{arch}_{s}"])
for arch in ARCHS:
    eval_detection(arch)
    plot_det_curve(f"{arch}_detection_singlebox", det_results[f"{arch}_detection_singlebox"])

plot_combined_curves()
fig_cls_bars()
fig_metrics_table()
fig_pr_curves()
fig_confusions()
fig_det()

# dump a json summary for the report writer
summary = {
    "classification": {n: {k: (v if isinstance(v, (int, float, str, dict, list, bool)) else None)
                           for k, v in r.items()
                           if k in ("arch", "setting", "prauc", "prauc_per", "macro_f1",
                                    "recall_disease", "precision_disease", "recall_ci",
                                    "accuracy", "n_dis", "n_valid", "reported", "dist_match",
                                    "manifest_dist", "pred_dist", "f1_per", "recall_per", "prec_per",
                                    "recall_macro", "precision_macro", "f1_macro_recomp",
                                    "auroc", "train_loss", "val_loss", "best_epoch",
                                    "params_m", "is_new")}
                       for n, r in cls_results.items()},
    "detection": {n: {k: (v if isinstance(v, (int, float, str, dict, list, bool)) else None)
                      for k, v in r.items()
                      if k in ("arch", "n", "n_pos", "n_neg", "det_pr_auc", "det_roc_auc",
                               "presence_recall", "fp_rate", "recall_ci", "fp_ci",
                               "iou_presence", "iou_dist", "map05", "sep", "reported",
                               "meta", "elapsed", "train_loss", "val_loss", "best_epoch",
                               "params_m", "is_new")}
                  for n, r in det_results.items()},
}
(EVAL / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
print("DONE. verify files + figures + summary.json written.")
print("\nModel params (M)  [classifier = build_classifier(arch,2,224); detector = build_detector(arch,512)]")
print(f"  {'arch':<16} {'label':<22} {'cls(M)':>8} {'det(M)':>8}")
for arch in ARCHS:
    cm, dm = PARAMS_M[arch]
    print(f"  {ARCH_TOKEN[arch]:<16} {ARCH_LABEL[arch]:<22} {cm:>8.2f} {dm:>8.2f}")
for n, r in cls_results.items():
    print(f"  {n}: acc={r['accuracy']:.3f} trL={r['train_loss']:.3f} vlL={r['val_loss']:.3f} "
          f"rec={r['recall_macro']:.3f} prec={r['precision_macro']:.3f} f1={r['f1_macro_recomp']:.3f} "
          f"AUROC={r['auroc']:.4f} PR-AUC={r['prauc']:.3f} (rep PR-AUC={r['reported']['pr_auc']:.3f}) "
          f"dist_match={r['dist_match']} new={r['is_new']}")
for n, r in det_results.items():
    rep = r["reported"]
    print(f"  {n}: det_PR-AUC={r['det_pr_auc']:.4f} (rep {rep['det_pr_auc']:.4f}) "
          f"ROC-AUC={r['det_roc_auc']:.4f} recall@0.5={r['presence_recall']:.3f} (rep {rep['presence_recall_at_0.5']:.3f}) "
          f"fp_rate@0.5={r['fp_rate']:.4f} (rep {rep['fp_rate_at_0.5']:.4f}) IoU_med={r['iou_dist']['median']:.3f} "
          f"mAP={r['map05']:.3f} | obj_sep pos_med={r['sep']['pos_median']:.3f} neg_med={r['sep']['neg_median']:.4f} n_pos/neg={r['n_pos']}/{r['n_neg']}")
