"""Metric helpers for radish baseline experiments.

These are used by `src/train.py` to record a FIRST-PASS set of metrics in
`experiments/<name>/metrics.json`. eval-reporter recomputes everything
independently from the dumped predictions, so anything here is for logging /
early-stop only and must not be relied on as the source of truth.

Classification metrics operate on probabilities (softmax) + integer labels.
Detection metrics operate on predicted single boxes (xyxy in [0,1]) + scores
and a list of GT boxes per image (xyxy in [0,1], possibly empty for normal).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def classification_metrics(probs: np.ndarray, labels: np.ndarray, class_names: list[str]) -> dict:
    """Compute first-pass classification metrics.

    Args:
        probs: [N, C] softmax probabilities.
        labels: [N] integer ground-truth labels.
        class_names: list of C class names (index 0 == normal).

    Returns dict with pr_auc (primary), macro_f1, recall_disease,
    precision_disease, accuracy, confusion, plus macro recall/precision/f1
    (recall_macro/precision_macro/f1_macro) and auroc (ROC-AUC).
    """
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    num_classes = len(class_names)
    preds = probs.argmax(axis=1)

    # accuracy (reported but NOT primary -- valid is imbalanced)
    accuracy = float((preds == labels).mean()) if len(labels) else 0.0

    # macro F1 over argmax predictions
    macro_f1 = float(
        f1_score(labels, preds, labels=list(range(num_classes)),
                 average="macro", zero_division=0)
    )
    # macro recall / precision / f1 over all classes (class-balanced view)
    recall_macro = float(
        recall_score(labels, preds, labels=list(range(num_classes)),
                     average="macro", zero_division=0)
    )
    precision_macro = float(
        precision_score(labels, preds, labels=list(range(num_classes)),
                        average="macro", zero_division=0)
    )
    f1_macro = macro_f1

    # disease class indices: everything except normal(0)
    disease_idx = list(range(1, num_classes))
    # binary disease membership (any disease vs normal)
    y_true_disease = np.isin(labels, disease_idx).astype(np.int64)
    y_pred_disease = np.isin(preds, disease_idx).astype(np.int64)
    recall_disease = float(
        recall_score(y_true_disease, y_pred_disease, zero_division=0)
    )
    precision_disease = float(
        precision_score(y_true_disease, y_pred_disease, zero_division=0)
    )

    # PR-AUC
    if num_classes == 2:
        # disease score = P(class 1)
        disease_score = probs[:, 1]
        if y_true_disease.sum() == 0 or y_true_disease.sum() == len(y_true_disease):
            pr_auc = float("nan")
        else:
            pr_auc = float(average_precision_score(y_true_disease, disease_score))
    else:
        # macro one-vs-rest PR-AUC over disease classes
        aps = []
        for c in disease_idx:
            y_c = (labels == c).astype(np.int64)
            if y_c.sum() == 0 or y_c.sum() == len(y_c):
                continue
            aps.append(float(average_precision_score(y_c, probs[:, c])))
        pr_auc = float(np.mean(aps)) if aps else float("nan")

    # ROC-AUC (AUROC)
    #   2-class: P(disease == positive class) vs binary disease label.
    #   3-class+: macro one-vs-rest ROC-AUC over ALL classes.
    # Degenerate cases (single class present, all-one-label) -> nan.
    try:
        if num_classes == 2:
            disease_score = probs[:, 1]
            if y_true_disease.sum() == 0 or y_true_disease.sum() == len(y_true_disease):
                auroc = float("nan")
            else:
                auroc = float(roc_auc_score(y_true_disease, disease_score))
        else:
            present = np.unique(labels)
            if len(present) < 2:
                auroc = float("nan")
            else:
                # macro OvR over the classes actually present in labels.
                aucs = []
                for c in range(num_classes):
                    y_c = (labels == c).astype(np.int64)
                    if y_c.sum() == 0 or y_c.sum() == len(y_c):
                        continue
                    aucs.append(float(roc_auc_score(y_c, probs[:, c])))
                auroc = float(np.mean(aucs)) if aucs else float("nan")
    except Exception:  # noqa: BLE001 -- any sklearn edge case -> nan, never crash logging
        auroc = float("nan")

    cm = confusion_matrix(labels, preds, labels=list(range(num_classes))).tolist()

    return {
        "pr_auc": pr_auc,
        "macro_f1": macro_f1,
        "recall_disease": recall_disease,
        "precision_disease": precision_disease,
        "accuracy": accuracy,
        "confusion": cm,
        # macro (class-averaged) recall/precision/f1 + AUROC
        "recall_macro": recall_macro,
        "precision_macro": precision_macro,
        "f1_macro": f1_macro,
        "auroc": auroc,
    }


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two single boxes (xyxy)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    iw = max(0.0, inter_x1 - inter_x0)
    ih = max(0.0, inter_y1 - inter_y0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def detection_metrics(
    pred_boxes: np.ndarray,
    scores: np.ndarray,
    gt_boxes_list: list[np.ndarray],
) -> dict:
    """Compute image-level disease-detection + localization + mAP metrics.

    This dataset supplies NORMAL images as negatives (empty GT box) and disease
    images as positives (one GT box). Metrics are therefore reframed around
    image-level disease detection (objectness score as the disease score),
    keeping localization quality separate so the dominant negatives do not mask a
    box-regression problem.

    Args:
        pred_boxes: [N, 4] predicted box per image (xyxy in [0,1]).
        scores: [N] objectness score per image (used as the disease score).
        gt_boxes_list: list length N; each is [Ki, 4] GT boxes (0 or 1 box).
                       Empty array == normal (negative) image.

    Returns dict with:
      Image-level disease detection (PRIMARY = det_pr_auc):
        det_pr_auc, det_roc_auc            (objectness vs disease/normal label)
        presence_recall_at_0.5             (frac of disease imgs w/ score>=0.5)
        fp_rate_at_0.5                     (frac of normal  imgs w/ score>=0.5)
      Localization (POSITIVES ONLY):
        iou_distribution, iou_at_0.5_presence
      Standard detection:
        map_at_0.5
      Counts:
        n_positive, n_negative, n_total
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    pred_boxes = np.asarray(pred_boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)

    # is_positive: image has a (disease) GT box.
    is_pos = np.array(
        [np.asarray(gt, dtype=np.float64).reshape(-1, 4).shape[0] > 0
         for gt in gt_boxes_list],
        dtype=bool,
    )
    n_total = int(len(gt_boxes_list))
    n_positive = int(is_pos.sum())
    n_negative = int((~is_pos).sum())

    # ---- image-level disease detection (objectness vs label) ----
    y_true = is_pos.astype(np.int64)
    if 0 < n_positive < n_total:
        det_pr_auc = float(average_precision_score(y_true, scores))
        det_roc_auc = float(roc_auc_score(y_true, scores))
    else:
        # degenerate (all-positive or all-negative valid set)
        det_pr_auc = float("nan")
        det_roc_auc = float("nan")

    presence_recall_at_0_5 = (
        float((scores[is_pos] >= 0.5).mean()) if n_positive > 0 else float("nan")
    )
    fp_rate_at_0_5 = (
        float((scores[~is_pos] >= 0.5).mean()) if n_negative > 0 else float("nan")
    )

    # ---- localization: IoU on POSITIVE (disease) images only ----
    ious = []
    for i, gt in enumerate(gt_boxes_list):
        gt = np.asarray(gt, dtype=np.float64).reshape(-1, 4)
        if gt.shape[0] == 0:
            continue
        ious.append(box_iou_xyxy(pred_boxes[i], gt[0]))
    ious_arr = np.asarray(ious, dtype=np.float64)

    if n_positive > 0:
        iou_distribution = {
            "mean": float(ious_arr.mean()),
            "median": float(np.median(ious_arr)),
            "p25": float(np.percentile(ious_arr, 25)),
            "p75": float(np.percentile(ious_arr, 75)),
            "min": float(ious_arr.min()),
            "max": float(ious_arr.max()),
        }
        iou_at_0_5_presence = float((ious_arr >= 0.5).mean())
    else:
        iou_distribution = {
            "mean": float("nan"), "median": float("nan"),
            "p25": float("nan"), "p75": float("nan"),
            "min": float("nan"), "max": float("nan"),
        }
        iou_at_0_5_presence = float("nan")

    # ---- standard mAP@0.5 (objectness-ranked, normals' preds are all FP) ----
    map_at_0_5 = _map_at_iou(pred_boxes, scores, gt_boxes_list, iou_thr=0.5)

    return {
        "det_pr_auc": det_pr_auc,
        "det_roc_auc": det_roc_auc,
        "presence_recall_at_0.5": presence_recall_at_0_5,
        "fp_rate_at_0.5": fp_rate_at_0_5,
        "iou_distribution": iou_distribution,
        "iou_at_0.5_presence": iou_at_0_5_presence,
        "map_at_0.5": map_at_0_5,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_total": n_total,
    }


def _map_at_iou(
    pred_boxes: np.ndarray,
    scores: np.ndarray,
    gt_boxes_list: list[np.ndarray],
    iou_thr: float = 0.5,
) -> float:
    """Single-class AP@iou_thr (VOC-style, one predicted box per image).

    Each image contributes exactly one detection (the single regressed box,
    ranked by objectness score). A detection is TP iff the image has a GT box
    and IoU >= thr; otherwise FP. AP via precision-recall integration.
    """
    n_gt = sum(1 for gt in gt_boxes_list
               if np.asarray(gt).reshape(-1, 4).shape[0] > 0)
    if n_gt == 0:
        return float("nan")

    order = np.argsort(-scores)
    tp = np.zeros(len(order), dtype=np.float64)
    fp = np.zeros(len(order), dtype=np.float64)

    for rank, i in enumerate(order):
        gt = np.asarray(gt_boxes_list[i], dtype=np.float64).reshape(-1, 4)
        if gt.shape[0] == 0:
            fp[rank] = 1.0
            continue
        iou = box_iou_xyxy(pred_boxes[i], gt[0])
        if iou >= iou_thr:
            tp[rank] = 1.0
        else:
            fp[rank] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / max(1, n_gt)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)

    # VOC-style: prepend/append sentinels, monotone envelope, integrate.
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for k in range(len(mpre) - 2, -1, -1):
        mpre[k] = max(mpre[k], mpre[k + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap
