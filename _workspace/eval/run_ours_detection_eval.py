"""Ours (DINOv3-B @512 frozen, head-only) detection eval (eval-reporter QA).

Adds the Ours detection row to §3 (orig distribution) and §3B (balanced valid)
of report/EXPERIMENTS.md. Reuses the exact metric primitives of run_eval.py
(§3) and the balanced loader procedure of run_balanced_detection_eval.py (§3B).

§3 (orig, prevalence ~0.071): recompute from saved predictions/valid.json
    -> det_pr_auc / det_roc_auc / presence_recall@0.5 / fp_rate@0.5 / mAP@0.5
       + positive-only IoU median/mean + objectness separation (pos vs neg).

§3B (balanced 1:1, N=200): reload best.pt (weights NOT modified), rebuild
    balance_valid=True loader (seed=42, img512), forward fp32, recompute the
    same set on the balanced valid set.

CAVEAT (best-epoch): det_pr_auc saturates at 1.0 from ep0, so the run's best
    checkpoint = ep0, whose saved predictions have the LOWEST localization IoU
    (median 0.5649). Later epochs reach IoU median ~0.64 (train.log). Detection
    metrics (det_pr_auc 1.0 / presence 1.0 / fp ~0) are saturated from ep0 so are
    unaffected. Both facts are emitted to ours_detection.json and the verify file.

Dumps _workspace/eval/ours_detection.json. sklearn cross-check on both sets.

Run: ./.venv/bin/python _workspace/eval/run_ours_detection_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import build_detection_loaders  # noqa: E402
from src.models import build_detector  # noqa: E402

from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    roc_auc_score,
)

EXP = ROOT / "experiments"
EVAL = ROOT / "_workspace" / "eval"
NAME = "dinov3_base_detection_singlebox"
ARCH = "dinov3_base"
IMG = 512
SEED = 42


# ---- metric primitives copied verbatim from run_eval.py (same conventions) ----
def pr_auc(y_true, scores):
    """PR-AUC(average precision)를 계산."""
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
    rprev = 0.0
    ap = 0.0
    for r, p in zip(recall, precision):
        ap += (r - rprev) * p
        rprev = r
    return float(ap)


def roc_auc(y_true, scores):
    """ROC-AUC를 계산."""
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
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    sum_ranks_pos = ranks[y_true == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def wilson_ci(k, n, z=1.96):
    """이항 비율의 Wilson 95% 신뢰구간(저·고) — 소표본 지표에 CI 병기용."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def iou_xyxy(a, b):
    """두 xyxy 박스의 IoU(교집합/합집합)."""
    ix0 = max(a[0], b[0]); iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2]); iy1 = min(a[3], b[3])
    iw = max(0.0, ix1 - ix0); ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


def detection_block(pred_boxes, scores, gt_boxes, is_pos):
    """Replicate run_eval.eval_detection metric block. Returns dict."""
    scores = np.asarray(scores, dtype=float)
    is_pos = np.asarray(is_pos, dtype=bool)
    n = len(pred_boxes)
    n_pos = int(is_pos.sum())
    n_neg = int((~is_pos).sum())

    det_pr = pr_auc(is_pos, scores)
    det_roc = roc_auc(is_pos, scores)
    thr = 0.5
    pred_disease = scores >= thr
    tp = int((pred_disease & is_pos).sum())
    fp = int((pred_disease & ~is_pos).sum())
    presence_recall = tp / n_pos if n_pos else float("nan")
    fp_rate = fp / n_neg if n_neg else float("nan")

    sp = scores[is_pos]
    sn = scores[~is_pos]
    sep = dict(
        pos_median=float(np.median(sp)), pos_p25=float(np.percentile(sp, 25)),
        pos_min=float(sp.min()),
        neg_median=float(np.median(sn)) if n_neg else float("nan"),
        neg_p75=float(np.percentile(sn, 75)) if n_neg else float("nan"),
        neg_p95=float(np.percentile(sn, 95)) if n_neg else float("nan"),
        neg_max=float(sn.max()) if n_neg else float("nan"),
        n_neg_ge_0p5=int((sn >= 0.5).sum()) if n_neg else 0,
    )

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
    iou_presence = float((pos_ious >= 0.5).mean())

    order = np.argsort(-scores)
    tp_arr = np.zeros(n); fp_arr = np.zeros(n)
    for rank, idx in enumerate(order):
        gb = gt_boxes[idx]
        if is_pos[idx] and isinstance(gb, list) and len(gb) > 0 and gb[0] is not None \
                and iou_xyxy(pred_boxes[idx], gb[0]) >= 0.5:
            tp_arr[rank] = 1
        else:
            fp_arr[rank] = 1
    ctp = np.cumsum(tp_arr); cfp = np.cumsum(fp_arr)
    recall = ctp / max(n_pos, 1)
    precision = ctp / np.maximum(ctp + cfp, 1)
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = precision[recall >= t].max() if np.any(recall >= t) else 0.0
        ap += p / 101
    map05 = float(ap)

    return dict(
        n=n, n_pos=n_pos, n_neg=n_neg,
        det_pr_auc=det_pr, det_roc_auc=det_roc,
        presence_recall=presence_recall, fp_rate=fp_rate,
        tp=tp, fp=fp,
        recall_ci=wilson_ci(tp, n_pos), fp_ci=wilson_ci(fp, n_neg),
        iou_presence=iou_presence, iou_dist=iou_dist, map05=map05, sep=sep,
        pos_ious=pos_ious,
    )


@torch.no_grad()
def forward_balanced(model, loader, device, img_size):
    """균형 valid 로더로 모델을 forward해 예측을 수집."""
    model.eval()
    pred_list, score_list, gt_list, pos_list = [], [], [], []
    sz = float(img_size)
    for images, targets in loader:
        imgs = torch.stack([im for im in images]).to(device, non_blocking=True)
        pred_boxes_t, obj_logit = model(imgs)
        scores = torch.sigmoid(obj_logit.float()).cpu().numpy()
        pred_boxes = pred_boxes_t.float().cpu().numpy()
        for j, t in enumerate(targets):
            pred_list.append([float(x) for x in pred_boxes[j]])
            score_list.append(float(scores[j]))
            b = t["boxes"]
            if b.numel():
                gt_list.append([[float(x) for x in (b[0] / sz).tolist()]])
            else:
                gt_list.append([])
            pos_list.append(bool(b.numel() > 0))
    return pred_list, score_list, gt_list, pos_list


def main() -> int:
    """스크립트 진입점: 예측/체크포인트 로드 → 지표 재계산 → JSON·그림·리포트 산출."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_dir = EXP / NAME
    mj = json.loads((run_dir / "metrics.json").read_text())
    best_ep = int(mj["final"]["epoch"])
    per_epoch = mj["per_epoch"]

    # params: build once (total + trainable=head-only since backbone frozen)
    model = build_detector(arch=ARCH, img_size=IMG, with_objectness=True).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ---------- §3 ORIG distribution: recompute from saved predictions ----------
    pj = json.loads((run_dir / "predictions" / "valid.json").read_text())
    orig = detection_block(pj["pred_boxes"], pj["scores"], pj["gt_boxes"],
                           pj["is_positive"])
    is_pos_o = np.asarray(pj["is_positive"], dtype=bool)
    sc_o = np.asarray(pj["scores"], dtype=float)
    sk_pr_o = float(average_precision_score(is_pos_o.astype(int), sc_o))
    sk_roc_o = float(roc_auc_score(is_pos_o.astype(int), sc_o))

    # ---------- §3B BALANCED valid: reload best.pt, forward fp32 ----------
    ckpt = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    _, valid_loader, meta = build_detection_loaders(
        img_size=IMG, batch_size=16, num_workers=8, seed=SEED,
        include_normal=True, balance_valid=True,
    )
    pb, sc, gt, ip = forward_balanced(model, valid_loader, device, IMG)
    bal = detection_block(pb, sc, gt, ip)
    ip_arr = np.asarray(ip, dtype=bool)
    sc_arr = np.asarray(sc, dtype=float)
    sk_pr_b = float(average_precision_score(ip_arr.astype(int), sc_arr))
    sk_roc_b = float(roc_auc_score(ip_arr.astype(int), sc_arr))

    # best-epoch IoU caveat evidence (from per_epoch in metrics.json)
    iou_by_ep = [(e["epoch"], e.get("iou_distribution", {}).get("median"),
                  e.get("iou_distribution", {}).get("mean")) for e in per_epoch]
    iou_meds = [m for _, m, _ in iou_by_ep if m is not None]
    late = [m for ep, m, _ in iou_by_ep if ep >= 28 and m is not None]

    def strip(d):  # drop ndarray for json (but keep pos_ious as a list for the figure)
        d = dict(d)
        pi = d.pop("pos_ious", None)
        if pi is not None:
            d["pos_ious"] = [float(x) for x in pi]
        return d

    out = {
        "name": NAME, "arch": ARCH, "img_size": IMG, "seed": SEED,
        "frozen_backbone": True, "with_objectness": True,
        "params": {
            "total_M": round(total_p / 1e6, 2),
            "trainable_K": round(train_p / 1e3, 1),
            "trainable_M": round(train_p / 1e6, 4),
        },
        "best_epoch": best_ep,
        "orig_full": strip(orig),
        "balanced": {**strip(bal),
                     "balanced_valid_counts": meta["valid_counts"]},
        "crosscheck": {
            "orig": {"recomputed_det_pr_auc": orig["det_pr_auc"],
                     "sklearn_det_pr_auc": sk_pr_o,
                     "recomputed_det_roc_auc": orig["det_roc_auc"],
                     "sklearn_det_roc_auc": sk_roc_o,
                     "pr_match": abs(orig["det_pr_auc"] - sk_pr_o) < 1e-9,
                     "roc_match": abs(orig["det_roc_auc"] - sk_roc_o) < 1e-9},
            "balanced": {"recomputed_det_pr_auc": bal["det_pr_auc"],
                         "sklearn_det_pr_auc": sk_pr_b,
                         "recomputed_det_roc_auc": bal["det_roc_auc"],
                         "sklearn_det_roc_auc": sk_roc_b,
                         "pr_match": abs(bal["det_pr_auc"] - sk_pr_b) < 1e-9,
                         "roc_match": abs(bal["det_roc_auc"] - sk_roc_b) < 1e-9},
        },
        "reported_final": {
            "det_pr_auc": mj["final"]["det_pr_auc"],
            "det_roc_auc": mj["final"]["det_roc_auc"],
            "presence_recall_at_0.5": mj["final"]["presence_recall_at_0.5"],
            "fp_rate_at_0.5": mj["final"]["fp_rate_at_0.5"],
            "map_at_0.5": mj["final"]["map_at_0.5"],
            "iou_median": mj["final"]["iou_distribution"]["median"],
            "iou_mean": mj["final"]["iou_distribution"]["mean"],
            "iou_at_0.5_presence": mj["final"]["iou_at_0.5_presence"],
        },
        "best_epoch_iou_caveat": {
            "best_epoch": best_ep,
            "best_epoch_iou_median": orig["iou_dist"]["median"],
            "best_epoch_iou_mean": orig["iou_dist"]["mean"],
            "late_epoch_iou_median_max": (max(late) if late else None),
            "late_epoch_iou_median_mean(ep>=28)": (float(np.mean(late)) if late else None),
            "all_epoch_iou_median_min": min(iou_meds),
            "all_epoch_iou_median_max": max(iou_meds),
            "note": ("best.pt = ep0 (det_pr_auc saturated 1.0 from ep0). Saved "
                     "predictions thus carry the LOWEST localization IoU "
                     f"(median {orig['iou_dist']['median']:.4f}); later epochs "
                     f"reach IoU median ~{max(late) if late else float('nan'):.2f} "
                     "(train.log). Detection metrics (det_pr_auc/presence/fp) are "
                     "saturated from ep0 so are unaffected by the epoch choice."),
        },
    }
    (EVAL / "ours_detection.json").write_text(json.dumps(out, indent=2, default=float))

    # console summary
    print(f"== Ours detection: {NAME} (arch={ARCH}, img={IMG}, frozen head-only) ==")
    print(f"params total={out['params']['total_M']}M trainable={out['params']['trainable_K']}K  best_ep={best_ep}")
    o, b = orig, bal
    print(f"[ORIG  N={o['n']} pos/neg={o['n_pos']}/{o['n_neg']}] "
          f"det_pr={o['det_pr_auc']:.4f} roc={o['det_roc_auc']:.4f} "
          f"presence@.5={o['presence_recall']:.3f}({o['tp']}/{o['n_pos']}) "
          f"fp@.5={o['fp_rate']:.4f}({o['fp']}/{o['n_neg']}) mAP={o['map05']:.4f} "
          f"IoU_med={o['iou_dist']['median']:.4f} mean={o['iou_dist']['mean']:.4f}")
    print(f"      objectness: disease median={o['sep']['pos_median']:.4f} | "
          f"normal median={o['sep']['neg_median']:.5f} (neg>=0.5: {o['sep']['n_neg_ge_0p5']}/{o['n_neg']})")
    print(f"[BAL   N={b['n']} pos/neg={b['n_pos']}/{b['n_neg']}] "
          f"det_pr={b['det_pr_auc']:.4f} roc={b['det_roc_auc']:.4f} "
          f"presence@.5={b['presence_recall']:.3f}({b['tp']}/{b['n_pos']}) "
          f"fp@.5={b['fp_rate']:.4f}({b['fp']}/{b['n_neg']}) mAP={b['map05']:.4f} "
          f"IoU_med={b['iou_dist']['median']:.4f} mean={b['iou_dist']['mean']:.4f}")
    print(f"      objectness: disease median={b['sep']['pos_median']:.4f} | "
          f"normal median={b['sep']['neg_median']:.5f} (neg>=0.5: {b['sep']['n_neg_ge_0p5']}/{b['n_neg']})")
    print("sklearn crosscheck orig:", out["crosscheck"]["orig"]["pr_match"],
          out["crosscheck"]["orig"]["roc_match"],
          "| balanced:", out["crosscheck"]["balanced"]["pr_match"],
          out["crosscheck"]["balanced"]["roc_match"])
    print("best-epoch IoU caveat:", out["best_epoch_iou_caveat"]["note"])
    print(f"wrote {EVAL / 'ours_detection.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
