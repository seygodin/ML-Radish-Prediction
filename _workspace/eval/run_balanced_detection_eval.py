"""Balanced-valid re-evaluation of saved detection baselines (eval-reporter QA).

Loads each saved best.pt (weights NOT modified), rebuilds the BALANCED detection
valid loader (balance_valid=True -> disease 100 : normal 100, 1:1, seed=42), runs
forward in fp32, and recomputes the image-level disease-detection + localization
metrics on the BALANCED valid set:

    det_pr_auc (primary), det_roc_auc, presence_recall@0.5, fp_rate@0.5,
    map_at_0.5, IoU distribution (median/mean) + iou_at_0.5_presence

Forward mirrors train.py run_detection eval:
  - objectness = sigmoid(obj_logit) as the image-level disease score
  - pred_box = model box output (already xyxy in [0,1])
  - gt_box = target box / img_size (disease only; normals empty)
  - eval in fp32 (no AMP) for ALL backbones (nextvit/nextvit20/mamba trained amp off).

Notes:
  - presence_recall@0.5 and positive-only IoU depend ONLY on the 100 disease
    images -> identical to §3 (full prevalence).  det_pr_auc / fp_rate / map
    change because #normal goes 1303 -> 100 (prevalence 0.5).

Dumps _workspace/eval/balanced_valid_detection.json. Cross-checks one run vs sklearn.

Run: ./.venv/bin/python _workspace/eval/run_balanced_detection_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data import build_detection_loaders  # noqa: E402
from src.models import build_detector  # noqa: E402
from src.metrics import detection_metrics  # noqa: E402

from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    roc_auc_score,
)

# nextvit20 (NeXtViT-base) excluded from report per user request (artifacts preserved on disk).
ARCHS = ["convnextv2", "efficientnetv2", "nextvit",
         "densenet121", "resnet50", "mamba"]
SEED = 42
EXP = _REPO_ROOT / "experiments"


@torch.no_grad()
def forward_collect(model, loader, device, img_size, with_obj):
    """Forward balanced valid -> (pred_boxes[N,4], scores[N], gt_list, is_pos).

    Mirrors train.py run_detection eval, but always fp32 (no autocast).
    """
    model.eval()
    pred_list, score_list, gt_list, pos_list = [], [], [], []
    sz = float(img_size)
    for images, targets in loader:
        imgs = torch.stack([im for im in images]).to(device, non_blocking=True)
        out = model(imgs)
        if with_obj:
            pred_boxes_t, obj_logit = out
            scores = torch.sigmoid(obj_logit.float()).cpu().numpy()
        else:
            pred_boxes_t = out
            scores = np.ones(pred_boxes_t.shape[0], dtype=np.float32)
        pred_boxes = pred_boxes_t.float().cpu().numpy()
        for j, t in enumerate(targets):
            pred_list.append(pred_boxes[j])
            score_list.append(float(scores[j]))
            b = t["boxes"]
            gt = (b[0] / sz).numpy() if b.numel() else np.zeros((0, 4))
            gt_list.append(np.asarray(gt, dtype=np.float64).reshape(-1, 4))
            pos_list.append(bool(b.numel() > 0))
    pred_arr = np.stack(pred_list, axis=0)
    score_arr = np.asarray(score_list, dtype=np.float64)
    return pred_arr, score_arr, gt_list, np.asarray(pos_list, dtype=bool)


def main() -> int:
    """스크립트 진입점: 예측/체크포인트 로드 → 지표 재계산 → JSON·그림·리포트 산출."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    num_workers = 8

    results = {}
    params_cache = {}
    crosscheck = None

    for arch in ARCHS:
        name = f"{arch}_detection_singlebox"
        run_dir = EXP / name
        snap = json.loads((run_dir / "config.snapshot").read_text())
        mj = json.loads((run_dir / "metrics.json").read_text())
        m_model = snap["spec"]["model"]
        arch_s = m_model["arch"]
        img_size = int(m_model["img_size"])
        with_obj = bool(m_model.get("with_objectness", True))
        assert arch_s == arch, f"{name}: arch mismatch {arch_s}"

        # ---- balanced detection valid loader (1:1, seed fixed) ----
        _, valid_loader, meta = build_detection_loaders(
            img_size=img_size, batch_size=batch_size, num_workers=num_workers,
            seed=SEED, include_normal=True, balance_valid=True,
        )
        valid_counts = meta["valid_counts"]

        # ---- build model + load best.pt (weights only loaded, never changed) ----
        if arch not in params_cache:
            m_tmp = build_detector(arch=arch, img_size=img_size,
                                   with_objectness=with_obj)
            params_cache[arch] = sum(p.numel() for p in m_tmp.parameters()) / 1e6
            del m_tmp
        params_m = params_cache[arch]

        model = build_detector(arch=arch, img_size=img_size,
                               with_objectness=with_obj).to(device)
        ckpt = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device)
        model.load_state_dict(ckpt["model"])

        pred_arr, score_arr, gt_list, is_pos = forward_collect(
            model, valid_loader, device, img_size, with_obj)

        # sanitize non-finite (same guard as train.py) -- should not trigger in fp32
        n_bad = int((~np.isfinite(score_arr)).sum() + (~np.isfinite(pred_arr)).sum())
        if n_bad:
            score_arr = np.nan_to_num(score_arr, nan=0.0, posinf=1.0, neginf=0.0)
            pred_arr = np.nan_to_num(pred_arr, nan=0.0, posinf=0.0, neginf=0.0)

        m = detection_metrics(pred_arr, score_arr, gt_list)
        best_ep = int(mj["final"]["epoch"])

        # ---- sklearn cross-check on the FIRST run (det PR-AUC / ROC-AUC) ----
        if crosscheck is None:
            y = is_pos.astype(np.int64)
            sk_pr = float(average_precision_score(y, score_arr))
            sk_roc = float(roc_auc_score(y, score_arr))
            crosscheck = {
                "run": name,
                "metrics_det_pr_auc": m["det_pr_auc"],
                "sklearn_det_pr_auc": sk_pr,
                "metrics_det_roc_auc": m["det_roc_auc"],
                "sklearn_det_roc_auc": sk_roc,
                "pr_match": abs(m["det_pr_auc"] - sk_pr) < 1e-9,
                "roc_match": abs(m["det_roc_auc"] - sk_roc) < 1e-9,
            }

        # original full-prevalence (§3) for delta comparison
        f = mj["final"]
        results[name] = {
            "arch": arch, "img_size": img_size, "with_objectness": with_obj,
            "params_M": round(params_m, 2), "best_epoch": best_ep,
            "balanced_valid_counts": valid_counts,
            "n_positive": m["n_positive"], "n_negative": m["n_negative"],
            "n_total": m["n_total"],
            # balanced metrics
            "det_pr_auc": m["det_pr_auc"],
            "det_roc_auc": m["det_roc_auc"],
            "presence_recall_at_0.5": m["presence_recall_at_0.5"],
            "fp_rate_at_0.5": m["fp_rate_at_0.5"],
            "map_at_0.5": m["map_at_0.5"],
            "iou_median": m["iou_distribution"]["median"],
            "iou_mean": m["iou_distribution"]["mean"],
            "iou_at_0.5_presence": m["iou_at_0.5_presence"],
            # original full-prevalence (§3, prevalence ~0.071)
            "orig_full": {
                "det_pr_auc": f.get("det_pr_auc"),
                "det_roc_auc": f.get("det_roc_auc"),
                "presence_recall_at_0.5": f.get("presence_recall_at_0.5"),
                "fp_rate_at_0.5": f.get("fp_rate_at_0.5"),
                "map_at_0.5": f.get("map_at_0.5"),
                "iou_median": f.get("iou_distribution", {}).get("median"),
                "iou_mean": f.get("iou_distribution", {}).get("mean"),
                "iou_at_0.5_presence": f.get("iou_at_0.5_presence"),
                "n_positive": f.get("n_positive"),
                "n_negative": f.get("n_negative"),
            },
        }
        print(f"{name:34s} N={m['n_total']:3d} (pos={m['n_positive']} "
              f"neg={m['n_negative']}) det_pr={m['det_pr_auc']:.4f} "
              f"roc={m['det_roc_auc']:.4f} pres@.5={m['presence_recall_at_0.5']:.3f} "
              f"fp@.5={m['fp_rate_at_0.5']:.3f} mAP={m['map_at_0.5']:.4f} "
              f"iou_med={m['iou_distribution']['median']:.3f}", flush=True)
        del model
        torch.cuda.empty_cache()

    print("\nsklearn cross-check:", json.dumps(crosscheck, indent=2))

    out = _REPO_ROOT / "_workspace" / "eval" / "balanced_valid_detection.json"
    out.write_text(json.dumps({"seed": SEED, "balance_valid": True,
                               "crosscheck": crosscheck,
                               "results": results}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
