"""Spec-driven training + evaluation for radish baseline experiments.

CLI:
    python -m src.train --spec _workspace/specs/exp_<name>.yaml [--device cuda:0]
    python -m src.train --spec _workspace/specs/exp_<name>.yaml --smoke --device cuda:0

Reads a self-contained spec yaml, builds data + model via the data/models
public APIs, trains with AdamW + cosine(warmup) schedule, evaluates on valid
each epoch, dumps predictions + metrics + checkpoints under the standard
experiment layout:

    experiments/<name>/
        metrics.json       {status, task, primary, per_epoch:[...], final:{...}, samples}
        config.snapshot     spec + package versions + command + seed + device + gpu
        train.log
        checkpoints/        best.pt, last.pt
        predictions/        valid.npz (classification) / valid.json (detection)

Failure contract: on any error a metrics.json with status="failed" + error
summary is still written, and the process exits with a non-zero code.

Smoke mode (--smoke): epochs forced to 2, train/valid subset-limited, output
written to experiments/_smoke_<name>/ so it never pollutes a real run dir.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW

# Repo root importable regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data import build_classification_loaders, build_detection_loaders  # noqa: E402
from src.losses import build_loss  # noqa: E402
from src.metrics import classification_metrics, detection_metrics  # noqa: E402
from src.models import build_classifier, build_detector  # noqa: E402

# Smoke caps.
_SMOKE_EPOCHS = 2
_SMOKE_TRAIN_BATCHES = 4
_SMOKE_VALID_BATCHES = 4
_SMOKE_BATCH_SIZE = 8


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """random/numpy/torch(+cuda) 시드를 고정해 재현성 확보."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(log_path: Path) -> logging.Logger:
    """run별 파일+stdout 로거 구성(experiments/<name>/train.log)."""
    logger = logging.getLogger(f"train.{log_path.parent.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def package_versions(pkgs: list[str]) -> dict:
    """지정 패키지들의 설치 버전을 dict로 수집(config.snapshot 기록용)."""
    import importlib
    versions = {}
    name_map = {"scikit-learn": "sklearn", "pyyaml": "yaml"}
    for p in pkgs:
        mod = name_map.get(p, p)
        try:
            m = importlib.import_module(mod)
            versions[p] = getattr(m, "__version__", "unknown")
        except Exception as e:  # noqa: BLE001
            versions[p] = f"<not importable: {e}>"
    return versions


def git_state() -> str:
    """현재 git HEAD 커밋 해시(없으면 표식) 반환 — 재현 추적용."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT),
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return "not-a-git-repo"


def cosine_warmup_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    """warmup 후 cosine 감쇠 학습률 스케줄 값을 step에 대해 계산."""
    if total_steps <= 0:
        return base_lr
    if step < warmup_steps and warmup_steps > 0:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def resolve_outdir(name: str, smoke: bool) -> Path:
    """출력 디렉토리 결정. 기존 run은 덮지 않고 _vN으로 분기, smoke는 _smoke_ 격리."""
    base = _REPO_ROOT / "experiments"
    if smoke:
        return base / f"_smoke_{name}"
    out = base / name
    if out.exists():
        # Skill rule: never overwrite an existing real run dir; branch _vN.
        v = 2
        while (base / f"{name}_v{v}").exists():
            v += 1
        return base / f"{name}_v{v}"
    return out


# ---------------------------------------------------------------------------
# Detection loss helpers
# ---------------------------------------------------------------------------
def giou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - GIoU, per-row, for xyxy boxes. pred/target: [M,4]."""
    px0, py0, px1, py1 = pred.unbind(-1)
    tx0, ty0, tx1, ty1 = target.unbind(-1)
    # ensure ordered corners for area computation
    px0_, px1_ = torch.min(px0, px1), torch.max(px0, px1)
    py0_, py1_ = torch.min(py0, py1), torch.max(py0, py1)
    area_p = (px1_ - px0_).clamp(min=0) * (py1_ - py0_).clamp(min=0)
    area_t = (tx1 - tx0).clamp(min=0) * (ty1 - ty0).clamp(min=0)

    ix0 = torch.max(px0_, tx0)
    iy0 = torch.max(py0_, ty0)
    ix1 = torch.min(px1_, tx1)
    iy1 = torch.min(py1_, ty1)
    inter = (ix1 - ix0).clamp(min=0) * (iy1 - iy0).clamp(min=0)
    union = area_p + area_t - inter + 1e-7
    iou = inter / union

    cx0 = torch.min(px0_, tx0)
    cy0 = torch.min(py0_, ty0)
    cx1 = torch.max(px1_, tx1)
    cy1 = torch.max(py1_, ty1)
    area_c = (cx1 - cx0).clamp(min=0) * (cy1 - cy0).clamp(min=0) + 1e-7
    giou = iou - (area_c - union) / area_c
    return 1.0 - giou


# ---------------------------------------------------------------------------
# Training: classification
# ---------------------------------------------------------------------------
def run_classification(spec: dict, args, outdir: Path, logger: logging.Logger,
                       device: torch.device, snapshot: dict) -> dict:
    """분류 spec 학습/평가 루프: 로더·모델 구성, head/trainable만 옵티마이저, epoch별 valid 평가·best 체크포인트·predictions·metrics.json."""
    d = spec["data"]
    o = spec["optim"]
    loss_cfg = spec.get("loss", {})
    seed = int(spec.get("seed", 42))

    batch_size = _SMOKE_BATCH_SIZE if args.smoke else int(d["batch_size"])
    num_workers = 2 if args.smoke else int(d.get("num_workers", 8))

    aug = d.get("aug", "default")
    logger.info("Building classification loaders: setting=%s img=%d bs=%d aug=%s",
                d["setting"], d["img_size"], batch_size, aug)
    train_loader, valid_loader, meta = build_classification_loaders(
        setting=d["setting"], img_size=int(d["img_size"]),
        batch_size=batch_size, num_workers=num_workers, seed=seed,
        aug=aug,
        train_ratio=float(d.get("train_ratio", 1.0)),
    )
    class_names = meta["class_names"]
    num_classes = meta["num_classes"]
    logger.info("class_names=%s train_counts=%s valid_counts=%s",
                class_names, meta["train_counts"], meta["valid_counts"])

    model = build_classifier(
        arch=spec["model"]["arch"], num_classes=num_classes,
        img_size=int(spec["model"]["img_size"]),
    ).to(device)

    # class weights (from data loader meta, per spec 'from_meta')
    use_weights = str(loss_cfg.get("class_weights", "")) == "from_meta"
    class_weights = meta["class_weights"].to(device) if use_weights else None
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))
    loss_type = str(loss_cfg.get("type", "cross_entropy"))
    # Focal is opt-in: only when spec sets loss.type=focal do we build a custom
    # criterion (built ONCE, weights already on device). Otherwise the criterion
    # stays None and the loop uses F.cross_entropy exactly as before (CE path
    # unchanged -> zero regression for existing baselines).
    if loss_type == "focal":
        criterion = build_loss(loss_cfg, meta["class_weights"].to(device))
        snapshot["loss"] = {
            "type": "focal", "gamma": float(loss_cfg.get("gamma", 2.0)),
            "class_weights_used": use_weights,
            "class_weights": meta["class_weights"].tolist(),
            "label_smoothing": label_smoothing,
        }
        logger.info("loss=focal gamma=%.2f class_weights_used=%s label_smoothing=%.3f",
                    float(loss_cfg.get("gamma", 2.0)), use_weights, label_smoothing)
    else:
        criterion = None
        snapshot["loss"] = {
            "type": "cross_entropy", "class_weights_used": use_weights,
            "class_weights": meta["class_weights"].tolist(),
            "label_smoothing": label_smoothing,
        }

    epochs = _SMOKE_EPOCHS if args.smoke else int(o["epochs"])
    warmup_epochs = int(o.get("warmup_epochs", 0))
    # Optimize only trainable params: for a frozen-backbone model (e.g. dinov3
    # "Ours") this excludes requires_grad=False backbone params from the
    # optimizer entirely. For fully-trainable baselines this filter is a no-op
    # (every param has requires_grad=True), so existing baseline runs are
    # unaffected.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info("optimizer params: trainable=%d total=%d (%.2f%%)",
                n_trainable, n_total, 100.0 * n_trainable / max(1, n_total))
    snapshot["trainable_params"] = {"trainable": n_trainable, "total": n_total}
    optimizer = AdamW(trainable_params, lr=float(o["lr"]), weight_decay=float(o["wd"]))

    steps_per_epoch = (_SMOKE_TRAIN_BATCHES if args.smoke else len(train_loader))
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    base_lr = float(o["lr"])

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    primary_key = "pr_auc"
    best_primary = -float("inf")
    per_epoch = []
    global_step = 0

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        n_seen = 0
        for bi, (images, labels) in enumerate(train_loader):
            if args.smoke and bi >= _SMOKE_TRAIN_BATCHES:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            lr = cosine_warmup_lr(global_step, total_steps, warmup_steps, base_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                if criterion is not None:
                    loss = criterion(logits, labels)
                else:
                    loss = F.cross_entropy(
                        logits, labels, weight=class_weights,
                        label_smoothing=label_smoothing,
                    )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss) * images.size(0)
            n_seen += images.size(0)
            global_step += 1
            if bi % 10 == 0:
                logger.info("ep%d step%d loss=%.4f lr=%.2e", epoch, bi, float(loss), lr)

        train_loss = running_loss / max(1, n_seen)

        # ---- validation ----
        model.eval()
        all_probs, all_labels = [], []
        val_loss_sum, val_n = 0.0, 0
        with torch.no_grad():
            for bi, (images, labels) in enumerate(valid_loader):
                if args.smoke and bi >= _SMOKE_VALID_BATCHES:
                    break
                images = images.to(device, non_blocking=True)
                labels_d = labels.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(images)
                    if criterion is not None:
                        vloss = criterion(logits, labels_d)
                    else:
                        vloss = F.cross_entropy(
                            logits, labels_d, weight=class_weights,
                            label_smoothing=label_smoothing,
                        )
                val_loss_sum += float(vloss) * images.size(0)
                val_n += images.size(0)
                probs = F.softmax(logits.float(), dim=1)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.numpy())
        probs = np.concatenate(all_probs, axis=0)
        labels_np = np.concatenate(all_labels, axis=0)
        val_loss = val_loss_sum / max(1, val_n)
        m = classification_metrics(probs, labels_np, class_names)
        primary = m[primary_key]
        dt = time.time() - t0
        logger.info(
            "[ep%d] train_loss=%.4f val_loss=%.4f val_pr_auc=%.4f auroc=%.4f "
            "f1_macro=%.4f recall_macro=%.4f prec_macro=%.4f acc=%.4f (%.1fs)",
            epoch, train_loss, val_loss, m["pr_auc"], m["auroc"],
            m["f1_macro"], m["recall_macro"], m["precision_macro"],
            m["accuracy"], dt,
        )
        per_epoch.append({"epoch": epoch, "train_loss": train_loss,
                          "val_loss": val_loss, "seconds": dt, **m})

        # checkpoints
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "metrics": m}, outdir / "checkpoints" / "last.pt")
        cmp_primary = primary if not math.isnan(primary) else -float("inf")
        if cmp_primary > best_primary:
            best_primary = cmp_primary
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "metrics": m}, outdir / "checkpoints" / "best.pt")
            # dump predictions for the best epoch (eval-reporter reads these)
            np.savez_compressed(
                outdir / "predictions" / "valid.npz",
                prob=probs.astype(np.float32),
                label=labels_np.astype(np.int64),
                class_names=np.array(class_names),
            )

    # final metrics = best epoch's full metric set
    best_epoch = max(per_epoch, key=lambda r: (r["pr_auc"]
                     if not math.isnan(r["pr_auc"]) else -1.0)) if per_epoch else {}
    # Required 7-key final contract (best epoch): accuracy, train_loss, val_loss,
    # recall(macro), precision(macro), f1(macro), auroc. Legacy keys (pr_auc,
    # recall_disease, ...) are preserved alongside for eval-reporter compatibility.
    final = {k: best_epoch.get(k) for k in
             ["epoch", "pr_auc", "macro_f1", "recall_disease",
              "precision_disease", "accuracy", "confusion",
              "recall_macro", "precision_macro", "f1_macro", "auroc",
              "train_loss", "val_loss"]}
    final["recall"] = best_epoch.get("recall_macro")
    final["precision"] = best_epoch.get("precision_macro")
    final["f1"] = best_epoch.get("f1_macro")
    return {
        "task": "classification",
        "primary": "pr_auc",
        "per_epoch": per_epoch,
        "final": final,
        "samples": {"train": int(sum(meta["train_counts"].values())),
                    "valid": int(sum(meta["valid_counts"].values()))},
        "meta": {"class_names": class_names,
                 "train_counts": meta["train_counts"],
                 "valid_counts": meta["valid_counts"]},
    }


# ---------------------------------------------------------------------------
# Training: detection
# ---------------------------------------------------------------------------
def run_detection(spec: dict, args, outdir: Path, logger: logging.Logger,
                  device: torch.device, snapshot: dict) -> dict:
    """detection spec 학습/평가 루프: 단일박스 GIoU+L1(양성만)+objectness BCE, 이미지단위 검출+IoU 평가."""
    d = spec["data"]
    o = spec["optim"]
    loss_cfg = spec.get("loss", {})
    seed = int(spec.get("seed", 42))

    batch_size = _SMOKE_BATCH_SIZE if args.smoke else int(d["batch_size"])
    num_workers = 2 if args.smoke else int(d.get("num_workers", 8))

    logger.info("Building detection loaders: img=%d bs=%d include_normal=%s",
                d["img_size"], batch_size, d.get("include_normal", False))
    train_loader, valid_loader, meta = build_detection_loaders(
        img_size=int(d["img_size"]), batch_size=batch_size,
        num_workers=num_workers, seed=seed,
        include_normal=bool(d.get("include_normal", False)),
    )
    logger.info("train_counts=%s valid_counts=%s",
                meta["train_counts"], meta["valid_counts"])

    with_obj = bool(spec["model"].get("with_objectness", True))
    model = build_detector(
        arch=spec["model"]["arch"], img_size=int(spec["model"]["img_size"]),
        with_objectness=with_obj,
    ).to(device)

    giou_w = float(loss_cfg.get("giou_weight", 2.0))
    l1_w = float(loss_cfg.get("l1_weight", 5.0))
    obj_w = float(loss_cfg.get("obj_weight", 1.0))
    snapshot["loss"] = {"type": "giou_l1_obj", "giou_weight": giou_w,
                        "l1_weight": l1_w, "obj_weight": obj_w,
                        "box_loss_on": "positive_only", "with_objectness": with_obj}

    epochs = _SMOKE_EPOCHS if args.smoke else int(o["epochs"])
    warmup_epochs = int(o.get("warmup_epochs", 0))
    optimizer = AdamW(model.parameters(), lr=float(o["lr"]), weight_decay=float(o["wd"]))
    steps_per_epoch = (_SMOKE_TRAIN_BATCHES if args.smoke else len(train_loader))
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    base_lr = float(o["lr"])

    # AMP can be disabled per-spec (optim.amp: false) -- NeXt-ViT detection was
    # observed to emit NaN logits under fp16 autocast, so it runs in fp32.
    amp_enabled = bool(o.get("amp", True))
    use_amp = device.type == "cuda" and amp_enabled
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    snapshot["amp"] = {"enabled": use_amp, "spec_amp": amp_enabled}
    if not amp_enabled:
        snapshot["adjustments"].append(
            "AMP disabled for detection (optim.amp=false) due to fp16 NaN "
            "instability (NeXt-ViT); training/eval run in fp32.")
        logger.info("AMP disabled (optim.amp=false): running detection in fp32")

    primary_key = "det_pr_auc"
    best_primary = -float("inf")
    per_epoch = []
    global_step = 0

    def _targets_to_tensors(targets):
        # box (xyxy in pixels per loader) -> normalize to [0,1] by img_size.
        # Loader resizes to (img_size, img_size); boxes are in that pixel space.
        """detection 타깃 리스트를 (gt_boxes, has_box) 텐서로 변환(빈 박스=음성→has_box 0)."""
        sz = float(spec["model"]["img_size"])
        boxes, has_box = [], []
        for t in targets:
            b = t["boxes"]
            if b.numel() == 0:
                boxes.append(torch.zeros(4))
                has_box.append(0.0)
            else:
                boxes.append(b[0] / sz)  # single box
                has_box.append(1.0)
        return torch.stack(boxes).to(device), torch.tensor(has_box, device=device)

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        n_batches = 0
        for bi, (images, targets) in enumerate(train_loader):
            if args.smoke and bi >= _SMOKE_TRAIN_BATCHES:
                break
            imgs = torch.stack([im for im in images]).to(device, non_blocking=True)
            gt_boxes, has_box = _targets_to_tensors(targets)

            lr = cosine_warmup_lr(global_step, total_steps, warmup_steps, base_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(imgs)
                if with_obj:
                    pred_boxes, obj_logit = out
                else:
                    pred_boxes, obj_logit = out, None
                pred_boxes = pred_boxes.float()

                pos = has_box > 0.5
                if pos.any():
                    pb = pred_boxes[pos]
                    gb = gt_boxes[pos]
                    l_giou = giou_loss(pb, gb).mean()
                    l_l1 = F.l1_loss(pb, gb)
                    box_loss = giou_w * l_giou + l1_w * l_l1
                else:
                    box_loss = pred_boxes.sum() * 0.0

                if with_obj:
                    obj_loss = F.binary_cross_entropy_with_logits(
                        obj_logit.float(), has_box)
                    loss = box_loss + obj_w * obj_loss
                else:
                    loss = box_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss)
            n_batches += 1
            global_step += 1
            if bi % 10 == 0:
                logger.info("ep%d step%d loss=%.4f lr=%.2e", epoch, bi, float(loss), lr)

        train_loss = running_loss / max(1, n_batches)

        # ---- validation ----
        model.eval()
        pred_list, score_list, gt_list, id_list, pos_list = [], [], [], [], []
        img_counter = 0
        val_loss_sum, val_n = 0.0, 0
        with torch.no_grad():
            for bi, (images, targets) in enumerate(valid_loader):
                if args.smoke and bi >= _SMOKE_VALID_BATCHES:
                    break
                imgs = torch.stack([im for im in images]).to(device, non_blocking=True)
                gt_boxes, has_box = _targets_to_tensors(targets)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(imgs)
                    if with_obj:
                        pred_boxes_t, obj_logit = out
                    else:
                        pred_boxes_t, obj_logit = out, None
                    pred_boxes_t = pred_boxes_t.float()
                    pos = has_box > 0.5
                    if pos.any():
                        vbox = (giou_w * giou_loss(pred_boxes_t[pos], gt_boxes[pos]).mean()
                                + l1_w * F.l1_loss(pred_boxes_t[pos], gt_boxes[pos]))
                    else:
                        vbox = pred_boxes_t.sum() * 0.0
                    if with_obj:
                        vbox = vbox + obj_w * F.binary_cross_entropy_with_logits(
                            obj_logit.float(), has_box)
                val_loss_sum += float(vbox) * imgs.size(0)
                val_n += imgs.size(0)
                if with_obj:
                    pred_boxes, obj_logit = out
                    scores = torch.sigmoid(obj_logit.float()).cpu().numpy()
                else:
                    pred_boxes = out
                    scores = np.ones(pred_boxes.shape[0], dtype=np.float32)
                pred_boxes = pred_boxes.float().cpu().numpy()
                sz = float(spec["model"]["img_size"])
                for j, t in enumerate(targets):
                    pred_list.append(pred_boxes[j])
                    score_list.append(float(scores[j]))
                    b = t["boxes"]
                    gt = (b[0] / sz).numpy() if b.numel() else np.zeros((0, 4))
                    gt_list.append(gt.reshape(-1, 4))
                    pos_list.append(bool(b.numel() > 0))
                    id_list.append(img_counter)
                    img_counter += 1

        pred_arr = np.stack(pred_list, axis=0)
        score_arr = np.asarray(score_list, dtype=np.float64)
        val_loss = val_loss_sum / max(1, val_n)
        # Guard against NaN/Inf model outputs (e.g. AMP fp16 overflow) so a single
        # bad eval cannot crash the whole run inside sklearn. Sanitize + log.
        n_bad_score = int((~np.isfinite(score_arr)).sum())
        n_bad_pred = int((~np.isfinite(pred_arr)).sum())
        if n_bad_score or n_bad_pred:
            logger.warning("ep%d: non-finite outputs (scores=%d preds=%d) -> "
                           "sanitized to 0 for metric computation",
                           epoch, n_bad_score, n_bad_pred)
            score_arr = np.nan_to_num(score_arr, nan=0.0, posinf=1.0, neginf=0.0)
            pred_arr = np.nan_to_num(pred_arr, nan=0.0, posinf=0.0, neginf=0.0)
        m = detection_metrics(pred_arr, score_arr, gt_list)
        primary = m[primary_key]
        dt = time.time() - t0
        logger.info(
            "[ep%d] train_loss=%.4f val_loss=%.4f det_pr_auc=%.4f det_roc_auc=%.4f "
            "presence_rec@0.5=%.4f fp_rate@0.5=%.4f iou_med=%.4f mAP@0.5=%.4f "
            "n_pos=%d n_neg=%d (%.1fs)",
            epoch, train_loss, val_loss, m["det_pr_auc"], m["det_roc_auc"],
            m["presence_recall_at_0.5"], m["fp_rate_at_0.5"],
            m["iou_distribution"]["median"], m["map_at_0.5"],
            m["n_positive"], m["n_negative"], dt,
        )
        per_epoch.append({"epoch": epoch, "train_loss": train_loss,
                          "val_loss": val_loss, "seconds": dt, **m})

        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "metrics": m}, outdir / "checkpoints" / "last.pt")
        cmp_primary = primary if not math.isnan(primary) else -float("inf")
        # Always keep at least one best/predictions even if primary is nan
        # (degenerate valid subset, e.g. smoke) by treating the first epoch as best.
        is_best = cmp_primary > best_primary or not (
            outdir / "predictions" / "valid.json").exists()
        if is_best:
            best_primary = max(best_primary, cmp_primary)
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "metrics": m}, outdir / "checkpoints" / "best.pt")
            # dump predictions (eval-reporter recomputes IoU/mAP)
            with open(outdir / "predictions" / "valid.json", "w") as f:
                json.dump({
                    "format": "xyxy_normalized_0_1",
                    "img_size": int(spec["model"]["img_size"]),
                    "score_meaning": "objectness = image-level disease score",
                    "pred_boxes": pred_arr.tolist(),
                    "scores": score_arr.tolist(),
                    "gt_boxes": [g.tolist() for g in gt_list],
                    "is_positive": [bool(p) for p in pos_list],
                    "image_ids": id_list,
                }, f)

    best_epoch = max(
        per_epoch,
        key=lambda r: (r[primary_key]
                       if not math.isnan(r[primary_key]) else -1.0),
    ) if per_epoch else {}
    return {
        "task": "detection",
        "primary": primary_key,
        "per_epoch": per_epoch,
        "final": {k: best_epoch.get(k) for k in
                  ["epoch", "det_pr_auc", "det_roc_auc",
                   "presence_recall_at_0.5", "fp_rate_at_0.5",
                   "iou_at_0.5_presence", "iou_distribution", "map_at_0.5",
                   "n_positive", "n_negative", "n_total"]},
        "samples": {"train": int(sum(meta["train_counts"].values())),
                    "valid": int(sum(meta["valid_counts"].values()))},
        "meta": {"train_counts": meta["train_counts"],
                 "valid_counts": meta["valid_counts"],
                 "include_normal": meta["include_normal"]},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    """CLI 진입점: spec 로드 → task 분기 → 표준 산출물 생성. 실패해도 metrics.json(status:failed) 기록."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, type=str)
    ap.add_argument("--device", default="cuda:0", type=str)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    spec_path = Path(args.spec).resolve()
    with open(spec_path) as f:
        spec = yaml.safe_load(f)

    name = spec["name"]
    task = spec["task"]
    outdir = resolve_outdir(name, args.smoke)
    (outdir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (outdir / "predictions").mkdir(parents=True, exist_ok=True)

    logger = setup_logger(outdir / "train.log")
    logger.info("=== experiment-runner :: %s (task=%s smoke=%s) ===",
                name, task, args.smoke)
    logger.info("spec=%s outdir=%s device=%s", spec_path, outdir, args.device)

    seed = int(spec.get("seed", 42))
    set_seed(seed)
    torch.backends.cudnn.benchmark = True  # variable-size friendly, speed
    torch.backends.cudnn.deterministic = False
    logger.info("seed=%d cudnn.benchmark=True deterministic=False", seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    gpu_name = (torch.cuda.get_device_name(device)
                if device.type == "cuda" else "cpu")
    logger.info("device=%s gpu=%s", device, gpu_name)

    # config snapshot (frozen reproducibility info)
    snapshot = {
        "name": name, "task": task, "spec_path": str(spec_path), "spec": spec,
        "command": "python -m src.train " + " ".join(sys.argv[1:]),
        "seed": seed, "device": str(device), "gpu_name": gpu_name,
        "smoke": args.smoke,
        "package_versions": package_versions(spec.get("packages", [])),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "git_head": git_state(),
        "cudnn": {"benchmark": True, "deterministic": False},
        "adjustments": [],
    }

    start = time.time()
    status = "ok"
    error = None
    result = {}
    try:
        if task == "classification":
            result = run_classification(spec, args, outdir, logger, device, snapshot)
        elif task == "detection":
            result = run_detection(spec, args, outdir, logger, device, snapshot)
        else:
            raise ValueError(f"unknown task {task!r}")
    except torch.cuda.OutOfMemoryError as e:  # noqa: BLE001
        status = "failed"
        error = f"CUDA OOM: {e}"
        snapshot["adjustments"].append(
            "OOM encountered; reduce batch_size in spec and rerun "
            "(runner does not silently mutate spec batch on real runs)."
        )
        logger.error(error)
        logger.error(traceback.format_exc())
    except Exception as e:  # noqa: BLE001
        status = "failed"
        error = f"{type(e).__name__}: {e}"
        logger.error(error)
        logger.error(traceback.format_exc())

    elapsed = time.time() - start
    metrics = {"status": status, "name": name, "task": task,
               "elapsed_seconds": elapsed, **result}
    if error:
        metrics["error"] = error

    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(outdir / "config.snapshot", "w") as f:
        json.dump(snapshot, f, indent=2, default=str)

    logger.info("=== DONE status=%s elapsed=%.1fs outdir=%s ===",
                status, elapsed, outdir)
    if status != "ok":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
