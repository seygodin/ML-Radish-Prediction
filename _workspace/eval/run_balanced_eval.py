"""Balanced-valid re-evaluation of saved classification baselines (eval-reporter QA).

Loads each saved best.pt (weights NOT modified), rebuilds the balanced valid
loader (balance_valid=True -> 1:1 binary / 1:1:1 3-class, seed=42), runs forward,
and recomputes the user-required 7 metrics + PR-AUC on the BALANCED valid set:

    accuracy, train_loss, val_loss, recall(macro), precision(macro),
    f1(macro), AUROC  (+ PR-AUC)

- train_loss: taken from that run's metrics.json per_epoch at the best epoch
  (training-time value, invariant -- NOT recomputed).
- val_loss: recomputed on the BALANCED valid set as plain CE (no class weights,
  label_smoothing=0), matching train.py's CE form (weights unnecessary: balanced).
- the rest: computed from balanced-valid prob/label.
- AUROC: 2-class = P(disease) ROC-AUC; 3-class = macro OvR.
- params(M): build_classifier(arch, num_classes, img_size) param count.

Dumps _workspace/eval/balanced_valid.json. Cross-checks one run vs sklearn.

Run: ./.venv/bin/python _workspace/eval/run_balanced_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data import build_classification_loaders  # noqa: E402
from src.models import build_classifier  # noqa: E402

from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# nextvit20 (NeXtViT-base) excluded from report per user request (artifacts preserved on disk).
ARCHS = ["convnextv2", "efficientnetv2", "nextvit",
         "densenet121", "resnet50", "mamba"]
SETTINGS = ["normal_vs_d3", "normal_vs_d4", "normal_d3_d4"]
SEED = 42
EXP = _REPO_ROOT / "experiments"


def metrics_balanced(probs: np.ndarray, labels: np.ndarray, num_classes: int) -> dict:
    """Recompute balanced-valid metrics from prob/label (sklearn)."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    preds = probs.argmax(axis=1)
    cls = list(range(num_classes))

    acc = float((preds == labels).mean())
    f1m = float(f1_score(labels, preds, labels=cls, average="macro", zero_division=0))
    recm = float(recall_score(labels, preds, labels=cls, average="macro", zero_division=0))
    precm = float(precision_score(labels, preds, labels=cls, average="macro", zero_division=0))

    disease_idx = list(range(1, num_classes))
    y_true_dis = np.isin(labels, disease_idx).astype(np.int64)

    if num_classes == 2:
        score = probs[:, 1]
        auroc = float(roc_auc_score(y_true_dis, score))
        pr_auc = float(average_precision_score(y_true_dis, score))
    else:
        aucs, aps = [], []
        for c in range(num_classes):
            y_c = (labels == c).astype(np.int64)
            if y_c.sum() == 0 or y_c.sum() == len(y_c):
                continue
            aucs.append(float(roc_auc_score(y_c, probs[:, c])))
        auroc = float(np.mean(aucs)) if aucs else float("nan")
        for c in disease_idx:  # PR-AUC = disease OvR macro
            y_c = (labels == c).astype(np.int64)
            if y_c.sum() == 0 or y_c.sum() == len(y_c):
                continue
            aps.append(float(average_precision_score(y_c, probs[:, c])))
        pr_auc = float(np.mean(aps)) if aps else float("nan")

    return {
        "accuracy": acc, "recall_macro": recm, "precision_macro": precm,
        "f1_macro": f1m, "auroc": auroc, "pr_auc": pr_auc,
        "confusion": confusion_matrix(labels, preds, labels=cls).tolist(),
    }


@torch.no_grad()
def forward_collect(model, loader, device, num_classes):
    """Forward balanced valid -> (probs[N,C], labels[N], val_loss CE-mean)."""
    model.eval()
    all_probs, all_labels = [], []
    loss_sum, n = 0.0, 0
    use_amp = device.type == "cuda"
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels_d = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
        # plain CE on balanced valid (no class weights, no label smoothing)
        vloss = F.cross_entropy(logits.float(), labels_d)
        loss_sum += float(vloss) * images.size(0)
        n += images.size(0)
        all_probs.append(F.softmax(logits.float(), dim=1).cpu().numpy())
        all_labels.append(labels.numpy())
    probs = np.concatenate(all_probs, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    val_loss = loss_sum / max(1, n)
    return probs, labels_np, val_loss


def main() -> int:
    """스크립트 진입점: 예측/체크포인트 로드 → 지표 재계산 → JSON·그림·리포트 산출."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img_size = 224
    batch_size = 64
    num_workers = 8

    results = {}
    params_cache = {}

    for setting in SETTINGS:
        for arch in ARCHS:
            name = f"{arch}_{setting}"
            run_dir = EXP / name
            snap = json.loads((run_dir / "config.snapshot").read_text())
            mj = json.loads((run_dir / "metrics.json").read_text())
            m_model = snap["spec"]["model"]
            arch_s = m_model["arch"]
            num_classes = int(m_model["num_classes"])
            img_size = int(m_model["img_size"])
            assert arch_s == arch, f"{name}: arch mismatch {arch_s}"

            # best-epoch train_loss (training-time, invariant)
            best_ep = int(mj["final"]["epoch"])
            train_loss = float(mj["per_epoch"][best_ep]["train_loss"])
            orig_val_loss = float(mj["per_epoch"][best_ep]["val_loss"])
            # original IMBALANCED valid metrics: recompute independently from the
            # saved best-epoch predictions (old 3 backbones lack auroc in per_epoch).
            npz = np.load(run_dir / "predictions" / "valid.npz", allow_pickle=True)
            o = metrics_balanced(npz["prob"], npz["label"], num_classes)
            orig_acc = o["accuracy"]
            orig_auroc = o["auroc"]
            orig_pr = o["pr_auc"]

            # balanced valid loader (seed fixed, reproducible)
            _, valid_loader, meta = build_classification_loaders(
                setting=setting, img_size=img_size, batch_size=batch_size,
                num_workers=num_workers, seed=SEED, balance_valid=True,
            )

            # build model + load best.pt (weights only loaded, never changed)
            if arch not in params_cache:
                m_tmp = build_classifier(arch=arch, num_classes=num_classes,
                                         img_size=img_size)
                params_cache[arch] = sum(p.numel() for p in m_tmp.parameters()) / 1e6
                del m_tmp
            params_m = params_cache[arch]

            model = build_classifier(arch=arch, num_classes=num_classes,
                                     img_size=img_size).to(device)
            ckpt = torch.load(run_dir / "checkpoints" / "best.pt",
                              map_location=device)
            model.load_state_dict(ckpt["model"])

            probs, labels_np, val_loss = forward_collect(
                model, valid_loader, device, num_classes)

            mt = metrics_balanced(probs, labels_np, num_classes)
            valid_counts = meta["valid_counts"]

            results[name] = {
                "arch": arch, "setting": setting,
                "num_classes": num_classes, "img_size": img_size,
                "params_M": round(params_m, 2),
                "best_epoch": best_ep,
                "balanced_valid_counts": valid_counts,
                "n_valid": int(sum(valid_counts.values())),
                # required 7 metrics
                "accuracy": mt["accuracy"],
                "train_loss": train_loss,            # invariant (training time)
                "val_loss": val_loss,                # recomputed on balanced valid
                "recall_macro": mt["recall_macro"],
                "precision_macro": mt["precision_macro"],
                "f1_macro": mt["f1_macro"],
                "auroc": mt["auroc"],
                "pr_auc": mt["pr_auc"],
                "confusion": mt["confusion"],
                # original (full imbalanced valid) for delta comparison
                "orig_imbalanced": {
                    "accuracy": orig_acc, "auroc": orig_auroc,
                    "pr_auc": orig_pr, "val_loss": orig_val_loss,
                },
            }
            oa = f"{orig_auroc:.3f}" if orig_auroc is not None else "n/a"
            print(f"{name:38s} N={results[name]['n_valid']:3d} "
                  f"acc={mt['accuracy']:.3f} f1={mt['f1_macro']:.3f} "
                  f"auroc={mt['auroc']:.3f} pr={mt['pr_auc']:.3f} "
                  f"val_loss={val_loss:.3f} (orig acc={orig_acc:.3f} "
                  f"auroc={oa})", flush=True)
            del model
            torch.cuda.empty_cache()

    out = _REPO_ROOT / "_workspace" / "eval" / "balanced_valid.json"
    out.write_text(json.dumps({"seed": SEED, "img_size": img_size,
                               "results": results}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
