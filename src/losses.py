"""Loss functions for radish disease classification (ml-researcher).

Currently provides a multiclass focal loss used by the "Ours+ focal+aug"
experiments (DINOv3 ViT-B/16 frozen @512). The module is intentionally
self-contained so that `train.py` can swap it in for `F.cross_entropy`
when a spec sets `loss.type: focal`.

Focal loss (Lin et al., 2017) reshapes standard cross-entropy so that
well-classified (easy) examples are down-weighted and the model focuses on
hard examples:

    FL = (1 - p_t)^gamma * CE

where p_t is the predicted probability of the true class. With gamma=0 this
reduces exactly to (weighted, label-smoothed) cross-entropy. The `weight`
argument plays the per-class alpha role (e.g. meta['class_weights']), and
`label_smoothing` matches torch's CE semantics so focal and CE runs stay
directly comparable.

Works for both binary (C=2) and multiclass (C=3) logits — it operates on
the full softmax, so 2-class and 3-class settings use the same code path.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multiclass focal loss on top of cross-entropy.

    Args:
        gamma: focusing parameter (>= 0). gamma=0 -> plain (weighted,
            label-smoothed) cross-entropy. gamma=2.0 is the paper default.
        weight: optional per-class weights, shape [C] (alpha role). Passed
            through to the CE term exactly like ``F.cross_entropy(weight=...)``.
        label_smoothing: in [0, 1), same semantics as torch CE label_smoothing.
        reduction: 'mean' | 'sum' | 'none'.

    forward(logits[B, C], targets[B]) -> scalar (or [B] if reduction='none').

    Numerical stability: uses log_softmax (no manual exp of large logits).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        """FocalLoss 설정(gamma, 클래스 가중 weight=alpha, label_smoothing)."""
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(
                f"label_smoothing must be in [0, 1), got {label_smoothing}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"unknown reduction {reduction!r}")
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.reduction = reduction
        # Register as buffer so .to(device)/.cuda() moves it with the module,
        # but allow None.
        if weight is not None:
            self.register_buffer("weight", weight.float())
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """logits/targets에 대해 FL=(1-p_t)^gamma·CE 계산(log_softmax로 수치 안정)."""
        if logits.dim() != 2:
            raise ValueError(
                f"expected logits [B, C], got shape {tuple(logits.shape)}")
        num_classes = logits.size(1)

        # log p(class) for every class, numerically stable.
        log_probs = F.log_softmax(logits, dim=1)            # [B, C]
        probs = log_probs.exp()                              # [B, C]

        # Per-sample CE term (with optional label smoothing + class weight),
        # computed WITHOUT reduction so we can apply the focal modulation
        # per-sample before reducing. We reproduce torch CE label-smoothing
        # semantics: loss_i = (1 - eps) * nll_i + eps * mean_c(-w_c * log p_ic),
        # where the smoothing term is a weighted mean over classes.
        weight = self.weight
        if weight is not None:
            weight = weight.to(dtype=logits.dtype, device=logits.device)

        # gather the true-class log-prob and prob
        tgt = targets.view(-1, 1)                            # [B, 1]
        logp_t = log_probs.gather(1, tgt).squeeze(1)         # [B]
        p_t = probs.gather(1, tgt).squeeze(1)                # [B]

        # focal modulation factor (1 - p_t)^gamma
        focal_factor = (1.0 - p_t).clamp(min=0.0).pow(self.gamma)  # [B]

        # per-class weight applied to the true class (alpha role)
        if weight is not None:
            w_t = weight.gather(0, targets)                  # [B]
        else:
            w_t = torch.ones_like(p_t)

        # main (hard-label) focal-NLL term
        nll = -logp_t                                        # [B]
        loss_main = w_t * focal_factor * nll                 # [B]

        eps = self.label_smoothing
        if eps > 0.0:
            # Per-sample smoothing term that reproduces torch CE semantics
            # exactly (verified empirically), so gamma=0 reduces to
            # F.cross_entropy(weight=..., label_smoothing=eps):
            #   unweighted: smooth_i = mean_c(-log p_ic)             (= /C)
            #   weighted:   smooth_i = sum_c(-w_c * log p_ic) / C
            # The mean reduction divides the batch-summed combined loss by
            #   sum_i w_{y_i}  (unweighted -> B).
            # We apply the focal factor to the smoothing term as well so that
            # smoothing stays consistent with the focused main term. With
            # gamma=0, focal_factor==1, so this is exact CE.
            if weight is not None:
                smooth_per_sample = (
                    -(log_probs * weight.view(1, -1)).sum(dim=1) / num_classes)
            else:
                smooth_per_sample = -log_probs.mean(dim=1)
            loss_smooth = focal_factor * smooth_per_sample
            loss = (1.0 - eps) * loss_main + eps * loss_smooth
        else:
            loss = loss_main

        if self.reduction == "mean":
            # Normalize by the sum of applied weights (mirrors torch CE's
            # weighted-mean behaviour) so weighted/unweighted are comparable.
            if weight is not None:
                return loss.sum() / w_t.sum().clamp(min=1e-12)
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss  # 'none'


def build_loss(loss_cfg: dict, class_weights: torch.Tensor | None):
    """Spec -> loss module. Contract used by train.py (experiment-runner wires).

    loss_cfg fields:
        type: 'cross_entropy' (default) | 'focal'
        gamma: float (focal only; default 2.0)
        class_weights: 'from_meta' -> pass class_weights; else None
        label_smoothing: float (default 0.0)

    For 'cross_entropy', train.py keeps using F.cross_entropy directly; this
    helper covers the 'focal' branch. Returned module's forward(logits,
    targets) -> scalar matches the F.cross_entropy call site.
    """
    use_weights = str(loss_cfg.get("class_weights", "")) == "from_meta"
    weight = class_weights if use_weights else None
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))
    ltype = str(loss_cfg.get("type", "cross_entropy"))
    if ltype == "focal":
        return FocalLoss(
            gamma=float(loss_cfg.get("gamma", 2.0)),
            weight=weight,
            label_smoothing=label_smoothing,
        )
    raise ValueError(
        f"build_loss only constructs 'focal' here; got type={ltype!r}. "
        "cross_entropy is handled inline by train.py via F.cross_entropy.")


# ---------------------------------------------------------------------------
# Smoke test: forward/backward on random tensors (2-class & 3-class) + a
# gamma=0 == cross_entropy sanity check.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    def _check(num_classes: int, gamma: float, weighted: bool, ls: float):
        """내부 sanity check 헬퍼."""
        B = 16
        logits = torch.randn(B, num_classes, requires_grad=True)
        targets = torch.randint(0, num_classes, (B,))
        weight = torch.rand(num_classes) + 0.5 if weighted else None
        fl = FocalLoss(gamma=gamma, weight=weight, label_smoothing=ls)
        loss = fl(logits, targets)
        loss.backward()
        gsum = float(logits.grad.abs().sum())
        lval = float(loss.detach())
        finite = torch.isfinite(loss).item() and (logits.grad.isfinite().all().item())
        print(f"C={num_classes} gamma={gamma} weighted={weighted} ls={ls}: "
              f"loss={lval:.4f} grad_sum={gsum:.4f} finite={finite}")
        return lval

    print("== forward/backward smoke ==")
    for C in (2, 3):
        for g in (0.0, 2.0):
            for w in (False, True):
                for ls in (0.0, 0.1):
                    _check(C, g, w, ls)

    print("\n== gamma=0 == cross_entropy sanity ==")
    for C in (2, 3):
        for weighted in (False, True):
            for ls in (0.0, 0.1):
                B = 32
                logits = torch.randn(B, C)
                targets = torch.randint(0, C, (B,))
                weight = torch.rand(C) + 0.5 if weighted else None
                fl0 = FocalLoss(gamma=0.0, weight=weight, label_smoothing=ls)
                fl_val = float(fl0(logits, targets))
                ce_val = float(F.cross_entropy(
                    logits, targets, weight=weight, label_smoothing=ls))
                diff = abs(fl_val - ce_val)
                ok = diff < 1e-5
                print(f"C={C} weighted={weighted} ls={ls}: focal={fl_val:.6f} "
                      f"ce={ce_val:.6f} diff={diff:.2e} match={ok}")
                assert ok, f"gamma=0 focal != CE (diff={diff})"

    print("\nALL CHECKS PASSED")
