"""Classifier builder for baseline experiments.

Wraps the `baseline/` backbone classifier classes (as-is, from-scratch, no
pretrained weights). Each baseline class already implements:
    forward(images, labels=None) -> logits[B, num_labels]  (or (loss, logits) if labels given)
with an internal CrossEntropyLoss. We reuse them directly so the runner only
needs `arch`, `num_classes`, `img_size`.

In addition to the as-is `baseline/` wrappers (convnextv2, efficientnetv2,
nextvit), four extra from-scratch backbones are supported for fair comparison:
    - densenet121, resnet50: timm backbones (pretrained=False, num_classes=0)
      wrapped with the same LayerNorm/Linear->GELU->Dropout->Linear head.
    - nextvit20: NeXtViT *base* variant (depths=[3,4,20,3]) via the baseline
      NextViTForImageClassification(model_variant='base').
    - mamba/mambavision: from-scratch Vision-Mamba on mambapy (pscan selective
      scan); see src/models/mamba_vision.py for the design rationale.

The original `mamba_ssm`-based MambaVision is excluded (cannot build in this env);
'mamba'/'mambavision' archs map to the mambapy implementation instead.
Supported archs: convnextv2, efficientnetv2, nextvit, nextvit20,
densenet121, resnet50, mamba, mambavision.

"Ours" arch (pretrained, not from-scratch):
    - dinov3: DINOv3 ViT-S/16 (timm 'vit_small_patch16_dinov3', pretrained=True),
      frozen backbone (requires_grad=False, no-grad/eval forward) + 2-layer MLP
      head (384->256->GELU->Dropout->num_classes). Only the head is trainable.
      See src/models/dinov3.py.
    - dinov3_base: "Ours+" — DINOv3 ViT-B/16 (timm 'vit_base_patch16_dinov3',
      pretrained=True) @512, frozen backbone + 2-layer MLP head
      (768->head_hidden->GELU->Dropout->num_classes, head_hidden default 512).
      Scaled-up variant for maximum performance; only the head is trainable.
      See src/models/dinov3.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn

# Ensure repo root is importable so `baseline.*` and `models.module` resolve
# regardless of the caller's working directory.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Default variant per backbone (smallest "standard" config that the baseline
# wrapper exposes). Kept explicit so specs/runner can rely on it.
_DEFAULT_VARIANT = {
    "convnextv2": "tiny",
    "efficientnetv2": "s",
    "nextvit": "small",
    "nextvit20": "base",  # NeXtViT base: depths=[3,4,20,3]
    "mamba": "base",
    "mambavision": "base",  # alias -> mambapy Vision-Mamba
}

# timm backbones (from-scratch). Mapped to internal timm model names.
_TIMM_ARCH = {
    "densenet121": "densenet121",
    "resnet50": "resnet50",
}


def build_classifier(
    arch: str,
    num_classes: int,
    img_size: int = 224,
    *,
    hidden_dim: int = 512,
    head_hidden: int | None = None,
    model_variant: str | None = None,
) -> nn.Module:
    """Build a baseline image classifier.

    Args:
        arch: one of {"convnextv2", "efficientnetv2", "nextvit"}.
        num_classes: number of output classes (-> num_labels).
        img_size: input resolution (square). Default 224.
        hidden_dim: width of the classifier MLP head. Default 512.
        model_variant: backbone size variant; defaults per-arch if None.

    Returns:
        nn.Module whose forward(images, labels=None) returns logits[B, num_classes]
        (or (loss, logits) when labels are provided). From-scratch init, no pretrained.
    """
    arch = arch.lower()

    # --- DINOv3 ViT-S/16 transfer ("Ours"): frozen self-supervised backbone
    #     + 2-layer MLP head. pretrained+frozen by default so the existing
    #     runner (which calls build_classifier with only arch/num_classes/
    #     img_size) gets the intended transfer/probe behavior. ---
    if arch == "dinov3":
        from src.models.dinov3 import build_dinov3_classifier

        return build_dinov3_classifier(
            num_classes=num_classes,
            img_size=img_size,
            variant="small",
            hidden_dim=256,
            pretrained=True,
            frozen_backbone=True,
        )

    # --- "Ours+": DINOv3 ViT-B/16 @512 frozen + 2-layer MLP head. Scaled-up
    #     variant (768-d feature, wider head) to maximize performance while
    #     keeping the frozen-backbone / no-forgetting recipe. ---
    if arch == "dinov3_base":
        from src.models.dinov3 import build_dinov3_classifier

        return build_dinov3_classifier(
            num_classes=num_classes,
            img_size=img_size,
            variant="base",
            hidden_dim=head_hidden if head_hidden is not None else 512,
            pretrained=True,
            frozen_backbone=True,
        )

    # --- timm backbones (densenet121, resnet50) ---
    if arch in _TIMM_ARCH:
        from src.models.timm_backbone import TimmForImageClassification

        return TimmForImageClassification(
            _TIMM_ARCH[arch], num_labels=num_classes, hidden_dim=hidden_dim
        )

    # --- mambapy Vision-Mamba (from-scratch) ---
    if arch in ("mamba", "mambavision"):
        from src.models.mamba_vision import VisionMambaForImageClassification

        variant = model_variant or _DEFAULT_VARIANT[arch]
        return VisionMambaForImageClassification(
            num_labels=num_classes,
            img_size=img_size,
            hidden_dim=hidden_dim,
            model_variant=variant,
        )

    # --- baseline wrappers (convnextv2, efficientnetv2, nextvit, nextvit20) ---
    if arch not in _DEFAULT_VARIANT:
        raise ValueError(
            f"unsupported arch {arch!r}; expected one of "
            f"{sorted(set(_DEFAULT_VARIANT) | set(_TIMM_ARCH))}"
        )
    variant = model_variant or _DEFAULT_VARIANT[arch]

    if arch == "convnextv2":
        from baseline.convnextv2 import ConvNeXtV2ForImageClassification as Cls
    elif arch == "efficientnetv2":
        from baseline.efficientnetv2 import EfficientNetV2ForImageClassification as Cls
    else:  # nextvit (small) or nextvit20 (base)
        from baseline.nextvit import NextViTForImageClassification as Cls

    return Cls(
        num_labels=num_classes,
        img_size=img_size,
        hidden_dim=hidden_dim,
        model_variant=variant,
    )


if __name__ == "__main__":
    import torch

    x = torch.randn(2, 3, 224, 224)
    y = torch.randint(0, 2, (2,))
    for arch in (
        "convnextv2", "efficientnetv2", "nextvit",
        "nextvit20", "densenet121", "resnet50", "mamba",
    ):
        m = build_classifier(arch, num_classes=2, img_size=224)
        m.eval()
        logits = m(x)
        loss, logits2 = m(x, y)
        n_params = sum(p.numel() for p in m.parameters())
        print(
            f"[classifier] {arch:14s} logits={tuple(logits.shape)} "
            f"loss={float(loss):.4f} params={n_params/1e6:.2f}M"
        )
    print("classifier smoke OK")
