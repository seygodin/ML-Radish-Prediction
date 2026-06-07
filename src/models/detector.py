"""Single-box detector builder for baseline experiments.

The dataset has exactly ONE coarse bounding box per image (see report/REPORT.md
and data_card.md). Rather than bolting an anchor/FPN detector onto backbones whose
`forward_features` only returns a *pooled* (B, C) vector, we use the most honest
baseline for these as-is backbones: direct single-box regression.

    backbone.forward_features(x) -> (B, C) pooled feature
    -> Linear head -> box[4] in [0,1] (xyxy, sigmoid)  (+ optional objectness logit)

Loss (computed in the runner, not here): GIoU + L1 on the box, optional BCE on
objectness. The head exposes `pred_boxes[B,4]` from forward(images); when
`with_objectness=True`, forward returns (pred_boxes[B,4], obj_logit[B]).

In addition to the as-is baseline backbones, four extra from-scratch backbones
are supported (same pooled-feature contract -> same SingleBoxDetector head):
    - densenet121, resnet50: timm pooled features (1024 / 2048).
    - nextvit20: NeXtViT base backbone (pooled 1024).
    - mamba/mambavision: mambapy Vision-Mamba backbone (pooled d_model=256).

The original mamba_ssm MambaVision is excluded; 'mamba'/'mambavision' map to the
mambapy implementation. Supported archs: convnextv2, efficientnetv2, nextvit,
nextvit20, densenet121, resnet50, mamba, mambavision.

"Ours" detection: 'dinov3_base' (and 'dinov3') wrap the self-supervised DINOv3
ViT backbone (frozen) behind the same SingleBoxDetector head. The backbone is a
DINOv3ForImageClassification configured with frozen_backbone=True, exposing the
same forward_features(x) -> (B, C) pooled contract (768-d for base @512, 384-d
for small @384). Backbone weights are preserved (requires_grad=False, no_grad
forward, eval mode); only the box/objectness head trains - the textbook
linear-probe transfer recipe applied to coarse single-box localization.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Pooled feature width produced by each baseline backbone's forward_features.
_FEATURE_DIM = {
    "convnextv2": 768,
    "efficientnetv2": 1280,
    "nextvit": 1024,
    "nextvit20": 1024,   # NeXtViT base, same final dim
    "densenet121": 1024,
    "resnet50": 2048,
    "mamba": 256,        # mambapy Vision-Mamba d_model (base)
    "mambavision": 256,
    "dinov3": 384,       # DINOv3 ViT-S/16 pooled feature ("Ours")
    "dinov3_base": 768,  # DINOv3 ViT-B/16 pooled feature ("Ours" detection)
}
_DEFAULT_VARIANT = {
    "convnextv2": "tiny",
    "efficientnetv2": "s",
    "nextvit": "small",
    "nextvit20": "base",
    "mamba": "base",
    "mambavision": "base",
    "dinov3": "small",
    "dinov3_base": "base",
}
# DINOv3 archs that wrap the frozen self-supervised backbone (Ours detection).
_DINOV3_ARCH = {"dinov3": "small", "dinov3_base": "base"}
_TIMM_ARCH = {"densenet121": "densenet121", "resnet50": "resnet50"}


def _build_backbone(
    arch: str,
    img_size: int,
    variant: str,
    *,
    pretrained: bool = True,
    frozen_backbone: bool = True,
) -> nn.Module:
    """Return a feature-extractor backbone exposing forward_features -> (B, C)."""
    if arch in _DINOV3_ARCH:
        # Reuse the frozen DINOv3 classifier as a pooled-feature backbone.
        # DINOv3ForImageClassification.forward_features(x) -> (B, feature_dim)
        # already matches the SingleBoxDetector backbone contract, and keeps the
        # backbone frozen (requires_grad=False + no_grad forward + eval()).
        from src.models.dinov3 import DINOv3ForImageClassification

        wrapper = DINOv3ForImageClassification(
            num_labels=2,  # unused: we only call forward_features
            img_size=img_size,
            variant=_DINOV3_ARCH[arch],
            pretrained=pretrained,
            frozen_backbone=frozen_backbone,
        )
        # The wrapper builds a classification head we never use; drop it so its
        # params do not count as trainable. Replace with Identity so any stray
        # .classifier reference stays valid. Only forward_features is called.
        wrapper.classifier = nn.Identity()
        return wrapper
    if arch in _TIMM_ARCH:
        from src.models.timm_backbone import build_timm_backbone

        return build_timm_backbone(_TIMM_ARCH[arch])
    if arch in ("mamba", "mambavision"):
        from src.models.mamba_vision import build_vision_mamba_backbone

        return build_vision_mamba_backbone(img_size=img_size, model_variant=variant)

    if arch == "convnextv2":
        from baseline.convnextv2 import ConvNeXtV2ForImageClassification as Cls
    elif arch == "efficientnetv2":
        from baseline.efficientnetv2 import EfficientNetV2ForImageClassification as Cls
    else:  # nextvit (small) or nextvit20 (base)
        from baseline.nextvit import NextViTForImageClassification as Cls
    # num_labels is irrelevant: we only keep `.backbone`.
    wrapper = Cls(num_labels=2, img_size=img_size, model_variant=variant)
    return wrapper.backbone


class SingleBoxDetector(nn.Module):
    """Pooled-feature single-box regressor.

    forward(images):
        with_objectness=False -> pred_boxes[B, 4] in [0,1] (xyxy)
        with_objectness=True  -> (pred_boxes[B, 4], obj_logit[B])
    """

    def __init__(
        self,
        backbone: nn.Module,
        feat_dim: int,
        *,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        with_objectness: bool = True,
    ):
        """동결 백본 + 단일박스 회귀(+objectness) 헤드 구성(헤드만 학습 대상)."""
        super().__init__()
        self.backbone = backbone
        self.with_objectness = with_objectness
        self.neck = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.box_head = nn.Linear(hidden_dim, 4)
        self.obj_head = nn.Linear(hidden_dim, 1) if with_objectness else None

    def forward(self, images: torch.Tensor):
        """백본 풀링특징 → 박스[B,4] xyxy∈[0,1] (+ with_objectness면 objectness logit) 반환."""
        feats = self.backbone.forward_features(images)  # (B, C) pooled
        h = self.neck(feats)
        pred_boxes = torch.sigmoid(self.box_head(h))  # (B, 4) in [0,1], xyxy
        if self.with_objectness:
            obj_logit = self.obj_head(h).squeeze(-1)  # (B,)
            return pred_boxes, obj_logit
        return pred_boxes


def build_detector(
    arch: str,
    img_size: int = 512,
    *,
    hidden_dim: int = 256,
    with_objectness: bool = True,
    model_variant: str | None = None,
    pretrained: bool = True,
    frozen_backbone: bool = True,
) -> nn.Module:
    """Build a single-box-regression detector on a baseline backbone.

    Args:
        arch: one of the supported archs (see _FEATURE_DIM), including the
            "Ours" detection archs "dinov3" / "dinov3_base".
        img_size: input resolution (square). Default 512.
        hidden_dim: neck width before the box/objectness heads.
        with_objectness: also predict an objectness logit (BCE in runner).
        model_variant: backbone variant; defaults per-arch if None.
        pretrained: load pretrained weights (only used by dinov3 archs).
        frozen_backbone: freeze the backbone so only the head trains (only
            used by dinov3 archs; from-scratch baselines train fully).

    Returns:
        SingleBoxDetector. forward(images) -> pred_boxes[B,4] in [0,1]
        (or (pred_boxes, obj_logit) when with_objectness=True).
    """
    arch = arch.lower()
    if arch not in _FEATURE_DIM:
        raise ValueError(
            f"unsupported arch {arch!r}; expected one of {sorted(_FEATURE_DIM)}"
        )
    # timm backbones have no size variant; default to "" for them.
    variant = model_variant or _DEFAULT_VARIANT.get(arch, "")
    backbone = _build_backbone(
        arch, img_size, variant,
        pretrained=pretrained, frozen_backbone=frozen_backbone,
    )
    return SingleBoxDetector(
        backbone,
        feat_dim=_FEATURE_DIM[arch],
        hidden_dim=hidden_dim,
        with_objectness=with_objectness,
    )


if __name__ == "__main__":
    x = torch.randn(2, 3, 512, 512)
    for arch in (
        "convnextv2", "efficientnetv2", "nextvit",
        "nextvit20", "densenet121", "resnet50", "mamba",
    ):
        m = build_detector(arch, img_size=512, with_objectness=True)
        m.eval()
        boxes, obj = m(x)
        assert boxes.shape == (2, 4), boxes.shape
        assert obj.shape == (2,), obj.shape
        assert float(boxes.min()) >= 0.0 and float(boxes.max()) <= 1.0
        n_params = sum(p.numel() for p in m.parameters())
        print(
            f"[detector] {arch:14s} boxes={tuple(boxes.shape)} "
            f"obj={tuple(obj.shape)} range=[{float(boxes.min()):.2f},{float(boxes.max()):.2f}] "
            f"params={n_params/1e6:.2f}M"
        )

    # --- "Ours" detection: DINOv3 ViT-B/16 frozen backbone + single-box head ---
    det = build_detector("dinov3_base", img_size=512, with_objectness=True)
    det.train()  # head trainable; DINOv3 backbone forced to eval()
    boxes, obj = det(x)
    assert boxes.shape == (2, 4), boxes.shape
    assert obj.shape == (2,), obj.shape
    assert float(boxes.min()) >= 0.0 and float(boxes.max()) <= 1.0

    n_total = sum(p.numel() for p in det.parameters())
    n_train = sum(p.numel() for p in det.parameters() if p.requires_grad)
    n_backbone = sum(p.numel() for p in det.backbone.parameters())
    backbone_grad = any(p.requires_grad for p in det.backbone.parameters())
    n_head = n_total - n_backbone
    print(
        f"[detector] dinov3_base    boxes={tuple(boxes.shape)} obj={tuple(obj.shape)} "
        f"range=[{float(boxes.min()):.2f},{float(boxes.max()):.2f}]"
    )
    print(
        f"[detector] dinov3_base    total={n_total/1e6:.3f}M "
        f"backbone={n_backbone/1e6:.3f}M head={n_head/1e6:.4f}M "
        f"trainable={n_train/1e6:.4f}M ({100*n_train/n_total:.2f}% of total)"
    )
    print(
        f"[detector] dinov3_base    backbone any requires_grad = {backbone_grad} "
        f"(expect False = frozen); trainable == head : {n_train == n_head}"
    )
    print("detector smoke OK")
