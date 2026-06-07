"""DINOv3 ViT transfer classifier ("Ours" / "Ours+").

Self-supervised pretrained backbone (frozen) + 2-layer MLP head.

Two variants are supported through a single implementation:
    - "small": DINOv3 ViT-S/16 @256, pooled feature 384-d (the original Ours).
    - "base" : DINOv3 ViT-B/16 @512, pooled feature 768-d (Ours+, scaled up
               for maximum performance while keeping the frozen-backbone /
               no-forgetting recipe).

Rationale (see _workspace/specs/design_notes.md "Ours" / "Ours+"):
    The from-scratch baselines (convnextv2/efficientnetv2/nextvit/densenet121/
    resnet50/mamba) have a low absolute ceiling on this small, imbalanced,
    single-season dataset. A DINOv3 ViT backbone brings strong self-supervised
    ImageNet features. We *freeze* the backbone (requires_grad=False + no_grad
    forward) so its pretrained knowledge is preserved (no catastrophic
    forgetting) and train only the MLP head - the textbook linear/MLP-probe
    transfer recipe. Ours+ scales the backbone (S->B), the input (256->512) and
    the head width to push representation power on the hardest 3-class setting.

Backbones (verified):
    timm.create_model("vit_small_patch16_dinov3", pretrained=True,
                       num_classes=0, img_size=256)
        -> pooled feature dim 384, backbone ~21.6M params.
    timm.create_model("vit_base_patch16_dinov3", pretrained=True,
                       num_classes=0, img_size=512)
        -> pooled feature dim 768, backbone ~85.6M params.
    pretrained_cfg: ImageNet mean/std (the data loader already supports
    img_size + ImageNet norm, so no data changes are needed).

Head:
    feat -> Linear(feat, hidden) -> GELU -> Dropout -> Linear(hidden, C)
    (hidden default 256 for small, 512 for base.)

forward(images, labels=None) -> logits[B, C]   (or (loss, logits) with labels)
matches the baseline classifier wrapper signature (internal CrossEntropyLoss).
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

# DINOv3 variant registry: timm model name + pooled feature width + default
# input resolution per pretrained_cfg.
DINOV3_VARIANTS = {
    "small": {
        "timm_name": "vit_small_patch16_dinov3",
        "feature_dim": 384,
        "default_img_size": 256,
        "default_hidden_dim": 256,
    },
    "base": {
        "timm_name": "vit_base_patch16_dinov3",
        "feature_dim": 768,
        "default_img_size": 512,
        "default_hidden_dim": 512,
    },
}

# Back-compat aliases (the original module exposed these for the small variant).
DINOV3_TIMM_NAME = DINOV3_VARIANTS["small"]["timm_name"]
DINOV3_FEATURE_DIM = DINOV3_VARIANTS["small"]["feature_dim"]


class DINOv3ForImageClassification(nn.Module):
    """Frozen DINOv3 ViT backbone + 2-layer MLP classification head.

    The backbone is loaded with self-supervised ImageNet pretrained weights and
    fully frozen (requires_grad=False). Its forward pass runs under
    torch.no_grad() and the backbone is kept in eval() mode so no statistics or
    weights ever change - only the head is trainable.

    Args:
        num_labels: number of output classes.
        img_size: input resolution (square). Defaults to the variant default.
        variant: "small" (ViT-S/16, 384-d) or "base" (ViT-B/16, 768-d).
        hidden_dim: MLP head width. Defaults to the variant default.
        dropout / pretrained / frozen_backbone: as named.
    """

    def __init__(
        self,
        num_labels: int = 2,
        img_size: int | None = None,
        *,
        variant: str = "small",
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        pretrained: bool = True,
        frozen_backbone: bool = True,
    ):
        """DINOv3 백본(pretrained, frozen) + 2-layer MLP 헤드 구성(헤드만 trainable)."""
        super().__init__()
        variant = variant.lower()
        if variant not in DINOV3_VARIANTS:
            raise ValueError(
                f"unsupported DINOv3 variant {variant!r}; "
                f"expected one of {sorted(DINOV3_VARIANTS)}"
            )
        cfg = DINOV3_VARIANTS[variant]
        self.variant = variant
        self.frozen_backbone = frozen_backbone

        if img_size is None:
            img_size = cfg["default_img_size"]
        if hidden_dim is None:
            hidden_dim = cfg["default_hidden_dim"]
        self.img_size = img_size

        self.backbone = timm.create_model(
            cfg["timm_name"],
            pretrained=pretrained,
            num_classes=0,  # pooled feature mode -> (B, feature_dim)
            img_size=img_size,
        )
        num_features = cfg["feature_dim"]

        if frozen_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()  # freeze norm/stats (ViT has no BN, but be explicit)

        self.classifier = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def train(self, mode: bool = True):
        """Keep a frozen backbone in eval() even when the module is trained.

        Only the head should switch between train/eval; the backbone must stay
        in eval so its (any) running stats and dropout never activate.
        """
        super().train(mode)
        if self.frozen_backbone:
            self.backbone.eval()
        return self

    def forward_features(self, images):
        """Pooled DINOv3 feature (B, feature_dim). Runs under no_grad if frozen."""
        if self.frozen_backbone:
            with torch.no_grad():
                return self.backbone(images)
        return self.backbone(images)

    def forward(self, images, labels=None):
        """frozen 백본 풀링특징 → head → logits(labels 주면 (loss, logits))."""
        feats = self.forward_features(images)
        logits = self.classifier(feats)
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits
        return logits

    def trainable_parameters(self):
        """Iterator over parameters that require grad (head only when frozen).

        experiment-runner can feed this to the optimizer. The existing runner
        uses model.parameters(); that is also correct because frozen params
        have requires_grad=False and therefore receive no gradient/update.
        """
        return (p for p in self.parameters() if p.requires_grad)


def build_dinov3_classifier(
    num_classes: int,
    img_size: int | None = None,
    *,
    variant: str = "small",
    hidden_dim: int | None = None,
    pretrained: bool = True,
    frozen_backbone: bool = True,
) -> nn.Module:
    """arch/variant에 맞는 DINOv3 frozen+2-layer head 분류기 생성 헬퍼."""
    return DINOv3ForImageClassification(
        num_labels=num_classes,
        img_size=img_size,
        variant=variant,
        hidden_dim=hidden_dim,
        pretrained=pretrained,
        frozen_backbone=frozen_backbone,
    )


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Ours+ : DINOv3 ViT-B/16 @512 (the new scaled-up variant) ---
    img_size = 512
    model = build_dinov3_classifier(
        num_classes=3, img_size=img_size, variant="base", hidden_dim=512
    ).to(dev)
    model.train()  # head trainable, backbone forced eval

    x = torch.randn(2, 3, img_size, img_size, device=dev)
    y = torch.randint(0, 3, (2,), device=dev)

    feats = model.forward_features(x)
    logits = model(x)
    loss, logits2 = model(x, y)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_backbone = sum(p.numel() for p in model.backbone.parameters())
    n_head = sum(p.numel() for p in model.classifier.parameters())
    backbone_grad = any(p.requires_grad for p in model.backbone.parameters())

    print(f"[dinov3-base] device={dev} variant={model.variant} "
          f"img_size={model.img_size} feat={tuple(feats.shape)} "
          f"logits={tuple(logits.shape)} loss={float(loss):.4f}")
    print(f"[dinov3-base] total={n_total/1e6:.3f}M backbone={n_backbone/1e6:.3f}M "
          f"head={n_head/1e6:.4f}M trainable={n_train/1e6:.4f}M "
          f"({100*n_train/n_total:.2f}% of total)")
    print(f"[dinov3-base] backbone any requires_grad = {backbone_grad} "
          f"(expect False = frozen)")
    print(f"[dinov3-base] trainable == head : {n_train == n_head}")

    # --- regression check: original small variant still builds (CPU is fine) ---
    small = build_dinov3_classifier(num_classes=3, variant="small")
    s_feat = small.forward_features(torch.randn(1, 3, 256, 256))
    s_logits = small(torch.randn(1, 3, 256, 256))
    s_head = sum(p.numel() for p in small.classifier.parameters())
    print(f"[dinov3-small] variant={small.variant} img_size={small.img_size} "
          f"feat={tuple(s_feat.shape)} logits={tuple(s_logits.shape)} "
          f"head={s_head/1e6:.4f}M (regression OK)")
    print("dinov3 smoke OK")
