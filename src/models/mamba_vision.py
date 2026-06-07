"""From-scratch Vision Mamba classifier built on `mambapy`.

WHY mambapy (clean Mamba) rather than reviving `baseline/mambavision.py`:
    `baseline/mambavision.py`'s `MambaVisionMixer` depends on
    `mamba_ssm.ops.selective_scan_interface.selective_scan_fn`, which cannot be
    built on this machine (no `mamba_ssm`). Two options were considered:
      (A) monkey-patch the MambaVision mixer to use a `mambapy.pscan`-based
          selective scan, keeping the MambaVision (windowed conv+mamba) arch;
      (B) build a clean ViT-style Mamba classifier with `mambapy.Mamba`.
    (A) is brittle: the MambaVision mixer splits the inner channels, uses a
    custom dt/B/C projection layout, applies a separate conv on the z gate, and
    concatenates the gate post-scan -- reproducing that exactly on top of
    pscan's (B, L, ED, N) convention is error-prone and unverifiable without the
    original CUDA kernel as a reference. (B) is from-scratch, fully verifiable,
    numerically stable (pscan parallel scan), and keeps the same forward
    interface as the other baselines. We chose (B).

Architecture (from-scratch, no pretrained):
    images (B,3,H,W)
      -> Conv2d patch-embed (stride=patch) -> tokens (B, N, d_model)
      -> + learnable positional embedding (interpolated if N differs)
      -> mambapy.Mamba(MambaConfig(d_model, n_layers, use_cuda=False))   # pscan scan
      -> final RMSNorm -> mean-pool over tokens -> (B, d_model)
      -> classifier head (LayerNorm -> Linear -> GELU -> Dropout -> Linear)

`use_cuda=False` forces mambapy's pure-PyTorch pscan selective scan (the CUDA
fast path needs mamba_ssm). The backbone exposes `forward_features(images) ->
(B, d_model)` so the same detector head can reuse it, exactly like the timm /
baseline backbones.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mambapy.mamba import Mamba, MambaConfig


# Default variant. ~patch16 @224 -> 14x14=196 tokens. d_model=256, 8 layers.
_VARIANTS = {
    "base": {"d_model": 256, "n_layers": 8, "patch_size": 16},
    "small": {"d_model": 192, "n_layers": 6, "patch_size": 16},
}


class _VisionMambaBackbone(nn.Module):
    """Conv patch-embed -> Mamba blocks -> pooled feature (B, d_model)."""

    def __init__(self, d_model: int, n_layers: int, patch_size: int, img_size: int):
        """Vision-Mamba(mambapy) 구성: patch-embed → mamba 블록 → 헤드."""
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        n_tokens = self.grid * self.grid

        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        cfg = MambaConfig(d_model=d_model, n_layers=n_layers, use_cuda=False)
        self.mamba = Mamba(cfg)
        self.norm = nn.LayerNorm(d_model)

    def _interp_pos(self, n_tokens: int) -> torch.Tensor:
        """입력 토큰 수에 맞춰 위치 임베딩을 보간."""
        if n_tokens == self.pos_embed.shape[1]:
            return self.pos_embed
        # Bilinear-interpolate the positional grid for a different resolution.
        g0 = int(self.pos_embed.shape[1] ** 0.5)
        g1 = int(n_tokens ** 0.5)
        pe = self.pos_embed.reshape(1, g0, g0, self.d_model).permute(0, 3, 1, 2)
        pe = F.interpolate(pe, size=(g1, g1), mode="bilinear", align_corners=False)
        return pe.permute(0, 2, 3, 1).reshape(1, g1 * g1, self.d_model)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        """patch 토큰 → mamba 인코더 → 평균풀링한 (B,C) 특징."""
        x = self.patch_embed(images)              # (B, d_model, g, g)
        x = x.flatten(2).transpose(1, 2)          # (B, N, d_model)
        x = x + self._interp_pos(x.shape[1])
        x = self.mamba(x)                         # (B, N, d_model)
        x = self.norm(x)
        return x.mean(dim=1)                      # (B, d_model)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """특징 → 헤드 → logits(labels 주면 (loss, logits))."""
        return self.forward_features(images)


class VisionMambaForImageClassification(nn.Module):
    """Vision-Mamba classifier with the project-standard forward signature.

    forward(images, labels=None) -> logits[B, num_labels]  (or (loss, logits)).
    """

    def __init__(self, num_labels=2, img_size=224, hidden_dim=512, model_variant="base"):
        """Vision-Mamba(mambapy) 구성: patch-embed → mamba 블록 → 헤드."""
        super().__init__()
        cfg = _VARIANTS.get(model_variant, _VARIANTS["base"])
        self.backbone = _VisionMambaBackbone(
            d_model=cfg["d_model"],
            n_layers=cfg["n_layers"],
            patch_size=cfg["patch_size"],
            img_size=img_size,
        )
        num_features = cfg["d_model"]
        self.classifier = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Linear(num_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, images, labels=None):
        """특징 → 헤드 → logits(labels 주면 (loss, logits))."""
        feats = self.backbone.forward_features(images)
        logits = self.classifier(feats)
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits
        return logits


def build_vision_mamba_backbone(img_size: int, model_variant: str = "base") -> nn.Module:
    """Return the bare backbone (exposes forward_features -> (B, d_model))."""
    cfg = _VARIANTS.get(model_variant, _VARIANTS["base"])
    return _VisionMambaBackbone(
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        patch_size=cfg["patch_size"],
        img_size=img_size,
    )


VISION_MAMBA_FEATURE_DIM = {k: v["d_model"] for k, v in _VARIANTS.items()}
