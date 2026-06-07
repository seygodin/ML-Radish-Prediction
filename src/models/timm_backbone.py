"""timm-backbone image classifier wrapper (from-scratch, no pretrained).

Wraps a timm backbone in pooled-feature mode (`num_classes=0`) and applies the
same classifier head pattern the `baseline/` wrappers use:
    LayerNorm -> Linear(feat, hidden) -> GELU -> Dropout -> Linear(hidden, num_labels)

Exposes `forward_features(images) -> (B, feat_dim)` on the backbone so the
single-box detector can reuse the same pooled features as the other backbones.

forward(images, labels=None) -> logits[B, num_labels]  (or (loss, logits)).
"""
from __future__ import annotations

import timm
import torch.nn as nn


# Pooled feature width for each supported timm backbone (num_classes=0).
TIMM_FEATURE_DIM = {
    "densenet121": 1024,
    "resnet50": 2048,
}


class TimmBackbone(nn.Module):
    """timm model in pooled mode; `forward_features` returns (B, C)."""

    def __init__(self, timm_name: str):
        """timm 백본 + 분류 헤드 구성."""
        super().__init__()
        self.model = timm.create_model(timm_name, pretrained=False, num_classes=0)

    def forward_features(self, images):
        """입력을 백본에 통과시켜 풀링된 (B,C) 특징 반환."""
        return self.model(images)  # (B, C) pooled (num_classes=0)

    def forward(self, images):
        """특징 추출 → 헤드 → logits(labels 주면 (loss, logits))."""
        return self.model(images)


class TimmForImageClassification(nn.Module):
    """timm 백본(num_classes=0 풀링특징) + 표준 분류 헤드 래퍼(baseline 인터페이스)."""
    def __init__(self, timm_name: str, num_labels=2, hidden_dim=512):
        """timm 백본 + 분류 헤드 구성."""
        super().__init__()
        if timm_name not in TIMM_FEATURE_DIM:
            raise ValueError(f"unsupported timm backbone {timm_name!r}")
        self.backbone = TimmBackbone(timm_name)
        num_features = TIMM_FEATURE_DIM[timm_name]
        self.classifier = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Linear(num_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, images, labels=None):
        """특징 추출 → 헤드 → logits(labels 주면 (loss, logits))."""
        feats = self.backbone.forward_features(images)
        logits = self.classifier(feats)
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits
        return logits


def build_timm_backbone(timm_name: str) -> nn.Module:
    """timm 모델명을 받아 풀링특징 백본을 생성(detector가 forward_features로 사용)."""
    return TimmBackbone(timm_name)
