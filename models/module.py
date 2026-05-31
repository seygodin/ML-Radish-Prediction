"""Shared custom modules for the radish disease models.

SAFEModule is referenced by the `*ForImageClassification_v2` wrappers in `baseline/`
(the SAFE-enhanced variants intended for the "Ours" model). The plain
`*ForImageClassification` **baseline** wrappers do NOT use it — but importing any
`baseline/*.py` file executes `from models.module import SAFEModule`, so this module
must exist for baselines to import at all.

For the BASELINE measurement phase this is a no-op passthrough so it has zero effect
on baseline numbers. Replace this with the real SAFE design during the "Ours" phase.
"""
import torch.nn as nn


class SAFEModule(nn.Module):
    """Placeholder feature module (identity passthrough).

    Accepts the backbone feature width and returns its input unchanged, so it works
    for both pooled (B, C) and spatial (B, C, H, W) features. The real "Ours"
    contribution should implement the actual SAFE mechanism here.
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features

    def forward(self, x):
        return x
