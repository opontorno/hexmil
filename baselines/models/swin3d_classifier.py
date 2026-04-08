"""
swin3d_classifier.py
====================
Swin Transformer 3D (Swin3D-T) baseline for CT volume binary classification.
Requires torchvision >= 0.14.

The model takes (1, K, H, W) single-channel volumes.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class Swin3DClassifier(nn.Module):
    """
    Swin3D-T adapted for single-channel CT volumes.

    Args:
        pretrained: load torchvision Kinetics-400 weights
        dropout:    dropout in classifier head
    """
    def __init__(self, pretrained: bool = False, dropout: float = 0.3):
        super().__init__()
        import torchvision.models.video as vm

        base = vm.swin3d_t(weights='DEFAULT' if pretrained else None)

        # Patch embed: conv3d maps (3, T, H, W) → (96, T', H', W')
        # Adapt to 1 input channel
        old_pe = base.patch_embed.proj
        base.patch_embed.proj = nn.Conv3d(
            1, old_pe.out_channels,
            kernel_size=old_pe.kernel_size,
            stride=old_pe.stride,
            padding=old_pe.padding,
        )
        if pretrained:
            with torch.no_grad():
                base.patch_embed.proj.weight.copy_(
                    old_pe.weight.mean(dim=1, keepdim=True)
                )

        feat_dim = base.head.in_features  # 768
        base.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 1),
        )
        self.model = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, K, H, W)
        Returns:
            logit: (B, 1)
        """
        return self.model(x)


def build_swin3d_classifier(pretrained: bool = False,
                             dropout: float = 0.3) -> Swin3DClassifier:
    return Swin3DClassifier(pretrained=pretrained, dropout=dropout)
