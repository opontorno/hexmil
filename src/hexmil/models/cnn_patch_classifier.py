"""
cnn_patch_classifier.py
-----------------------
Phase A — CNN backbone patch classifier.

Takes a single-channel 2D patch (1, H, W) centred on the nodule and classifies it
as real (0) or fake (1).  The backbone feature map is also used to produce a
**spatial attention map** that highlights which spatial regions are most relevant
to the classification — this serves as the CNN-based XAI signal.

Architecture:
    ┌──────────────────────────────┐
    │  Backbone (e.g. ResNet-50)   │  → feature map  F ∈ (B, C, h, w)
    └──────────────────────────────┘
                │
    ┌───────────▼──────────────────┐
    │  Spatial Attention Module     │  → attention map  α ∈ (B, 1, h, w)
    │  (1×1 conv → sigmoid)        │     (upsampled to input resolution)
    └──────────────────────────────┘
                │
    ┌───────────▼──────────────────┐
    │  Weighted GAP                 │  → attended feature  v ∈ (B, C)
    │  (F ⊙ α).mean(spatial)       │
    └──────────────────────────────┘
                │
    ┌───────────▼──────────────────┐
    │  Classification Head          │  → logit ∈ (B, 1)
    │  (FC → ReLU → FC)            │
    └──────────────────────────────┘

Supported backbones (via timm):
    resnet50, resnet34, efficientnet_b4, vgg16_bn, densenet121, convnext_tiny
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class SpatialAttention(nn.Module):
    """Learnable 1×1 conv that squashes the channel dimension to a single
    spatial attention map, normalised by sigmoid."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: (B, C, h, w) backbone feature map
        Returns:
            attn: (B, 1, h, w)  values in [0, 1]
        """
        return torch.sigmoid(self.conv(feat))


class CNNPatchClassifier(nn.Module):
    """
    CNN backbone + spatial attention + binary classifier.

    Args:
        backbone_name:  any timm model name (features_only must be supported).
        pretrained:     use ImageNet pre-trained weights.
        freeze_ratio:   fraction of backbone layers to freeze (0.0 = none, 0.5 = first half).
        in_chans:       number of input channels (1 for grayscale CT).
    """

    def __init__(
        self,
        backbone_name: str = 'resnet50',
        pretrained: bool = True,
        freeze_ratio: float = 0.5,
        in_chans: int = 1,
    ):
        super().__init__()

        # ── Backbone (take last feature map) ─────────────────────────────
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            features_only=True,
        )
        # The last feature map has the richest semantics
        feat_info  = self.backbone.feature_info
        self.feat_dim = feat_info[-1]['num_chs']

        # ── Freeze first `freeze_ratio` of backbone parameters ───────────
        if freeze_ratio > 0:
            all_params = list(self.backbone.parameters())
            n_freeze   = int(len(all_params) * freeze_ratio)
            for p in all_params[:n_freeze]:
                p.requires_grad = False

        # ── Spatial attention ────────────────────────────────────────────
        self.spatial_attn = SpatialAttention(self.feat_dim)

        # ── Classification head ──────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(self.feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        """
        Args:
            x:           (B, 1, H, W) input patch
            return_attn: if True, also return the spatial attention map

        Returns:
            logits: (B, 1) raw classification logits
            attn:   (B, 1, H, W) attention map upsampled to input size (if requested)
        """
        # Backbone features — take last stage
        feats = self.backbone(x)[-1]                       # (B, C, h, w)

        # Spatial attention
        attn = self.spatial_attn(feats)                    # (B, 1, h, w)

        # Weighted global average pooling
        weighted = feats * attn                            # (B, C, h, w)
        v = weighted.mean(dim=[2, 3])                      # (B, C)

        # Classify
        logits = self.head(v)                              # (B, 1)

        if return_attn:
            attn_up = F.interpolate(attn, size=x.shape[2:],
                                    mode='bilinear', align_corners=False)
            return logits, attn_up
        return logits


def build_cnn_classifier(backbone: str = 'resnet50', pretrained: bool = True,
                          freeze_ratio: float = 0.5) -> CNNPatchClassifier:
    """Factory function."""
    return CNNPatchClassifier(
        backbone_name=backbone,
        pretrained=pretrained,
        freeze_ratio=freeze_ratio,
        in_chans=1,
    )
