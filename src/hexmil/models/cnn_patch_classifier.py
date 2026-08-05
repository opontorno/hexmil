from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class SpatialAttention(nn.Module):

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.conv(feat))


class CNNPatchClassifier(nn.Module):

    def __init__(
        self,
        backbone_name: str = 'resnet50',
        pretrained: bool = True,
        freeze_ratio: float = 0.5,
        in_chans: int = 1,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            features_only=True,
        )
        # The last feature map has the richest semantics
        feat_info  = self.backbone.feature_info
        self.feat_dim = feat_info[-1]['num_chs']

        if freeze_ratio > 0:
            all_params = list(self.backbone.parameters())
            n_freeze   = int(len(all_params) * freeze_ratio)
            for p in all_params[:n_freeze]:
                p.requires_grad = False

        self.spatial_attn = SpatialAttention(self.feat_dim)

        self.head = nn.Sequential(
            nn.Linear(self.feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        # Backbone features — take last stage
        feats = self.backbone(x)[-1]                       # (B, C, h, w)

        attn = self.spatial_attn(feats)                    # (B, 1, h, w)

        # Weighted global average pooling
        weighted = feats * attn                            # (B, C, h, w)
        v = weighted.mean(dim=[2, 3])                      # (B, C)

        logits = self.head(v)                              # (B, 1)

        if return_attn:
            attn_up = F.interpolate(attn, size=x.shape[2:],
                                    mode='bilinear', align_corners=False)
            return logits, attn_up
        return logits


def build_cnn_classifier(backbone: str = 'resnet50', pretrained: bool = True,
                          freeze_ratio: float = 0.5) -> CNNPatchClassifier:
    return CNNPatchClassifier(
        backbone_name=backbone,
        pretrained=pretrained,
        freeze_ratio=freeze_ratio,
        in_chans=1,
    )
