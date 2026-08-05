from __future__ import annotations

import torch
import torch.nn as nn

from hexmil.models.cnn_patch_classifier import CNNPatchClassifier


class GatedAttention(nn.Module):

    def __init__(self, feat_dim: int, attn_dim: int = 128, dropout: float = 0.25):
        super().__init__()
        self.V       = nn.Linear(feat_dim, attn_dim)
        self.U       = nn.Linear(feat_dim, attn_dim)
        self.w       = nn.Linear(attn_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h  = self.dropout(h)
        A  = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h)))  # (B, N, 1)
        A  = torch.softmax(A.squeeze(-1), dim=1)                        # (B, N)
        z  = (A.unsqueeze(-1) * h).sum(dim=1)                          # (B, M)
        return z, A


class SliceMIL(nn.Module):

    def __init__(
        self,
        encoder: CNNPatchClassifier,
        proj_dim: int = 512,
        attn_dim: int = 128,
        dropout: float = 0.25,
    ):
        super().__init__()

        self.encoder = encoder

        raw_feat_dim: int = encoder.feat_dim   # e.g. 2048 for ResNet-50

        if proj_dim is not None and proj_dim != raw_feat_dim:
            self.projector = nn.Sequential(
                nn.Linear(raw_feat_dim, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self._feat_dim = proj_dim
        else:
            self.projector = nn.Identity()
            self._feat_dim = raw_feat_dim

        self.aggregator = GatedAttention(self._feat_dim, attn_dim, dropout)

        self.head = nn.Sequential(
            nn.Linear(self._feat_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    @property
    def feat_dim(self) -> int:
        return self._feat_dim

    def encode_patches(self, patches: torch.Tensor) -> torch.Tensor:
        B, N, C, P_h, P_w = patches.shape
        flat = patches.view(B * N, C, P_h, P_w)

        if hasattr(self.encoder, 'encode_patch'):
            # ViT path: CLS token as patch feature
            v = self.encoder.encode_patch(flat)            # (B*N, embed_dim)
        else:
            # CNN path: backbone + spatial attention + GAP
            feat_map = self.encoder.backbone(flat)[-1]     # (B*N, C_feat, h, w)
            attn_map = self.encoder.spatial_attn(feat_map) # (B*N, 1, h, w)
            v        = (feat_map * attn_map).mean(dim=[2, 3])  # (B*N, C_feat)

        return v.view(B, N, -1)                            # (B, N, raw_feat_dim)

    def forward(
        self,
        patches: torch.Tensor,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.encode_patches(patches)              # (B, N, raw_feat_dim)

        B, N, _ = features.shape
        h = self.projector(features.view(B * N, -1)).view(B, N, self._feat_dim)

        z, attn_weights = self.aggregator(h)                  # z:(B,M), attn:(B,N)

        logits = self.head(z)                                  # (B, 1)

        if return_attn:
            return logits, attn_weights
        return logits


def build_slicemil(
    backbone:        str   = 'resnet50',
    pretrained:      bool  = True,
    proj_dim:        int   = 512,
    attn_dim:        int   = 128,
    dropout:         float = 0.25,
    patch_size:      int   = 64,
) -> SliceMIL:
    from hexmil.models.cnn_patch_classifier import build_cnn_classifier

    encoder = build_cnn_classifier(
        backbone=backbone,
        pretrained=pretrained,
        freeze_ratio=0.0,
    )
    return SliceMIL(
        encoder=encoder,
        proj_dim=proj_dim,
        attn_dim=attn_dim,
        dropout=dropout,
    )
