"""
abmil_slice_classifier.py
--------------------------
SliceMIL — Stage 1 CT slice classifier via Attention-Based MIL (Ilse et al., 2018).

Architecture:
    Patch encoder (backbone + spatial attention → GAP)  → f_k ∈ R^D  per patch
    Linear projector  D → proj_dim (M)                  → h_k ∈ R^M
    Gated ABMIL:  a_k = softmax(w^T (tanh(Vh_k) ⊙ σ(Uh_k)))
                  z   = Σ_k a_k h_k                     → bag repr.
    Classification head  M → 256 → 1                    → logit

Attention weights a_k map back to the slice grid to produce a localisation
heatmap without extra supervision.

Factory: build_abmil_classifier_scratch(backbone, ...) → ABMILSliceClassifier
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hexmil.models.cnn_patch_classifier import CNNPatchClassifier


# ---------------------------------------------------------------------------
#  Gated Attention module
# ---------------------------------------------------------------------------

class GatedAttention(nn.Module):
    """
    Gated Attention-Based MIL pooling (Ilse et al. 2018).

        A = softmax( w^T ( tanh(V H^T) ⊙ σ(U H^T) ) )
        z = A H

    Args:
        feat_dim:  dimensionality of input instance features (M).
        attn_dim:  internal attention dimension (L).
        dropout:   dropout probability applied before attention computation.
    """

    def __init__(self, feat_dim: int, attn_dim: int = 128, dropout: float = 0.25):
        super().__init__()
        self.V       = nn.Linear(feat_dim, attn_dim)
        self.U       = nn.Linear(feat_dim, attn_dim)
        self.w       = nn.Linear(attn_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (B, N, M)  bag of instance feature vectors.
        Returns:
            z:    (B, M)  aggregated bag representation.
            attn: (B, N)  normalised attention weights (sums to 1 over N).
        """
        h  = self.dropout(h)
        A  = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h)))  # (B, N, 1)
        A  = torch.softmax(A.squeeze(-1), dim=1)                        # (B, N)
        z  = (A.unsqueeze(-1) * h).sum(dim=1)                          # (B, M)
        return z, A


# ---------------------------------------------------------------------------
#  Main model
# ---------------------------------------------------------------------------

class ABMILSliceClassifier(nn.Module):
    """
    Attention-Based MIL classifier for full CT slices.

    The entire model (encoder, projector, ABMIL aggregator, classification
    head) is trained end-to-end.  The encoder is a CNNPatchClassifier whose
    backbone can be initialised from ImageNet weights.

    Args:
        encoder:   CNNPatchClassifier instance.
        proj_dim:  dimension to project each patch feature to before ABMIL.
                   Set to None to skip the projection (keep raw feat_dim).
        attn_dim:  internal ABMIL attention dimension.
        dropout:   dropout used inside ABMIL and the classification head.
    """

    def __init__(
        self,
        encoder: CNNPatchClassifier,
        proj_dim: int = 512,
        attn_dim: int = 128,
        dropout: float = 0.25,
    ):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────────
        self.encoder = encoder

        raw_feat_dim: int = encoder.feat_dim   # e.g. 2048 for ResNet-50

        # ── Feature projector ────────────────────────────────────────────
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

        # ── Gated ABMIL aggregation ──────────────────────────────────────
        self.aggregator = GatedAttention(self._feat_dim, attn_dim, dropout)

        # ── Classification head ──────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(self._feat_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    # ------------------------------------------------------------------
    @property
    def feat_dim(self) -> int:
        return self._feat_dim

    # ------------------------------------------------------------------
    def encode_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Extract a feature vector for each patch using the encoder.

        Args:
            patches: (B, N, 1, P, P)
        Returns:
            features: (B, N, raw_feat_dim)
        """
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

    # ------------------------------------------------------------------
    def forward(
        self,
        patches: torch.Tensor,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            patches:     (B, N, 1, P, P)
            return_attn: if True, also return attention weights (B, N).

        Returns:
            logits: (B, 1)
            attn:   (B, N)  [only if return_attn=True]
        """
        # 1. Encode
        features = self.encode_patches(patches)              # (B, N, raw_feat_dim)

        # 2. Project
        B, N, _ = features.shape
        h = self.projector(features.view(B * N, -1)).view(B, N, self._feat_dim)

        # 3. ABMIL aggregation
        z, attn_weights = self.aggregator(h)                  # z:(B,M), attn:(B,N)

        # 4. Classify
        logits = self.head(z)                                  # (B, 1)

        if return_attn:
            return logits, attn_weights
        return logits


# ---------------------------------------------------------------------------
#  Factory helper
# ---------------------------------------------------------------------------


def build_abmil_classifier_scratch(
    backbone:        str   = 'resnet50',
    pretrained:      bool  = True,
    proj_dim:        int   = 512,
    attn_dim:        int   = 128,
    dropout:         float = 0.25,
    patch_size:      int   = 64,
) -> ABMILSliceClassifier:
    """
    Build an ABMILSliceClassifier with a freshly initialised encoder.
    The entire model (encoder + projector + ABMIL + head) is trained
    end-to-end.

    Args:
        backbone:   CNN name (resnet50, densenet121, …), 'vit_base', or 'clip'.
        pretrained: ImageNet init for CNN backbones; ignored for ViT/CLIP
                    (pretrained weights always loaded from their respective sources).
        proj_dim:   patch feature projection dimension.
        attn_dim:   gated ABMIL internal attention dimension.
        dropout:    dropout probability.
        patch_size: spatial size of each patch (P×P). Used by ViT/CLIP encoders
                    for positional-embedding interpolation; CNNs ignore it.

    Returns:
        ABMILSliceClassifier ready for end-to-end training.
    """
    if backbone == 'clip':
        from hexmil.models.clip_patch_encoder import build_clip_encoder

        # OpenAI CLIP ViT-B/16 visual backbone. Contrastive projection head
        # removed; raw 768-dim CLS token used as patch feature.
        # Patches are resized to 224×224 and CLIP-normalised inside the encoder.
        encoder = build_clip_encoder()

    elif backbone.startswith('vit'):
        from hexmil.models.vit_patch_classifier import build_pretrained_vit_encoder

        # ViT-Base/16 pretrained on ImageNet-21k/1k via timm.
        # patch embed adapted to in_chans=1 (weights averaged 3→1);
        # positional embeddings interpolated to patch_size/16 token grid.
        encoder = build_pretrained_vit_encoder(
            patch_size=patch_size,
            freeze_ratio=0.0,
        )
    else:
        from hexmil.models.cnn_patch_classifier import build_cnn_classifier

        encoder = build_cnn_classifier(
            backbone=backbone,
            pretrained=pretrained,
            freeze_ratio=0.0,
        )
    return ABMILSliceClassifier(
        encoder=encoder,
        proj_dim=proj_dim,
        attn_dim=attn_dim,
        dropout=dropout,
    )
