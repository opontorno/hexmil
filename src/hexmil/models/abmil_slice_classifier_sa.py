"""
abmil_slice_classifier_sa.py
-----------------------------
Phase B — ABMIL Slice Classifier with Patch Self-Attention (SA-ABMIL).

A Transformer encoder is inserted between the CNN feature extractor and the
Gated ABMIL aggregator.  This lets each patch attend to every other patch in
the same slice before the pooling step, capturing spatial inconsistency
signals that are invisible when patches are treated independently.

Architecture:
    ┌──────────────────────────────────────────────────────────────┐
    │  CNN Encoder (ResNet/EfficientNet + spatial_attn + GAP)      │
    │  → f_k ∈ R^D                                                 │
    └──────────────────────────────────────────────────────────────┘
                          │  N vectors (D-dim)
    ┌─────────────────────▼────────────────────────────────────────┐
    │  Linear Projector  D → proj_dim (M)                          │
    └──────────────────────────────────────────────────────────────┘
                          │
    ┌─────────────────────▼────────────────────────────────────────┐
    │  PatchTransformer  (Pre-LN, sa_n_layers × sa_n_heads)        │
    │  h_k' = TransformerEncoderLayer(h_1…h_N)[k]                 │
    │  Each patch attends to ALL other patches in the same slice.  │
    │  → (N, M)  contextualised patch representations              │
    └──────────────────────────────────────────────────────────────┘
                          │
    ┌─────────────────────▼────────────────────────────────────────┐
    │  Gated ABMIL Attention                                       │
    │  a_k = softmax(w^T (tanh(Vh_k')⊙σ(Uh_k')))                 │
    │  z   = Σ_k a_k h_k'                                         │
    └──────────────────────────────────────────────────────────────┘
                          │
    ┌─────────────────────▼────────────────────────────────────────┐
    │  Classification Head  M → 256 → 1                            │
    └──────────────────────────────────────────────────────────────┘

The ABMIL attention weights a_k can still be reshaped to (n_rows, n_cols)
and upsampled to slice resolution for localisation heatmaps.

Key hyper-parameters vs vanilla ABMIL:
    sa_n_heads  (int, default 8)  — number of attention heads in Transformer.
                                    proj_dim must be divisible by sa_n_heads.
    sa_n_layers (int, default 2)  — depth of the Transformer encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hexmil.models.cnn_patch_classifier import CNNPatchClassifier
from hexmil.models.abmil_slice_classifier import GatedAttention


# ---------------------------------------------------------------------------
#  Patch Transformer
# ---------------------------------------------------------------------------

class PatchTransformer(nn.Module):
    """
    Lightweight Transformer encoder over a bag of N patch embeddings.

    Uses Pre-LN (norm_first=True) for stable training without warmup.
    batch_first=True so tensors are (B, N, D) throughout.

    Args:
        feat_dim:   embedding dimension (must match proj_dim).
        n_heads:    number of self-attention heads. feat_dim % n_heads == 0.
        n_layers:   number of Transformer encoder layers.
        dropout:    dropout applied inside each encoder layer.
        ff_mult:    feedforward hidden dim = feat_dim * ff_mult.
    """

    def __init__(
        self,
        feat_dim: int,
        n_heads:  int   = 8,
        n_layers: int   = 2,
        dropout:  float = 0.1,
        ff_mult:  int   = 4,
    ):
        super().__init__()
        assert feat_dim % n_heads == 0, (
            f"feat_dim ({feat_dim}) must be divisible by n_heads ({n_heads})"
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = feat_dim,
            nhead           = n_heads,
            dim_feedforward = feat_dim * ff_mult,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,   # Pre-LN: more stable
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers = n_layers,
            enable_nested_tensor = False,
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h:  (B, N, D)  bag of patch embeddings.
        Returns:
            h': (B, N, D)  contextualised patch embeddings.
        """
        return self.transformer(h)


# ---------------------------------------------------------------------------
#  Main model
# ---------------------------------------------------------------------------

class SABMILSliceClassifier(nn.Module):
    """
    Self-Attention ABMIL slice classifier (Phase B).

    Identical to ABMILSliceClassifier except that a PatchTransformer
    contextualisation step is applied after projection and before the
    Gated ABMIL aggregator.

    Args:
        encoder:    CNNPatchClassifier instance (trained end-to-end).
        proj_dim:   patch feature projection dimension.
        attn_dim:   ABMIL internal attention dimension.
        dropout:    dropout for projector, ABMIL and classification head.
        sa_n_heads: Transformer attention heads (proj_dim % sa_n_heads == 0).
        sa_n_layers: number of Transformer encoder layers.
    """

    def __init__(
        self,
        encoder:     CNNPatchClassifier,
        proj_dim:    int   = 512,
        attn_dim:    int   = 128,
        dropout:     float = 0.25,
        sa_n_heads:  int   = 8,
        sa_n_layers: int   = 2,
    ):
        super().__init__()

        self.encoder      = encoder
        raw_feat_dim: int = encoder.feat_dim

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

        # ── Patch-level Transformer (self-attention over N patches) ──────
        self.patch_transformer = PatchTransformer(
            feat_dim = self._feat_dim,
            n_heads  = sa_n_heads,
            n_layers = sa_n_layers,
            dropout  = dropout * 0.5,   # slightly lower inside Transformer
        )

        # ── Gated ABMIL aggregation ──────────────────────────────────────
        self.aggregator = GatedAttention(self._feat_dim, attn_dim, dropout)

        # ── Classification head ──────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(self._feat_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    @property
    def feat_dim(self) -> int:
        return self._feat_dim

    # ------------------------------------------------------------------
    def encode_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Extract raw CNN feature vector for each patch.

        Args:
            patches: (B, N, 1, P, P)
        Returns:
            features: (B, N, raw_feat_dim)
        """
        B, N, C, P_h, P_w = patches.shape
        flat     = patches.view(B * N, C, P_h, P_w)
        feat_map = self.encoder.backbone(flat)[-1]
        attn_map = self.encoder.spatial_attn(feat_map)
        v        = (feat_map * attn_map).mean(dim=[2, 3])
        return v.view(B, N, -1)

    # ------------------------------------------------------------------
    def forward(
        self,
        patches:     torch.Tensor,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            patches:     (B, N, 1, P, P)
            return_attn: if True, also return ABMIL attention weights (B, N).

        Returns:
            logits: (B, 1)
            attn:   (B, N)  [only if return_attn=True]
        """
        # 1. CNN encode
        features = self.encode_patches(patches)            # (B, N, raw_feat_dim)

        # 2. Project
        B, N, _ = features.shape
        h = self.projector(
            features.view(B * N, -1)
        ).view(B, N, self._feat_dim)                       # (B, N, M)

        # 3. Patch Transformer — contextualise across patches
        h = self.patch_transformer(h)                      # (B, N, M)

        # 4. Gated ABMIL aggregation
        z, attn_weights = self.aggregator(h)               # (B, M), (B, N)

        # 5. Classify
        logits = self.head(z)                              # (B, 1)

        if return_attn:
            return logits, attn_weights
        return logits


# ---------------------------------------------------------------------------
#  Factory
# ---------------------------------------------------------------------------

def build_sa_classifier_scratch(
    backbone:    str   = 'resnet50',
    pretrained:  bool  = True,
    proj_dim:    int   = 512,
    attn_dim:    int   = 128,
    dropout:     float = 0.25,
    sa_n_heads:  int   = 8,
    sa_n_layers: int   = 2,
) -> SABMILSliceClassifier:
    """
    Build an SABMILSliceClassifier with a freshly initialised CNN encoder.
    The entire model is trained end-to-end.

    Args:
        backbone:    ResNet/EfficientNet backbone name.
        pretrained:  if True, initialise backbone from ImageNet weights.
        proj_dim:    patch feature projection dimension.
        attn_dim:    ABMIL internal attention dimension.
        dropout:     dropout probability.
        sa_n_heads:  Transformer attention heads (proj_dim % sa_n_heads == 0).
        sa_n_layers: Transformer encoder depth.
    """
    from hexmil.models.cnn_patch_classifier import build_cnn_classifier

    encoder = build_cnn_classifier(
        backbone   = backbone,
        pretrained = pretrained,
        freeze_ratio = 0.0,
    )
    return SABMILSliceClassifier(
        encoder     = encoder,
        proj_dim    = proj_dim,
        attn_dim    = attn_dim,
        dropout     = dropout,
        sa_n_heads  = sa_n_heads,
        sa_n_layers = sa_n_layers,
    )
