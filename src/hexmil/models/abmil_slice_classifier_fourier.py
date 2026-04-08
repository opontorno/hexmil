"""
abmil_slice_classifier_fourier.py
-----------------------------------
Phase B — Fourier-Augmented ABMIL Slice Classifier (FA-ABMIL).

Each patch is processed through two parallel streams before aggregation:

  1. **Spatial stream**  : patch → CNN backbone + spatial_attn + GAP → f_s ∈ R^D
  2. **Frequency stream**: patch → rfft2 → log-magnitude → small CNN → f_f ∈ R^F

The two vectors are concatenated and projected to *proj_dim* before the
standard Gated ABMIL pooling and classification head.

The motivation is that GAN / diffusion artefacts often leave structured
traces in the frequency domain (spectral peaks, checkerboard patterns) that
are hard to capture in the spatial domain alone.

Architecture:
    ┌──────────────────────────────────────────────────────────────┐
    │  Per-patch (B*N inputs of shape 1×P×P):                      │
    │                                                              │
    │  Spatial branch                                              │
    │  patch ──→ CNN encoder (backbone + spatial_attn + GAP)       │
    │         ──→ f_s ∈ R^D                                        │
    │                                                              │
    │  Frequency branch                                            │
    │  patch ──→ rfft2 ──→ |·| ──→ log1p ──→ fftshift ──→ resize  │
    │         ──→ FourierMagnitudeNet (3-layer conv + GAP + FC)    │
    │         ──→ f_f ∈ R^F                                        │
    └──────────────────────────────────────────────────────────────┘
               │  [f_s ; f_f]  ∈ R^(D+F)
    ┌──────────▼───────────────────────────────────────────────────┐
    │  Linear Projector  (D+F) → proj_dim (M)                      │
    └──────────────────────────────────────────────────────────────┘
               │
    ┌──────────▼───────────────────────────────────────────────────┐
    │  Gated ABMIL Attention                                       │
    │  a_k = softmax(w^T (tanh(Vh_k) ⊙ σ(Uh_k)))                 │
    │  z   = Σ_k a_k h_k                                          │
    └──────────────────────────────────────────────────────────────┘
               │
    ┌──────────▼───────────────────────────────────────────────────┐
    │  Classification Head  M → 256 → 1                            │
    └──────────────────────────────────────────────────────────────┘

Key hyper-parameters:
    fourier_feat_dim (int, default 256) — output dimension of the frequency
        branch.  Total fused dim before projection = CNN feat_dim + fourier_feat_dim.
    patch_size (int, default 128) — spatial size of input patches.  Must
        match the value used in the patch-extraction pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hexmil.models.cnn_patch_classifier import CNNPatchClassifier
from hexmil.models.abmil_slice_classifier import GatedAttention


# ---------------------------------------------------------------------------
#  Fourier magnitude branch
# ---------------------------------------------------------------------------

class FourierMagnitudeNet(nn.Module):
    """
    Lightweight CNN that encodes the log-magnitude FFT spectrum of a patch.

    Pipeline per patch:
        1. ``torch.fft.rfft2``  →  complex spectrum (1, P, P//2+1)
        2. ``.abs()``           →  real magnitude
        3. ``log1p``            →  log-compress (suppresses DC dominance)
        4. ``roll`` along H     →  DC shifted to centre row
        5. bilinear resize      →  (1, P, P) for a uniform square input
        6. instance normalise   →  zero-mean / unit-std per image
        7. 3-layer conv stack → AdaptiveAvgPool2d(1) → linear → feat_dim

    Args:
        in_size:  spatial size P of the patch (before rfft2).
        feat_dim: output feature dimension F.
    """

    def __init__(self, in_size: int = 128, feat_dim: int = 256):
        super().__init__()
        self.in_size  = in_size
        self.feat_dim = feat_dim

        self.cnn = nn.Sequential(
            # block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                         # → P/2
            # block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                         # → P/4
            # block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                 # → 1×1
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, P, P)  — raw spatial patches.
        Returns:
            (B, feat_dim)
        """
        # ── 1. Compute log-magnitude spectrum ────────────────────────────
        spec = torch.fft.rfft2(x, norm='ortho')              # (B,1,P,P//2+1) complex
        mag  = spec.abs()                                     # real magnitude
        mag  = torch.log1p(mag)                               # log-compress

        # ── 2. Shift DC component to centre (aids conv learning) ─────────
        mag  = torch.roll(mag, shifts=mag.shape[-2] // 2, dims=-2)

        # ── 3. Resize to square (P × P) for uniform spatial structure ────
        mag  = F.interpolate(
            mag, size=(self.in_size, self.in_size),
            mode='bilinear', align_corners=False,
        )

        # ── 4. Instance-wise normalisation ───────────────────────────────
        mu  = mag.mean(dim=[-2, -1], keepdim=True)
        std = mag.std (dim=[-2, -1], keepdim=True).clamp(min=1e-6)
        mag = (mag - mu) / std

        # ── 5. Encode ─────────────────────────────────────────────────────
        return self.head(self.cnn(mag))   # (B, feat_dim)


# ---------------------------------------------------------------------------
#  Main model
# ---------------------------------------------------------------------------

class FABMILSliceClassifier(nn.Module):
    """
    Fourier-Augmented ABMIL slice classifier (Phase B).

    Identical to :class:`ABMILSliceClassifier` except that each patch is
    encoded by both the standard CNN spatial encoder **and** a dedicated
    Fourier magnitude branch.  The two feature vectors are concatenated,
    projected to *proj_dim*, and then fed to the Gated ABMIL aggregator.

    Args:
        encoder:         CNNPatchClassifier instance (trained end-to-end).
        patch_size:      spatial size P of input patches.
        fourier_feat_dim: output dimension of the Fourier branch (F).
        proj_dim:        dimension to project (D+F) to before ABMIL.
        attn_dim:        ABMIL internal attention dimension.
        dropout:         dropout for projector, ABMIL, and classification head.
    """

    def __init__(
        self,
        encoder:          CNNPatchClassifier,
        patch_size:       int   = 128,
        fourier_feat_dim: int   = 256,
        proj_dim:         int   = 512,
        attn_dim:         int   = 128,
        dropout:          float = 0.25,
    ):
        super().__init__()

        self.encoder      = encoder
        raw_feat_dim: int = encoder.feat_dim          # e.g. 2048 for ResNet-50
        fused_dim: int    = raw_feat_dim + fourier_feat_dim

        # ── Fourier branch ────────────────────────────────────────────────
        self.fourier_branch = FourierMagnitudeNet(
            in_size  = patch_size,
            feat_dim = fourier_feat_dim,
        )

        # ── Feature projector (fused_dim → proj_dim) ─────────────────────
        if proj_dim is not None and proj_dim != fused_dim:
            self.projector = nn.Sequential(
                nn.Linear(fused_dim, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self._feat_dim = proj_dim
        else:
            self.projector = nn.Identity()
            self._feat_dim = fused_dim

        # ── Gated ABMIL aggregation ───────────────────────────────────────
        self.aggregator = GatedAttention(self._feat_dim, attn_dim, dropout)

        # ── Classification head ───────────────────────────────────────────
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
        Encode each patch through the spatial CNN + Fourier branch and
        return the concatenated feature vectors.

        Args:
            patches: (B, N, 1, P, P)
        Returns:
            features: (B, N, raw_feat_dim + fourier_feat_dim)
        """
        B, N, C, P_h, P_w = patches.shape
        flat = patches.view(B * N, C, P_h, P_w)   # (B*N, 1, P, P)

        # Spatial stream
        feat_map  = self.encoder.backbone(flat)[-1]       # (B*N, C_feat, h, w)
        attn_map  = self.encoder.spatial_attn(feat_map)   # (B*N, 1, h, w)
        f_spatial = (feat_map * attn_map).mean(dim=[2, 3])  # (B*N, D)

        # Frequency stream
        f_freq = self.fourier_branch(flat)                 # (B*N, F)

        # Fuse
        fused = torch.cat([f_spatial, f_freq], dim=1)     # (B*N, D+F)
        return fused.view(B, N, -1)                        # (B, N, D+F)

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
        # 1. Dual-stream encoding
        features = self.encode_patches(patches)            # (B, N, D+F)

        # 2. Project
        B, N, _ = features.shape
        h = self.projector(
            features.view(B * N, -1)
        ).view(B, N, self._feat_dim)                       # (B, N, M)

        # 3. Gated ABMIL aggregation
        z, attn_weights = self.aggregator(h)               # (B, M), (B, N)

        # 4. Classify
        logits = self.head(z)                              # (B, 1)

        if return_attn:
            return logits, attn_weights
        return logits


# ---------------------------------------------------------------------------
#  Factory
# ---------------------------------------------------------------------------

def build_fabmil_classifier_scratch(
    backbone:          str   = 'resnet50',
    pretrained:        bool  = True,
    patch_size:        int   = 128,
    fourier_feat_dim:  int   = 256,
    proj_dim:          int   = 512,
    attn_dim:          int   = 128,
    dropout:           float = 0.25,
) -> FABMILSliceClassifier:
    """
    Build a :class:`FABMILSliceClassifier` with a freshly initialised encoder.
    The entire model (CNN encoder + Fourier branch + projector + ABMIL + head)
    is trained end-to-end.

    Args:
        backbone:         ResNet/EfficientNet backbone name.
        pretrained:       if True, initialise backbone from ImageNet weights.
        patch_size:       spatial size P of input patches.
        fourier_feat_dim: output dimension of the Fourier branch.
        proj_dim:         feature dimension after fusion projection.
        attn_dim:         ABMIL internal attention dimension.
        dropout:          dropout probability.

    Returns:
        FABMILSliceClassifier ready for end-to-end training.
    """
    from hexmil.models.cnn_patch_classifier import build_cnn_classifier

    encoder = build_cnn_classifier(
        backbone     = backbone,
        pretrained   = pretrained,
        freeze_ratio = 0.0,
    )
    return FABMILSliceClassifier(
        encoder          = encoder,
        patch_size       = patch_size,
        fourier_feat_dim = fourier_feat_dim,
        proj_dim         = proj_dim,
        attn_dim         = attn_dim,
        dropout          = dropout,
    )
