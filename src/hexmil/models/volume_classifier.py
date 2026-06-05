"""
volume_classifier.py
---------------------
HexMIL — Stage 2 volume-level forgery classifier.

Architecture:
    SliceMIL encoder (FROZEN)  → h_k + sinusoidal positional encoding
    Gated Volume Aggregator (TRAINABLE):
        β = softmax(gated_attn([h_1…h_K]))  →  v = Σ β_k h_k
    MLP head (TRAINABLE): v → logit  (binary BCE)

3-D heatmap: heatmap[k, y, x] = β_k × α̃_k[y, x]

Factory: build_volume_classifier(slice_ckpt_dir, K, attn_dim, dropout, device)
"""

from __future__ import annotations

import json
import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn

from hexmil.models.abmil_slice_classifier import (
    ABMILSliceClassifier,
    GatedAttention,
    build_abmil_classifier_scratch,
)


# ---------------------------------------------------------------------------
#  Positional encoding (sinusoidal, 1-D)
# ---------------------------------------------------------------------------

class SinPosEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding over a Z-axis index.

    Given an integer z (or a batch of integers), returns a vector of
    dimension `d_model` following the standard Vaswani formulation.
    Works for arbitrary z values (not limited to a fixed sequence length).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        # pre-compute divisor terms: 10000^(2i/d_model)
        div = torch.exp(
            -math.log(10000.0) *
            torch.arange(0, d_model, 2, dtype=torch.float32) / d_model
        )
        self.register_buffer("div", div)        # (d_model//2,)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z:  Long or Float tensor of any shape (…)
        Returns:
            encoding of shape (…, d_model)
        """
        z = z.float().unsqueeze(-1)             # (…, 1)
        div = self.div                          # (d_model//2,)
        angles = z * div                        # (…, d_model//2)
        pe = torch.zeros(*z.shape[:-1], self.d_model, device=z.device, dtype=z.dtype)
        pe[..., 0::2] = torch.sin(angles)
        pe[..., 1::2] = torch.cos(angles)
        return pe


# ---------------------------------------------------------------------------
#  Volume Classifier
# ---------------------------------------------------------------------------

class VolumeClassifier(nn.Module):
    """
    Stage 2 volume-level binary classifier.

    Args:
        slice_encoder:  ABMILSliceClassifier instance – will be FROZEN.
        feat_dim:       feature dimension of the slice encoder output.
        K:              fixed window length (number of slices).
        attn_dim:       hidden dim for the gated volume attention.
        dropout:        dropout rate for the MLP head.
    """

    def __init__(
        self,
        slice_encoder: ABMILSliceClassifier,
        feat_dim: int,
        K: int,
        attn_dim: int = 256,
        dropout: float = 0.25,
    ):
        super().__init__()

        self.K        = K
        self.feat_dim = feat_dim

        # ── frozen Stage 1 encoder ───────────────────────────────────────
        self.slice_encoder = slice_encoder
        for p in self.slice_encoder.parameters():
            p.requires_grad_(False)
        self.slice_encoder.eval()

        # ── positional encoding (fixed, no learnable params) ────────────
        self.pos_enc = SinPosEncoding(feat_dim)

        # ── gated MIL aggregator over Z ──────────────────────────────────
        self.z_aggregator = GatedAttention(feat_dim, attn_dim, dropout)

        # ── classification head ─────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    # ------------------------------------------------------------------
    def train(self, mode: bool = True):
        """Override train() to always keep slice_encoder in eval mode."""
        super().train(mode)
        self.slice_encoder.eval()
        return self

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _encode_slice(
        self, patches: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode one slice via the frozen Stage 1 model.

        Args:
            patches:  (N, 1, P, P) — patches for a single slice.
        Returns:
            h:     (feat_dim,)  weighted feature vector
            alpha: (N,)         per-patch attention weights
        """
        enc       = self.slice_encoder
        patches_b = patches.unsqueeze(0)            # (1, N, 1, P, P)

        # 1. Per-patch backbone features (backbone + spatial_attn + weighted GAP)
        features  = enc.encode_patches(patches_b)   # (1, N, raw_feat_dim)
        B, N, _   = features.shape

        # 2. Linear projection
        h = enc.projector(
            features.view(B * N, -1)
        ).view(B, N, enc._feat_dim)                 # (1, N, feat_dim)

        # 3. Gated ABMIL aggregation → feature + per-patch attention
        z, alpha_b = enc.aggregator(h)              # (1, feat_dim), (1, N)

        return z.squeeze(0), alpha_b.squeeze(0)     # (feat_dim,), (N,)

    # ------------------------------------------------------------------
    def forward(
        self,
        patches_seq: torch.Tensor,
        z_indices:   torch.Tensor,
        valid_mask:  Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            patches_seq: (K, N, 1, P, P)  — patch bags per slice.
            z_indices:   (K,)  LongTensor  — actual Z coordinates (−1 = padded).
            valid_mask:  (K,)  BoolTensor  — True for real slices, False for padding.
                         If None, derived from z_indices >= 0.
            return_attn: if True, also return (beta, alpha_list).

        Returns:
            logit:  scalar tensor               (1,)
            attn:   (beta (K,), alpha_list list[(N,)] per slice)  iff return_attn
        """
        K = self.K
        device = patches_seq.device

        if valid_mask is None:
            valid_mask = (z_indices >= 0)           # (K,)

        # ── Per-slice encoding ───────────────────────────────────────────
        h_list     = []
        alpha_list = []
        for k in range(K):
            if valid_mask[k]:
                with torch.no_grad():
                    h_k, alpha_k = self._encode_slice(patches_seq[k])
            else:
                # padding slice → zero vector
                h_k     = torch.zeros(self.feat_dim, device=device,
                                      dtype=patches_seq.dtype)
                alpha_k = torch.zeros(patches_seq.shape[1], device=device,
                                      dtype=patches_seq.dtype)
            h_list.append(h_k)
            alpha_list.append(alpha_k)

        H = torch.stack(h_list, dim=0).unsqueeze(0)    # (1, K, feat_dim)

        # ── Positional encoding ─────────────────────────────────────────
        z_clamp  = z_indices.clamp(min=0)               # (K,)
        pe       = self.pos_enc(z_clamp)                # (K, feat_dim)
        H = H + pe.unsqueeze(0)                         # (1, K, feat_dim)

        # ── Gated volume aggregation ────────────────────────────────────
        v, beta = self.z_aggregator(H)                  # (1, feat_dim), (1, K)
        v    = v.squeeze(0)                             # (feat_dim,)
        beta = beta.squeeze(0)                          # (K,)

        # zero-out beta for padded slices (numerical hygiene)
        if valid_mask is not None:
            beta = beta * valid_mask.float()
            beta_sum = beta.sum().clamp(min=1e-8)
            beta = beta / beta_sum

        # ── Classification head ─────────────────────────────────────────
        logit = self.head(v)                            # (1,)

        if return_attn:
            return logit, (beta, alpha_list)
        return logit, None


# ---------------------------------------------------------------------------
#  Factory
# ---------------------------------------------------------------------------

def build_volume_classifier(
    slice_ckpt_dir: str,
    K:             int   = 16,
    attn_dim:      int   = 256,
    dropout:       float = 0.25,
    device:        Optional[torch.device] = None,
) -> Tuple[VolumeClassifier, dict]:
    """
    Build a VolumeClassifier:
      1. Load Stage 1 args.json from `slice_ckpt_dir`.
      2. Reconstruct ABMILSliceClassifier and load best_model.pt weights.
      3. Wrap in VolumeClassifier with gated Z aggregator.

    Returns:
        model:     VolumeClassifier (Stage 1 encoder frozen)
        slice_args: dict with Stage 1 configuration (patch_size, stride, …)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Read Stage 1 args ────────────────────────────────────────────────
    args_path = os.path.join(slice_ckpt_dir, "args.json")
    with open(args_path) as f:
        sargs = json.load(f)

    # ── Re-build Stage 1 model ───────────────────────────────────────────
    slice_model: ABMILSliceClassifier = build_abmil_classifier_scratch(
            backbone   = sargs.get("backbone",   "resnet50"),
            pretrained = False,
            patch_size = sargs.get("patch_size", 64),
            proj_dim   = sargs.get("proj_dim",   512),
            attn_dim   = sargs.get("attn_dim",   256),
            dropout    = sargs.get("dropout",    0.25),
        )
    ckpt_path = os.path.join(slice_ckpt_dir, "best_model.pt")
    ckpt      = torch.load(ckpt_path, map_location="cpu")
    # support both plain state_dict and {"model_state_dict": …}
    state_dict = ckpt.get("model_state_dict", ckpt)
    slice_model.load_state_dict(state_dict, strict=True)
    slice_model.to(device).eval()

    feat_dim = slice_model.feat_dim

    # ── Wrap in VolumeClassifier ─────────────────────────────────────────
    model = VolumeClassifier(
        slice_encoder = slice_model,
        feat_dim      = feat_dim,
        K             = K,
        attn_dim      = attn_dim,
        dropout       = dropout,
    ).to(device)

    return model, sargs
