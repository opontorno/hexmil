"""
volume_classifier_sa.py
-----------------------
SA-HexMIL — Stage 2 volume classifier with Slice Self-Attention.

Extends HexMIL by inserting a SliceTransformer between positional encoding
and the Gated Volume Aggregator, letting each slice attend to all other
slices in the K-slice window before pooling.

Architecture:
    SliceMIL encoder (FROZEN) + sinusoidal pos-enc
    → SliceTransformer (TRAINABLE, pre-LN)
    → Gated Volume Aggregator (TRAINABLE): β = softmax(...)  v = Σ β_k h_k'
    → MLP head: v → logit

Accepts checkpoints from both ABMILSliceClassifier and SABMILSliceClassifier
(auto-detected from args.json).

Factory: build_sa_volume_classifier(slice_ckpt_dir, K, ...) → SAVolumeClassifier
"""

from __future__ import annotations

import json
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn

from hexmil.models.volume_classifier import SinPosEncoding
from hexmil.models.abmil_slice_classifier import GatedAttention


# ---------------------------------------------------------------------------
#  Slice Transformer
# ---------------------------------------------------------------------------

class SliceTransformer(nn.Module):
    """
    Transformer encoder over K slice embeddings (Pre-LN, batch_first=True).

    Each slice can attend to every other slice in the window, capturing
    cross-slice inconsistencies that are invisible in independent aggregation.

    Args:
        feat_dim:  slice embedding dimension.
        n_heads:   number of attention heads (feat_dim % n_heads == 0).
        n_layers:  Transformer depth.
        dropout:   dropout inside each encoder layer.
        ff_mult:   feedforward hidden dim = feat_dim * ff_mult.
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
            norm_first      = True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers = n_layers,
            enable_nested_tensor = False,
        )

    def forward(
        self,
        h:           torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            h:                      (B, K, D)  slice embeddings.
            src_key_padding_mask:   (B, K) bool mask — True = padded slice to ignore.
        Returns:
            h': (B, K, D)  contextualised slice embeddings.
        """
        return self.transformer(h, src_key_padding_mask=src_key_padding_mask)


# ---------------------------------------------------------------------------
#  SA Volume Classifier
# ---------------------------------------------------------------------------

class SAVolumeClassifier(nn.Module):
    """
    Stage 2 volume-level binary classifier with Slice Self-Attention.

    Args:
        slice_encoder:  ABMILSliceClassifier (or SA variant) — FROZEN.
        feat_dim:       feature dimension of the slice encoder output.
        K:              fixed window length (number of slices).
        attn_dim:       hidden dim for the gated volume aggregator.
        dropout:        dropout for MLP head and volume aggregator.
        sa_n_heads:     Transformer attention heads.
        sa_n_layers:    Transformer encoder depth.
    """

    def __init__(
        self,
        slice_encoder,
        feat_dim:    int,
        K:           int,
        attn_dim:    int   = 256,
        dropout:     float = 0.25,
        sa_n_heads:  int   = 8,
        sa_n_layers: int   = 2,
    ):
        super().__init__()

        self.K        = K
        self.feat_dim = feat_dim

        # ── frozen Stage 1 encoder ───────────────────────────────────────
        self.slice_encoder = slice_encoder
        for p in self.slice_encoder.parameters():
            p.requires_grad_(False)
        self.slice_encoder.eval()

        # ── positional encoding (fixed) ──────────────────────────────────
        self.pos_enc = SinPosEncoding(feat_dim)

        # ── Slice Transformer (self-attention over K slices) ─────────────
        self.slice_transformer = SliceTransformer(
            feat_dim = feat_dim,
            n_heads  = sa_n_heads,
            n_layers = sa_n_layers,
            dropout  = dropout * 0.5,
        )

        # ── Gated aggregator over contextualised slice representations ────
        self.z_aggregator = GatedAttention(feat_dim, attn_dim, dropout)

        # ── Classification head ──────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    # ------------------------------------------------------------------
    def train(self, mode: bool = True):
        """Always keep slice_encoder in eval mode."""
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
            patches: (N, 1, P, P)
        Returns:
            h:     (feat_dim,)
            alpha: (N,)  per-patch attention weights
        """
        enc       = self.slice_encoder
        patches_b = patches.unsqueeze(0)            # (1, N, 1, P, P)
        features  = enc.encode_patches(patches_b)   # (1, N, raw_feat_dim)
        B, N, _   = features.shape
        h = enc.projector(
            features.view(B * N, -1)
        ).view(B, N, enc._feat_dim)                 # (1, N, feat_dim)

        # SA encoder also has patch_transformer — apply if present
        if hasattr(enc, 'patch_transformer'):
            h = enc.patch_transformer(h)            # (1, N, feat_dim)

        z, alpha_b = enc.aggregator(h)              # (1, feat_dim), (1, N)
        return z.squeeze(0), alpha_b.squeeze(0)

    # ------------------------------------------------------------------
    def forward(
        self,
        patches_seq: torch.Tensor,
        z_indices:   torch.Tensor,
        valid_mask:  Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, list]]]:
        """
        Args:
            patches_seq: (K, N, 1, P, P)
            z_indices:   (K,)  LongTensor — actual Z coordinates (−1 = padding).
            valid_mask:  (K,)  BoolTensor — True for real slices.
            return_attn: if True, also return (beta (K,), alpha_list).

        Returns:
            logit:  (1,)
            attn:   (beta, alpha_list)  iff return_attn
        """
        K      = self.K
        device = patches_seq.device

        if valid_mask is None:
            valid_mask = (z_indices >= 0)

        # ── Per-slice encoding ───────────────────────────────────────────
        h_list     = []
        alpha_list = []
        for k in range(K):
            if valid_mask[k]:
                with torch.no_grad():
                    h_k, alpha_k = self._encode_slice(patches_seq[k])
            else:
                h_k     = torch.zeros(self.feat_dim, device=device,
                                      dtype=patches_seq.dtype)
                alpha_k = torch.zeros(patches_seq.shape[1], device=device,
                                      dtype=patches_seq.dtype)
            h_list.append(h_k)
            alpha_list.append(alpha_k)

        H = torch.stack(h_list, dim=0).unsqueeze(0)    # (1, K, feat_dim)

        # ── Positional encoding ─────────────────────────────────────────
        z_clamp = z_indices.clamp(min=0)
        pe      = self.pos_enc(z_clamp)                # (K, feat_dim)
        H       = H + pe.unsqueeze(0)                  # (1, K, feat_dim)

        # ── Slice Transformer — contextualise across slices ──────────────
        # Build padding mask: True = padded (PyTorch convention)
        pad_mask = (~valid_mask).unsqueeze(0)           # (1, K) bool
        H = self.slice_transformer(H, src_key_padding_mask=pad_mask)
                                                        # (1, K, feat_dim)

        # ── Gated aggregation ────────────────────────────────────────────
        v, beta = self.z_aggregator(H)                  # (1, feat_dim), (1, K)
        v    = v.squeeze(0)                             # (feat_dim,)
        beta = beta.squeeze(0)                          # (K,)

        # zero-out padded slices
        beta     = beta * valid_mask.float()
        beta_sum = beta.sum().clamp(min=1e-8)
        beta     = beta / beta_sum

        # ── Head ─────────────────────────────────────────────────────────
        logit = self.head(v)                            # (1,)

        if return_attn:
            return logit, (beta, alpha_list)
        return logit, None


# ---------------------------------------------------------------------------
#  Factory
# ---------------------------------------------------------------------------

def build_sa_volume_classifier(
    slice_ckpt_dir: str,
    K:             int   = 16,
    attn_dim:      int   = 256,
    dropout:       float = 0.25,
    sa_n_heads:    int   = 8,
    sa_n_layers:   int   = 2,
    device:        Optional[torch.device] = None,
) -> Tuple['SAVolumeClassifier', dict]:
    """
    Build an SAVolumeClassifier from a Stage 1 checkpoint.

    Automatically detects whether the checkpoint uses ABMILSliceClassifier
    or SABMILSliceClassifier by checking for 'use_sa' in args.json.

    Returns:
        model:      SAVolumeClassifier (Stage 1 encoder frozen)
        slice_args: dict with Stage 1 configuration
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args_path = os.path.join(slice_ckpt_dir, "args.json")
    with open(args_path) as f:
        sargs = json.load(f)

    use_sa = sargs.get("use_sa", False)

    if use_sa:
        from hexmil.models.abmil_slice_classifier_sa import (
            build_sa_classifier_scratch,
        )
        slice_model = build_sa_classifier_scratch(
            backbone    = sargs.get("backbone",    "resnet50"),
            pretrained  = False,
            proj_dim    = sargs.get("proj_dim",    512),
            attn_dim    = sargs.get("attn_dim",    256),
            dropout     = sargs.get("dropout",     0.25),
            sa_n_heads  = sargs.get("sa_n_heads",  8),
            sa_n_layers = sargs.get("sa_n_layers", 2),
        )
    else:
        from hexmil.models.abmil_slice_classifier import (
            build_abmil_classifier_scratch,
        )
        slice_model = build_abmil_classifier_scratch(
            backbone   = sargs.get("backbone",   "resnet50"),
            pretrained = False,
            patch_size = sargs.get("patch_size", 64),
            proj_dim   = sargs.get("proj_dim",   512),
            attn_dim   = sargs.get("attn_dim",   256),
            dropout    = sargs.get("dropout",    0.25),
        )

    ckpt_path  = os.path.join(slice_ckpt_dir, "best_model.pt")
    ckpt       = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    slice_model.load_state_dict(state_dict, strict=True)
    slice_model.to(device).eval()

    feat_dim = slice_model.feat_dim

    model = SAVolumeClassifier(
        slice_encoder = slice_model,
        feat_dim      = feat_dim,
        K             = K,
        attn_dim      = attn_dim,
        dropout       = dropout,
        sa_n_heads    = sa_n_heads,
        sa_n_layers   = sa_n_layers,
    ).to(device)

    print(f"  Loaded Stage 1 {'SA-ABMIL' if use_sa else 'ABMIL'} encoder "
          f"(frozen)  feat_dim={feat_dim}")

    return model, sargs
