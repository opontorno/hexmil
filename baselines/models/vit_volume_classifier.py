"""
vit_volume_classifier.py
========================
ViT-ABMIL: Vision Transformer slice encoder + Gated Attention volume aggregator.

Architecture:
    For each of K slices:
        slice_k (H, W) → ViT (processes slice as non-overlapping patch tokens)
                       → [CLS] embedding h_k  (embed_dim,)
                       + spatial attention map α_k from last attn layer
    Volume level:
        [h_1 ... h_K] + sinusoidal Z-encoding → GatedAttention
                      → β_k weights
                      → v = Σ β_k h_k
                      → MLP head → logit

Key difference from ABMIL:
  - ABMIL: patch-level GatedAttention WITHIN each slice → pooled slice embedding
  - ViT-ABMIL: ViT processes entire slice image → CLS token as slice embedding
               (richer spatial reasoning, no manual patch extraction)
"""
from __future__ import annotations
import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hexmil.models.abmil_slice_classifier import GatedAttention


# ---------------------------------------------------------------------------
#  ViT slice encoder (processes one 2D slice as a sequence of patch tokens)
# ---------------------------------------------------------------------------

class ViTSliceEncoder(nn.Module):
    """
    Lightweight ViT that encodes a single CT slice (H×W) to a CLS embedding.

    Args:
        slice_size:  expected HxW of input slice (square, default 512 — will be resized internally)
        token_size:  spatial size of each patch token in pixels (default 32)
        in_chans:    input channels (1 for grayscale CT)
        embed_dim:   transformer embedding dimension
        depth:       number of transformer layers
        num_heads:   attention heads
        target_size: resize slice to this before tokenisation (for memory efficiency)
    """
    def __init__(
        self,
        token_size: int = 32,
        in_chans:   int = 3,
        embed_dim:  int = 256,
        depth:      int = 4,
        num_heads:  int = 8,
        target_size: int = 256,   # resize slice to this before tokenisation
        drop_rate:   float = 0.1,
    ):
        super().__init__()
        self.target_size = target_size
        self.token_size  = token_size
        self.embed_dim   = embed_dim

        G = target_size // token_size        # grid size, e.g. 256//32 = 8
        self.grid_size   = G
        num_patches      = G * G

        # Patch embedding (conv2d)
        self.patch_embed = nn.Conv2d(in_chans, embed_dim,
                                     kernel_size=token_size, stride=token_size)

        # CLS token + positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop  = nn.Dropout(drop_rate)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=drop_rate, batch_first=True,
                norm_first=True,             # pre-LN, more stable
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

        # Hook for last-block attention
        self._last_attn: Optional[torch.Tensor] = None
        self.blocks[-1].self_attn.register_forward_hook(self._save_attn)

    def _save_attn(self, module, inp, out):
        # out is (attn_output, attn_weights) when need_weights=True
        # but TransformerEncoderLayer doesn't expose attn_weights directly
        # We'll use rollout-style: just store the raw output for now
        pass   # See forward() for attention extraction

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, slices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            slices: (B, 3, H, W) — one slice per batch item (1-ch repeated to 3-ch)
        Returns:
            cls_emb:  (B, embed_dim) — CLS token embedding
            attn_map: (B, G, G)      — spatial attention (CLS→patches, last layer)
        """
        B, C, H, W = slices.shape

        # Resize to target_size if needed
        if H != self.target_size or W != self.target_size:
            slices = F.interpolate(slices, size=(self.target_size, self.target_size),
                                   mode='bilinear', align_corners=False)

        # Tokenise: (B, embed_dim, G, G) → (B, G*G, embed_dim)
        tokens = self.patch_embed(slices).flatten(2).transpose(1, 2)

        # Prepend CLS
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)        # (B, G*G+1, D)
        tokens = self.pos_drop(tokens + self.pos_embed)

        # Transformer (standard: no per-layer attention extraction)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        cls_emb = tokens[:, 0]                          # (B, D)

        # Approximate spatial attention: cosine similarity of CLS with each patch token
        patch_tokens = tokens[:, 1:]                    # (B, N, D)
        cls_exp      = cls_emb.unsqueeze(1)             # (B, 1, D)
        attn_scores  = F.cosine_similarity(cls_exp, patch_tokens, dim=-1)  # (B, N)
        attn_scores  = attn_scores.softmax(dim=-1)
        G = self.grid_size
        attn_map = attn_scores.reshape(B, G, G)         # (B, G, G)

        return cls_emb, attn_map


# ---------------------------------------------------------------------------
#  Sinusoidal positional encoding (reuse from volume_classifier.py logic)
# ---------------------------------------------------------------------------

class SinPosEnc1D(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        div = torch.exp(
            -math.log(10000.0) *
            torch.arange(0, d_model, 2, dtype=torch.float32) / d_model
        )
        self.register_buffer("div", div)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z   = z.float().unsqueeze(-1)
        ang = z * self.div
        pe  = torch.zeros(*z.shape[:-1], self.div.shape[0] * 2, device=z.device)
        pe[..., 0::2] = ang.sin()
        pe[..., 1::2] = ang.cos()
        return pe


# ---------------------------------------------------------------------------
#  ViTVolumeClassifier
# ---------------------------------------------------------------------------

class ViTVolumeClassifier(nn.Module):
    """
    Full ViT-ABMIL volume classifier.

    Args:
        slice_encoder:  ViTSliceEncoder instance (or None to build with defaults)
        K:              number of slices per window
        attn_dim:       hidden dim for GatedAttention
        dropout:        dropout in MLP head
    """
    def __init__(
        self,
        K:            int  = 16,
        token_size:   int  = 32,
        embed_dim:    int  = 256,
        depth:        int  = 4,
        num_heads:    int  = 8,
        target_size:  int  = 256,
        attn_dim:     int  = 128,
        dropout:      float = 0.25,
    ):
        super().__init__()
        self.K           = K
        self.embed_dim   = embed_dim

        self.slice_enc   = ViTSliceEncoder(
            token_size=token_size, in_chans=3, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads, target_size=target_size,
        )
        self.pos_enc     = SinPosEnc1D(embed_dim)
        self.vol_attn    = GatedAttention(embed_dim, attn_dim, dropout=dropout)
        self.head        = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        slices:    torch.Tensor,    # (K, 1, H, W) — one volume window
        z_indices: torch.Tensor,    # (K,) int
        valid:     torch.Tensor,    # (K,) bool
        return_attn: bool = False,
    ):
        """
        Args:
            slices:    (K, 1, H, W)
            z_indices: (K,) absolute z positions (for positional encoding)
            valid:     (K,) mask of valid slices
        Returns:
            logit:     scalar
            (beta, alpha_list) if return_attn=True
        """
        K = slices.shape[0]

        # Encode each slice
        h_list:     List[torch.Tensor] = []
        alpha_list: List[torch.Tensor] = []   # spatial attn maps (G, G) per slice

        for k in range(K):
            sl  = slices[k:k+1]                              # (1, 1, H, W)
            h_k, a_k = self.slice_enc(sl)                    # (1, D), (1, G, G)
            h_list.append(h_k.squeeze(0))                    # (D,)
            alpha_list.append(a_k.squeeze(0))                # (G, G)

        H_mat = torch.stack(h_list, dim=0)   # (K, D)

        # Add sinusoidal Z positional encoding
        pe    = self.pos_enc(z_indices)       # (K, D)
        H_mat = H_mat + pe

        # Mask invalid slices (set to zero before attention)
        valid_f = valid.float().unsqueeze(-1)   # (K, 1)
        H_mat   = H_mat * valid_f

        # Gated volume-level attention → β weights
        H_T  = H_mat.unsqueeze(0)                       # (1, K, D)
        beta = self.vol_attn(H_T)[1].squeeze(0)         # (K,)

        # Zero-out invalid slices in beta
        beta = beta * valid.float()
        beta = beta / (beta.sum() + 1e-8)

        # Weighted sum → volume embedding → logit
        vol_emb = (beta.unsqueeze(-1) * H_mat).sum(dim=0)   # (D,)
        logit   = self.head(vol_emb.unsqueeze(0)).squeeze()  # scalar

        if return_attn:
            return logit, (beta, alpha_list)
        return logit


def build_vit_volume_classifier(
    K: int = 16, token_size: int = 32, embed_dim: int = 256,
    depth: int = 4, num_heads: int = 8, target_size: int = 256,
    attn_dim: int = 128, dropout: float = 0.25,
) -> ViTVolumeClassifier:
    return ViTVolumeClassifier(
        K=K, token_size=token_size, embed_dim=embed_dim,
        depth=depth, num_heads=num_heads, target_size=target_size,
        attn_dim=attn_dim, dropout=dropout,
    )
