"""
vit_patch_classifier.py
-----------------------
Phase A — ViT-style patch-tokenised classifier.

Takes a single-channel 2D patch (1, H, W) centred on the nodule, divides it
into a grid of non-overlapping sub-patches (tokens), and uses a Vision Transformer
to classify it as real (0) or fake (1).

The CLS-token attention (averaged across heads in the last layer) naturally produces
a **spatial attention map** over the sub-patch grid, showing which spatial tokens
the network found most relevant — this is the ViT-based XAI signal.

Architecture:
    ┌────────────────────────────────────────┐
    │  Patch Embedding                        │  input (B, 1, H, W)
    │  (Conv2d stride=token_size)             │  → (B, N, D)   N = (H/ts)*(W/ts)
    └────────────────────────────────────────┘
                │
    ┌───────────▼────────────────────────────┐
    │  [CLS] token + Positional Embedding     │  → (B, N+1, D)
    └────────────────────────────────────────┘
                │
    ┌───────────▼────────────────────────────┐
    │  Transformer Encoder                    │  L layers, H heads
    │  (self-attention + FFN)                 │
    └────────────────────────────────────────┘
                │
    ┌───────────▼────────────────────────────┐
    │  CLS token output                       │  → (B, D)
    │  + attention weights for XAI            │
    └────────────────────────────────────────┘
                │
    ┌───────────▼────────────────────────────┐
    │  Classification Head                    │  → logit ∈ (B, 1)
    │  (LayerNorm → FC)                       │
    └────────────────────────────────────────┘

Also supports loading ViT-Small/Base from timm with in_chans=1.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
#  Custom lightweight ViT (from scratch — no pre-training)
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Split image into non-overlapping patches and linearly embed them."""

    def __init__(self, img_size: int = 128, token_size: int = 16,
                 in_chans: int = 1, embed_dim: int = 384):
        super().__init__()
        self.num_patches = (img_size // token_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=token_size, stride=token_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)  →  (B, N, D)
        return self.proj(x).flatten(2).transpose(1, 2)


class TransformerBlock(nn.Module):
    """Standard pre-norm Transformer block with multi-head self-attention."""

    def __init__(self, dim: int, num_heads: int = 6, mlp_ratio: float = 4.0,
                 drop: float = 0.1, attn_drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads,
                                           dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor, need_weights: bool = False):
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(x_norm, x_norm, x_norm,
                                            need_weights=need_weights)
        x = x + attn_out
        # FFN with residual
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights


class ViTPatchClassifier(nn.Module):
    """
    Custom ViT for binary patch classification.

    Args:
        img_size:     spatial size of the input patch (square).
        token_size:   size of each non-overlapping sub-patch (token).
        in_chans:     input channels (1 for grayscale).
        embed_dim:    transformer embedding dimension.
        depth:        number of transformer layers.
        num_heads:    number of attention heads.
        mlp_ratio:    expansion ratio of the FFN.
        drop_rate:    dropout rate.
    """

    def __init__(
        self,
        img_size: int = 128,
        token_size: int = 16,
        in_chans: int = 1,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.img_size   = img_size
        self.token_size = token_size
        self.grid_size  = img_size // token_size      # e.g. 128 / 16 = 8

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, token_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS token + positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop  = nn.Dropout(drop_rate)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        """
        Args:
            x:           (B, 1, H, W) input patch
            return_attn: if True, return spatial attention map from last layer

        Returns:
            logits: (B, 1)
            attn:   (B, 1, H, W) attention map upsampled to input size (if requested)
        """
        B = x.shape[0]

        # Tokenise
        tokens = self.patch_embed(x)                        # (B, N, D)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)              # (B, 1, D)
        tokens = torch.cat([cls, tokens], dim=1)            # (B, N+1, D)
        tokens = self.pos_drop(tokens + self.pos_embed)

        # Transformer
        attn_weights = None
        for blk in self.blocks:
            tokens, attn_weights = blk(tokens, need_weights=return_attn)

        tokens = self.norm(tokens)

        # CLS output → classification
        cls_out = tokens[:, 0]                              # (B, D)
        logits  = self.head(cls_out)                        # (B, 1)

        if return_attn and attn_weights is not None:
            # attn_weights: (B, N+1, N+1) — take CLS row, skip CLS col
            cls_attn = attn_weights[:, 0, 1:]               # (B, N)
            # Reshape to spatial grid
            g = self.grid_size
            attn_map = cls_attn.reshape(B, 1, g, g)         # (B, 1, g, g)
            # Upsample to input resolution
            attn_map = F.interpolate(attn_map, size=(self.img_size, self.img_size),
                                     mode='bilinear', align_corners=False)
            return logits, attn_map

        return logits


def build_vit_classifier(
    img_size: int = 128,
    token_size: int = 16,
    embed_dim: int = 384,
    depth: int = 6,
    num_heads: int = 6,
) -> ViTPatchClassifier:
    """Factory function."""
    return ViTPatchClassifier(
        img_size=img_size,
        token_size=token_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
    )
