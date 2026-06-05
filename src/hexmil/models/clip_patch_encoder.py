"""
clip_patch_encoder.py
---------------------
Stage 1 — OpenAI CLIP ViT-B/16 visual encoder used as a patch-level
feature extractor inside ABMILSliceClassifier.

The CLIP projection head (768→512, trained for contrastive image-text
alignment) is removed; the raw 768-dim CLS token from the ViT backbone
is used instead, which is richer for discriminative fine-tuning.

Preprocessing pipeline applied inside encode_patch():
  (B, 1, P, P)
    → channel repeat   → (B, 3, P, P)
    → bilinear resize  → (B, 3, 224, 224)   [CLIP fixed input size]
    → CLIP normalise   → mean=(0.4815,0.4578,0.4082), std=(0.2686,0.2613,0.2758)

Requirements:
    pip install git+https://github.com/openai/CLIP.git

Usage:
    from hexmil.models.clip_patch_encoder import CLIPPatchEncoder
    enc = CLIPPatchEncoder()
    feat = enc.encode_patch(x)   # x: (B, 1, P, P)  → (B, 768)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPPatchEncoder(nn.Module):
    """
    OpenAI CLIP ViT-B/16 visual encoder adapted for single-channel CT patch
    encoding.

    The contrastive projection head is discarded so that the raw 768-dim CLS
    token is exposed — this preserves the full representational capacity of the
    ViT backbone for discriminative fine-tuning within the ABMIL pipeline.

    All parameters are trainable by default (fine-tuned end-to-end together
    with the ABMIL projector, aggregator, and head).
    """

    FEAT_DIM: int = 768   # ViT-B/16 width, before CLIP's contrastive projection

    # CLIP normalisation constants (ImageNet-derived, used by OpenAI CLIP)
    _CLIP_MEAN = (0.48145466, 0.4578275,  0.40821073)
    _CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self):
        super().__init__()
        try:
            import clip as _clip
        except ImportError as exc:
            raise ImportError(
                "CLIPPatchEncoder requires the openai/clip package.\n"
                "Install with:  pip install git+https://github.com/openai/CLIP.git"
            ) from exc

        clip_model, _ = _clip.load("ViT-B/16", device="cpu")

        # Keep only the visual backbone; discard the text encoder and the
        # contrastive projection head (visual.proj) to expose raw 768-dim
        # CLS token features.
        self.visual = clip_model.visual
        self.visual.proj = None   # forward() now returns ln_post(CLS) ∈ R^768

        # Register as non-trainable buffers so they move with .to(device)
        self.register_buffer(
            "clip_mean",
            torch.tensor(self._CLIP_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "clip_std",
            torch.tensor(self._CLIP_STD,  dtype=torch.float32).view(1, 3, 1, 1),
        )

    # ------------------------------------------------------------------

    @property
    def feat_dim(self) -> int:
        return self.FEAT_DIM

    # ------------------------------------------------------------------

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prepare a batch of single-channel CT patches for CLIP.

        Args:
            x: (B, 1, P, P)  float32  values in [0, 1]
        Returns:
            (B, 3, 224, 224)  normalised for CLIP
        """
        x = x.repeat(1, 3, 1, 1)                               # (B, 3, P, P)
        if x.shape[2] != 224 or x.shape[3] != 224:
            x = F.interpolate(
                x, size=(224, 224),
                mode="bilinear", align_corners=False,
            )
        x = (x - self.clip_mean) / self.clip_std               # CLIP normalise
        return x

    # ------------------------------------------------------------------

    def encode_patch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract the 768-dim CLS token feature for a batch of patches.
        Called by ABMILSliceClassifier.encode_patches().

        Args:
            x: (B, 1, P, P)
        Returns:
            (B, 768)
        """
        return self.visual(self._preprocess(x))

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> torch.Tensor:
        """Alias of encode_patch; returns (B, 768) CLS token."""
        return self.encode_patch(x)


def build_clip_encoder() -> CLIPPatchEncoder:
    """Factory function — returns a CLIPPatchEncoder ready to be embedded in ABMILSliceClassifier."""
    return CLIPPatchEncoder()
