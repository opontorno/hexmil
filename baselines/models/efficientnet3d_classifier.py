"""
efficientnet3d_classifier.py
============================
EfficientNet 3D baseline for CT volume binary classification.
Built on MONAI EfficientNetBN with single-channel input support and GradCAM.

Usage:
    from baselines.models.efficientnet3d_classifier import (
        EfficientNet3DClassifier,
        build_efficientnet3d_classifier,
    )
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientNet3DClassifier(nn.Module):
    """
    EfficientNet-BN adapted for single-channel CT volume binary classification.

    Args:
        model_name: EfficientNet variant, e.g. 'efficientnet-b0'.
        pretrained: use MONAI pretrained initialization when available.
        dropout: dropout rate before the final linear layer.
    """

    def __init__(
        self,
        model_name: str = "efficientnet-b0",
        pretrained: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()

        try:
            from monai.networks.nets import EfficientNetBN
        except ImportError as exc:
            raise ImportError(
                "EfficientNet3DClassifier requires MONAI. Install it with `pip install monai`."
            ) from exc

        base = EfficientNetBN(
            model_name=model_name,
            pretrained=pretrained,
            spatial_dims=3,
            in_channels=3,
            num_classes=1,
        )

        # Keep temporal resolution for K=16 windows by avoiding Z downsampling.
        for mod in base.modules():
            if isinstance(mod, nn.Conv3d):
                if len(mod.stride) == 3 and mod.stride[0] != 1:
                    mod.stride = (1, mod.stride[1], mod.stride[2])
            elif isinstance(mod, (nn.MaxPool3d, nn.AvgPool3d)):
                stride = mod.stride if isinstance(mod.stride, tuple) else (mod.stride,) * 3
                kernel = mod.kernel_size if isinstance(mod.kernel_size, tuple) else (mod.kernel_size,) * 3
                padding = mod.padding if isinstance(mod.padding, tuple) else (mod.padding,) * 3
                mod.stride = (1, stride[1], stride[2])
                mod.kernel_size = (1, kernel[1], kernel[2])
                mod.padding = (0, padding[1], padding[2])

        # Keep classifier output shape (B, 1) and use user-defined dropout.
        base._dropout = nn.Dropout(dropout)
        self.model = base

        # For GradCAM hook
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self.model._blocks[-1].register_forward_hook(self._save_activation)
        self.model._blocks[-1].register_full_backward_hook(self._save_gradient)

    @staticmethod
    def _as_tensor(out) -> torch.Tensor:
        if isinstance(out, torch.Tensor):
            return out
        if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
            return out[0]
        raise RuntimeError("GradCAM hook did not receive a tensor output.")

    def _save_activation(self, module, inp, out):
        self._activations = self._as_tensor(out).detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self._gradients = self._as_tensor(grad_out).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, K, H, W) single-channel 3D volume
        Returns:
            logit: (B, 1)
        """
        return self.model(x)

    def gradcam_3d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute GradCAM attention map for a single volume.

        Args:
            x: (1, 1, K, H, W)
        Returns:
            cam: (K, H, W) float32, normalized to [0, 1]
        """
        x = x.requires_grad_(True)
        logit = self.forward(x)
        self.zero_grad()
        logit.sum().backward()

        if self._gradients is None or self._activations is None:
            raise RuntimeError("GradCAM hooks did not capture activations/gradients.")

        grads = self._gradients
        acts = self._activations

        weights = grads.mean(dim=(2, 3, 4), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        k, h, w = x.shape[2], x.shape[3], x.shape[4]
        cam = F.interpolate(cam, size=(k, h, w), mode="trilinear", align_corners=False)
        cam = cam.squeeze()
        lo, hi = cam.min(), cam.max()
        cam = (cam - lo) / (hi - lo + 1e-8)
        return cam.detach().cpu()


def build_efficientnet3d_classifier(
    model_name: str = "efficientnet-b0",
    pretrained: bool = False,
    dropout: float = 0.3,
) -> EfficientNet3DClassifier:
    return EfficientNet3DClassifier(
        model_name=model_name,
        pretrained=pretrained,
        dropout=dropout,
    )
