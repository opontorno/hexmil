"""
densenet3d_classifier.py
========================
DenseNet121 3D baseline for CT volume binary classification.
Built on MONAI DenseNet121 with single-channel input support and GradCAM.

Usage:
    from baselines.models.densenet3d_classifier import (
        DenseNet3DClassifier,
        build_densenet3d_classifier,
    )
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseNet3DClassifier(nn.Module):
    """
    DenseNet121 3D adapted for single-channel CT volume binary classification.

    Args:
        pretrained: MONAI pretrained weights flag. Not supported for 3D DenseNet.
        dropout: dropout rate before the final linear layer.
    """

    def __init__(self, pretrained: bool = False, dropout: float = 0.3):
        super().__init__()

        try:
            from monai.networks.nets import DenseNet121
        except ImportError as exc:
            raise ImportError(
                "DenseNet3DClassifier requires MONAI. Install it with `pip install monai`."
            ) from exc

        if pretrained:
            raise ValueError(
                "MONAI DenseNet121 does not provide pretrained weights for spatial_dims=3. "
                "Use pretrained=False."
            )

        base = DenseNet121(
            spatial_dims=3,
            in_channels=3,
            out_channels=1,
            pretrained=False,
        )

        # Keep temporal resolution for K=16 windows by downsampling only in H/W.
        conv0 = base.features.conv0
        base.features.conv0 = nn.Conv3d(
            conv0.in_channels,
            conv0.out_channels,
            kernel_size=conv0.kernel_size,
            stride=(1, conv0.stride[1], conv0.stride[2]),
            padding=conv0.padding,
            bias=False,
        )
        with torch.no_grad():
            base.features.conv0.weight.copy_(conv0.weight)

        base.features.pool0 = nn.MaxPool3d(
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )

        for trans in (base.features.transition1, base.features.transition2, base.features.transition3):
            trans.pool = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)

        # Insert dropout in classifier head while keeping DenseNet121 output shape (B, 1).
        out_layer = base.class_layers.out
        base.class_layers = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(output_size=1),
            nn.Flatten(start_dim=1),
            nn.Dropout(dropout),
            nn.Linear(out_layer.in_features, 1),
        )
        self.model = base

        # For GradCAM hook
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self.model.features.denseblock4.register_forward_hook(self._save_activation)
        self.model.features.denseblock4.register_full_backward_hook(self._save_gradient)

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


def build_densenet3d_classifier(
    pretrained: bool = False,
    dropout: float = 0.3,
) -> DenseNet3DClassifier:
    return DenseNet3DClassifier(pretrained=pretrained, dropout=dropout)
