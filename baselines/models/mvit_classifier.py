from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

VALID_ARCHS = ('mvit_v1_b', 'mvit_v2_s', 'mvit_v2_b')


class MViTClassifier(nn.Module):

    def __init__(
        self,
        arch: str = 'mvit_v2_s',
        pretrained: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()
        if arch not in VALID_ARCHS:
            raise ValueError(f'Unknown arch "{arch}". Choose from: {VALID_ARCHS}')

        import torchvision.models.video as vm

        weights_map = {
            'mvit_v1_b': 'DEFAULT',
            'mvit_v2_s': 'DEFAULT',
            'mvit_v2_b': 'DEFAULT',
        }
        weights = weights_map[arch] if pretrained else None

        if arch == 'mvit_v1_b':
            base = vm.mvit_v1_b(weights=weights)
        elif arch == 'mvit_v2_s':
            base = vm.mvit_v2_s(weights=weights)
        else:
            base = vm.mvit_v2_b(weights=weights)

        # Replace classifier head with binary head
        # torchvision MViT head: nn.Sequential(nn.Dropout(...), nn.Linear(D, 400))
        in_features = base.head[-1].in_features
        base.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1),
        )
        self.model = base

        # GradCAM hook on the final normalisation layer before the head
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        # torchvision MViT exposes norm as a plain LayerNorm (not Sequential)
        norm_layer = base.norm if isinstance(base.norm, nn.Module) else list(base.norm.children())[-1]
        norm_layer.register_forward_hook(self._save_activation)
        norm_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self._activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def gradcam_3d(self, x: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True)
        logit = self.forward(x)
        self.zero_grad()
        logit.sum().backward()

        if self._gradients is None or self._activations is None:
            raise RuntimeError('GradCAM hooks did not capture activations/gradients.')

        acts  = self._activations    # (1, D) or (1, N, D) depending on MViT version
        grads = self._gradients

        if acts.dim() == 2:
            # Global average pooled output — spatial CAM not meaningful; return uniform
            cam = torch.ones(x.shape[2], x.shape[3], x.shape[4])
            return cam / cam.max()

        # (1, N, D): pool over embed dim, weighted by gradients
        weights = grads.mean(dim=-1, keepdim=True)              # (1, N, 1)
        cam_flat = F.relu((weights * acts).sum(dim=-1))         # (1, N)

        # Infer spatial/temporal grid size from N
        k, h, w = x.shape[2], x.shape[3], x.shape[4]
        N = cam_flat.shape[1]
        cam_flat = cam_flat.reshape(1, 1, N, 1, 1)
        cam = F.interpolate(
            cam_flat.expand(1, 1, N, 1, 1),
            size=(k, h, w), mode='nearest',
        )
        # Better: reshape to closest factor trio and interpolate
        import math
        t_stride = 2  # MViT typically downsamples T by 2x
        n_t = max(1, k // t_stride)
        n_s = max(1, N // n_t)
        n_s_sq = int(math.sqrt(n_s))
        if n_t * n_s_sq * n_s_sq == N:
            cam = cam_flat.squeeze().reshape(1, 1, n_t, n_s_sq, n_s_sq)
            cam = F.interpolate(cam, size=(k, h, w), mode='trilinear', align_corners=False)
        else:
            cam = cam_flat.reshape(1, 1, -1, 1, 1).expand(1, 1, k, h, w).float()

        cam = cam.squeeze()
        lo, hi = cam.min(), cam.max()
        return ((cam - lo) / (hi - lo + 1e-8)).detach().cpu()


def build_mvit_classifier(
    arch: str = 'mvit_v2_s',
    pretrained: bool = False,
    dropout: float = 0.3,
) -> MViTClassifier:
    return MViTClassifier(arch=arch, pretrained=pretrained, dropout=dropout)
