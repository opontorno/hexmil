from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class R3DClassifier(nn.Module):
    def __init__(self, arch: str = 'r3d_18', pretrained: bool = False, dropout: float = 0.3):
        super().__init__()
        import torchvision.models.video as vm

        if arch == 'r3d_18':
            base = vm.r3d_18(weights='DEFAULT' if pretrained else None)
        elif arch == 'mc3_18':
            base = vm.mc3_18(weights='DEFAULT' if pretrained else None)
        else:
            raise ValueError(f"Unknown arch: {arch}. Use 'r3d_18' or 'mc3_18'")

        # Keep original 3-ch stem — 1-ch CT slices are repeated to 3-ch in the training loop

        # Keep all layers except final FC
        self.stem     = base.stem
        self.layer1   = base.layer1
        self.layer2   = base.layer2
        self.layer3   = base.layer3
        self.layer4   = base.layer4
        self.avgpool  = base.avgpool

        feat_dim = base.fc.in_features   # 512
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 1),
        )

        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self.layer4[-1].register_forward_hook(self._save_activation)
        self.layer4[-1].register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self._activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def gradcam_3d(self, x: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True)
        logit = self.forward(x)           # (1,1)
        self.zero_grad()
        logit.sum().backward()

        grads = self._gradients           # (1, C, k, h, w)
        acts  = self._activations         # (1, C, k, h, w)

        weights = grads.mean(dim=(2,3,4), keepdim=True)   # (1, C, 1, 1, 1)
        cam = (weights * acts).sum(dim=1, keepdim=True)    # (1, 1, k, h, w)
        cam = F.relu(cam)

        K, H, W = x.shape[2], x.shape[3], x.shape[4]
        cam = F.interpolate(cam, size=(K, H, W), mode='trilinear', align_corners=False)
        cam = cam.squeeze()                                 # (K, H, W)
        lo, hi = cam.min(), cam.max()
        cam = (cam - lo) / (hi - lo + 1e-8)
        return cam.detach().cpu()


def build_r3d_classifier(arch: str = 'r3d_18', pretrained: bool = False,
                          dropout: float = 0.3) -> R3DClassifier:
    return R3DClassifier(arch=arch, pretrained=pretrained, dropout=dropout)
