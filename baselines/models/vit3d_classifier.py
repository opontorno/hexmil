from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


ARCH_CONFIGS: dict[str, dict] = {
    'vit3d_tiny':  dict(embed_dim=192, depth=12, num_heads=3),
    'vit3d_small': dict(embed_dim=384, depth=12, num_heads=6),
    'vit3d_base':  dict(embed_dim=768, depth=12, num_heads=12),
}

_TIMM_MODELS: dict[str, str] = {
    'vit3d_tiny':  'vit_tiny_patch16_224',
    'vit3d_small': 'vit_small_patch16_224',
    'vit3d_base':  'vit_base_patch16_224',
}


class PatchEmbed3D(nn.Module):

    def __init__(
        self,
        K: int = 16,
        img_size: int = 224,
        patch_t: int = 2,
        patch_s: int = 32,
        in_chans: int = 3,
        embed_dim: int = 384,
    ):
        super().__init__()
        self.n_t = K // patch_t
        self.n_s_sqrt = img_size // patch_s
        self.n_s = self.n_s_sqrt ** 2
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(patch_t, patch_s, patch_s),
            stride=(patch_t, patch_s, patch_s),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                          # (B, D, n_t, n_h, n_w)
        B, D, nt, nh, nw = x.shape
        return x.flatten(2).transpose(1, 2)       # (B, N, D)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, dropout: float = 0.0):
        super().__init__()
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2  = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, f'dim={dim} must be divisible by num_heads={num_heads}'
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv       = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        attn = self.attn_drop((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = Attention(dim, num_heads, attn_drop, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = Mlp(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT3DClassifier(nn.Module):

    def __init__(
        self,
        arch: str = 'vit3d_base',
        K: int = 16,
        img_size: int = 224,
        patch_t: int = 2,
        patch_s: int = 32,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_drop: float = 0.0,
        pretrained: bool = False,
    ):
        super().__init__()
        self.arch = arch

        self.patch_embed = PatchEmbed3D(K, img_size, patch_t, patch_s, in_chans, embed_dim)
        n_tokens = self.patch_embed.n_t * self.patch_embed.n_s

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens + 1, embed_dim))
        self.pos_drop  = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, dropout, attn_drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

        self._init_weights()

        if pretrained:
            self._inflate_from_2d(arch, patch_t, patch_s)

        # GradCAM hooks on the last transformer block
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self.blocks[-1].register_forward_hook(self._save_activation)
        self.blocks[-1].register_full_backward_hook(self._save_gradient)


    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _inflate_from_2d(self, arch: str, patch_t: int, patch_s: int):
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                'pretrained=True requires timm. Install with: pip install timm'
            ) from exc

        timm_name = _TIMM_MODELS[arch]
        src_sd = timm.create_model(timm_name, pretrained=True).state_dict()

        def _copy(key: str, dst: nn.Parameter | torch.Tensor):
            if key in src_sd:
                dst.data.copy_(src_sd[key])

        with torch.no_grad():
            w2d = src_sd['patch_embed.proj.weight']    # (D, 3, 16, 16)
            D = w2d.shape[0]

            if patch_s != 16:
                w2d_rs = w2d.reshape(D * 3, 1, 16, 16)
                w2d = F.interpolate(
                    w2d_rs, size=(patch_s, patch_s),
                    mode='bilinear', align_corners=False,
                ).reshape(D, 3, patch_s, patch_s)

            # Tile along temporal dim and scale to preserve pre-activation magnitude
            w3d = w2d.unsqueeze(2).repeat(1, 1, patch_t, 1, 1) / patch_t
            self.patch_embed.proj.weight.copy_(w3d)
            _copy('patch_embed.proj.bias', self.patch_embed.proj.bias)

            _copy('cls_token', self.cls_token)

            src_pe   = src_sd['pos_embed']             # (1, 197, D)
            cls_pe   = src_pe[:, :1, :]                # (1, 1, D)
            patch_pe = src_pe[:, 1:, :]                # (1, 196, D)

            n_s     = self.patch_embed.n_s
            n_t     = self.patch_embed.n_t
            src_gs  = int(math.sqrt(patch_pe.shape[1]))  # typically 14
            dst_gs  = self.patch_embed.n_s_sqrt

            if src_gs != dst_gs:
                patch_pe = (
                    patch_pe.reshape(1, src_gs, src_gs, D)
                    .permute(0, 3, 1, 2)                  # (1, D, gs, gs)
                )
                patch_pe = F.interpolate(
                    patch_pe, size=(dst_gs, dst_gs),
                    mode='bilinear', align_corners=False,
                )
                patch_pe = (
                    patch_pe.permute(0, 2, 3, 1)
                    .reshape(1, n_s, D)
                )

            # Tile spatial PE across temporal positions
            patch_pe_3d = patch_pe.repeat(1, n_t, 1)  # (1, n_t*n_s, D)
            new_pe = torch.cat([cls_pe, patch_pe_3d], dim=1)
            self.pos_embed.copy_(new_pe)

            for i, blk in enumerate(self.blocks):
                p = f'blocks.{i}'
                _copy(f'{p}.norm1.weight',   blk.norm1.weight)
                _copy(f'{p}.norm1.bias',     blk.norm1.bias)
                _copy(f'{p}.norm2.weight',   blk.norm2.weight)
                _copy(f'{p}.norm2.bias',     blk.norm2.bias)
                _copy(f'{p}.attn.qkv.weight', blk.attn.qkv.weight)
                _copy(f'{p}.attn.qkv.bias',   blk.attn.qkv.bias)
                _copy(f'{p}.attn.proj.weight', blk.attn.proj.weight)
                _copy(f'{p}.attn.proj.bias',   blk.attn.proj.bias)
                _copy(f'{p}.mlp.fc1.weight',  blk.mlp.fc1.weight)
                _copy(f'{p}.mlp.fc1.bias',    blk.mlp.fc1.bias)
                _copy(f'{p}.mlp.fc2.weight',  blk.mlp.fc2.weight)
                _copy(f'{p}.mlp.fc2.bias',    blk.mlp.fc2.bias)

            _copy('norm.weight', self.norm.weight)
            _copy('norm.bias',   self.norm.bias)

        print(f'  Inflated 2D weights from {timm_name}')


    def _save_activation(self, module, inp, out):
        self._activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x)                       # (B, N, D)
        B = tokens.shape[0]
        cls = self.cls_token.expand(B, -1, -1)             # (B, 1, D)
        tokens = torch.cat([cls, tokens], dim=1)           # (B, N+1, D)
        tokens = self.pos_drop(tokens + self.pos_embed)

        for blk in self.blocks:
            tokens = blk(tokens)

        tokens = self.norm(tokens)
        return self.head(tokens[:, 0])                     # CLS token -> (B, 1)


    def gradcam_3d(self, x: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True)
        logit = self.forward(x)
        self.zero_grad()
        logit.sum().backward()

        if self._gradients is None or self._activations is None:
            raise RuntimeError('GradCAM hooks did not capture activations/gradients.')

        acts  = self._activations[:, 1:, :]    # (1, N, D) — skip CLS
        grads = self._gradients[:, 1:, :]      # (1, N, D)

        weights = grads.mean(dim=-1, keepdim=True)         # (1, N, 1)
        cam = F.relu((weights * acts).sum(dim=-1))         # (1, N)

        n_t = self.patch_embed.n_t
        n_s = self.patch_embed.n_s_sqrt
        cam = cam.reshape(1, 1, n_t, n_s, n_s)            # (1, 1, n_t, n_h, n_w)

        k, h, w = x.shape[2], x.shape[3], x.shape[4]
        cam = F.interpolate(cam, size=(k, h, w), mode='trilinear', align_corners=False)
        cam = cam.squeeze()
        lo, hi = cam.min(), cam.max()
        return ((cam - lo) / (hi - lo + 1e-8)).detach().cpu()


# ViViT Factorised Dot-Product Attention (ViViT-F)

class FactorisedBlock(nn.Module):

    def __init__(
        self,
        n_t: int,
        n_s: int,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.n_t = n_t
        self.n_s = n_s

        self.norm1_s = nn.LayerNorm(dim)
        self.attn_s  = Attention(dim, num_heads, attn_drop, dropout)
        self.norm1_t = nn.LayerNorm(dim)
        self.attn_t  = Attention(dim, num_heads, attn_drop, dropout)
        self.norm2   = nn.LayerNorm(dim)
        self.mlp     = Mlp(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)  where  N = n_t * n_s
        B, N, D = x.shape
        n_t, n_s = self.n_t, self.n_s

        # Spatial attention: within each temporal frame
        x_s = x.reshape(B * n_t, n_s, D)
        x_s = x_s + self.attn_s(self.norm1_s(x_s))
        x   = x_s.reshape(B, N, D)

        # Temporal attention: across frames at each spatial position
        x_t = x.reshape(B, n_t, n_s, D).permute(0, 2, 1, 3).reshape(B * n_s, n_t, D)
        x_t = x_t + self.attn_t(self.norm1_t(x_t))
        x   = x_t.reshape(B, n_s, n_t, D).permute(0, 2, 1, 3).reshape(B, N, D)

        x = x + self.mlp(self.norm2(x))
        return x


class ViViTFactorised3DClassifier(nn.Module):

    def __init__(
        self,
        arch: str = 'vit3d_base',
        K: int = 16,
        img_size: int = 224,
        patch_t: int = 2,
        patch_s: int = 32,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_drop: float = 0.0,
        pretrained: bool = False,
    ):
        super().__init__()
        self.arch = arch

        self.patch_embed = PatchEmbed3D(K, img_size, patch_t, patch_s, in_chans, embed_dim)
        n_t = self.patch_embed.n_t
        n_s = self.patch_embed.n_s

        # Factorised positional encoding (additive spatial + temporal)
        self.spatial_pe  = nn.Parameter(torch.zeros(1, n_s, embed_dim))
        self.temporal_pe = nn.Parameter(torch.zeros(1, n_t, embed_dim))
        self.pos_drop    = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            FactorisedBlock(n_t, n_s, embed_dim, num_heads, mlp_ratio, dropout, attn_drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

        self._init_weights()

        if pretrained:
            self._inflate_from_2d(arch, patch_t, patch_s)

        # GradCAM hooks on last block
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self.blocks[-1].register_forward_hook(self._save_activation)
        self.blocks[-1].register_full_backward_hook(self._save_gradient)

    def _init_weights(self):
        nn.init.trunc_normal_(self.spatial_pe,  std=0.02)
        nn.init.trunc_normal_(self.temporal_pe, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _inflate_from_2d(self, arch: str, patch_t: int, patch_s: int):
        try:
            import timm
        except ImportError as exc:
            raise ImportError('pretrained=True requires timm: pip install timm') from exc

        timm_name = _TIMM_MODELS[arch]
        src_sd = timm.create_model(timm_name, pretrained=True).state_dict()

        def _copy(key: str, dst):
            if key in src_sd:
                dst.data.copy_(src_sd[key])

        with torch.no_grad():
            w2d = src_sd['patch_embed.proj.weight']
            D = w2d.shape[0]
            if patch_s != 16:
                w2d = F.interpolate(
                    w2d.reshape(D * 3, 1, 16, 16), size=(patch_s, patch_s),
                    mode='bilinear', align_corners=False,
                ).reshape(D, 3, patch_s, patch_s)
            w3d = w2d.unsqueeze(2).repeat(1, 1, patch_t, 1, 1) / patch_t
            self.patch_embed.proj.weight.copy_(w3d)
            _copy('patch_embed.proj.bias', self.patch_embed.proj.bias)

            # Spatial PE (strip CLS, interpolate grid if needed)
            src_pe  = src_sd['pos_embed']               # (1, 197, D)
            patch_pe = src_pe[:, 1:, :]                 # (1, 196, D)
            src_gs  = int(math.sqrt(patch_pe.shape[1]))
            dst_gs  = self.patch_embed.n_s_sqrt
            n_s     = self.patch_embed.n_s
            if src_gs != dst_gs:
                patch_pe = (patch_pe.reshape(1, src_gs, src_gs, D)
                            .permute(0, 3, 1, 2))
                patch_pe = F.interpolate(
                    patch_pe, size=(dst_gs, dst_gs),
                    mode='bilinear', align_corners=False,
                )
                patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, n_s, D)
            self.spatial_pe.copy_(patch_pe)
            # temporal_pe: no 2D equivalent, stays randomly initialised

            # Transformer blocks — spatial + temporal attention both init from 2D ViT
            for i, blk in enumerate(self.blocks):
                p = f'blocks.{i}'
                _copy(f'{p}.norm1.weight',    blk.norm1_s.weight)
                _copy(f'{p}.norm1.bias',      blk.norm1_s.bias)
                _copy(f'{p}.attn.qkv.weight', blk.attn_s.qkv.weight)
                _copy(f'{p}.attn.qkv.bias',   blk.attn_s.qkv.bias)
                _copy(f'{p}.attn.proj.weight', blk.attn_s.proj.weight)
                _copy(f'{p}.attn.proj.bias',   blk.attn_s.proj.bias)
                # Temporal attention (factorised space-time init: copy same weights)
                _copy(f'{p}.norm1.weight',    blk.norm1_t.weight)
                _copy(f'{p}.norm1.bias',      blk.norm1_t.bias)
                _copy(f'{p}.attn.qkv.weight', blk.attn_t.qkv.weight)
                _copy(f'{p}.attn.qkv.bias',   blk.attn_t.qkv.bias)
                _copy(f'{p}.attn.proj.weight', blk.attn_t.proj.weight)
                _copy(f'{p}.attn.proj.bias',   blk.attn_t.proj.bias)
                _copy(f'{p}.norm2.weight',   blk.norm2.weight)
                _copy(f'{p}.norm2.bias',     blk.norm2.bias)
                _copy(f'{p}.mlp.fc1.weight', blk.mlp.fc1.weight)
                _copy(f'{p}.mlp.fc1.bias',   blk.mlp.fc1.bias)
                _copy(f'{p}.mlp.fc2.weight', blk.mlp.fc2.weight)
                _copy(f'{p}.mlp.fc2.bias',   blk.mlp.fc2.bias)

            _copy('norm.weight', self.norm.weight)
            _copy('norm.bias',   self.norm.bias)

        print(f'  Inflated 2D weights from {timm_name} (factorised space-time init)')

    def _save_activation(self, module, inp, out):
        self._activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x)               # (B, N, D)
        B, N, D = tokens.shape
        n_t = self.patch_embed.n_t
        n_s = self.patch_embed.n_s

        # Additive factorised PE: spatial PE tiled across n_t, temporal PE across n_s
        sp = self.spatial_pe.unsqueeze(1).expand(1, n_t, n_s, D).reshape(1, N, D)
        tp = self.temporal_pe.unsqueeze(2).expand(1, n_t, n_s, D).reshape(1, N, D)
        tokens = self.pos_drop(tokens + sp + tp)

        for blk in self.blocks:
            tokens = blk(tokens)

        tokens = self.norm(tokens)
        return self.head(tokens.mean(dim=1))       # global avg pool → (B, 1)

    def gradcam_3d(self, x: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True)
        logit = self.forward(x)
        self.zero_grad()
        logit.sum().backward()

        if self._gradients is None or self._activations is None:
            raise RuntimeError('GradCAM hooks did not capture activations/gradients.')

        acts  = self._activations
        grads = self._gradients
        weights = grads.mean(dim=-1, keepdim=True)
        cam = F.relu((weights * acts).sum(dim=-1))         # (1, N)

        n_t = self.patch_embed.n_t
        n_s = self.patch_embed.n_s_sqrt
        cam = cam.reshape(1, 1, n_t, n_s, n_s)
        k, h, w = x.shape[2], x.shape[3], x.shape[4]
        cam = F.interpolate(cam, size=(k, h, w), mode='trilinear', align_corners=False)
        cam = cam.squeeze()
        lo, hi = cam.min(), cam.max()
        return ((cam - lo) / (hi - lo + 1e-8)).detach().cpu()


def build_vivit_factorised_classifier(
    arch: str = 'vit3d_base',
    K: int = 16,
    img_size: int = 224,
    patch_t: int = 2,
    patch_s: int = 32,
    dropout: float = 0.1,
    attn_drop: float = 0.0,
    pretrained: bool = False,
) -> ViViTFactorised3DClassifier:
    if arch not in ARCH_CONFIGS:
        raise ValueError(f'Unknown arch "{arch}". Choose from: {list(ARCH_CONFIGS)}')
    cfg = ARCH_CONFIGS[arch]
    return ViViTFactorised3DClassifier(
        arch=arch, K=K, img_size=img_size,
        patch_t=patch_t, patch_s=patch_s,
        in_chans=3, dropout=dropout, attn_drop=attn_drop,
        pretrained=pretrained, **cfg,
    )


# Factory — plain ViT3D

def build_vit3d_classifier(
    arch: str = 'vit3d_base',
    K: int = 16,
    img_size: int = 224,
    patch_t: int = 2,
    patch_s: int = 32,
    dropout: float = 0.1,
    attn_drop: float = 0.0,
    pretrained: bool = False,
) -> ViT3DClassifier:
    if arch not in ARCH_CONFIGS:
        raise ValueError(f'Unknown arch "{arch}". Choose from: {list(ARCH_CONFIGS)}')
    cfg = ARCH_CONFIGS[arch]
    return ViT3DClassifier(
        arch=arch,
        K=K,
        img_size=img_size,
        patch_t=patch_t,
        patch_s=patch_s,
        in_chans=3,
        dropout=dropout,
        attn_drop=attn_drop,
        pretrained=pretrained,
        **cfg,
    )
