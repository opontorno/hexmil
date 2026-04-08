#!/usr/bin/env python3
"""
eval_robustness.py
==================
Post-processing robustness evaluation for all forensics models.

Perturbations applied only at test time (no re-training):

  Tier 1 — standard:
    gaussian_noise     sigma ∈ {0.02, 0.05, 0.10}
    gaussian_blur      sigma ∈ {1, 2, 3} px
    jpeg_compression   quality ∈ {90, 70, 50}

  Tier 2 — CT-specific:
    intensity_shift    delta ∈ {0.10, 0.20, 0.30}  (uniform HU offset, clipped)
    downscale          factor ∈ {0.75, 0.50, 0.25}  (zoom-out then zoom back in)

Evaluation granularity:
  Volume-level (K slices per volume): ABMIL, R3D-18, FlatCNN, ViT-ABMIL, PoolMIL
  Slice-level  (one slice at a time): HP-FCN, TruFor, MVSS-Net, ManTraNet

Usage:
    python experiments/robustness/eval_robustness.py \\
        --abmil_run_dir  PATH            \\
        [--baselines_runs_dir  PATH]     \\
        [--data_dir            PATH]     \\
        [--out_dir             PATH]     \\
        [--K                   16]       \\  # slices per volume (volume models)
        [--n_samples           500]      \\  # max fake volumes per modality; 0 = all
        [--gpu_id              0]        \\
        [--seed                42]
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# =============================================================================
#  Paths
# =============================================================================

WORK_DIR  = Path(__file__).resolve().parent.parent.parent
BASE_DIR  = WORK_DIR / 'experiments' / 'baselines'
ABMIL_DIR = WORK_DIR / 'experiments' / 'ABMIL'
MVSS_DIR  = WORK_DIR / 'baselines' / 'MVSS-Net'
sys.path.insert(0, str(WORK_DIR))
sys.path.insert(0, str(WORK_DIR / 'src'))
from config import DATA_DIR

from hexmil.data.patch_dataset import load_split_table
from hexmil.data.slice_dataset  import build_patch_grid
from hexmil.models.abmil_slice_classifier         import build_abmil_classifier_scratch
from hexmil.models.abmil_slice_classifier_fourier import build_fabmil_classifier_scratch
from hexmil.models.volume_classifier              import VolumeClassifier
from hexmil.utils.tiff_utils import (
    get_shape_tiff_scan, load_slice_tiff_scan,
    get_percentile_tiff_scan, apply_percentile,
)

ALL_FAKES = ['pix2pix', 'cycle', 'diffusion']

# =============================================================================
#  Baseline auto-discovery
# =============================================================================

# Order matters: more-specific patterns before less-specific
_BASELINE_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r'^hp_fcn_K(?P<K>\d+)$'),           'hp_fcn',       'HP-FCN'),
    (re.compile(r'^trufor_mitb2_K(?P<K>\d+)$'),     'trufor_mitb2', 'TruFor-MiT-B2'),
    (re.compile(r'^trufor_K(?P<K>\d+)$'),            'trufor',       'TruFor'),
    (re.compile(r'^mvssnet_full_K(?P<K>\d+)$'),      'mvssnet_full', 'MVSSNet-Full'),
    (re.compile(r'^mvssnet_K(?P<K>\d+)$'),            'mvssnet',      'MVSSNet'),
    (re.compile(r'^mantranet_K(?P<K>\d+)$'),          'mantranet',    'ManTraNet'),
    (re.compile(r'^r3d\d*_K(?P<K>\d+)$'),            'r3d',          'R3D-18'),
    (re.compile(r'^flat_cnn\w*_K(?P<K>\d+)$'),       'flat_cnn',     'FlatCNN'),
    (re.compile(r'^vit_abmil_K(?P<K>\d+)$'),         'vit_abmil',    'ViT-ABMIL'),
    (re.compile(r'^pool_mil_\w+_K(?P<K>\d+)$'),      'pool_mil',     'PoolMIL'),
]

def _discover_baselines(
    baselines_runs_dir: Path,
) -> list[tuple[str, str, str, Path]]:
    """Returns list of (arch_key, scorer_name, trained_on, run_dir)."""
    entries: list[tuple[str, str, str, Path]] = []
    if not baselines_runs_dir.exists():
        print(f"[WARN] Baselines runs dir not found: {baselines_runs_dir}")
        return entries
    for model_dir in sorted(baselines_runs_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        arch_key = display_name = None
        for pat, ak, dn in _BASELINE_PATTERNS:
            m = pat.match(model_dir.name)
            if m:
                arch_key = ak
                display_name = f'{dn}_K{m.group("K")}'
                break
        if arch_key is None:
            continue
        for mod in ALL_FAKES:
            trained_on_dir = model_dir / f'trained_on_{mod}'
            if trained_on_dir.is_dir() and (trained_on_dir / 'best_model.pt').exists():
                entries.append((arch_key, display_name, mod, trained_on_dir))
    return entries

def _discover_abmil(abmil_run_dir: Path) -> list[tuple[str, Path]]:
    """
    Returns list of (trained_on_label, run_dir).
    If best_model.pt is directly in abmil_run_dir → [('all', abmil_run_dir)].
    If trained_on_{mod} subdirs with best_model.pt exist → one entry per mod.
    """
    if (abmil_run_dir / 'best_model.pt').exists():
        return [('all', abmil_run_dir)]
    entries = []
    for mod in ALL_FAKES:
        sub = abmil_run_dir / f'trained_on_{mod}'
        if sub.is_dir() and (sub / 'best_model.pt').exists():
            entries.append((mod, sub))
    return entries

# =============================================================================
#  Perturbation configuration
# =============================================================================

PERTURBATIONS: dict[str, list] = {
    'gaussian_noise':   [0.02, 0.05, 0.10],
    'gaussian_blur':    [1,    2,    3   ],
    'jpeg_compression': [90,   70,   50  ],
    'intensity_shift':  [0.10, 0.20, 0.30],
    'downscale':        [0.75, 0.50, 0.25],
}

PERTURBATION_LABELS: dict[str, list[str]] = {
    'gaussian_noise':   ['σ=0.02', 'σ=0.05', 'σ=0.10'],
    'gaussian_blur':    ['σ=1',    'σ=2',    'σ=3'   ],
    'jpeg_compression': ['q=90',   'q=70',   'q=50'  ],
    'intensity_shift':  ['δ=0.10', 'δ=0.20', 'δ=0.30'],
    'downscale':        ['×0.75',  '×0.50',  '×0.25' ],
}

# =============================================================================
#  Perturbation functions  —  img: float32 (H, W) ∈ [0, 1]
# =============================================================================

def perturb_gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(0.0, sigma, img.shape).astype(np.float32)
    return np.clip(img + noise, 0.0, 1.0)

def perturb_gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    return gaussian_filter(img, sigma=sigma).astype(np.float32)

def perturb_jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    uint8 = (img * 255.0).clip(0, 255).astype(np.uint8)
    buf   = io.BytesIO()
    Image.fromarray(uint8, mode='L').save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return np.asarray(Image.open(buf), dtype=np.float32) / 255.0

def perturb_intensity_shift(img: np.ndarray, delta: float) -> np.ndarray:
    return np.clip(img + delta, 0.0, 1.0).astype(np.float32)

def perturb_downscale(img: np.ndarray, factor: float) -> np.ndarray:
    H, W  = img.shape
    small = zoom(img, factor, order=1)
    back  = zoom(small, (H / small.shape[0], W / small.shape[1]), order=1)
    out   = np.zeros((H, W), dtype=np.float32)
    h = min(back.shape[0], H)
    w = min(back.shape[1], W)
    out[:h, :w] = back[:h, :w].astype(np.float32)
    return out

_PERTURB_FN: dict[str, Callable] = {
    'gaussian_noise':   perturb_gaussian_noise,
    'gaussian_blur':    perturb_gaussian_blur,
    'jpeg_compression': perturb_jpeg,
    'intensity_shift':  perturb_intensity_shift,
    'downscale':        perturb_downscale,
}

def apply_perturbation(img: np.ndarray, name: str, param) -> np.ndarray:
    return _PERTURB_FN[name](img, param)

def apply_perturbation_volume(vol: dict, name: str, param) -> dict:
    """Apply perturbation to all valid slices of a volume dict."""
    return {
        **vol,
        'slices': [
            apply_perturbation(img, name, param) if valid else img
            for img, valid in zip(vol['slices'], vol['valid_mask'])
        ],
    }

# =============================================================================
#  Utilities
# =============================================================================

def _load_module(unique_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(unique_name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def _select_device(gpu_id: int | None) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device('cpu')
    if gpu_id is not None:
        return torch.device(f'cuda:{gpu_id}')
    import subprocess
    try:
        out  = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
            capture_output=True, text=True, check=True,
        ).stdout.strip().split('\n')
        free = [int(x) for x in out]
        best = free.index(max(free))
        print(f"Auto-selected GPU {best} ({free[best]} MiB free)")
        return torch.device(f'cuda:{best}')
    except Exception:
        return torch.device('cuda')

# =============================================================================
#  Scorer base class
# =============================================================================

class Scorer:
    """
    Unified interface for all models.

    is_volume_model = True   → score_batch receives volume dicts
                               (keys: slices, z_indices, valid_mask, label, mod, img_id)
    is_volume_model = False  → score_batch receives list of (H,W) float32 images
    """
    name: str
    is_volume_model: bool = False

    def score_batch(self, data) -> list[float]:
        raise NotImplementedError

# =============================================================================
#  Slice-level scorer  (HP-FCN, TruFor, MVSS-Net, ManTraNet)
# =============================================================================

class PixelModelScorer(Scorer):
    """
    2D pixel-level model.  Score = max(sigmoid(logits)) over all pixels.
    Some models return (edge_map, logits) — set returns_tuple=True.
    """
    is_volume_model = False

    def __init__(
        self,
        name:          str,
        model:         torch.nn.Module,
        device:        torch.device,
        target_size:   int  = 224,
        returns_tuple: bool = False,
        batch_size:    int  = 8,
    ):
        self.name          = name
        self.model         = model.eval().to(device)
        self.device        = device
        self.target_size   = target_size
        self.returns_tuple = returns_tuple
        self.batch_size    = batch_size

    @torch.no_grad()
    def score_batch(self, imgs: list[np.ndarray]) -> list[float]:
        scores: list[float] = []
        for i in range(0, len(imgs), self.batch_size):
            chunk   = imgs[i : i + self.batch_size]
            tensors = []
            for img in chunk:
                t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
                if img.shape[0] != self.target_size or img.shape[1] != self.target_size:
                    t = F.interpolate(t, size=self.target_size,
                                      mode='bilinear', align_corners=False)
                tensors.append(t)
            batch  = torch.cat(tensors, dim=0).to(self.device)
            out    = self.model(batch)
            logits = out[1] if self.returns_tuple else out
            probs  = torch.sigmoid(logits)
            mx     = probs.flatten(1).max(dim=1).values
            scores.extend(mx.cpu().tolist())
        return scores

# =============================================================================
#  Volume-level scorers
# =============================================================================

class ABMILVolumeScorer(Scorer):
    """
    ABMIL / FA-ABMIL: concatenate patches from all K valid slices into one
    bag and run a single forward pass.  Faithful to how the model is trained.
    """
    is_volume_model = True

    def __init__(
        self,
        name:       str,
        model:      torch.nn.Module,
        device:     torch.device,
        patch_size: int = 128,
        stride:     int = 64,
    ):
        self.name       = name
        self.model      = model.eval().to(device)
        self.device     = device
        self.patch_size = patch_size
        self.stride     = stride

    @torch.no_grad()
    def score_batch(self, volumes: list[dict]) -> list[float]:
        scores: list[float] = []
        for vol in volumes:
            all_patches = []
            for img, valid in zip(vol['slices'], vol['valid_mask']):
                if valid:
                    p, _, _ = build_patch_grid(img, self.patch_size, self.stride)
                    all_patches.append(p)          # (N, P, P)
            if not all_patches:
                scores.append(0.5)
                continue
            patches_np = np.concatenate(all_patches, axis=0)   # (K*N, P, P)
            patches = (torch.from_numpy(patches_np)
                       .unsqueeze(1)    # (K*N, 1, P, P)
                       .unsqueeze(0)    # (1, K*N, 1, P, P)
                       .to(self.device))
            logit = self.model(patches)
            scores.append(torch.sigmoid(logit).flatten()[0].item())
        return scores

class R3DVolumeScorer(Scorer):
    """
    R3D-18 / MC3-18: pack K slices into a (1, 1, K, H, W) 3D tensor.
    """
    is_volume_model = True

    def __init__(
        self,
        name:        str,
        model:       torch.nn.Module,
        device:      torch.device,
        target_size: int = 224,
    ):
        self.name        = name
        self.model       = model.eval().to(device)
        self.device      = device
        self.target_size = target_size

    @torch.no_grad()
    def score_batch(self, volumes: list[dict]) -> list[float]:
        scores: list[float] = []
        for vol in volumes:
            frames = []
            for img, valid in zip(vol['slices'], vol['valid_mask']):
                t = torch.from_numpy(img).unsqueeze(0)           # (1, H, W)
                if img.shape[0] != self.target_size or img.shape[1] != self.target_size:
                    t = F.interpolate(t.unsqueeze(0), size=self.target_size,
                                      mode='bilinear', align_corners=False).squeeze(0)
                frames.append(t if valid else torch.zeros_like(t))
            volume = torch.stack(frames, dim=1).unsqueeze(0).to(self.device)  # (1,1,K,H,W)
            logit  = self.model(volume)                           # (1,1) or (1,)
            scores.append(torch.sigmoid(logit).flatten()[0].item())
        return scores

class FlatCNNVolumeScorer(Scorer):
    """
    2D CNN (e.g. ResNet-50) applied independently to each slice;
    volume score = max or mean of slice scores.
    """
    is_volume_model = True

    def __init__(
        self,
        name:        str,
        model:       torch.nn.Module,
        device:      torch.device,
        target_size: int = 224,
        vol_agg:     str = 'max',
        batch_size:  int = 8,
    ):
        self.name        = name
        self.model       = model.eval().to(device)
        self.device      = device
        self.target_size = target_size
        self.vol_agg     = vol_agg
        self.batch_size  = batch_size

    @torch.no_grad()
    def score_batch(self, volumes: list[dict]) -> list[float]:
        scores: list[float] = []
        for vol in volumes:
            valid_imgs = [img for img, v in zip(vol['slices'], vol['valid_mask']) if v]
            if not valid_imgs:
                scores.append(0.5)
                continue
            slice_scores: list[float] = []
            for i in range(0, len(valid_imgs), self.batch_size):
                chunk = valid_imgs[i : i + self.batch_size]
                tensors = []
                for img in chunk:
                    t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
                    if img.shape[0] != self.target_size or img.shape[1] != self.target_size:
                        t = F.interpolate(t, size=self.target_size,
                                          mode='bilinear', align_corners=False)
                    tensors.append(t)
                batch  = torch.cat(tensors, dim=0).to(self.device)
                logits = self.model(batch).flatten()
                slice_scores.extend(torch.sigmoid(logits).cpu().tolist())
            vol_score = max(slice_scores) if self.vol_agg == 'max' else (
                sum(slice_scores) / len(slice_scores))
            scores.append(vol_score)
        return scores

class ViTVolumeScorer(Scorer):
    """
    ViT-ABMIL: forward(slices (K,1,H,W), z_indices (K,), valid (K,)) → logit.
    """
    is_volume_model = True

    def __init__(
        self,
        name:        str,
        model:       torch.nn.Module,
        device:      torch.device,
        target_size: int = 256,
    ):
        self.name        = name
        self.model       = model.eval().to(device)
        self.device      = device
        self.target_size = target_size

    @torch.no_grad()
    def score_batch(self, volumes: list[dict]) -> list[float]:
        scores: list[float] = []
        for vol in volumes:
            frames = []
            for img in vol['slices']:
                t = torch.from_numpy(img).unsqueeze(0)           # (1, H, W)
                if img.shape[0] != self.target_size or img.shape[1] != self.target_size:
                    t = F.interpolate(t.unsqueeze(0), size=self.target_size,
                                      mode='bilinear', align_corners=False).squeeze(0)
                frames.append(t)
            slices_t = torch.stack(frames, dim=0).to(self.device)           # (K,1,H,W)
            z_t      = torch.from_numpy(vol['z_indices']).to(self.device)   # (K,)
            valid_t  = torch.from_numpy(vol['valid_mask']).to(self.device)  # (K,)
            logit    = self.model(slices_t, z_t, valid_t)
            scores.append(torch.sigmoid(logit).flatten()[0].item())
        return scores

class PoolMILVolumeScorer(Scorer):
    """
    PoolMIL-Volume: forward(patches (K,N,1,P,P), z_indices (K,), valid (K,)) → logit.
    All slices are padded to the same number of patches N.
    """
    is_volume_model = True

    def __init__(
        self,
        name:       str,
        model:      torch.nn.Module,
        device:     torch.device,
        patch_size: int = 64,
        stride:     int = 32,
    ):
        self.name       = name
        self.model      = model.eval().to(device)
        self.device     = device
        self.patch_size = patch_size
        self.stride     = stride

    @torch.no_grad()
    def score_batch(self, volumes: list[dict]) -> list[float]:
        scores: list[float] = []
        for vol in volumes:
            patches_per_slice = [
                build_patch_grid(img, self.patch_size, self.stride)[0]   # (N_k, P, P)
                for img in vol['slices']
            ]
            N_max = max(p.shape[0] for p in patches_per_slice)
            K     = len(vol['slices'])
            padded = np.zeros((K, N_max, 1, self.patch_size, self.patch_size), dtype=np.float32)
            for k, p in enumerate(patches_per_slice):
                n = p.shape[0]
                padded[k, :n, 0] = p                        # (n, P, P) → (n, 1, P, P)[k]
            patches_t = torch.from_numpy(padded).to(self.device)          # (K,N,1,P,P)
            z_t       = torch.from_numpy(vol['z_indices']).to(self.device)
            valid_t   = torch.from_numpy(vol['valid_mask']).to(self.device)
            logit     = self.model(patches_t, z_t, valid_t)
            scores.append(torch.sigmoid(logit).flatten()[0].item())
        return scores

class VolumeClassifierScorer(Scorer):
    """
    Phase-C VolumeClassifier: encodes each slice independently with a frozen
    ABMILSliceClassifier, then aggregates K slice embeddings with gated attention.
    Model signature: forward(patches_seq (K,N,1,P,P), z_indices (K,), valid (K,))
                     → (logit, attn_or_None)
    """
    is_volume_model = True

    def __init__(
        self,
        name:       str,
        model:      torch.nn.Module,
        device:     torch.device,
        patch_size: int = 64,
        stride:     int = 32,
    ):
        self.name       = name
        self.model      = model.eval().to(device)
        self.device     = device
        self.patch_size = patch_size
        self.stride     = stride

    @torch.no_grad()
    def score_batch(self, volumes: list[dict]) -> list[float]:
        scores: list[float] = []
        for vol in volumes:
            patches_per_slice = [
                build_patch_grid(img, self.patch_size, self.stride)[0]  # (N_k, P, P)
                for img in vol['slices']
            ]
            N_max  = max(p.shape[0] for p in patches_per_slice)
            K      = len(vol['slices'])
            padded = np.zeros((K, N_max, 1, self.patch_size, self.patch_size), dtype=np.float32)
            for k, p in enumerate(patches_per_slice):
                padded[k, :p.shape[0], 0] = p
            patches_t = torch.from_numpy(padded).to(self.device)            # (K, N, 1, P, P)
            z_t       = torch.from_numpy(vol['z_indices']).to(self.device)  # (K,)
            valid_t   = torch.from_numpy(vol['valid_mask']).to(self.device)  # (K,)
            logit, _  = self.model(patches_t, z_t, valid_t)
            scores.append(torch.sigmoid(logit).flatten()[0].item())
        return scores

# =============================================================================
#  Model loaders
# =============================================================================

def _load_ckpt(run_dir: Path):
    ckpt_path = run_dir / 'best_model.pt'
    args_path = run_dir / 'args.json'
    if not ckpt_path.exists():
        return None, None
    with open(args_path) as f:
        a = json.load(f)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    return ckpt, a

def load_abmil(run_dir: Path, device: torch.device):
    ckpt, a = _load_ckpt(run_dir)
    if ckpt is None:
        print(f"[ABMIL] best_model.pt not found in {run_dir}"); return None

    state         = ckpt.get('model_state_dict', ckpt)
    is_volume_cls = any(k.startswith('slice_encoder.') for k in state)

    if is_volume_cls:
        # ── Phase C: VolumeClassifier ─────────────────────────────────────
        sargs = a.get('slice_args') or {}
        if not sargs:
            slice_ckpt_dir = Path(a.get('slice_ckpt_dir', ''))
            if slice_ckpt_dir.exists():
                with open(slice_ckpt_dir / 'args.json') as f:
                    sargs = json.load(f)
        if sargs.get('use_fourier', False):
            slice_model = build_fabmil_classifier_scratch(
                backbone         = sargs.get('backbone', 'resnet50'),
                pretrained       = False,
                patch_size       = sargs.get('patch_size', 64),
                fourier_feat_dim = sargs.get('fourier_feat_dim', 256),
                proj_dim         = sargs.get('proj_dim', 512),
                attn_dim         = sargs.get('attn_dim', 128),
                dropout          = sargs.get('dropout', 0.25),
            )
        else:
            slice_model = build_abmil_classifier_scratch(
                backbone   = sargs.get('backbone', 'resnet50'),
                pretrained = False,
                proj_dim   = sargs.get('proj_dim', 512),
                attn_dim   = sargs.get('attn_dim', 128),
                dropout    = sargs.get('dropout', 0.25),
            )
        model = VolumeClassifier(
            slice_encoder = slice_model,
            feat_dim      = slice_model.feat_dim,
            K             = a.get('K', 16),
            attn_dim      = a.get('attn_dim', 256),
            dropout       = a.get('dropout', 0.25),
        )
        model.load_state_dict(state)
        K      = a.get('K', 16)
        ps     = sargs.get('patch_size', 64)
        stride = sargs.get('stride') or (ps // 2)
        print(f"[VolumeClassifier] Loaded K={K} epoch={ckpt.get('epoch','?')}  from {run_dir}")
        return VolumeClassifierScorer('HexMIL-VC', model, device,
                                      patch_size=ps, stride=stride)

    # ── Phase B: ABMILSliceClassifier ─────────────────────────────────────
    if a.get('use_fourier', False):
        model = build_fabmil_classifier_scratch(
            backbone         = a.get('backbone', 'resnet50'),
            pretrained       = False,
            patch_size       = a.get('patch_size', 128),
            fourier_feat_dim = a.get('fourier_feat_dim', 256),
            proj_dim         = a.get('proj_dim', 512),
            attn_dim         = a.get('attn_dim', 128),
            dropout          = a.get('dropout', 0.25),
        )
        label = 'FA-ABMIL'
    else:
        model = build_abmil_classifier_scratch(
            backbone   = a.get('backbone', 'resnet50'),
            pretrained = False,
            proj_dim   = a.get('proj_dim', 512),
            attn_dim   = a.get('attn_dim', 128),
            dropout    = a.get('dropout', 0.25),
        )
        label = 'ABMIL'
    model.load_state_dict(state)
    print(f"[{label}] Loaded epoch={ckpt.get('epoch','?')}  from {run_dir}")
    stride = a.get('stride') or (a.get('patch_size', 128) // 2)
    return ABMILVolumeScorer(label, model, device,
                             patch_size=a.get('patch_size', 128), stride=stride)

def load_hp_fcn(run_dir: Path, device: torch.device) -> PixelModelScorer | None:
    ckpt, a = _load_ckpt(run_dir)
    if ckpt is None:
        print(f"[HP-FCN] best_model.pt not found in {run_dir}"); return None
    train_mod = _load_module('_train_hp_fcn', BASE_DIR / 'train_hp_fcn.py')
    model     = train_mod.HPFCN(filter_learnable=a.get('filter_learnable', True))
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"[HP-FCN] Loaded epoch={ckpt.get('epoch','?')}  from {run_dir}")
    return PixelModelScorer('HP-FCN', model, device,
                            target_size=a.get('target_size', 224))

def load_trufor(run_dir: Path, device: torch.device) -> PixelModelScorer | None:
    ckpt, a = _load_ckpt(run_dir)
    if ckpt is None:
        print(f"[TruFor] best_model.pt not found in {run_dir}"); return None
    train_mod = _load_module('_train_trufor', BASE_DIR / 'train_trufor.py')
    model     = train_mod.TruForCT(pretrained=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"[TruFor] Loaded epoch={ckpt.get('epoch','?')}  from {run_dir}")
    return PixelModelScorer('TruFor', model, device,
                            target_size=a.get('target_size', 224))

def load_mvssnet(run_dir: Path, device: torch.device) -> PixelModelScorer | None:
    """Load MVSSNet with Sobel branch (mvssnet_full checkpoints)."""
    ckpt, a = _load_ckpt(run_dir)
    if ckpt is None:
        print(f"[MVSS-Net†] best_model.pt not found in {run_dir}"); return None
    train_mod = _load_module('_train_mvssnet', BASE_DIR / 'train_mvssnet.py')
    model     = train_mod.build_mvssnet_1ch(pretrained=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"[MVSS-Net†] Loaded epoch={ckpt.get('epoch','?')}  from {run_dir}")
    return PixelModelScorer('MVSS-Net†', model, device,
                            target_size=a.get('target_size', 224),
                            returns_tuple=True)

def load_mvssnet_base(run_dir: Path, device: torch.device) -> PixelModelScorer | None:
    """Load MVSSNet without Sobel branch (mvssnet base checkpoints)."""
    ckpt, a = _load_ckpt(run_dir)
    if ckpt is None:
        print(f"[MVSS-Net] best_model.pt not found in {run_dir}"); return None
    train_mod = _load_module('_train_mvssnet', BASE_DIR / 'train_mvssnet.py')
    model     = train_mod.build_mvssnet_1ch_no_sobel(pretrained=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"[MVSS-Net] Loaded epoch={ckpt.get('epoch','?')}  from {run_dir}")
    return PixelModelScorer('MVSS-Net', model, device,
                            target_size=a.get('target_size', 224),
                            returns_tuple=True)

def load_mantranet(run_dir: Path, device: torch.device) -> PixelModelScorer | None:
    ckpt, a = _load_ckpt(run_dir)
    if ckpt is None:
        print(f"[ManTraNet] best_model.pt not found in {run_dir}"); return None
    train_mod = _load_module('_train_mantranet', BASE_DIR / 'train_mantranet.py')
    model     = train_mod.ManTraNet()
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"[ManTraNet] Loaded epoch={ckpt.get('epoch','?')}  from {run_dir}")
    return PixelModelScorer('ManTraNet', model, device,
                            target_size=a.get('target_size', 224))

def load_r3d(run_dir: Path, device: torch.device) -> R3DVolumeScorer | None:
    ckpt, a = _load_ckpt(run_dir)
    if ckpt is None:
        print(f"[R3D] best_model.pt not found in {run_dir}"); return None
    train_mod = _load_module('_train_3d_resnet', BASE_DIR / 'train_3d_resnet.py')
    model     = train_mod.build_r3d_classifier(
        arch      = a.get('arch', 'r3d_18'),
        pretrained= False,
        dropout   = a.get('dropout', 0.3),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"[R3D] Loaded arch={a.get('arch','r3d_18')} epoch={ckpt.get('epoch','?')}  from {run_dir}")
    return R3DVolumeScorer('R3D-18', model, device, target_size=224)

def load_flat_cnn(run_dir: Path, device: torch.device) -> FlatCNNVolumeScorer | None:
    ckpt, a = _load_ckpt(run_dir)
    if ckpt is None:
        print(f"[FlatCNN] best_model.pt not found in {run_dir}"); return None
    train_mod = _load_module('_train_flat_cnn', BASE_DIR / 'train_flat_cnn.py')
    model     = train_mod.build_flat_cnn(pretrained=False,
                                         dropout=a.get('dropout', 0.25))
    model.load_state_dict(ckpt['model_state_dict'])
    vol_agg = a.get('vol_agg', 'max')
    print(f"[FlatCNN] Loaded vol_agg={vol_agg} epoch={ckpt.get('epoch','?')}  from {run_dir}")
    return FlatCNNVolumeScorer('FlatCNN', model, device,
                               target_size=a.get('target_size', 224),
                               vol_agg=vol_agg)

def load_vit_abmil(run_dir: Path, device: torch.device) -> ViTVolumeScorer | None:
    ckpt, a = _load_ckpt(run_dir)
    if ckpt is None:
        print(f"[ViT-ABMIL] best_model.pt not found in {run_dir}"); return None
    train_mod = _load_module('_train_vit_abmil', BASE_DIR / 'train_vit_abmil.py')
    model     = train_mod.build_vit_volume_classifier(
        K         = a.get('K', 16),
        token_size= a.get('token_size', 32),
        embed_dim = a.get('embed_dim', 256),
        depth     = a.get('depth', 4),
        num_heads = a.get('num_heads', 8),
        target_size=a.get('target_size', 256),
        attn_dim  = a.get('attn_dim', 128),
        dropout   = a.get('dropout', 0.25),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"[ViT-ABMIL] Loaded K={a.get('K',16)} epoch={ckpt.get('epoch','?')}  from {run_dir}")
    return ViTVolumeScorer('ViT-ABMIL', model, device,
                           target_size=a.get('target_size', 256))

def load_pool_mil(run_dir: Path, device: torch.device) -> PoolMILVolumeScorer | None:
    ckpt, a = _load_ckpt(run_dir)
    if ckpt is None:
        print(f"[PoolMIL] best_model.pt not found in {run_dir}"); return None
    train_mod  = _load_module('_train_pool_mil', BASE_DIR / 'train_pool_mil.py')
    slice_cls  = train_mod.build_pool_mil_slice(
        backbone  = a.get('backbone', 'resnet50'),
        pretrained= False,
        proj_dim  = a.get('proj_dim', 512),
        pool_mode = a.get('pool_mode', 'mean'),
        dropout   = a.get('dropout', 0.25),
    )
    model = train_mod.PoolMILVolumeClassifier(
        slice_encoder = slice_cls,
        feat_dim      = a.get('proj_dim', 512),
        K             = a.get('K', 16),
        pool_mode     = a.get('pool_mode', 'mean'),
        dropout       = a.get('dropout', 0.25),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"[PoolMIL] Loaded K={a.get('K',16)} epoch={ckpt.get('epoch','?')}  from {run_dir}")
    return PoolMILVolumeScorer('PoolMIL', model, device,
                               patch_size=a.get('patch_size', 64),
                               stride=a.get('stride', 32))

# Arch key → loader function
_ARCH_LOADER_FN: dict[str, Callable] = {
    'hp_fcn':       load_hp_fcn,
    'trufor':       load_trufor,
    'trufor_mitb2': load_trufor,
    'mvssnet':      load_mvssnet_base,
    'mvssnet_full': load_mvssnet,
    'mantranet':    load_mantranet,
    'r3d':          load_r3d,
    'flat_cnn':     load_flat_cnn,
    'vit_abmil':    load_vit_abmil,
    'pool_mil':     load_pool_mil,
}

# =============================================================================
#  Data loading  —  always returns volume dicts
# =============================================================================

def load_test_volumes(
    data_dir: str,
    K:        int       = 16,
    n_samples: int | None = None,
) -> list[dict]:
    """
    Returns list of volume dicts:
      slices:     list of K (H,W) float32 arrays (zero-padded if scan has < K slices)
      z_indices:  (K,) int64 — absolute z positions; -1 for padding slices
      valid_mask: (K,) bool
      label:      0 (real) or 1 (fake)
      mod:        synthesis method (or 'real')
      img_id:     scan identifier

    Fake volumes are capped at n_samples per modality.
    K slices are sampled uniformly from the scan's available slices in the test split.
    """
    tab_test = load_split_table(data_dir, 'test', ['real'] + ALL_FAKES)

    # Group available slice positions per (mod, img_id)
    vol_z: dict[tuple, list[int]] = defaultdict(list)
    for _, row in tab_test.iterrows():
        vol_z[(row['mod'], str(row['img_id']))].append(int(row['coord_z']))
    for key in vol_z:
        vol_z[key] = sorted(set(vol_z[key]))

    volumes:    list[dict]      = []
    mod_counts: dict[str, int]  = {}

    for (mod, img_id), z_list in tqdm(sorted(vol_z.items()), desc='Loading volumes'):
        label = 0 if mod == 'real' else 1

        if label == 1 and n_samples is not None:
            cnt = mod_counts.get(mod, 0)
            if cnt >= n_samples:
                continue
            mod_counts[mod] = cnt + 1

        # Uniformly sample up to K slice positions
        if len(z_list) >= K:
            idx      = np.linspace(0, len(z_list) - 1, K, dtype=int)
            selected = [z_list[i] for i in idx]
        else:
            selected = z_list

        scan_dir  = os.path.join(data_dir, mod, 'scan', img_id)
        shape     = get_shape_tiff_scan(scan_dir)
        low, high = get_percentile_tiff_scan(scan_dir, np.uint16)

        slices, z_indices, valid = [], [], []
        for z in selected:
            raw = load_slice_tiff_scan(scan_dir, shape, np.uint16, z, z + 1)[0]
            slices.append(apply_percentile(raw.astype(np.float32), low, high).astype(np.float32))
            z_indices.append(z)
            valid.append(True)

        # Pad to exactly K
        blank = np.zeros_like(slices[0])
        while len(slices) < K:
            slices.append(blank)
            z_indices.append(-1)
            valid.append(False)

        volumes.append(dict(
            slices    = slices,
            z_indices = np.array(z_indices, dtype=np.int64),
            valid_mask= np.array(valid,     dtype=bool),
            label     = label,
            mod       = mod,
            img_id    = img_id,
        ))

    n_real = sum(1 for v in volumes if v['label'] == 0)
    n_fake = sum(1 for v in volumes if v['label'] == 1)
    print(f"Loaded {len(volumes)} volumes  (real={n_real}, fake={n_fake}  "
          f"[{', '.join(f'{m}={mod_counts.get(m,0)}' for m in ALL_FAKES)}])")
    return volumes

# =============================================================================
#  Evaluation
# =============================================================================

def _auc(labels: list[int], scores: list[float]) -> float:
    y = np.array(labels); s = np.array(scores)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float('nan')

def _per_mod_auc(samples: list[dict], scores: list[float]) -> dict[str, float]:
    """Works for both volume dicts and slice dicts — both have 'mod' and 'label'."""
    real_mask = np.array([s['mod'] == 'real' for s in samples])
    out: dict[str, float] = {}
    for fm in ALL_FAKES:
        sel = real_mask | np.array([s['mod'] == fm for s in samples])
        if sel.sum() < 2:
            continue
        y  = np.array([s['label'] for s in samples])[sel]
        sc = np.array(scores)[sel]
        out[fm] = float(roc_auc_score(y, sc)) if len(np.unique(y)) > 1 else float('nan')
    return out

def evaluate_scorer(
    scorer:        Scorer,
    volumes:       list[dict],
    perturb_name:  str  | None = None,
    perturb_param             = None,
) -> dict:
    """
    Evaluate scorer on the volume list.

    For volume models: perturbation is applied to all K slices of each volume;
                       one score is produced per volume.
    For slice models:  volumes are flattened to individual valid slices;
                       perturbation is applied per slice; one score per slice.
    """
    if perturb_name is not None:
        vols_in = [apply_perturbation_volume(v, perturb_name, perturb_param)
                   for v in volumes]
    else:
        vols_in = volumes

    if scorer.is_volume_model:
        scores  = scorer.score_batch(vols_in)
        samples = vols_in                         # label/mod keys present at volume level
    else:
        # Flatten valid slices; each inherits label/mod from its parent volume
        samples = [
            {'img': img, 'label': v['label'], 'mod': v['mod'], 'img_id': v['img_id']}
            for v in vols_in
            for img, valid in zip(v['slices'], v['valid_mask']) if valid
        ]
        scores = scorer.score_batch([s['img'] for s in samples])

    return {
        'auc':     _auc([s['label'] for s in samples], scores),
        'per_mod': _per_mod_auc(samples, scores),
    }

def _evaluate_all(scorer: Scorer, volumes: list[dict]) -> dict:
    """Run clean + all perturbations. Returns full results dict."""
    r: dict = {}
    r['clean'] = evaluate_scorer(scorer, volumes)
    for p_name, p_vals in PERTURBATIONS.items():
        for p_val in p_vals:
            r[f'{p_name}_{p_val}'] = evaluate_scorer(scorer, volumes, p_name, p_val)
    return r

# =============================================================================
#  Output — CSV, figure, console table
# =============================================================================

def _delta_table(results: dict) -> dict[str, dict[str, float]]:
    delta: dict[str, dict[str, float]] = {}
    for model_name, configs in results.items():
        clean_auc = configs.get('clean', {}).get('auc', float('nan'))
        delta[model_name] = {}
        for key, m in configs.items():
            if key == 'clean':
                continue
            delta[model_name][key] = clean_auc - m.get('auc', float('nan'))
    return delta

def _save_csv(delta: dict, path: Path) -> None:
    import csv
    all_keys = _ordered_keys(delta)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model'] + all_keys)
        for mn, d in delta.items():
            w.writerow([mn] + [f"{d.get(k, float('nan')):.4f}" for k in all_keys])
    print(f"Saved ΔAUC CSV  →  {path}")

def _ordered_keys(delta: dict) -> list[str]:
    seen, keys = set(), []
    for p_name, p_vals in PERTURBATIONS.items():
        for v in p_vals:
            k = f'{p_name}_{v}'
            if k not in seen:
                keys.append(k); seen.add(k)
    for v in {k for d in delta.values() for k in d}:
        if v not in seen:
            keys.append(v)
    return keys

def _save_figure(results: dict, path: Path) -> None:
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        warnings.warn("matplotlib not available — skipping figure"); return

    n_cols = len(PERTURBATIONS)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 4.0), sharey=True)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    model_names = list(results.keys())

    for ax, (p_name, p_vals) in zip(axes, PERTURBATIONS.items()):
        x_labels = ['clean'] + PERTURBATION_LABELS[p_name]
        x_pos    = list(range(len(x_labels)))
        for ci, mn in enumerate(model_names):
            clean_auc = results[mn].get('clean', {}).get('auc', float('nan'))
            perturbed = [results[mn].get(f'{p_name}_{v}', {}).get('auc', float('nan'))
                         for v in p_vals]
            ax.plot(x_pos, [clean_auc] + perturbed,
                    marker='o', linewidth=1.8, markersize=5,
                    label=mn, color=colors[ci % len(colors)])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=30, ha='right', fontsize=8)
        ax.set_title(p_name.replace('_', ' ').title(), fontsize=9, pad=4)
        ax.set_ylim(0.40, 1.02)
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
        ax.grid(True, which='major', alpha=0.3)
        ax.grid(True, which='minor', alpha=0.12, linestyle=':')
        if ax is axes[0]:
            ax.set_ylabel('AUC', fontsize=9)

    handles, labels_leg = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc='lower center',
               ncol=min(len(model_names), 4), fontsize=8,
               bbox_to_anchor=(0.5, -0.08), frameon=True)
    fig.suptitle('Robustness: AUC under post-processing perturbations', fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved figure     →  {path}")

def _print_delta_table(delta: dict, group_tag: str) -> None:
    all_keys = _ordered_keys(delta)
    col_w    = 12
    sep = '=' * (16 + col_w * len(all_keys))
    print(f"\n{sep}")
    print(f"ΔAUC — group: {group_tag}  (clean − perturbed; positive = degradation)")
    print(sep)
    print(f"{'model':16s}" + ''.join(f"{k[:col_w]:>{col_w}s}" for k in all_keys))
    print('-' * (16 + col_w * len(all_keys)))
    for mn, d in delta.items():
        cells = ''.join(f"{d.get(k, float('nan')):+{col_w}.4f}" for k in all_keys)
        print(f"{mn:16s}{cells}")
    print(sep)

# =============================================================================
#  Args
# =============================================================================

def get_args():
    p = argparse.ArgumentParser(
        description='Robustness evaluation — volume + slice level, auto-discovery')
    p.add_argument('--abmil_run_dir', type=str, default=None,
                   help='ABMIL run dir (best_model.pt) or parent with trained_on_{mod} subdirs')
    p.add_argument('--baselines_runs_dir', type=str, default=None,
                   help='Dir containing baseline run dirs (default: experiments/baselines/runs/)')
    p.add_argument('--data_dir',  type=str, default=DATA_DIR)
    p.add_argument('--out_dir',   type=str, default=None)
    p.add_argument('--K',         type=int, default=32,
                   help='Slices per volume for volume-level models (default: 32)')
    p.add_argument('--n_samples', type=int, default=500,
                   help='Max fake volumes per modality (0 = use all)')
    p.add_argument('--gpu_id',    type=int, default=None)
    p.add_argument('--seed',      type=int, default=42)
    return p.parse_args()

# =============================================================================
#  Main
# =============================================================================

def main():
    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = _select_device(args.gpu_id)

    if args.out_dir is None:
        args.out_dir = str(WORK_DIR / 'experiments' / 'robustness' / 'evaluation')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover model configs ─────────────────────────────────────────────────
    abmil_entries: list[tuple[str, Path]] = []
    if args.abmil_run_dir is not None:
        abmil_entries = _discover_abmil(Path(args.abmil_run_dir))
        if not abmil_entries:
            print(f"[WARN] No ABMIL best_model.pt found in {args.abmil_run_dir}")

    baselines_runs_dir = (
        Path(args.baselines_runs_dir) if args.baselines_runs_dir
        else BASE_DIR / 'runs'
    )
    baseline_entries = _discover_baselines(baselines_runs_dir)

    print(f"\nDiscovered {len(baseline_entries)} baseline configs:")
    for arch_key, scorer_name, trained_on, run_dir in baseline_entries:
        vol_tag = '(vol)' if _ARCH_LOADER_FN[arch_key] in (
            load_r3d, load_flat_cnn, load_vit_abmil, load_pool_mil) else '(slice)'
        print(f"  {scorer_name:30s}  trained_on={trained_on:12s}  {vol_tag}  {run_dir.name}")

    if not abmil_entries and not baseline_entries:
        print("No models found — provide --abmil_run_dir and/or --baselines_runs_dir")
        return

    # ── Load test volumes once (all models share the same data) ──────────────
    n_samp  = args.n_samples if args.n_samples > 0 else None
    volumes = load_test_volumes(args.data_dir, K=args.K, n_samples=n_samp)

    # ── Evaluate all scorers ──────────────────────────────────────────────────
    results: dict[str, dict] = {}

    def _run_and_store(scorer: Scorer) -> None:
        granularity = 'volume' if scorer.is_volume_model else 'slice'
        print(f"\n── Evaluating {scorer.name}  [{granularity}-level] ──")
        r = _evaluate_all(scorer, volumes)
        results[scorer.name] = r
        per = '  '.join(
            f"{fm}={r['clean']['per_mod'].get(fm, float('nan')):.4f}"
            for fm in ALL_FAKES
        )
        print(f"  clean AUC={r['clean']['auc']:.4f}  [{per}]")

    # ABMIL: one scorer per discovered trained_on config
    abmil_groups: dict[str, list[str]] = {m: [] for m in ALL_FAKES}
    abmil_all_names: list[str] = []

    for trained_on, run_dir in abmil_entries:
        scorer = load_abmil(run_dir, device)
        if scorer is None:
            continue
        if len(abmil_entries) > 1:
            scorer.name = f'{scorer.name}[{trained_on}]'
        _run_and_store(scorer)
        abmil_all_names.append(scorer.name)
        if trained_on == 'all':
            for mod in ALL_FAKES:
                abmil_groups[mod].append(scorer.name)
        else:
            abmil_groups[trained_on].append(scorer.name)
        del scorer; torch.cuda.empty_cache()

    # Baselines
    baseline_groups: dict[str, list[str]] = {m: [] for m in ALL_FAKES}
    for arch_key, scorer_name, trained_on, run_dir in baseline_entries:
        scorer = _ARCH_LOADER_FN[arch_key](run_dir, device)
        if scorer is None:
            continue
        scorer.name = f'{scorer_name}[{trained_on}]'
        _run_and_store(scorer)
        baseline_groups[trained_on].append(scorer.name)
        del scorer; torch.cuda.empty_cache()

    # ── Save combined JSON ────────────────────────────────────────────────────
    json_path = out_dir / 'robustness_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved combined JSON  →  {json_path}")

    # ── Per-group outputs ─────────────────────────────────────────────────────
    groups: dict[str, list[str]] = {}
    for mod in ALL_FAKES:
        group = abmil_groups[mod] + baseline_groups[mod]
        if group:
            groups[mod] = group

    all_baseline_names = [n for names in baseline_groups.values() for n in names]
    if len(groups) > 1 or (abmil_all_names and all_baseline_names):
        groups['all'] = abmil_all_names + all_baseline_names

    if not groups and results:
        groups['all'] = list(results.keys())

    for group_tag, scorer_names in groups.items():
        group_results = {n: results[n] for n in scorer_names if n in results}
        if not group_results:
            continue
        delta = _delta_table(group_results)
        _save_csv(delta, out_dir / f'robustness_delta_auc_{group_tag}.csv')
        _save_figure(group_results, out_dir / f'robustness_curves_{group_tag}.png')
        _print_delta_table(delta, group_tag)

    print(f"\nDone. Results saved to: {out_dir}")

if __name__ == '__main__':
    main()
