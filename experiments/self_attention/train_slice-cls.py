#!/usr/bin/env python3
"""
train_slice-cls.py  [SA-ABMIL variant]
=======================================
Phase B training script: full-slice binary classification via SA-ABMIL.

This is the Self-Attention version.  A lightweight Transformer encoder is
inserted between the CNN patch embedder and the Gated ABMIL aggregator,
allowing each patch to attend to all other patches in the same slice before
pooling.  This helps the model detect spatial inconsistency — a more
domain-agnostic forgery cue than absolute patch-level artefacts.

Each CT slice is converted into a bag of N overlapping patches, encoded by a
CNN encoder (backbone + spatial attention) trained end-to-end together with
the Gated Attention MIL module and classification head.

Key features:
  - Automatic GPU selection via GPUtil (or --gpu_id for manual override)
  - Optional auxiliary attention supervision: ABMIL weights guided by the
    fraction of GT mask covered by each patch  (--aux_attn_loss)
  - WandB logging (--wandb)
  - Checkpointing on minimum validation loss
  - Periodic qualitative visualisations (--vis_every)
  - args.json saved at the start of every run

Usage:
    python experiments/train_slice-cls.py \\
        --backbone resnet50 --pretrained
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score,
)
from scipy.special import expit as sigmoid_np
import GPUtil

# ── project imports ──────────────────────────────────────────────────────────

from hexmil.data.patch_dataset import load_split_table
from hexmil.data.slice_dataset import SliceDataset, build_patch_grid, reconstruct_heatmap
from hexmil.models.abmil_slice_classifier_sa import (
    build_sa_classifier_scratch, SABMILSliceClassifier,
)

# ── paths ────────────────────────────────────────────────────────────────────

sys.path.insert(0, str(WORK_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import DATA_DIR, WORK_DIR
ALL_FAKES = ['pix2pix', 'cycle', 'diffusion']

# =============================================================================
#  GPU selection
# =============================================================================

def select_best_gpu() -> int | None:
    if not torch.cuda.is_available():
        print("No CUDA GPUs available, using CPU")
        return None
    gpus = GPUtil.getGPUs()
    if not gpus:
        print("No GPUs found by GPUtil, using cuda:0")
        return 0
    best = max(gpus, key=lambda g: g.memoryFree)
    print(f"🎯 Auto-selected GPU {best.id}: {best.name} "
          f"(Free: {best.memoryFree}MB / {best.memoryTotal}MB)")
    return best.id

# =============================================================================
#  Args
# =============================================================================

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Phase B: slice-level MIL classification')

    # ── Backbone ──────────────────────────────────────────────────────────
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet50', 'resnet34', 'efficientnet_b0',
                                 'efficientnet_b4', 'densenet121'],
                        help='CNN backbone for the patch encoder')
    parser.add_argument('--mods', nargs='+', default=['pix2pix', 'cycle', 'diffusion'],
                        help='Fake modalities for train/val. real is always included. '
                             'Use a subset for cross-domain: e.g. --mods pix2pix')
    parser.add_argument('--test_mods', nargs='+', default=None,
                        help='Fake modalities for test set (default: same as --mods). '
                             'Set to test cross-domain generalisation, e.g. --test_mods cycle diffusion')
    parser.add_argument('--pretrained',    action='store_true', default=True,
                        help='Initialise backbone from ImageNet weights')

    # ── MIL / model ──────────────────────────────────────────────────────────
    parser.add_argument('--patch_size', type=int, default=64, choices=[32, 64, 128],
                        help='Patch size for the sliding-window tiling')
    parser.add_argument('--stride',     type=int, default=None,
                        help='Sliding-window stride (default: patch_size // 2)')
    parser.add_argument('--proj_dim',   type=int, default=512,
                        help='Patch feature projection dimension before ABMIL')
    parser.add_argument('--attn_dim',   type=int, default=128,
                        help='Gated ABMIL internal attention dimension')
    parser.add_argument('--dropout',    type=float, default=0.25,
                        help='Dropout probability in projector + head + ABMIL')

    # ── Self-Attention Transformer ────────────────────────────────────────────
    parser.add_argument('--sa_n_heads',  type=int, default=8,
                        help='Number of self-attention heads in PatchTransformer '
                             '(proj_dim must be divisible by sa_n_heads)')
    parser.add_argument('--sa_n_layers', type=int, default=2,
                        help='Number of Transformer encoder layers in PatchTransformer')

    # ── Auxiliary attention loss ──────────────────────────────────────────────
    parser.add_argument('--aux_attn_loss', action='store_true',
                        help='Guide ABMIL weights with per-patch GT mask coverage')
    parser.add_argument('--aux_weight',    type=float, default=0.3,
                        help='Weight of the auxiliary attention loss')

    # ── Training ─────────────────────────────────────────────────────────────
    parser.add_argument('--epochs',          type=int,   default=50)
    parser.add_argument('--batch_size',      type=int,   default=32,
                        help='Number of slices per batch')
    parser.add_argument('--lr',              type=float, default=1e-4)
    parser.add_argument('--weight_decay',    type=float, default=1e-4)
    parser.add_argument('--balance_classes', action='store_true', default=True,
                        help='Weighted sampler to balance real/fake slices')

    # ── Scheduler / early stopping ────────────────────────────────────────────
    parser.add_argument('--scheduler',      action='store_true',
                        help='ReduceLROnPlateau on val loss')
    parser.add_argument('--warmup_epochs',  type=int, default=0)
    parser.add_argument('--patience',       type=int, default=10)

    # ── System ───────────────────────────────────────────────────────────────
    parser.add_argument('--num_workers', type=int,  default=4)
    parser.add_argument('--amp',         action='store_true', default=True,
                        help='Mixed precision for projector/aggregator/head')
    parser.add_argument('--seed',        type=int,  default=42)
    parser.add_argument('--wandb_mode',  default='online', choices=['online', 'offline', 'disabled'],
                        help='WandB logging mode')
    parser.add_argument('--debug',       action='store_true',
                        help='Debug mode: sets wandb_mode to disabled')
    parser.add_argument('--run_name',    type=str,  default=None)
    parser.add_argument('--gpu_id',      type=int,  default=None)

    # ── Visualisation ─────────────────────────────────────────────────────────
    parser.add_argument('--vis_every',    type=int, default=5)
    parser.add_argument('--vis_n_samples', type=int, default=8)

    return parser.parse_args()

# =============================================================================
#  Data
# =============================================================================

def build_dataloaders(args) -> tuple[DataLoader, DataLoader, DataLoader]:
    fake_mods  = [m for m in args.mods if m != 'real']
    train_mods = ['real'] + fake_mods
    # Test always includes ALL fake mods (unless explicitly overridden)
    # so in-domain vs out-of-domain metrics are always available.
    if args.test_mods is not None:
        test_mods = ['real'] + [m for m in args.test_mods if m != 'real']
    else:
        test_mods = ['real'] + ALL_FAKES

    tab_train = load_split_table(DATA_DIR, 'train', train_mods)
    tab_valid = load_split_table(DATA_DIR, 'valid', train_mods)
    tab_test  = load_split_table(DATA_DIR, 'test',  test_mods)

    stride = args.stride   # may be None → default inside SliceDataset

    ds_train = SliceDataset(DATA_DIR, tab_train, patch_size=args.patch_size,
                             stride=stride, augment=True)
    ds_valid = SliceDataset(DATA_DIR, tab_valid, patch_size=args.patch_size,
                             stride=stride, augment=False)
    ds_test  = SliceDataset(DATA_DIR, tab_test,  patch_size=args.patch_size,
                             stride=stride, augment=False)

    if args.balance_classes:
        labels = tab_train['mod'].apply(lambda m: 0 if m == 'real' else 1).values
        counts = np.bincount(labels)
        w_samp = 1.0 / counts[labels]
        sampler     = WeightedRandomSampler(w_samp, len(w_samp), replacement=True)
        shuffle_tr  = False
    else:
        sampler    = None
        shuffle_tr = True

    dl_train = DataLoader(ds_train, batch_size=args.batch_size,
                          shuffle=shuffle_tr, sampler=sampler,
                          num_workers=args.num_workers, pin_memory=True,
                          drop_last=True)
    dl_valid = DataLoader(ds_valid, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)
    dl_test  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    print(f"  Train: {len(ds_train)} slices  ({len(dl_train)} batches)")
    print(f"  Valid: {len(ds_valid)} slices  ({len(dl_valid)} batches)")
    print(f"  Test:  {len(ds_test)} slices  ({len(dl_test)} batches)")

    # Actual stride used (computed once via dataset attribute)
    # OOD validation + test loaders (real + fakes not seen during training)
    ood_fakes   = [m for m in ALL_FAKES if m not in set(fake_mods)]
    dl_val_ood  = None
    dl_test_ood = None
    if ood_fakes:
        tab_val_ood  = load_split_table(DATA_DIR, 'valid', ['real'] + ood_fakes)
        tab_test_ood = load_split_table(DATA_DIR, 'test',  ['real'] + ood_fakes)
        ds_val_ood   = SliceDataset(DATA_DIR, tab_val_ood,  patch_size=args.patch_size,
                                    stride=stride, augment=False)
        ds_test_ood  = SliceDataset(DATA_DIR, tab_test_ood, patch_size=args.patch_size,
                                    stride=stride, augment=False)
        dl_val_ood  = DataLoader(ds_val_ood,  batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)
        dl_test_ood = DataLoader(ds_test_ood, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)
        print(f"  Val OOD: {len(ds_val_ood):5d} slices  |  "
              f"Test OOD: {len(ds_test_ood):5d}  (OOD fakes: {ood_fakes})")

    actual_stride = ds_train.stride
    print(f"  Patches/slice (approx): {_count_patches(ds_train)}")
    return dl_train, dl_valid, dl_test, actual_stride, dl_val_ood, dl_test_ood

def _count_patches(ds: SliceDataset) -> str:
    """Quick estimate for the first valid sample."""
    try:
        sample = ds[0]
        n = sample['patches'].shape[0]
        return str(n)
    except Exception:
        return '?'

# =============================================================================
#  Loss
# =============================================================================

class SliceLoss(nn.Module):
    """
    BCE on the bag-level logit + optional MSE on ABMIL attention weights
    guided by per-patch GT mask coverage.
    """

    def __init__(self, aux_attn: bool = False, aux_weight: float = 0.3):
        super().__init__()
        self.bce        = nn.BCEWithLogitsLoss()
        self.aux_attn   = aux_attn
        self.aux_weight = aux_weight

    @staticmethod
    def compute_patch_mask_coverage(
        masks: torch.Tensor,           # (B, 1, H, W)
        patches_shape: torch.Tensor,   # (B, N, 1, P, P) — only used for N
        grid_hw: torch.Tensor,         # (B, 2)  [n_rows, n_cols]
        slice_hw: torch.Tensor,        # (B, 2)  [H, W]
        patch_size: int,
        stride: int,
    ) -> torch.Tensor:
        """
        For each patch in the bag, compute the fraction of pixels that are
        inside the GT manipulation mask.  Returns (B, N) float32 tensor.
        This is used as a soft target for the ABMIL attention weights.
        """
        B, _, H_t, W_t = masks.shape
        N  = patches_shape.shape[1]
        targets = torch.zeros(B, N, device=masks.device)

        half = patch_size // 2

        for b in range(B):
            n_rows = int(grid_hw[b, 0].item())
            n_cols = int(grid_hw[b, 1].item())
            H      = int(slice_hw[b, 0].item())
            W      = int(slice_hw[b, 1].item())
            mask_b = masks[b, 0]           # (H, W) on device

            ys = list(range(half, H - half + 1, stride))
            xs = list(range(half, W - half + 1, stride))
            if not ys or ys[-1] < H - half:
                ys.append(H - half)
            if not xs or xs[-1] < W - half:
                xs.append(W - half)
            ys = sorted(set(ys))
            xs = sorted(set(xs))

            idx = 0
            for cy in ys:
                for cx in xs:
                    y0, y1 = max(cy - half, 0), min(cy + half, H)
                    x0, x1 = max(cx - half, 0), min(cx + half, W)
                    region = mask_b[y0:y1, x0:x1]
                    targets[b, idx] = region.mean()
                    idx += 1

        return targets   # (B, N)

    def forward(
        self,
        logits: torch.Tensor,       # (B, 1)
        labels: torch.Tensor,       # (B,)  0/1  int
        attn_weights: torch.Tensor | None = None,   # (B, N)
        masks: torch.Tensor | None = None,           # (B, 1, H, W)
        patches: torch.Tensor | None = None,         # (B, N, 1, P, P)  shape ref
        grid_hw: torch.Tensor | None = None,         # (B, 2)
        slice_hw: torch.Tensor | None = None,        # (B, 2)
        patch_size: int = 128,
        stride: int = 64,
    ) -> dict[str, torch.Tensor]:
        cls_loss = self.bce(logits.squeeze(1), labels.float())
        result   = {'cls': cls_loss}

        if (self.aux_attn
                and attn_weights is not None
                and masks is not None
                and patches is not None
                and grid_hw is not None
                and slice_hw is not None):

            # Only supervise fake samples (where mask may be non-zero)
            fake_idx = labels > 0
            if fake_idx.any():
                cov = self.compute_patch_mask_coverage(
                    masks[fake_idx],
                    patches[fake_idx],
                    grid_hw[fake_idx],
                    slice_hw[fake_idx],
                    patch_size, stride,
                )
                # Normalise coverage to a soft probability distribution
                cov = cov / (cov.sum(dim=1, keepdim=True) + 1e-8)
                attn_sup = attn_weights[fake_idx]

                # KL divergence: guide attention toward coverage distribution
                # attn is already softmax'd; clamp to avoid log(0)
                attn_loss = F_torch.kl_div(
                    (attn_sup + 1e-8).log(),
                    cov,
                    reduction='batchmean',
                )
                result['attn'] = attn_loss
            else:
                result['attn'] = torch.tensor(0.0, device=logits.device)

            result['total'] = cls_loss + self.aux_weight * result['attn']
        else:
            result['total'] = cls_loss

        return result

# =============================================================================
#  Evaluation
# =============================================================================

def _balanced_mod_metrics(
    bin_labels: np.ndarray,
    probs:      np.ndarray,
    preds:      np.ndarray,
    mods_arr:   np.ndarray,
    mod:        str,
    seed:       int = 42,
) -> dict[str, float]:
    """
    AUC/ACC/F1/AP for `mod` vs real using equal balanced sampling (fixed seed).
    """
    real_idx = np.where(mods_arr == 'real')[0]
    fake_idx = np.where(mods_arr == mod)[0]
    if len(fake_idx) == 0 or len(real_idx) == 0:
        return {k: float('nan') for k in ['auc', 'acc', 'f1', 'ap']}

    n   = min(len(real_idx), len(fake_idx))
    rng = np.random.default_rng(seed)
    sel = np.concatenate([
        rng.choice(real_idx, n, replace=False),
        rng.choice(fake_idx, n, replace=False),
    ])
    lbl = bin_labels[sel]
    prb = probs[sel]
    prd = preds[sel]

    if lbl.sum() < 1 or (lbl == 0).sum() < 1:
        return {k: float('nan') for k in ['auc', 'acc', 'f1', 'ap']}

    try:
        auc = float(roc_auc_score(lbl, prb))
    except Exception:
        auc = float('nan')
    return {
        'auc': auc,
        'acc': float(accuracy_score(lbl, prd)),
        'f1':  float(f1_score(lbl, prd, zero_division=0)),
        'ap':  float(average_precision_score(lbl, prb)),
    }

@torch.no_grad()
def evaluate(
    model: ABMILSliceClassifier,
    dataloader: DataLoader,
    criterion: SliceLoss,
    device: torch.device,
    patch_size: int,
    stride: int,
) -> dict:
    model.eval()

    all_logits: list = []
    all_labels: list = []
    all_mods:   list = []
    all_img_ids: list = []
    running_loss = 0.0
    n_batches    = 0

    for batch in dataloader:
        patches  = batch['patches'].to(device)   # (B, N, 1, P, P)
        labels   = batch['label'].to(device)
        masks    = batch['mask'].to(device)
        grid_hw  = batch['grid_hw'].to(device)
        slice_hw = batch['slice_hw'].to(device)

        logits, attn_w = model(patches, return_attn=True)

        loss_dict = criterion(
            logits, labels, attn_w, masks, patches, grid_hw, slice_hw,
            patch_size=patch_size, stride=stride,
        )
        running_loss += loss_dict['total'].item()
        n_batches    += 1

        all_logits.append(logits.squeeze(1).cpu())
        all_labels.append(batch['label'])
        all_mods.extend(batch['mod'])
        all_img_ids.extend(batch['img_id'])

    all_logits    = torch.cat(all_logits).numpy()
    all_labels    = torch.cat(all_labels).numpy()   # binary 0/1
    probs         = sigmoid_np(all_logits)
    preds         = (probs >= 0.5).astype(int)
    mods_arr      = np.array(all_mods)
    ids_arr       = np.array(all_img_ids)

    metrics = {
        'loss':     running_loss / max(n_batches, 1),
        'auc':      float(roc_auc_score(all_labels, probs)),
        'accuracy': float(accuracy_score(all_labels, preds)),
        'ap':       float(average_precision_score(all_labels, probs)),
        'f1':       float(f1_score(all_labels, preds, zero_division=0)),
    }

    def _mod_metrics(lbl, prb, prd, ids):
        """AUC / Acc / AP / F1 per manipulation type (rem / inj).

        The overall metric is the macro-average of rem and inj sub-breakdowns
        so that it stays coherent with the displayed sub-breakdown.  Computing
        the overall directly over (all-real + mod-fakes) would inflate the real
        pool with unpaired real samples that carry neither a rem_ nor inj_
        prefix, causing the overall to diverge from its own sub-items.
        """
        _nan4 = {k: float('nan') for k in [
            'auc', 'acc', 'ap', 'f1',
            'rem_auc', 'rem_acc', 'rem_ap', 'rem_f1',
            'inj_auc', 'inj_acc', 'inj_ap', 'inj_f1',
        ]}
        if len(lbl) < 2 or lbl.sum() < 1 or (lbl == 0).sum() < 1:
            return _nan4
        sub: dict = {}
        for ty, pfx in [('rem', 'rem_'), ('inj', 'inj_')]:
            msk = np.array([str(i).startswith(pfx) for i in ids])
            tl, tp, td = lbl[msk], prb[msk], prd[msk]
            if msk.sum() < 2 or tl.sum() < 1 or (tl == 0).sum() < 1:
                sub[ty] = {k: float('nan') for k in ['auc', 'acc', 'ap', 'f1']}
            else:
                sub[ty] = {
                    'auc': float(roc_auc_score(tl, tp)),
                    'acc': float(accuracy_score(tl, td)),
                    'ap':  float(average_precision_score(tl, tp)),
                    'f1':  float(f1_score(tl, td, zero_division=0)),
                }

        def _macro(k: str) -> float:
            vals = [sub[ty][k] for ty in ('rem', 'inj') if not np.isnan(sub[ty][k])]
            return float(np.mean(vals)) if vals else float('nan')

        out = {'auc': _macro('auc'), 'acc': _macro('acc'),
               'ap':  _macro('ap'),  'f1':  _macro('f1')}
        for ty in ('rem', 'inj'):
            for k, v in sub[ty].items():
                out[f'{ty}_{k}'] = v
        return out

    # Per-modality metrics (each fake mod vs all real)
    real_mask = mods_arr == 'real'
    for mod in sorted(m for m in set(all_mods) if m != 'real'):
        fake_mask = mods_arr == mod
        sel = real_mask | fake_mask
        for k, v in _mod_metrics(
            all_labels[sel], probs[sel], preds[sel], ids_arr[sel],
        ).items():
            metrics[f'{mod}_{k}'] = v

    return metrics

# =============================================================================
#  Training
# =============================================================================

def train_one_epoch(
    model: ABMILSliceClassifier,
    dataloader: DataLoader,
    criterion: SliceLoss,
    optimiser: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    patch_size: int,
    stride: int,
) -> dict:
    model.train()

    running = {'cls': 0.0, 'total': 0.0}
    if criterion.aux_attn:
        running['attn'] = 0.0
    n_batches = 0

    for batch in dataloader:
        patches  = batch['patches'].to(device)   # (B, N, 1, P, P)
        labels   = batch['label'].to(device)
        masks    = batch['mask'].to(device)
        grid_hw  = batch['grid_hw'].to(device)
        slice_hw = batch['slice_hw'].to(device)

        optimiser.zero_grad(set_to_none=True)

        # encode_patches runs under @torch.no_grad() — AMP is fine here
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits, attn_w = model(patches, return_attn=True)

        # Loss outside autocast for numerical stability (BCEWithLogitsLoss)
        loss_dict = criterion(
            logits, labels, attn_w, masks, patches, grid_hw, slice_hw,
            patch_size=patch_size, stride=stride,
        )
        loss = loss_dict['total']

        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        scaler.step(optimiser)
        scaler.update()

        for k in running:
            if k in loss_dict:
                running[k] += loss_dict[k].item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in running.items()}

# =============================================================================
#  Periodic visualisation
# =============================================================================

# Grid column order for visualisations: one representative sample per modality
_VIS_COL_MODS = ['real', 'cycle', 'diffusion', 'pix2pix']
_VIS_TYS      = ['removal', 'injection']   # manipulation type subsets

def _get_ty(img_id: str) -> str:
    """Infer manipulation type: 'rem_' prefix → 'removal', else 'injection'."""
    return 'removal' if str(img_id).startswith('rem_') else 'injection'

def _smooth_attn(a: np.ndarray) -> np.ndarray:
    """Smooth an attention map with adaptive Gaussian and renormalise to [0, 1]."""
    from scipy.ndimage import gaussian_filter
    sigma = max(1.0, max(a.shape) * 0.03)
    a = gaussian_filter(a.astype(np.float32), sigma=sigma)
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo + 1e-8)

@torch.no_grad()
def save_epoch_vis(
    model: ABMILSliceClassifier,
    dataloader: DataLoader,
    device: torch.device,
    vis_dir: Path,
    patch_size: int,
    stride: int,
    n_samples: int = 8,
    in_domain_fakes: set | None = None,
    ood_dataloader: DataLoader | None = None,
) -> None:
    """
    Save two visualisation grids from the validation set:
        samples_removal.png   — one sample per modality from removal scans
        samples_injection.png — one sample per modality from injection scans

    Layout:
        rows  → [ Image | Attention heatmap | Overlay + mask contour ]
        cols  → [ real | in-domain fakes | OOD fakes ]  (separator drawn when both present)
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    vis_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.encoder.eval()

    def _collect(loader: DataLoader) -> None:
        """Fill `cands` from a dataloader (in-place)."""
        for batch in loader:
            if all(len(cands[ty][m]) >= _N_CAND for ty in _VIS_TYS for m in _VIS_COL_MODS):
                break
            patches  = batch['patches'].to(device)
            logits, attn_w = model(patches, return_attn=True)
            for i in range(len(logits)):
                mod    = batch['mod'][i]
                img_id = batch['img_id'][i]
                ty     = _get_ty(img_id)
                if mod in _VIS_COL_MODS and len(cands[ty][mod]) < _N_CAND:
                    gh      = (int(batch['grid_hw'][i, 0]), int(batch['grid_hw'][i, 1]))
                    sh      = (int(batch['slice_hw'][i, 0]), int(batch['slice_hw'][i, 1]))
                    w_i     = attn_w[i].cpu().float().numpy()
                    heatmap = reconstruct_heatmap(w_i, gh, sh, patch_size, stride)
                    grid    = _assemble_slice_preview(batch['patches'][i], gh, sh, patch_size, stride)
                    cands[ty][mod].append({
                        'grid':    grid,
                        'mask':    batch['mask'][i, 0].numpy(),
                        'heatmap': heatmap,
                        'label':   int(batch['label'][i]),
                        'prob':    float(sigmoid_np(logits[i, 0].cpu().item())),
                        'mod':     mod,
                        'img_id':  img_id,
                    })

    # Collect candidates per (ty, mod) — first in-domain, then OOD
    _N_CAND = 40
    cands: dict[str, dict[str, list]] = {ty: {m: [] for m in _VIS_COL_MODS} for ty in _VIS_TYS}

    _collect(dataloader)
    if ood_dataloader is not None:
        _collect(ood_dataloader)

    # Select shared img_id so all columns show the same nodule
    by_ty: dict[str, dict[str, dict]] = {'removal': {}, 'injection': {}}
    for ty in _VIS_TYS:
        id_sets = [{s['img_id'] for s in cands[ty][m]} for m in _VIS_COL_MODS if cands[ty][m]]
        shared  = id_sets[0].intersection(*id_sets[1:]) if len(id_sets) > 1 else (id_sets[0] if id_sets else set())
        anchor  = next(iter(shared)) if shared else None
        for mod in _VIS_COL_MODS:
            clist = cands[ty][mod]
            if not clist:
                continue
            match = next((s for s in clist if s['img_id'] == anchor), None) if anchor else None
            by_ty[ty][mod] = match if match else clist[0]

    def _draw_vis_grid(by_mod: dict, suffix: str) -> None:
        # Order: real | in-domain fakes | OOD fakes
        _all_f = [m for m in _VIS_COL_MODS if m != 'real']
        if in_domain_fakes is not None:
            _in_c  = [m for m in sorted(_all_f) if m in in_domain_fakes and m in by_mod]
            _ood_c = [m for m in sorted(_all_f) if m not in in_domain_fakes and m in by_mod]
        else:
            _in_c  = [m for m in sorted(_all_f) if m in by_mod]
            _ood_c = []
        _real = ['real'] if 'real' in by_mod else []
        cols  = _real + _in_c + _ood_c
        if not cols:
            return
        n_in       = len(_real) + len(_in_c)
        row_labels = ['Image', 'Attention', 'Overlay']
        fig, axes  = plt.subplots(3, len(cols), figsize=(3.5 * len(cols), 9.5),
                                  squeeze=False)
        fig.suptitle(suffix.capitalize(), fontsize=11, fontweight='bold', y=1.01)
        for col_idx, mod in enumerate(cols):
            s        = by_mod[mod]
            is_ood   = col_idx >= n_in
            gt_str   = 'real' if s['label'] == 0 else s['mod']
            pred_str = 'fake' if s['prob'] > 0.5 else 'real'
            correct  = (s['label'] == 0) == (s['prob'] <= 0.5)
            col_title = (
                f"{mod.upper()}\n"
                f"{s['img_id']}\n"
                f"GT:{gt_str}  Pred:{pred_str}(p={s['prob']:.2f})  {'✓' if correct else '✗'}"
            )
            attn_sm = _smooth_attn(s['heatmap'])
            axes[0, col_idx].imshow(s['grid'], cmap='gray')
            axes[0, col_idx].set_title(col_title, fontsize=7,
                                       color='#e65100' if is_ood else 'black')
            axes[0, col_idx].axis('off')
            if s['label'] > 0 and s['mask'].sum() > 0:
                axes[0, col_idx].contour(s['mask'], levels=[0.5], colors='lime', linewidths=1.2)
            axes[1, col_idx].imshow(attn_sm, cmap='turbo', vmin=0, vmax=1)
            axes[1, col_idx].axis('off')
            axes[2, col_idx].imshow(s['grid'], cmap='gray')
            axes[2, col_idx].imshow(attn_sm, cmap='turbo', alpha=0.45, vmin=0, vmax=1)
            if s['label'] > 0 and s['mask'].sum() > 0:
                axes[2, col_idx].contour(s['mask'], levels=[0.5], colors='lime', linewidths=1.2)
            axes[2, col_idx].axis('off')
        for row_idx, rl in enumerate(row_labels):
            axes[row_idx, 0].set_ylabel(rl, fontsize=9, fontweight='bold',
                                        rotation=90, labelpad=4)
        plt.tight_layout()
        if _ood_c:
            fig.canvas.draw()
            in0  = axes[0, 0].get_position()
            in1  = axes[0, n_in - 1].get_position()
            od0  = axes[0, n_in].get_position()
            od1  = axes[0, len(cols) - 1].get_position()
            top  = axes[0, 0].get_position().y1
            bot  = axes[2, 0].get_position().y0
            fig.text((in0.x0 + in1.x1) / 2, top + 0.025,
                     '── In-domain ──', ha='center', va='bottom',
                     fontsize=9, fontweight='bold', color='#1565c0')
            fig.text((od0.x0 + od1.x1) / 2, top + 0.025,
                     '── OOD ──', ha='center', va='bottom',
                     fontsize=9, fontweight='bold', color='#e65100')
            sep_x = (in1.x1 + od0.x0) / 2
            fig.add_artist(plt.Line2D(
                [sep_x, sep_x], [bot - 0.02, top + 0.05],
                transform=fig.transFigure,
                color='#e65100', linewidth=2.0, linestyle='--',
            ))
        plt.savefig(vis_dir / f'samples_{suffix}.png', dpi=120, bbox_inches='tight')
        plt.close(fig)

    for ty in _VIS_TYS:
        _draw_vis_grid(by_ty[ty], ty)
    model.train()

def _assemble_slice_preview(
    patches: torch.Tensor,         # (N, 1, P, P) CPU
    grid_hw: tuple[int, int],
    slice_hw: tuple[int, int],
    patch_size: int,
    stride: int,
) -> np.ndarray:
    """Reconstruct a low-res slice preview by averaging overlapping patches."""
    H, W    = slice_hw
    n_rows, n_cols = grid_hw
    half    = patch_size // 2

    accum = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)

    ys = list(range(half, H - half + 1, stride))
    xs = list(range(half, W - half + 1, stride))
    if not ys or ys[-1] < H - half:
        ys.append(H - half)
    if not xs or xs[-1] < W - half:
        xs.append(W - half)
    ys = sorted(set(ys))
    xs = sorted(set(xs))

    idx = 0
    patches_np = patches[:, 0].numpy()   # (N, P, P)
    for cy in ys:
        for cx in xs:
            y0, y1 = max(cy - half, 0), min(cy + half, H)
            x0, x1 = max(cx - half, 0), min(cx + half, W)
            ph, pw  = y1 - y0, x1 - x0
            accum[y0:y1, x0:x1] += patches_np[idx, :ph, :pw]
            count[y0:y1, x0:x1] += 1.0
            idx += 1

    return accum / np.maximum(count, 1e-6)

# =============================================================================
#  Test evaluation
# =============================================================================

def save_evaluation_grid(
    grid_samples:    dict,
    eval_dir:        Path,
    filename:        str = 'grid.png',
    in_domain_fakes: set | None = None,
) -> None:
    """
    3 rows × 4 cols summary grid.  All columns share the same anchor (img_id, coord_z).
      Row 0: CT slice + green mask contour (bbox)
      Row 1: attention heatmap
      Row 2: CT + attention overlay
      Columns: real | in-domain fake(s) | OOD fake(s)
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    eval_dir.mkdir(parents=True, exist_ok=True)

    _fakes = [m for m in _VIS_COL_MODS if m != 'real']
    if in_domain_fakes is not None:
        _in_f  = [m for m in sorted(_fakes) if m in in_domain_fakes and m in grid_samples]
        _ood_f = [m for m in sorted(_fakes) if m not in in_domain_fakes and m in grid_samples]
    else:
        _in_f  = [m for m in sorted(_fakes) if m in grid_samples]
        _ood_f = []
    _real = ['real'] if 'real' in grid_samples else []
    cols  = _real + _in_f + _ood_f
    n_in  = len(_real) + len(_in_f)
    if not cols:
        return

    N_COLS     = len(cols)
    row_labels = ['CT + Mask', 'Attention', 'Overlay']

    fig, axes = plt.subplots(3, N_COLS, figsize=(3.5 * N_COLS, 9.5), squeeze=False,
                             facecolor='#111111')
    fig.suptitle(filename.replace('.png', '').replace('_', ' ').title(),
                 fontsize=11, fontweight='bold', color='white')

    for col_idx, mod in enumerate(cols):
        s      = grid_samples[mod]
        is_ood = col_idx >= n_in
        gt_str = 'real' if s['label'] == 0 else mod
        ok     = (s['label'] == 0) == (s['prob'] <= 0.5)
        attn_s = _smooth_attn(s['heatmap'])
        title  = (f"{mod.upper()}\n{s['img_id']}\n"
                  f"GT:{gt_str}  p={s['prob']:.2f}  {'OK' if ok else 'X'}")

        # Row 0: CT slice + green mask contour
        axes[0, col_idx].imshow(s['grid'], cmap='gray')
        if s['label'] > 0 and s['mask'].sum() > 0:
            axes[0, col_idx].contour(s['mask'], levels=[0.5], colors='lime', linewidths=1.2)
        axes[0, col_idx].set_title(title, fontsize=7, fontweight='bold',
                                   color='#e65100' if is_ood else 'white')
        axes[0, col_idx].axis('off')

        # Row 1: attention heatmap
        axes[1, col_idx].imshow(attn_s, cmap='turbo', vmin=0, vmax=1)
        axes[1, col_idx].axis('off')

        # Row 2: overlay
        axes[2, col_idx].imshow(s['grid'], cmap='gray')
        axes[2, col_idx].imshow(attn_s, cmap='turbo', alpha=0.45, vmin=0, vmax=1)
        if s['label'] > 0 and s['mask'].sum() > 0:
            axes[2, col_idx].contour(s['mask'], levels=[0.5], colors='lime', linewidths=1.2)
        axes[2, col_idx].axis('off')

    for row_idx, rl in enumerate(row_labels):
        axes[row_idx, 0].set_ylabel(rl, fontsize=9, fontweight='bold',
                                    color='#cccccc', rotation=90, labelpad=4)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if _ood_f and n_in < N_COLS:
        fig.canvas.draw()
        in0 = axes[0, 0].get_position()
        in1 = axes[0, n_in - 1].get_position()
        od0 = axes[0, n_in].get_position()
        od1 = axes[0, N_COLS - 1].get_position()
        top = axes[0, 0].get_position().y1
        bot = axes[2, 0].get_position().y0
        sep_x = (in1.x1 + od0.x0) / 2
        fig.text((in0.x0 + in1.x1) / 2, top + 0.025,
                 '── In-domain ──', ha='center', va='bottom',
                 fontsize=9, fontweight='bold', color='white')
        fig.text((od0.x0 + od1.x1) / 2, top + 0.025,
                 '── OOD ──', ha='center', va='bottom',
                 fontsize=9, fontweight='bold', color='#e65100')
        fig.add_artist(plt.Line2D(
            [sep_x, sep_x], [bot - 0.02, top + 0.04],
            transform=fig.transFigure,
            color='#e65100', linewidth=2.0, linestyle='--',
        ))

    plt.savefig(eval_dir / filename, dpi=120, bbox_inches='tight', facecolor='#111111')
    plt.close(fig)

def _save_single_vis(
    vis_dir: Path,
    name:    str,
    grid:    np.ndarray,
    mask:    np.ndarray,
    heatmap: np.ndarray,
    prob:    float,
    label:   int,
    mod:     str,
    img_id:  str,
) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    vis_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    gt   = 'real' if label == 0 else mod
    pred = 'fake' if prob > 0.5 else 'real'
    axes[0].imshow(grid,    cmap='gray');          axes[0].set_title('Slice',        fontsize=8); axes[0].axis('off')
    axes[1].imshow(mask,    cmap='gray');          axes[1].set_title('GT Mask',      fontsize=8); axes[1].axis('off')
    axes[2].imshow(heatmap, cmap='jet', vmin=0, vmax=1); axes[2].set_title('Attn Heatmap', fontsize=8); axes[2].axis('off')
    axes[3].imshow(grid, cmap='gray')
    axes[3].imshow(heatmap, cmap='jet', alpha=0.45, vmin=0, vmax=1)
    if label > 0 and mask.sum() > 0:
        axes[3].contour(mask, levels=[0.5], colors='lime', linewidths=1.2)
    axes[3].set_title(f'Overlay  GT:{gt}  Pred:{pred}(p={prob:.2f})', fontsize=7)
    axes[3].axis('off')
    fig.suptitle(f'{img_id} | {mod}', fontsize=9)
    plt.tight_layout()
    plt.savefig(vis_dir / f'{name.replace("/", "_")}.png', dpi=100, bbox_inches='tight')
    plt.close(fig)

_N_CAND = 50

@torch.no_grad()
def run_test_evaluation_slice(
    model,
    dl_test:         DataLoader,
    device:          torch.device,
    eval_dir:        Path,
    epoch:           int,
    run_dir:         Path,
    patch_size:      int,
    stride:          int,
    in_domain_fakes: set | None = None,
    seed:            int = 42,
    save_vis:        bool = True,
    max_vis:         int = 20,
) -> dict:
    """
    Full test evaluation. Saves to eval_dir/:
        metrics.json, per_sample_metrics.csv,
        grid_removal.png, grid_injection.png, vis/*.png

    Per-modality metrics use balanced equal sampling (fixed seed).
    """
    eval_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = eval_dir / 'vis' if save_vis else None
    model.eval()

    all_logits:  list = []
    all_labels:  list = []   # multiclass
    all_mods:    list = []
    all_img_ids: list = []
    per_sample_records: list = []

    # Candidates for visualization: {ty: {mod: [sample_dict, ...]}}
    cands: dict = {ty: {m: [] for m in _VIS_COL_MODS} for ty in _VIS_TYS}
    vis_count = 0

    progress = tqdm(dl_test, desc='Evaluating on test set', unit='batch')

    for batch in progress:
        patches        = batch['patches'].to(device)
        logits, attn_w = model(patches, return_attn=True)

        all_logits.append(logits.squeeze(1).cpu())
        all_labels.append(batch['label'])
        all_mods.extend(batch['mod'])
        all_img_ids.extend(batch['img_id'])

        attn_np  = attn_w.cpu().float().numpy()
        masks_np = batch['mask'][:, 0].numpy()
        probs_b  = sigmoid_np(logits.squeeze(1).cpu().numpy())

        for i in range(len(logits)):
            prob   = float(probs_b[i])
            label  = int(batch['label'][i])     # multiclass
            mod    = batch['mod'][i]
            img_id = batch['img_id'][i]

            per_sample_records.append({
                'img_id':    img_id,
                'mod':       mod,
                'label':     label,
                'bin_label': int(label > 0),
                'prob':      prob,
                'pred':      int(prob >= 0.5),
                'correct':   int((label == 0) == (prob <= 0.5)),
            })

            ty = _get_ty(img_id)
            need_heatmap = (
                (mod in _VIS_COL_MODS and len(cands[ty][mod]) < _N_CAND) or
                (save_vis and vis_count < max_vis)
            )
            if need_heatmap:
                gh   = (int(batch['grid_hw'][i, 0]), int(batch['grid_hw'][i, 1]))
                sh   = (int(batch['slice_hw'][i, 0]), int(batch['slice_hw'][i, 1]))
                mask = masks_np[i]
                hm   = reconstruct_heatmap(attn_np[i], gh, sh, patch_size, stride)
                grid = _assemble_slice_preview(batch['patches'][i], gh, sh, patch_size, stride)

                if mod in _VIS_COL_MODS and len(cands[ty][mod]) < _N_CAND:
                    cands[ty][mod].append({
                        'grid': grid, 'heatmap': hm, 'mask': mask,
                        'label': label, 'prob': prob, 'mod': mod,
                        'img_id': img_id, 'coord_z': int(batch['coord_z'][i]),
                    })

                if save_vis and vis_dir is not None and vis_count < max_vis:
                    _save_single_vis(
                        vis_dir=vis_dir, name=f'{img_id}_{mod}',
                        grid=grid, mask=mask, heatmap=_smooth_attn(hm),
                        prob=prob, label=label, mod=mod, img_id=img_id,
                    )
                    vis_count += 1

    # ── Aggregate classification metrics ──────────────────────────────────────
    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()   # multiclass
    bin_labels = (all_labels > 0).astype(int)
    probs      = sigmoid_np(all_logits)
    preds      = (probs >= 0.5).astype(int)
    mods_arr   = np.array(all_mods)

    cls_metrics: dict = {
        'auc':       float(roc_auc_score(bin_labels, probs)),
        'accuracy':  float(accuracy_score(bin_labels, preds)),
        'precision': float(precision_score(bin_labels, preds, zero_division=0)),
        'recall':    float(recall_score(bin_labels, preds, zero_division=0)),
        'f1':        float(f1_score(bin_labels, preds, zero_division=0)),
        'ap':        float(average_precision_score(bin_labels, probs)),
    }
    for mod in sorted(m for m in set(all_mods) if m != 'real'):
        for k, v in _balanced_mod_metrics(bin_labels, probs, preds, mods_arr, mod, seed).items():
            cls_metrics[f'{mod}_{k}'] = v

    results = {
        'split':          'test',
        'run_dir':        str(run_dir),
        'epoch':          epoch,
        'classification': cls_metrics,
    }

    # ── Save outputs ──────────────────────────────────────────────────────────
    with open(eval_dir / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Metrics JSON      -> {eval_dir / 'metrics.json'}")

    if per_sample_records:
        csv_path = eval_dir / 'per_sample_metrics.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(per_sample_records[0].keys()))
            writer.writeheader()
            writer.writerows(per_sample_records)
        print(f"  Per-sample CSV    -> {csv_path}")

    # ── Build visualization grids (shared anchor per ty) ──────────────────────
    grid_samples: dict = {ty: {} for ty in _VIS_TYS}
    for ty in _VIS_TYS:
        key_sets = [
            {(s['img_id'], s['coord_z']) for s in cands[ty][m]}
            for m in _VIS_COL_MODS if cands[ty][m]
        ]
        shared = set()
        if key_sets:
            shared = key_sets[0].intersection(*key_sets[1:]) if len(key_sets) > 1 else key_sets[0]
        anchor = next(iter(shared)) if shared else None
        for mod in _VIS_COL_MODS:
            clist = cands[ty][mod]
            if not clist:
                continue
            match = next((s for s in clist if anchor and (s['img_id'], s['coord_z']) == anchor), None)
            grid_samples[ty][mod] = match if match else clist[0]

        if grid_samples[ty]:
            fname = f'grid_{ty}.png'
            save_evaluation_grid(grid_samples[ty], eval_dir, filename=fname,
                                 in_domain_fakes=in_domain_fakes)
            print(f"  Summary grid ({ty}) -> {eval_dir / fname}")

    model.train()
    return results

# =============================================================================
#  Champion / Challenger promotion
# =============================================================================

def promote_if_better(temp_dir: Path, canonical_dir: Path) -> None:
    """
    Compare the challenger run (temp_dir) against the existing canonical model
    (canonical_dir) using test accuracy stored in test_metrics.json.

      - No canonical exists              -> rename temp to canonical.
      - Challenger test acc >= champion  -> replace canonical with temp.
      - Otherwise                        -> delete temp, leave canonical unchanged.
    """
    def _test_acc(run_dir: Path) -> float:
        jf = run_dir / 'test_metrics.json'
        if not jf.exists():
            return -1.0
        m = json.loads(jf.read_text())
        # Prefer OOD test accuracy — champion = best generalisation to unseen fakes
        v = m.get('ood_test_accuracy')
        if v is not None:
            try:
                fv = float(v)
                if fv == fv:  # not NaN
                    return fv
            except (TypeError, ValueError):
                pass
        return float(m.get('accuracy', m.get('global_acc', -1.0)))

    challenger_acc = _test_acc(temp_dir)
    print(f"\n{'─'*60}")
    print(f"  Champion/Challenger  ->  {canonical_dir.name}")
    print(f"{'─'*60}")

    if not canonical_dir.exists():
        print('  No champion found — promoting challenger as canonical.')
        shutil.move(str(temp_dir), str(canonical_dir))
        print(f'  ✓ Canonical: {canonical_dir}')
        return

    champion_acc = _test_acc(canonical_dir)
    print(f'  Champion   OOD test acc = {champion_acc:.4f}  ({canonical_dir.name})')
    print(f'  Challenger OOD test acc = {challenger_acc:.4f}  ({temp_dir.name})')

    if challenger_acc >= champion_acc:
        print('  -> Challenger wins! Replacing champion.')
        shutil.rmtree(str(canonical_dir))
        shutil.move(str(temp_dir), str(canonical_dir))
        print(f'  ✓ New canonical: {canonical_dir}')
    else:
        print('  -> Champion holds. Deleting challenger run.')
        shutil.rmtree(str(temp_dir))
        print(f'  ✓ Canonical unchanged: {canonical_dir}')

# =============================================================================
#  Main
# =============================================================================

def main(args: argparse.Namespace) -> None:

    # ── Debug mode ───────────────────────────────────────────────────────────
    if args.debug:
        args.wandb_mode = 'disabled'

    # ── Seed ─────────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Device ───────────────────────────────────────────────────────────────
    if args.gpu_id is not None:
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        gid = select_best_gpu()
        device = torch.device(f'cuda:{gid}') if gid is not None else torch.device('cpu')

    print(f"\n{'='*60}")
    print(f"  Phase B Training — SA-ABMIL slice classifier")
    print(f"{'='*60}")
    print(f"  Device:       {device}")
    print(f"  Backbone:     {args.backbone} (pretrained={args.pretrained})")
    print(f"  Patch size:   {args.patch_size}  stride: {args.stride or 'auto (patch_size//2)'}")
    print(f"  proj_dim:     {args.proj_dim}   attn_dim: {args.attn_dim}   dropout: {args.dropout}")
    print(f"  SA Transformer: {args.sa_n_layers} layers × {args.sa_n_heads} heads")
    print(f"  Loss:         BCE{' + attn KL (w=' + str(args.aux_weight) + ')' if args.aux_attn_loss else ''}")
    print(f"  LR={args.lr}  WD={args.weight_decay}  BS={args.batch_size}  epochs={args.epochs}")
    print(f"  AMP={args.amp}  balance={args.balance_classes}  seed={args.seed}")

    # ── Run name / directory structure ────────────────────────────────────────
    #   runs/
    #     slice-cls_{backbone}_p{ps}_s{st}/
    #       trained_on_all/             ← canonical
    #       trained_on_pix2pix/         ← canonical (cross-domain run)
    #       trained_on_all_{hp_tag}/    ← challenger temp dir
    stride_eff = args.stride or (args.patch_size // 2)
    loss_tag   = 'bce' if not args.aux_attn_loss else 'bce+attn'
    all_fake   = set(ALL_FAKES)
    fake_mods  = [m for m in args.mods if m != 'real']
    ood_fakes  = [m for m in ALL_FAKES if m not in set(fake_mods)]
    mods_tag   = 'all' if set(fake_mods) == all_fake else '+'.join(sorted(fake_mods))
    base_dir   = WORK_DIR / 'experiments' / 'SelfAttention' / 'runs' / f"sa-slice-cls_{args.backbone}_p{args.patch_size}_s{stride_eff}"
    if args.run_name is None:
        run_tag      = f"trained_on_{mods_tag}_{loss_tag}_bs{args.batch_size}_lr{args.lr}"
        wandb_group  = f"SA_trained_on_{mods_tag}"
        wandb_name   = f"sa-slice-cls_{args.backbone}_p{args.patch_size}_s{stride_eff}_{loss_tag}_bs{args.batch_size}_lr{args.lr}"
    else:
        run_tag      = args.run_name
        wandb_group  = f"trained_on_{mods_tag}"
        wandb_name   = args.run_name
    canonical_dir = base_dir / f'trained_on_{mods_tag}'
    out_dir       = base_dir / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output dir:   {out_dir}\n")

    # Save args (include use_sa flag so volume-cls SA can auto-detect)
    saved_args = vars(args).copy()
    saved_args['use_sa'] = True
    with open(out_dir / 'args.json', 'w') as f:
        json.dump(saved_args, f, indent=2)

    # ── WandB ────────────────────────────────────────────────────────────────
    if args.wandb_mode != 'disabled':
        import wandb
        wandb.init(project='MedForensics', group=wandb_group, name=wandb_name, config=vars(args), mode=args.wandb_mode)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("Loading data...")
    dl_train, dl_valid, dl_test, actual_stride, dl_val_ood, dl_test_ood = build_dataloaders(args)
    print()

    # ── Model ────────────────────────────────────────────────────────────────
    print(f"Building SA-ABMIL model...")
    model = build_sa_classifier_scratch(
        backbone    = args.backbone,
        pretrained  = args.pretrained,
        proj_dim    = args.proj_dim,
        attn_dim    = args.attn_dim,
        dropout     = args.dropout,
        sa_n_heads  = args.sa_n_heads,
        sa_n_layers = args.sa_n_layers,
    ).to(device)

    total_p   = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {total_p:,} params  (full model — end-to-end)")
    print()

    # ── Optimiser & scheduler ────────────────────────────────────────────────
    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    main_scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode='min', factor=0.5, patience=10, min_lr=1e-6,
        )
        if args.scheduler else None
    )

    # ── Loss ─────────────────────────────────────────────────────────────────
    criterion = SliceLoss(aux_attn=args.aux_attn_loss, aux_weight=args.aux_weight)

    # ── AMP scaler ───────────────────────────────────────────────────────────
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_loss    = float('inf')
    patience_counter = 0
    history          = []

    print("Starting training...\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Warmup LR
        if epoch <= args.warmup_epochs:
            wlr = args.lr * (epoch / args.warmup_epochs)
            for pg in optimiser.param_groups:
                pg['lr'] = wlr

        train_losses = train_one_epoch(
            model, dl_train, criterion, optimiser, scaler, device,
            args.amp, args.patch_size, actual_stride,
        )

        val_metrics = evaluate(
            model, dl_valid, criterion, device, args.patch_size, actual_stride,
        )
        ood_metrics = (
            evaluate(model, dl_val_ood, criterion, device, args.patch_size, actual_stride)
            if dl_val_ood is not None else {}
        )

        if epoch > args.warmup_epochs and main_scheduler is not None:
            main_scheduler.step(val_metrics['loss'])

        current_lr = optimiser.param_groups[0]['lr']
        elapsed    = time.time() - t0

        _ood_loss = f" | Val-OOD {ood_metrics['loss']:.4f}" if ood_metrics else ""
        log_str = (
            f"\nEpoch {epoch:03d}/{args.epochs} | "
            f"Train {train_losses['total']:.4f}"
        )
        if 'attn' in train_losses:
            log_str += f" (cls={train_losses['cls']:.4f}, attn={train_losses['attn']:.4f})"
        log_str += (
            f" | Val {val_metrics['loss']:.4f}{_ood_loss}"
            f" | In:  AUC {val_metrics['auc']:.4f}  AP {val_metrics['ap']:.4f}"
            f"  Acc {val_metrics['accuracy']:.4f}  F1 {val_metrics['f1']:.4f}"
        )
        if ood_metrics and ood_fakes:
            # Macro-average over OOD modalities — consistent with per-mod breakdown below.
            # (Global ood_metrics aggregate is skewed by 1:2 class ratio in dl_val_ood.)
            def _ood_macro(k: str) -> float:
                vals = [ood_metrics[f'{m}_{k}'] for m in ood_fakes
                        if not np.isnan(ood_metrics.get(f'{m}_{k}', float('nan')))]
                return float(np.mean(vals)) if vals else float('nan')
            log_str += (
                f" | OOD: AUC {_ood_macro('auc'):.4f}  AP {_ood_macro('ap'):.4f}"
                f"  Acc {_ood_macro('acc'):.4f}  F1 {_ood_macro('f1'):.4f}"
            )
        log_str += f" | {elapsed:.1f}s"
        print(log_str)

        if fake_mods:
            print("  ─── In-domain " + "─" * 58)
            for m in sorted(fake_mods):
                a, ac, ap, f1 = (val_metrics.get(f'{m}_{k}', float('nan'))
                                 for k in ['auc', 'acc', 'ap', 'f1'])
                print(f"  {m:<10s}  AUC {a:.4f}  Acc {ac:.4f}  AP {ap:.4f}  F1 {f1:.4f}")
                for ty_label, ty in [("Removal", "rem"), ("Injection", "inj")]:
                    ra, rac, rap, rf = (val_metrics.get(f'{m}_{ty}_{k}', float('nan'))
                                        for k in ['auc', 'acc', 'ap', 'f1'])
                    print(f"    {ty_label:<9s}  AUC {ra:.4f}  Acc {rac:.4f}  AP {rap:.4f}  F1 {rf:.4f}")
        if ood_metrics and ood_fakes:
            print("  ─── OOD " + "─" * 64)
            for m in sorted(ood_fakes):
                a, ac, ap, f1 = (ood_metrics.get(f'{m}_{k}', float('nan'))
                                 for k in ['auc', 'acc', 'ap', 'f1'])
                print(f"  {m:<10s}  AUC {a:.4f}  Acc {ac:.4f}  AP {ap:.4f}  F1 {f1:.4f}")
                for ty_label, ty in [("Removal", "rem"), ("Injection", "inj")]:
                    ra, rac, rap, rf = (ood_metrics.get(f'{m}_{ty}_{k}', float('nan'))
                                        for k in ['auc', 'acc', 'ap', 'f1'])
                    print(f"    {ty_label:<9s}  AUC {ra:.4f}  Acc {rac:.4f}  AP {rap:.4f}  F1 {rf:.4f}")

        # Visualisation
        if args.vis_every > 0 and epoch % args.vis_every == 0:
            vis_dir = out_dir / 'vis' / f'epoch_{epoch:03d}'
            try:
                save_epoch_vis(
                    model, dl_valid, device, vis_dir,
                    args.patch_size, actual_stride, args.vis_n_samples,
                    in_domain_fakes=set(fake_mods),
                    ood_dataloader=dl_val_ood,
                )
                print(f"  → Vis saved: {vis_dir}/samples_removal.png + samples_injection.png")
                if args.wandb_mode != 'disabled':
                    vis_log = {}
                    for png in sorted(vis_dir.glob('samples_*.png')):
                        vis_log[f'vis/{png.stem}'] = wandb.Image(str(png))
                    if vis_log:
                        wandb.log(vis_log)
            except Exception as e:
                print(f"  [WARN] save_epoch_vis failed at epoch {epoch}: {e}")

        # WandB
        if args.wandb_mode != 'disabled':
            log_dict = {
                'train/loss':  train_losses['total'],
                'val/loss':    val_metrics['loss'],
                'val/auc':     val_metrics['auc'],
                'val/acc':     val_metrics['accuracy'],
                'val/ap':      val_metrics['ap'],
                'val/f1':      val_metrics['f1'],
            }
            if 'attn' in train_losses:
                log_dict['train/loss_cls']  = train_losses['cls']
                log_dict['train/loss_attn'] = train_losses['attn']
            for m in sorted(fake_mods):
                for k in ['auc', 'acc', 'ap', 'f1']:
                    log_dict[f'val/in/{m}/{k}'] = val_metrics.get(f'{m}_{k}', float('nan'))
                for ty in ['rem', 'inj']:
                    for k in ['auc', 'acc', 'ap', 'f1']:
                        log_dict[f'val/in/{m}/{ty}_{k}'] = val_metrics.get(f'{m}_{ty}_{k}', float('nan'))
            if ood_metrics:
                log_dict['val/ood/loss'] = ood_metrics['loss']
                log_dict['val/ood/auc']  = ood_metrics['auc']
                log_dict['val/ood/acc']  = ood_metrics['accuracy']
                log_dict['val/ood/ap']   = ood_metrics['ap']
                log_dict['val/ood/f1']   = ood_metrics['f1']
                for m in sorted(ood_fakes):
                    for k in ['auc', 'acc', 'ap', 'f1']:
                        log_dict[f'val/ood/{m}/{k}'] = ood_metrics.get(f'{m}_{k}', float('nan'))
                    for ty in ['rem', 'inj']:
                        for k in ['auc', 'acc', 'ap', 'f1']:
                            log_dict[f'val/ood/{m}/{ty}_{k}'] = ood_metrics.get(f'{m}_{ty}_{k}', float('nan'))
            wandb.log(log_dict)

        # History
        history.append({
            'epoch':       epoch,
            'lr':          current_lr,
            'train_loss':  train_losses,
            'val_metrics': val_metrics,
            'ood_metrics': ood_metrics,
            'time_s':      elapsed,
        })

        # Checkpoint
        if val_metrics['loss'] < best_val_loss:
            best_val_loss    = val_metrics['loss']
            patience_counter = 0
            torch.save({
                'epoch':              epoch,
                'model_state_dict':   model.state_dict(),
                'optimiser_state_dict': optimiser.state_dict(),
                'best_val_loss':      best_val_loss,
                'best_val_auc':       val_metrics['auc'],
                'args':               vars(args),
            }, out_dir / 'best_model.pt')
            print(f"  ★ New best val loss: {best_val_loss:.4f}  (AUC {val_metrics['auc']:.4f}) — saved")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  Early stopping at epoch {epoch} (patience={args.patience})")
                break

    # Save training history
    with open(out_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2, default=str)

    # ── Final test evaluation ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Test Evaluation (best model)")
    print(f"{'='*60}\n")

    ckpt = torch.load(out_dir / 'best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    test_metrics = evaluate(
        model, dl_test, criterion, device, args.patch_size, actual_stride,
    )

    for k, v in sorted(test_metrics.items()):
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.4f}")

    # ── OOD test evaluation (champion metric) ────────────────────────────────
    ood_fakes = [m for m in ALL_FAKES if m not in set(fake_mods)]
    if dl_test_ood is not None:
        print("\n  [OOD Test] evaluating on held-out fakes...")
        ood_test_m = evaluate(model, dl_test_ood, criterion, device, args.patch_size, actual_stride)
        test_metrics['ood_test_accuracy'] = ood_test_m['accuracy']
        test_metrics['ood_test_auc']      = ood_test_m['auc']
        test_metrics['ood_test_ap']       = ood_test_m['ap']
        test_metrics['ood_test_f1']       = ood_test_m['f1']
        print(f"  OOD Test  AUC {ood_test_m['auc']:.4f}  AP {ood_test_m['ap']:.4f}"
              f"  Acc {ood_test_m['accuracy']:.4f}  F1 {ood_test_m['f1']:.4f}"
              f"  \u2190 champion metric")

    with open(out_dir / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2, default=str)

    # ── In-domain vs Out-of-domain summary ───────────────────────────────────
    if ood_fakes:
        print(f"\n  In-domain fakes : {fake_mods}")
        print(f"  Out-of-domain   : {ood_fakes}")
        print("  ─── In-domain ───────────────────────────────────")
        for m in fake_mods:
            k = f'{m}_auc'
            if k in test_metrics and not np.isnan(test_metrics[k]):
                print(f"    {m:12s}: AUC {test_metrics[k]:.4f}")
        print("  ─── Out-of-domain (OOD) ─────────────────────────")
        for m in ood_fakes:
            k = f'{m}_auc'
            if k in test_metrics and not np.isnan(test_metrics[k]):
                print(f"    {m:12s}: AUC {test_metrics[k]:.4f}")
            else:
                print(f"    {m:12s}: not in test set — add --test_mods to include OOD fakes")

    # ── Test-phase visualisation ──────────────────────────────────────────────
    print("\nSaving test visualisations...")
    test_vis_dir = out_dir / 'vis' / 'test'
    try:
        save_epoch_vis(
            model, dl_test, device, test_vis_dir,
            args.patch_size, actual_stride, args.vis_n_samples,
            in_domain_fakes=set(fake_mods),
        )
        print(f"  → Test vis saved: {test_vis_dir}")
    except Exception as e:
        print(f"  [WARN] save_epoch_vis failed: {e}")

    if args.wandb_mode != 'disabled':
        wandb.log({f'test/{k}': v for k, v in test_metrics.items()})
        vis_log = {}
        for png in sorted(test_vis_dir.glob('samples_*.png')):
            vis_log[f'vis/test/{png.stem}'] = wandb.Image(str(png))
        if vis_log:
            wandb.log(vis_log)
            print(f"  → Logged {len(vis_log)} test visuals to WandB")
        wandb.finish()

    promote_if_better(out_dir, canonical_dir)

    print(f"\nAll outputs saved to: {out_dir}")
    print("Done.")

if __name__ == '__main__':
    args = get_args()
    main(args)
