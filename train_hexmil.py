#!/usr/bin/env python3
"""
train_hexmil.py
===============
Stage 2 — HexMIL training: volume-level binary forgery classification.

Loads a frozen SliceMIL (Stage 1) checkpoint as the slice encoder and trains
a Gated MIL aggregator over K slices plus a binary classification head.

Usage:
    python train_hexmil.py \\
        --slice_ckpt_dir runs/slice-cls_resnet50_p64_s32/trained_on_all \\
        --K 32 --batch_size 8 --epochs 60
"""

from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score,
)
from scipy.special import expit as sigmoid_np
import GPUtil

from hexmil.data.patch_dataset import load_split_table
from hexmil.data.volume_dataset import VolumeDataset
from hexmil.data.slice_dataset import reconstruct_heatmap
from hexmil.models.volume_classifier import VolumeClassifier, build_volume_classifier

# ── paths ────────────────────────────────────────────────────────────────────
from config import DATA_DIR, WORK_DIR
ALL_FAKES = ['pix2pix', 'cycle', 'diffusion', 'ctgan']


# =============================================================================
#  GPU selection
# =============================================================================

def select_best_gpu() -> int | None:
    if not torch.cuda.is_available():
        print("No CUDA available — using CPU")
        return None
    gpus = GPUtil.getGPUs()
    if not gpus:
        return 0
    best = max(gpus, key=lambda g: g.memoryFree)
    print(f"Auto-selected GPU {best.id}: {best.name} "
          f"(Free: {best.memoryFree}MB / {best.memoryTotal}MB)")
    return best.id


# =============================================================================
#  Args
# =============================================================================

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Stage 2: volume-level classification')

    p.add_argument('-ckpt_dir', '--slice_ckpt_dir', type=str, required=True,
                   help='Path to the Stage 1 run directory containing args.json '
                        'and best_model.pt')

    # ── Volume window ─────────────────────────────────────────────────────────
    p.add_argument('--K', type=int, default=32,
                   help='Number of slices in the window')

    # ── Modalities ───────────────────────────────────────────────────────────
    p.add_argument('--mods', nargs='+', default=None,
                   help='Modalities for train/val. If not set, inferred automatically from --slice_ckpt_dir args.json.')

    # ── Volume aggregator ──────────────────────────────────────────────────────
    p.add_argument('--attn_dim', type=int, default=256,
                   help='Hidden dim for the gated Z-aggregator')
    p.add_argument('--dropout',  type=float, default=0.25)

    # ── Training ─────────────────────────────────────────────────────────────
    p.add_argument('--epochs',       type=int,   default=60)
    p.add_argument('--batch_size',   type=int,   default=16,
                   help='Volumes per batch (keep small because K slices each)')
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--balance_classes', action='store_true', default=True)

    # ── Scheduler / early stopping ────────────────────────────────────────────
    p.add_argument('--scheduler',      action='store_true', default=True)
    p.add_argument('--patience',       type=int, default=15)
    p.add_argument('--warmup_epochs',  type=int, default=3)

    # ── Misc ─────────────────────────────────────────────────────────────────
    p.add_argument('--amp',            action='store_true', default=True)
    p.add_argument('--num_workers',    type=int, default=4)
    p.add_argument('--seed',           type=int, default=42)
    p.add_argument('--gpu_id',         type=int, default=None)
    p.add_argument('--run_name',       type=str, default=None)

    # ── Visualisation ─────────────────────────────────────────────────────────
    p.add_argument('--vis_every',     type=int, default=5,
                   help='Save visualisations every N epochs (0 = never)')

    # ── WandB ────────────────────────────────────────────────────────────────
    p.add_argument('--wandb_mode', default='online', choices=['online', 'offline', 'disabled'],
                   help='WandB logging mode')
    p.add_argument('--debug',      action='store_true',
                   help='Debug mode: sets wandb_mode to disabled')

    return p.parse_args()


# =============================================================================
#  Data utilities
# =============================================================================

def build_dataloaders(args: argparse.Namespace, sargs: dict):
    """Build train / in_valid / valid / test DataLoaders for VolumeDataset."""
    patch_size = sargs['patch_size']
    stride     = sargs.get('stride') or (patch_size // 2)

    fake_mods  = [m for m in args.mods if m != 'real']
    train_mods = ['real'] + fake_mods
    # valid and test always use ALL fake mods
    all_mods   = ['real'] + ALL_FAKES

    tab_train    = load_split_table(DATA_DIR, 'train', train_mods)
    tab_in_valid = load_split_table(DATA_DIR, 'valid', train_mods)   # in-domain: overfitting monitor
    tab_valid    = load_split_table(DATA_DIR, 'valid', all_mods)     # full valid: checkpoint selection
    tab_test     = load_split_table(DATA_DIR, 'test',  all_mods)

    ds_train    = VolumeDataset(DATA_DIR, tab_train,    K=args.K, patch_size=patch_size, stride=stride, augment=True)
    ds_in_valid = VolumeDataset(DATA_DIR, tab_in_valid, K=args.K, patch_size=patch_size, stride=stride, augment=False)
    ds_valid    = VolumeDataset(DATA_DIR, tab_valid,    K=args.K, patch_size=patch_size, stride=stride, augment=False)
    ds_test     = VolumeDataset(DATA_DIR, tab_test,     K=args.K, patch_size=patch_size, stride=stride, augment=False)

    # Weighted sampler for binary class balance (real vs any-fake)
    sampler = None
    if args.balance_classes:
        labels    = tab_train['mod'].apply(lambda m: 0 if m == 'real' else 1).values
        cls_count = np.bincount(labels)
        cls_w     = 1.0 / np.maximum(cls_count, 1)
        sample_w  = cls_w[labels]
        sampler   = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

    loader_kw   = dict(num_workers=args.num_workers, pin_memory=True)
    dl_train    = DataLoader(ds_train,    batch_size=args.batch_size, sampler=sampler, shuffle=(sampler is None), **loader_kw)
    dl_in_valid = DataLoader(ds_in_valid, batch_size=args.batch_size, shuffle=False, **loader_kw)
    dl_valid    = DataLoader(ds_valid,    batch_size=args.batch_size, shuffle=False, **loader_kw)
    dl_test     = DataLoader(ds_test,     batch_size=args.batch_size, shuffle=False, **loader_kw)

    print(f"  Train:{len(ds_train):5d}  |  In-Valid:{len(ds_in_valid):5d}  |  "
          f"Valid:{len(ds_valid):5d}  |  Test:{len(ds_test):5d}")
    return dl_train, dl_in_valid, dl_valid, dl_test


# =============================================================================
#  Pd@1% helper
# =============================================================================

def _pd_at_1pct(y_true, y_score):
    """TPR at 1% FPR on the ROC curve (Pd@1%)."""
    if len(np.unique(y_true)) < 2:
        return float('nan')
    fpr, tpr, _ = roc_curve(np.array(y_true), np.array(y_score))
    idx = int(np.argmin(np.abs(fpr - 0.01)))
    return float(tpr[idx])


# =============================================================================
#  Balanced per-modality metrics helper
# =============================================================================

def _balanced_mod_metrics(
    bin_labels: np.ndarray,
    probs:      np.ndarray,
    preds:      np.ndarray,
    mods_arr:   np.ndarray,
    mod:        str,
    seed:       int = 42,
) -> dict:
    """AUC/ACC/F1/AP/Pd@1% for `mod` vs real using equal balanced sampling (fixed seed)."""
    real_idx = np.where(mods_arr == 'real')[0]
    fake_idx = np.where(mods_arr == mod)[0]
    if len(fake_idx) == 0 or len(real_idx) == 0:
        return {k: float('nan') for k in ['auc', 'acc', 'f1', 'ap', 'pd_at_1']}
    n   = min(len(real_idx), len(fake_idx))
    rng = np.random.default_rng(seed)
    sel = np.concatenate([
        rng.choice(real_idx, n, replace=False),
        rng.choice(fake_idx, n, replace=False),
    ])
    lbl = bin_labels[sel]; prb = probs[sel]; prd = preds[sel]
    if lbl.sum() < 1 or (lbl == 0).sum() < 1:
        return {k: float('nan') for k in ['auc', 'acc', 'f1', 'ap', 'pd_at_1']}
    try:
        auc = float(roc_auc_score(lbl, prb))
    except Exception:
        auc = float('nan')
    return {
        'auc':      auc,
        'acc':      float(accuracy_score(lbl, prd)),
        'f1':       float(f1_score(lbl, prd, zero_division=0)),
        'ap':       float(average_precision_score(lbl, prb)),
        'pd_at_1':  _pd_at_1pct(lbl, prb),
    }


# =============================================================================
#  Evaluate
# =============================================================================

@torch.no_grad()
def evaluate(
    model: VolumeClassifier,
    dataloader: DataLoader,
    criterion: nn.BCEWithLogitsLoss,
    device: torch.device,
    seed: int = 42,
) -> dict:
    model.eval()

    all_logits: list = []
    all_labels: list = []   # multiclass (0-3)
    all_mods:   list = []
    running_loss = 0.0
    n_batches    = 0

    for batch in dataloader:
        patches_seq = batch['patches'].to(device)
        z_indices   = batch['z_indices'].to(device)
        labels      = batch['label'].to(device)         # multiclass
        labels_bin  = (labels > 0).float()              # binary for BCE

        B = patches_seq.shape[0]
        logits_list = []
        for b in range(B):
            valid_mask = (z_indices[b] >= 0)
            logit, _   = model(patches_seq[b], z_indices[b], valid_mask)
            logits_list.append(logit)

        logits = torch.stack(logits_list).squeeze(-1)
        loss   = criterion(logits, labels_bin)
        running_loss += loss.item()
        n_batches    += 1

        all_logits.append(logits.cpu())
        all_labels.append(batch['label'])
        all_mods.extend(batch['mod'])

    all_logits_np = torch.cat(all_logits).numpy()
    all_labels_np = torch.cat(all_labels).numpy()   # multiclass
    bin_labels    = (all_labels_np > 0).astype(int)
    probs         = sigmoid_np(all_logits_np)
    preds         = (probs >= 0.5).astype(int)
    mods_arr      = np.array(all_mods)

    metrics = {
        'loss':     running_loss / max(n_batches, 1),
        'auc':      float(roc_auc_score(bin_labels, probs)),
        'accuracy': float(accuracy_score(bin_labels, preds)),
        'ap':       float(average_precision_score(bin_labels, probs)),
        'f1':       float(f1_score(bin_labels, preds, zero_division=0)),
        'pd_at_1':  _pd_at_1pct(bin_labels, probs),
    }

    for mod in sorted(m for m in set(all_mods) if m != 'real'):
        for k, v in _balanced_mod_metrics(bin_labels, probs, preds, mods_arr, mod, seed).items():
            metrics[f'{mod}_{k}'] = v

    return metrics


# =============================================================================
#  Training
# =============================================================================

def train_one_epoch(
    model: VolumeClassifier,
    dataloader: DataLoader,
    criterion: nn.BCEWithLogitsLoss,
    optimiser: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
) -> dict:
    model.train()

    running_loss = 0.0
    n_batches    = 0

    for batch in dataloader:
        patches_seq = batch['patches'].to(device)     # (B, K, N, 1, P, P)
        z_indices   = batch['z_indices'].to(device)
        labels      = batch['label'].to(device)         # multiclass (0-3)
        labels_bin  = (labels > 0).float()              # binary for BCE

        optimiser.zero_grad(set_to_none=True)

        B = patches_seq.shape[0]
        logits_list = []
        with torch.amp.autocast('cuda', enabled=use_amp):
            for b in range(B):
                valid_mask = (z_indices[b] >= 0)
                logit, _   = model(patches_seq[b], z_indices[b], valid_mask)
                logits_list.append(logit)

        logits = torch.stack(logits_list).squeeze(-1)   # (B,)
        loss   = criterion(logits, labels_bin)

        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        scaler.step(optimiser)
        scaler.update()

        running_loss += loss.item()
        n_batches    += 1

    return {'cls': running_loss / max(n_batches, 1),
            'total': running_loss / max(n_batches, 1)}


# =============================================================================
#  Periodic visualisation: β heatmaps per volume
# =============================================================================

def _get_ty(img_id: str) -> str:
    return 'removal' if str(img_id).startswith('rem_') else 'injection'


# Kept for eval_volume.py backward compatibility via importlib.
_VIS_COL_MODS = ['real', 'cycle', 'diffusion', 'pix2pix']
_VIS_TYS      = ['removal', 'injection']


def _smooth_attn(a: np.ndarray) -> np.ndarray:
    """Smooth a 2-D attention map and renormalise to [0, 1]."""
    from scipy.ndimage import gaussian_filter
    sigma = max(1.0, max(a.shape) * 0.03)
    a = gaussian_filter(a.astype(np.float32), sigma=sigma)
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo + 1e-8)


def compute_xai_metrics(heatmap: np.ndarray, mask: np.ndarray) -> dict:
    """
    Compute pixel-level XAI localisation metrics for a single 2-D slice.

    Args:
        heatmap: float32 (H, W)   — already in [0,1]
        mask:    float32 (H, W)   — binary GT mask

    Returns dict with: pixel_auc, iou_03, iou_05, iou_07, pointing_game
    """
    result = {}
    m_bin = (mask > 0.5).astype(bool)

    # Pixel-level ROC AUC
    if m_bin.sum() > 0 and (~m_bin).sum() > 0:
        result['pixel_auc'] = float(roc_auc_score(m_bin.ravel(), heatmap.ravel()))
    else:
        result['pixel_auc'] = float('nan')

    # IoU at thresholds
    for thr in [0.3, 0.5, 0.7]:
        h_bin = heatmap >= thr
        inter = (h_bin & m_bin).sum()
        union = (h_bin | m_bin).sum()
        result[f'iou_{int(thr*10):02d}'] = float(inter / union) if union > 0 else 0.0

    # Pointing game: peak of heatmap inside mask
    ym, xm = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    result['pointing_game'] = float(m_bin[ym, xm])

    return result


# =============================================================================
#  3-D visualisation utilities (shared by train & eval)
# =============================================================================

def _norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-8)


def _build_volume_3d(
    patches_cpu: torch.Tensor,    # (K, N, 1, P, P)  CPU
    grid_hw_arr: np.ndarray,      # (K, 2)
    slice_hw:    tuple,           # (H, W)
    patch_size:  int,
    stride:      int,
    valid_mask:  np.ndarray,      # (K,) bool
) -> np.ndarray:                  # (K, H, W) float32
    """Reconstruct a 3-D volume by averaging overlapping patches slice by slice."""
    K    = patches_cpu.shape[0]
    H, W = slice_hw
    vol  = np.zeros((K, H, W), dtype=np.float32)
    half = patch_size // 2
    ys   = sorted(set(list(range(half, H - half + 1, stride)) + [H - half]))
    xs   = sorted(set(list(range(half, W - half + 1, stride)) + [W - half]))
    for k in range(K):
        if not valid_mask[k]:
            continue
        pnp   = patches_cpu[k, :, 0].numpy()   # (N, P, P)
        accum = np.zeros((H, W), dtype=np.float32)
        cnt   = np.zeros((H, W), dtype=np.float32)
        idx   = 0
        for cy in ys:
            for cx in xs:
                y0, y1 = max(cy - half, 0), min(cy + half, H)
                x0, x1 = max(cx - half, 0), min(cx + half, W)
                accum[y0:y1, x0:x1] += pnp[idx, :y1 - y0, :x1 - x0]
                cnt[y0:y1, x0:x1]   += 1.0
                idx += 1
        vol[k] = accum / np.maximum(cnt, 1e-6)
    return vol


def _build_heatmap_3d(
    alpha_cpu:   list,            # list[(N,) float32 np]  len=K
    beta_np:     np.ndarray,      # (K,)
    grid_hw_arr: np.ndarray,      # (K, 2)
    slice_hw:    tuple,           # (H, W)
    patch_size:  int,
    stride:      int,
    valid_mask:  np.ndarray,      # (K,) bool
) -> np.ndarray:                  # (K, H, W) float32
    """Build β-weighted per-slice attention heatmap volume."""
    K    = len(alpha_cpu)
    H, W = slice_hw
    hmap = np.zeros((K, H, W), dtype=np.float32)
    for k in range(K):
        if not valid_mask[k]:
            continue
        gh      = (int(grid_hw_arr[k, 0]), int(grid_hw_arr[k, 1]))
        a2d     = reconstruct_heatmap(alpha_cpu[k], gh, slice_hw, patch_size, stride)
        hmap[k] = np.clip(a2d * float(beta_np[k]), 0, 1)
    return hmap


def _smooth_3d(hmap3d: np.ndarray) -> np.ndarray:
    """Apply per-slice Gaussian smoothing and normalise whole volume to [0, 1]."""
    from scipy.ndimage import gaussian_filter
    out = np.zeros_like(hmap3d)
    for k in range(hmap3d.shape[0]):
        sl = hmap3d[k]
        if sl.max() > 0:
            sl = gaussian_filter(sl.astype(np.float32),
                                 sigma=max(1.0, max(sl.shape) * 0.03))
        out[k] = sl
    lo, hi = out.min(), out.max()
    return (out - lo) / (hi - lo + 1e-8)


def _save_volume_gif(
    volume_3d:   np.ndarray,
    heatmap_3d:  np.ndarray,
    masks_3d:    np.ndarray,
    beta_np:     np.ndarray,
    z_indices:   np.ndarray,
    valid_mask:  np.ndarray,
    label:       int,
    prob:        float,
    mod:         str,
    img_id:      str,
    save_path:   Path,
    duration_ms: int = 250,
) -> None:
    """Animated GIF scrolling through K axial slices (used by eval_volume.py)."""
    try:
        from PIL import Image
        import io
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    gt_str = 'real' if label == 0 else mod
    pr_str = 'fake' if prob > 0.5 else 'real'
    frames = []

    for k in range(volume_3d.shape[0]):
        if not valid_mask[k]:
            continue
        sl_n = _norm01(volume_3d[k])
        hm_n = _norm01(heatmap_3d[k]) if heatmap_3d[k].max() > 0 else heatmap_3d[k]
        mk   = masks_3d[k]
        b_k  = float(beta_np[k])
        z_k  = int(z_indices[k])

        fig, axs = plt.subplots(1, 3, figsize=(9, 3.2))
        axs[0].imshow(sl_n, cmap='gray');  axs[0].set_title('Volume', fontsize=11);  axs[0].axis('off')
        if label > 0 and mk.sum() > 0:
            axs[0].contour(mk, levels=[0.5], colors='lime', linewidths=0.8)
        axs[1].imshow(hm_n, cmap='turbo', vmin=0, vmax=1)
        axs[1].set_title(f'Attention  β={b_k:.3f}', fontsize=11);  axs[1].axis('off')
        axs[2].imshow(sl_n, cmap='gray')
        axs[2].imshow(hm_n, cmap='turbo', alpha=0.45, vmin=0, vmax=1)
        axs[2].set_title('Overlay', fontsize=11);  axs[2].axis('off')
        if label > 0 and mk.sum() > 0:
            axs[2].contour(mk, levels=[0.5], colors='lime', linewidths=0.8)
        fig.suptitle(f'{img_id} | {mod} | z={z_k}  GT:{gt_str} Pred:{pr_str}(p={prob:.2f})', fontsize=11)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()

    if frames:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)


@torch.no_grad()
def save_nodule_grid(
    grid_samples:    dict,
    out_dir:         Path,
    filename:        str = 'nodule_grid.png',
    in_domain_fakes: set | None = None,
) -> None:
    """
    5 rows × 12 cols nodule-context summary grid (Stage 2).
    Rows (top→bottom): z+2 | z+1 | ★ nodule (bbox) | z-1 | z-2
    Col groups (×4 mods): [ CT  |  Attention  |  Overlay ]
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import matplotlib.patches as mpatches
    except ImportError:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    _all_f = sorted(m for m in grid_samples if m != 'real')
    if in_domain_fakes is not None:
        _in_fake  = [m for m in sorted(_all_f) if m in in_domain_fakes and m in grid_samples]
        _ood_fake = [m for m in sorted(_all_f) if m not in in_domain_fakes and m in grid_samples]
    else:
        _in_fake  = [m for m in sorted(_all_f) if m in grid_samples]
        _ood_fake = []
    _real = ['real'] if 'real' in grid_samples else []
    mods  = _real + _in_fake + _ood_fake
    n_in  = len(_real) + len(_in_fake)
    if not mods:
        return

    N_OFF  = 2
    N_ROWS = 2 * N_OFF + 1
    N_IC   = 3
    N_BLK  = len(mods)

    fig = plt.figure(
        figsize=(3.0 * N_IC * N_BLK + 0.8, 2.7 * N_ROWS + 1.5),
        facecolor='#111111',
    )
    fig.suptitle(
        filename.replace('.png', '').replace('_', ' ').title(),
        fontsize=18, fontweight='bold', color='white', y=0.99,
    )
    outer_gs = gridspec.GridSpec(
        1, N_BLK, figure=fig,
        wspace=0.07, left=0.06, right=0.99, top=0.94, bottom=0.02,
    )
    row_lbls = []
    for ri in range(N_ROWS):
        off = N_OFF - ri
        if off > 0:
            row_lbls.append(f'z+{off}')
        elif off == 0:
            row_lbls.append('★nodule')
        else:
            row_lbls.append(f'z{off}')

    for bi, mod in enumerate(mods):
        s      = grid_samples[mod]
        vol3d  = s['volume_3d']
        attn3d = _smooth_3d(s['heatmap_3d'])
        msk3d  = s['masks_3d']
        label  = s['label']
        prob   = s['prob']
        img_id = s['img_id']
        K      = vol3d.shape[0]
        ctr    = int(s.get('coord_z_local', K // 2))

        inner  = outer_gs[bi].subgridspec(N_ROWS, N_IC, wspace=0.018, hspace=0.018)
        ok     = (label == 0) == (prob <= 0.5)
        gt_str = 'real' if label == 0 else mod

        for ri in range(N_ROWS):
            off    = N_OFF - ri
            ki     = int(np.clip(ctr + off, 0, K - 1))
            is_ctr = (ri == N_OFF)

            sl = _norm01(vol3d[ki])
            hm = attn3d[ki]
            if hm.max() > 0:
                hm = _norm01(hm)
            mk = msk3d[ki]

            bb = None
            if is_ctr and label > 0:
                ry = np.any(mk > 0.5, axis=1)
                cx = np.any(mk > 0.5, axis=0)
                if ry.any() and cx.any():
                    rmin, rmax = int(np.where(ry)[0][0]),  int(np.where(ry)[0][-1])
                    cmin, cmax = int(np.where(cx)[0][0]),  int(np.where(cx)[0][-1])
                    bb = (rmin, rmax, cmin, cmax)

            for ci in range(N_IC):
                ax = fig.add_subplot(inner[ri, ci])
                ax.axis('off')

                if ci == 0:
                    ax.imshow(sl, cmap='gray', aspect='auto')
                    if bb:
                        ax.add_patch(mpatches.Rectangle(
                            (bb[2], bb[0]), bb[3] - bb[2], bb[1] - bb[0],
                            lw=1.5, edgecolor='lime', facecolor='none',
                        ))
                    if bi == 0:
                        ax.set_ylabel(row_lbls[ri], fontsize=12,
                                      color='#ff8c00' if is_ctr else '#aaaaaa',
                                      fontweight='bold' if is_ctr else 'normal',
                                      rotation=0, labelpad=36, va='center')
                    if ri == 0:
                        ood_m = ' [OOD]' if (in_domain_fakes is not None
                                             and mod != 'real'
                                             and mod not in in_domain_fakes) else ''
                        ax.set_title(
                            f"{mod.upper()}{ood_m} | {img_id[:8]}\n"
                            f"GT:{gt_str}  p={prob:.2f} {'✓' if ok else '✗'}",
                            fontsize=11, fontweight='bold', color='white', pad=2,
                        )
                elif ci == 1:
                    ax.imshow(hm, cmap='turbo', vmin=0, vmax=1, aspect='auto')
                else:
                    ax.imshow(sl, cmap='gray', aspect='auto')
                    ax.imshow(hm, cmap='turbo', alpha=0.42, vmin=0, vmax=1, aspect='auto')
                    if bb:
                        ax.add_patch(mpatches.Rectangle(
                            (bb[2], bb[0]), bb[3] - bb[2], bb[1] - bb[0],
                            lw=1.5, edgecolor='lime', facecolor='none',
                        ))

                if is_ctr:
                    for sp in ax.spines.values():
                        sp.set_edgecolor('#ff8c00')
                        sp.set_linewidth(2.0)
                        sp.set_visible(True)

    if _ood_fake and n_in < N_BLK:
        p_in0 = outer_gs[0].get_position(fig)
        p_in1 = outer_gs[n_in - 1].get_position(fig)
        p_od0 = outer_gs[n_in].get_position(fig)
        p_od1 = outer_gs[-1].get_position(fig)
        sep_x = (p_in1.x1 + p_od0.x0) / 2
        fig.text((p_in0.x0 + p_in1.x1) / 2, 0.955,
                 '── In-domain ──', ha='center', va='bottom',
                 fontsize=14, fontweight='bold', color='white')
        fig.text((p_od0.x0 + p_od1.x1) / 2, 0.955,
                 '── OOD ──', ha='center', va='bottom',
                 fontsize=14, fontweight='bold', color='#e65100')
        fig.add_artist(plt.Line2D(
            [sep_x, sep_x], [0.02, 0.96],
            transform=fig.transFigure,
            color='#e65100', linewidth=2.0, linestyle='--',
        ))
    plt.savefig(out_dir / filename, dpi=120, bbox_inches='tight',
                facecolor='#111111')
    plt.close(fig)


def _save_combined_gif(
    grid_samples_ty: dict,          # {mod: sample_dict} for one ty (injection/removal)
    in_domain_fakes: set | None,
    save_path:       Path,
    duration_ms:     int = 300,
) -> None:
    """
    Single animated GIF showing real | in-domain | OOD side-by-side.
    Each frame = one z slice (CT overlay), 4 columns separated by colored bars.
    """
    try:
        from PIL import Image, ImageDraw
        import io
    except ImportError:
        return
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    _all_f = sorted(m for m in grid_samples_ty if m != 'real')
    if in_domain_fakes is not None:
        _in_f  = [m for m in sorted(_all_f) if m in in_domain_fakes and m in grid_samples_ty]
        _ood_f = [m for m in sorted(_all_f) if m not in in_domain_fakes and m in grid_samples_ty]
    else:
        _in_f  = [m for m in sorted(_all_f) if m in grid_samples_ty]
        _ood_f = []
    _real = ['real'] if 'real' in grid_samples_ty else []
    mods  = _real + _in_f + _ood_f
    if not mods:
        return

    n_in  = len(_real) + len(_in_f)
    K_min = min(grid_samples_ty[m]['volume_3d'].shape[0] for m in mods)

    frames = []
    for k in range(K_min):
        n_cols = len(mods)
        fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 3.8),
                                 facecolor='#111111')
        if n_cols == 1:
            axes = [axes]
        fig.subplots_adjust(left=0.01, right=0.99, top=0.85, bottom=0.02, wspace=0.06)

        for ci, mod in enumerate(mods):
            s      = grid_samples_ty[mod]
            is_ood = ci >= n_in
            ctr    = int(s.get('coord_z_local', s['volume_3d'].shape[0] // 2))
            ki     = int(np.clip(k, 0, s['volume_3d'].shape[0] - 1))

            sl  = _norm01(s['volume_3d'][ki])
            hm  = _smooth_3d(s['heatmap_3d'])[ki]
            hm  = _norm01(hm) if hm.max() > 0 else hm
            mk  = s['masks_3d'][ki]

            axes[ci].imshow(sl, cmap='gray', aspect='auto')
            axes[ci].imshow(np.ma.masked_where(hm < 0.05, hm),
                            cmap='turbo', alpha=0.55, vmin=0.05, vmax=1, aspect='auto')
            if s['label'] > 0 and mk.sum() > 0:
                axes[ci].contour(mk, levels=[0.5], colors='lime', linewidths=0.8)
            axes[ci].axis('off')

            label_color = '#e65100' if is_ood else 'white'
            tag = '[OOD]' if is_ood else ''
            axes[ci].set_title(
                f"{mod.upper()} {tag}\np={s['prob']:.2f}",
                fontsize=10, color=label_color, fontweight='bold',
            )
            # orange border on centre slice
            if ki == ctr:
                for sp in axes[ci].spines.values():
                    sp.set_edgecolor('#ff8c00'); sp.set_linewidth(2.5); sp.set_visible(True)

        # separator line between in-domain and OOD
        if _ood_f and n_in < len(mods):
            sep_frac = n_in / len(mods)
            fig.add_artist(plt.Line2D(
                [sep_frac, sep_frac], [0.0, 1.0],
                transform=fig.transFigure,
                color='#e65100', linewidth=2.0, linestyle='--',
            ))

        fig.suptitle(f'Frame z={k}', fontsize=11, color='white', y=0.97)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=80, bbox_inches='tight', facecolor='#111111')
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()

    if frames:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            save_path, save_all=True, append_images=frames[1:],
            duration=duration_ms, loop=0,
        )


def save_epoch_vis(
    model:      VolumeClassifier,
    dataloader: DataLoader,
    device:     torch.device,
    vis_dir:    Path,
    sargs:      dict,
) -> None:
    """
    Save one animated GIF per modality detected in the dataloader.
    Each GIF scrolls through K slices; each frame shows:
      Fake mods: 2 rows x 3 cols (row 0 = removal, row 1 = injection)
      Real mod:  1 row x 3 cols
    Columns: CT + GT contour | Attention | Overlay
    Files: vis_dir/{mod}.gif
    """
    vis_dir.mkdir(parents=True, exist_ok=True)
    patch_size = sargs['patch_size']
    stride     = sargs.get('stride') or (patch_size // 2)
    model.eval()

    cands: dict = {}

    for batch in dataloader:
        if cands and all(
            cands[m].get('removal') is not None and cands[m].get('injection') is not None
            for m in cands if m != 'real'
        ) and all(
            (cands[m].get('injection') or cands[m].get('removal')) is not None
            for m in cands if m == 'real'
        ):
            break
        patches_seq = batch['patches'].to(device)
        z_indices_t = batch['z_indices'].to(device)
        B = patches_seq.shape[0]
        for b in range(B):
            mod    = batch['mod'][b]
            img_id = batch['img_id'][b]
            ty     = _get_ty(img_id)
            if mod not in cands:
                cands[mod] = {'removal': None, 'injection': None}
            slot = 'injection' if mod == 'real' else ty
            if cands[mod][slot] is not None:
                continue
            valid_t = (z_indices_t[b] >= 0)
            with torch.no_grad():
                logit, attn_tup = model(patches_seq[b], z_indices_t[b], valid_t, return_attn=True)
            beta_t, alpha_list = attn_tup
            beta_np    = beta_t.cpu().float().numpy()
            valid_np   = valid_t.cpu().numpy()
            alpha_cpu  = [a.cpu().float().numpy() for a in alpha_list]
            sh         = (int(batch['slice_hw'][b, 0]), int(batch['slice_hw'][b, 1]))
            gh_arr     = batch['grid_hw'][b].numpy()
            masks_cpu  = batch['masks'][b, :, 0].numpy()
            volume_3d  = _build_volume_3d(batch['patches'][b], gh_arr, sh, patch_size, stride, valid_np)
            heatmap_3d = _build_heatmap_3d(alpha_cpu, beta_np, gh_arr, sh, patch_size, stride, valid_np)
            cands[mod][slot] = {
                'label':      int(batch['label'][b]),
                'prob':       float(sigmoid_np(logit.item())),
                'img_id':     img_id,
                'valid_mask': valid_np,
                'volume_3d':  volume_3d,
                'heatmap_3d': heatmap_3d,
                'masks_3d':   masks_cpu,
            }

    _render_vol_gifs(cands, vis_dir)
    model.train()


def _render_vol_gifs(
    cands:       dict,
    vis_dir:     Path,
    duration_ms: int = 250,
) -> None:
    """
    Render one animated GIF per modality in vis_dir/{mod}.gif.
    Fake mods: 2 rows x 3 cols (removal | injection), animated together.
    Real mod:  1 row x 3 cols.
    Columns: CT + GT contour | Attention | Overlay.
    """
    try:
        from PIL import Image
        import io
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    vis_dir.mkdir(parents=True, exist_ok=True)

    for mod, slots in cands.items():
        is_real    = (mod == 'real')
        rows_data  = []
        row_labels = []

        if is_real:
            s = slots.get('injection') or slots.get('removal')
            if s:
                rows_data.append(s)
                row_labels.append('')
        else:
            for ty in ['removal', 'injection']:
                s = slots.get(ty)
                if s:
                    rows_data.append(s)
                    row_labels.append(ty.capitalize())

        if not rows_data:
            continue

        valid_indices = []
        for s in rows_data:
            vi = np.where(s['valid_mask'])[0]
            valid_indices.append(vi)
        K_frames = min(len(vi) for vi in valid_indices)
        if K_frames == 0:
            continue

        n_rows   = len(rows_data)
        attn_3ds = [_smooth_3d(s['heatmap_3d']) for s in rows_data]
        frames   = []

        for k in range(K_frames):
            fig, axes = plt.subplots(n_rows, 3, figsize=(10.5, 3.5 * n_rows), squeeze=False)
            fig.suptitle(mod.upper(), fontsize=12, fontweight='bold')
            if k == 0:
                for ci, ct in enumerate(['CT + GT', 'Attention', 'Overlay']):
                    axes[0, ci].set_title(ct, fontsize=9)

            for ri, (s, rl, attn3d) in enumerate(zip(rows_data, row_labels, attn_3ds)):
                ki = int(valid_indices[ri][k])
                sl = _norm01(s['volume_3d'][ki])
                hm = attn3d[ki]
                hm = _norm01(hm) if hm.max() > 0 else hm
                mk = s['masks_3d'][ki]

                axes[ri, 0].imshow(sl, cmap='gray', aspect='auto')
                if s['label'] > 0 and mk.sum() > 0:
                    axes[ri, 0].contour(mk, levels=[0.5], colors='lime', linewidths=0.8)
                axes[ri, 0].set_ylabel(rl, fontsize=9, rotation=0, labelpad=52, va='center')
                axes[ri, 0].axis('off')

                axes[ri, 1].imshow(hm, cmap='turbo', vmin=0, vmax=1, aspect='auto')
                axes[ri, 1].axis('off')

                axes[ri, 2].imshow(sl, cmap='gray', aspect='auto')
                axes[ri, 2].imshow(hm, cmap='turbo', alpha=0.45, vmin=0, vmax=1, aspect='auto')
                if s['label'] > 0 and mk.sum() > 0:
                    axes[ri, 2].contour(mk, levels=[0.5], colors='lime', linewidths=0.8)
                axes[ri, 2].axis('off')

            fig.text(0.5, 0.02, f'frame {k+1}/{K_frames}', ha='center', fontsize=7)
            plt.tight_layout(rect=[0, 0.03, 1, 0.96])
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            frames.append(Image.open(buf).copy())
            buf.close()

        if frames:
            frames[0].save(
                vis_dir / f'{mod}.gif', save_all=True, append_images=frames[1:],
                duration=duration_ms, loop=0,
            )

# =============================================================================
#  Unified test evaluation (matches eval_volume-cls.py, full_volume=False)
# =============================================================================

@torch.no_grad()
def run_test_evaluation_volume(
    model,
    dl_test:         DataLoader,
    device:          torch.device,
    eval_dir:        Path,
    epoch:           int,
    run_dir:         Path,
    sargs:           dict,
    in_domain_fakes: set | None = None,
) -> dict:
    """
    Full test evaluation. Saves to eval_dir/:
        metrics.json      — nested {cls, xai}
        per_sample.csv    — one row per volume block
        {mod}.gif         — one animated GIF per modality
    Returns the full results dict.
    """
    eval_dir.mkdir(parents=True, exist_ok=True)

    patch_size = sargs['patch_size']
    stride     = sargs.get('stride') or (patch_size // 2)
    K          = model.K

    model.eval()

    all_logits:  list = []
    all_labels:  list = []
    all_mods:    list = []
    all_img_ids: list = []
    per_sample_records: list = []
    cands: dict = {}

    for batch in tqdm(dl_test, desc='Evaluating', unit='batch'):
        patches_seq = batch['patches'].to(device)
        z_indices_t = batch['z_indices'].to(device)

        B  = patches_seq.shape[0]
        sh = (int(batch['slice_hw'][0, 0]), int(batch['slice_hw'][0, 1]))

        for b in range(B):
            valid_t  = (z_indices_t[b] >= 0)
            valid_np = valid_t.cpu().numpy().astype(bool)

            logit, attn_tup = model(
                patches_seq[b], z_indices_t[b], valid_t, return_attn=True
            )
            beta_t, alpha_list = attn_tup
            beta_np    = beta_t.cpu().float().numpy()
            alpha_cpu  = [a.cpu().float().numpy() for a in alpha_list]
            z_np       = z_indices_t[b].cpu().numpy()
            gh_arr     = batch['grid_hw'][b].numpy()
            masks_cpu  = batch['masks'][b, :, 0].numpy()

            prob     = float(sigmoid_np(logit.item()))
            label_mc = int(batch['label'][b])
            label    = int(label_mc > 0)
            mod      = batch['mod'][b]
            img_id   = batch['img_id'][b]

            all_logits.append(torch.tensor(logit.item()))
            all_labels.append(torch.tensor(label))
            all_mods.append(mod)
            all_img_ids.append(img_id)

            coord_z_b = int(batch['coord_z'][b]) if 'coord_z' in batch else None
            if coord_z_b is not None and valid_np.any():
                diffs = np.abs(z_np.astype(int) - coord_z_b)
                diffs[~valid_np] = 99999
                centre_k = int(diffs.argmin())
            else:
                centre_k = K // 2

            heatmap_3d = _build_heatmap_3d(
                alpha_cpu, beta_np, gh_arr, sh, patch_size, stride, valid_np
            )
            hm_centre   = _smooth_attn(heatmap_3d[centre_k])
            mask_centre = masks_cpu[centre_k]

            xai_keys = ['pixel_auc', 'iou_03', 'iou_05', 'iou_07', 'pointing_game']
            rec = {
                'img_id': img_id, 'mod': mod, 'label': label,
                'prob': prob, 'pred': int(prob >= 0.5),
                'correct': int((label == 0) == (prob <= 0.5)),
                'coord_z': coord_z_b,
            }
            if label > 0 and mask_centre.sum() > 0:
                xai = compute_xai_metrics(hm_centre, mask_centre)
                rec.update({k: xai.get(k, float('nan')) for k in xai_keys})
            else:
                rec.update({k: float('nan') for k in xai_keys})
            per_sample_records.append(rec)

            ty   = _get_ty(img_id)
            if mod not in cands:
                cands[mod] = {'removal': None, 'injection': None}
            slot = 'injection' if mod == 'real' else ty
            if cands[mod][slot] is None:
                volume_3d = _build_volume_3d(
                    batch['patches'][b], gh_arr, sh, patch_size, stride, valid_np
                )
                cands[mod][slot] = {
                    'label':      label,
                    'prob':       prob,
                    'img_id':     img_id,
                    'valid_mask': valid_np,
                    'volume_3d':  volume_3d,
                    'heatmap_3d': heatmap_3d,
                    'masks_3d':   masks_cpu,
                }

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    all_logits_np = torch.stack(all_logits).numpy()
    all_labels_np = torch.stack(all_labels).numpy()   # binary (0/1)
    probs         = sigmoid_np(all_logits_np)
    preds         = (probs >= 0.5).astype(int)
    mods_arr      = np.array(all_mods)

    cls_metrics: dict = {
        'auc':      float(roc_auc_score(all_labels_np, probs)),
        'accuracy': float(accuracy_score(all_labels_np, preds)),
        'ap':       float(average_precision_score(all_labels_np, probs)),
        'f1':       float(f1_score(all_labels_np, preds, zero_division=0)),
        'pd_at_1':  _pd_at_1pct(all_labels_np, probs),
    }
    for mod in sorted(m for m in set(all_mods) if m != 'real'):
        for k, v in _balanced_mod_metrics(all_labels_np, probs, preds, mods_arr, mod).items():
            cls_metrics[f'{mod}_{k}'] = v

    xai_records = [r for r in per_sample_records
                   if not np.isnan(r.get('pixel_auc', float('nan')))]
    xai_keys = ['pixel_auc', 'iou_03', 'iou_05', 'iou_07',
                'pointing_game']
    xai_metrics: dict = {}
    if xai_records:
        for k in xai_keys:
            vals = [r[k] for r in xai_records if not np.isnan(r[k])]
            xai_metrics[k] = float(np.mean(vals)) if vals else float('nan')
        by_mod: dict = {}
        for r in xai_records:
            by_mod.setdefault(r['mod'], []).append(r)
        for mod, xs in by_mod.items():
            for k in xai_keys:
                vals = [x[k] for x in xs if not np.isnan(x[k])]
                xai_metrics[f'{k}_{mod}'] = float(np.mean(vals)) if vals else float('nan')

    results = {
        'split':   'test',
        'run_dir': str(run_dir),
        'epoch':   epoch,
        'cls':     cls_metrics,
        'xai':     xai_metrics,
    }

    # ── Save metrics.json ─────────────────────────────────────────────────────
    with open(eval_dir / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Metrics JSON  → {eval_dir / 'metrics.json'}")

    # ── Save per_sample.csv ───────────────────────────────────────────────────
    if per_sample_records:
        import pandas as pd
        pd.DataFrame(per_sample_records).to_csv(eval_dir / 'per_sample.csv', index=False)
        print(f"  Per-sample CSV → {eval_dir / 'per_sample.csv'}")

    # ── Visualization GIFs (one per modality) ────────────────────────────────
    _render_vol_gifs(cands, eval_dir)
    print(f"  Vis GIFs          -> {eval_dir}/<mod>.gif")

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
        jf = run_dir / 'evaluation' / 'metrics.json'
        if not jf.exists():
            jf = run_dir / 'test_metrics.json'  # legacy fallback
            if not jf.exists():
                return -1.0
        m   = json.loads(jf.read_text())
        cls = m.get('cls', m.get('classification', m))
        try:
            return float(cls.get('accuracy', cls.get('global_acc', -1.0)))
        except (TypeError, ValueError):
            return -1.0

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
    print(f'  Champion   test acc = {champion_acc:.4f}  ({canonical_dir.name})')
    print(f'  Challenger test acc = {challenger_acc:.4f}  ({temp_dir.name})')

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

    # ── Debug mode ────────────────────────────────────────────────────────────
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
        gid    = select_best_gpu()
        device = torch.device(f'cuda:{gid}') if gid is not None else torch.device('cpu')

    # ── Load Stage 1 checkpoint + build model ────────────────────────────────
    print(f"\nLoading Stage 1 checkpoint from: {args.slice_ckpt_dir}")
    model, sargs = build_volume_classifier(
        slice_ckpt_dir = args.slice_ckpt_dir,
        K              = args.K,
        attn_dim       = args.attn_dim,
        dropout        = args.dropout,
        device         = device,
    )

    # ── Auto-infer --mods from slice checkpoint if not explicitly passed ──────
    _slice_args = json.loads((Path(args.slice_ckpt_dir) / 'args.json').read_text())
    if args.mods is None:
        _slice_fake = [m for m in (_slice_args.get('mods') or []) if m != 'real']
        if not _slice_fake:
            _slice_fake = ALL_FAKES
        args.mods = ['real'] + _slice_fake
        print(f"  [auto] --mods inferred from slice checkpoint: {args.mods}")

    backbone_tag = sargs.get('backbone', 'unknown')
    patch_size   = sargs['patch_size']
    stride       = sargs.get('stride') or (patch_size // 2)
    print(f"\n{'='*60}")
    print(f"  Stage 2 Training — Volume-level classifier")
    print(f"{'='*60}")
    print(f"  Device:        {device}")
    print(f"  Stage 1:       {args.slice_ckpt_dir}")
    print(f"  Backbone:      {backbone_tag}  patch={patch_size}  stride={stride}")
    print(f"  Window:        K={args.K}  (random coord_z offset)")
    print(f"  attn_dim={args.attn_dim}   dropout={args.dropout}")
    print(f"  LR={args.lr}  WD={args.weight_decay}  BS={args.batch_size}  epochs={args.epochs}")
    print(f"  AMP={args.amp}  balance={args.balance_classes}  seed={args.seed}")

    # Trainable parameter count
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Params:        {n_train:,} trainable / {n_total:,} total\n")

    # ── Run name ─────────────────────────────────────────────────────────────
    all_fake      = set(ALL_FAKES)
    fake_mods_set = {m for m in args.mods if m != 'real'}
    mods_tag      = 'all' if fake_mods_set == all_fake else '+'.join(sorted(fake_mods_set))
    attn_tag      = '_attn' if _slice_args.get('aux_attn_loss') else ''
    base_dir      = Path(WORK_DIR) / 'runs' / f"volume-cls_{backbone_tag}_p{patch_size}_s{stride}_K{args.K}{attn_tag}"

    if args.run_name is None:
        run_tag     = f"trained_on_{mods_tag}_bce_bs{args.batch_size}_lr{args.lr}{attn_tag}"
        wandb_group = f"trained_on_{mods_tag}"
        wandb_name  = f"volume-cls_{backbone_tag}_p{patch_size}_s{stride}_K{args.K}_bce_bs{args.batch_size}_lr{args.lr}{attn_tag}"
    else:
        run_tag     = args.run_name
        wandb_group = f"trained_on_{mods_tag}"
        wandb_name  = args.run_name
    canonical_dir = base_dir / f'trained_on_{mods_tag}'
    out_dir       = base_dir / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output dir:    {out_dir}\n")

    # Save args (merge with sargs under 'slice_args' key)
    full_args = vars(args)
    full_args['slice_args'] = sargs
    with open(out_dir / 'args.json', 'w') as f:
        json.dump(full_args, f, indent=2)

    # ── WandB ────────────────────────────────────────────────────────────────
    if args.wandb_mode != 'disabled':
        import wandb
        wandb.init(project='MedForensics', group=wandb_group, name=wandb_name, config=vars(args), mode=args.wandb_mode)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("Loading data...")
    dl_train, dl_in_valid, dl_valid, dl_test = build_dataloaders(args, sargs)
    print()

    # ── Optimiser & criterion ─────────────────────────────────────────────────
    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode='min', factor=0.5, patience=10, min_lr=1e-6,
        ) if args.scheduler else None
    )
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_loss    = float('inf')
    patience_counter = 0
    history          = []

    print("Starting training...\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Warmup
        if epoch <= args.warmup_epochs:
            wlr = args.lr * (epoch / args.warmup_epochs)
            for pg in optimiser.param_groups:
                pg['lr'] = wlr

        train_losses  = train_one_epoch(model, dl_train, criterion, optimiser, scaler, device, args.amp)
        val_metrics   = evaluate(model, dl_valid,    criterion, device)
        inval_metrics = evaluate(model, dl_in_valid, criterion, device)

        if epoch > args.warmup_epochs and scheduler is not None:
            scheduler.step(val_metrics['loss'])

        current_lr = optimiser.param_groups[0]['lr']
        elapsed    = time.time() - t0

        log_str = (
            f"\nEpoch {epoch:03d}/{args.epochs} | "
            f"Train {train_losses['total']:.4f}"
            f" | Valid {val_metrics['loss']:.4f}"
            f" | In-Valid {inval_metrics['loss']:.4f}"
            f" | AUC {val_metrics['auc']:.4f}  AP {val_metrics['ap']:.4f}"
            f"  Acc {val_metrics['accuracy']:.4f}  F1 {val_metrics['f1']:.4f}"
            f" | {elapsed:.1f}s"
        )
        print(log_str)
        all_in_val = sorted(m for m in ALL_FAKES if f'{m}_auc' in val_metrics)
        if all_in_val:
            print("  ─── Per-arch (balanced) " + "─" * 48)
            for m in all_in_val:
                tag = '' if m in fake_mods_set else ' [OOD]'
                a, ac, ap, f1 = (val_metrics.get(f'{m}_{k}', float('nan'))
                                 for k in ['auc', 'acc', 'ap', 'f1'])
                print(f"  {m:<12s}{tag}  AUC {a:.4f}  Acc {ac:.4f}  AP {ap:.4f}  F1 {f1:.4f}")

        # Visualisation
        if args.vis_every > 0 and epoch % args.vis_every == 0:
            ep_vis_dir = out_dir / 'vis' / f'epoch_{epoch:03d}'
            try:
                save_epoch_vis(model, dl_valid, device, ep_vis_dir, sargs)
                print(f"  → Vis saved: {ep_vis_dir}/")
                if args.wandb_mode != 'disabled':
                    import wandb
                    vis_log = {
                        f'vis/{gif.stem}': wandb.Video(str(gif), fps=4, format='gif')
                        for gif in sorted(ep_vis_dir.glob('*.gif'))
                    }
                    if vis_log:
                        wandb.log(vis_log)
            except Exception as e:
                print(f"  [WARN] save_epoch_vis failed at epoch {epoch}: {e}")

        # WandB
        if args.wandb_mode != 'disabled':
            import wandb
            log_d = {
                'train/loss':    train_losses['total'],
                'val/loss':      val_metrics['loss'],
                'in_val/loss':   inval_metrics['loss'],
                'val/auc':       val_metrics['auc'],
                'val/acc':       val_metrics['accuracy'],
                'val/ap':        val_metrics['ap'],
                'val/f1':        val_metrics['f1'],
                'in_val/auc':    inval_metrics['auc'],
                'in_val/acc':    inval_metrics['accuracy'],
            }
            for m in sorted(m for m in ALL_FAKES if f'{m}_auc' in val_metrics):
                for k in ['auc', 'acc', 'ap', 'f1']:
                    log_d[f'val/{m}/{k}'] = val_metrics.get(f'{m}_{k}', float('nan'))
            wandb.log(log_d)

        history.append({
            'epoch':         epoch,
            'lr':            current_lr,
            'train_loss':    train_losses,
            'val_metrics':   val_metrics,
            'inval_metrics': inval_metrics,
            'time_s':        elapsed,
        })

        # Checkpoint
        if val_metrics['loss'] < best_val_loss:
            best_val_loss    = val_metrics['loss']
            patience_counter = 0
            torch.save({
                'epoch':                   epoch,
                'model_state_dict':        model.state_dict(),
                'optimiser_state_dict':    optimiser.state_dict(),
                'best_val_loss':           best_val_loss,
                'best_val_auc':            val_metrics['auc'],
                'args':                    full_args,
            }, out_dir / 'best_model.pt')
            print(f"  ★ New best val loss: {best_val_loss:.4f}  "
                  f"(AUC {val_metrics['auc']:.4f}) — saved")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  Early stopping at epoch {epoch} (patience={args.patience})")
                break

    with open(out_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2, default=str)

    # ── Test evaluation → evaluation/ (matches eval_volume-cls.py, full_volume=False) ─
    print(f"\n{'='*60}")
    print("  Test Evaluation (best model)")
    print(f"{'='*60}\n")

    ckpt = torch.load(out_dir / 'best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    eval_dir = out_dir / 'evaluation'
    results = run_test_evaluation_volume(
        model           = model,
        dl_test         = dl_test,
        device          = device,
        eval_dir        = eval_dir,
        epoch           = int(ckpt['epoch']),
        run_dir         = out_dir,
        sargs           = sargs,
        in_domain_fakes = fake_mods_set,
    )
    cls_m = results['cls']

    for k, v in sorted(cls_m.items()):
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.4f}")

    if args.wandb_mode != 'disabled':
        import wandb
        wandb.log({f'test/{k}': v for k, v in cls_m.items()})
        wandb.log({f'test/xai/{k}': v for k, v in results.get('xai', {}).items()})
        vis_log = {
            f'vis/gif/{gif.stem}': wandb.Video(str(gif), fps=4, format='gif')
            for gif in sorted(eval_dir.glob('*.gif'))
        }
        if vis_log:
            wandb.log(vis_log)
            print(f"  → Logged {len(vis_log)} visuals to WandB")
        wandb.finish()

    promote_if_better(out_dir, canonical_dir)

    print(f"\nAll outputs saved to: {out_dir}")
    print("Done.")


if __name__ == '__main__':
    args = get_args()
    main(args)
