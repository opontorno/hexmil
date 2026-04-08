#!/usr/bin/env python3
"""
eval_phase_a.py
===============
Phase A evaluation script.

Loads a trained checkpoint and computes:
  1. **Classification metrics**: AUC-ROC, Accuracy, Precision, Recall, F1 — overall and per-modality.
  2. **XAI / Attention metrics** (fake samples only):
     - Pixel-AUC: AUC of attention map vs binary mask (pixel-level).
     - IoU@t: Intersection-over-Union at various thresholds on the attention map.
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import DATA_DIR
     - Pointing Game accuracy: does the max-attention pixel fall inside the mask?
     - Energy inside mask: what fraction of total attention falls inside the GT mask?
  3. Saves attention map visualisations for qualitative inspection.

Usage:
    python experiments/eval_phase_a.py --checkpoint experiments/runs/<run>/best_model.pt
    python experiments/eval_phase_a.py --checkpoint experiments/runs/<run>/best_model.pt --split test --save_vis
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
from scipy.special import expit as sigmoid_np  # numerically stable sigmoid

from hexmil.data.patch_dataset import NodulePatchDataset, load_split_table
from hexmil.models.cnn_patch_classifier import build_cnn_classifier
from hexmil.models.vit_patch_classifier import build_vit_classifier

# =========================================================================
#  CLI
# =========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate patch classifier on a saved checkpoint.")
    parser.add_argument('--run_dir', type=str, required=True,
                        help="Path to the training run directory containing best_model.pt")
    parser.add_argument('--split', type=str, default='test', choices=['train', 'valid', 'test'],
                        help="Which data split to evaluate on")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for evaluation")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument('--device', type=str, default='cuda', help="Device to run evaluation on")
    parser.add_argument('--save_vis', action='store_true', help="Whether to save individual attention visualisations")
    parser.add_argument('--max_vis', type=int, default=50, help="Maximum number of individual visualisations to save")
    return parser.parse_args()

# =========================================================================
#  Build model from saved args
# =========================================================================

def load_model_from_checkpoint(ckpt_path: str, device: torch.device):
    """Load model + args from a training checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = argparse.Namespace(**ckpt['args'])

    if args.arch == 'cnn':
        model = build_cnn_classifier(
            backbone=args.backbone,
            pretrained=False,          # no need to download again
            freeze_ratio=0.0,          # nothing to freeze at eval
        )
    elif args.arch == 'vit':
        model = build_vit_classifier(
            img_size=args.patch_size,
            token_size=args.token_size,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
        )
    else:
        raise ValueError(f"Unknown arch: {args.arch}")

    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()
    return model, args

# =========================================================================
#  XAI metrics (per-sample, fake only)
# =========================================================================

def compute_xai_metrics(attn: np.ndarray, mask: np.ndarray) -> dict:
    """
    Compute XAI quality metrics for a single sample.

    Args:
        attn: (H, W) attention map, values in [0, 1]
        mask: (H, W) binary ground-truth mask {0, 1}

    Returns:
        dict with pixel_auc, iou_03, iou_05, iou_07, pointing_game, energy_in_mask
    """
    assert attn.shape == mask.shape, f"Shape mismatch: attn {attn.shape} vs mask {mask.shape}"

    mask_bin = (mask > 0.5).astype(np.float32)
    has_mask = mask_bin.sum() > 0

    result = {}

    # ── Pixel-AUC ────────────────────────────────────────────────────────
    if has_mask and len(np.unique(mask_bin)) == 2:
        result['pixel_auc'] = roc_auc_score(mask_bin.ravel(), attn.ravel())
    else:
        result['pixel_auc'] = float('nan')

    # ── IoU at different thresholds ──────────────────────────────────────
    for t in [0.3, 0.5, 0.7]:
        attn_bin = (attn >= t).astype(np.float32)
        inter = (attn_bin * mask_bin).sum()
        union = attn_bin.sum() + mask_bin.sum() - inter
        iou = inter / (union + 1e-7)
        result[f'iou_{int(t*10):02d}'] = float(iou)

    # ── Pointing Game ────────────────────────────────────────────────────
    if has_mask:
        max_idx = np.unravel_index(attn.argmax(), attn.shape)
        result['pointing_game'] = float(mask_bin[max_idx] > 0.5)
    else:
        result['pointing_game'] = float('nan')

    # ── Energy inside mask ───────────────────────────────────────────────
    total_energy = attn.sum()
    if has_mask and total_energy > 1e-7:
        result['energy_in_mask'] = float((attn * mask_bin).sum() / total_energy)
    else:
        result['energy_in_mask'] = float('nan')

    return result

# =========================================================================
#  Main evaluation
# =========================================================================

# Column order for the 3×4 evaluation grid
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
def run_evaluation(model, dataloader, device, save_vis=False, vis_dir=None,
                   max_vis=50):
    """
    Run full evaluation.

    Returns:
        cls_metrics       – overall + per-modality classification metrics
        xai_metrics       – XAI summary metrics (fake samples only)
        per_sample_records – list of dicts, one per sample (for CSV)
        grid_samples      – dict mod → sample data (for 3×4 grid)
    """
    model.eval()

    all_logits, all_labels, all_mods, all_img_ids = [], [], [], []
    all_xai: list[dict] = []
    per_sample_records: list[dict] = []
    grid_cands: dict[str, dict[str, list]] = {ty: {m: [] for m in _VIS_COL_MODS} for ty in _VIS_TYS}
    _N_CAND_GRID = 40

    vis_count = 0

    for batch in tqdm(dataloader, desc='Evaluating patches', unit='batch', dynamic_ncols=True):
        images = batch['image'].to(device)
        logits, attn = model(images, return_attn=True)

        logits_np = logits.squeeze(1).cpu().numpy()
        attn_np   = attn.squeeze(1).cpu().numpy()       # (B, H, W)
        labels_np = batch['label'].numpy()
        masks_np  = batch['mask'].squeeze(1).numpy()     # (B, H, W)

        all_logits.append(logits_np)
        all_labels.append(labels_np)
        all_mods.extend(batch['mod'])
        all_img_ids.extend(batch['img_id'])

        for i in range(len(labels_np)):
            mod     = batch['mod'][i]
            img_id  = batch['img_id'][i]
            prob_i  = float(sigmoid_np(logits_np[i]))
            pred_i  = int(prob_i >= 0.5)
            label_i = int(labels_np[i] > 0)   # binary

            # XAI metrics for fake samples
            xai_row: dict = {}
            if labels_np[i] > 0:
                xai = compute_xai_metrics(attn_np[i], masks_np[i])
                xai['img_id'] = img_id
                xai['mod']    = mod
                all_xai.append(xai)
                xai_row = {k: xai[k] for k in
                           ['pixel_auc', 'iou_03', 'iou_05', 'iou_07',
                            'pointing_game', 'energy_in_mask']}
            else:
                xai_row = {k: float('nan') for k in
                           ['pixel_auc', 'iou_03', 'iou_05', 'iou_07',
                            'pointing_game', 'energy_in_mask']}

            per_sample_records.append({
                'img_id':  img_id,
                'mod':     mod,
                'label':   label_i,
                'prob':    round(prob_i, 6),
                'pred':    pred_i,
                'correct': int(label_i == pred_i),
                **xai_row,
            })

            # Collect candidates for summary grids (shared-ID selection after loop)
            ty = _get_ty(img_id)
            if mod in _VIS_COL_MODS and len(grid_cands[ty][mod]) < _N_CAND_GRID:
                a = attn_np[i].copy()
                lo, hi = a.min(), a.max()
                if hi > lo:
                    a = (a - lo) / (hi - lo)
                grid_cands[ty][mod].append({
                    'image':  images[i, 0].cpu().numpy(),
                    'attn':   a,
                    'mask':   masks_np[i],
                    'label':  label_i,
                    'mod':    mod,
                    'img_id': img_id,
                    'prob':   prob_i,
                })

            # Individual visualisations
            if save_vis and vis_dir and vis_count < max_vis:
                _save_single_vis(
                    images[i].cpu().numpy()[0],
                    attn_np[i],
                    masks_np[i],
                    labels_np[i],
                    mod,
                    img_id,
                    logits_np[i],
                    vis_dir,
                    vis_count,
                )
                vis_count += 1
    # ── Build summary grids (shared img_id across modalities) ──────────────────
    grid_samples: dict[str, dict] = {'removal': {}, 'injection': {}}
    for ty in _VIS_TYS:
        id_sets = [{s['img_id'] for s in grid_cands[ty][m]} for m in _VIS_COL_MODS if grid_cands[ty][m]]
        shared  = id_sets[0].intersection(*id_sets[1:]) if len(id_sets) > 1 else (id_sets[0] if id_sets else set())
        anchor  = next(iter(shared)) if shared else None
        for mod in _VIS_COL_MODS:
            clist = grid_cands[ty][mod]
            if not clist:
                continue
            match = next((s for s in clist if s['img_id'] == anchor), None) if anchor else None
            grid_samples[ty][mod] = match if match else clist[0]
    # ── Aggregate classification metrics ─────────────────────────────────
    all_logits    = np.concatenate(all_logits)
    all_labels    = np.concatenate(all_labels)   # multi-class
    binary_labels = (all_labels > 0).astype(int)
    probs         = sigmoid_np(all_logits)
    preds         = (probs >= 0.5).astype(int)
    mods_arr      = np.array(all_mods)

    cls_metrics = {
        'auc':       float(roc_auc_score(binary_labels, probs)),
        'accuracy':  float(accuracy_score(binary_labels, preds)),
        'precision': float(precision_score(binary_labels, preds, zero_division=0)),
        'recall':    float(recall_score(binary_labels, preds, zero_division=0)),
        'f1':        float(f1_score(binary_labels, preds, zero_division=0)),
    }

    # Confusion matrix
    cm = confusion_matrix(binary_labels, preds)
    cls_metrics['confusion_matrix'] = cm.tolist()

    # Per-modality
    per_mod = {}
    real_idx_arr = mods_arr == 'real'
    for mod in sorted(set(all_mods)):
        idx = mods_arr == mod
        if idx.sum() == 0:
            continue
        lab_m  = binary_labels[idx]
        prob_m = probs[idx]
        pred_m = preds[idx]
        m = {
            'n_samples': int(idx.sum()),
            'accuracy':  float(accuracy_score(lab_m, pred_m)),
        }
        if mod != 'real':
            # AUC: this fake mod vs real only
            combined = real_idx_arr | idx
            if combined.sum() > 0 and idx.sum() > 0:
                m['auc']       = float(roc_auc_score(binary_labels[combined], probs[combined]))
                m['precision'] = float(precision_score(binary_labels[combined], preds[combined], zero_division=0))
                m['recall']    = float(recall_score(binary_labels[combined], preds[combined], zero_division=0))
                m['f1']        = float(f1_score(binary_labels[combined], preds[combined], zero_division=0))
        per_mod[mod] = m
    cls_metrics['per_modality'] = per_mod

    # ── Per manipulation type (removal / injection) ───────────────────────
    for ty, prefix in [('removal', 'rem_'), ('injection', 'inj_')]:
        ty_mask = np.array([str(id_).startswith(prefix) for id_ in all_img_ids])
        bl_ty   = binary_labels[ty_mask]
        pr_ty   = probs[ty_mask]
        pd_ty   = preds[ty_mask]
        if ty_mask.sum() > 0 and bl_ty.sum() > 0 and (bl_ty == 0).sum() > 0:
            cls_metrics[f'{ty}_auc']       = float(roc_auc_score(bl_ty, pr_ty))
            cls_metrics[f'{ty}_accuracy']  = float(accuracy_score(bl_ty, pd_ty))
            cls_metrics[f'{ty}_precision'] = float(precision_score(bl_ty, pd_ty, zero_division=0))
            cls_metrics[f'{ty}_recall']    = float(recall_score(bl_ty, pd_ty, zero_division=0))
            cls_metrics[f'{ty}_f1']        = float(f1_score(bl_ty, pd_ty, zero_division=0))
        else:
            for stat in ['auc', 'accuracy', 'precision', 'recall', 'f1']:
                cls_metrics[f'{ty}_{stat}'] = float('nan')

    # ── Aggregate XAI metrics ────────────────────────────────────────────
    xai_metrics = {}
    if all_xai:
        keys = ['pixel_auc', 'iou_03', 'iou_05', 'iou_07', 'pointing_game', 'energy_in_mask']
        for k in keys:
            vals = [x[k] for x in all_xai if not np.isnan(x[k])]
            if vals:
                xai_metrics[k] = {
                    'mean': float(np.mean(vals)),
                    'std':  float(np.std(vals)),
                    'min':  float(np.min(vals)),
                    'max':  float(np.max(vals)),
                    'n':    len(vals),
                }

        # Per-modality XAI
        xai_per_mod = {}
        for mod in sorted(set(x['mod'] for x in all_xai)):
            mod_xai = [x for x in all_xai if x['mod'] == mod]
            mod_metrics = {}
            for k in keys:
                vals = [x[k] for x in mod_xai if not np.isnan(x[k])]
                if vals:
                    mod_metrics[k] = {
                        'mean': float(np.mean(vals)),
                        'std':  float(np.std(vals)),
                        'n':    len(vals),
                    }
            xai_per_mod[mod] = mod_metrics
        xai_metrics['per_modality'] = xai_per_mod

        # ── Per manipulation type XAI ────────────────────────────────
        for ty, prefix in [('removal', 'rem_'), ('injection', 'inj_')]:
            ty_xai = [x for x in all_xai if str(x['img_id']).startswith(prefix)]
            if ty_xai:
                ty_m: dict = {}
                for k in keys:
                    vals = [x[k] for x in ty_xai if not np.isnan(x[k])]
                    if vals:
                        ty_m[k] = {
                            'mean': float(np.mean(vals)),
                            'std':  float(np.std(vals)),
                            'n':    len(vals),
                        }
                xai_metrics[ty] = ty_m

    return cls_metrics, xai_metrics, per_sample_records, grid_samples

def save_evaluation_grid(
    grid_samples: dict,
    eval_dir: Path,
    filename: str = 'grid.png',
) -> None:
    """
    3 rows × (4 blocks × 3 cols) summary grid with modality block separators.
    Rows:  [ CT Patch  |  Attention  |  Overlay + contour ]
    Col groups (×4 mods):  [ patch  |  attn  |  overlay ]
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    eval_dir.mkdir(parents=True, exist_ok=True)
    mods = [m for m in _VIS_COL_MODS if m in grid_samples]
    if not mods:
        return

    N_ROWS  = 3
    N_IC    = 3
    N_BLK   = len(mods)

    fig = plt.figure(
        figsize=(3.0 * N_IC * N_BLK + 0.8, 2.7 * N_ROWS + 1.5),
        facecolor='#111111',
    )
    fig.suptitle(filename.replace('.png', '').replace('_', ' ').title(),
                 fontsize=11, fontweight='bold', color='white', y=0.99)
    outer_gs = gridspec.GridSpec(
        1, N_BLK, figure=fig,
        wspace=0.07, left=0.06, right=0.99, top=0.94, bottom=0.02,
    )
    row_lbls = ['CT Patch', 'Attention', 'Overlay']

    for bi, mod in enumerate(mods):
        s      = grid_samples[mod]
        img    = s['image']
        attn   = _smooth_attn(s['attn'])
        mask   = s['mask']
        label, prob, img_id = s['label'], s['prob'], s['img_id']

        inner  = outer_gs[bi].subgridspec(N_ROWS, N_IC, wspace=0.018, hspace=0.018)
        ok     = (label == 0) == (prob <= 0.5)
        gt_str = 'real' if label == 0 else mod

        for ri in range(N_ROWS):
            for ci in range(N_IC):
                ax = fig.add_subplot(inner[ri, ci])
                ax.axis('off')
                if ci == 0:
                    ax.imshow(img, cmap='gray', aspect='auto')
                    if label > 0 and mask.sum() > 0:
                        ax.contour(mask, levels=[0.5], colors='lime', linewidths=0.9)
                    if bi == 0:
                        ax.set_ylabel(row_lbls[ri], fontsize=7.5, color='#cccccc',
                                      fontweight='bold', rotation=0, labelpad=50,
                                      va='center')
                    if ri == 0:
                        ax.set_title(
                            f"{mod.upper()} | {img_id[:8]}\n"
                            f"GT:{gt_str}  p={prob:.2f} {'✓' if ok else '✗'}",
                            fontsize=7, fontweight='bold', color='white', pad=2,
                        )
                elif ci == 1:
                    ax.imshow(attn, cmap='turbo', vmin=0, vmax=1, aspect='auto')
                else:
                    ax.imshow(img, cmap='gray', aspect='auto')
                    ax.imshow(attn, cmap='turbo', alpha=0.42, vmin=0, vmax=1,
                              aspect='auto')
                    if label > 0 and mask.sum() > 0:
                        ax.contour(mask, levels=[0.5], colors='lime', linewidths=0.9)

    plt.savefig(eval_dir / filename, dpi=120, bbox_inches='tight',
                facecolor='#111111')
    plt.close(fig)

def _save_single_vis(image, attn, mask, label, mod, img_id, logit, vis_dir, idx):
    """Save individual patch vis: input | attention | overlay."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pred_str = 'fake' if logit > 0 else 'real'
    gt_str   = 'fake' if label > 0 else 'real'
    prob     = float(sigmoid_np(logit))

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    fig.suptitle(f'{img_id} ({mod})  GT:{gt_str}  Pred:{pred_str}(p={prob:.3f})', fontsize=9)

    axes[0].imshow(image, cmap='gray');  axes[0].set_title('Image', fontsize=8);     axes[0].axis('off')
    axes[1].imshow(attn,  cmap='jet', vmin=0, vmax=1)
    axes[1].set_title('Attention', fontsize=8); axes[1].axis('off')
    axes[2].imshow(image, cmap='gray')
    axes[2].imshow(attn,  cmap='jet', alpha=0.45, vmin=0, vmax=1)
    if label > 0:
        axes[2].contour(mask, levels=[0.5], colors='lime', linewidths=1.5)
    axes[2].set_title('Overlay', fontsize=8); axes[2].axis('off')

    plt.tight_layout()
    safe = str(img_id).replace('/', '_')
    plt.savefig(os.path.join(vis_dir, f'{idx:04d}_{safe}_{mod}.png'), dpi=100,
                bbox_inches='tight')
    plt.close(fig)

def main(args):

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ── Resolve checkpoint from run dir ──────────────────────────────────
    run_dir   = Path(args.run_dir)
    ckpt_path = run_dir / 'best_model.pt'
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No best_model.pt found in: {run_dir}")

    # ── Load model ─────────────────────────────────────────────────────
    print(f"Loading checkpoint: {ckpt_path}")
    model, train_args = load_model_from_checkpoint(str(ckpt_path), device)
    print(f"  Architecture: {train_args.arch}")
    if train_args.arch == 'cnn':
        print(f"  Backbone: {train_args.backbone}")
    print(f"  Patch size: {train_args.patch_size}")
    print(f"  Trained for {train_args.epochs} epochs")
    print()

    # ── Data ─────────────────────────────────────────────────────────────
    mods = ['real', 'pix2pix', 'cycle', 'diffusion']
    tab = load_split_table(DATA_DIR, args.split, mods)
    ds = NodulePatchDataset(DATA_DIR, tab, patch_size=train_args.patch_size, augment=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)
    print(f"Evaluating on {args.split}: {len(ds)} samples")
    print()

    # ── Output directory ─────────────────────────────────────────────────
    eval_dir = run_dir / 'evaluation'
    eval_dir.mkdir(exist_ok=True)

    vis_dir = None
    if args.save_vis:
        vis_dir = str(eval_dir / 'vis')
        os.makedirs(vis_dir, exist_ok=True)

    # ── Run ──────────────────────────────────────────────────────────────
    cls_metrics, xai_metrics, per_sample_records, grid_samples = run_evaluation(
        model, dl, device,
        save_vis=args.save_vis,
        vis_dir=vis_dir,
        max_vis=args.max_vis,
    )

    # ── Print results ────────────────────────────────────────────────────
    print(f"{'='*60}")
    print(f"  Classification Metrics ({args.split})")
    print(f"{'='*60}")
    for k in ['auc', 'accuracy', 'precision', 'recall', 'f1']:
        print(f"  {k:20s}: {cls_metrics[k]:.4f}")
    print()

    if 'per_modality' in cls_metrics:
        print("  Per-modality:")
        for mod, m in cls_metrics['per_modality'].items():
            line = f"    {mod:20s}: n={m['n_samples']:4d}, acc={m['accuracy']:.4f}"
            if 'auc' in m:
                line += f", auc={m['auc']:.4f}, f1={m['f1']:.4f}"
            print(line)
        print()

    print("  Removal:   AUC {:.4f}  Acc {:.4f}  F1 {:.4f}".format(
        cls_metrics.get('removal_auc', float('nan')),
        cls_metrics.get('removal_accuracy', float('nan')),
        cls_metrics.get('removal_f1', float('nan')),
    ))
    print("  Injection: AUC {:.4f}  Acc {:.4f}  F1 {:.4f}".format(
        cls_metrics.get('injection_auc', float('nan')),
        cls_metrics.get('injection_accuracy', float('nan')),
        cls_metrics.get('injection_f1', float('nan')),
    ))
    print()

    if 'confusion_matrix' in cls_metrics:
        cm = np.array(cls_metrics['confusion_matrix'])
        print("  Confusion matrix (rows=GT, cols=Pred):")
        print(f"                Pred Real  Pred Fake")
        print(f"    GT Real       {cm[0,0]:5d}      {cm[0,1]:5d}")
        print(f"    GT Fake       {cm[1,0]:5d}      {cm[1,1]:5d}")
        print()

    if xai_metrics:
        print(f"{'='*60}")
        print(f"  XAI / Attention Quality (fake samples only)")
        print(f"{'='*60}")
        for k in ['pixel_auc', 'iou_03', 'iou_05', 'iou_07', 'pointing_game', 'energy_in_mask']:
            if k in xai_metrics:
                m = xai_metrics[k]
                print(f"  {k:20s}: {m['mean']:.4f} ± {m['std']:.4f}  (n={m['n']})")
        print()

        if 'per_modality' in xai_metrics:
            print("  Per-modality XAI:")
            for mod, mod_m in xai_metrics['per_modality'].items():
                print(f"    {mod}:")
                for k, v in mod_m.items():
                    print(f"      {k:20s}: {v['mean']:.4f} ± {v['std']:.4f}  (n={v['n']})")
            print()

    # ── 3×4 summary grids (removal + injection) ───────────────────────
    for ty in _VIS_TYS:
        fname = f'grid_{ty}.png'
        save_evaluation_grid(grid_samples[ty], eval_dir, filename=fname)
        print(f"  Summary grid ({ty}) → {eval_dir / fname}")

    # ── Per-sample CSV ───────────────────────────────────────────────────
    import csv
    csv_path = eval_dir / 'per_sample_metrics.csv'
    if per_sample_records:
        fieldnames = list(per_sample_records[0].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_sample_records)
        print(f"  Per-sample CSV    → {csv_path}")

    # ── Save JSON ────────────────────────────────────────────────────────
    results = {
        'split':          args.split,
        'checkpoint':     str(ckpt_path),
        'classification': cls_metrics,
        'xai':            xai_metrics,
    }
    out_path = eval_dir / 'metrics.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Metrics JSON      → {out_path}")

    if args.save_vis:
        print(f"  Individual vis    → {vis_dir}/")

    print("Done.")

if __name__ == '__main__':
    args = parse_args()
    main(args)
