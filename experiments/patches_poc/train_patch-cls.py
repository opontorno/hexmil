#!/usr/bin/env python3
"""
train_phase_a.py
================
Phase A training script: single-patch binary classification (real vs fake).

Supports:
  - CNN backbone (ResNet-50, EfficientNet-B4, …) via `--arch cnn`
  - ViT (custom from-scratch) via `--arch vit`
  - Two patch sizes (64, 128) via `--patch_size`
  - Pure BCE loss *or* BCE + auxiliary attention loss via `--aux_attn_loss`
  - Early stopping on validation AUC
  - WandB logging (optional, --wandb)
  - Checkpointing best model

Usage examples:
    python experiments/train_phase_a.py --arch cnn --backbone resnet50 --patch_size 128
    python experiments/train_phase_a.py --arch vit --patch_size 128 --embed_dim 384 --depth 6
    python experiments/train_phase_a.py --arch cnn --aux_attn_loss --aux_weight 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from scipy.special import expit as sigmoid_np  # numerically stable sigmoid for numpy arrays
import GPUtil

from hexmil.data.patch_dataset import NodulePatchDataset, load_split_table
from hexmil.models.cnn_patch_classifier import build_cnn_classifier
from hexmil.models.vit_patch_classifier import build_vit_classifier

# ── paths ────────────────────────────────────────────────────────────────

WORK_DIR = Path(__file__).resolve().parent.parent   # MedForensics root
sys.path.insert(0, str(WORK_DIR))
from config import DATA_DIR

def select_best_gpu():
    """Automatically select the GPU with the most free memory."""
    if not torch.cuda.is_available():
        print("No CUDA GPUs available, using CPU")
        return None
    gpus = GPUtil.getGPUs()
    if not gpus:
        print("No GPUs found by GPUtil, using cuda:0")
        return 0
    best_gpu = max(gpus, key=lambda gpu: gpu.memoryFree)
    print(f"🎯 Auto-selected GPU {best_gpu.id}: {best_gpu.name} "
          f"(Free: {best_gpu.memoryFree}MB / {best_gpu.memoryTotal}MB)")
    return best_gpu.id

def get_args():
    parser = argparse.ArgumentParser(description='Phase A: patch classification')

    # ── Architecture ─────────────────────────────────────────────────────
    parser.add_argument('--arch', type=str, default='cnn', choices=['cnn', 'vit'], help='Model architecture family')
    parser.add_argument('--patch_size', type=int, default=128, choices=[64, 128], help='Input patch spatial size')

    # CNN-specific
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet50', 'efficientnet_b4', 'vgg16_bn', 'densenet121', 'convnext_tiny'],
                        help='timm backbone name (CNN only)')
    parser.add_argument('--pretrained', action='store_true', default=True, help='Use ImageNet pretrained weights (CNN)')
    parser.add_argument('--freeze_ratio', type=float, default=0.5, help='Fraction of backbone layers to freeze (CNN)')

    # ViT-specific
    parser.add_argument('--token_size', type=int, default=16, choices=[8, 16],
                        help='ViT sub-patch (token) size')
    parser.add_argument('--embed_dim', type=int, default=384, choices=[256, 384, 512],
                        help='ViT embedding dimension')
    parser.add_argument('--depth', type=int, default=6, choices=[4, 6, 8, 12],
                        help='ViT transformer depth')
    parser.add_argument('--num_heads', type=int, default=6, choices=[4, 6, 8],
                        help='ViT number of attention heads')

    # ── Loss ─────────────────────────────────────────────────────────────
    parser.add_argument('--aux_attn_loss', action='store_true', help='Add auxiliary attention supervision loss')
    parser.add_argument('--aux_weight', type=float, default=0.5, help='Weight of auxiliary attention loss')
    parser.add_argument('--attn_loss_type', type=str, default='bce', choices=['bce', 'mse', 'dice'], help='Type of auxiliary attention loss')

    # ── Training ─────────────────────────────────────────────────────────
    parser.add_argument('--epochs', type=int, default=50, help='Maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='AdamW weight decay')
    parser.add_argument('--jitter_px', type=int, default=8, help='Max random crop jitter in pixels')
    parser.add_argument('--balance_classes', action='store_true', default=True, help='Use weighted sampler for class balance')

    # ── Scheduler ────────────────────────────────────────────────────────
    parser.add_argument('--scheduler', action='store_true', help='Learning rate scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=3, help='Number of linear warmup epochs')

    # ── Early stopping ───────────────────────────────────────────────────
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (epochs without improvement)')

    # ── System ───────────────────────────────────────────────────────────
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader num_workers')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision (AMP)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--wandb_mode', default='online', choices=['online', 'offline', 'disabled'],
                        help='WandB logging mode')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode: sets wandb_mode to disabled')
    parser.add_argument('--run_name', type=str, default=None, help='Custom run name (defaults to auto-generated)')
    parser.add_argument('--gpu_id', type=int, default=None, help="Manually specify GPU ID to use (default: auto-select GPU with most free memory)")

    # ── Visualisation ────────────────────────────────────────────────────
    parser.add_argument('--vis_every', type=int, default=5,
                        help='Save qualitative visualisations every N epochs (0 = disable)')
    parser.add_argument('--vis_n_samples', type=int, default=16,
                        help='Number of validation samples to visualise per checkpoint')

    args = parser.parse_args()
    return args

def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def build_model(args) -> nn.Module:
    if args.arch == 'cnn':
        model = build_cnn_classifier(
            backbone=args.backbone,
            pretrained=args.pretrained,
            freeze_ratio=args.freeze_ratio,
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
    return model

def build_dataloaders(args) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / valid / test DataLoaders with optional class balancing."""

    mods = ['real', 'pix2pix', 'cycle', 'diffusion']

    tab_train = load_split_table(DATA_DIR, 'train', mods)
    tab_valid = load_split_table(DATA_DIR, 'valid', mods)
    tab_test  = load_split_table(DATA_DIR, 'test',  mods)

    ds_train = NodulePatchDataset(DATA_DIR, tab_train, patch_size=args.patch_size,
                                   augment=True, jitter_px=args.jitter_px)
    ds_valid = NodulePatchDataset(DATA_DIR, tab_valid, patch_size=args.patch_size,
                                   augment=False)
    ds_test  = NodulePatchDataset(DATA_DIR, tab_test,  patch_size=args.patch_size,
                                   augment=False)

    # ── Weighted sampler to balance real vs fake in each batch ────────────
    if args.balance_classes:
        labels = tab_train['mod'].apply(lambda m: 0 if m == 'real' else 1).values
        class_counts = np.bincount(labels)
        weights = 1.0 / class_counts[labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle_train = False
    else:
        sampler = None
        shuffle_train = True

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=shuffle_train,
                          sampler=sampler, num_workers=args.num_workers,
                          pin_memory=True, drop_last=True)
    dl_valid = DataLoader(ds_valid, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)
    dl_test  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    print(f"  Train: {len(ds_train)} samples  ({len(dl_train)} batches)")
    print(f"  Valid: {len(ds_valid)} samples  ({len(dl_valid)} batches)")
    print(f"  Test:  {len(ds_test)} samples  ({len(dl_test)} batches)")

    return dl_train, dl_valid, dl_test

# =========================================================================
#  Loss functions
# =========================================================================

class PhaseALoss(nn.Module):
    """
    BCE classification loss, optionally combined with an auxiliary
    attention‐supervision loss that encourages the predicted attention map
    to overlap with the ground-truth manipulation mask.

    loss = BCE(logits, label) + aux_weight * attn_loss(attn, mask)

    The auxiliary loss is only applied to **fake** samples (where mask != 0),
    so the classifier remains free to attend anywhere on real samples.
    """

    def __init__(self, aux_attn: bool = False, aux_weight: float = 0.5,
                 attn_loss_type: str = 'bce'):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.aux_attn = aux_attn
        self.aux_weight = aux_weight
        self.attn_loss_type = attn_loss_type

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                attn: torch.Tensor | None = None,
                masks: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """
        Args:
            logits: (B, 1) raw classification logits
            labels: (B,) int labels 0/1
            attn:   (B, 1, H, W) predicted attention map  [0, 1]
            masks:  (B, 1, H, W) ground-truth mask  {0, 1}
        Returns:
            dict with 'total', 'cls', and optionally 'attn'
        """
        # ── Classification loss — target is binary: real=0, any fake>0 ────
        binary_labels = (labels > 0).float()
        cls_loss = self.bce(logits.squeeze(1), binary_labels)
        result = {'cls': cls_loss}

        # ── Auxiliary attention loss (fake samples only) ─────────────────
        if self.aux_attn and attn is not None and masks is not None:
            fake_mask = (labels > 0)
            if fake_mask.any():
                attn_fake  = attn[fake_mask].float()     # (Nf, 1, H, W) — float32 required by BCE
                masks_fake = masks[fake_mask].float()    # (Nf, 1, H, W)

                if self.attn_loss_type == 'bce':
                    # Treat attention as probability, mask as target
                    attn_loss = F_torch.binary_cross_entropy(
                        attn_fake.clamp(1e-7, 1 - 1e-7),
                        masks_fake,
                    )
                elif self.attn_loss_type == 'mse':
                    attn_loss = F_torch.mse_loss(attn_fake, masks_fake)
                elif self.attn_loss_type == 'dice':
                    # Soft Dice loss
                    inter = (attn_fake * masks_fake).sum(dim=[1, 2, 3])
                    union = attn_fake.sum(dim=[1, 2, 3]) + masks_fake.sum(dim=[1, 2, 3])
                    attn_loss = 1.0 - (2.0 * inter / (union + 1e-7)).mean()
                else:
                    raise ValueError(f"Unknown attn_loss_type: {self.attn_loss_type}")

                result['attn'] = attn_loss
            else:
                result['attn'] = torch.tensor(0.0, device=logits.device)

            result['total'] = cls_loss + self.aux_weight * result['attn']
        else:
            result['total'] = cls_loss

        return result

# =========================================================================
#  Training & validation loops
# =========================================================================

@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, criterion: PhaseALoss,
             device: torch.device) -> dict:
    """Run evaluation and return metrics dict."""
    model.eval()

    all_logits, all_labels, all_mods, all_img_ids = [], [], [], []
    running_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        masks  = batch['mask'].to(device)

        logits, attn = model(images, return_attn=True)
        loss_dict = criterion(logits, labels, attn, masks)

        running_loss += loss_dict['total'].item()
        n_batches += 1

        all_logits.append(logits.squeeze(1).cpu())
        all_labels.append(labels.cpu())
        all_mods.extend(batch['mod'])
        all_img_ids.extend(batch['img_id'])

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()          # multi-class: 0=real,1=pix2pix,2=cycle,3=diffusion
    binary_labels = (all_labels > 0).astype(int)        # real=0, any fake=1
    probs = sigmoid_np(all_logits)                      # numerically stable, no overflow
    preds = (probs >= 0.5).astype(int)

    metrics = {
        'loss':      running_loss / max(n_batches, 1),
        'auc':       roc_auc_score(binary_labels, probs),
        'accuracy':  accuracy_score(binary_labels, preds),
        'precision': precision_score(binary_labels, preds, zero_division=0),
        'recall':    recall_score(binary_labels, preds, zero_division=0),
        'f1':        f1_score(binary_labels, preds, zero_division=0),
    }

    # ── Per-modality AUC: each fake mod vs real (binary) ─────────────────
    mods_arr = np.array(all_mods)
    real_idx = mods_arr == 'real'
    for mod in sorted(set(all_mods)):
        if mod == 'real':
            continue
        fake_idx = mods_arr == mod
        idx = real_idx | fake_idx                        # real + this fake mod
        if idx.sum() > 0 and fake_idx.sum() > 0:
            metrics[f'auc_{mod}'] = roc_auc_score(binary_labels[idx], probs[idx])
        else:
            metrics[f'auc_{mod}'] = float('nan')

    # ── Per manipulation type (removal / injection) ───────────────────────
    for ty, prefix in [('removal', 'rem_'), ('injection', 'inj_')]:
        ty_mask = np.array([str(id_).startswith(prefix) for id_ in all_img_ids])
        bl_ty   = binary_labels[ty_mask]
        pr_ty   = probs[ty_mask]
        pd_ty   = preds[ty_mask]
        if ty_mask.sum() > 0 and bl_ty.sum() > 0 and (bl_ty == 0).sum() > 0:
            metrics[f'{ty}_auc']      = float(roc_auc_score(bl_ty, pr_ty))
            metrics[f'{ty}_accuracy'] = float(accuracy_score(bl_ty, pd_ty))
            metrics[f'{ty}_f1']       = float(f1_score(bl_ty, pd_ty, zero_division=0))
        else:
            metrics[f'{ty}_auc']      = float('nan')
            metrics[f'{ty}_accuracy'] = float('nan')
            metrics[f'{ty}_f1']       = float('nan')

    return metrics

def train_one_epoch(model: nn.Module, dataloader: DataLoader,
                    criterion: PhaseALoss, optimiser: torch.optim.Optimizer,
                    scaler: torch.amp.GradScaler, device: torch.device,
                    use_amp: bool, aux_attn: bool) -> dict:
    """Train for one epoch, return average loss dict."""
    model.train()
    running = {'cls': 0.0, 'total': 0.0}
    if aux_attn:
        running['attn'] = 0.0
    n_batches = 0

    for batch in dataloader:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        masks  = batch['mask'].to(device)

        optimiser.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=use_amp):
            # Always request attention when aux loss is active
            if aux_attn:
                logits, attn = model(images, return_attn=True)
            else:
                logits = model(images, return_attn=False)
                attn = None

        # Loss runs outside autocast — F.binary_cross_entropy is unsafe inside it
        loss_dict = criterion(logits, labels, attn, masks)
        loss = loss_dict['total']

        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimiser)
        scaler.update()

        for k in running:
            if k in loss_dict:
                running[k] += loss_dict[k].item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in running.items()}

# =========================================================================
#  Periodic visualisation
# =========================================================================

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
def save_epoch_vis(model: nn.Module, dataloader: DataLoader, device: torch.device,
                  vis_dir: Path, n_samples: int = 16) -> None:
    """
    Save two 3 × 4 visualisation grids from the validation set:
        samples_removal.png   — one sample per modality from removal scans
        samples_injection.png — one sample per modality from injection scans

    Layout:
        rows  → [ Image | Attention map | Overlay + mask contour ]
        cols  → [ real | cycle | diffusion | pix2pix ]
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    vis_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    # Collect candidates per (ty, mod) then select a shared img_id across columns
    _N_CAND = 40
    cands: dict[str, dict[str, list]] = {ty: {m: [] for m in _VIS_COL_MODS} for ty in _VIS_TYS}

    for batch in dataloader:
        if all(len(cands[ty][m]) >= _N_CAND for ty in _VIS_TYS for m in _VIS_COL_MODS):
            break
        images = batch['image'].to(device)
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            logits, attn = model(images, return_attn=True)
        for i in range(len(images)):
            mod    = batch['mod'][i]
            img_id = batch['img_id'][i]
            ty     = _get_ty(img_id)
            if mod in _VIS_COL_MODS and len(cands[ty][mod]) < _N_CAND:
                cands[ty][mod].append({
                    'image':  images[i, 0].cpu().numpy(),
                    'attn':   attn[i, 0].cpu().float().numpy(),
                    'mask':   batch['mask'][i, 0].numpy(),
                    'label':  int(batch['label'][i]),
                    'mod':    mod,
                    'img_id': img_id,
                    'prob':   float(sigmoid_np(logits[i, 0].cpu().item())),
                })

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
        cols = [m for m in _VIS_COL_MODS if m in by_mod]
        if not cols:
            return
        row_labels = ['Image', 'Attention', 'Overlay']
        fig, axes = plt.subplots(3, len(cols), figsize=(3.5 * len(cols), 9),
                                 squeeze=False)
        fig.suptitle(suffix.capitalize(), fontsize=11, fontweight='bold')
        for col_idx, mod in enumerate(cols):
            s = by_mod[mod]
            gt_str   = 'real' if s['label'] == 0 else s['mod']
            pred_str = 'fake' if s['prob'] > 0.5 else 'real'
            correct  = (s['label'] == 0) == (s['prob'] <= 0.5)
            col_title = (
                f"{mod.upper()}\n"
                f"{s['img_id']}\n"
                f"GT:{gt_str}  Pred:{pred_str}(p={s['prob']:.2f})  {'✓' if correct else '✗'}"
            )
            attn_sm = _smooth_attn(s['attn'])
            axes[0, col_idx].imshow(s['image'], cmap='gray')
            axes[0, col_idx].set_title(col_title, fontsize=7)
            axes[0, col_idx].axis('off')
            if s['label'] > 0 and s['mask'].sum() > 0:
                axes[0, col_idx].contour(s['mask'], levels=[0.5], colors='lime', linewidths=1.2)
            axes[1, col_idx].imshow(attn_sm, cmap='turbo', vmin=0, vmax=1)
            axes[1, col_idx].axis('off')
            axes[2, col_idx].imshow(s['image'], cmap='gray')
            axes[2, col_idx].imshow(attn_sm, cmap='turbo', alpha=0.45, vmin=0, vmax=1)
            if s['label'] > 0 and s['mask'].sum() > 0:
                axes[2, col_idx].contour(s['mask'], levels=[0.5], colors='lime', linewidths=1.2)
            axes[2, col_idx].axis('off')
        for row_idx, rl in enumerate(row_labels):
            axes[row_idx, 0].set_ylabel(rl, fontsize=9, fontweight='bold',
                                        rotation=90, labelpad=4)
        plt.tight_layout()
        plt.savefig(vis_dir / f'samples_{suffix}.png', dpi=120, bbox_inches='tight')
        plt.close(fig)

    for ty in _VIS_TYS:
        _draw_vis_grid(by_ty[ty], ty)
    model.train()

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
    print(f'  Champion   test_acc = {champion_acc:.4f}  ({canonical_dir.name})')
    print(f'  Challenger test_acc = {challenger_acc:.4f}  ({temp_dir.name})')

    if challenger_acc >= champion_acc:
        print('  -> Challenger wins! Replacing champion.')
        shutil.rmtree(str(canonical_dir))
        shutil.move(str(temp_dir), str(canonical_dir))
        print(f'  ✓ New canonical: {canonical_dir}')
    else:
        print('  -> Champion holds. Deleting challenger run.')
        shutil.rmtree(str(temp_dir))
        print(f'  ✓ Canonical unchanged: {canonical_dir}')

# =========================================================================
#  Main
# =========================================================================

def main(args):

    # ── Debug mode
    if args.debug:
        args.wandb_mode = 'disabled'

    # ── Seed ─────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Device ───────────────────────────────────────────────────────────
    if args.gpu_id is not None:
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"📌 Using manually specified GPU {args.gpu_id}")
    else:
        gpu_id = select_best_gpu()
        if gpu_id is not None:
            device = torch.device(f'cuda:{gpu_id}')
        else:
            device = torch.device('cpu')
            print("⚠️  No GPU available, using CPU")
    print(f"\n{'='*60}")
    print(f"  Phase A Training — {args.arch.upper()}")
    print(f"{'='*60}")
    print(f"  Device:     {device}")
    print(f"  Patch size: {args.patch_size}")
    if args.arch == 'cnn':
        print(f"  Backbone:   {args.backbone} (pretrained={args.pretrained}, freeze={args.freeze_ratio})")
    else:
        print(f"  ViT config: dim={args.embed_dim}, depth={args.depth}, heads={args.num_heads}, token={args.token_size}")
    print(f"  Loss:       BCE{' + aux_attn (' + args.attn_loss_type + ', w=' + str(args.aux_weight) + ')' if args.aux_attn_loss else ''}")
    print(f"  LR={args.lr}, WD={args.weight_decay}, BS={args.batch_size}, epochs={args.epochs}")
    print(f"  Scheduler:  {args.scheduler}, warmup={args.warmup_epochs}")
    print(f"  AMP={args.amp}, balance={args.balance_classes}, seed={args.seed}")

    # ── Run name & output dir ────────────────────────────────────────────
    arch_tag = args.backbone if args.arch == 'cnn' else f'vit_d{args.depth}_e{args.embed_dim}'
    if args.run_name is None:
        loss_tag = 'bce' if not args.aux_attn_loss else f'bce+{args.attn_loss_type}'
        args.run_name = f"patch-cls_{arch_tag}_p{args.patch_size}_{loss_tag}_bs{args.batch_size}_lr{args.lr}_wd{args.weight_decay}"

    canonical_name = f"patch-cls_{arch_tag}_p{args.patch_size}"
    canonical_dir  = WORK_DIR / 'experiments' / 'runs' / canonical_name
    out_dir        = WORK_DIR / 'experiments' / 'runs' / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output dir: {out_dir}")
    print()

    # Save args
    with open(out_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # ── WandB (optional) ─────────────────────────────────────────────────
    if args.wandb_mode != 'disabled':
        import wandb
        wandb.init(project='MedForensics-PhaseA', name=args.run_name, config=vars(args), mode=args.wandb_mode)

    # ── Data ─────────────────────────────────────────────────────────────
    print("Loading data...")
    dl_train, dl_valid, dl_test = build_dataloaders(args)
    print()

    # ── Model ────────────────────────────────────────────────────────────
    model = build_model(args).to(device)
    total_p, train_p = count_parameters(model)
    print(f"Model: {total_p:,} total params, {train_p:,} trainable")
    print()

    # ── Optimiser & scheduler ────────────────────────────────────────────
    optimiser = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if args.scheduler:
        main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )
    else:
        main_scheduler = None

    # ── Loss ─────────────────────────────────────────────────────────────
    criterion = PhaseALoss(
        aux_attn=args.aux_attn_loss,
        aux_weight=args.aux_weight,
        attn_loss_type=args.attn_loss_type,
    )

    # ── AMP scaler ───────────────────────────────────────────────────────
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    # ── Training loop ────────────────────────────────────────────────────
    best_val_loss = float('inf')
    patience_counter = 0
    history = []

    print("Starting training...\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Warmup LR ───────────────────────────────────────────────────
        if epoch <= args.warmup_epochs:
            warmup_lr = args.lr * (epoch / args.warmup_epochs)
            for pg in optimiser.param_groups:
                pg['lr'] = warmup_lr

        # ── Train ────────────────────────────────────────────────────────
        train_losses = train_one_epoch(
            model, dl_train, criterion, optimiser, scaler,
            device, args.amp, args.aux_attn_loss,
        )

        # ── Validate ─────────────────────────────────────────────────────
        val_metrics = evaluate(model, dl_valid, criterion, device)

        # ── Scheduler step ───────────────────────────────────────────────
        if epoch > args.warmup_epochs and main_scheduler is not None:
            if args.scheduler:
                main_scheduler.step(val_metrics['loss'])
            else:
                main_scheduler.step()

        current_lr = optimiser.param_groups[0]['lr']
        elapsed = time.time() - t0

        # ── Logging ──────────────────────────────────────────────────────
        log_str = (
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"LR {current_lr:.2e} | "
            f"Train loss {train_losses['total']:.4f}"
        )
        if 'attn' in train_losses:
            log_str += f" (cls={train_losses['cls']:.4f}, attn={train_losses['attn']:.4f})"

        log_str += (
            f" | Val loss {val_metrics['loss']:.4f} | "
            f"AUC {val_metrics['auc']:.4f} | "
            f"Acc {val_metrics['accuracy']:.4f} | "
            f"F1 {val_metrics['f1']:.4f} | "
            f"{elapsed:.1f}s"
        )
        print(log_str)

        # Per-modality AUC
        mod_aucs = {k: v for k, v in val_metrics.items() if k.startswith('auc_')}
        if mod_aucs:
            mod_str = "  Mod AUC: " + " | ".join(f"{k.replace('auc_','')}: {v:.4f}" for k, v in sorted(mod_aucs.items()))
            print(mod_str)
        print("  Removal:   AUC {:.4f}  Acc {:.4f}  F1 {:.4f}".format(
            val_metrics.get('removal_auc', float('nan')),
            val_metrics.get('removal_accuracy', float('nan')),
            val_metrics.get('removal_f1', float('nan')),
        ))
        print("  Injection: AUC {:.4f}  Acc {:.4f}  F1 {:.4f}".format(
            val_metrics.get('injection_auc', float('nan')),
            val_metrics.get('injection_accuracy', float('nan')),
            val_metrics.get('injection_f1', float('nan')),
        ))

        # ── Periodic visualisation ───────────────────────────────────────
        if args.vis_every > 0 and epoch % args.vis_every == 0:
            vis_epoch_dir = out_dir / 'vis' / f'epoch_{epoch:03d}'
            save_epoch_vis(model, dl_valid, device, vis_epoch_dir, args.vis_n_samples)
            print(f"  → Vis saved: {vis_epoch_dir}/samples_removal.png + samples_injection.png")

        # ── WandB ────────────────────────────────────────────────────────
        if args.wandb_mode != 'disabled':
            log_dict = {
                'epoch': epoch,
                'lr': current_lr,
                'train/loss_total': train_losses['total'],
                'train/loss_cls': train_losses['cls'],
                'val/loss': val_metrics['loss'],
                'val/auc': val_metrics['auc'],
                'val/accuracy': val_metrics['accuracy'],
                'val/precision': val_metrics['precision'],
                'val/recall': val_metrics['recall'],
                'val/f1': val_metrics['f1'],
            }
            if 'attn' in train_losses:
                log_dict['train/loss_attn'] = train_losses['attn']
            for k, v in mod_aucs.items():
                log_dict[f'val/{k}'] = v
            for ty in ['removal', 'injection']:
                for metric in ['auc', 'accuracy', 'f1']:
                    k = f'{ty}_{metric}'
                    if k in val_metrics:
                        log_dict[f'val/{k}'] = val_metrics[k]
            wandb.log(log_dict)

        # ── History ──────────────────────────────────────────────────────
        epoch_record = {
            'epoch': epoch,
            'lr': current_lr,
            'train_loss': train_losses,
            'val_metrics': val_metrics,
            'time_s': elapsed,
        }
        history.append(epoch_record)

        # ── Checkpointing (best = lowest val loss) ───────────────────────
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimiser_state_dict': optimiser.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_auc':  val_metrics['auc'],
                'args': vars(args),
            }
            torch.save(ckpt, out_dir / 'best_model.pt')
            print(f"  ★ New best val loss: {best_val_loss:.4f} (AUC {val_metrics['auc']:.4f}) — saved checkpoint")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  Early stopping at epoch {epoch} (patience={args.patience})")
                break

    # ── Test evaluation ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Test Evaluation (best model)")
    print(f"{'='*60}\n")

    best_ckpt = torch.load(out_dir / 'best_model.pt', map_location=device, weights_only=True)
    model.load_state_dict(best_ckpt['model_state_dict'])

    test_metrics = evaluate(model, dl_test, criterion, device)

    for k, v in sorted(test_metrics.items()):
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.4f}")

    # Save test metrics
    with open(out_dir / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2, default=str)

    if args.wandb_mode != 'disabled':
        wandb.log({f'test/{k}': v for k, v in test_metrics.items()})
        wandb.finish()

    promote_if_better(out_dir, canonical_dir)

    print(f"\nAll outputs saved to: {out_dir}")
    print("Done.")

if __name__ == '__main__':
    args = get_args()
    main(args)
