#!/usr/bin/env python3
"""
train_flat_cnn.py
=================
Baseline: standard ResNet-50 on full 2D CT slices (no patch extraction,
no MIL attention). Volume-level prediction via max/mean probability
across K slices at inference time.

Proves: hierarchical patch-level MIL adds value over a flat 2D classifier.

Usage
-----
    python baselines/train_flat_cnn.py \
        --data_dir /mnt/.../M3DSynth \
        --out_dir  baselines/runs/flat_cnn_resnet50 \
        --epochs 50 --batch_size 16 --lr 1e-4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, average_precision_score,
)
from scipy.special import expit as sigmoid_np
from tqdm import tqdm

# Ensure hexmil is importable
WORK_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORK_DIR))
sys.path.insert(0, str(WORK_DIR / 'src'))
sys.path.insert(0, str(WORK_DIR / 'baselines'))
from config import DATA_DIR

from hexmil.data.patch_dataset import load_split_table, MOD_LABEL
from hexmil.utils.tiff_utils import (
    get_shape_tiff_scan,
    load_slice_tiff_scan,
    get_percentile_tiff_scan,
    apply_percentile,
)

ALL_FAKES = ['pix2pix', 'cycle', 'diffusion']


def select_device(gpu_id: int | None = None) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device('cpu')
    if gpu_id is not None:
        return torch.device(f'cuda:{gpu_id}')
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
            capture_output=True, text=True, check=True,
        ).stdout.strip().split('\n')
        free = [int(x) for x in out]
        best = free.index(max(free))
        return torch.device(f'cuda:{best}')
    except Exception:
        return torch.device('cuda')


# =============================================================================
#  Datasets
# =============================================================================

class FlatSliceDataset(Dataset):
    """One sample = one full axial slice at coord_z, resized to target_size."""

    def __init__(self, data_dir, tab, target_size=224, augment=False):
        self.data_dir    = data_dir
        self.tab         = tab.reset_index(drop=True)
        self.target_size = target_size
        self.augment     = augment

    def __len__(self):
        return len(self.tab)

    def __getitem__(self, idx):
        row    = self.tab.iloc[idx]
        mod    = row['mod']
        img_id = str(row['img_id'])
        cz     = int(row['coord_z'])

        scan_dir = os.path.join(self.data_dir, mod, 'scan', img_id)
        shape    = get_shape_tiff_scan(scan_dir)
        low, high = get_percentile_tiff_scan(scan_dir, np.uint16)

        sl = load_slice_tiff_scan(scan_dir, shape, np.uint16, cz, cz + 1)[0]
        sl = apply_percentile(sl.astype(np.float32), low, high)

        if self.augment:
            if np.random.rand() > 0.5:
                sl = sl[::-1, :].copy()
            if np.random.rand() > 0.5:
                sl = sl[:, ::-1].copy()

        img = torch.from_numpy(sl).unsqueeze(0).float()  # (1, H, W)
        ts  = self.target_size
        if img.shape[1] != ts or img.shape[2] != ts:
            img = F.interpolate(
                img.unsqueeze(0), size=(ts, ts),
                mode='bilinear', align_corners=False,
            ).squeeze(0)

        label = 0 if mod == 'real' else 1
        return dict(image=img, label=label, mod=mod, img_id=img_id, coord_z=cz)

class VolumeSliceDataset(Dataset):
    """K-slice window dataset for volume-level evaluation of a 2D model."""

    def __init__(self, data_dir, tab, K=16, target_size=224):
        self.data_dir    = data_dir
        self.tab         = tab.reset_index(drop=True)
        self.K           = K
        self.target_size = target_size

    def __len__(self):
        return len(self.tab)

    def __getitem__(self, idx):
        row    = self.tab.iloc[idx]
        mod    = row['mod']
        img_id = str(row['img_id'])
        cz     = int(row['coord_z'])

        scan_dir = os.path.join(self.data_dir, mod, 'scan', img_id)
        shape    = get_shape_tiff_scan(scan_dir)
        Z, H, W  = shape
        low, high = get_percentile_tiff_scan(scan_dir, np.uint16)

        K = self.K
        if mod == 'real':
            z_start = int(np.random.randint(0, max(1, Z - K + 1)))
        else:
            rand_off = int(np.random.randint(0, K))
            z_start  = cz - rand_off
            z_end    = z_start + K
            if z_end > Z:
                z_end   = Z
                z_start = z_end - K
            if z_start < 0:
                z_start = 0
        z_end = z_start + K

        chunk = load_slice_tiff_scan(scan_dir, shape, np.uint16, z_start, z_end)
        chunk = apply_percentile(chunk.astype(np.float32), low, high)

        n = chunk.shape[0]
        ts = self.target_size
        slices_out = torch.zeros(K, 1, ts, ts, dtype=torch.float32)
        valid      = torch.zeros(K, dtype=torch.bool)

        for i in range(n):
            sl = torch.from_numpy(chunk[i]).unsqueeze(0).unsqueeze(0).float()
            if sl.shape[2] != ts or sl.shape[3] != ts:
                sl = F.interpolate(sl, size=(ts, ts), mode='bilinear',
                                   align_corners=False)
            slices_out[i] = sl.squeeze(0)
            valid[i] = True

        label = 0 if mod == 'real' else 1
        return dict(slices=slices_out, valid=valid, label=label,
                    mod=mod, img_id=img_id)

# =============================================================================
#  Model
# =============================================================================

def build_flat_cnn(pretrained=True, dropout=0.25):
    """ResNet-50 original 3-ch backbone → binary logit.
    1-ch CT slices are repeated to 3-ch in the training loop."""
    import timm
    model = timm.create_model(
        'resnet50', pretrained=pretrained, num_classes=1,
    )
    feat_dim = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(feat_dim, 1),
    )
    return model

# =============================================================================
#  Metrics
# =============================================================================

def per_mod_metrics(labels, scores, mods):
    results = {}
    real_mask = np.array([m == 'real' for m in mods])
    for fake in ALL_FAKES:
        fake_mask = np.array([m == fake for m in mods])
        subset = real_mask | fake_mask
        if subset.sum() < 2 or len(set(labels[subset].tolist())) < 2:
            continue
        sl, ss = labels[subset], scores[subset]
        sp = (ss >= 0.5).astype(int)
        results[fake] = dict(
            auc=float(roc_auc_score(sl, ss)),
            acc=float(accuracy_score(sl, sp)),
            f1=float(f1_score(sl, sp, zero_division=0)),
            ap=float(average_precision_score(sl, ss)),
        )
    return results

# =============================================================================
#  Training helpers
# =============================================================================

def evaluate_slices(model, loader, criterion, device):
    model.eval()
    all_logits, all_labels, all_mods = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            imgs   = batch['image'].to(device).repeat(1, 3, 1, 1)   # 1-ch → 3-ch
            labels = batch['label'].float().to(device)
            logits = model(imgs)
            if logits.dim() == 2:
                logits = logits.squeeze(1)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            all_logits.extend(logits.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_mods.extend(batch['mod'])

    scores = sigmoid_np(np.array(all_logits))
    labels = np.array(all_labels)
    preds  = (scores >= 0.5).astype(int)

    m = dict(loss=total_loss / max(len(labels), 1))
    if len(set(labels.tolist())) > 1:
        m['auc'] = float(roc_auc_score(labels, scores))
        m['acc'] = float(accuracy_score(labels, preds))
        m['f1']  = float(f1_score(labels, preds, zero_division=0))
        m['ap']  = float(average_precision_score(labels, scores))
    return m, labels, scores, all_mods

@torch.no_grad()
def evaluate_volumes(model, loader, device, agg='max'):
    """Volume-level evaluation: aggregate 2D predictions over K slices."""
    model.eval()
    all_probs, all_labels, all_mods = [], [], []

    for batch in tqdm(loader, desc='Vol eval', leave=False, dynamic_ncols=True):
        slices = batch['slices'].to(device)   # (B, K, 1, H, W)
        valid  = batch['valid']               # (B, K)
        labels = batch['label']
        mods   = batch['mod']

        B, K = slices.shape[0], slices.shape[1]
        for b in range(B):
            valid_idx = valid[b].nonzero(as_tuple=True)[0]
            if len(valid_idx) == 0:
                all_probs.append(0.5)
                all_labels.append(labels[b].item())
                all_mods.append(mods[b])
                continue

            sl_batch = slices[b, valid_idx].repeat(1, 3, 1, 1)  # (n_valid, 1, H, W) → 3-ch
            logits   = model(sl_batch)
            if logits.dim() == 2:
                logits = logits.squeeze(1)
            probs = torch.sigmoid(logits)

            if agg == 'max':
                vol_prob = probs.max().item()
            else:
                vol_prob = probs.mean().item()

            all_probs.append(vol_prob)
            all_labels.append(labels[b].item())
            all_mods.append(mods[b])

    scores = np.array(all_probs)
    labels = np.array(all_labels)
    preds  = (scores >= 0.5).astype(int)

    m = {}
    if len(set(labels.tolist())) > 1:
        m['auc'] = float(roc_auc_score(labels, scores))
        m['acc'] = float(accuracy_score(labels, preds))
        m['f1']  = float(f1_score(labels, preds, zero_division=0))
        m['ap']  = float(average_precision_score(labels, scores))
        m['per_mod'] = per_mod_metrics(labels, scores, all_mods)
    return m

# =============================================================================
#  Args
# =============================================================================

def get_args():
    p = argparse.ArgumentParser(description='Flat CNN baseline (ResNet-50 on slices)')
    p.add_argument('--data_dir',     type=str, default=DATA_DIR)
    p.add_argument('--out_dir',      type=str, default=None)
    p.add_argument('--K',            type=int, default=16,
                   help='K-slice window for volume-level evaluation')
    p.add_argument('--target_size',  type=int, default=224)
    p.add_argument('--pretrained',   action='store_true', default=True)
    p.add_argument('--dropout',      type=float, default=0.25)
    p.add_argument('--epochs',       type=int, default=50)
    p.add_argument('--batch_size',   type=int, default=16)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--patience',     type=int, default=10)
    p.add_argument('--num_workers',  type=int, default=4)
    p.add_argument('--seed',         type=int, default=42)
    p.add_argument('--gpu_id',       type=int, default=None)
    p.add_argument('--train_mods',   nargs='+', default=None,
                   help='Fake modalities for training (default: all)')
    p.add_argument('--vol_agg',      type=str, default='max',
                   choices=['max', 'mean'])
    p.add_argument('--use_wandb',    action='store_true', default=False)
    p.add_argument('--run_name',     type=str, default=None)
    p.add_argument('--amp',          action='store_true', default=True)
    return p.parse_args()

# =============================================================================
#  Main
# =============================================================================

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = select_device(args.gpu_id)
    print(f"Device: {device}")

    # Output directory
    train_mods = args.train_mods or ALL_FAKES
    mods_tag   = '+'.join(sorted(train_mods))
    if args.out_dir is None:
        args.out_dir = str(
            WORK_DIR / 'baselines' / 'runs'
            / f'flat_cnn_resnet50_K{args.K}' / f'trained_on_{mods_tag}'
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    args_dict = vars(args)
    with open(out_dir / 'args.json', 'w') as f:
        json.dump(args_dict, f, indent=2)

    # Data — slice-level for training
    train_mods_with_real = ['real'] + train_mods
    test_mods            = ['real'] + ALL_FAKES

    tab_train = load_split_table(args.data_dir, 'train', train_mods_with_real)
    tab_valid = load_split_table(args.data_dir, 'valid', train_mods_with_real)
    tab_test  = load_split_table(args.data_dir, 'test',  test_mods)

    ds_train = FlatSliceDataset(args.data_dir, tab_train, args.target_size, augment=True)
    ds_valid = FlatSliceDataset(args.data_dir, tab_valid, args.target_size, augment=False)
    ds_test  = FlatSliceDataset(args.data_dir, tab_test,  args.target_size, augment=False)

    # Weighted sampler
    labels_arr = tab_train['mod'].apply(lambda m: 0 if m == 'real' else 1).values
    cls_count  = np.bincount(labels_arr)
    cls_w      = 1.0 / np.maximum(cls_count, 1)
    sample_w   = cls_w[labels_arr]
    sampler    = WeightedRandomSampler(sample_w, num_samples=len(sample_w),
                                       replacement=True)

    kw = dict(num_workers=args.num_workers, pin_memory=True)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size,
                          sampler=sampler, **kw)
    dl_valid = DataLoader(ds_valid, batch_size=args.batch_size,
                          shuffle=False, **kw)
    dl_test  = DataLoader(ds_test,  batch_size=args.batch_size,
                          shuffle=False, **kw)

    # Model
    model = build_flat_cnn(pretrained=args.pretrained, dropout=args.dropout)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Flat CNN (ResNet-50)  params={n_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=args.lr * 0.01,
    )
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    if args.use_wandb:
        import wandb
        run_name = args.run_name or f"flat_cnn_K{args.K}_{mods_tag}"
        wandb.init(project='MedForensics-baselines', name=run_name,
                   config=args_dict)

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss    = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss, n_train = 0.0, 0

        for batch in tqdm(dl_train, desc=f'Epoch {epoch:03d}',
                          leave=False, dynamic_ncols=True):
            imgs   = batch['image'].to(device).repeat(1, 3, 1, 1)   # 1-ch → 3-ch
            labels = batch['label'].float().to(device)

            optimiser.zero_grad()
            with torch.amp.autocast('cuda', enabled=args.amp):
                logits = model(imgs)
                if logits.dim() == 2:
                    logits = logits.squeeze(1)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimiser)
            scaler.update()

            train_loss += loss.item() * len(labels)
            n_train    += len(labels)

        scheduler.step()
        train_loss /= max(n_train, 1)

        val_m, _, _, _ = evaluate_slices(model, dl_valid, criterion, device)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"Train {train_loss:.4f}  "
            f"Val loss={val_m['loss']:.4f}  "
            f"AUC={val_m.get('auc', float('nan')):.4f}  "
            f"({elapsed:.1f}s)"
        )

        if args.use_wandb:
            import wandb
            wandb.log({
                'train/loss': train_loss,
                'val/loss':   val_m['loss'],
                'val/auc':    val_m.get('auc', float('nan')),
                'lr':         scheduler.get_last_lr()[0],
            })

        if val_m['loss'] < best_val_loss:
            best_val_loss    = val_m['loss']
            patience_counter = 0
            torch.save(
                dict(epoch=epoch, model_state_dict=model.state_dict(),
                     best_val_loss=best_val_loss, args=args_dict),
                out_dir / 'best_model.pt',
            )
            print(f"  * New best val loss: {best_val_loss:.4f} — saved")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # ── Test evaluation (slice-level) ─────────────────────────────────────
    print("\n=== Slice-Level Test ===")
    ckpt = torch.load(out_dir / 'best_model.pt', map_location=device,
                       weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    test_m, t_labels, t_scores, t_mods = evaluate_slices(
        model, dl_test, criterion, device,
    )
    test_m['per_mod'] = per_mod_metrics(t_labels, t_scores, t_mods)
    print(f"  Slice AUC={test_m.get('auc', float('nan')):.4f}  "
          f"Acc={test_m.get('acc', float('nan')):.4f}")
    for mod, mm in test_m.get('per_mod', {}).items():
        print(f"    {mod:12s}  AUC={mm['auc']:.4f}  F1={mm['f1']:.4f}")

    # ── Test evaluation (volume-level, max/mean over K slices) ────────────
    print(f"\n=== Volume-Level Test (agg={args.vol_agg}, K={args.K}) ===")
    ds_test_vol = VolumeSliceDataset(
        args.data_dir, tab_test, K=args.K, target_size=args.target_size,
    )
    dl_test_vol = DataLoader(ds_test_vol, batch_size=args.batch_size,
                             shuffle=False, **kw)
    vol_m = evaluate_volumes(model, dl_test_vol, device, agg=args.vol_agg)
    print(f"  Volume AUC={vol_m.get('auc', float('nan')):.4f}  "
          f"Acc={vol_m.get('acc', float('nan')):.4f}")
    for mod, mm in vol_m.get('per_mod', {}).items():
        print(f"    {mod:12s}  AUC={mm['auc']:.4f}  F1={mm['f1']:.4f}")

    # ── Save results ──────────────────────────────────────────────────────
    all_metrics = dict(
        slice=test_m,
        volume=vol_m,
        args=args_dict,
    )
    eval_dir = out_dir / 'evaluation'
    eval_dir.mkdir(exist_ok=True)
    with open(eval_dir / 'metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)

    if args.use_wandb:
        import wandb
        wandb.log({f'test/slice/{k}': v for k, v in test_m.items()
                   if isinstance(v, (int, float))})
        wandb.log({f'test/volume/{k}': v for k, v in vol_m.items()
                   if isinstance(v, (int, float))})
        wandb.finish()

    print(f"\nDone. Outputs: {out_dir}")

if __name__ == '__main__':
    main()
