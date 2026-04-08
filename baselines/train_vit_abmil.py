#!/usr/bin/env python3
"""
train_vit_abmil.py
==================
Training script for the ViT-ABMIL volume classifier.

Architecture:
  - ViTSliceEncoder: lightweight ViT that encodes each CT slice to a CLS token
  - Sinusoidal Z positional encoding
  - GatedAttention aggregator over K slices
  - MLP head → binary logit

End-to-end training (ViT + GatedAttention + head trained jointly).

Usage
-----
    python baselines/train_vit_abmil.py \\
        --data_dir /mnt/.../M3DSynth \\
        --out_dir  baselines/runs/vit_abmil_K16_d256 \\
        --K 16 --embed_dim 256 --depth 4 --epochs 60
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, average_precision_score,
)
from scipy.special import expit as sigmoid_np
from tqdm import tqdm

# Ensure packages are importable
WORK_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORK_DIR / 'src'))
sys.path.insert(0, str(WORK_DIR / 'baselines'))

from hexmil.data.patch_dataset import load_split_table
from hexmil.utils.tiff_utils import (
    get_shape_tiff_scan,
    load_slice_tiff_scan,
    get_percentile_tiff_scan,
    apply_percentile,
)
from models.vit_volume_classifier import build_vit_volume_classifier

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
#  Dataset
# =============================================================================

class ViTVolumeDataset(Dataset):
    """
    Dataset for ViT-ABMIL: provides K full slice images (not patch bags).

    Each sample returns:
        slices:    (K, 1, H, W) float32 — full normalised slice images
        z_indices: (K,)         int64   — absolute Z coordinates (-1 if padded)
        valid:     (K,)         bool    — True for real slices
        label:     int          0=real / 1=fake
        mod:       str
        img_id:    str
    """

    def __init__(
        self,
        data_dir: str,
        tab,
        K: int = 16,
        augment: bool = False,
    ):
        self.data_dir = data_dir
        self.tab      = tab.reset_index(drop=True)
        self.K        = K
        self.augment  = augment

    def __len__(self) -> int:
        return len(self.tab)

    def __getitem__(self, idx: int) -> dict:
        row    = self.tab.iloc[idx]
        mod    = row['mod']
        img_id = str(row['img_id'])
        cz     = int(row['coord_z'])

        scan_dir  = os.path.join(self.data_dir, mod, 'scan', img_id)
        shape     = get_shape_tiff_scan(scan_dir)
        Z_total, H, W = shape
        low, high = get_percentile_tiff_scan(scan_dir, np.uint16)

        # Build K-slice window (same logic as VolumeDataset)
        K = self.K
        if mod == 'real':
            max_start = max(0, Z_total - K)
            z_start   = int(np.random.randint(0, max_start + 1))
        else:
            rand_off = int(np.random.randint(0, K))
            z_start  = cz - rand_off
            z_end_   = z_start + K
            if z_end_ > Z_total:
                z_start = Z_total - K
            if z_start < 0:
                z_start = 0
        z_end       = min(z_start + K, Z_total)
        avail_start = max(z_start, 0)
        avail_end   = min(z_end, Z_total)

        # Allocate outputs
        slices_out   = torch.zeros(K, 1, H, W, dtype=torch.float32)
        z_indices    = torch.full((K,), fill_value=-1, dtype=torch.long)
        valid        = torch.zeros(K, dtype=torch.bool)

        if avail_end > avail_start:
            chunk = load_slice_tiff_scan(scan_dir, shape, np.uint16, avail_start, avail_end)
            chunk = apply_percentile(chunk.astype(np.float32), low, high)  # [0,1]

            for local_z in range(avail_end - avail_start):
                global_z = avail_start + local_z
                win_idx  = global_z - z_start
                if win_idx < 0 or win_idx >= K:
                    continue

                sl = chunk[local_z]

                if self.augment:
                    if np.random.rand() > 0.5:
                        sl = sl[::-1, :].copy()
                    if np.random.rand() > 0.5:
                        sl = sl[:, ::-1].copy()

                slices_out[win_idx, 0]  = torch.from_numpy(sl.copy()).float()
                z_indices[win_idx]      = global_z
                valid[win_idx]          = True

        label = 0 if mod == 'real' else 1
        return {
            'slices':    slices_out,   # (K, 1, H, W)
            'z_indices': z_indices,    # (K,)
            'valid':     valid,        # (K,)
            'label':     label,
            'mod':       mod,
            'img_id':    img_id,
        }


# =============================================================================
#  Custom collate (handle variable H,W by padding to max in batch)
# =============================================================================

def vit_collate(batch: list[dict]) -> dict:
    """Collate ViTVolumeDataset samples, padding slices to the max H,W in batch."""
    K = batch[0]['slices'].shape[0]
    max_H = max(s['slices'].shape[2] for s in batch)
    max_W = max(s['slices'].shape[3] for s in batch)

    slices_list   = []
    z_indices_list = []
    valid_list    = []
    labels_list   = []
    mods_list     = []
    img_ids_list  = []

    for s in batch:
        sl = s['slices']                                # (K, 1, H, W)
        H, W = sl.shape[2], sl.shape[3]
        if H != max_H or W != max_W:
            # Pad with zeros on right/bottom
            padded = torch.zeros(K, 1, max_H, max_W, dtype=sl.dtype)
            padded[:, :, :H, :W] = sl
            sl = padded
        slices_list.append(sl)
        z_indices_list.append(s['z_indices'])
        valid_list.append(s['valid'])
        labels_list.append(s['label'])
        mods_list.append(s['mod'])
        img_ids_list.append(s['img_id'])

    return {
        'slices':    torch.stack(slices_list),      # (B, K, 1, H, W)
        'z_indices': torch.stack(z_indices_list),   # (B, K)
        'valid':     torch.stack(valid_list),       # (B, K)
        'label':     torch.tensor(labels_list, dtype=torch.long),
        'mod':       mods_list,
        'img_id':    img_ids_list,
    }


# =============================================================================
#  Training helpers
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


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device) -> tuple:
    model.eval()
    all_logits, all_labels, all_mods = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            slices    = batch['slices'].to(device)     # (B, K, 1, H, W)
            z_indices = batch['z_indices'].to(device)  # (B, K)
            valid     = batch['valid'].to(device)      # (B, K)
            labels    = batch['label'].float().to(device)

            B = slices.shape[0]
            logits_batch = []
            for b in range(B):
                logit = model(slices[b].repeat(1, 3, 1, 1), z_indices[b], valid[b])  # (K,1,H,W)→3-ch
                if isinstance(logit, (tuple, list)):
                    logit = logit[0]
                logits_batch.append(logit)

            logits = torch.stack(logits_batch)         # (B,)
            if logits.dim() > 1:
                logits = logits.squeeze(-1)
            loss = criterion(logits, labels)
            total_loss += loss.item() * B
            all_logits.extend(logits.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_mods.extend(batch['mod'])

    scores_np = sigmoid_np(np.array(all_logits))
    labels_np = np.array(all_labels)
    preds_np  = (scores_np >= 0.5).astype(int)

    metrics = {'loss': total_loss / max(len(all_labels), 1)}
    if len(set(labels_np.tolist())) > 1:
        metrics['auc']      = float(roc_auc_score(labels_np, scores_np))
        metrics['accuracy'] = float(accuracy_score(labels_np, preds_np))
        metrics['f1']       = float(f1_score(labels_np, preds_np, zero_division=0))
        metrics['ap']       = float(average_precision_score(labels_np, scores_np))
    else:
        metrics['auc']      = float('nan')
        metrics['accuracy'] = float('nan')
        metrics['f1']       = float('nan')
        metrics['ap']       = float('nan')
    return metrics, labels_np, scores_np, all_mods


# =============================================================================
#  Args
# =============================================================================

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train ViT-ABMIL volume classifier')
    p.add_argument('--data_dir',     type=str, required=True)
    p.add_argument('--out_dir',      type=str, required=True)
    p.add_argument('--K',            type=int, default=16)
    p.add_argument('--embed_dim',    type=int, default=256)
    p.add_argument('--depth',        type=int, default=4)
    p.add_argument('--num_heads',    type=int, default=8)
    p.add_argument('--token_size',   type=int, default=32)
    p.add_argument('--target_size',  type=int, default=256,
                   help='Resize slices to this HxW before ViT tokenisation')
    p.add_argument('--attn_dim',     type=int, default=128)
    p.add_argument('--dropout',      type=float, default=0.25)
    p.add_argument('--epochs',       type=int, default=60)
    p.add_argument('--batch_size',   type=int, default=4)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--num_workers',  type=int, default=4)
    p.add_argument('--patience',     type=int, default=15)
    p.add_argument('--seed',         type=int, default=42)
    p.add_argument('--gpu_id',       type=int, default=None)
    p.add_argument('--train_mods',   nargs='+',
                   default=['real', 'cycle', 'diffusion', 'pix2pix'])
    p.add_argument('--use_wandb',    action='store_true', default=False)
    p.add_argument('--run_name',     type=str, default=None)
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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    args_dict = vars(args)
    with open(out_dir / 'args.json', 'w') as f:
        json.dump(args_dict, f, indent=2)

    # Data
    train_mods = args.train_mods
    fake_mods  = [m for m in train_mods if m != 'real']
    train_mods_with_real = ['real'] + fake_mods
    test_mods  = ['real', 'cycle', 'diffusion', 'pix2pix']

    tab_train = load_split_table(args.data_dir, 'train', train_mods_with_real)
    tab_valid = load_split_table(args.data_dir, 'valid', train_mods_with_real)
    tab_test  = load_split_table(args.data_dir, 'test',  test_mods)

    ds_train = ViTVolumeDataset(args.data_dir, tab_train, K=args.K, augment=True)
    ds_valid = ViTVolumeDataset(args.data_dir, tab_valid, K=args.K, augment=False)
    ds_test  = ViTVolumeDataset(args.data_dir, tab_test,  K=args.K, augment=False)

    # Weighted sampler
    labels_arr = tab_train['mod'].apply(lambda m: 0 if m == 'real' else 1).values
    cls_count  = np.bincount(labels_arr)
    cls_w      = 1.0 / np.maximum(cls_count, 1)
    sample_w   = cls_w[labels_arr]
    sampler    = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

    kw = dict(num_workers=args.num_workers, pin_memory=True, collate_fn=vit_collate)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler, **kw)
    dl_valid = DataLoader(ds_valid, batch_size=args.batch_size, shuffle=False, **kw)
    dl_test  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False, **kw)

    # Model
    model = build_vit_volume_classifier(
        K=args.K,
        token_size=args.token_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        target_size=args.target_size,
        attn_dim=args.attn_dim,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"ViT-ABMIL params: {n_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    if args.use_wandb:
        import wandb
        run_name = args.run_name or f"vit_abmil_K{args.K}_d{args.embed_dim}"
        wandb.init(project='MedForensics-baselines', name=run_name, config=args_dict)

    best_val_auc     = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n_train    = 0

        for batch in tqdm(dl_train, desc=f'Epoch {epoch:03d}', leave=False, dynamic_ncols=True):
            slices    = batch['slices'].to(device)     # (B, K, 1, H, W)
            z_indices = batch['z_indices'].to(device)  # (B, K)
            valid     = batch['valid'].to(device)      # (B, K)
            labels    = batch['label'].float().to(device)

            B = slices.shape[0]
            optimiser.zero_grad()

            logits_batch = []
            for b in range(B):
                logit = model(slices[b].repeat(1, 3, 1, 1), z_indices[b], valid[b])  # (K,1,H,W)→3-ch
                if isinstance(logit, (tuple, list)):
                    logit = logit[0]
                logits_batch.append(logit)

            logits = torch.stack(logits_batch)
            if logits.dim() > 1:
                logits = logits.squeeze(-1)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            train_loss += loss.item() * B
            n_train    += B

        scheduler.step()
        train_loss /= max(n_train, 1)

        val_metrics, _, _, _ = evaluate(model, dl_valid, criterion, device)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"Train {train_loss:.4f}  "
            f"Val loss={val_metrics['loss']:.4f}  "
            f"AUC={val_metrics['auc']:.4f}  "
            f"Acc={val_metrics['accuracy']:.4f}  "
            f"({elapsed:.1f}s)"
        )

        if args.use_wandb:
            import wandb
            wandb.log({
                'train/loss': train_loss,
                'val/loss':   val_metrics['loss'],
                'val/auc':    val_metrics['auc'],
                'val/acc':    val_metrics['accuracy'],
                'lr':         scheduler.get_last_lr()[0],
            })

        val_auc = val_metrics.get('auc', 0.0)
        if not np.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc     = val_auc
            patience_counter = 0
            torch.save(
                {
                    'epoch':            epoch,
                    'model_state_dict': model.state_dict(),
                    'best_val_auc':     best_val_auc,
                    'args':             args_dict,
                },
                out_dir / 'best_model.pt'
            )
            print(f"  * New best val AUC: {best_val_auc:.4f} — saved")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Test evaluation
    print("\n=== Test Evaluation ===")
    ckpt = torch.load(out_dir / 'best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    test_metrics, t_labels, t_scores, t_mods = evaluate(model, dl_test, criterion, device)
    test_metrics['per_mod'] = per_mod_metrics(t_labels, t_scores, t_mods)
    print(f"  Test AUC={test_metrics['auc']:.4f}  Acc={test_metrics['accuracy']:.4f}  "
          f"F1={test_metrics['f1']:.4f}  AP={test_metrics['ap']:.4f}")
    for mod, mm in test_metrics.get('per_mod', {}).items():
        print(f"    {mod:12s}  AUC={mm['auc']:.4f}  F1={mm['f1']:.4f}  AP={mm['ap']:.4f}")

    with open(out_dir / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)

    if args.use_wandb:
        import wandb
        wandb.log({f'test/{k}': v for k, v in test_metrics.items()
                   if isinstance(v, (int, float))})
        wandb.finish()

    print(f"\nDone. Outputs: {out_dir}")


if __name__ == '__main__':
    main()
