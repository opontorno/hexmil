#!/usr/bin/env python3
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
    roc_auc_score,
    accuracy_score,
    f1_score,
    average_precision_score,
)
from scipy.special import expit as sigmoid_np
from tqdm import tqdm

# Ensure hexmil package is importable
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
from models.vit3d_classifier import (
    build_vit3d_classifier,
    build_vivit_factorised_classifier,
    ARCH_CONFIGS,
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


class Volume3DDataset(Dataset):

    TARGET_HW = 224

    def __init__(
        self,
        data_dir: str,
        tab,
        K: int = 16,
        augment: bool = False,
    ):
        self.data_dir = data_dir
        self.tab = tab.reset_index(drop=True)
        self.K = K
        self.augment = augment

    def __len__(self) -> int:
        return len(self.tab)

    def __getitem__(self, idx: int) -> dict:
        row = self.tab.iloc[idx]
        mod = row['mod']
        img_id = str(row['img_id'])
        cz = int(row['coord_z'])

        scan_dir = os.path.join(self.data_dir, mod, 'scan', img_id)
        shape = get_shape_tiff_scan(scan_dir)
        z_total, h, w = shape
        low, high = get_percentile_tiff_scan(scan_dir, np.uint16)

        K = self.K
        if mod == 'real':
            max_start = max(0, z_total - K)
            z_start = int(np.random.randint(0, max_start + 1))
        else:
            rand_off = int(np.random.randint(0, K))
            z_start = cz - rand_off
            z_end_ = z_start + K
            if z_end_ > z_total:
                z_start = z_total - K
            if z_start < 0:
                z_start = 0
        z_end = min(z_start + K, z_total)

        chunk = load_slice_tiff_scan(scan_dir, shape, np.uint16, z_start, z_end)
        chunk = apply_percentile(chunk.astype(np.float32), low, high)

        n_loaded = chunk.shape[0]
        if n_loaded < K:
            pad = np.zeros((K - n_loaded, h, w), dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=0)

        if self.augment:
            if np.random.rand() > 0.5:
                chunk = chunk[:, ::-1, :].copy()
            if np.random.rand() > 0.5:
                chunk = chunk[:, :, ::-1].copy()

        vol_t = torch.from_numpy(chunk).unsqueeze(0).float()   # (1, K, H, W)
        if h != self.TARGET_HW or w != self.TARGET_HW:
            vol_t = F.interpolate(
                vol_t.unsqueeze(0),
                size=(K, self.TARGET_HW, self.TARGET_HW),
                mode='trilinear',
                align_corners=False,
            ).squeeze(0)

        return {
            'volume': vol_t,
            'label':  0 if mod == 'real' else 1,
            'mod':    mod,
            'img_id': img_id,
        }


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


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    model.eval()
    all_logits, all_labels, all_mods = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            vol = batch['volume'].to(device).repeat(1, 3, 1, 1, 1)  # (B,1,K,H,W)->3-ch
            labels = batch['label'].float().to(device)
            logits = model(vol).squeeze(1)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
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
        metrics['auc'] = metrics['accuracy'] = metrics['f1'] = metrics['ap'] = float('nan')
    return metrics, labels_np, scores_np, all_mods


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train ViT 3D / ViViT-F baseline')
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--out_dir',  type=str, required=True)
    p.add_argument('--arch', type=str, default='vit3d_base',
                   choices=list(ARCH_CONFIGS),
                   help='vit3d_tiny (~5M), vit3d_small (~22M), vit3d_base (~86M)')
    p.add_argument('--variant', type=str, default='plain',
                   choices=['plain', 'factorised'],
                   help='plain: full joint self-attention (ViT3D); '
                        'factorised: spatial+temporal factorised attention (ViViT-F)')
    p.add_argument('--patch_t', type=int, default=2,
                   help='Temporal patch size (must divide K)')
    p.add_argument('--patch_s', type=int, default=32,
                   help='Spatial patch size (must divide img_size=224); '
                        'use 16 to match pretrained 2D ViT patch grid exactly')
    p.add_argument('--attn_drop', type=float, default=0.0,
                   help='Dropout inside attention softmax')
    p.add_argument('--K',           type=int,   default=16)
    p.add_argument('--pretrained',  action='store_true', default=False,
                   help='Inflate 2D ImageNet ViT weights (timm) into the 3D model')
    p.add_argument('--dropout',     type=float, default=0.1)
    p.add_argument('--epochs',      type=int,   default=60)
    p.add_argument('--batch_size',  type=int,   default=4)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--weight_decay',type=float, default=1e-4)
    p.add_argument('--num_workers', type=int,   default=4)
    p.add_argument('--patience',    type=int,   default=15)
    p.add_argument('--seed',        type=int,   default=42)
    p.add_argument('--gpu_id',      type=int,   default=None)
    p.add_argument('--train_mods',  nargs='+',
                   default=['real', 'cycle', 'diffusion', 'pix2pix'])
    p.add_argument('--use_wandb',   action='store_true', default=False)
    p.add_argument('--run_name',    type=str,   default=None)
    return p.parse_args()


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = select_device(args.gpu_id)
    print(f'Device: {device}')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    args_dict = vars(args)
    with open(out_dir / 'args.json', 'w') as f:
        json.dump(args_dict, f, indent=2)

    train_mods = args.train_mods
    fake_mods  = [m for m in train_mods if m != 'real']
    test_mods  = ['real', 'cycle', 'diffusion', 'pix2pix']

    tab_train = load_split_table(args.data_dir, 'train', ['real'] + fake_mods)
    tab_valid = load_split_table(args.data_dir, 'valid', ['real'] + fake_mods)
    tab_test  = load_split_table(args.data_dir, 'test',  test_mods)

    ds_train = Volume3DDataset(args.data_dir, tab_train, K=args.K, augment=True)
    ds_valid = Volume3DDataset(args.data_dir, tab_valid, K=args.K, augment=False)
    ds_test  = Volume3DDataset(args.data_dir, tab_test,  K=args.K, augment=False)

    labels_arr = tab_train['mod'].apply(lambda m: 0 if m == 'real' else 1).values
    cls_count  = np.bincount(labels_arr)
    cls_w      = 1.0 / np.maximum(cls_count, 1)
    sample_w   = cls_w[labels_arr]
    sampler    = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

    kw = dict(num_workers=args.num_workers, pin_memory=True)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler, **kw)
    dl_valid = DataLoader(ds_valid, batch_size=args.batch_size, shuffle=False, **kw)
    dl_test  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False, **kw)

    n_t = args.K // args.patch_t
    n_s = (224 // args.patch_s) ** 2
    cls_info = '+ 1 CLS' if args.variant == 'plain' else '(global avg pool)'
    print(
        f'ViT3D config: variant={args.variant}  arch={args.arch}  '
        f'patch_t={args.patch_t}  patch_s={args.patch_s}  '
        f'tokens={n_t}t x {n_s}s = {n_t * n_s} patch {cls_info}'
    )

    _build_kw = dict(
        arch=args.arch, K=args.K, img_size=224,
        patch_t=args.patch_t, patch_s=args.patch_s,
        dropout=args.dropout, attn_drop=args.attn_drop,
        pretrained=args.pretrained,
    )
    if args.variant == 'factorised':
        model = build_vivit_factorised_classifier(**_build_kw).to(device)
        model_tag = f'ViViT-F-{args.arch}'
    else:
        model = build_vit3d_classifier(**_build_kw).to(device)
        model_tag = f'ViT3D-{args.arch}'
    print(f'Model: {model_tag}  params={sum(p.numel() for p in model.parameters()):,}')

    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    if args.use_wandb:
        import wandb
        run_name = args.run_name or f'{args.variant}_{args.arch}_K{args.K}'
        wandb.init(project='MedForensics-baselines', name=run_name, config=args_dict)

    best_val_auc    = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n_train    = 0

        for batch in tqdm(dl_train, desc=f'Epoch {epoch:03d}', leave=False, dynamic_ncols=True):
            vol    = batch['volume'].to(device).repeat(1, 3, 1, 1, 1)  # (B,1,K,H,W)->3-ch
            labels = batch['label'].float().to(device)
            optimiser.zero_grad()
            logits = model(vol).squeeze(1)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            train_loss += loss.item() * len(labels)
            n_train    += len(labels)

        scheduler.step()
        train_loss /= max(n_train, 1)

        val_metrics, _, _, _ = evaluate(model, dl_valid, criterion, device)
        elapsed = time.time() - t0

        print(
            f'Epoch {epoch:03d}/{args.epochs}  '
            f'Train {train_loss:.4f}  '
            f"Val loss={val_metrics['loss']:.4f}  "
            f"AUC={val_metrics['auc']:.4f}  "
            f"Acc={val_metrics['accuracy']:.4f}  "
            f'({elapsed:.1f}s)'
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
                out_dir / 'best_model.pt',
            )
            print(f'  * New best val AUC: {best_val_auc:.4f} - saved')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'  Early stopping at epoch {epoch}')
                break

    print('\n=== Test Evaluation ===')
    ckpt = torch.load(out_dir / 'best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    test_metrics, t_labels, t_scores, t_mods = evaluate(model, dl_test, criterion, device)
    test_metrics['per_mod'] = per_mod_metrics(t_labels, t_scores, t_mods)

    print(
        f"  Test AUC={test_metrics['auc']:.4f}  Acc={test_metrics['accuracy']:.4f}  "
        f"F1={test_metrics['f1']:.4f}  AP={test_metrics['ap']:.4f}"
    )
    for mod, mm in test_metrics.get('per_mod', {}).items():
        print(f"    {mod:12s}  AUC={mm['auc']:.4f}  F1={mm['f1']:.4f}  AP={mm['ap']:.4f}")

    with open(out_dir / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)

    if args.use_wandb:
        import wandb
        wandb.log({f'test/{k}': v for k, v in test_metrics.items()
                   if isinstance(v, (int, float))})
        wandb.finish()

    print(f'\nDone. Outputs: {out_dir}')


if __name__ == '__main__':
    main()
