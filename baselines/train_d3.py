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
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, average_precision_score
from tqdm import tqdm

WORK_DIR = Path(__file__).resolve().parent.parent
D3_DIR   = WORK_DIR / 'baselines' / 'git_repo' / 'D3'
sys.path.insert(0, str(WORK_DIR))
sys.path.insert(0, str(WORK_DIR / 'src'))
sys.path.insert(0, str(D3_DIR))          # enables `from networks.resnet_lpf import ...`
from config import DATA_DIR

from networks.resnet_lpf import resnet50 as d3_resnet50

from hexmil.data.patch_dataset import load_split_table
from hexmil.utils.tiff_utils import (
    get_shape_tiff_scan, load_slice_tiff_scan,
    get_percentile_tiff_scan, apply_percentile,
)

ALL_FAKES = ['pix2pix', 'cycle', 'diffusion']

_IMG_MEAN = 0.449   # (0.485 + 0.456 + 0.406) / 3
_IMG_STD  = 0.226   # (0.229 + 0.224 + 0.225) / 3


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
        print(f"Auto-selected GPU {best} ({free[best]} MiB free)")
        return torch.device(f'cuda:{best}')
    except Exception:
        return torch.device('cuda')


class SliceDataset(Dataset):
    def __init__(self, data_dir: str, tab, target_size: int = 224, augment: bool = False):
        self.data_dir    = data_dir
        self.tab         = tab.reset_index(drop=True)
        self.target_size = target_size
        self.augment     = augment

    def __len__(self) -> int:
        return len(self.tab)

    def __getitem__(self, idx: int) -> dict:
        row    = self.tab.iloc[idx]
        mod    = row['mod']
        img_id = str(row['img_id'])
        cz     = int(row['coord_z'])

        scan_dir  = os.path.join(self.data_dir, mod, 'scan', img_id)
        shape     = get_shape_tiff_scan(scan_dir)
        low, high = get_percentile_tiff_scan(scan_dir, np.uint16)

        raw = load_slice_tiff_scan(scan_dir, shape, np.uint16, cz, cz + 1)[0]
        img = (apply_percentile(raw.astype(np.float32), low, high) - _IMG_MEAN) / _IMG_STD

        img_t = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        if img.shape[0] != self.target_size or img.shape[1] != self.target_size:
            img_t = F.interpolate(img_t, size=self.target_size, mode='bilinear', align_corners=False)
        img_t = img_t.squeeze(0).float()   # (1, H, W)

        if self.augment:
            if torch.rand(1).item() > 0.5:
                img_t = torch.flip(img_t, [2])
            if torch.rand(1).item() > 0.5:
                img_t = torch.flip(img_t, [1])

        return dict(image=img_t, label=0 if mod == 'real' else 1, mod=mod, img_id=img_id)


class VolumeDataset(Dataset):
    def __init__(self, data_dir: str, tab, K: int = 16, target_size: int = 224):
        self.data_dir    = data_dir
        self.K           = K
        self.target_size = target_size
        self.volumes = (tab[['mod', 'img_id', 'coord_z']]
                        .drop_duplicates(subset=['mod', 'img_id'])
                        .reset_index(drop=True))

    def __len__(self) -> int:
        return len(self.volumes)

    def __getitem__(self, idx: int) -> dict:
        row    = self.volumes.iloc[idx]
        mod    = row['mod']
        img_id = str(row['img_id'])
        cz     = int(row['coord_z'])

        scan_dir  = os.path.join(self.data_dir, mod, 'scan', img_id)
        shape     = get_shape_tiff_scan(scan_dir)
        Z         = shape[0]
        low, high = get_percentile_tiff_scan(scan_dir, np.uint16)

        K = self.K
        if mod == 'real':
            z0 = int(np.random.randint(0, max(1, Z - K + 1)))
        else:
            rand_off = int(np.random.randint(0, K))
            z0 = cz - rand_off
            z1 = z0 + K
            if z1 > Z:
                z1 = Z
                z0 = z1 - K
            if z0 < 0:
                z0 = 0
        z1 = z0 + K
        raw = load_slice_tiff_scan(scan_dir, shape, np.uint16, z0, z1)
        imgs = np.stack([(apply_percentile(s.astype(np.float32), low, high) - _IMG_MEAN) / _IMG_STD
                         for s in raw])

        imgs_t = torch.from_numpy(imgs.astype(np.float32)).unsqueeze(1)  # (K, 1, H, W)
        if imgs.shape[1] != self.target_size or imgs.shape[2] != self.target_size:
            imgs_t = F.interpolate(imgs_t, size=self.target_size, mode='bilinear', align_corners=False)
        return dict(images=imgs_t, label=0 if mod == 'real' else 1, mod=mod, img_id=img_id)


def _compute_metrics(labels, scores, mods) -> dict:
    y = np.array(labels)
    s = np.nan_to_num(np.array(scores, dtype=np.float64), nan=0.5, posinf=1.0, neginf=0.0)
    p = (s >= 0.5).astype(int)

    def _auc(y, s):
        return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float('nan')

    out = dict(auc=_auc(y, s), acc=float(accuracy_score(y, p)),
               f1=float(f1_score(y, p, zero_division=0)),
               ap=float(average_precision_score(y, s)) if len(np.unique(y)) > 1 else float('nan'))
    per_mod = {}
    rm = np.array(mods) == 'real'
    for fm in ALL_FAKES:
        sel = rm | (np.array(mods) == fm)
        if sel.sum() < 2:
            continue
        per_mod[fm] = dict(
            auc=_auc(y[sel], s[sel]),
            acc=float(accuracy_score(y[sel], p[sel])),
            f1=float(f1_score(y[sel], p[sel], zero_division=0)),
            ap=float(average_precision_score(y[sel], s[sel]))
               if len(np.unique(y[sel])) > 1 else float('nan'),
        )
    out['per_mod'] = per_mod
    return out


def evaluate_slices(model, loader, device) -> tuple:
    model.eval()
    labels, scores, mods = [], [], []
    with torch.no_grad():
        for b in loader:
            imgs = b['image'].to(device).repeat(1, 3, 1, 1)   # 1-ch → 3-ch
            logits = model(imgs).squeeze(1)
            labels.extend(b['label'].tolist())
            scores.extend(torch.sigmoid(logits).cpu().tolist())
            mods.extend(b['mod'])
    return _compute_metrics(labels, scores, mods), labels, scores, mods


@torch.no_grad()
def evaluate_volumes(model, loader, device) -> dict:
    model.eval()
    labels, scores, mods = [], [], []
    for b in loader:
        imgs = b['images'].squeeze(0).to(device).repeat(1, 3, 1, 1)   # (K,1,H,W) → (K,3,H,W)
        logits = model(imgs).squeeze(1)
        labels.append(b['label'].item())
        scores.append(torch.sigmoid(logits).max().item())
        mods.append(b['mod'][0])
    return _compute_metrics(labels, scores, mods)


def get_args():
    p = argparse.ArgumentParser(description='D3 LPF-ResNet50 baseline — 1-ch CT adaptation')
    p.add_argument('--data_dir',     type=str,   default=DATA_DIR)
    p.add_argument('--out_dir',      type=str,   default=None)
    p.add_argument('--K',            type=int,   default=16)
    p.add_argument('--filter_size',  type=int,   default=4,
                   help='LPF filter size for BlurPool (D3 default=4)')
    p.add_argument('--target_size',  type=int,   default=224)
    p.add_argument('--epochs',       type=int,   default=50)
    p.add_argument('--batch_size',   type=int,   default=16)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--patience',     type=int,   default=10)
    p.add_argument('--num_workers',  type=int,   default=4)
    p.add_argument('--seed',         type=int,   default=42)
    p.add_argument('--gpu_id',       type=int,   default=None)
    p.add_argument('--train_mods',   nargs='+',  default=None)
    p.add_argument('--use_wandb',    action='store_true', default=False)
    p.add_argument('--run_name',     type=str,   default=None)
    p.add_argument('--eval_only',    action='store_true', default=False)
    return p.parse_args()


def main():
    args = get_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = select_device(args.gpu_id)

    train_mods = args.train_mods or ALL_FAKES
    mods_tag   = '+'.join(sorted(train_mods))

    if args.out_dir is None:
        args.out_dir = str(WORK_DIR / 'baselines' / 'runs' / f'd3_K{args.K}' / f'trained_on_{mods_tag}')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    args_dict = vars(args)
    with open(out_dir / 'args.json', 'w') as f:
        json.dump(args_dict, f, indent=2)

    fake_mods = [m for m in train_mods if m != 'real']
    tr_mods   = ['real'] + fake_mods
    ts_mods   = ['real'] + ALL_FAKES

    tab_train = load_split_table(args.data_dir, 'train', tr_mods)
    tab_valid = load_split_table(args.data_dir, 'valid', tr_mods)
    tab_test  = load_split_table(args.data_dir, 'test',  ts_mods)

    ds_train = SliceDataset(args.data_dir, tab_train, args.target_size, augment=True)
    ds_valid = SliceDataset(args.data_dir, tab_valid, args.target_size)
    ds_test  = SliceDataset(args.data_dir, tab_test,  args.target_size)

    lbls_arr = tab_train['mod'].apply(lambda m: 0 if m == 'real' else 1).values
    cls_w    = 1.0 / np.maximum(np.bincount(lbls_arr), 1)
    sampler  = WeightedRandomSampler(cls_w[lbls_arr], len(lbls_arr), replacement=True)

    kw = dict(num_workers=args.num_workers, pin_memory=True)
    dl_train = DataLoader(ds_train, args.batch_size, sampler=sampler, **kw)
    dl_valid = DataLoader(ds_valid, args.batch_size, shuffle=False, **kw)
    dl_test  = DataLoader(ds_test,  args.batch_size, shuffle=False, **kw)

    # D3 ResNet-50 with LPF anti-aliasing; replace fc for binary classification
    model = d3_resnet50(pretrained=False, filter_size=args.filter_size, pool_only=True,
                        num_classes=1)
    model = model.to(device)
    print(f"D3-ResNet50 (filter_size={args.filter_size})  "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs, eta_min=args.lr * 0.01)

    if args.use_wandb:
        import wandb
        wandb.init(project='MedForensics-baselines',
                   name=args.run_name or f"d3_K{args.K}_{mods_tag}", config=args_dict)

    if not args.eval_only:
        best_auc, patience_counter = 0.0, 0

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            model.train()
            tloss, n_tr = 0.0, 0

            for b in tqdm(dl_train, desc=f'Epoch {epoch:03d}', leave=False):
                imgs   = b['image'].to(device).repeat(1, 3, 1, 1)   # 1-ch → 3-ch
                labels = b['label'].float().to(device)
                optimiser.zero_grad()
                logits = model(imgs).squeeze(1)
                loss   = criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                tloss += loss.item() * len(labels); n_tr += len(labels)

            scheduler.step()
            val_m, *_ = evaluate_slices(model, dl_valid, device)
            auc = val_m.get('auc', float('nan'))
            print(f"Epoch {epoch:03d}/{args.epochs}  loss={tloss/n_tr:.4f}  val_AUC={auc:.4f}"
                  f"  ({time.time()-t0:.1f}s)")

            if args.use_wandb:
                import wandb; wandb.log({'train/loss': tloss/n_tr, 'val/auc': auc})

            if not np.isnan(auc) and auc > best_auc:
                best_auc = auc; patience_counter = 0
                torch.save(dict(epoch=epoch, model_state_dict=model.state_dict(),
                                best_auc=best_auc, args=args_dict),
                           out_dir / 'best_model.pt')
                print(f"  * New best AUC={best_auc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch}"); break

    ckpt = torch.load(out_dir / 'best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    print("\n=== Slice-Level Test ===")
    sm, *_ = evaluate_slices(model, dl_test, device)
    print(f"  AUC={sm.get('auc', float('nan')):.4f}  Acc={sm.get('acc', float('nan')):.4f}")

    print(f"\n=== Volume-Level Test (K={args.K}) ===")
    ds_vol = VolumeDataset(args.data_dir, tab_test, args.K, args.target_size)
    dl_vol = DataLoader(ds_vol, batch_size=1, shuffle=False, **kw)
    vm = evaluate_volumes(model, dl_vol, device)
    print(f"  AUC={vm.get('auc', float('nan')):.4f}  Acc={vm.get('acc', float('nan')):.4f}")
    for mod, mm in vm.get('per_mod', {}).items():
        print(f"    {mod:12s}  AUC={mm['auc']:.4f}  F1={mm['f1']:.4f}  AP={mm['ap']:.4f}")

    results = dict(slice=sm, volume=vm, args=args_dict)
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    if args.use_wandb:
        import wandb
        wandb.log({f'test/volume/{k}': v for k, v in vm.items() if isinstance(v, (int, float))})
        wandb.finish()

    print(f"\nDone. Results: {out_dir / 'results.json'}")

if __name__ == '__main__':
    main()
