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
import torch.utils.model_zoo as model_zoo
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, average_precision_score
from tqdm import tqdm

WORK_DIR   = Path(__file__).resolve().parent.parent
MVSS_DIR   = WORK_DIR / 'baselines' / 'git_repo' / 'MVSS-Net'
sys.path.insert(0, str(WORK_DIR))
sys.path.insert(0, str(WORK_DIR / 'src'))
sys.path.insert(0, str(MVSS_DIR))   # needed so mvssnet.py can import resfcn
from config import DATA_DIR

# Import MVSSNet and get_mvss from the MVSS-Net repo
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location('mvssnet', MVSS_DIR / 'models' / 'mvssnet.py')
_mod  = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
MVSSNet  = _mod.MVSSNet
get_mvss = _mod.get_mvss

from hexmil.data.patch_dataset import load_split_table, MOD_LABEL
from hexmil.utils.tiff_utils import (
    get_shape_tiff_scan, load_slice_tiff_scan,
    get_percentile_tiff_scan, apply_percentile,
)

ALL_FAKES = ['pix2pix', 'cycle', 'diffusion']

_RESNET50_URL = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'

# ImageNet normalization adapted to 1-channel grayscale (average of RGB stats).
# The original MVSS-Net applies this before feeding images to the ResNet-50 backbone.
_IMG_MEAN = 0.449   # (0.485 + 0.456 + 0.406) / 3
_IMG_STD  = 0.226   # (0.229 + 0.224 + 0.225) / 3

#  Device selection — picks GPU with most free memory

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

#  Model — Full MVSSNet with original 3-ch backbone

def build_mvssnet(pretrained: bool = True) -> MVSSNet:
    model = MVSSNet(nclass=1, sobel=True, constrain=False, n_input=3)
    n = sum(p.numel() for p in model.parameters())
    print(f"MVSSNet  params={n:,}")
    return model

def build_mvssnet_no_sobel(pretrained: bool = True) -> MVSSNet:
    model = MVSSNet(nclass=1, sobel=False, constrain=False, n_input=3)
    n = sum(p.numel() for p in model.parameters())
    print(f"MVSSNet (no-sobel)  params={n:,}")
    return model

#  Loss — focal + dice

def focal_loss(logits: torch.Tensor, targets: torch.Tensor,
               gamma: float = 2.0, pos_weight: float = 1.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(
        logits, targets,
        pos_weight=torch.tensor(pos_weight, device=logits.device),
        reduction='none',
    )
    return ((1 - torch.exp(-bce)) ** gamma * bce).mean()

def dice_loss(logits: torch.Tensor, targets: torch.Tensor,
              eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = (prob * targets).sum(dim=(-1, -2))
    union = prob.sum(dim=(-1, -2)) + targets.sum(dim=(-1, -2))
    return (1 - (2 * inter + eps) / (union + eps)).mean()

def combined_loss(logits, targets, gamma=2.0, pos_weight=1.0, dice_w=0.5):
    return focal_loss(logits, targets, gamma, pos_weight) + dice_w * dice_loss(logits, targets)


class SliceDataset(Dataset):
    def __init__(self, data_dir: str, tab, target_size: int = 224,
                 augment: bool = False):
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

        if mod == 'real':
            mask = np.zeros_like(img, dtype=np.float32)
        else:
            ldir = os.path.join(self.data_dir, mod, 'label', img_id)
            mask = load_slice_tiff_scan(ldir, shape, np.bool_, cz, cz + 1)[0].astype(np.float32)

        img_t  = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)

        if img.shape[0] != self.target_size or img.shape[1] != self.target_size:
            img_t  = F.interpolate(img_t,  size=self.target_size, mode='bilinear', align_corners=False)
            mask_t = F.interpolate(mask_t, size=self.target_size, mode='nearest')

        img_t  = img_t.squeeze(0).float()
        mask_t = mask_t.squeeze(0).float()

        if self.augment:
            if torch.rand(1).item() > 0.5:
                img_t  = torch.flip(img_t,  [2]); mask_t = torch.flip(mask_t,  [2])
            if torch.rand(1).item() > 0.5:
                img_t  = torch.flip(img_t,  [1]); mask_t = torch.flip(mask_t,  [1])

        return dict(image=img_t, mask=mask_t,
                    label=0 if mod == 'real' else 1, mod=mod, img_id=img_id)

class VolumeDataset(Dataset):
    def __init__(self, data_dir: str, tab, K: int = 16, target_size: int = 224):
        self.data_dir    = data_dir
        self.K           = K
        self.target_size = target_size
        self.volumes = (tab[['mod', 'img_id', 'coord_z']]
                        .drop_duplicates(subset=['mod', 'img_id'])
                        .reset_index(drop=True))

    def __len__(self) -> int: return len(self.volumes)

    def __getitem__(self, idx: int) -> dict:
        row    = self.volumes.iloc[idx]
        mod    = row['mod']
        img_id = str(row['img_id'])
        cz     = int(row['coord_z'])

        scan_dir  = os.path.join(self.data_dir, mod, 'scan', img_id)
        shape     = get_shape_tiff_scan(scan_dir)
        Z         = shape[0]
        low, high = get_percentile_tiff_scan(scan_dir, np.uint16)

        z0  = max(0, min(cz - self.K // 2, Z - self.K))
        z1  = min(z0 + self.K, Z)
        raw = load_slice_tiff_scan(scan_dir, shape, np.uint16, z0, z1)
        imgs = np.stack([(apply_percentile(s.astype(np.float32), low, high) - _IMG_MEAN) / _IMG_STD for s in raw])

        imgs_t = torch.from_numpy(imgs.astype(np.float32)).unsqueeze(1)
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
            _, logits = model(imgs)   # x0 is the main segmentation output
            prob = torch.sigmoid(logits)
            labels.extend(b['label'].tolist())
            scores.extend(prob.flatten(1).max(1).values.cpu().tolist())
            mods.extend(b['mod'])
    return _compute_metrics(labels, scores, mods), labels, scores, mods

@torch.no_grad()
def evaluate_volumes(model, loader, device) -> dict:
    model.eval()
    labels, scores, mods = [], [], []
    for b in loader:
        imgs = b['images'].squeeze(0).to(device).repeat(1, 3, 1, 1)   # 1-ch → 3-ch
        _, logits = model(imgs)   # x0 is the main segmentation output
        prob = torch.sigmoid(logits)
        labels.append(b['label'].item())
        scores.append(prob.flatten().max().item())
        mods.append(b['mod'][0])
    return _compute_metrics(labels, scores, mods)

#  Localization metrics (pixel-level, fake slices only)

def evaluate_localization(model, loader, device) -> dict:
    model.eval()
    _T = (0.3, 0.5, 0.7)
    pg_all, eom_all, pauc_all = [], [], []
    iou_all: dict[float, list] = {t: [] for t in _T}
    per_mod: dict[str, dict] = {
        fm: {'pg': [], 'eom': [], 'pauc': [], **{f'iou_{t}': [] for t in _T}}
        for fm in ALL_FAKES
    }

    with torch.no_grad():
        for b in loader:
            imgs  = b['image'].to(device).repeat(1, 3, 1, 1)   # 1-ch → 3-ch
            masks = b['mask']
            mods  = b['mod']
            lbls  = b['label'].tolist()
            _, logits = model(imgs)   # MVSSNet returns (res1, x0); use x0
            pmaps = torch.sigmoid(logits).cpu().numpy()

            for i, lbl in enumerate(lbls):
                if lbl == 0:
                    continue
                pm   = pmaps[i, 0]
                mask = masks[i, 0].numpy()
                if mask.sum() == 0:
                    continue
                mod  = mods[i]
                ay, ax = np.unravel_index(pm.argmax(), pm.shape)
                pg  = int(mask[ay, ax] > 0.5)
                eom = float((pm * mask).sum() / (pm.sum() + 1e-8))
                y_flat = mask.flatten().astype(int)
                pauc = float(roc_auc_score(y_flat, pm.flatten())) \
                    if len(np.unique(y_flat)) > 1 else float('nan')
                mask_bin = mask > 0.5
                ious = {}
                for t in _T:
                    pred_bin = pm >= t
                    inter    = float((pred_bin & mask_bin).sum())
                    union    = float((pred_bin | mask_bin).sum())
                    ious[t]  = inter / (union + 1e-8)

                pg_all.append(pg); eom_all.append(eom)
                if not np.isnan(pauc):
                    pauc_all.append(pauc)
                for t in _T:
                    iou_all[t].append(ious[t])
                if mod in per_mod:
                    per_mod[mod]['pg'].append(pg)
                    per_mod[mod]['eom'].append(eom)
                    if not np.isnan(pauc):
                        per_mod[mod]['pauc'].append(pauc)
                    for t in _T:
                        per_mod[mod][f'iou_{t}'].append(ious[t])

    def _m(v): return float(np.mean(v)) if v else float('nan')
    return {
        'pointing_game':  _m(pg_all),
        'energy_on_mask': _m(eom_all),
        'pixel_auc':      _m(pauc_all),
        'iou_0.3':        _m(iou_all[0.3]),
        'iou_0.5':        _m(iou_all[0.5]),
        'iou_0.7':        _m(iou_all[0.7]),
        'n_fake_slices':  len(pg_all),
        'per_mod': {
            mod: {
                'pointing_game': _m(v['pg']),
                'energy_on_mask': _m(v['eom']),
                'pixel_auc':      _m(v['pauc']),
                'iou_0.3':        _m(v['iou_0.3']),
                'iou_0.5':        _m(v['iou_0.5']),
                'iou_0.7':        _m(v['iou_0.7']),
            }
            for mod, v in per_mod.items() if v['pg']
        },
    }


def get_args():
    p = argparse.ArgumentParser(description='MVSS-Net (full, sobel, 1-ch CT) baseline')
    p.add_argument('--data_dir',     type=str,   default=DATA_DIR)
    p.add_argument('--out_dir',      type=str,   default=None)
    p.add_argument('--K',            type=int,   default=16)
    p.add_argument('--target_size',  type=int,   default=224)
    p.add_argument('--pretrained',   action='store_true', default=True)
    p.add_argument('--focal_gamma',  type=float, default=2.0)
    p.add_argument('--pos_weight',   type=float, default=5.0)
    p.add_argument('--dice_weight',  type=float, default=0.5,
                   help='Weight for dice loss component')
    p.add_argument('--epochs',       type=int,   default=50)
    p.add_argument('--batch_size',   type=int,   default=8)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--patience',     type=int,   default=10)
    p.add_argument('--num_workers',  type=int,   default=4)
    p.add_argument('--seed',         type=int,   default=42)
    p.add_argument('--gpu_id',       type=int,   default=None,
                   help='Specific GPU id; omit for auto-selection')
    p.add_argument('--train_mods',   nargs='+',  default=None,
                   help='Generative families to train on (default: all three)')
    p.add_argument('--amp',          action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--use_wandb',    action='store_true', default=False)
    p.add_argument('--run_name',     type=str,   default=None)
    p.add_argument('--eval_only',    action='store_true', default=False,
                   help='Skip training; load best_model.pt and run test only')
    return p.parse_args()


def main():
    args = get_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = select_device(args.gpu_id)

    train_mods = args.train_mods or ALL_FAKES
    mods_tag   = '+'.join(sorted(train_mods))

    if args.out_dir is None:
        args.out_dir = str(
            WORK_DIR / 'baselines' / 'runs'
            / f'mvssnet_full_K{args.K}' / f'trained_on_{mods_tag}'
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    args_dict = vars(args)
    with open(out_dir / 'args.json', 'w') as f:
        json.dump(args_dict, f, indent=2)

    tr_mods = ['real'] + train_mods
    ts_mods = ['real'] + ALL_FAKES

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

    model = build_mvssnet(pretrained=args.pretrained).to(device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    if args.use_wandb:
        import wandb
        wandb.init(project='MedForensics-baselines',
                   name=args.run_name or f"mvssnet_K{args.K}_{mods_tag}",
                   config=args_dict)

    if not args.eval_only:
        best_auc, patience_counter = 0.0, 0

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            model.train()
            tloss, n_tr = 0.0, 0

            for b in tqdm(dl_train, desc=f'Epoch {epoch:03d}', leave=False):
                imgs  = b['image'].to(device).repeat(1, 3, 1, 1)   # 1-ch → 3-ch
                masks = b['mask'].to(device)
                optimiser.zero_grad()
                with torch.amp.autocast('cuda', enabled=args.amp):
                    res1, x0 = model(imgs)   # MVSSNet: edge branch (H/4) + main head (H)
                    # Upsample res1 from H/4 to H for multi-task loss
                    res1_up = F.interpolate(res1.float(), size=imgs.shape[2:],
                                            mode='bilinear', align_corners=False)
                    loss = (combined_loss(x0.float(),    masks, args.focal_gamma,
                                          args.pos_weight, args.dice_weight)
                          + combined_loss(res1_up, masks, args.focal_gamma,
                                          args.pos_weight, args.dice_weight))
                if not loss.isfinite():
                    optimiser.zero_grad(set_to_none=True)
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimiser)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimiser); scaler.update()
                tloss += loss.item() * len(masks); n_tr += len(masks)

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
        print(f"    {mod:12s}  AUC={mm['auc']:.4f}  F1={mm['f1']:.4f}")

    print("\n=== Localization Metrics (fake slices only) ===")
    loc_m = evaluate_localization(model, dl_test, device)
    print(f"  Pointing Game={loc_m['pointing_game']:.4f}  "
          f"EoM={loc_m['energy_on_mask']:.4f}  "
          f"pAUC={loc_m['pixel_auc']:.4f}  "
          f"IoU@0.3={loc_m['iou_0.3']:.4f}  "
          f"IoU@0.5={loc_m['iou_0.5']:.4f}  "
          f"IoU@0.7={loc_m['iou_0.7']:.4f}  "
          f"(n={loc_m['n_fake_slices']})")
    for mod, mm in loc_m.get('per_mod', {}).items():
        print(f"    {mod:12s}  PG={mm['pointing_game']:.4f}  "
              f"EoM={mm['energy_on_mask']:.4f}  pAUC={mm['pixel_auc']:.4f}  "
              f"IoU@0.3={mm['iou_0.3']:.4f}  IoU@0.5={mm['iou_0.5']:.4f}  "
              f"IoU@0.7={mm['iou_0.7']:.4f}")

    eval_dir = out_dir / 'evaluation'
    eval_dir.mkdir(exist_ok=True)
    with open(eval_dir / 'metrics.json', 'w') as f:
        json.dump(dict(slice=sm, volume=vm, localization=loc_m, args=args_dict), f, indent=2)

    if args.use_wandb:
        import wandb
        wandb.log({f'test/volume/{k}': v for k, v in vm.items() if isinstance(v, (int, float))})
        wandb.finish()

    print(f"\nDone. Results: {eval_dir / 'metrics.json'}")

if __name__ == '__main__':
    main()
