#!/usr/bin/env python3
"""
train_deepfeaturex.py
=====================

Adapted to M3DSynth binary (pristine vs tampered) task.
DeepFeatureX is modular: the number of Siamese encoders depends on the task.
For binary classification we use ONE encoder.

───────────────────────────────────────────────────────────────────────
STAGE 1 — Siamese encoder  (ContrastiveLoss)
───────────────────────────────────────────────────────────────────────
  A single ResNet-50 backbone (fc=Identity() → 2048-dim features) is
  trained with Contrastive Loss on random image pairs:

    similar pair    (label=0): both from the same class  (real+real OR fake+fake)
    dissimilar pair (label=1): one real, one fake

  The encoder learns a metric space where pristine and tampered images
  are separated by at least margin m=2.

───────────────────────────────────────────────────────────────────────
STAGE 2 — Binary classifier  (frozen encoder + head)
───────────────────────────────────────────────────────────────────────
  The Siamese encoder is frozen and used as a feature extractor.
  A lightweight binary head is trained on top:

    frozen encoder → (B, 2048) → Linear(2048, 512) → ReLU → Dropout
                   → Linear(512, 1)  [BCEWithLogitsLoss]

───────────────────────────────────────────────────────────────────────
EVALUATION
───────────────────────────────────────────────────────────────────────
  Slice-level: sigmoid(logit) >= 0.5 → fake
  Volume-level: max sigmoid(logit) over K slices
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
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, average_precision_score
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
WORK_DIR = Path(__file__).resolve().parent.parent
DFX_DIR  = WORK_DIR / 'baselines' / 'git_repo' / 'DeepFeatureX-SN'
sys.path.insert(0, str(WORK_DIR))
sys.path.insert(0, str(WORK_DIR / 'src'))
from config import DATA_DIR

# Load backbone() from import_classifiers.py directly to bypass dfx/__init__.py
# which imports dfx.wd (a path-config file not needed here).
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    'dfx_import_classifiers',
    DFX_DIR / 'src' / 'dfx' / 'import_classifiers.py',
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
dfx_backbone = _mod.backbone

from hexmil.data.patch_dataset import load_split_table
from hexmil.utils.tiff_utils import (
    get_shape_tiff_scan, load_slice_tiff_scan,
    get_percentile_tiff_scan, apply_percentile,
)

ALL_FAKES = ['pix2pix', 'cycle', 'diffusion']
_IMG_MEAN = 0.449
_IMG_STD  = 0.226


# =============================================================================
#  Device
# =============================================================================

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


# =============================================================================
#  Contrastive loss  (verbatim from dfx/losses.py)
# =============================================================================

class ContrastiveLoss(nn.Module):
    """Pairwise contrastive loss with margin m.

    label=0  →  similar pair:    minimise d (push embeddings together)
    label=1  →  dissimilar pair: push d to at least m
    """
    def __init__(self, m: float = 2.0):
        super().__init__()
        self.m = m

    def forward(self, phi_i: torch.Tensor, phi_j: torch.Tensor,
                label: torch.Tensor) -> torch.Tensor:
        d = F.pairwise_distance(phi_i, phi_j)
        loss = (0.5 * (1.0 - label.float()) * d.pow(2)
                + 0.5 * label.float() * torch.clamp(self.m - d, min=0.0).pow(2))
        return loss.mean()


# =============================================================================
#  Model
# =============================================================================

def build_encoder() -> nn.Module:
    """ResNet-50 feature extractor: fc=Identity() → (B, 2048) embeddings."""
    return dfx_backbone('resnet50', pretrained=True,
                        finetuning=False, as_feature_extractor=True)


class DeepFeatureXBinary(nn.Module):
    """Frozen Siamese encoder + trainable binary classifier head."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder          # frozen after Siamese training
        self.head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)          # (B, 2048)
        return self.head(feat)          # (B, 1)


# =============================================================================
#  Image loading helper
# =============================================================================

def _load_slice(data_dir: str, mod: str, img_id: str,
                cz: int, target_size: int) -> torch.Tensor:
    """Load one CT slice, normalise, resize → (1, H, W) float32 tensor."""
    scan_dir  = os.path.join(data_dir, mod, 'scan', img_id)
    shape     = get_shape_tiff_scan(scan_dir)
    low, high = get_percentile_tiff_scan(scan_dir, np.uint16)
    raw = load_slice_tiff_scan(scan_dir, shape, np.uint16, cz, cz + 1)[0]
    img = (apply_percentile(raw.astype(np.float32), low, high) - _IMG_MEAN) / _IMG_STD
    t   = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    if img.shape[0] != target_size or img.shape[1] != target_size:
        t = F.interpolate(t, size=target_size, mode='bilinear', align_corners=False)
    return t.squeeze(0)   # (1, H, W)


# =============================================================================
#  Datasets
# =============================================================================

class SiamesePairDataset(Dataset):
    """Random pairs for Siamese contrastive training (pristine vs tampered).

    similar pair    (label=0): (real, real)  or  (fake, fake — same mod)
    dissimilar pair (label=1): (real, fake)

    Pairs are generated randomly with 50% same / 50% different per item.
    `n_pairs` is the epoch length (re-sampled at every __getitem__).
    """

    def __init__(self, data_dir: str, real_tab, fake_tab,
                 n_pairs: int, target_size: int = 224, augment: bool = False):
        self.data_dir    = data_dir
        self.target_size = target_size
        self.augment     = augment
        self.n_pairs     = n_pairs
        self.real = real_tab.reset_index(drop=True)
        self.fake = fake_tab.reset_index(drop=True)
        # Group fake rows by modality for within-mod positive pairs
        self.fake_by_mod: dict[str, list[int]] = {}
        for i, row in self.fake.iterrows():
            self.fake_by_mod.setdefault(row['mod'], []).append(i)

    def __len__(self) -> int:
        return self.n_pairs

    def _load(self, tab, idx: int) -> torch.Tensor:
        row = tab.iloc[idx]
        t   = _load_slice(self.data_dir, str(row['mod']), str(row['img_id']),
                          int(row['coord_z']), self.target_size)
        if self.augment:
            if np.random.rand() > 0.5:
                t = torch.flip(t, [2])
            if np.random.rand() > 0.5:
                t = torch.flip(t, [1])
        return t.repeat(3, 1, 1)   # (1,H,W) → (3,H,W) for pretrained backbone

    def __getitem__(self, _idx: int) -> tuple:
        same = np.random.rand() < 0.5

        if same:
            # Positive pair: both from same class
            if np.random.rand() < 0.5 or len(self.fake) == 0:
                # real + real
                i1 = np.random.randint(len(self.real))
                i2 = np.random.randint(len(self.real))
                img1 = self._load(self.real, i1)
                img2 = self._load(self.real, i2)
            else:
                # fake + fake (same mod, to avoid confounding domain shift)
                mod  = np.random.choice(list(self.fake_by_mod.keys()))
                idxs = self.fake_by_mod[mod]
                i1   = idxs[np.random.randint(len(idxs))]
                i2   = idxs[np.random.randint(len(idxs))]
                img1 = self._load(self.fake, i1)
                img2 = self._load(self.fake, i2)
            label = torch.tensor(0, dtype=torch.float32)
        else:
            # Negative pair: real + fake
            i1   = np.random.randint(len(self.real))
            i2   = np.random.randint(len(self.fake))
            img1 = self._load(self.real, i1)
            img2 = self._load(self.fake, i2)
            label = torch.tensor(1, dtype=torch.float32)

        return img1, img2, label


class SliceDataset(Dataset):
    """Single-slice dataset for Stage 2 binary classifier training."""

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
        mod    = str(row['mod'])
        img_id = str(row['img_id'])
        cz     = int(row['coord_z'])
        t      = _load_slice(self.data_dir, mod, img_id, cz, self.target_size)
        if self.augment:
            if np.random.rand() > 0.5:
                t = torch.flip(t, [2])
            if np.random.rand() > 0.5:
                t = torch.flip(t, [1])
        return dict(
            image  = t.repeat(3, 1, 1),       # (3, H, W)
            label  = 0 if mod == 'real' else 1,
            mod    = mod,
            img_id = img_id,
        )


class VolumeDataset(Dataset):
    """K-slice window for volume-level evaluation."""

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
        mod    = str(row['mod'])
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
                z1 = Z; z0 = z1 - K
            if z0 < 0:
                z0 = 0
        z1  = z0 + K
        raw = load_slice_tiff_scan(scan_dir, shape, np.uint16, z0, z1)
        imgs = np.stack([(apply_percentile(s.astype(np.float32), low, high) - _IMG_MEAN) / _IMG_STD
                         for s in raw])
        imgs_t = torch.from_numpy(imgs.astype(np.float32)).unsqueeze(1)  # (K,1,H,W)
        if imgs.shape[1] != self.target_size or imgs.shape[2] != self.target_size:
            imgs_t = F.interpolate(imgs_t, size=self.target_size, mode='bilinear', align_corners=False)
        return dict(images=imgs_t, label=0 if mod == 'real' else 1, mod=mod, img_id=img_id)


# =============================================================================
#  Metrics
# =============================================================================

def _compute_metrics(labels, scores, mods) -> dict:
    y = np.array(labels)
    s = np.nan_to_num(np.array(scores, dtype=np.float64), nan=0.5)
    p = (s >= 0.5).astype(int)

    def _auc(y, s):
        return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float('nan')

    out = dict(auc=_auc(y, s), acc=float(accuracy_score(y, p)),
               f1=float(f1_score(y, p, zero_division=0)),
               ap=float(average_precision_score(y, s)) if len(np.unique(y)) > 1 else float('nan'))
    rm = np.array(mods) == 'real'
    per_mod = {}
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


@torch.no_grad()
def evaluate_slices(model, loader, device) -> dict:
    model.eval()
    labels, scores, mods = [], [], []
    for b in loader:
        imgs   = b['image'].to(device)
        logits = model(imgs).squeeze(1)
        labels.extend(b['label'].tolist())
        scores.extend(torch.sigmoid(logits).cpu().tolist())
        mods.extend(b['mod'])
    return _compute_metrics(labels, scores, mods)


@torch.no_grad()
def evaluate_volumes(model, loader, device) -> dict:
    model.eval()
    labels, scores, mods = [], [], []
    for b in loader:
        imgs   = b['images'].squeeze(0).to(device).repeat(1, 3, 1, 1)  # (K,3,H,W)
        logits = model(imgs).squeeze(1)
        labels.append(b['label'].item())
        scores.append(torch.sigmoid(logits).max().item())
        mods.append(b['mod'][0])
    return _compute_metrics(labels, scores, mods)


# =============================================================================
#  Stage 1 — Siamese encoder training
# =============================================================================

def train_siamese(encoder: nn.Module, real_tab, fake_tab,
                  args, device) -> nn.Module:
    """Train the Siamese encoder with ContrastiveLoss on pristine/tampered pairs."""
    print(f"\n{'─'*60}")
    print(f"  Stage 1 — Siamese encoder (ContrastiveLoss)")
    print(f"  real: {len(real_tab)}  fake: {len(fake_tab)}")
    print(f"{'─'*60}")

    n_pairs = max((len(real_tab) + len(fake_tab)) * 2, 1024)
    ds_tr   = SiamesePairDataset(args.data_dir, real_tab, fake_tab,
                                  n_pairs=n_pairs,
                                  target_size=args.target_size, augment=True)

    # Validation: smaller fixed set
    tab_all_valid = args._valid_tabs  # injected by main()
    real_vl = tab_all_valid[tab_all_valid['mod'] == 'real']
    fake_vl = tab_all_valid[tab_all_valid['mod'] != 'real']
    n_vl    = max((len(real_vl) + len(fake_vl)), 256)
    ds_vl   = SiamesePairDataset(args.data_dir, real_vl, fake_vl,
                                  n_pairs=n_vl,
                                  target_size=args.target_size, augment=False)

    kw    = dict(num_workers=args.num_workers, pin_memory=True)
    dl_tr = DataLoader(ds_tr, args.batch_size, shuffle=True,  **kw)
    dl_vl = DataLoader(ds_vl, args.batch_size, shuffle=False, **kw)

    criterion = ContrastiveLoss(m=2.0)
    optimiser = torch.optim.AdamW(encoder.parameters(),
                                   lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.s_epochs, eta_min=args.lr * 0.01)

    best_loss   = float('inf')
    best_state  = None
    patience_ctr = 0

    for epoch in range(1, args.s_epochs + 1):
        t0 = time.time()
        encoder.train()
        tloss, n_seen = 0.0, 0

        for img1, img2, lbl in tqdm(dl_tr, desc=f'  siamese ep{epoch:03d}', leave=False):
            img1, img2, lbl = img1.to(device), img2.to(device), lbl.to(device)
            code1 = encoder(img1)
            code2 = encoder(img2)
            loss  = criterion(code1, code2, lbl)
            optimiser.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimiser.step()
            tloss += loss.item() * len(lbl); n_seen += len(lbl)

        scheduler.step()

        encoder.eval()
        vloss, vn = 0.0, 0
        with torch.no_grad():
            for img1, img2, lbl in dl_vl:
                img1, img2, lbl = img1.to(device), img2.to(device), lbl.to(device)
                code1 = encoder(img1); code2 = encoder(img2)
                vloss += criterion(code1, code2, lbl).item() * len(lbl); vn += len(lbl)
        vloss /= vn if vn > 0 else 1

        print(f"  siamese  ep{epoch:03d}/{args.s_epochs}"
              f"  tr_loss={tloss/n_seen:.4f}  val_loss={vloss:.4f}  ({time.time()-t0:.1f}s)")

        if vloss < best_loss:
            best_loss = vloss; patience_ctr = 0
            best_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"  Early stopping Siamese at epoch {epoch}"); break

    if best_state is not None:
        encoder.load_state_dict(best_state)
    return encoder


# =============================================================================
#  Stage 2 — Binary classifier training (frozen encoder)
# =============================================================================

def train_classifier(model: DeepFeatureXBinary, train_tab, valid_tab,
                     args, device) -> DeepFeatureXBinary:
    """Freeze encoder, train binary head with BCEWithLogitsLoss."""
    print(f"\n{'─'*60}")
    print(f"  Stage 2 — Binary classifier (frozen encoder)")
    print(f"{'─'*60}")

    # Freeze encoder
    model.encoder.eval()
    for p in model.encoder.parameters():
        p.requires_grad = False

    ds_tr = SliceDataset(args.data_dir, train_tab, args.target_size, augment=True)
    ds_vl = SliceDataset(args.data_dir, valid_tab, args.target_size)

    lbls_arr = train_tab['mod'].apply(lambda m: 0 if m == 'real' else 1).values
    cls_w    = 1.0 / np.maximum(np.bincount(lbls_arr), 1)
    sampler  = WeightedRandomSampler(cls_w[lbls_arr], len(lbls_arr), replacement=True)

    kw    = dict(num_workers=args.num_workers, pin_memory=True)
    dl_tr = DataLoader(ds_tr, args.batch_size, sampler=sampler, **kw)
    dl_vl = DataLoader(ds_vl, args.batch_size, shuffle=False,   **kw)

    criterion = nn.BCEWithLogitsLoss()
    trainable = list(model.head.parameters())
    optimiser = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.c_epochs, eta_min=args.lr * 0.01)

    best_auc     = 0.0
    best_state   = None
    patience_ctr = 0

    for epoch in range(1, args.c_epochs + 1):
        t0 = time.time()
        model.train()
        model.encoder.eval()   # keep encoder frozen in eval mode
        tloss, n_seen = 0.0, 0

        for b in tqdm(dl_tr, desc=f'  classifier ep{epoch:03d}', leave=False):
            imgs   = b['image'].to(device)
            labels = b['label'].float().to(device)
            optimiser.zero_grad()
            logits = model(imgs).squeeze(1)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimiser.step()
            tloss += loss.item() * len(labels); n_seen += len(labels)

        scheduler.step()
        val_m = evaluate_slices(model, dl_vl, device)
        auc   = val_m.get('auc', float('nan'))

        print(f"  classifier ep{epoch:03d}/{args.c_epochs}"
              f"  loss={tloss/n_seen:.4f}  val_AUC={auc:.4f}  ({time.time()-t0:.1f}s)")

        if not np.isnan(auc) and auc > best_auc:
            best_auc = auc; patience_ctr = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"  Early stopping classifier at epoch {epoch}"); break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# =============================================================================
#  Args
# =============================================================================

def get_args():
    p = argparse.ArgumentParser(
        description='DeepFeatureX — Siamese encoder + binary classifier')
    p.add_argument('--data_dir',     type=str,   default=DATA_DIR)
    p.add_argument('--out_dir',      type=str,   default=None)
    p.add_argument('--K',            type=int,   default=16)
    p.add_argument('--target_size',  type=int,   default=224)
    p.add_argument('--s_epochs',     type=int,   default=30,
                   help='Epochs for Siamese encoder (Stage 1)')
    p.add_argument('--c_epochs',     type=int,   default=30,
                   help='Epochs for binary classifier (Stage 2)')
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
    p.add_argument('--eval_only',    action='store_true', default=False,
                   help='Skip training, load saved model and evaluate')
    return p.parse_args()


# =============================================================================
#  Main
# =============================================================================

def main():
    args = get_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = select_device(args.gpu_id)

    train_mods = args.train_mods or ALL_FAKES
    fake_mods  = [m for m in train_mods if m != 'real']
    mods_tag   = '+'.join(sorted(fake_mods)) if set(fake_mods) != set(ALL_FAKES) else 'all'

    if args.out_dir is None:
        args.out_dir = str(
            WORK_DIR / 'baselines' / 'runs'
            / f'deepfeaturex_K{args.K}' / f'trained_on_{mods_tag}'
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    args_dict = vars(args)
    with open(out_dir / 'args.json', 'w') as f:
        json.dump({k: v for k, v in args_dict.items() if not k.startswith('_')},
                  f, indent=2)

    # ── Data ──────────────────────────────────────────────────────────────────
    tr_mods = ['real'] + fake_mods
    ts_mods = ['real'] + ALL_FAKES

    tab_train = load_split_table(args.data_dir, 'train', tr_mods)
    tab_valid = load_split_table(args.data_dir, 'valid', tr_mods)
    tab_test  = load_split_table(args.data_dir, 'test',  ts_mods)

    print(f"  Train: {len(tab_train)}  Valid: {len(tab_valid)}  Test: {len(tab_test)}")
    print(f"  Train mods: {tr_mods}  |  Test mods: {ts_mods}")

    kw = dict(num_workers=args.num_workers, pin_memory=True)

    if not args.eval_only:
        real_tr = tab_train[tab_train['mod'] == 'real']
        fake_tr = tab_train[tab_train['mod'] != 'real']

        # Inject valid tables for Siamese validation (used inside train_siamese)
        args._valid_tabs = tab_valid

        # ── Stage 1: Siamese encoder ───────────────────────────────────────────
        encoder = build_encoder().to(device)
        encoder = train_siamese(encoder, real_tr, fake_tr, args, device)
        torch.save(encoder.state_dict(), out_dir / 'encoder.pt')
        print(f"\n  Encoder saved → {out_dir / 'encoder.pt'}")

        # ── Stage 2: Binary classifier (frozen encoder) ────────────────────────
        model = DeepFeatureXBinary(encoder).to(device)
        model = train_classifier(model, tab_train, tab_valid, args, device)
        torch.save(model.state_dict(), out_dir / 'best_model.pt')
        print(f"  Complete model saved → {out_dir / 'best_model.pt'}")

    else:
        encoder = build_encoder().to(device)
        model   = DeepFeatureXBinary(encoder).to(device)
        ckpt    = out_dir / 'best_model.pt'
        if not ckpt.exists():
            sys.exit(f'[ERROR] No best_model.pt found in {out_dir}')
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

    model.eval()

    # ── Evaluation ────────────────────────────────────────────────────────────
    ds_sl = SliceDataset(args.data_dir, tab_test, args.target_size)
    dl_sl = DataLoader(ds_sl, args.batch_size, shuffle=False, **kw)

    print("\n=== Slice-Level Test ===")
    sm = evaluate_slices(model, dl_sl, device)
    print(f"  AUC={sm.get('auc', float('nan')):.4f}  Acc={sm.get('acc', float('nan')):.4f}")

    print(f"\n=== Volume-Level Test (K={args.K}) ===")
    ds_vol = VolumeDataset(args.data_dir, tab_test, args.K, args.target_size)
    dl_vol = DataLoader(ds_vol, batch_size=1, shuffle=False, **kw)
    vm = evaluate_volumes(model, dl_vol, device)
    print(f"  AUC={vm.get('auc', float('nan')):.4f}  Acc={vm.get('acc', float('nan')):.4f}")
    for mod, mm in vm.get('per_mod', {}).items():
        print(f"    {mod:12s}  AUC={mm['auc']:.4f}  F1={mm['f1']:.4f}  AP={mm['ap']:.4f}")

    results = dict(slice=sm, volume=vm, args={k: v for k, v in args_dict.items()
                                               if not k.startswith('_')})
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    if args.use_wandb:
        import wandb
        wandb.log({f'test/volume/{k}': v for k, v in vm.items() if isinstance(v, (int, float))})
        wandb.finish()

    print(f"\nDone. Results → {out_dir / 'results.json'}")


if __name__ == '__main__':
    main()
