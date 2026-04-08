#!/usr/bin/env python3
"""
train_pool_mil.py
=================
Ablation baseline: same patch-based MIL pipeline as HEXMIL but with
**simple mean/max pooling** instead of Gated Attention at both slice
and volume aggregation levels.

Proves: gated attention adds value over naive aggregation.

Two-stage training (mirrors HEXMIL):
  Stage 1 (slice-level): patches → encoder → projector → pool → head (BCE)
  Stage 2 (volume-level): frozen Stage 1 encoder → pos_enc → pool → head (BCE)

Usage
-----
    # Stage 1 (slice)
    python baselines/train_pool_mil.py \
        --data_dir /mnt/.../M3DSynth \
        --stage slice --pool_mode mean \
        --epochs 100 --batch_size 8 --lr 1e-4

    # Stage 2 (volume)
    python baselines/train_pool_mil.py \
        --data_dir /mnt/.../M3DSynth \
        --stage volume --pool_mode mean \
        --slice_run_dir baselines/runs/pool_mil_mean_slice/trained_on_... \
        --K 16 --epochs 100 --batch_size 4 --lr 1e-4
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
from hexmil.data.slice_dataset import SliceDataset, build_patch_grid
from hexmil.utils.tiff_utils import (
    get_shape_tiff_scan,
    load_slice_tiff_scan,
    get_percentile_tiff_scan,
    apply_percentile,
)
from hexmil.models.cnn_patch_classifier import CNNPatchClassifier

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
#  Pool MIL Models
# =============================================================================

class PoolMILSliceClassifier(nn.Module):
    """
    Same architecture as ABMILSliceClassifier but replaces GatedAttention
    with simple mean or max pooling over patch features.
    """

    def __init__(
        self,
        encoder,
        proj_dim: int = 512,
        pool_mode: str = 'mean',
        dropout: float = 0.25,
    ):
        super().__init__()
        self.encoder   = encoder
        self.pool_mode = pool_mode

        raw_feat_dim: int = encoder.feat_dim

        if proj_dim is not None and proj_dim != raw_feat_dim:
            self.projector = nn.Sequential(
                nn.Linear(raw_feat_dim, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self._feat_dim = proj_dim
        else:
            self.projector = nn.Identity()
            self._feat_dim = raw_feat_dim

        self.head = nn.Sequential(
            nn.Linear(self._feat_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    @property
    def feat_dim(self) -> int:
        return self._feat_dim

    def encode_patches(self, patches: torch.Tensor) -> torch.Tensor:
        B, N, C, P_h, P_w = patches.shape
        flat = patches.view(B * N, C, P_h, P_w)
        feat_map = self.encoder.backbone(flat)[-1]
        attn_map = self.encoder.spatial_attn(feat_map)
        v = (feat_map * attn_map).mean(dim=[2, 3])
        return v.view(B, N, -1)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        features = self.encode_patches(patches)
        B, N, _ = features.shape
        h = self.projector(features.view(B * N, -1)).view(B, N, self._feat_dim)

        if self.pool_mode == 'max':
            z = h.max(dim=1).values
        else:
            z = h.mean(dim=1)

        logits = self.head(z)
        return logits

def build_pool_mil_slice(
    backbone='resnet50', pretrained=True, proj_dim=512,
    pool_mode='mean', dropout=0.25,
) -> PoolMILSliceClassifier:
    encoder = CNNPatchClassifier(
        backbone_name=backbone, pretrained=pretrained, freeze_ratio=0.0, in_chans=3,
    )
    return PoolMILSliceClassifier(
        encoder=encoder, proj_dim=proj_dim,
        pool_mode=pool_mode, dropout=dropout,
    )

# ── Volume-level pool aggregator ─────────────────────────────────────────

class SinPosEncoding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        import math
        div = torch.exp(
            -math.log(10000.0) *
            torch.arange(0, d_model, 2, dtype=torch.float32) / d_model
        )
        self.register_buffer("div", div)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z.float().unsqueeze(-1)
        angles = z * self.div
        pe = torch.zeros(*z.shape[:-1], self.d_model, device=z.device, dtype=z.dtype)
        pe[..., 0::2] = torch.sin(angles)
        pe[..., 1::2] = torch.cos(angles)
        return pe

class PoolMILVolumeClassifier(nn.Module):
    """
    Volume classifier that mirrors VolumeClassifier but uses mean/max pool
    instead of GatedAttention over slices.
    """

    def __init__(
        self,
        slice_encoder: PoolMILSliceClassifier,
        feat_dim: int,
        K: int,
        pool_mode: str = 'mean',
        dropout: float = 0.25,
    ):
        super().__init__()
        self.K         = K
        self.feat_dim  = feat_dim
        self.pool_mode = pool_mode

        self.slice_encoder = slice_encoder
        for p in self.slice_encoder.parameters():
            p.requires_grad_(False)
        self.slice_encoder.eval()

        self.pos_enc = SinPosEncoding(feat_dim)

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def train(self, mode=True):
        super().train(mode)
        self.slice_encoder.eval()
        return self

    @torch.no_grad()
    def _encode_slice(self, patches):
        enc = self.slice_encoder
        patches_b = patches.unsqueeze(0)
        features = enc.encode_patches(patches_b)
        B, N, _ = features.shape
        h = enc.projector(features.view(B * N, -1)).view(B, N, enc._feat_dim)
        if enc.pool_mode == 'max':
            z = h.max(dim=1).values
        else:
            z = h.mean(dim=1)
        return z.squeeze(0)

    def forward(self, patches_seq, z_indices, valid_mask=None):
        K = self.K
        device = patches_seq.device

        if valid_mask is None:
            valid_mask = (z_indices >= 0)

        h_list = []
        for k in range(K):
            if valid_mask[k]:
                with torch.no_grad():
                    h_k = self._encode_slice(patches_seq[k])
            else:
                h_k = torch.zeros(self.feat_dim, device=device,
                                  dtype=patches_seq.dtype)
            h_list.append(h_k)

        H = torch.stack(h_list, dim=0).unsqueeze(0)   # (1, K, feat_dim)

        z_clamp = z_indices.clamp(min=0)
        pe = self.pos_enc(z_clamp)
        H = H + pe.unsqueeze(0)

        # Pool over valid slices
        valid_f = valid_mask.float().unsqueeze(0).unsqueeze(-1)  # (1, K, 1)
        if self.pool_mode == 'max':
            H_masked = H - 1e9 * (1 - valid_f)
            v = H_masked.max(dim=1).values.squeeze(0)
        else:
            H_masked = H * valid_f
            n_valid = valid_f.sum(dim=1).clamp(min=1)
            v = (H_masked.sum(dim=1) / n_valid).squeeze(0)

        logit = self.head(v)
        return logit, None

# =============================================================================
#  Volume Dataset (for Stage 2)
# =============================================================================

class PoolVolumeDataset(Dataset):
    """K-slice windows with per-slice patch bags for volume-level training."""

    def __init__(self, data_dir, tab, K=16, patch_size=64, stride=32,
                 augment=False):
        self.data_dir   = data_dir
        self.tab        = tab.reset_index(drop=True)
        self.K          = K
        self.patch_size = patch_size
        self.stride     = stride
        self.augment    = augment

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

        ps, st = self.patch_size, self.stride
        half   = ps // 2
        dummy  = np.zeros((H, W), dtype=np.float32)
        _, grid, _ = build_patch_grid(dummy, ps, st)
        N = len(grid)

        patches_seq = torch.zeros(K, N, 1, ps, ps, dtype=torch.float32)
        z_indices   = torch.full((K,), -1, dtype=torch.long)
        valid       = torch.zeros(K, dtype=torch.bool)

        n_loaded = chunk.shape[0]
        for i in range(n_loaded):
            sl = chunk[i]
            if self.augment:
                if np.random.rand() > 0.5:
                    sl = sl[::-1, :].copy()
                if np.random.rand() > 0.5:
                    sl = sl[:, ::-1].copy()

            for j, (cy, cx) in enumerate(grid):
                patches_seq[i, j, 0] = torch.from_numpy(
                    sl[cy-half:cy+half, cx-half:cx+half].copy()
                ).float()

            z_indices[i] = z_start + i
            valid[i]     = True

        label = 0 if mod == 'real' else 1
        return dict(
            patches_seq=patches_seq,
            z_indices=z_indices,
            valid=valid,
            label=label,
            mod=mod,
            img_id=img_id,
        )

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
#  Evaluation helpers
# =============================================================================

def evaluate_slices(model, loader, criterion, device):
    model.eval()
    all_logits, all_labels, all_mods = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            patches = batch['patches'].to(device).repeat(1, 1, 3, 1, 1)  # (B,N,1,H,W)→3-ch
            labels  = (batch['label'] > 0).float().to(device)
            logits  = model(patches)
            if logits.dim() == 2:
                logits = logits.squeeze(1)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            all_logits.extend(logits.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_mods.extend(batch['mod'])

    scores = sigmoid_np(np.array(all_logits))
    labels = np.array(all_labels).astype(int)
    preds  = (scores >= 0.5).astype(int)

    m = dict(loss=total_loss / max(len(labels), 1))
    if len(set(labels.tolist())) > 1:
        m['auc'] = float(roc_auc_score(labels, scores))
        m['acc'] = float(accuracy_score(labels, preds))
        m['f1']  = float(f1_score(labels, preds, zero_division=0))
        m['ap']  = float(average_precision_score(labels, scores))
    return m, labels, scores, all_mods

def evaluate_volumes(model, loader, criterion, device):
    model.eval()
    all_logits, all_labels, all_mods = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            ps      = batch['patches_seq'].to(device)
            z_idx   = batch['z_indices'].to(device)
            valid   = batch['valid'].to(device)
            labels  = (batch['label'] > 0).float().to(device)

            B = ps.shape[0]
            logits_batch = []
            for b in range(B):
                logit, _ = model(ps[b].repeat(1, 1, 3, 1, 1), z_idx[b], valid[b])  # (K,N,1,H,W)→3-ch
                logits_batch.append(logit)

            logits = torch.stack(logits_batch)
            if logits.dim() > 1:
                logits = logits.squeeze(-1)
            loss = criterion(logits, labels)
            total_loss += loss.item() * B
            all_logits.extend(logits.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_mods.extend(batch['mod'])

    scores = sigmoid_np(np.array(all_logits))
    labels = np.array(all_labels).astype(int)
    preds  = (scores >= 0.5).astype(int)

    m = dict(loss=total_loss / max(len(labels), 1))
    if len(set(labels.tolist())) > 1:
        m['auc'] = float(roc_auc_score(labels, scores))
        m['acc'] = float(accuracy_score(labels, preds))
        m['f1']  = float(f1_score(labels, preds, zero_division=0))
        m['ap']  = float(average_precision_score(labels, scores))
    return m, labels, scores, all_mods

# =============================================================================
#  Args
# =============================================================================

def get_args():
    p = argparse.ArgumentParser(
        description='Pool MIL baseline (mean/max pooling instead of gated attention)')
    p.add_argument('--data_dir',     type=str, default=DATA_DIR)
    p.add_argument('--out_dir',      type=str, default=None)
    p.add_argument('--stage',        type=str, default='slice',
                   choices=['slice', 'volume'])
    p.add_argument('--pool_mode',    type=str, default='mean',
                   choices=['mean', 'max'])
    p.add_argument('--backbone',     type=str, default='resnet50')
    p.add_argument('--pretrained',   action='store_true', default=True)
    p.add_argument('--proj_dim',     type=int, default=512)
    p.add_argument('--dropout',      type=float, default=0.25)
    p.add_argument('--patch_size',   type=int, default=64)
    p.add_argument('--stride',       type=int, default=32)
    p.add_argument('--K',            type=int, default=16,
                   help='K-slice window for volume stage')
    p.add_argument('--slice_run_dir', type=str, default=None,
                   help='Stage 1 run dir for volume stage (loads frozen encoder)')
    p.add_argument('--epochs',       type=int, default=100)
    p.add_argument('--batch_size',   type=int, default=8)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--patience',     type=int, default=15)
    p.add_argument('--num_workers',  type=int, default=4)
    p.add_argument('--seed',         type=int, default=42)
    p.add_argument('--gpu_id',       type=int, default=None)
    p.add_argument('--train_mods',   nargs='+', default=None,
                   help='Fake modalities for training (default: all)')
    p.add_argument('--use_wandb',    action='store_true', default=False)
    p.add_argument('--run_name',     type=str, default=None)
    p.add_argument('--amp',          action='store_true', default=True)
    return p.parse_args()

# =============================================================================
#  Main — Stage 1 (Slice)
# =============================================================================

def train_slice(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = select_device(args.gpu_id)
    print(f"Device: {device}")

    train_mods = args.train_mods or ALL_FAKES
    mods_tag   = '+'.join(sorted(train_mods))
    if args.out_dir is None:
        args.out_dir = str(
            WORK_DIR / 'baselines' / 'runs'
            / f'pool_mil_{args.pool_mode}_slice' / f'trained_on_{mods_tag}'
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    args_dict = vars(args)
    with open(out_dir / 'args.json', 'w') as f:
        json.dump(args_dict, f, indent=2)

    # Data
    train_mods_with_real = ['real'] + train_mods
    test_mods = ['real'] + ALL_FAKES

    tab_train = load_split_table(args.data_dir, 'train', train_mods_with_real)
    tab_valid = load_split_table(args.data_dir, 'valid', train_mods_with_real)
    tab_test  = load_split_table(args.data_dir, 'test',  test_mods)

    ds_train = SliceDataset(args.data_dir, tab_train,
                            patch_size=args.patch_size, stride=args.stride,
                            augment=True)
    ds_valid = SliceDataset(args.data_dir, tab_valid,
                            patch_size=args.patch_size, stride=args.stride,
                            augment=False)
    ds_test  = SliceDataset(args.data_dir, tab_test,
                            patch_size=args.patch_size, stride=args.stride,
                            augment=False)

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
    model = build_pool_mil_slice(
        backbone=args.backbone, pretrained=args.pretrained,
        proj_dim=args.proj_dim, pool_mode=args.pool_mode,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Pool-MIL ({args.pool_mode}) slice  params={n_params:,}")

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
        run_name = args.run_name or f"pool_{args.pool_mode}_slice_{mods_tag}"
        wandb.init(project='MedForensics-baselines', name=run_name,
                   config=args_dict)

    best_val_loss    = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss, n_train = 0.0, 0

        for batch in tqdm(dl_train, desc=f'Epoch {epoch:03d}',
                          leave=False, dynamic_ncols=True):
            patches = batch['patches'].to(device).repeat(1, 1, 3, 1, 1)  # (B,N,1,H,W)→3-ch
            labels  = (batch['label'] > 0).float().to(device)

            optimiser.zero_grad()
            with torch.amp.autocast('cuda', enabled=args.amp):
                logits = model(patches)
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

    # Test
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

    eval_dir = out_dir / 'evaluation'
    eval_dir.mkdir(exist_ok=True)
    with open(eval_dir / 'metrics.json', 'w') as f:
        json.dump(dict(slice=test_m, args=args_dict), f, indent=2)

    if args.use_wandb:
        import wandb
        wandb.log({f'test/{k}': v for k, v in test_m.items()
                   if isinstance(v, (int, float))})
        wandb.finish()

    print(f"\nDone. Outputs: {out_dir}")

# =============================================================================
#  Main — Stage 2 (Volume)
# =============================================================================

def train_volume(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = select_device(args.gpu_id)
    print(f"Device: {device}")

    assert args.slice_run_dir is not None, \
        "--slice_run_dir required for volume stage"

    # Load Stage 1 args
    with open(Path(args.slice_run_dir) / 'args.json') as f:
        slice_args = json.load(f)

    ps   = slice_args.get('patch_size', args.patch_size)
    st   = slice_args.get('stride', args.stride)
    pool = slice_args.get('pool_mode', args.pool_mode)

    train_mods = args.train_mods or ALL_FAKES
    mods_tag   = '+'.join(sorted(train_mods))
    if args.out_dir is None:
        args.out_dir = str(
            WORK_DIR / 'baselines' / 'runs'
            / f'pool_mil_{pool}_volume_K{args.K}' / f'trained_on_{mods_tag}'
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    args_dict = vars(args)
    with open(out_dir / 'args.json', 'w') as f:
        json.dump(args_dict, f, indent=2)

    # Rebuild and load Stage 1 model
    slice_model = build_pool_mil_slice(
        backbone=slice_args.get('backbone', 'resnet50'),
        pretrained=False,
        proj_dim=slice_args.get('proj_dim', 512),
        pool_mode=pool,
        dropout=slice_args.get('dropout', 0.25),
    )
    ckpt = torch.load(
        Path(args.slice_run_dir) / 'best_model.pt',
        map_location='cpu', weights_only=False,
    )
    slice_model.load_state_dict(ckpt['model_state_dict'])
    slice_model.to(device).eval()

    feat_dim = slice_model.feat_dim

    # Build volume classifier
    model = PoolMILVolumeClassifier(
        slice_encoder=slice_model,
        feat_dim=feat_dim,
        K=args.K,
        pool_mode=pool,
        dropout=args.dropout,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Pool-MIL ({pool}) volume  trainable={trainable:,}  total={total:,}")

    # Data
    train_mods_with_real = ['real'] + train_mods
    test_mods = ['real'] + ALL_FAKES

    tab_train = load_split_table(args.data_dir, 'train', train_mods_with_real)
    tab_valid = load_split_table(args.data_dir, 'valid', train_mods_with_real)
    tab_test  = load_split_table(args.data_dir, 'test',  test_mods)

    ds_train = PoolVolumeDataset(args.data_dir, tab_train, K=args.K,
                                 patch_size=ps, stride=st, augment=True)
    ds_valid = PoolVolumeDataset(args.data_dir, tab_valid, K=args.K,
                                 patch_size=ps, stride=st, augment=False)
    ds_test  = PoolVolumeDataset(args.data_dir, tab_test, K=args.K,
                                 patch_size=ps, stride=st, augment=False)

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

    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    if args.use_wandb:
        import wandb
        run_name = args.run_name or f"pool_{pool}_vol_K{args.K}_{mods_tag}"
        wandb.init(project='MedForensics-baselines', name=run_name,
                   config=args_dict)

    best_val_loss    = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss, n_train = 0.0, 0

        for batch in tqdm(dl_train, desc=f'Epoch {epoch:03d}',
                          leave=False, dynamic_ncols=True):
            ps_t    = batch['patches_seq'].to(device)
            z_idx   = batch['z_indices'].to(device)
            valid   = batch['valid'].to(device)
            labels  = (batch['label'] > 0).float().to(device)

            B = ps_t.shape[0]
            optimiser.zero_grad()

            logits_batch = []
            for b in range(B):
                logit, _ = model(ps_t[b].repeat(1, 1, 3, 1, 1), z_idx[b], valid[b])  # (K,N,1,H,W)→3-ch
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

        val_m, _, _, _ = evaluate_volumes(model, dl_valid, criterion, device)
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

    # Test
    print("\n=== Volume-Level Test ===")
    ckpt = torch.load(out_dir / 'best_model.pt', map_location=device,
                       weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    test_m, t_labels, t_scores, t_mods = evaluate_volumes(
        model, dl_test, criterion, device,
    )
    test_m['per_mod'] = per_mod_metrics(t_labels, t_scores, t_mods)
    print(f"  Volume AUC={test_m.get('auc', float('nan')):.4f}  "
          f"Acc={test_m.get('acc', float('nan')):.4f}")
    for mod, mm in test_m.get('per_mod', {}).items():
        print(f"    {mod:12s}  AUC={mm['auc']:.4f}  F1={mm['f1']:.4f}")

    eval_dir = out_dir / 'evaluation'
    eval_dir.mkdir(exist_ok=True)
    with open(eval_dir / 'metrics.json', 'w') as f:
        json.dump(dict(volume=test_m, args=args_dict), f, indent=2)

    if args.use_wandb:
        import wandb
        wandb.log({f'test/{k}': v for k, v in test_m.items()
                   if isinstance(v, (int, float))})
        wandb.finish()

    print(f"\nDone. Outputs: {out_dir}")

# =============================================================================
#  Entry point
# =============================================================================

def main():
    args = get_args()
    if args.stage == 'slice':
        train_slice(args)
    else:
        train_volume(args)

if __name__ == '__main__':
    main()
