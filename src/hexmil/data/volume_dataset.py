from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from hexmil.utils.tiff_utils import (
    get_shape_tiff_scan,
    load_slice_tiff_scan,
    get_percentile_tiff_scan,
    apply_percentile,
)
from hexmil.data.patch_dataset import MOD_LABEL, load_split_table
from hexmil.data.slice_dataset import build_patch_grid


class VolumeDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        tab: pd.DataFrame,
        K: int = 16,
        patch_size: int = 128,
        stride: Optional[int] = None,
        augment: bool = False,
    ):
        self.data_dir   = data_dir
        self.tab        = tab.reset_index(drop=True)
        self.K          = K
        self.patch_size = patch_size
        self.stride     = stride if stride is not None else patch_size // 2
        self.augment    = augment

    def __len__(self) -> int:
        return len(self.tab)

    def __getitem__(self, idx: int) -> dict:
        row    = self.tab.iloc[idx]
        mod    = row['mod']
        img_id = str(row['img_id'])
        cz     = int(row['coord_z'])

        scan_dir   = os.path.join(self.data_dir, mod, 'scan', img_id)
        shape      = get_shape_tiff_scan(scan_dir)       # (Z_total, H, W)
        Z_total, H, W = shape
        low, high  = get_percentile_tiff_scan(scan_dir, np.uint16)

        label_dir  = None if mod == 'real' else \
                     os.path.join(self.data_dir, mod, 'label', img_id)

        # Real: random consecutive K-slice subvolume (no anchor)
        # Fake: coord_z at a uniformly random position within the window
        K = self.K
        if mod == 'real':
            max_start = max(0, Z_total - K)
            z_start   = np.random.randint(0, max_start + 1)
            z_end     = z_start + K
        else:
            rand_offset = np.random.randint(0, K)   # [0, K-1]
            z_start     = cz - rand_offset
            z_end       = z_start + K
            # Clamp to volume bounds, preserving window length
            if z_end > Z_total:
                z_end   = Z_total
                z_start = z_end - K
            if z_start < 0:
                z_start = 0
                z_end   = z_start + K

        # actual slice indices available from the volume
        avail_start = max(z_start, 0)
        avail_end   = min(z_end, Z_total)

        # patches: compute N patches for a reference slice
        dummy_slice = np.zeros((H, W), dtype=np.float32)
        _, _, grid_hw = build_patch_grid(dummy_slice, self.patch_size, self.stride)
        n_rows, n_cols = grid_hw
        N = n_rows * n_cols

        patches_out = torch.zeros(self.K, N, 1, self.patch_size, self.patch_size,
                                  dtype=torch.float32)
        masks_out   = torch.zeros(self.K, 1, H, W, dtype=torch.float32)
        z_indices   = torch.full((self.K,), fill_value=-1, dtype=torch.long)
        grid_hw_out = torch.tensor([[n_rows, n_cols]] * self.K, dtype=torch.long)

        if avail_end > avail_start:
            scan_chunk = load_slice_tiff_scan(
                scan_dir, shape, np.uint16, avail_start, avail_end
            )    # (avail_end - avail_start, H, W) uint16

            if label_dir is not None:
                mask_chunk = load_slice_tiff_scan(
                    label_dir, shape, np.bool_, avail_start, avail_end
                ).astype(np.float32)   # (n_avail, H, W) float32
            else:
                mask_chunk = np.zeros_like(scan_chunk, dtype=np.float32)

            for local_z in range(avail_end - avail_start):
                global_z = avail_start + local_z
                # Position inside the K-window
                win_idx  = global_z - z_start

                if win_idx < 0 or win_idx >= self.K:
                    continue

                sl = apply_percentile(
                    scan_chunk[local_z].astype(np.float32), low, high
                )   # [0, 1] float32
                mk = mask_chunk[local_z]

                # Optional spatial augmentation (flip)
                if self.augment:
                    if np.random.rand() > 0.5:
                        sl = np.flip(sl, axis=0).copy()
                        mk = np.flip(mk, axis=0).copy()
                    if np.random.rand() > 0.5:
                        sl = np.flip(sl, axis=1).copy()
                        mk = np.flip(mk, axis=1).copy()

                patches_np, _, _ = build_patch_grid(sl, self.patch_size, self.stride)

                patches_out[win_idx] = torch.from_numpy(
                    patches_np
                ).unsqueeze(1).float()   # (N, 1, P, P)
                masks_out[win_idx]   = torch.from_numpy(mk).unsqueeze(0).float()
                z_indices[win_idx]   = global_z

        label     = MOD_LABEL.get(mod, 0)   # multiclass: 0=real,1=pix2pix,2=cycle,3=diffusion
        mod_label = label                    # alias kept for compatibility

        return dict(
            patches   = patches_out,                       # (K, N, 1, P, P)
            label     = label,                             # int
            mod_label = mod_label,                         # int
            masks     = masks_out,                         # (K, 1, H, W)
            z_indices = z_indices,                         # (K,) actual z or -1 if padded
            grid_hw   = grid_hw_out,                       # (K, 2)
            slice_hw  = torch.tensor([H, W], dtype=torch.long),
            mod       = mod,
            img_id    = img_id,
            coord_z   = cz,
        )
