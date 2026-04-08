"""
slice_dataset.py
----------------
Dataset for Phase B: MIL bag of patches from a full CT slice.

Each sample is one annotated axial slice (coord_z from data.csv).
The slice is tiled into N patches via a sliding window.
The entire bag shares a multiclass label (real=0, pix2pix=1, cycle=2, diffusion=3).
Binary labels (real=0, fake=1) are derived downstream by callers via (label > 0).

Returns a dict with:
    patches   – Float32 (N, 1, P, P)  all patches in the bag
    label     – int   0=real, 1=pix2pix, 2=cycle, 3=diffusion
    mask      – Float32 (1, H, W)  full-resolution GT manipulation mask
    grid_hw   – LongTensor [n_rows, n_cols]  shape of the patch grid
    slice_hw  – LongTensor [H, W]  original slice resolution
    mod       – str   modality name
    img_id    – str   volume id
    coord_z   – int   axial slice index
"""

from __future__ import annotations

import os
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
from hexmil.data.patch_dataset import MOD_LABEL, load_split_table   # re-export for convenience


# ---------------------------------------------------------------------------
#  Grid extraction helper
# ---------------------------------------------------------------------------

def build_patch_grid(
    arr: np.ndarray,   # (H, W) float32
    patch_size: int,
    stride: int,
) -> tuple[np.ndarray, list[tuple[int, int]], tuple[int, int]]:
    """
    Extract a regular grid of patches from a 2-D array using a sliding window.

    Patches are centred at positions (cy, cx).  Centres are spaced `stride`
    apart starting at `patch_size // 2` so the first patch is fully within the
    image (with reflect-padding at the border if the slice is smaller than the
    patch size).

    Args:
        arr:        (H, W) numpy array, any dtype.
        patch_size: spatial size of each square patch.
        stride:     step between consecutive patch centres.

    Returns:
        patches:   (N, patch_size, patch_size) float32 array
        positions: list of (cy, cx) centre coordinates (length N)
        grid_hw:   (n_rows, n_cols)
    """
    H, W = arr.shape
    half  = patch_size // 2

    ys = list(range(half, H - half + 1, stride))
    xs = list(range(half, W - half + 1, stride))

    # Make sure we always cover up to the last column / row
    if not ys or ys[-1] < H - half:
        ys.append(H - half)
    if not xs or xs[-1] < W - half:
        xs.append(W - half)

    # De-duplicate (can happen when H == patch_size)
    ys = sorted(set(ys))
    xs = sorted(set(xs))

    n_rows, n_cols = len(ys), len(xs)
    patches   = np.empty((n_rows * n_cols, patch_size, patch_size), dtype=np.float32)
    positions: list[tuple[int, int]] = []

    idx = 0
    for cy in ys:
        for cx in xs:
            y0, y1 = cy - half, cy + half
            x0, x1 = cx - half, cx + half

            pt  = max(0, -y0);  pb = max(0, y1 - H)
            pl  = max(0, -x0);  pr = max(0, x1 - W)
            y0c, y1c = max(y0, 0), min(y1, H)
            x0c, x1c = max(x0, 0), min(x1, W)

            patch = arr[y0c:y1c, x0c:x1c]
            if pt or pb or pl or pr:
                patch = np.pad(patch, ((pt, pb), (pl, pr)), mode='reflect')

            patches[idx]  = patch
            positions.append((cy, cx))
            idx += 1

    return patches, positions, (n_rows, n_cols)


def reconstruct_heatmap(
    weights: np.ndarray,          # (N,) per-patch attention / score
    grid_hw: tuple[int, int],     # (n_rows, n_cols)
    slice_hw: tuple[int, int],    # (H, W) target resolution
    patch_size: int,
    stride: int,
) -> np.ndarray:
    """
    Place per-patch scalar values back into a 2-D heatmap by averaging
    contributions at each pixel position.

    Returns:
        heatmap: (H, W) float32 in [0, 1] after min-max normalisation.
    """
    H, W = slice_hw
    n_rows, n_cols = grid_hw
    half  = patch_size // 2

    ys = list(range(half, H - half + 1, stride))
    xs = list(range(half, W - half + 1, stride))
    if not ys or ys[-1] < H - half:
        ys.append(H - half)
    if not xs or xs[-1] < W - half:
        xs.append(W - half)
    ys = sorted(set(ys))
    xs = sorted(set(xs))

    accumulator = np.zeros((H, W), dtype=np.float32)
    count       = np.zeros((H, W), dtype=np.float32)

    idx = 0
    for cy in ys:
        for cx in xs:
            y0, y1 = max(cy - half, 0), min(cy + half, H)
            x0, x1 = max(cx - half, 0), min(cx + half, W)
            accumulator[y0:y1, x0:x1] += weights[idx]
            count[y0:y1, x0:x1]       += 1.0
            idx += 1

    heatmap = accumulator / np.maximum(count, 1e-6)

    # Min-max normalise to [0, 1]
    lo, hi = heatmap.min(), heatmap.max()
    if hi > lo:
        heatmap = (heatmap - lo) / (hi - lo)
    return heatmap


# ---------------------------------------------------------------------------
#  Dataset
# ---------------------------------------------------------------------------

class SliceDataset(Dataset):
    """
    Phase B dataset — one sample = one annotated axial slice as a MIL bag.

    Args:
        data_dir:   M3DSynth root (contains data.csv, sets.csv and per-mod dirs).
        tab:        DataFrame returned by load_split_table().
        patch_size: spatial size of each patch fed to the encoder (px).
        stride:     sliding-window step.  Default = patch_size // 2 (50% overlap).
        augment:    if True, apply random horizontal/vertical flips to the slice
                    before tiling (training only).
    """

    def __init__(
        self,
        data_dir: str,
        tab: pd.DataFrame,
        patch_size: int = 128,
        stride: int | None = None,
        augment: bool = False,
    ):
        self.data_dir   = data_dir
        self.tab        = tab.reset_index(drop=True)
        self.patch_size = patch_size
        self.stride     = stride if stride is not None else patch_size // 2
        self.augment    = augment

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.tab)

    def __getitem__(self, idx: int) -> dict:
        row    = self.tab.iloc[idx]
        mod    = row['mod']
        img_id = str(row['img_id'])
        cz     = int(row['coord_z'])

        # ── Load full axial slice ────────────────────────────────────────
        scan_dir   = os.path.join(self.data_dir, mod, 'scan', img_id)
        shape      = get_shape_tiff_scan(scan_dir)              # (Z, H, W)
        low, high  = get_percentile_tiff_scan(scan_dir, np.uint16)

        scan_slice = load_slice_tiff_scan(scan_dir, shape, np.uint16, cz, cz + 1)[0]
        scan_slice = apply_percentile(scan_slice.astype(np.float32), low, high)  # [0,1] float32

        # ── Load full-resolution manipulation mask ───────────────────────
        if mod == 'real':
            mask_slice = np.zeros_like(scan_slice, dtype=np.float32)
        else:
            label_dir  = os.path.join(self.data_dir, mod, 'label', img_id)
            mask_slice = load_slice_tiff_scan(label_dir, shape, np.bool_, cz, cz + 1)[0]
            mask_slice = mask_slice.astype(np.float32)

        # ── Slice-level augmentation (training only) ─────────────────────
        if self.augment:
            if np.random.rand() > 0.5:
                scan_slice = np.flip(scan_slice, axis=0).copy()
                mask_slice = np.flip(mask_slice, axis=0).copy()
            if np.random.rand() > 0.5:
                scan_slice = np.flip(scan_slice, axis=1).copy()
                mask_slice = np.flip(mask_slice, axis=1).copy()

        # ── Build patch bag ──────────────────────────────────────────────
        patches, _positions, grid_hw = build_patch_grid(
            scan_slice, self.patch_size, self.stride
        )
        # patches: (N, P, P)  float32

        # ── To tensors ───────────────────────────────────────────────────
        patches_t  = torch.from_numpy(patches).unsqueeze(1).float()   # (N, 1, P, P)
        mask_t     = torch.from_numpy(mask_slice).unsqueeze(0).float()  # (1, H, W)

        label = MOD_LABEL.get(mod, 0)   # multiclass: 0=real, 1=pix2pix, 2=cycle, 3=diffusion

        H, W = scan_slice.shape

        return dict(
            patches   = patches_t,                          # (N, 1, P, P)
            label     = label,                              # int  (multiclass)
            mask      = mask_t,                             # (1, H, W)
            grid_hw   = torch.tensor([grid_hw[0], grid_hw[1]], dtype=torch.long),
            slice_hw  = torch.tensor([H, W], dtype=torch.long),
            mod       = mod,
            img_id    = img_id,
            coord_z   = cz,
        )
