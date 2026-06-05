"""
slice_dataset.py
----------------
Stage 1 (SliceMIL) dataset: MIL bag of patches from a full CT slice.

For fake volumes, slices with sufficient GT mask coverage around coord_z are
selected; for real volumes, slices are sampled uniformly at random.
Each slice is tiled into N overlapping patches via a sliding window.

Returns a dict with:
    patches  – (N, 1, P, P)  patch bag
    label    – 0=real, 1-4=fake modality (use label > 0 for binary)
    mask     – (1, H, W)  GT manipulation mask
    grid_hw  – [n_rows, n_cols]
    slice_hw – [H, W]
    mod, img_id, coord_z, ty
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from tqdm import tqdm

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
    Stage 1 dataset — one sample = one axial slice as a MIL bag of patches.

    Slice selection per volume
    --------------------------
    Fake volumes:
        coord_z (the annotated manipulation centre) is always included.
        Up to ``neighborhood_slices`` additional slices are drawn uniformly
        at random from the neighbourhood window
        [coord_z - neighborhood_radius, coord_z + neighborhood_radius],
        clipped to [0, Z_total).  Total slices per fake volume =
        1 + neighborhood_slices.

    Real volumes:
        ``1 + neighborhood_slices`` slices are drawn uniformly at random
        from [0, Z_total) so the per-volume count matches fake volumes.

    No 3-D mask loading at init: only volume shapes are read (fast header
    reads), so dataset construction is near-instantaneous.

    Args:
        data_dir:             M3DSynth root directory.
        tab:                  DataFrame from load_split_table() — one row per volume.
        patch_size:           Spatial size of each patch (px).
        stride:               Sliding-window step.  Default = patch_size // 2.
        augment:              Random H/V flips (training only).
        neighborhood_radius:  Half-width of the window around coord_z from which
                              extra slices are sampled.  Default 5 → window of
                              ±5 slices (10 candidates, manipulated in all
                              typical M3DSynth volumes whose span ≈ 23 slices).
        neighborhood_slices:  Number of extra slices sampled beyond coord_z.
                              Default 4 → 5 slices per fake volume total.
    """

    def __init__(
        self,
        data_dir: str,
        tab: pd.DataFrame,
        patch_size: int = 128,
        stride: int | None = None,
        augment: bool = False,
        neighborhood_radius: int = 5,
        neighborhood_slices: int = 4,
    ):
        self.data_dir   = data_dir
        self.patch_size = patch_size
        self.stride     = stride if stride is not None else patch_size // 2
        self.augment    = augment

        self.samples = self._build_samples(
            tab.reset_index(drop=True),
            neighborhood_radius,
            neighborhood_slices,
        )

    # ------------------------------------------------------------------
    def _build_samples(
        self,
        tab: pd.DataFrame,
        radius: int,
        n_extra: int,
        seed: int = 42,
    ) -> list[dict]:
        """
        Expand the volume-level table into a flat list of (volume, slice_z) samples.
        No mask loading — uses only volume shapes and coord_z from the CSV.
        """
        rng     = np.random.default_rng(seed)
        samples: list[dict] = []

        fake_tab = tab[tab['mod'] != 'real']
        real_tab = tab[tab['mod'] == 'real']

        # ── Fake volumes: coord_z + neighbourhood ────────────────────────
        for _, row in fake_tab.iterrows():
            mod    = row['mod']
            img_id = str(row['img_id'])
            label  = MOD_LABEL.get(mod, 0)
            cz     = int(row['coord_z'])

            scan_dir = os.path.join(self.data_dir, mod, 'scan', img_id)
            try:
                Z_total = get_shape_tiff_scan(scan_dir)[0]
            except Exception as exc:
                print(f'  [SKIP] {mod}/{img_id}: {exc}')
                continue

            # Window around coord_z, clipped to valid range, excluding coord_z
            lo   = max(0, cz - radius)
            hi   = min(Z_total - 1, cz + radius)
            pool = [z for z in range(lo, hi + 1) if z != cz]

            chosen = [cz]
            if n_extra > 0 and pool:
                k      = min(n_extra, len(pool))
                chosen += rng.choice(pool, size=k, replace=False).tolist()

            ty_raw  = str(row.get('ty', 'injection'))
            ty_norm = 'removal' if ty_raw in ('rem', 'removal') else 'injection'
            for z in sorted(chosen):
                samples.append({
                    'mod':     mod,
                    'img_id':  img_id,
                    'label':   label,
                    'slice_z': int(z),
                    'ty':      ty_norm,
                })

        # ── Real volumes: sample enough slices to balance fake modalities ───
        # Each fake volume contributes (1 + n_extra) slices; there are
        # n_fake_mods fake modalities per real volume, so to equalise the
        # total fake/real sample count we sample (1 + n_extra) * n_fake_mods
        # slices from each real volume.
        n_fake_mods   = fake_tab['mod'].nunique() if len(fake_tab) > 0 else 1
        total_per_vol = (1 + n_extra) * n_fake_mods
        for _, row in real_tab.iterrows():
            img_id   = str(row['img_id'])
            scan_dir = os.path.join(self.data_dir, 'real', 'scan', img_id)
            try:
                Z_total = get_shape_tiff_scan(scan_dir)[0]
            except Exception as exc:
                print(f'  [SKIP] real/{img_id}: {exc}')
                continue

            ty_real = 'removal' if img_id.startswith('rem_') else 'injection'
            zs = rng.choice(Z_total, size=min(total_per_vol, Z_total), replace=False)
            for z in zs:
                samples.append({
                    'mod':     'real',
                    'img_id':  img_id,
                    'label':   0,
                    'slice_z': int(z),
                    'ty':      ty_real,
                })

        n_fake = sum(1 for s in samples if s['mod'] != 'real')
        n_real = len(samples) - n_fake
        print(f'[SliceDataset] {len(samples)} samples  '
              f'(fake: {n_fake}, real: {n_real})')
        return samples

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s      = self.samples[idx]
        mod    = s['mod']
        img_id = s['img_id']
        cz     = s['slice_z']

        # ── Load full axial slice ────────────────────────────────────────
        scan_dir  = os.path.join(self.data_dir, mod, 'scan', img_id)
        shape     = get_shape_tiff_scan(scan_dir)
        low, high = get_percentile_tiff_scan(scan_dir, np.uint16)

        scan_slice = load_slice_tiff_scan(scan_dir, shape, np.uint16, cz, cz + 1)[0]
        scan_slice = apply_percentile(scan_slice.astype(np.float32), low, high)

        # ── Load GT mask for this slice ──────────────────────────────────
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

        patches_t = torch.from_numpy(patches).unsqueeze(1).float()   # (N, 1, P, P)
        mask_t    = torch.from_numpy(mask_slice).unsqueeze(0).float() # (1, H, W)

        H, W = scan_slice.shape

        return dict(
            patches  = patches_t,
            label    = s['label'],
            mask     = mask_t,
            grid_hw  = torch.tensor([grid_hw[0], grid_hw[1]], dtype=torch.long),
            slice_hw = torch.tensor([H, W], dtype=torch.long),
            mod      = mod,
            img_id   = img_id,
            coord_z  = cz,
            ty       = s.get('ty', 'injection'),
        )
