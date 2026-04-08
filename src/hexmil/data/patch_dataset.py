"""
patch_dataset.py
----------------
Dataset for Phase A: single 2D patch centred on the nodule.

Each sample returns:
    image   – float32 tensor [C, H, W] where C=1 (grayscale), H=W=patch_size
    label   – int, multi-class: 0=real, 1=pix2pix, 2=cycle, 3=diffusion
               use (label > 0) for binary real/fake classification
    mask    – float32 tensor [1, H, W] binary manipulation mask (zeros for real)
    meta    – dict with img_id, mod, ty, orig_id, coord_z/y/x
"""

import os
import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment,misc]

from hexmil.utils.tiff_utils import (
    get_shape_tiff_scan,
    load_slice_tiff_scan,
    get_percentile_tiff_scan,
    apply_percentile,
)


# ---------------------------------------------------------------------------
#  Metadata helpers
# ---------------------------------------------------------------------------

# Multi-class modality labels — use (label > 0) for binary real/fake
MOD_LABEL = {'real': 0, 'pix2pix': 1, 'cycle': 2, 'diffusion': 3}

def load_split_table(data_dir: str, split: str, mods: list[str] | None = None) -> pd.DataFrame:
    """
    Merge data.csv + sets.csv and filter by split and modalities.
    
    Args:
        data_dir: root of M3DSynth.
        split:    'train', 'valid', or 'test'.
        mods:     list of modalities to keep.  None → all four.
    """
    data = pd.read_csv(os.path.join(data_dir, 'data.csv'))
    sets = pd.read_csv(os.path.join(data_dir, 'sets.csv'))
    tab  = data.merge(sets, on='orig_id', how='inner')
    tab  = tab[tab['set'] == split].reset_index(drop=True)
    if mods is not None:
        tab = tab[tab['mod'].isin(mods)].reset_index(drop=True)
    return tab


# ---------------------------------------------------------------------------
#  Patch extraction
# ---------------------------------------------------------------------------

def _extract_2d_patch(
    arr: np.ndarray,        # (H, W) single slice
    cy: int, cx: int,       # centre coords
    patch_size: int,
) -> np.ndarray:
    """Crop a square patch of `patch_size` around (cy, cx), with reflect-padding at borders."""
    half = patch_size // 2
    H, W = arr.shape

    y0, y1 = cy - half, cy + half
    x0, x1 = cx - half, cx + half

    # Determine how much padding we need
    pad_top    = max(0, -y0)
    pad_bottom = max(0, y1 - H)
    pad_left   = max(0, -x0)
    pad_right  = max(0, x1 - W)

    # Clamp to valid range
    y0c, y1c = max(y0, 0), min(y1, H)
    x0c, x1c = max(x0, 0), min(x1, W)

    patch = arr[y0c:y1c, x0c:x1c]

    if pad_top or pad_bottom or pad_left or pad_right:
        patch = np.pad(patch,
                       ((pad_top, pad_bottom), (pad_left, pad_right)),
                       mode='reflect')
    return patch


# ---------------------------------------------------------------------------
#  Dataset
# ---------------------------------------------------------------------------

class NodulePatchDataset(Dataset):
    """
    Phase A dataset: extract a single 2D patch centred on the nodule from
    the axial slice at coord_z.

    Args:
        data_dir:    M3DSynth root.
        tab:         DataFrame with columns: img_id, mod, ty, orig_id, sdir_id, 
                     coord_z, coord_y, coord_x, set.
        patch_size:  spatial extent of the cropped patch.
        augment:     if True, apply random flips + slight jitter during training.
        jitter_px:   max random shift (in pixels) of the crop centre.
    """

    def __init__(
        self,
        data_dir: str,
        tab: pd.DataFrame,
        patch_size: int = 128,
        augment: bool = False,
        jitter_px: int = 8,
    ):
        self.data_dir   = data_dir
        self.tab        = tab.reset_index(drop=True)
        self.patch_size = patch_size
        self.augment    = augment
        self.jitter_px  = jitter_px

    def __len__(self) -> int:
        return len(self.tab)

    def __getitem__(self, idx: int) -> dict:
        row = self.tab.iloc[idx]
        mod    = row['mod']
        img_id = str(row['img_id'])
        cz     = int(row['coord_z'])
        cy     = int(row['coord_y'])
        cx     = int(row['coord_x'])

        # ── Load single axial slice ──────────────────────────────────────
        scan_dir = os.path.join(self.data_dir, mod, 'scan', img_id)
        shape    = get_shape_tiff_scan(scan_dir)            # (Z, H, W)
        low, high = get_percentile_tiff_scan(scan_dir, np.uint16)

        scan_slice = load_slice_tiff_scan(scan_dir, shape, np.uint16, cz, cz + 1)[0]  # (H, W)
        scan_slice = apply_percentile(scan_slice.astype(np.float32), low, high)         # [0, 1]

        # ── Load mask (zeros for real) ───────────────────────────────────
        if mod == 'real':
            mask_slice = np.zeros_like(scan_slice, dtype=np.float32)
        else:
            label_dir = os.path.join(self.data_dir, mod, 'label', img_id)
            mask_slice = load_slice_tiff_scan(label_dir, shape, np.bool_, cz, cz + 1)[0]
            mask_slice = mask_slice.astype(np.float32)

        # ── Apply jitter (training augmentation) ─────────────────────────
        crop_cy, crop_cx = cy, cx
        if self.augment and self.jitter_px > 0:
            crop_cy += np.random.randint(-self.jitter_px, self.jitter_px + 1)
            crop_cx += np.random.randint(-self.jitter_px, self.jitter_px + 1)

        # ── Crop patch ───────────────────────────────────────────────────
        patch = _extract_2d_patch(scan_slice, crop_cy, crop_cx, self.patch_size)
        mask  = _extract_2d_patch(mask_slice, crop_cy, crop_cx, self.patch_size)

        # ── Augmentation: random flips ───────────────────────────────────
        if self.augment:
            if np.random.rand() > 0.5:
                patch = np.flip(patch, axis=0).copy()
                mask  = np.flip(mask,  axis=0).copy()
            if np.random.rand() > 0.5:
                patch = np.flip(patch, axis=1).copy()
                mask  = np.flip(mask,  axis=1).copy()

        # ── To tensor (ensure float32 — apply_percentile can promote to float64) ─
        image_t = torch.from_numpy(patch).unsqueeze(0).float()   # (1, H, W)
        mask_t  = torch.from_numpy(mask).unsqueeze(0).float()    # (1, H, W)
        label   = MOD_LABEL.get(mod, 1)

        return dict(
            image   = image_t,
            label   = label,
            mask    = mask_t,
            img_id  = img_id,
            mod     = mod,
            ty      = row['ty'],
            orig_id = row['orig_id'],
            coord   = (cz, cy, cx),
        )
