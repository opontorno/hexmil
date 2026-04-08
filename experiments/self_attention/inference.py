#!/usr/bin/env python3
"""
inference.py
============
Standalone single-volume inference for the SA-ABMIL VolumeClassifier.

Reads all hyperparameters from the run's args.json.
The model type (SA or standard ABMIL) is auto-detected from the ``use_sa``
flag stored in args.json.
Divides the volume into non-overlapping K-slice windows (stride = K),
runs inference on each, and sets vol_score = max(window probabilities).

Usage
-----
    python experiments/SelfAttention/inference.py \\
        --run_dir  experiments/runs/volume-cls-sa_resnet50_p128_s64_K16 \\
        --scan_dir /mnt/.../M3DSynth/cycle/scan/rem_0001 \\
        [--label_dir /mnt/.../M3DSynth/cycle/label/rem_0001] \\
        [--out_dir /path/to/output] \\
        [--save_3d] [--save_nifti] \\
        [--beta_thresh 0.0] [--attn_thresh_3d 0.3] \\
        [--gpu_id 0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import torch
from scipy.special import expit as sigmoid_np
from tqdm import tqdm

from hexmil.data.slice_dataset import reconstruct_heatmap, build_patch_grid
from hexmil.models.volume_classifier import VolumeClassifier, build_volume_classifier
from hexmil.models.volume_classifier_sa import SAVolumeClassifier, build_sa_volume_classifier
from hexmil.utils.tiff_utils import (
    get_shape_tiff_scan,
    load_slice_tiff_scan,
    get_percentile_tiff_scan,
    apply_percentile,
)

# ── paths ─────────────────────────────────────────────────────────────────────

# =============================================================================
#  Args
# =============================================================================

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='SA-ABMIL single-volume inference')
    p.add_argument('--run_dir',   type=str, required=True,
                   help='Path to Phase C run directory (contains args.json + best_model.pt)')
    p.add_argument('--scan_dir',  type=str, required=True,
                   help='Path to scan directory (TIFF stack)')
    p.add_argument('--label_dir', type=str, default=None,
                   help='Optional path to GT label directory (binary TIFF masks)')
    p.add_argument('--out_dir',   type=str, default=None,
                   help='Output directory (default: WORK_DIR/.pictures/<vol_name>/)')
    p.add_argument('--gpu_id',    type=int, default=None)
    # ── visualisation flags ───────────────────────────────────────────────────
    p.add_argument('--save_3d',        action='store_true',
                   help='Save triplanar MIP + 3D scatter projection')
    p.add_argument('--save_nifti',     action='store_true',
                   help='Export volume + attention heatmap as .nii.gz')
    p.add_argument('--beta_thresh',    type=float, default=0.2,
                   help='Zero out slices with β_k < beta_thresh from the 3D heatmap '
                        '(suppresses non-suspicious slices; with K=16, uniform β≈0.0625)')
    p.add_argument('--attn_thresh_3d', type=float, default=0.7,
                   help='Attention threshold for 3D scatter voxel display')
    return p.parse_args()

# =============================================================================
#  Visualisation helpers  (standalone; duplicated from eval_volume-cls.py)
# =============================================================================

def _norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-8)

def _build_volume_3d(
    patches_cpu: torch.Tensor,   # (K, N, 1, P, P)
    grid_hw_arr: np.ndarray,     # (K, 2)
    slice_hw:    tuple,
    patch_size:  int,
    stride:      int,
    valid_mask:  np.ndarray,     # (K,) bool
) -> np.ndarray:                 # (K, H, W) float32
    K    = patches_cpu.shape[0]
    H, W = slice_hw
    vol  = np.zeros((K, H, W), dtype=np.float32)
    half = patch_size // 2
    ys   = sorted(set(list(range(half, H - half + 1, stride)) + [H - half]))
    xs   = sorted(set(list(range(half, W - half + 1, stride)) + [W - half]))
    for k in range(K):
        if not valid_mask[k]:
            continue
        pnp   = patches_cpu[k, :, 0].numpy()
        accum = np.zeros((H, W), dtype=np.float32)
        cnt   = np.zeros((H, W), dtype=np.float32)
        idx   = 0
        for cy in ys:
            for cx in xs:
                y0, y1 = max(cy - half, 0), min(cy + half, H)
                x0, x1 = max(cx - half, 0), min(cx + half, W)
                accum[y0:y1, x0:x1] += pnp[idx, :y1 - y0, :x1 - x0]
                cnt[y0:y1, x0:x1]   += 1.0
                idx += 1
        vol[k] = accum / np.maximum(cnt, 1e-6)
    return vol

def _smooth_3d(hmap3d: np.ndarray) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    out = np.zeros_like(hmap3d)
    for k in range(hmap3d.shape[0]):
        sl = hmap3d[k]
        if sl.max() > 0:
            sl = gaussian_filter(sl.astype(np.float32),
                                 sigma=max(1.0, max(sl.shape) * 0.03))
        out[k] = sl
    lo, hi = out.min(), out.max()
    return (out - lo) / (hi - lo + 1e-8)

def _mean_proj(vol3d: np.ndarray) -> tuple:
    return vol3d.mean(axis=0), vol3d.mean(axis=1), vol3d.mean(axis=2)

def _mask_proj_bbox(mask3d: np.ndarray) -> dict:
    def _bb(m2d):
        r = np.any(m2d, axis=1); c = np.any(m2d, axis=0)
        if not r.any() or not c.any():
            return None
        rmin, rmax = np.where(r)[0][[0, -1]]
        cmin, cmax = np.where(c)[0][[0, -1]]
        return int(rmin), int(rmax), int(cmin), int(cmax)
    return {
        'axial':    _bb((mask3d.max(axis=0) > 0.5)),
        'coronal':  _bb((mask3d.max(axis=1) > 0.5)),
        'sagittal': _bb((mask3d.max(axis=2) > 0.5)),
    }

def _save_volume_gif(
    volume_3d:   np.ndarray,
    heatmap_3d:  np.ndarray,
    masks_3d:    np.ndarray,
    beta_np:     np.ndarray,
    z_indices:   np.ndarray,
    valid_mask:  np.ndarray,
    label:       int | None,
    prob:        float,
    mod:         str,
    img_id:      str,
    save_path:   Path,
    duration_ms: int = 250,
) -> None:
    try:
        from PIL import Image
        import io
    except ImportError:
        print('[WARN] Pillow not installed; skipping GIF (pip install Pillow)')
        return

    gt_str = ('real' if label == 0 else mod) if label is not None else None
    pr_str = 'fake' if prob > 0.5 else 'real'
    frames = []

    for k in range(volume_3d.shape[0]):
        if not valid_mask[k]:
            continue
        sl_n = _norm01(volume_3d[k])
        hm_n = _norm01(heatmap_3d[k]) if heatmap_3d[k].max() > 0 else heatmap_3d[k]
        mk   = masks_3d[k]
        b_k  = float(beta_np[k])
        z_k  = int(z_indices[k])

        fig, axs = plt.subplots(1, 3, figsize=(9, 3.2))
        axs[0].imshow(sl_n, cmap='gray'); axs[0].set_title('Volume', fontsize=8); axs[0].axis('off')
        if label is not None and label > 0 and mk.sum() > 0:
            axs[0].contour(mk, levels=[0.5], colors='lime', linewidths=0.8)
        axs[1].imshow(hm_n, cmap='turbo', vmin=0, vmax=1)
        axs[1].set_title(f'Attention  β={b_k:.3f}', fontsize=8); axs[1].axis('off')
        axs[2].imshow(sl_n, cmap='gray')
        axs[2].imshow(hm_n, cmap='turbo', alpha=0.45, vmin=0, vmax=1)
        axs[2].set_title('Overlay', fontsize=8); axs[2].axis('off')
        if label is not None and label > 0 and mk.sum() > 0:
            axs[2].contour(mk, levels=[0.5], colors='lime', linewidths=0.8)
        gt_part = f'  GT:{gt_str}' if gt_str is not None else ''
        fig.suptitle(
            f'{img_id} | {mod} | z={z_k}{gt_part}  Pred:{pr_str}(p={prob:.2f})',
            fontsize=8,
        )
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()

    if frames:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            save_path, save_all=True, append_images=frames[1:],
            duration=duration_ms, loop=0,
        )
        print(f"  → GIF: {save_path}")

def save_volume_3d_projection(
    volume_3d:   np.ndarray,
    heatmap_3d:  np.ndarray,
    masks_3d:    np.ndarray,
    label:       int | None,
    prob:        float,
    mod:         str,
    img_id:      str,
    save_path:   Path,
    attn_thresh: float = 0.3,
) -> None:
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    gt_str = ('real' if label == 0 else mod) if label is not None else None
    pr_str = 'FAKE' if prob > 0.5 else 'REAL'
    hm     = np.clip(heatmap_3d, 0.0, 1.0)

    ct_ax, ct_co, ct_sa = _mean_proj(_norm01(volume_3d))
    bboxes = _mask_proj_bbox(masks_3d) if (label is not None and label > 0) else {}

    # 3-D attention bounding box for coronal/sagittal cube overlay
    _hm_mask = hm > max(attn_thresh, 0.05)
    if _hm_mask.any():
        _az, _ay, _ax_b = np.where(_hm_mask)
        _att_bbox = dict(
            z=(int(_az.min()),   int(_az.max())),
            y=(int(_ay.min()),   int(_ay.max())),
            x=(int(_ax_b.min()), int(_ax_b.max())),
        )
    else:
        _att_bbox = {}

    def _draw_bb(ax, bb):
        if bb is None:
            return
        r0, r1, c0, c1 = bb
        ax.add_patch(mpatches.Rectangle(
            (c0, r0), c1 - c0, r1 - r0, lw=1.5, edgecolor='lime', facecolor='none',
        ))

    fig = plt.figure(figsize=(20, 10), facecolor='#111111')
    fig.suptitle(
        (f'{img_id}  |  {mod}  GT:{gt_str}  Pred:{pr_str}  p={prob:.3f}'
         if gt_str is not None else
         f'{img_id}  |  {mod}  Pred:{pr_str}  p={prob:.3f}'),
        fontsize=25, fontweight='bold', color='white', y=1.01,
    )
    gs = gridspec.GridSpec(
        2, 4, figure=fig, wspace=0.06, hspace=0.06,
        left=0.03, right=0.97, top=0.89, bottom=0.02,
        width_ratios=[1, 1, 1, 1.3],
    )

    view_labels = ['Axial (z→)', 'Coronal (y→)', 'Sagittal (x→)']
    ct_imgs = [ct_ax, ct_co, ct_sa]
    bb_keys = ['axial', 'coronal', 'sagittal']

    for ci in range(3):
        ax0 = fig.add_subplot(gs[0, ci])
        ax0.imshow(ct_imgs[ci], cmap='gray', origin='upper', aspect='auto')
        _draw_bb(ax0, bboxes.get(bb_keys[ci]))
        ax0.set_title(view_labels[ci], fontsize=20, color='#aaaaaa', pad=2)
        ax0.axis('off')

        ax1 = fig.add_subplot(gs[1, ci])
        ax1.imshow(ct_imgs[ci], cmap='gray', origin='upper', aspect='auto')
        _draw_bb(ax1, bboxes.get(bb_keys[ci]))
        # attention bounding-box rectangle on all three views
        if ci == 0 and _att_bbox:   # axial: rows=y, cols=x
            y0, y1 = _att_bbox['y']; x0, x1 = _att_bbox['x']
            ax1.add_patch(mpatches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                lw=2, edgecolor='yellow', facecolor='red', alpha=0.4))
        if ci == 1 and _att_bbox:   # coronal: rows=z, cols=x
            z0, z1 = _att_bbox['z']; x0, x1 = _att_bbox['x']
            ax1.add_patch(mpatches.Rectangle(
                (x0, z0), x1 - x0, z1 - z0,
                lw=2, edgecolor='yellow', facecolor='red', alpha=0.4))
        if ci == 2 and _att_bbox:   # sagittal: rows=z, cols=y
            z0, z1 = _att_bbox['z']; y0, y1 = _att_bbox['y']
            ax1.add_patch(mpatches.Rectangle(
                (y0, z0), y1 - y0, z1 - z0,
                lw=2, edgecolor='yellow', facecolor='red', alpha=0.4))
        ax1.axis('off')

    for ri, lbl in enumerate(['CT (mean proj)', 'CT + BBox']):
        fig.text(0.01, 0.73 - ri * 0.47, lbl,
                 va='center', ha='left', fontsize=8, color='white', rotation=90)

    ax3d = fig.add_subplot(gs[:, 3], projection='3d')
    ax3d.set_facecolor('#111111')
    ax3d.set_xlabel('X', fontsize=10, color='white')
    ax3d.set_ylabel('Y', fontsize=10, color='white')
    ax3d.set_zlabel('Z', fontsize=10, color='white')
    ax3d.tick_params(colors='white', labelsize=10)
    ax3d.set_title('3-D CT projections', fontsize=20, color='white', pad=4)
    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('#333333')

    K_vol, H, W = hm.shape

    # ── CT wall projections (3 planes) ────────────────────────────────
    xr = np.arange(W); yr = np.arange(H); zr = np.arange(K_vol)
    Xw,  Yw  = np.meshgrid(xr, yr)
    Xw2, Zw2 = np.meshgrid(xr, zr)
    Yw3, Zw3 = np.meshgrid(yr, zr)
    ax3d.contourf(Xw,                       Yw,                   ct_ax,  zdir='z', offset=0, cmap='gray', alpha=0.35, levels=20)
    ax3d.contourf(np.rot90(Xw2, 2),  np.rot90(ct_co, 2),          Zw2,   zdir='y', offset=0, cmap='gray', alpha=0.35, levels=20)
    ax3d.contourf(np.rot90(ct_sa, 2),  np.rot90(Yw3, 2),          Zw3,   zdir='x', offset=0, cmap='gray', alpha=0.35, levels=20)

    # ── 3-D bounding box (red fill + yellow edges) ─────────────────────
    if _att_bbox:
        bx0, bx1 = _att_bbox['x']
        by0, by1 = _att_bbox['y']
        bz0, bz1 = _att_bbox['z']
        faces = [
            [[bx0,by0,bz0],[bx1,by0,bz0],[bx1,by1,bz0],[bx0,by1,bz0]],
            [[bx0,by0,bz1],[bx1,by0,bz1],[bx1,by1,bz1],[bx0,by1,bz1]],
            [[bx0,by0,bz0],[bx1,by0,bz0],[bx1,by0,bz1],[bx0,by0,bz1]],
            [[bx0,by1,bz0],[bx1,by1,bz0],[bx1,by1,bz1],[bx0,by1,bz1]],
            [[bx0,by0,bz0],[bx0,by1,bz0],[bx0,by1,bz1],[bx0,by0,bz1]],
            [[bx1,by0,bz0],[bx1,by1,bz0],[bx1,by1,bz1],[bx1,by0,bz1]],
        ]
        poly = Poly3DCollection(faces, alpha=0.18, facecolor='red', edgecolor='none')
        ax3d.add_collection3d(poly)
        for xs, ys, zs in [
            ([bx0,bx1],[by0,by0],[bz0,bz0]), ([bx0,bx1],[by1,by1],[bz0,bz0]),
            ([bx0,bx1],[by0,by0],[bz1,bz1]), ([bx0,bx1],[by1,by1],[bz1,bz1]),
            ([bx0,bx0],[by0,by1],[bz0,bz0]), ([bx1,bx1],[by0,by1],[bz0,bz0]),
            ([bx0,bx0],[by0,by1],[bz1,bz1]), ([bx1,bx1],[by0,by1],[bz1,bz1]),
            ([bx0,bx0],[by0,by0],[bz0,bz1]), ([bx1,bx1],[by0,by0],[bz0,bz1]),
            ([bx0,bx0],[by1,by1],[bz0,bz1]), ([bx1,bx1],[by1,by1],[bz0,bz1]),
        ]:
            ax3d.plot3D(xs, ys, zs, color='yellow', lw=1.5)

    ax3d.set_xlim(0, W); ax3d.set_ylim(0, H); ax3d.set_zlim(0, K_vol)
    ax3d.invert_yaxis()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor='#111111')
    plt.close(fig)
    print(f"  → 3D projection: {save_path}")

def save_as_nifti(
    volume_3d:  np.ndarray,
    heatmap_3d: np.ndarray,
    out_dir:    Path,
    prefix:     str = '',
) -> None:
    try:
        import nibabel as nib
    except ImportError:
        print('[WARN] nibabel not installed; skipping NIfTI export (pip install nibabel)')
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    affine = np.eye(4, dtype=np.float32)
    tag    = f'{prefix}_' if prefix else ''

    nib.save(
        nib.Nifti1Image(volume_3d.transpose(2, 1, 0).astype(np.float32), affine),
        str(out_dir / f'{tag}volume.nii.gz'),
    )
    nib.save(
        nib.Nifti1Image(heatmap_3d.transpose(2, 1, 0).astype(np.float32), affine),
        str(out_dir / f'{tag}attention.nii.gz'),
    )
    print(f"  → NIfTI: {out_dir}/{tag}volume.nii.gz  +  {tag}attention.nii.gz")

# =============================================================================
#  Inference core
# =============================================================================

@torch.no_grad()
def run_inference(
    model,
    scan_dir:           str,
    label_dir:          str | None,
    device:             torch.device,
    patch_size:         int,
    stride:             int,
    K:                  int,
    beta_thresh:        float = 0.0,
) -> dict:
    """
    Full-volume sliding-window inference (stride = K, non-overlapping).
    Returns dict with vol_score, pred, vol_full, hm_full, beta_full,
    valid_full, mask_full, z_indices_full, Z_total, H, W.
    """
    shape         = get_shape_tiff_scan(scan_dir)
    Z_total, H, W = shape
    low, high     = get_percentile_tiff_scan(scan_dir, np.uint16)

    scan_full = load_slice_tiff_scan(scan_dir, shape, np.uint16, 0, Z_total)
    mask_full = (
        load_slice_tiff_scan(label_dir, shape, np.bool_, 0, Z_total).astype(np.float32)
        if label_dir else np.zeros((Z_total, H, W), dtype=np.float32)
    )

    _, _, grid_hw = build_patch_grid(np.zeros((H, W), dtype=np.float32), patch_size, stride)
    n_rows, n_cols = grid_hw
    N              = n_rows * n_cols

    vol_full   = np.zeros((Z_total, H, W), dtype=np.float32)
    hm_full    = np.zeros((Z_total, H, W), dtype=np.float32)
    beta_full  = np.zeros(Z_total, dtype=np.float32)
    valid_full = np.zeros(Z_total, dtype=bool)
    win_probs  = []

    windows = []
    for z_start in range(0, Z_total, K):
        z_end       = z_start + K
        patches_out = torch.zeros(K, N, 1, patch_size, patch_size, dtype=torch.float32)
        z_indices   = torch.full((K,), -1, dtype=torch.long)
        avail_end   = min(z_end, Z_total)
        for local_z in range(avail_end - z_start):
            global_z = z_start + local_z
            sl = apply_percentile(scan_full[global_z].astype(np.float32), low, high)
            patches_np, _, _ = build_patch_grid(sl, patch_size, stride)
            patches_out[local_z] = torch.from_numpy(patches_np).unsqueeze(1).float()
            z_indices[local_z]   = global_z
        windows.append({
            'patches':    patches_out,
            'z_indices':  z_indices,
            'valid_mask': z_indices >= 0,
            'z_start':    z_start,
            'z_end':      min(z_end, Z_total),
        })

    for win in tqdm(windows, desc='Sliding-window inference', unit='win', dynamic_ncols=True):
        patches_t = win['patches'].to(device)
        z_t       = win['z_indices'].to(device)
        valid_t   = win['valid_mask'].to(device)
        valid_np  = win['valid_mask'].numpy()

        logit, attn_tup    = model(patches_t, z_t, valid_t, return_attn=True)
        beta_t, alpha_list = attn_tup
        beta_np_w  = beta_t.cpu().float().numpy()
        alpha_cpu  = [a.cpu().float().numpy() for a in alpha_list]

        hmap = np.zeros((K, H, W), dtype=np.float32)
        for k in range(K):
            if not valid_np[k]:
                continue
            if float(beta_np_w[k]) < beta_thresh:
                continue
            a2d = reconstruct_heatmap(alpha_cpu[k], grid_hw, (H, W), patch_size, stride)
            hmap[k] = np.clip(a2d * float(beta_np_w[k]), 0, 1)

        gh_arr  = np.tile(np.array(grid_hw), (K, 1))
        vol_win = _build_volume_3d(win['patches'], gh_arr, (H, W), patch_size, stride, valid_np)

        prob = float(sigmoid_np(logit.item()))
        win_probs.append(prob)

        z0, z1 = win['z_start'], win['z_end']
        k_eff  = z1 - z0
        vol_full[z0:z1] = vol_win[:k_eff]
        hm_full[z0:z1]  = hmap[:k_eff]
        for lk in range(k_eff):
            if valid_np[lk]:
                beta_full[z0 + lk]  = beta_np_w[lk]
                valid_full[z0 + lk] = True

    vol_score = max(win_probs)
    return {
        'vol_score':      vol_score,
        'pred':           'fake' if vol_score > 0.5 else 'real',
        'vol_full':       vol_full,
        'hm_full':        hm_full,
        'beta_full':      beta_full,
        'valid_full':     valid_full,
        'mask_full':      mask_full,
        'z_indices_full': np.arange(Z_total),
        'Z_total':        Z_total,
        'H':              H,
        'W':              W,
    }

# =============================================================================
#  Main
# =============================================================================

def main() -> None:
    args = get_args()

    run_dir   = Path(args.run_dir)
    ckpt_path = run_dir / 'best_model.pt'
    args_path = run_dir / 'args.json'

    if not ckpt_path.exists():
        raise FileNotFoundError(f"No best_model.pt in {run_dir}")
    if not args_path.exists():
        raise FileNotFoundError(f"No args.json in {run_dir}")

    with open(args_path) as f:
        saved_args = json.load(f)

    K          = saved_args['K']
    sargs      = saved_args.get('slice_args', saved_args)
    patch_size = sargs['patch_size']
    stride     = sargs.get('stride') or (patch_size // 2)
    use_sa     = saved_args.get('use_sa', False)

    # ── Device ────────────────────────────────────────────────────────────
    if args.gpu_id is not None:
        device = torch.device(f'cuda:{args.gpu_id}')
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    vol_name = Path(args.scan_dir).name
    out_dir  = Path(args.out_dir) if args.out_dir else Path(WORK_DIR) / '.pictures' / vol_name
    out_dir.mkdir(parents=True, exist_ok=True)

    arch_tag = 'SA-ABMIL' if use_sa else 'ABMIL'
    print(f"\n{'='*60}")
    print(f"  {arch_tag} Inference (SelfAttention experiment)")
    print(f"{'='*60}")
    print(f"  Run:     {run_dir}")
    print(f"  Volume:  {args.scan_dir}")
    print(f"  Out dir: {out_dir}")
    print(f"  Device:  {device}")
    print(f"  K={K}  patch_size={patch_size}  stride={stride}  use_sa={use_sa}\n")

    # ── Load model (SA or standard, auto-detected from args.json) ─────────
    slice_ckpt_dir = saved_args.get('slice_ckpt_dir', '')
    if slice_ckpt_dir and Path(slice_ckpt_dir).exists():
        if use_sa:
            model, _ = build_sa_volume_classifier(
                slice_ckpt_dir = slice_ckpt_dir,
                K              = K,
                attn_dim       = saved_args.get('attn_dim', 256),
                dropout        = saved_args.get('dropout',  0.25),
                sa_n_heads     = saved_args.get('sa_n_heads',  8),
                sa_n_layers    = saved_args.get('sa_n_layers', 2),
                device         = device,
            )
        else:
            model, _ = build_volume_classifier(
                slice_ckpt_dir = slice_ckpt_dir,
                K              = K,
                attn_dim       = saved_args.get('attn_dim', 256),
                dropout        = saved_args.get('dropout',  0.25),
                device         = device,
            )
    else:
        print("[WARN] slice_ckpt_dir not found; rebuilding slice encoder from sargs only")
        if use_sa:
            from hexmil.models.abmil_slice_classifier_sa import build_sa_classifier_scratch
            slice_model = build_sa_classifier_scratch(
                backbone    = sargs.get('backbone', 'resnet50'),
                pretrained  = False,
                proj_dim    = sargs.get('proj_dim', 512),
                attn_dim    = sargs.get('attn_dim', 256),
                dropout     = sargs.get('dropout',  0.25),
                sa_n_heads  = sargs.get('sa_n_heads',  8),
                sa_n_layers = sargs.get('sa_n_layers', 2),
            )
            model = SAVolumeClassifier(
                slice_encoder = slice_model,
                feat_dim      = slice_model.feat_dim,
                K             = K,
                attn_dim      = saved_args.get('attn_dim', 256),
                dropout       = saved_args.get('dropout',  0.25),
                sa_n_heads    = saved_args.get('sa_n_heads',  8),
                sa_n_layers   = saved_args.get('sa_n_layers', 2),
            ).to(device)
        else:
            from hexmil.models.abmil_slice_classifier import build_abmil_classifier_scratch
            slice_model = build_abmil_classifier_scratch(
                backbone   = sargs.get('backbone', 'resnet50'),
                pretrained = False,
                proj_dim   = sargs.get('proj_dim', 512),
                attn_dim   = sargs.get('attn_dim', 256),
                dropout    = sargs.get('dropout',  0.25),
            )
            model = VolumeClassifier(
                slice_encoder = slice_model,
                feat_dim      = slice_model.feat_dim,
                K             = K,
                attn_dim      = saved_args.get('attn_dim', 256),
                dropout       = saved_args.get('dropout',  0.25),
            ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=True)
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
          f"(val_loss={ckpt.get('best_val_loss', float('nan')):.4f})\n")

    # ── Run inference ─────────────────────────────────────────────────────
    results = run_inference(
        model              = model,
        scan_dir           = args.scan_dir,
        label_dir          = args.label_dir,
        device             = device,
        patch_size         = patch_size,
        stride             = stride,
        K                  = K,
        beta_thresh        = args.beta_thresh,
    )

    vol_score = results['vol_score']
    pred      = results['pred']
    label     = (1 if results['mask_full'].sum() > 0 else 0) if args.label_dir else None
    hm_smooth = _smooth_3d(results['hm_full'])

    print(f"\n  vol_score = {vol_score:.4f}  →  {pred.upper()}\n")

    # ── GIF ───────────────────────────────────────────────────────────────
    _save_volume_gif(
        volume_3d  = results['vol_full'],
        heatmap_3d = hm_smooth,
        masks_3d   = results['mask_full'],
        beta_np    = results['beta_full'],
        z_indices  = results['z_indices_full'],
        valid_mask = results['valid_full'],
        label      = label,
        prob       = vol_score,
        mod        = vol_name,
        img_id     = vol_name,
        save_path  = out_dir / f'{vol_name}.gif',
    )

    # ── 3-D triplanar projection ───────────────────────────────────────────
    if args.save_3d:
        save_volume_3d_projection(
            volume_3d   = results['vol_full'],
            heatmap_3d  = hm_smooth,
            masks_3d    = results['mask_full'],
            label       = label,
            prob        = vol_score,
            mod         = vol_name,
            img_id      = vol_name,
            save_path   = out_dir / f'{vol_name}_3d.png',
            attn_thresh = args.attn_thresh_3d,
        )

    # ── NIfTI export ──────────────────────────────────────────────────────
    if args.save_nifti:
        save_as_nifti(
            volume_3d  = results['vol_full'],
            heatmap_3d = hm_smooth,
            out_dir    = out_dir,
            prefix     = vol_name,
        )

    print(f"\nAll outputs written to: {out_dir}")
    print("Done.")

if __name__ == '__main__':
    main()
