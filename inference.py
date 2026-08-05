#!/usr/bin/env python3

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
from hexmil.models.hexmil import HexMIL, build_hexmil
from hexmil.utils.tiff_utils import (
    get_shape_tiff_scan,
    load_slice_tiff_scan,
    get_percentile_tiff_scan,
    apply_percentile,
)

WORK_DIR = Path(__file__).resolve().parent


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='ABMIL single-volume inference')
    p.add_argument('--run_dir',   type=str, required=True,
                   help='Path to Stage 2 run directory (contains args.json + best_model.pt)')
    p.add_argument('--scan_dir',  type=str, required=True,
                   help='Path to scan directory (TIFF stack)')
    p.add_argument('--label_dir', type=str, default=None,
                   help='Optional path to GT label directory (binary TIFF masks)')
    p.add_argument('--out_dir',   type=str, default=None,
                   help='Output directory (default: WORK_DIR/.pictures/<vol_name>/)')
    p.add_argument('--gpu_id',    type=int, default=None)
    p.add_argument('--save_3d',        action='store_true',
                   help='Save triplanar MIP + 3D scatter projection')
    p.add_argument('--save_nifti',     action='store_true',
                   help='Export volume + attention heatmap as .nii.gz')
    p.add_argument('--beta_thresh',    type=float, default=0.1,
                   help='Zero out slices with β_k < beta_thresh from the 3D heatmap')
    p.add_argument('--win_stride',     type=int,   default=None,
                   help='Step between window starts in slices '
                        '(default: K = non-overlapping). Use K//2 for 50%% overlap.')
    p.add_argument('--attn_thresh_3d', type=float, default=0.65,
                   help='Attention threshold for 3D scatter voxel display')
    p.add_argument('--show_title',     action='store_true', default=False,
                   help='Show volume-level title on _3d.png and _3d_attn.png (default: off)')
    return p.parse_args()


#  Visualisation helpers  (standalone; duplicated from eval_volume.py)

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


def _masked_mean_proj(hm_filt: np.ndarray, axis: int):
    import numpy.ma as ma
    count  = (hm_filt > 0).sum(axis=axis).astype(np.float32)
    total  = hm_filt.sum(axis=axis)
    result = np.where(count > 0, total / np.maximum(count, 1.0), 0.0)
    return ma.masked_equal(result.astype(np.float32), 0.0)


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
        axs[0].imshow(sl_n, cmap='gray'); axs[0].set_title('Volume', fontsize=15); axs[0].axis('off')
        if label is not None and label > 0 and mk.sum() > 0:
            axs[0].contour(mk, levels=[0.5], colors='lime', linewidths=0.8)
        axs[1].imshow(hm_n, cmap='turbo', vmin=0, vmax=1)
        axs[1].set_title(f'Attention  β={b_k:.3f}', fontsize=15); axs[1].axis('off')
        axs[2].imshow(sl_n, cmap='gray')
        axs[2].imshow(hm_n, cmap='turbo', alpha=0.45, vmin=0, vmax=1)
        axs[2].set_title('Overlay', fontsize=15); axs[2].axis('off')
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
    show_title:  bool  = False,
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

    fig = plt.figure(figsize=(20, 10), facecolor='none')
    if show_title:
        fig.suptitle(
            (f'{img_id}  |  {mod}  GT:{gt_str}  Pred:{pr_str}  p={prob:.3f}'
             if gt_str is not None else
             f'{img_id}  |  {mod}  Pred:{pr_str}  p={prob:.3f}'),
            fontsize=25, fontweight='bold', color='black', y=1.01,
        )
    gs = gridspec.GridSpec(
        2, 4, figure=fig, wspace=0.005, hspace=0.005,
        left=0.03, right=0.99, top=0.95, bottom=0.01,
        width_ratios=[1, 1, 1, 1.3],
    )

    view_labels = ['Axial (z→)', 'Coronal (y→)', 'Sagittal (x→)']
    ct_imgs = [ct_ax, ct_co, ct_sa]
    bb_keys = ['axial', 'coronal', 'sagittal']

    for ci in range(3):
        ax0 = fig.add_subplot(gs[0, ci])
        ax0.imshow(ct_imgs[ci], cmap='gray', origin='upper', aspect='auto')
        _draw_bb(ax0, bboxes.get(bb_keys[ci]))
        ax0.set_title(view_labels[ci], fontsize=30, color='#444444', pad=2)
        ax0.axis('off')

        ax1 = fig.add_subplot(gs[1, ci])
        ax1.imshow(ct_imgs[ci], cmap='gray', origin='upper', aspect='auto')
        _draw_bb(ax1, bboxes.get(bb_keys[ci]))
        # attention bounding-box rectangle on all three views
        if ci == 0 and _att_bbox:   # axial: rows=y, cols=x
            y0, y1 = _att_bbox['y']; x0, x1 = _att_bbox['x']
            ax1.add_patch(mpatches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                lw=2, edgecolor='darkorange', facecolor='red', alpha=0.4))
        if ci == 1 and _att_bbox:   # coronal: rows=z, cols=x
            z0, z1 = _att_bbox['z']; x0, x1 = _att_bbox['x']
            ax1.add_patch(mpatches.Rectangle(
            (x0, z0), x1 - x0, z1 - z0,
            lw=2, edgecolor='darkorange', facecolor='red', alpha=0.4))
        if ci == 2 and _att_bbox:   # sagittal: rows=z, cols=y
            z0, z1 = _att_bbox['z']; y0, y1 = _att_bbox['y']
            ax1.add_patch(mpatches.Rectangle(
                (y0, z0), y1 - y0, z1 - z0,
                lw=2, edgecolor='darkorange', facecolor='red', alpha=0.4))
        ax1.axis('off')

    for ri, lbl in enumerate(['CT (mean proj)', 'CT + BBox']):
        fig.text(0.01, 0.73 - ri * 0.47, lbl,
                 va='center', ha='left', fontsize=15, color='black', rotation=90)

    ax3d = fig.add_subplot(gs[:, 3], projection='3d')
    ax3d.set_facecolor('none')
    ax3d.set_xlabel('X', fontsize=10, color='black')
    ax3d.set_ylabel('Y', fontsize=10, color='black')
    ax3d.set_zlabel('Z', fontsize=10, color='black')
    ax3d.tick_params(colors='black', labelsize=10)
    ax3d.set_title('3-D CT projections', fontsize=20, color='black', pad=4)
    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('#aaaaaa')

    K_vol, H, W = hm.shape

    xr = np.arange(W); yr = np.arange(H); zr = np.arange(K_vol)
    Xw,  Yw  = np.meshgrid(xr, yr)
    Xw2, Zw2 = np.meshgrid(xr, zr)
    Yw3, Zw3 = np.meshgrid(yr, zr)
    ax3d.contourf(Xw,                    Yw, ct_ax,                   zdir='z', offset=0, cmap='gray', alpha=0.35, levels=20)
    ax3d.contourf(np.rot90(Xw2, 2),   np.rot90(ct_co, 2),   Zw2,                     zdir='y', offset=0, cmap='gray', alpha=0.35, levels=20)
    ax3d.contourf(np.rot90(ct_sa, 2),   np.rot90(Yw3, 2),   Zw3,                     zdir='x', offset=0, cmap='gray', alpha=0.35, levels=20)

    if _att_bbox:
        bx0, bx1 = _att_bbox['x']
        by0, by1 = _att_bbox['y']
        bz0, bz1 = _att_bbox['z']
        # 6 faces for red fill
        faces = [
            [[bx0,by0,bz0],[bx1,by0,bz0],[bx1,by1,bz0],[bx0,by1,bz0]],  # bottom
            [[bx0,by0,bz1],[bx1,by0,bz1],[bx1,by1,bz1],[bx0,by1,bz1]],  # top
            [[bx0,by0,bz0],[bx1,by0,bz0],[bx1,by0,bz1],[bx0,by0,bz1]],  # front
            [[bx0,by1,bz0],[bx1,by1,bz0],[bx1,by1,bz1],[bx0,by1,bz1]],  # back
            [[bx0,by0,bz0],[bx0,by1,bz0],[bx0,by1,bz1],[bx0,by0,bz1]],  # left
            [[bx1,by0,bz0],[bx1,by1,bz0],[bx1,by1,bz1],[bx1,by0,bz1]],  # right
        ]
        poly = Poly3DCollection(faces, alpha=0.18, facecolor='red', edgecolor='none')
        ax3d.add_collection3d(poly)
        # wall projections of the bbox (shadow on each axis plane)
        proj_kw = dict(alpha=0.30, facecolor='red', edgecolor='darkorange', lw=1.5)
        ax3d.add_collection3d(Poly3DCollection([  # axial wall  (z=0)
            [[bx0,by0,0],[bx1,by0,0],[bx1,by1,0],[bx0,by1,0]]], **proj_kw))
        ax3d.add_collection3d(Poly3DCollection([  # coronal wall (y=0)
            [[bx0,0,bz0],[bx1,0,bz0],[bx1,0,bz1],[bx0,0,bz1]]], **proj_kw))
        ax3d.add_collection3d(Poly3DCollection([  # sagittal wall (x=0)
            [[0,by0,bz0],[0,by1,bz0],[0,by1,bz1],[0,by0,bz1]]], **proj_kw))
        for xs, ys, zs in [
            ([bx0,bx1],[by0,by0],[bz0,bz0]), ([bx0,bx1],[by1,by1],[bz0,bz0]),
            ([bx0,bx1],[by0,by0],[bz1,bz1]), ([bx0,bx1],[by1,by1],[bz1,bz1]),
            ([bx0,bx0],[by0,by1],[bz0,bz0]), ([bx1,bx1],[by0,by1],[bz0,bz0]),
            ([bx0,bx0],[by0,by1],[bz1,bz1]), ([bx1,bx1],[by0,by1],[bz1,bz1]),
            ([bx0,bx0],[by0,by0],[bz0,bz1]), ([bx1,bx1],[by0,by0],[bz0,bz1]),
            ([bx0,bx0],[by1,by1],[bz0,bz1]), ([bx1,bx1],[by1,by1],[bz0,bz1]),
        ]:
            ax3d.plot3D(xs, ys, zs, color='darkorange', lw=2.0, zorder=10)

    ax3d.set_xlim(0, W); ax3d.set_ylim(0, H); ax3d.set_zlim(0, K_vol)
    ax3d.invert_yaxis()
    ax3d.dist = 7

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches='tight', pad_inches=0.02,
                facecolor='none', transparent=True)
    plt.close(fig)
    print(f"  → 3D projection: {save_path}")


def save_attn_3d_projection(
    volume_3d:   np.ndarray,
    heatmap_3d:  np.ndarray,
    masks_3d:    np.ndarray,
    label:       int | None,
    prob:        float,
    mod:         str,
    img_id:      str,
    save_path:   Path,
    attn_thresh: float = 0.3,
    proj_thresh: float = 0.2,
    show_title:  bool  = False,
) -> None:
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    hm     = np.clip(heatmap_3d, 0.0, 1.0)
    gt_str = ('real' if label == 0 else mod) if label is not None else None
    pr_str = 'FAKE' if prob > 0.5 else 'REAL'

    # CT mean projections (background for 2D views)
    ct_ax, ct_co, ct_sa = _mean_proj(_norm01(volume_3d))

    # Attention projections filtered at proj_thresh (for 2D overlays and 3D walls)
    hm_proj = hm.copy()
    hm_proj[hm_proj < proj_thresh] = 0.0
    hm_ax = _masked_mean_proj(hm_proj, axis=0)   # (H, W)
    hm_co = _masked_mean_proj(hm_proj, axis=1)   # (K, W)
    hm_sa = _masked_mean_proj(hm_proj, axis=2)   # (K, H)
    # Keep masked arrays for 3D walls so contourf does not paint background
    hm_ax_w = hm_ax
    hm_co_w = hm_co
    hm_sa_w = hm_sa

    # Threshold used only for bbox/scatter
    hm_filt = hm.copy()
    hm_filt[hm_filt < attn_thresh] = 0.0
    _hm_mask = hm > max(attn_thresh, 0.05)

    K_vol, H, W = hm.shape
    view_labels = ['Axial (z→)', 'Coronal (y→)', 'Sagittal (x→)']
    ct_imgs     = [ct_ax, ct_co, ct_sa]
    hm_imgs     = [hm_ax, hm_co, hm_sa]

    fig = plt.figure(figsize=(20, 6), facecolor='none')
    if show_title:
        fig.suptitle(
            (f'{img_id}  |  {mod}  GT:{gt_str}  Pred:{pr_str}  p={prob:.3f}'
             if gt_str is not None else
             f'{img_id}  |  {mod}  Pred:{pr_str}  p={prob:.3f}'),
            fontsize=25, fontweight='bold', color='black', y=1.04,
        )
    gs = gridspec.GridSpec(
        1, 4, figure=fig, wspace=0.005,
        left=0.01, right=0.99, top=0.95, bottom=0.01,
        width_ratios=[1, 1, 1, 1.3],
    )

    im = None
    for ci in range(3):
        ax = fig.add_subplot(gs[0, ci])
        ax.imshow(ct_imgs[ci], cmap='gray', origin='upper', aspect='auto')
        im = ax.imshow(hm_imgs[ci], cmap='turbo', vmin=0.0, vmax=1.0,
                       alpha=0.6, origin='upper', aspect='auto')
        ax.set_title(view_labels[ci], fontsize=30, color='#444444', pad=2)
        ax.axis('off')

    # Col 3: CT walls (gray) + 3D scatter filtered at proj_thresh
    ax3d = fig.add_subplot(gs[0, 3], projection='3d')
    ax3d.set_facecolor('none')
    ax3d.set_xlabel('X', fontsize=10, color='black')
    ax3d.set_ylabel('Y', fontsize=10, color='black')
    ax3d.set_zlabel('Z', fontsize=10, color='black')
    ax3d.tick_params(colors='black', labelsize=8)
    ax3d.set_title('CT + Attention (3-D)', fontsize=20, color='black', pad=4)
    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('#aaaaaa')

    xr = np.arange(W); yr = np.arange(H); zr = np.arange(K_vol)
    Xw,  Yw  = np.meshgrid(xr, yr)
    Xw2, Zw2 = np.meshgrid(xr, zr)
    Yw3, Zw3 = np.meshgrid(yr, zr)
    ax3d.contourf(Xw,              Yw,              ct_ax,              zdir='z', offset=0, cmap='gray', alpha=0.35, levels=20)
    ax3d.contourf(np.rot90(Xw2,2), np.rot90(ct_co,2), Zw2,             zdir='y', offset=0, cmap='gray', alpha=0.35, levels=20)
    ax3d.contourf(np.rot90(ct_sa,2), np.rot90(Yw3,2), Zw3,             zdir='x', offset=0, cmap='gray', alpha=0.35, levels=20)

    # Attention wall overlays: draw only valid (unmasked) regions above proj_thresh
    wall_levels = np.linspace(proj_thresh, 1.0, 20)
    if np.ma.count(hm_ax_w) > 0:
        ax3d.contourf(Xw, Yw, hm_ax_w, zdir='z', offset=0,
                      cmap='turbo', vmin=proj_thresh, vmax=1.0,
                      alpha=0.40, levels=wall_levels)
    if np.ma.count(hm_co_w) > 0:
        ax3d.contourf(np.rot90(Xw2, 2), np.rot90(hm_co_w, 2), Zw2,
                      zdir='y', offset=0,
                      cmap='turbo', vmin=proj_thresh, vmax=1.0,
                      alpha=0.40, levels=wall_levels)
    if np.ma.count(hm_sa_w) > 0:
        ax3d.contourf(np.rot90(hm_sa_w, 2), np.rot90(Yw3, 2), Zw3,
                      zdir='x', offset=0,
                      cmap='turbo', vmin=proj_thresh, vmax=1.0,
                      alpha=0.40, levels=wall_levels)

    # Scatter filtered at proj_thresh (same as 2D projections)
    proj_mask = hm_proj > 0
    if proj_mask.any():
        vz, vy, vx = np.where(proj_mask)
        vals = hm[vz, vy, vx]
        ax3d.scatter(vx, vy, vz, c=vals, cmap='turbo',
                     vmin=0.0, vmax=1.0,
                     s=1.5, alpha=0.7, depthshade=True)

    ax3d.set_xlim(0, W); ax3d.set_ylim(0, H); ax3d.set_zlim(0, K_vol)
    ax3d.invert_yaxis()
    ax3d.dist = 7

    # Colorbar to the right of the 3D panel, without resizing it
    if im is not None:
        fig.canvas.draw()
        pos3d = ax3d.get_position()
        cax   = fig.add_axes([pos3d.x1 + 0.004, pos3d.y0 + pos3d.height * 0.1,
                               0.010, pos3d.height * 0.8])
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(colors='black', labelsize=7)
        cb.set_label('Attention', color='black', fontsize=10)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=400, bbox_inches='tight', facecolor='none', transparent=True)
    plt.close(fig)
    print(f"  → 3D attention projection: {save_path}")


def save_slice_attention_grid(
    volume_3d:    np.ndarray,   # (Z, H, W) CT volume [0,1]
    heatmap_3d:   np.ndarray,   # (Z, H, W) combined attention beta*alpha
    beta_full:    np.ndarray,   # (Z,) slice-level beta weights
    save_path:    Path,
    patch_size:   int   = 64,   # to reconstruct patch centers
    stride:       int   = 32,   # to reconstruct patch centers
    beta_thresh:  float = 0.0,  # absolute β threshold (same as --beta_thresh);
                                # normalised internally and shown on β bar
    alpha_thresh: float = 0.5,  # shown as horizontal dashed line on α bar chart
) -> None:
    import matplotlib.gridspec as gridspec
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    import matplotlib.cm as mcm

    Z, H, W = volume_3d.shape
    hm      = np.clip(heatmap_3d, 0.0, 1.0)
    hm_vmax = float(hm.max()) if hm.max() > 0 else 1.0

    valid = beta_full > 0
    if not valid.any():
        print(f"  → [skip] no valid beta values for slice attention grid")
        return

    center_z     = int(np.argmax(beta_full))
    valid_zs     = np.where(valid)[0]
    z_min, z_max = int(valid_zs.min()), int(valid_zs.max())
    row_zs       = [int(np.clip(center_z + dz, z_min, z_max))
                    for dz in (-3, -2, -1, 0, 1, 2, 3)]

    beta_n        = beta_full / max(float(beta_full.max()), 1e-8)
    # normalise the absolute threshold onto the same 0-1 scale as beta_n
    beta_thresh_n = beta_thresh / max(float(beta_full.max()), 1e-8)

    half    = patch_size // 2
    ys_c    = sorted(set(list(range(half, H - half + 1, stride)) + [H - half]))
    xs_c    = sorted(set(list(range(half, W - half + 1, stride)) + [W - half]))
    n_rows_p, n_cols_p = len(ys_c), len(xs_c)
    N_patches           = n_rows_p * n_cols_p
    x_pos               = np.arange(N_patches)
    turbo               = mcm.get_cmap('turbo')

    col_titles = ['β  |  α per patch', 'CT slice', 'Attention  α', 'Overlay']

    fig = plt.figure(figsize=(18, 33), facecolor='none')
    gs  = gridspec.GridSpec(
        7, 4, figure=fig,
        wspace=0.012, hspace=0.012,
        left=0.02, right=0.99, top=0.965, bottom=0.005,
        width_ratios=[1.5, 1, 1, 1],
    )

    for ri, z in enumerate(row_zs):
        ct_sl = volume_3d[z]
        hm_sl = hm[z]

        # pure α map (divide out β to recover per-patch attention weights)
        if beta_full[z] > 0:
            alpha_map = np.clip(hm_sl / max(float(beta_full[z]), 1e-8), 0.0, 1.0)
        else:
            alpha_map = np.zeros_like(hm_sl)

        gs_c0 = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs[ri, 0],
            height_ratios=[1, 3], hspace=0.06,
        )

        # — β scalar bar —
        ax0b = fig.add_subplot(gs_c0[0])
        ax0b.set_facecolor('none')
        ax0b.set_xlim(0, 1.08)
        ax0b.set_ylim(0, 1)
        ax0b.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax0b.spines.values():
            spine.set_visible(False)

        ax0b.barh(0.5, beta_n[z], height=0.55, left=0,
                  color='steelblue', alpha=0.90, zorder=2)
        # threshold marker — beta_thresh_n is beta_thresh normalised to [0,1] scale
        ax0b.axvline(beta_thresh_n, color='#ff6600', lw=2.0, ls='--',
                     zorder=4, alpha=0.9)
        ax0b.text(beta_thresh_n + 0.01, 0.93, r'$\tau_{\beta}$ = '+f'{beta_thresh:.2f}',
              va='top', ha='left', fontsize=25, color='#ff6600', fontweight='bold')
        if beta_n[z] > 0.22:
            ax0b.text(0.04, 0.5, f'{beta_full[z]:.3f}',
                      va='center', ha='left', fontsize=25, color='white', zorder=3)
        else:
            ax0b.text(min(beta_n[z] + 0.03, 1.04), 0.5, f'{beta_full[z]:.3f}',
                      va='center', ha='left', fontsize=25, color='#222222', zorder=3)
        ax0b.text(-0.04, 0.5, 'β',
                  va='center', ha='right', fontsize=34,
                  color='steelblue', fontweight='bold',
                  transform=ax0b.transData)
        if ri == 0:
            ax0b.set_title(col_titles[0], fontsize=30, color='black', pad=6)

        # — α per-patch bar chart (raster order: row 0 left→right, row 1, …) —
        ax0a = fig.add_subplot(gs_c0[1])
        ax0a.set_facecolor('#111111')
        for spine in ax0a.spines.values():
            spine.set_color('#444444')

        if beta_full[z] > 0:
            # sample alpha at each patch center (value is constant within patch)
            alpha_bars = np.array([
                alpha_map[min(ys_c[r], H - 1), min(xs_c[c], W - 1)]
                for r in range(n_rows_p) for c in range(n_cols_p)
            ])
            bar_colors = turbo(alpha_bars)   # (N, 4) RGBA via turbo
            ax0a.bar(x_pos, alpha_bars, color=bar_colors,
                     width=1.0, linewidth=0, zorder=2)
            # faint vertical separators at row boundaries
            for sep in range(n_cols_p, N_patches, n_cols_p):
                ax0a.axvline(sep - 0.5, color='white', lw=0.4, alpha=0.4, zorder=3)
        else:
            ax0a.text(0.5, 0.5, 'no attn', va='center', ha='center',
                      fontsize=25, color='#aaaaaa', transform=ax0a.transAxes)

        ax0a.set_xlim(-0.5, N_patches - 0.5)
        ax0a.set_ylim(0, 1.05)
        ax0a.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        # horizontal threshold line on α bar chart
        ax0a.axhline(alpha_thresh, color='#ff6600', lw=2.0, ls='--',
                     zorder=4, alpha=0.9)
        ax0a.text(N_patches - 1, alpha_thresh + 0.02, r'$\tau_{3D}$ = '+f'{alpha_thresh:.2f}',
                  va='bottom', ha='right', fontsize=25, color='#ff6600', fontweight='bold')
        ax0a.text(0.01, 0.97, 'α', va='top', ha='left', fontsize=34,
                  color='red', fontweight='bold',
                  transform=ax0a.transAxes)

        ax1 = fig.add_subplot(gs[ri, 1])
        ax1.imshow(ct_sl, cmap='gray', vmin=0, vmax=1,
                   origin='upper', aspect='auto')
        ax1.axis('off')
        if ri == 0:
            ax1.set_title(col_titles[1], fontsize=30, color='black', pad=6)

        ax2 = fig.add_subplot(gs[ri, 2])
        if beta_n[z] >= beta_thresh_n:
            ax2.imshow(hm_sl, cmap='turbo', vmin=0.0, vmax=hm_vmax,
                       origin='upper', aspect='auto')
        else:
            ax2.imshow(np.zeros_like(hm_sl), cmap='turbo', vmin=0, vmax=1,
                       origin='upper', aspect='auto')
        ax2.axis('off')
        if ri == 0:
            ax2.set_title(col_titles[2], fontsize=30, color='black', pad=6)

        ax3 = fig.add_subplot(gs[ri, 3])
        ax3.imshow(ct_sl, cmap='gray', vmin=0, vmax=1,
                   origin='upper', aspect='auto')
        if beta_n[z] >= beta_thresh_n:
            ax3.imshow(np.ma.masked_where(hm_sl < 0.02, hm_sl),
                       cmap='turbo', vmin=0.0, vmax=hm_vmax,
                       alpha=0.60, origin='upper', aspect='auto')
        ax3.axis('off')
        if ri == 0:
            ax3.set_title(col_titles[3], fontsize=30, color='black', pad=6)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=400, bbox_inches='tight',
                facecolor='none', transparent=True)
    plt.close(fig)
    print(f"  → Slice attention grid: {save_path}")


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


def save_combined_3d_projection(
    volume_3d:   np.ndarray,
    heatmap_3d:  np.ndarray,
    label:       int | None,
    prob:        float,
    mod:         str,
    img_id:      str,
    save_path:   Path,
    attn_thresh: float = 0.3,
    proj_thresh: float = 0.2,
    show_title:  bool  = False,
) -> None:
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    hm     = np.clip(heatmap_3d, 0.0, 1.0)
    gt_str = ('real' if label == 0 else mod) if label is not None else None
    pr_str = 'FAKE' if prob > 0.5 else 'REAL'
    ct_ax, ct_co, ct_sa = _mean_proj(_norm01(volume_3d))
    ct_imgs = [ct_ax, ct_co, ct_sa]

    # Attention projections filtered at proj_thresh (2D overlays + 3D walls)
    hm_proj = hm.copy()
    hm_proj[hm_proj < proj_thresh] = 0.0
    hm_imgs = [
        _masked_mean_proj(hm_proj, axis=0),  # axial   (H, W)
        _masked_mean_proj(hm_proj, axis=1),  # coronal (K, W)
        _masked_mean_proj(hm_proj, axis=2),  # sagittal(K, H)
    ]
    hm_ax_w, hm_co_w, hm_sa_w = hm_imgs

    # Threshold used only for bbox/scatter
    hm_filt = hm.copy()
    hm_filt[hm_filt < attn_thresh] = 0.0

    _hm_mask = hm > max(attn_thresh, 0.05)
    if _hm_mask.any():
        _az, _ay, _ax_b = np.where(_hm_mask)
        _att_bbox = dict(
            z=(int(_az.min()), int(_az.max())),
            y=(int(_ay.min()), int(_ay.max())),
            x=(int(_ax_b.min()), int(_ax_b.max())),
        )
    else:
        _att_bbox = {}

    bb_keys     = ['axial', 'coronal', 'sagittal']
    view_labels = ['Axial (z→)', 'Coronal (y→)', 'Sagittal (x→)']
    K_vol, H, W = hm.shape

    fig = plt.figure(figsize=(26, 15), facecolor='none')
    if show_title:
        fig.suptitle(
            (f'{img_id}  |  {mod}  GT:{gt_str}  Pred:{pr_str}  p={prob:.3f}'
             if gt_str is not None else
             f'{img_id}  |  {mod}  Pred:{pr_str}  p={prob:.3f}'),
            fontsize=22, fontweight='bold', color='black', y=1.01,
        )
    gs = gridspec.GridSpec(
        3, 4, figure=fig, wspace=0.005, hspace=0.005,
        left=0.04, right=0.99, top=0.96, bottom=0.01,
        width_ratios=[1, 1, 1, 2],
    )

    ax_left = {}   # leftmost ax per row — used later for label alignment

    for ci in range(3):
        ax = fig.add_subplot(gs[0, ci])
        ax.imshow(ct_imgs[ci], cmap='gray', origin='upper', aspect='auto')
        ax.set_title(view_labels[ci], fontsize=40, color='#444444', pad=2)
        ax.axis('off')
        if ci == 0:
            ax_left[0] = ax
    for ci in range(3):
        ax = fig.add_subplot(gs[1, ci])
        ax.imshow(ct_imgs[ci], cmap='gray', origin='upper', aspect='auto')
        im_attn = ax.imshow(hm_imgs[ci], cmap='turbo', vmin=0.0, vmax=1.0,
                            alpha=0.6, origin='upper', aspect='auto')
        ax.axis('off')
        if ci == 0:
            ax_left[1] = ax

    # col 3 all rows: two 3D panels filling the full column height
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    gs_3d     = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0:3, 3], hspace=0.04)
    ax3d_attn = fig.add_subplot(gs_3d[0], projection='3d')
    ax3d_bb   = fig.add_subplot(gs_3d[1], projection='3d')

    xr = np.arange(W); yr = np.arange(H); zr = np.arange(K_vol)
    Xw, Yw   = np.meshgrid(xr, yr)
    Xw2, Zw2 = np.meshgrid(xr, zr)
    Yw3, Zw3 = np.meshgrid(yr, zr)
    proj_mask = hm_proj > 0

    def _setup_3d(ax):
        ax.set_facecolor('none')
        ax.set_xlabel('X', fontsize=20, color='black')
        ax.set_ylabel('Y', fontsize=20, color='black')
        ax.set_zlabel('Z', fontsize=20, color='black')
        ax.tick_params(colors='black', labelsize=15)
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False; pane.set_edgecolor('#aaaaaa')

    def _ct_walls(ax):
        ax.contourf(Xw,              Yw,              ct_ax,   zdir='z', offset=0, cmap='gray', alpha=0.30, levels=20)
        ax.contourf(np.rot90(Xw2,2), np.rot90(ct_co,2), Zw2,  zdir='y', offset=0, cmap='gray', alpha=0.30, levels=20)
        ax.contourf(np.rot90(ct_sa,2), np.rot90(Yw3,2), Zw3,  zdir='x', offset=0, cmap='gray', alpha=0.30, levels=20)

    def _attn_walls(ax):
        wall_levels = np.linspace(proj_thresh, 1.0, 20)
        if np.ma.count(hm_ax_w) > 0:
            ax.contourf(Xw, Yw, hm_ax_w, zdir='z', offset=0,
                        cmap='turbo', vmin=proj_thresh, vmax=1.0,
                        alpha=0.40, levels=wall_levels)
        if np.ma.count(hm_co_w) > 0:
            ax.contourf(np.rot90(Xw2, 2), np.rot90(hm_co_w, 2), Zw2,
                        zdir='y', offset=0,
                        cmap='turbo', vmin=proj_thresh, vmax=1.0,
                        alpha=0.40, levels=wall_levels)
        if np.ma.count(hm_sa_w) > 0:
            ax.contourf(np.rot90(hm_sa_w, 2), np.rot90(Yw3, 2), Zw3,
                        zdir='x', offset=0,
                        cmap='turbo', vmin=proj_thresh, vmax=1.0,
                        alpha=0.40, levels=wall_levels)

    def _scatter_attn(ax, s=1.5, alpha=0.7):
        if proj_mask.any():
            vz, vy, vx = np.where(proj_mask)
            vals = hm[vz, vy, vx]
            ax.scatter(vx, vy, vz, c=vals, cmap='turbo',
                       vmin=0.0, vmax=1.0, s=s, alpha=alpha, depthshade=True)

    def _finalize_3d(ax):
        ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_zlim(0, K_vol)
        ax.invert_yaxis()
        ax.dist = 7

    _setup_3d(ax3d_attn)
    _ct_walls(ax3d_attn)
    _attn_walls(ax3d_attn)
    _scatter_attn(ax3d_attn, s=1.5, alpha=0.7)
    _finalize_3d(ax3d_attn)

    for ci in range(3):
        ax = fig.add_subplot(gs[2, ci])
        ax.imshow(ct_imgs[ci], cmap='gray', origin='upper', aspect='auto')
        if ci == 0:
            ax_left[2] = ax
        if _att_bbox:
            if ci == 0:
                y0, y1 = _att_bbox['y']; x0, x1 = _att_bbox['x']
                ax.add_patch(mpatches.Rectangle(
                    (x0, y0), x1 - x0, y1 - y0,
                    lw=2, edgecolor='darkorange', facecolor='red', alpha=0.25))
            elif ci == 1:
                z0, z1 = _att_bbox['z']; x0, x1 = _att_bbox['x']
                ax.add_patch(mpatches.Rectangle(
                    (x0, z0), x1 - x0, z1 - z0,
                    lw=2, edgecolor='darkorange', facecolor='red', alpha=0.25))
            elif ci == 2:
                z0, z1 = _att_bbox['z']; y0, y1 = _att_bbox['y']
                ax.add_patch(mpatches.Rectangle(
                    (y0, z0), y1 - y0, z1 - z0,
                    lw=2, edgecolor='darkorange', facecolor='red', alpha=0.25))
        ax.axis('off')

    _setup_3d(ax3d_bb)
    _ct_walls(ax3d_bb)
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
        # bbox wall projections (shadow on each wall)
        proj_kw = dict(alpha=0.30, facecolor='red', edgecolor='darkorange', lw=1.5)
        ax3d_bb.add_collection3d(Poly3DCollection([  # axial wall  (z=0)
            [[bx0,by0,0],[bx1,by0,0],[bx1,by1,0],[bx0,by1,0]]], **proj_kw))
        ax3d_bb.add_collection3d(Poly3DCollection([  # coronal wall (y=0)
            [[bx0,0,bz0],[bx1,0,bz0],[bx1,0,bz1],[bx0,0,bz1]]], **proj_kw))
        ax3d_bb.add_collection3d(Poly3DCollection([  # sagittal wall (x=0)
            [[0,by0,bz0],[0,by1,bz0],[0,by1,bz1],[0,by0,bz1]]], **proj_kw))
        # bbox faces first (behind), then wireframe last (on top)
        ax3d_bb.add_collection3d(Poly3DCollection(
            faces, alpha=0.15, facecolor='red', edgecolor='darkorange', lw=1.5))
        for xs, ys, zs in [
            ([bx0,bx1],[by0,by0],[bz0,bz0]),([bx0,bx1],[by1,by1],[bz0,bz0]),
            ([bx0,bx1],[by0,by0],[bz1,bz1]),([bx0,bx1],[by1,by1],[bz1,bz1]),
            ([bx0,bx0],[by0,by1],[bz0,bz0]),([bx1,bx1],[by0,by1],[bz0,bz0]),
            ([bx0,bx0],[by0,by1],[bz1,bz1]),([bx1,bx1],[by0,by1],[bz1,bz1]),
            ([bx0,bx0],[by0,by0],[bz0,bz1]),([bx1,bx1],[by0,by0],[bz0,bz1]),
            ([bx0,bx0],[by1,by1],[bz0,bz1]),([bx1,bx1],[by1,by1],[bz0,bz1]),
        ]:
            ax3d_bb.plot3D(xs, ys, zs, color='darkorange', lw=3.0, zorder=10)
    _finalize_3d(ax3d_bb)

    fig.canvas.draw()   # force layout so get_position() is accurate

    # tighten layout: move both 3D panels slightly left, closer to 2D grid
    shift_3d_left = 0.03
    for ax3d in (ax3d_attn, ax3d_bb):
        pos = ax3d.get_position()
        ax3d.set_position([pos.x0 - shift_3d_left, pos.y0, pos.width, pos.height])

    row_labels = ['CT projections', 'CT + Attention', 'CT + Attn Volume']
    for ri, lbl in enumerate(row_labels):
        if ri in ax_left:
            pos = ax_left[ri].get_position()
            cy  = (pos.y0 + pos.y1) / 2
            fig.text(0.01, cy, lbl,
                     va='center', ha='left', fontsize=40, color='black', rotation=90)

    # Column title centred above both 3D panels
    pos_top = ax3d_attn.get_position()
    cx_3d   = (pos_top.x0 + pos_top.x1) / 2
    fig.text(cx_3d, pos_top.y1 + 0.01, '3D projections',
             va='bottom', ha='center', fontsize=40, color='black', fontweight=None)

    # Colorbar to the right of the attention 3D, without stealing its space
    pos3d = ax3d_attn.get_position()
    cax   = fig.add_axes([pos3d.x1 + 0.03, pos3d.y0 + pos3d.height * 0.1,
                          0.010, pos3d.height * 0.8])
    cb = fig.colorbar(im_attn, cax=cax)
    cb.ax.tick_params(colors='black', labelsize=15)
    cb.set_label('Attention', color='black', fontsize=20)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=400, bbox_inches='tight', facecolor='none', transparent=True)
    plt.close(fig)
    print(f"  → Combined 3D projection: {save_path}")


@torch.no_grad()
def run_inference(
    model:              HexMIL,
    scan_dir:           str,
    label_dir:          str | None,
    device:             torch.device,
    patch_size:         int,
    stride:             int,
    K:                  int,
    beta_thresh:        float = 0.0,
    win_stride:         int | None = None,
) -> dict:
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
    hm_accum   = np.zeros((Z_total, H, W), dtype=np.float32)
    beta_accum = np.zeros(Z_total, dtype=np.float32)
    count_full = np.zeros(Z_total, dtype=np.float32)
    valid_full = np.zeros(Z_total, dtype=bool)
    win_probs  = []

    _win_stride = win_stride if win_stride is not None else K
    windows = []
    for z_start in range(0, Z_total, _win_stride):
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
        vol_full[z0:z1] = vol_win[:k_eff]   # CT intensities: deterministic, last-write fine
        for lk in range(k_eff):
            if valid_np[lk]:
                gz = z0 + lk
                hm_accum[gz]   += hmap[lk]
                beta_accum[gz] += beta_np_w[lk]
                count_full[gz] += 1.0
                valid_full[gz]  = True

    # Average heatmap and beta over all windows that covered each slice
    cnt       = np.maximum(count_full, 1.0)
    hm_full   = hm_accum   / cnt[:, None, None]
    beta_full = beta_accum / cnt

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

    if args.gpu_id is not None:
        device = torch.device(f'cuda:{args.gpu_id}')
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    vol_name = Path(args.scan_dir).name
    out_dir  = Path(args.out_dir) if args.out_dir else WORK_DIR / '.pictures' / vol_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  ABMIL Inference")
    print(f"{'='*60}")
    print(f"  Run:     {run_dir}")
    print(f"  Volume:  {args.scan_dir}")
    print(f"  Out dir: {out_dir}")
    print(f"  Device:  {device}")
    print(f"  K={K}  patch_size={patch_size}  stride={stride}\n")

    slice_ckpt_dir = saved_args.get('slice_ckpt_dir', '')
    if slice_ckpt_dir and Path(slice_ckpt_dir).exists():
        model, _ = build_hexmil(
            slice_ckpt_dir = slice_ckpt_dir,
            K              = K,
            attn_dim       = saved_args.get('attn_dim', 256),
            dropout        = saved_args.get('dropout',  0.25),
            device         = device,
        )
    else:
        print("[WARN] slice_ckpt_dir not found; rebuilding slice encoder from sargs only")
        from hexmil.models.slicemil import build_slicemil
        slice_model = build_slicemil(
            backbone   = sargs.get('backbone',   'resnet50'),
            pretrained = False,
            patch_size = sargs.get('patch_size', 64),
            proj_dim   = sargs.get('proj_dim',   512),
            attn_dim   = sargs.get('attn_dim',   256),
            dropout    = sargs.get('dropout',    0.25),
        )
        model = HexMIL(
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

    label_dir = args.label_dir
    if label_dir is None:
        _auto = args.scan_dir.replace('/scan/', '/label/')
        if Path(_auto).exists():
            label_dir = _auto
            print(f"  [auto] label_dir detected: {label_dir}")

    results = run_inference(
        model              = model,
        scan_dir           = args.scan_dir,
        label_dir          = label_dir,
        device             = device,
        patch_size         = patch_size,
        stride             = stride,
        K                  = K,
        beta_thresh        = args.beta_thresh,
        win_stride         = args.win_stride,
    )

    vol_score = results['vol_score']
    pred      = results['pred']
    label     = (1 if results['mask_full'].sum() > 0 else 0) if label_dir else None
    hm_smooth = _smooth_3d(results['hm_full'])

    print(f"\n  vol_score = {vol_score:.4f}  →  {pred.upper()}\n")

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
            show_title  = args.show_title,
        )
        save_attn_3d_projection(
            volume_3d   = results['vol_full'],
            heatmap_3d  = hm_smooth,
            masks_3d    = results['mask_full'],
            label       = label,
            prob        = vol_score,
            mod         = vol_name,
            img_id      = vol_name,
            save_path   = out_dir / f'{vol_name}_3d_attn.png',
            attn_thresh = args.attn_thresh_3d,
            show_title  = args.show_title,
        )
        save_combined_3d_projection(
            volume_3d   = results['vol_full'],
            heatmap_3d  = hm_smooth,
            label       = label,
            prob        = vol_score,
            mod         = vol_name,
            img_id      = vol_name,
            save_path   = out_dir / f'{vol_name}_3d_combined.png',
            attn_thresh = args.attn_thresh_3d,
            show_title  = args.show_title,
        )

    if args.save_3d:
        save_slice_attention_grid(
            volume_3d    = results['vol_full'],
            heatmap_3d   = hm_smooth,
            beta_full    = results['beta_full'],
            save_path    = out_dir / f'{vol_name}_slice_attn.png',
            patch_size   = patch_size,
            stride       = stride,
            beta_thresh  = args.beta_thresh,
            alpha_thresh = args.attn_thresh_3d,
        )

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
