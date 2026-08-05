#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from scipy.special import expit as sigmoid_np

_TRAIN_SCRIPT = Path(__file__).parent / 'train_hexmil.py'
_spec = importlib.util.spec_from_file_location('_train_hexmil', _TRAIN_SCRIPT)
_tm   = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tm)

DATA_DIR  = _tm.DATA_DIR
WORK_DIR  = _tm.WORK_DIR
ALL_FAKES = _tm.ALL_FAKES

_VIS_COL_MODS              = _tm._VIS_COL_MODS
_VIS_TYS                   = _tm._VIS_TYS
_build_volume_3d           = _tm._build_volume_3d
_build_heatmap_3d          = _tm._build_heatmap_3d
_smooth_3d                 = _tm._smooth_3d
_save_volume_gif           = _tm._save_volume_gif
_save_combined_gif         = _tm._save_combined_gif
_norm01                    = _tm._norm01
_smooth_attn               = _tm._smooth_attn
_get_ty                    = _tm._get_ty
compute_xai_metrics        = _tm.compute_xai_metrics
save_nodule_grid           = _tm.save_nodule_grid
_balanced_mod_metrics      = _tm._balanced_mod_metrics
_pd_at_1pct                = _tm._pd_at_1pct
run_test_evaluation_volume = _tm.run_test_evaluation_volume
select_best_gpu            = _tm.select_best_gpu

from hexmil.data.patch_dataset    import load_split_table
from hexmil.data.volume_dataset   import VolumeDataset
from hexmil.data.slice_dataset    import reconstruct_heatmap, build_patch_grid
from hexmil.models.hexmil import HexMIL, build_hexmil
from hexmil.utils.tiff_utils import (
    get_shape_tiff_scan, load_slice_tiff_scan,
    get_percentile_tiff_scan, apply_percentile,
)

_N_CAND_GRID = 40


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Stage 2 evaluation')
    p.add_argument('--run_dir',     type=str, required=True,
                   help='Path to the Stage 2 run directory')
    p.add_argument('--save_vis',    action='store_true', default=True,
                   help='Save individual per-volume visualisations')
    p.add_argument('--max_vis',     type=int, default=50)
    p.add_argument('--gpu_id',      type=int, default=None)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--full_volume', action=argparse.BooleanOptionalAction, default=True,
                   help='Evaluate on full volumes (sliding K-slice windows). Default: on; '
                        'pass --no-full_volume for pre-chunked VolumeDataset blocks.')
    p.add_argument('--win_stride', type=int, default=None,
                   help='Step between window starts in slices '
                        '(default: K = non-overlapping). '
                        'Use K//2 for 50%% overlap.')
    p.add_argument('--beta_thresh', type=float, default=0.2,
                   help='Zero out slices with β_k < beta_thresh from the 3D heatmap '
                        '(with K=16, uniform β≈0.0625)')
    p.add_argument('--save_3d',        action='store_true',
                   help='Save triplanar MIP + 3D scatter projection for summary samples')
    p.add_argument('--attn_thresh_3d', type=float, default=0.3,
                   help='Attention threshold for 3D scatter voxel display')
    p.add_argument('--save_nifti', action='store_true',
                   help='Export volume + attention heatmap as .nii.gz (for 3D Slicer / ITK-SNAP)')
    return p.parse_args()


def compute_bbox_iou_3d(
    heatmap: np.ndarray,   # (K, H, W) float32
    mask:    np.ndarray,   # (K, H, W) float32
    thresh:  float = 0.3,
) -> float:
    az, ay, ax = np.where(heatmap > thresh)
    mz, my, mx = np.where(mask > 0.5)
    if len(az) == 0 or len(mz) == 0:
        return float('nan')
    az0, az1 = int(az.min()), int(az.max())
    ay0, ay1 = int(ay.min()), int(ay.max())
    ax0, ax1 = int(ax.min()), int(ax.max())
    mz0, mz1 = int(mz.min()), int(mz.max())
    my0, my1 = int(my.min()), int(my.max())
    mx0, mx1 = int(mx.min()), int(mx.max())
    iz = max(0, min(az1, mz1) - max(az0, mz0) + 1)
    iy = max(0, min(ay1, my1) - max(ay0, my0) + 1)
    ix = max(0, min(ax1, mx1) - max(ax0, mx0) + 1)
    inter = iz * iy * ix
    vol_a = (az1 - az0 + 1) * (ay1 - ay0 + 1) * (ax1 - ax0 + 1)
    vol_m = (mz1 - mz0 + 1) * (my1 - my0 + 1) * (mx1 - mx0 + 1)
    union = vol_a + vol_m - inter
    return float(inter / union) if union > 0 else 0.0


def save_volume_3d_projection(
    volume_3d:   np.ndarray,
    heatmap_3d:  np.ndarray,
    masks_3d:    np.ndarray,
    label:       int,
    prob:        float,
    mod:         str,
    img_id:      str,
    save_path:   Path,
    attn_thresh: float = 0.3,
) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import matplotlib.patches as mpatches
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        return

    gt_str = 'real' if label == 0 else mod
    pr_str = 'FAKE' if prob > 0.5 else 'REAL'
    hm     = np.clip(heatmap_3d, 0.0, 1.0)

    def _mean_proj(v): return v.mean(axis=0), v.mean(axis=1), v.mean(axis=2)
    def _mip(v):       return v.max(axis=0),  v.max(axis=1),  v.max(axis=2)

    ct_ax, ct_co, ct_sa = _mean_proj(_norm01(volume_3d))
    hm_ax, hm_co, hm_sa = _mip(hm)

    def _bb(m2d):
        r = np.any(m2d, axis=1); c = np.any(m2d, axis=0)
        if not r.any() or not c.any(): return None
        rmin, rmax = np.where(r)[0][[0, -1]]
        cmin, cmax = np.where(c)[0][[0, -1]]
        return int(rmin), int(rmax), int(cmin), int(cmax)

    bboxes_dict = {}
    if label > 0:
        bboxes_dict = {
            'axial':    _bb((masks_3d.max(axis=0) > 0.5)),
            'coronal':  _bb((masks_3d.max(axis=1) > 0.5)),
            'sagittal': _bb((masks_3d.max(axis=2) > 0.5)),
        }

    def _draw_bb(ax, bb):
        if bb is None: return
        r0, r1, c0, c1 = bb
        ax.add_patch(mpatches.Rectangle(
            (c0, r0), c1 - c0, r1 - r0,
            lw=1.5, edgecolor='lime', facecolor='none',
        ))

    fig = plt.figure(figsize=(20, 9), facecolor='#111111')
    fig.suptitle(
        f'{img_id}  |  {mod}  GT:{gt_str}  Pred:{pr_str}  p={prob:.3f}',
        fontsize=18, fontweight='bold', color='white', y=0.99,
    )
    gs = gridspec.GridSpec(
        2, 4, figure=fig, wspace=0.06, hspace=0.06,
        left=0.03, right=0.97, top=0.93, bottom=0.02,
        width_ratios=[1, 1, 1, 1.3],
    )

    view_labels = ['Axial (z→)', 'Coronal (y→)', 'Sagittal (x→)']
    ct_imgs = [ct_ax, ct_co, ct_sa]
    hm_imgs = [hm_ax, hm_co, hm_sa]
    bb_keys = ['axial', 'coronal', 'sagittal']

    for ci in range(3):
        ax0 = fig.add_subplot(gs[0, ci])
        ax0.imshow(ct_imgs[ci], cmap='gray', origin='upper', aspect='auto')
        _draw_bb(ax0, bboxes_dict.get(bb_keys[ci]))
        ax0.set_title(view_labels[ci], fontsize=13, color='#aaaaaa', pad=2)
        ax0.axis('off')

        ax1 = fig.add_subplot(gs[1, ci])
        ax1.imshow(ct_imgs[ci], cmap='gray', origin='upper', aspect='auto')
        im = ax1.imshow(hm_imgs[ci], cmap='turbo', alpha=0.55,
                        vmin=0, vmax=1, origin='upper', aspect='auto')
        _draw_bb(ax1, bboxes_dict.get(bb_keys[ci]))
        ax1.axis('off')

    cbar_ax = fig.add_axes([0.695, 0.08, 0.008, 0.38])
    fig.colorbar(im, cax=cbar_ax, label='Attention')
    cbar_ax.yaxis.label.set_color('white')
    cbar_ax.tick_params(colors='white', labelsize=6)

    for ri, lbl in enumerate(['CT (mean proj)', 'CT + Attention']):
        fig.text(0.01, 0.73 - ri * 0.47, lbl,
                 va='center', ha='left', fontsize=11, color='white', rotation=90)

    K, H, W = hm.shape
    ax3d = fig.add_subplot(gs[:, 3], projection='3d')
    ax3d.set_facecolor('#111111')
    ax3d.set_xlabel('X', fontsize=11, color='white')
    ax3d.set_ylabel('Y', fontsize=11, color='white')
    ax3d.set_zlabel('Z', fontsize=11, color='white')
    ax3d.tick_params(colors='white', labelsize=5)
    ax3d.set_title('3-D attention scatter', fontsize=13, color='white', pad=4)
    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('#333333')

    zz, yy, xx = np.where(hm > attn_thresh)
    if len(zz) > 0:
        vals = hm[zz, yy, xx]
        ax3d.scatter(xx, yy, zz, c=vals, cmap='turbo',
                     vmin=attn_thresh, vmax=1.0, s=2.5, alpha=0.6, linewidths=0)

    if label > 0:
        mz, my, mx = np.where(masks_3d > 0.5)
        if len(mz) > 0:
            ax3d.scatter(mx, my, mz, c='lime', s=6, alpha=0.35,
                         linewidths=0, label='GT mask')
            ax3d.legend(fontsize=11, facecolor='#222222', labelcolor='white',
                        loc='upper left', markerscale=2)

    ax3d.set_xlim(0, W); ax3d.set_ylim(0, H); ax3d.set_zlim(0, K)
    ax3d.invert_yaxis()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor='#111111')
    plt.close(fig)


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


def load_full_volume_windows(
    mod:        str,
    img_id:     str,
    K:          int,
    patch_size: int,
    stride:     int,
    win_stride: int | None = None,
) -> dict:
    scan_dir      = f'{DATA_DIR}/{mod}/scan/{img_id}'
    shape         = get_shape_tiff_scan(scan_dir)
    Z_total, H, W = shape
    low, high     = get_percentile_tiff_scan(scan_dir, np.uint16)

    label_dir = f'{DATA_DIR}/{mod}/label/{img_id}' if mod != 'real' else None

    scan_full = load_slice_tiff_scan(scan_dir, shape, np.uint16, 0, Z_total)
    mask_full = (
        load_slice_tiff_scan(label_dir, shape, np.bool_, 0, Z_total).astype(np.float32)
        if label_dir else np.zeros((Z_total, H, W), dtype=np.float32)
    )

    _, _, grid_hw = build_patch_grid(np.zeros((H, W), dtype=np.float32), patch_size, stride)
    n_rows, n_cols = grid_hw
    N              = n_rows * n_cols
    _win_stride    = win_stride if win_stride is not None else K

    windows = []
    for z_start in range(0, Z_total, _win_stride):
        z_end       = z_start + K
        patches_out = torch.zeros(K, N, 1, patch_size, patch_size, dtype=torch.float32)
        z_indices   = torch.full((K,), -1, dtype=torch.long)
        masks_3d    = np.zeros((K, H, W), dtype=np.float32)

        avail_end = min(z_end, Z_total)
        for local_z in range(avail_end - z_start):
            global_z = z_start + local_z
            sl = apply_percentile(scan_full[global_z].astype(np.float32), low, high)
            patches_np, _, _ = build_patch_grid(sl, patch_size, stride)
            patches_out[local_z] = torch.from_numpy(patches_np).unsqueeze(1).float()
            z_indices[local_z]   = global_z
            masks_3d[local_z]    = mask_full[global_z]

        windows.append({
            'patches':    patches_out,
            'z_indices':  z_indices,
            'valid_mask': z_indices >= 0,
            'masks_3d':   masks_3d,
            'z_start':    z_start,
            'z_end':      min(z_end, Z_total),
        })

    return {
        'windows': windows,
        'meta': {
            'img_id':     img_id,
            'mod':        mod,
            'Z_total':    Z_total,
            'H':          H,
            'W':          W,
            'n_windows':  len(windows),
            'grid_hw':    grid_hw,
            'win_stride': _win_stride,
        },
        'scan_full': scan_full,
        'mask_full': mask_full,
        'low':       low,
        'high':      high,
    }


@torch.no_grad()
def _run_window(
    model:       HexMIL,
    window:      dict,
    device:      torch.device,
    patch_size:  int,
    stride:      int,
    H:           int,
    W:           int,
    grid_hw:     tuple,
    beta_thresh: float,
) -> dict:
    patches_t = window['patches'].to(device)
    z_t       = window['z_indices'].to(device)
    valid_t   = window['valid_mask'].to(device)

    logit, attn_tup    = model(patches_t, z_t, valid_t, return_attn=True)
    beta_t, alpha_list = attn_tup
    beta_np   = beta_t.cpu().float().numpy()
    alpha_cpu = [a.cpu().float().numpy() for a in alpha_list]
    valid_np  = window['valid_mask'].numpy()
    K_eff     = patches_t.shape[0]

    gh   = grid_hw
    sh   = (H, W)
    hmap = np.zeros((K_eff, H, W), dtype=np.float32)
    for k in range(K_eff):
        if not valid_np[k]:
            continue
        if float(beta_np[k]) < beta_thresh:
            continue
        a2d = reconstruct_heatmap(alpha_cpu[k], gh, sh, patch_size, stride)
        hmap[k] = np.clip(a2d * float(beta_np[k]), 0, 1)

    gh_arr  = np.tile(np.array(gh), (K_eff, 1))
    vol_win = _build_volume_3d(window['patches'], gh_arr, sh, patch_size, stride, valid_np)

    prob = float(sigmoid_np(logit.item()))
    return {
        'z_start':    window['z_start'],
        'z_end':      window['z_end'],
        'prob':       prob,
        'beta':       beta_np,
        'alpha_cpu':  alpha_cpu,
        'heatmap_3d': hmap,
        'volume_3d':  vol_win,
        'masks_3d':   window['masks_3d'],
        'valid_mask': valid_np,
        'z_indices':  window['z_indices'].numpy(),
    }


@torch.no_grad()
def run_full_volume_eval(
    model:       HexMIL,
    tab:         pd.DataFrame,
    device:      torch.device,
    patch_size:  int,
    stride:      int,
    K:           int,
    save_vis:    bool,
    vis_dir:     Path | None,
    max_vis:     int,
    beta_thresh: float = 0.0,
    win_stride:  int | None = None,
) -> dict:
    model.eval()

    all_scores:  list = []
    all_labels:  list = []
    all_mods:    list = []
    all_img_ids: list = []
    all_xai:     list = []
    per_sample_records: list = []

    grid_cands: dict = {ty: {m: [] for m in _VIS_COL_MODS} for ty in _VIS_TYS}
    vis_count = 0

    for _, row in tqdm(tab.iterrows(), total=len(tab),
                       desc='Full-volume eval', unit='vol', dynamic_ncols=True):
        mod     = row['mod']
        img_id  = str(row['img_id'])
        coord_z = int(row['coord_z']) if mod != 'real' else None
        gt      = 0 if mod == 'real' else 1

        try:
            data = load_full_volume_windows(mod, img_id, K, patch_size, stride,
                                            win_stride=win_stride)
        except Exception as exc:
            print(f'  [SKIP] {mod}/{img_id}: {exc}')
            continue

        meta          = data['meta']
        Z_total, H, W = meta['Z_total'], meta['H'], meta['W']
        grid_hw       = meta['grid_hw']
        mask_full     = data['mask_full']

        win_results = [
            _run_window(model, win, device, patch_size, stride, H, W, grid_hw, beta_thresh)
            for win in data['windows']
        ]

        vol_score = max(r['prob'] for r in win_results)
        pred      = 1 if vol_score > 0.5 else 0

        # CT intensities: last-write is fine (same data for every window).
        # Heatmap and beta: accumulate then average over all windows that cover
        # each slice — with no overlap this reduces to a single write;
        # with overlap each slice gets the consensus across its windows.
        vol_full   = np.zeros((Z_total, H, W), dtype=np.float32)
        hm_accum   = np.zeros((Z_total, H, W), dtype=np.float32)
        beta_accum = np.zeros(Z_total, dtype=np.float32)
        count_full = np.zeros(Z_total, dtype=np.float32)
        valid_full = np.zeros(Z_total, dtype=bool)

        for res in win_results:
            z0, z1 = res['z_start'], res['z_end']
            k_eff  = z1 - z0
            vol_full[z0:z1] = res['volume_3d'][:k_eff]
            for lk in range(k_eff):
                if res['valid_mask'][lk]:
                    gz = z0 + lk
                    hm_accum[gz]   += res['heatmap_3d'][lk]
                    beta_accum[gz] += res['beta'][lk]
                    count_full[gz] += 1.0
                    valid_full[gz]  = True

        cnt        = np.maximum(count_full, 1.0)
        hm_full    = hm_accum   / cnt[:, None, None]
        beta_full  = beta_accum / cnt

        z_indices_full = np.arange(Z_total)

        # Use hm_full[coord_z] which already averages across all windows
        # that covered that slice (works for both overlap and no-overlap).
        xai_keys_blank = ['pixel_auc', 'iou_03', 'iou_05', 'iou_07',
                          'pointing_game']
        xai_row = {k: float('nan') for k in xai_keys_blank}

        if gt == 1 and coord_z is not None and valid_full[coord_z]:
            mask_2d = mask_full[coord_z]
            hm_2d   = hm_full[coord_z]
            if mask_2d.sum() > 0:
                xai = compute_xai_metrics(_smooth_attn(hm_2d), mask_2d)
                xai['img_id'] = img_id
                xai['mod']    = mod
                all_xai.append(xai)
                xai_row = {k: xai.get(k, float('nan')) for k in xai_keys_blank}

        per_sample_records.append({
            'img_id':           img_id,
            'mod':              mod,
            'label':            gt,
            'prob':             round(vol_score, 6),
            'pred':             pred,
            'correct':          int(gt == pred),
            'coord_z':          coord_z,
            'n_windows':        len(win_results),
            **xai_row,
        })

        all_scores.append(vol_score)
        all_labels.append(gt)
        all_mods.append(mod)
        all_img_ids.append(img_id)

        if save_vis and vis_dir is not None and vis_count < max_vis:
            _save_volume_gif(
                volume_3d  = vol_full,
                heatmap_3d = _smooth_3d(hm_full),
                masks_3d   = mask_full,
                beta_np    = beta_full,
                z_indices  = z_indices_full,
                valid_mask = valid_full,
                label      = gt,
                prob       = vol_score,
                mod        = mod,
                img_id     = img_id,
                save_path  = vis_dir / f'{img_id.replace("/", "_")}_{mod}.gif',
            )
            vis_count += 1

        ty_grid = _get_ty(img_id)
        if mod in _VIS_COL_MODS and len(grid_cands[ty_grid][mod]) < _N_CAND_GRID:
            ctr_z   = coord_z if coord_z is not None else Z_total // 2
            k_start = max(0, ctr_z - K // 2)
            k_end   = min(Z_total, k_start + K)
            k_start = max(0, k_end - K)
            kk      = k_end - k_start
            coord_z_local = int(coord_z - k_start) if coord_z is not None else K // 2
            coord_z_local = int(np.clip(coord_z_local, 0, K - 1))

            vol_sub  = vol_full[k_start:k_end]
            hm_sub   = hm_full[k_start:k_end]
            mk_sub   = mask_full[k_start:k_end]
            beta_sub = beta_full[k_start:k_end]
            z_sub    = z_indices_full[k_start:k_end]
            val_sub  = valid_full[k_start:k_end]

            if kk < K:
                pad     = K - kk
                vol_sub  = np.pad(vol_sub,  ((0, pad), (0, 0), (0, 0)))
                hm_sub   = np.pad(hm_sub,   ((0, pad), (0, 0), (0, 0)))
                mk_sub   = np.pad(mk_sub,   ((0, pad), (0, 0), (0, 0)))
                beta_sub = np.pad(beta_sub, (0, pad))
                z_sub    = np.pad(z_sub,    (0, pad), constant_values=-1)
                val_sub  = np.pad(val_sub,  (0, pad), constant_values=False)

            grid_cands[ty_grid][mod].append({
                'volume_3d':     vol_sub,
                'heatmap_3d':    hm_sub,
                'masks_3d':      mk_sub,
                'beta':          beta_sub,
                'z_indices':     z_sub,
                'valid_mask':    val_sub,
                'label':         gt,
                'prob':          vol_score,
                'mod':           mod,
                'img_id':        img_id,
                'coord_z_local': coord_z_local,
                'ctr_z':         ctr_z if coord_z is not None else None,
                'k_start':       k_start,
            })

    grid_samples: dict = {'removal': {}, 'injection': {}}
    for ty in _VIS_TYS:
        id_sets = [{s['img_id'] for s in grid_cands[ty][m]}
                   for m in _VIS_COL_MODS if grid_cands[ty][m]]
        shared  = id_sets[0].intersection(*id_sets[1:]) if len(id_sets) > 1 else \
                  (id_sets[0] if id_sets else set())
        anchor  = next(iter(shared)) if shared else None
        for mod in _VIS_COL_MODS:
            clist = grid_cands[ty][mod]
            if not clist:
                continue
            match = next((s for s in clist if s['img_id'] == anchor), None) \
                    if anchor else None
            grid_samples[ty][mod] = match if match else clist[0]

        # Align real's coord_z_local to the first fake's ctr_z
        fake_ctr_z = None
        for m in _VIS_COL_MODS:
            if m != 'real':
                s_fake = grid_samples[ty].get(m)
                if s_fake is not None and s_fake.get('ctr_z') is not None:
                    fake_ctr_z = s_fake['ctr_z']
                    break
        if fake_ctr_z is not None:
            s_real = grid_samples[ty].get('real')
            if s_real is not None:
                k_real = s_real.get('k_start', 0)
                local  = int(np.clip(fake_ctr_z - k_real, 0,
                                     s_real['volume_3d'].shape[0] - 1))
                s_real['coord_z_local'] = local

    scores_np = np.array(all_scores)
    labels_np = np.array(all_labels)
    preds_np  = (scores_np >= 0.5).astype(int)
    mods_arr  = np.array(all_mods)

    cls_metrics: dict = {
        'auc':      float(roc_auc_score(labels_np, scores_np)),
        'accuracy': float(accuracy_score(labels_np, preds_np)),
        'f1':       float(f1_score(labels_np, preds_np, zero_division=0)),
        'pd_at_1':  _pd_at_1pct(labels_np, scores_np),
    }
    for mod in sorted(m for m in set(all_mods) if m != 'real'):
        for k, v in _balanced_mod_metrics(labels_np, scores_np, preds_np, mods_arr, mod).items():
            cls_metrics[f'{mod}_{k}'] = v

    xai_metrics: dict = {}
    if all_xai:
        xai_keys_base = [k for k in all_xai[0] if k not in ('img_id', 'mod')]
        for k in xai_keys_base:
            vals = [x[k] for x in all_xai if not np.isnan(float(x[k]))]
            xai_metrics[k] = float(np.mean(vals)) if vals else float('nan')
        by_mod: dict = {}
        for x in all_xai:
            by_mod.setdefault(x['mod'], []).append(x)
        for mod, xs in by_mod.items():
            for k in xai_keys_base:
                vals = [x[k] for x in xs if not np.isnan(float(x[k]))]
                xai_metrics[f'{k}_{mod}'] = float(np.mean(vals)) if vals else float('nan')

    return {
        'split':              'test',
        'cls':                cls_metrics,
        'xai':                xai_metrics,
        'grid_samples':       grid_samples,
        'per_sample_records': per_sample_records,
    }


def main() -> None:
    args = get_args()

    run_dir   = Path(args.run_dir)
    eval_dir  = run_dir / ('evaluation_full' if args.full_volume else 'evaluation')
    ckpt_path = run_dir / 'best_model.pt'
    args_path = run_dir / 'args.json'

    if not ckpt_path.exists():
        raise FileNotFoundError(f"No best_model.pt in {run_dir}")
    if not args_path.exists():
        raise FileNotFoundError(f"No args.json in {run_dir}")

    saved_args = json.loads(args_path.read_text())
    K          = saved_args['K']
    sargs      = saved_args.get('slice_args', saved_args)
    patch_size = sargs['patch_size']
    stride     = sargs.get('stride') or (patch_size // 2)

    if args.gpu_id is not None:
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        gid    = select_best_gpu()
        device = torch.device(f'cuda:{gid}') if gid is not None else torch.device('cpu')

    print(f"\n{'='*60}")
    print(f"  Stage 2 Evaluation")
    print(f"{'='*60}")
    print(f"  Run:    {run_dir}")
    print(f"  Mode:   {'full_volume' if args.full_volume else 'block'}")
    print(f"  Device: {device}")
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
            backbone   = sargs.get('backbone', 'resnet50'),
            pretrained = False,
            proj_dim   = sargs.get('proj_dim', 512),
            attn_dim   = sargs.get('attn_dim', 256),
            dropout    = sargs.get('dropout',  0.25),
        )
        model = HexMIL(
            slice_encoder = slice_model,
            feat_dim      = slice_model.feat_dim,
            K             = K,
            attn_dim      = saved_args.get('attn_dim', 256),
            dropout       = saved_args.get('dropout', 0.25),
        ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=True)
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
          f"(val_loss={ckpt.get('best_val_loss', float('nan')):.4f})\n")

    tab = load_split_table(DATA_DIR, 'test', ['real'] + ALL_FAKES)
    print(f"Evaluating {len(tab)} volumes on real + ALL_FAKES (test split)\n")

    eval_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = eval_dir / 'per_volume' if args.save_vis else None

    if args.full_volume:
        results = run_full_volume_eval(
            model       = model,
            tab         = tab,
            device      = device,
            patch_size  = patch_size,
            stride      = stride,
            K           = K,
            save_vis    = args.save_vis,
            vis_dir     = vis_dir,
            max_vis     = args.max_vis,
            beta_thresh = args.beta_thresh,
            win_stride  = args.win_stride,
        )
        grid_samples       = results.pop('grid_samples')
        per_sample_records = results.pop('per_sample_records')
    else:
        ds = VolumeDataset(
            DATA_DIR, tab, K=K, patch_size=patch_size, stride=stride, augment=False,
        )
        dl = torch.utils.data.DataLoader(
            ds, batch_size=1, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )
        results = run_test_evaluation_volume(
            model           = model,
            dl_test         = dl,
            device          = device,
            eval_dir        = eval_dir,
            epoch           = int(ckpt.get('epoch', 0)),
            run_dir         = run_dir,
            sargs           = sargs,
            in_domain_fakes = set(ALL_FAKES),
        )
        # Normal mode: run_test_evaluation_volume already saved metrics.json,
        # per_sample.csv, GIFs, and nodule grids.
        print("\nClassification metrics:")
        for k, v in sorted(results['cls'].items()):
            if isinstance(v, float):
                print(f"  {k:30s}: {v:.4f}")
        if results.get('xai'):
            print("\nXAI metrics (annotated slice):")
            for k in ['pixel_auc', 'iou_03', 'iou_05', 'iou_07',
                      'pointing_game']:
                v = results['xai'].get(k, float('nan'))
                if isinstance(v, float):
                    print(f"  {k:30s}: {v:.4f}")
        print(f"\nAll outputs written to: {eval_dir}")
        print("Done.")
        return

    cls_m = results['cls']
    xai_m = results.get('xai', {})

    print("\nClassification metrics:")
    for k, v in sorted(cls_m.items()):
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4f}")
    if xai_m:
        print("\nXAI metrics:")
        for k in ['pixel_auc', 'iou_03', 'iou_05', 'iou_07',
                  'pointing_game']:
            v = xai_m.get(k, float('nan'))
            if isinstance(v, float):
                print(f"  {k:30s}: {v:.4f}")

    with open(eval_dir / 'metrics.json', 'w') as f:
        json.dump({'cls': cls_m, 'xai': xai_m,
                   'split': 'test', 'run_dir': str(run_dir)}, f, indent=2, default=str)
    print(f"  Metrics JSON  → {eval_dir / 'metrics.json'}")

    if per_sample_records:
        pd.DataFrame(per_sample_records).to_csv(eval_dir / 'per_sample.csv', index=False)
        print(f"  Per-sample CSV → {eval_dir / 'per_sample.csv'}")

    gif_dir = eval_dir / 'gifs'
    for ty in _VIS_TYS:
        for mod, s in grid_samples.get(ty, {}).items():
            _save_volume_gif(
                volume_3d  = s['volume_3d'],
                heatmap_3d = _smooth_3d(s['heatmap_3d']),
                masks_3d   = s['masks_3d'],
                beta_np    = s['beta'],
                z_indices  = s['z_indices'],
                valid_mask = s['valid_mask'],
                label      = s['label'],
                prob       = s['prob'],
                mod        = s['mod'],
                img_id     = s['img_id'],
                save_path  = gif_dir / f'{ty}_{mod}.gif',
            )
            print(f"  → GIF:  {gif_dir}/{ty}_{mod}.gif")
    for ty in _VIS_TYS:
        if grid_samples.get(ty):
            _save_combined_gif(
                grid_samples_ty = grid_samples[ty],
                in_domain_fakes = set(ALL_FAKES),
                save_path       = gif_dir / f'{ty}_combined.gif',
            )
            print(f"  → Combined GIF: {gif_dir}/{ty}_combined.gif")

    if args.save_3d:
        proj_dir = eval_dir / 'projections_3d'
        for ty in _VIS_TYS:
            for mod, s in grid_samples.get(ty, {}).items():
                save_volume_3d_projection(
                    volume_3d   = s['volume_3d'],
                    heatmap_3d  = _smooth_3d(s['heatmap_3d']),
                    masks_3d    = s['masks_3d'],
                    label       = s['label'],
                    prob        = s['prob'],
                    mod         = s['mod'],
                    img_id      = s['img_id'],
                    save_path   = proj_dir / f'{ty}_{mod}_3d.png',
                    attn_thresh = args.attn_thresh_3d,
                )
                print(f"  → 3D:  {proj_dir}/{ty}_{mod}_3d.png")

    if args.save_nifti:
        nii_dir = eval_dir / 'nifti'
        for ty in _VIS_TYS:
            for mod, s in grid_samples.get(ty, {}).items():
                save_as_nifti(
                    volume_3d  = s['volume_3d'],
                    heatmap_3d = _smooth_3d(s['heatmap_3d']),
                    out_dir    = nii_dir,
                    prefix     = f'{ty}_{mod}',
                )
                print(f"  → NII: {nii_dir}/{ty}_{mod}_volume.nii.gz")

    for ty in _VIS_TYS:
        if grid_samples.get(ty):
            save_nodule_grid(
                grid_samples[ty], eval_dir,
                filename=f'nodule_{ty}.png',
                in_domain_fakes=set(ALL_FAKES),
            )
            print(f"  → Grid: {eval_dir}/nodule_{ty}.png")

    print(f"\nAll outputs written to: {eval_dir}")
    print("Done.")


if __name__ == '__main__':
    main()
