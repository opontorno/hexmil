#!/usr/bin/env python3
"""
benchmark_xai_3d.py
===================
Quantitative 3D XAI benchmark for Phase C (volume-level) checkpoints.

The script evaluates explanation methods on full 3D volumes and compares
localisation quality against GT 3D masks.

Default methods:
  - abmil                  (ante-hoc: beta * alpha reconstructed in 3D)
  - grad_feat_alpha        (post-hoc: gradient slice importance x alpha map)
  - slice_occ_uniform      (post-hoc: leave-one-slice-out occlusion, uniform slice map)
  - random3d               (sanity baseline)
  - center_prior3d         (sanity baseline)

Outputs (out_dir):
  - per_volume_metrics.csv
  - summary_global.csv
  - summary_by_mod.csv
  - summary_by_type.csv
    - pairwise_tests.csv
    - figures/*.{png,pdf,svg}

Usage example:
  python experiments/ABMIL/benchmark_xai_3d.py \
      --run_dir experiments/ABMIL/runs/volume-cls_resnet50_p64_s32_K16/trained_on_all \
      --methods abmil grad_feat_alpha slice_occ_uniform random3d center_prior3d \
      --max_volumes_per_mod 40 --bootstrap_iters 1000
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter
from scipy.special import expit as sigmoid_np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# Import eval_volume-cls.py so we reuse the exact same full-volume pipeline.
_EVAL_PY = Path(__file__).resolve().parent / 'eval_volume.py'
if not _EVAL_PY.exists():
    sys.exit(f'[ERROR] eval_volume.py not found at {_EVAL_PY}')

_spec = importlib.util.spec_from_file_location('eval_volume_cls_bench3d', _EVAL_PY)
_ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ev)

DATA_DIR = _ev.DATA_DIR
ALL_FAKES = _ev.ALL_FAKES
load_split_table = _ev.load_split_table
load_full_volume_windows = _ev.load_full_volume_windows
_run_window = _ev._run_window
compute_bbox_iou_3d = _ev.compute_bbox_iou_3d
reconstruct_heatmap = _ev.reconstruct_heatmap
build_volume_classifier = _ev.build_volume_classifier
VolumeClassifier = _ev.VolumeClassifier
select_best_gpu = _ev.select_best_gpu


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='3D XAI benchmark for ABMIL volume model')
    p.add_argument('--run_dir', type=str, required=True,
                   help='Path to a Phase C run directory (best_model.pt + args.json)')
    p.add_argument('--out_dir', type=str, default=None,
                   help='Output directory (default: <run_dir>/evaluation_xai3d_benchmark)')
    p.add_argument('--gpu_id', type=int, default=None)
    p.add_argument('--seed', type=int, default=42)

    p.add_argument('--mods', nargs='+', default=None,
                   help='Fake modalities to evaluate (default: all fakes)')
    p.add_argument('--max_volumes_per_mod', type=int, default=0,
                   help='Maximum volumes per modality (0 = all)')

    p.add_argument('--methods', nargs='+',
                   default=['abmil', 'grad_feat_alpha', 'slice_occ_uniform', 'random3d', 'center_prior3d'],
                   choices=[
                       'abmil',
                       'beta_uniform',
                       'grad_feat_uniform', 'grad_feat_alpha',
                       'slice_occ_uniform', 'slice_occ_alpha',
                       'random3d', 'center_prior3d',
                   ])

    p.add_argument('--beta_thresh', type=float, default=0.0,
                   help='Threshold on beta_k for abmil method (same semantics as eval_volume-cls)')
    p.add_argument('--smooth_sigma_xy_frac', type=float, default=0.03,
                   help='Per-slice Gaussian sigma fraction over max(H,W)')
    p.add_argument('--smooth_sigma_z', type=float, default=0.0,
                   help='Optional Gaussian sigma on z-axis for 3D maps (0 disables)')

    p.add_argument('--bootstrap_iters', type=int, default=1000,
                   help='Bootstrap iterations for confidence intervals (0 disables)')
    p.add_argument('--ci_alpha', type=float, default=0.95,
                   help='Confidence level for bootstrap CI')

    p.add_argument('--extended_stats', action=argparse.BooleanOptionalAction, default=True,
                   help='Compute extended descriptive statistics in summaries')
    p.add_argument('--save_pairwise_tests', action=argparse.BooleanOptionalAction, default=True,
                   help='Save pairwise Wilcoxon tests across methods')

    p.add_argument('--save_figures', action=argparse.BooleanOptionalAction, default=True,
                   help='Generate paper-ready figures')
    p.add_argument('--figure_dpi', type=int, default=300)
    p.add_argument('--figure_formats', nargs='+', default=['png', 'pdf'],
                   choices=['png', 'pdf', 'svg'],
                   help='Output figure formats')
    p.add_argument('--primary_metric', type=str, default='voxel_auc',
                   choices=[
                       'voxel_auc',
                       'iou3d_03', 'iou3d_05', 'iou3d_07',
                       'pointing_game_3d',
                       'energy_in_mask_3d',
                       'bbox_iou3d_03', 'bbox_iou3d_05',
                   ],
                   help='Primary metric used for ranking plots')

    # Qualitative visual examples (enabled by default).
    p.add_argument('--save_visual_examples', action=argparse.BooleanOptionalAction, default=True,
                   help='Save representative slice comparisons across methods')
    p.add_argument('--n_visual_examples_per_mod', type=int, default=3,
                   help='How many qualitative volume examples to save per modality')
    p.add_argument('--visual_overlay_alpha', type=float, default=0.50,
                   help='Heatmap overlay alpha for qualitative figures')
    p.add_argument('--qualitative_only', action=argparse.BooleanOptionalAction, default=False,
                   help='Generate only qualitative visual examples (skip metrics/statistics)')
    return p.parse_args()


def _get_ty(img_id: str) -> str:
    return 'removal' if str(img_id).startswith('rem_') else 'injection'


def _norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo + 1e-8)


def _smooth_3d_map(hm: np.ndarray, sigma_xy_frac: float, sigma_z: float) -> np.ndarray:
    out = hm.astype(np.float32).copy()

    # Smooth each slice in-plane.
    if sigma_xy_frac > 0:
        for k in range(out.shape[0]):
            sl = out[k]
            if sl.max() > 0:
                sigma_xy = max(1.0, max(sl.shape) * sigma_xy_frac)
                out[k] = gaussian_filter(sl, sigma=sigma_xy)

    # Optional smoothing across z.
    if sigma_z > 0:
        out = gaussian_filter(out, sigma=(sigma_z, 0.0, 0.0))

    return _norm01(out)


def _voxel_metrics(hm_3d: np.ndarray, mask_3d: np.ndarray) -> dict[str, float]:
    m = (mask_3d > 0.5)
    hm = np.clip(hm_3d.astype(np.float32), 0.0, 1.0)
    res: dict[str, float] = {}

    # Voxel AUC.
    if m.sum() > 0 and (~m).sum() > 0:
        res['voxel_auc'] = float(roc_auc_score(m.ravel().astype(np.uint8), hm.ravel()))
    else:
        res['voxel_auc'] = float('nan')

    for thr in (0.3, 0.5, 0.7):
        h_bin = hm >= thr
        inter = float((h_bin & m).sum())
        union = float((h_bin | m).sum())
        res[f'iou3d_{int(thr * 10):02d}'] = inter / union if union > 0 else 0.0

    z, y, x = np.unravel_index(np.argmax(hm), hm.shape)
    res['pointing_game_3d'] = float(m[z, y, x])

    e_tot = float(hm.sum())
    e_in = float((hm * m.astype(np.float32)).sum())
    res['energy_in_mask_3d'] = e_in / e_tot if e_tot > 0 else float('nan')

    res['bbox_iou3d_03'] = float(compute_bbox_iou_3d(hm, m.astype(np.float32), thresh=0.3))
    res['bbox_iou3d_05'] = float(compute_bbox_iou_3d(hm, m.astype(np.float32), thresh=0.5))

    return res


def _center_prior_3d(shape: tuple[int, int, int]) -> np.ndarray:
    z, h, w = shape
    zz, yy, xx = np.mgrid[0:z, 0:h, 0:w].astype(np.float32)

    cz = (z - 1) / 2.0
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0

    sz = max(1.0, z * 0.2)
    sy = max(2.0, h * 0.2)
    sx = max(2.0, w * 0.2)

    g = np.exp(-(
        ((zz - cz) ** 2) / (2 * sz * sz)
        + ((yy - cy) ** 2) / (2 * sy * sy)
        + ((xx - cx) ** 2) / (2 * sx * sx)
    ))
    return _norm01(g)


def _window_slice_occlusion_weights(
    model: VolumeClassifier,
    window: dict,
    device: torch.device,
    base_prob: float,
) -> np.ndarray:
    """
    Post-hoc leave-one-slice-out importance for one K-slice window.
    Returns normalized weights over K slices (invalid slices are 0).

    Optimised: pre-encode all K slices once (O(K) encoder calls) then run
    only the cheap aggregator+head for each occlusion pass, instead of the
    original O(K²) full-model re-encoding.
    """
    patches = window['patches']   # CPU: (K, N, 1, P, P)
    z_idx = window['z_indices']
    valid = window['valid_mask']

    K = patches.shape[0]
    w = np.zeros(K, dtype=np.float32)

    with torch.no_grad():
        # 1. Encode all slices once.
        h_cache = []
        for k in range(K):
            if bool(valid[k]):
                h_k, _ = model._encode_slice(patches[k].to(device))
            else:
                h_k = torch.zeros(model.feat_dim, device=device,
                                  dtype=torch.float32)
            h_cache.append(h_k)

        # 2. Encode a blank slice once (reused for every occlusion pass).
        h_blank, _ = model._encode_slice(torch.zeros_like(patches[0]).to(device))

        # 3. Build base H = embeddings + positional encoding.
        pe = model.pos_enc(z_idx.clamp(min=0).to(device))   # (K, feat_dim)
        H_base = torch.stack(h_cache).unsqueeze(0) + pe.unsqueeze(0)  # (1, K, F)

        # 4. For each valid slice k, occlude it and measure score drop.
        h_blank_pe = h_blank  # will be shifted by pe[k] inside loop
        for k in range(K):
            if not bool(valid[k]):
                continue

            saved = H_base[0, k].clone()
            H_base[0, k] = h_blank_pe + pe[k]   # blank encoding + same PE
            v, _ = model.z_aggregator(H_base)
            logit_occ = model.head(v).squeeze()
            prob_occ = float(torch.sigmoid(logit_occ).item())
            w[k] = max(0.0, base_prob - prob_occ)
            H_base[0, k] = saved  # restore for next iteration

    s = float(w.sum())
    if s > 0:
        w = w / s
    return w


def _window_grad_feat_weights(
    model: VolumeClassifier,
    window: dict,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    """
    Post-hoc gradient slice importance from d(logit)/d(H_k), where H_k is
    the per-slice embedding before z-aggregator.

    Returns:
      - normalized non-negative weights over K slices
      - probability for this window from the gradient-path forward pass
    """
    patches = window['patches'].to(device)
    z_idx = window['z_indices'].to(device)
    valid = window['valid_mask'].to(device)

    K = patches.shape[0]
    feat_dim = model.feat_dim

    # Recreate per-slice embeddings using the frozen encoder.
    h_list = []
    for k in range(K):
        if bool(valid[k]):
            h_k, _ = model._encode_slice(patches[k])
        else:
            h_k = torch.zeros(feat_dim, device=device, dtype=patches.dtype)
        h_list.append(h_k)

    H0 = torch.stack(h_list, dim=0).unsqueeze(0).detach().requires_grad_(True)  # (1, K, F)

    z_clamp = z_idx.clamp(min=0)
    pe = model.pos_enc(z_clamp).unsqueeze(0)
    H = H0 + pe

    v, _beta = model.z_aggregator(H)
    logit = model.head(v.squeeze(0)).squeeze()

    model.zero_grad(set_to_none=True)
    if H0.grad is not None:
        H0.grad.zero_()
    logit.backward()

    grad = H0.grad.squeeze(0)  # (K, F)
    score = torch.norm(torch.relu(grad), p=2, dim=1) * valid.float()

    s = score.sum().clamp(min=1e-8)
    w = (score / s).detach().cpu().numpy().astype(np.float32)
    prob = float(torch.sigmoid(logit).detach().cpu().item())
    return w, prob


def _alpha_map(alpha_k: np.ndarray, grid_hw: tuple[int, int], slice_hw: tuple[int, int],
               patch_size: int, stride: int) -> np.ndarray:
    a2d = reconstruct_heatmap(alpha_k, grid_hw, slice_hw, patch_size, stride)
    return _norm01(a2d)


def _build_hm_from_window(
    method: str,
    res: dict,
    window: dict,
    model: VolumeClassifier,
    device: torch.device,
    grid_hw: tuple[int, int],
    slice_hw: tuple[int, int],
    patch_size: int,
    stride: int,
) -> np.ndarray:
    """Build method-specific heatmap for one window; returns (K, H, W)."""
    K = window['patches'].shape[0]
    H, W = slice_hw
    hm = np.zeros((K, H, W), dtype=np.float32)

    valid = res['valid_mask']
    beta = res['beta']
    alpha_list = res['alpha_cpu']

    if method == 'abmil':
        return res['heatmap_3d'].copy()

    if method == 'beta_uniform':
        for k in range(K):
            if valid[k]:
                hm[k] = float(beta[k])
        return hm

    if method in {'slice_occ_uniform', 'slice_occ_alpha'}:
        w_occ = _window_slice_occlusion_weights(model, window, device, base_prob=float(res['prob']))
        for k in range(K):
            if not valid[k]:
                continue
            if method == 'slice_occ_uniform':
                hm[k] = w_occ[k]
            else:
                hm[k] = w_occ[k] * _alpha_map(alpha_list[k], grid_hw, slice_hw, patch_size, stride)
        return hm

    if method in {'grad_feat_uniform', 'grad_feat_alpha'}:
        w_grad, _ = _window_grad_feat_weights(model, window, device)
        for k in range(K):
            if not valid[k]:
                continue
            if method == 'grad_feat_uniform':
                hm[k] = w_grad[k]
            else:
                hm[k] = w_grad[k] * _alpha_map(alpha_list[k], grid_hw, slice_hw, patch_size, stride)
        return hm

    raise ValueError(f'Unsupported method in window builder: {method}')


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, iters: int, alpha: float) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    n = values.size
    if n == 0:
        return float('nan'), float('nan')
    if n == 1 or iters <= 0:
        m = float(values.mean())
        return m, m

    means = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        idx = rng.integers(0, n, size=n)
        means[i] = values[idx].mean()
    q_lo = (1.0 - alpha) / 2.0
    q_hi = 1.0 - q_lo
    return float(np.quantile(means, q_lo)), float(np.quantile(means, q_hi))


def _extended_stats(arr: np.ndarray) -> dict[str, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            'median': float('nan'),
            'p25': float('nan'),
            'p75': float('nan'),
            'min': float('nan'),
            'max': float('nan'),
            'sem': float('nan'),
        }
    std = float(np.std(finite))
    sem = std / float(np.sqrt(max(1, finite.size)))
    return {
        'median': float(np.median(finite)),
        'p25': float(np.quantile(finite, 0.25)),
        'p75': float(np.quantile(finite, 0.75)),
        'min': float(np.min(finite)),
        'max': float(np.max(finite)),
        'sem': sem,
    }


def _bh_fdr(pvals: list[float]) -> list[float]:
    if not pvals:
        return []
    n = len(pvals)
    order = np.argsort(pvals)
    ordered = np.array([pvals[i] for i in order], dtype=np.float64)
    q = np.empty(n, dtype=np.float64)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = min(prev, ordered[i] * n / rank)
        q[i] = val
        prev = val
    out = np.empty(n, dtype=np.float64)
    for i, idx in enumerate(order):
        out[idx] = q[i]
    return [float(x) for x in out]


def _pairwise_tests(
    df: pd.DataFrame,
    metric_cols: list[str],
    sample_id_cols: list[str],
) -> pd.DataFrame:
    rows: list[dict] = []
    methods = sorted(df['method'].unique().tolist())

    for metric in metric_cols:
        pivot = df.pivot_table(index=sample_id_cols, columns='method', values=metric, aggfunc='mean')
        for a, b in combinations(methods, 2):
            if a not in pivot.columns or b not in pivot.columns:
                continue
            sub = pivot[[a, b]].dropna()
            n = int(len(sub))
            if n < 5:
                rows.append({
                    'metric': metric,
                    'method_a': a,
                    'method_b': b,
                    'n': n,
                    'mean_diff_a_minus_b': float('nan'),
                    'wilcoxon_stat': float('nan'),
                    'p_value': float('nan'),
                })
                continue

            diff = sub[a].to_numpy(dtype=np.float64) - sub[b].to_numpy(dtype=np.float64)
            mean_diff = float(np.mean(diff))
            try:
                stat, pval = wilcoxon(sub[a], sub[b], zero_method='wilcox', correction=False)
            except Exception:
                stat, pval = float('nan'), float('nan')

            rows.append({
                'metric': metric,
                'method_a': a,
                'method_b': b,
                'n': n,
                'mean_diff_a_minus_b': mean_diff,
                'wilcoxon_stat': float(stat),
                'p_value': float(pval),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    qvals = []
    for metric, sub in out.groupby('metric', dropna=False):
        ps = sub['p_value'].fillna(1.0).to_list()
        q = _bh_fdr(ps)
        qvals.extend([(idx, qv) for idx, qv in zip(sub.index.tolist(), q)])
    qvals = dict(qvals)
    out['p_value_fdr_bh'] = out.index.map(lambda i: qvals.get(i, float('nan')))
    return out


def _save_paper_figures(
    df: pd.DataFrame,
    summary_global: pd.DataFrame,
    summary_by_mod: pd.DataFrame,
    summary_by_type: pd.DataFrame,
    metric: str,
    out_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        print('[WARN] matplotlib unavailable: skipping figures.')
        return []

    fig_dir = out_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    g = summary_global.sort_values(f'{metric}_mean', ascending=False).reset_index(drop=True)
    methods = g['method'].tolist()

    # 1) Global ranking with CI.
    vals = g[f'{metric}_mean'].to_numpy(dtype=np.float64)
    lo = g[f'{metric}_ci_low'].to_numpy(dtype=np.float64)
    hi = g[f'{metric}_ci_high'].to_numpy(dtype=np.float64)
    yerr = np.vstack([np.maximum(0.0, vals - lo), np.maximum(0.0, hi - vals)])

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    x = np.arange(len(g))
    ax.bar(x, vals, yerr=yerr, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha='right')
    ax.set_ylabel(metric)
    ax.set_title(f'Global Ranking by {metric}')
    ax.grid(True, axis='y', alpha=0.25)
    fig.tight_layout()
    for ext in formats:
        p = fig_dir / f'global_ranking_{metric}.{ext}'
        fig.savefig(p, dpi=dpi, bbox_inches='tight')
        saved.append(p)
    plt.close(fig)

    # 2) Per-volume boxplot.
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    data = [df.loc[df['method'] == m, metric].dropna().to_numpy(dtype=np.float64) for m in methods]
    ax.boxplot(data, labels=methods, showfliers=False)
    ax.set_ylabel(metric)
    ax.set_title(f'Distribution of {metric} (per volume)')
    ax.tick_params(axis='x', rotation=25)
    ax.grid(True, axis='y', alpha=0.25)
    fig.tight_layout()
    for ext in formats:
        p = fig_dir / f'boxplot_{metric}.{ext}'
        fig.savefig(p, dpi=dpi, bbox_inches='tight')
        saved.append(p)
    plt.close(fig)

    # 3) Heatmap by modality.
    hm_mod = summary_by_mod.pivot(index='method', columns='mod', values=f'{metric}_mean')
    hm_mod = hm_mod.reindex(index=methods)
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    im = ax.imshow(hm_mod.to_numpy(dtype=np.float64), aspect='auto')
    ax.set_yticks(np.arange(hm_mod.shape[0]))
    ax.set_yticklabels(hm_mod.index.tolist())
    ax.set_xticks(np.arange(hm_mod.shape[1]))
    ax.set_xticklabels(hm_mod.columns.tolist(), rotation=20, ha='right')
    ax.set_title(f'{metric} by Modality')
    for i in range(hm_mod.shape[0]):
        for j in range(hm_mod.shape[1]):
            v = hm_mod.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f'{float(v):.3f}', ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    for ext in formats:
        p = fig_dir / f'heatmap_by_mod_{metric}.{ext}'
        fig.savefig(p, dpi=dpi, bbox_inches='tight')
        saved.append(p)
    plt.close(fig)

    # 4) Heatmap by type.
    hm_ty = summary_by_type.pivot(index='method', columns='type', values=f'{metric}_mean')
    hm_ty = hm_ty.reindex(index=methods)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    im = ax.imshow(hm_ty.to_numpy(dtype=np.float64), aspect='auto')
    ax.set_yticks(np.arange(hm_ty.shape[0]))
    ax.set_yticklabels(hm_ty.index.tolist())
    ax.set_xticks(np.arange(hm_ty.shape[1]))
    ax.set_xticklabels(hm_ty.columns.tolist())
    ax.set_title(f'{metric} by Type')
    for i in range(hm_ty.shape[0]):
        for j in range(hm_ty.shape[1]):
            v = hm_ty.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f'{float(v):.3f}', ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    for ext in formats:
        p = fig_dir / f'heatmap_by_type_{metric}.{ext}'
        fig.savefig(p, dpi=dpi, bbox_inches='tight')
        saved.append(p)
    plt.close(fig)

    return saved


def _save_visual_comparison_slice(
    vis_dir: Path,
    mod: str,
    img_id: str,
    z_idx: int,
    ct_slice: np.ndarray,
    mask_slice: np.ndarray,
    hm_by_method: dict[str, np.ndarray],
    methods: list[str],
    prob: float,
    overlay_alpha: float,
) -> Path | None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return None

    n_cols = 2 + len(methods)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.0 * n_cols, 3.4), facecolor='white')

    axes[0].imshow(ct_slice, cmap='gray')
    axes[0].set_title('CT slice', fontsize=9)
    axes[0].axis('off')

    axes[1].imshow(mask_slice, cmap='gray')
    axes[1].set_title('GT mask', fontsize=9)
    axes[1].axis('off')

    for j, m in enumerate(methods, start=2):
        hm = hm_by_method[m][z_idx]
        axes[j].imshow(ct_slice, cmap='gray')
        axes[j].imshow(hm, cmap='turbo', alpha=overlay_alpha, vmin=0, vmax=1)
        if mask_slice.sum() > 0:
            axes[j].contour(mask_slice, levels=[0.5], colors='lime', linewidths=1.0)
        axes[j].set_title(m, fontsize=9)
        axes[j].axis('off')

    fig.suptitle(f'{mod} | {img_id} | z={z_idx} | p={prob:.3f}', fontsize=10)
    fig.tight_layout()

    vis_dir.mkdir(parents=True, exist_ok=True)
    img_tag = img_id.replace('/', '_')
    out_path = vis_dir / f'xai3d_visual_{mod}_{img_tag}_z{z_idx}.png'
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    return out_path


def _summarize(
    df: pd.DataFrame,
    by_cols: list[str],
    metric_cols: list[str],
    rng: np.random.Generator,
    boot_iters: int,
    ci_alpha: float,
    extended_stats: bool = True,
) -> pd.DataFrame:
    rows: list[dict] = []
    for keys, sub in df.groupby(by_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {k: v for k, v in zip(by_cols, keys)}
        row['n'] = int(len(sub))

        for m in metric_cols:
            arr = sub[m].to_numpy(dtype=np.float64)
            finite = arr[np.isfinite(arr)]
            row[f'{m}_mean'] = float(np.mean(finite)) if finite.size else float('nan')
            row[f'{m}_std'] = float(np.std(finite)) if finite.size else float('nan')
            if extended_stats:
                extra = _extended_stats(finite)
                for k, v in extra.items():
                    row[f'{m}_{k}'] = v
            lo, hi = _bootstrap_ci(finite, rng, boot_iters, ci_alpha)
            row[f'{m}_ci_low'] = lo
            row[f'{m}_ci_high'] = hi
        rows.append(row)

    return pd.DataFrame(rows)


def _load_model_from_run(run_dir: Path, saved_args: dict, device: torch.device):
    ckpt_path = run_dir / 'best_model.pt'
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    K = int(saved_args['K'])
    slice_ckpt_dir = saved_args.get('slice_ckpt_dir', '')
    if slice_ckpt_dir and Path(slice_ckpt_dir).exists():
        model, _ = build_volume_classifier(
            slice_ckpt_dir=slice_ckpt_dir,
            K=K,
            attn_dim=int(saved_args.get('attn_dim', 256)),
            dropout=float(saved_args.get('dropout', 0.25)),
            device=device,
        )
    else:
        # Fallback: same pattern used in eval_volume-cls.py
        sargs = saved_args.get('slice_args', saved_args)
        from hexmil.models.abmil_slice_classifier import build_abmil_classifier_scratch

        slice_model = build_abmil_classifier_scratch(
            backbone=sargs.get('backbone', 'resnet50'),
            pretrained=False,
            proj_dim=sargs.get('proj_dim', 512),
            attn_dim=sargs.get('attn_dim', 256),
            dropout=sargs.get('dropout', 0.25),
        )
        model = VolumeClassifier(
            slice_encoder=slice_model,
            feat_dim=slice_model.feat_dim,
            K=K,
            attn_dim=int(saved_args.get('attn_dim', 256)),
            dropout=float(saved_args.get('dropout', 0.25)),
        ).to(device)

    model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=True)
    model.eval()
    return model, ckpt


def main() -> None:
    args = get_args()
    rng = np.random.default_rng(args.seed)

    run_dir = Path(args.run_dir)
    args_path = run_dir / 'args.json'
    ckpt_path = run_dir / 'best_model.pt'
    if not args_path.exists():
        sys.exit(f'[ERROR] Missing args.json: {args_path}')
    if not ckpt_path.exists():
        sys.exit(f'[ERROR] Missing best_model.pt: {ckpt_path}')

    saved_args = json.loads(args_path.read_text())
    K = int(saved_args['K'])
    sargs = saved_args.get('slice_args', saved_args)
    patch_size = int(sargs['patch_size'])
    stride = sargs.get('stride') or (patch_size // 2)

    if args.gpu_id is not None:
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        gid = select_best_gpu()
        device = torch.device(f'cuda:{gid}') if gid is not None else torch.device('cpu')

    out_dir = Path(args.out_dir) if args.out_dir else run_dir / 'evaluation_xai3d_benchmark'
    out_dir.mkdir(parents=True, exist_ok=True)

    mods = set(args.mods) if args.mods else set(ALL_FAKES)
    mods = {m for m in mods if m in set(ALL_FAKES)}
    if not mods:
        sys.exit('[ERROR] No valid fake modality selected. Use pix2pix/cycle/diffusion.')

    print('\n' + '=' * 72)
    print('3D XAI Benchmark (Phase C)')
    print('=' * 72)
    print(f'run_dir             : {run_dir}')
    print(f'device              : {device}')
    print(f'K                   : {K}')
    print(f'patch/stride        : {patch_size}/{stride}')
    print(f'methods             : {args.methods}')
    print(f'modalities          : {sorted(mods)}')
    print(f'max_volumes_per_mod : {args.max_volumes_per_mod}')
    print(f'output              : {out_dir}')

    model, ckpt = _load_model_from_run(run_dir, saved_args, device)
    print(f'loaded epoch        : {ckpt.get("epoch", "?")}')

    tab = load_split_table(DATA_DIR, 'test', ['real'] + ALL_FAKES)
    tab = tab[tab['mod'].isin(sorted(mods))].copy()  # fake-only for 3D localization benchmark

    if args.max_volumes_per_mod > 0:
        kept = []
        for mod in sorted(mods):
            sub = tab[tab['mod'] == mod].copy()
            if len(sub) > args.max_volumes_per_mod:
                idx = rng.choice(len(sub), size=args.max_volumes_per_mod, replace=False)
                sub = sub.iloc[np.sort(idx)]
            kept.append(sub)
        tab = pd.concat(kept, axis=0).reset_index(drop=True)

    if args.qualitative_only:
        if not args.save_visual_examples:
            args.save_visual_examples = True
        n_vis = max(1, int(args.n_visual_examples_per_mod))
        kept = []
        for mod in sorted(mods):
            sub = tab[tab['mod'] == mod].copy()
            if len(sub) > n_vis:
                idx = rng.choice(len(sub), size=n_vis, replace=False)
                sub = sub.iloc[np.sort(idx)]
            kept.append(sub)
        tab = pd.concat(kept, axis=0).reset_index(drop=True)
        print('[INFO] qualitative_only=True -> metrics/statistics are skipped.')

    if len(tab) == 0:
        sys.exit('[ERROR] No volumes selected for evaluation.')

    print(f'evaluated volumes    : {len(tab)}')

    rows: list[dict] = []
    vis_left_by_mod = {m: int(args.n_visual_examples_per_mod) for m in sorted(mods)}
    vis_saved: list[Path] = []

    for _, row in tqdm(tab.iterrows(), total=len(tab), desc='Benchmark 3D', unit='vol', dynamic_ncols=True):
        mod = row['mod']
        img_id = str(row['img_id'])

        try:
            data = load_full_volume_windows(mod, img_id, K, patch_size, stride)
        except Exception as exc:
            print(f'[SKIP] {mod}/{img_id}: {exc}')
            continue

        meta = data['meta']
        Z_total, H, W = int(meta['Z_total']), int(meta['H']), int(meta['W'])
        grid_hw = tuple(meta['grid_hw'])
        slice_hw = (H, W)
        mask_full = data['mask_full'].astype(np.float32)

        # First pass: ABMIL forward once per window (shared by all methods).
        win_results = [
            _run_window(model, win, device, patch_size, stride, H, W, grid_hw, args.beta_thresh)
            for win in data['windows']
        ]

        vol_score = max(float(r['prob']) for r in win_results)
        pred = int(vol_score >= 0.5)

        # Heatmaps by method.
        hm_by_method: dict[str, np.ndarray] = {
            m: np.zeros((Z_total, H, W), dtype=np.float32) for m in args.methods
        }

        for m in args.methods:
            if m == 'random3d':
                hm_by_method[m] = rng.random((Z_total, H, W), dtype=np.float32)
            elif m == 'center_prior3d':
                hm_by_method[m] = _center_prior_3d((Z_total, H, W))

        # Window-based methods.
        window_methods = [
            m for m in args.methods
            if m not in {'random3d', 'center_prior3d'}
        ]

        for win, res in zip(data['windows'], win_results):
            z0, z1 = int(res['z_start']), int(res['z_end'])
            k_eff = z1 - z0

            for m in window_methods:
                hm_win = _build_hm_from_window(
                    method=m,
                    res=res,
                    window=win,
                    model=model,
                    device=device,
                    grid_hw=grid_hw,
                    slice_hw=slice_hw,
                    patch_size=patch_size,
                    stride=stride,
                )
                hm_by_method[m][z0:z1] = hm_win[:k_eff]

        # Score each method.
        for m in args.methods:
            hm = hm_by_method[m]
            hm = _smooth_3d_map(hm, args.smooth_sigma_xy_frac, args.smooth_sigma_z)
            hm_by_method[m] = hm
            if not args.qualitative_only:
                met = _voxel_metrics(hm, mask_full)

                rows.append({
                    'method': m,
                    'mod': mod,
                    'type': _get_ty(img_id),
                    'img_id': img_id,
                    'prob': vol_score,
                    'pred': pred,
                    'mask_voxels': int((mask_full > 0.5).sum()),
                    **met,
                })

        if args.save_visual_examples and vis_left_by_mod.get(mod, 0) > 0:
            coord_z = row['coord_z'] if 'coord_z' in row and pd.notna(row['coord_z']) else None
            if coord_z is not None:
                z_show = int(np.clip(int(coord_z), 0, Z_total - 1))
            else:
                z_show = int(np.clip(np.argmax(mask_full.sum(axis=(1, 2))), 0, Z_total - 1))

            ct_slice = _norm01(data['scan_full'][z_show].astype(np.float32))
            mask_slice = mask_full[z_show]

            p_vis = _save_visual_comparison_slice(
                vis_dir=out_dir / 'visual_examples',
                mod=str(mod),
                img_id=str(img_id),
                z_idx=z_show,
                ct_slice=ct_slice,
                mask_slice=mask_slice,
                hm_by_method=hm_by_method,
                methods=args.methods,
                prob=vol_score,
                overlay_alpha=args.visual_overlay_alpha,
            )
            if p_vis is not None:
                vis_saved.append(p_vis)
                vis_left_by_mod[mod] = max(0, vis_left_by_mod[mod] - 1)

    if args.qualitative_only:
        print('\nSaved qualitative visual examples:')
        if vis_saved:
            for p in vis_saved:
                print(f'  - {p}')
        else:
            print('  [WARN] No visual examples were generated.')
        return

    if not rows:
        sys.exit('[ERROR] No metrics were produced (all samples may have been skipped).')

    df = pd.DataFrame(rows)

    metric_cols = [
        'voxel_auc',
        'iou3d_03', 'iou3d_05', 'iou3d_07',
        'pointing_game_3d',
        'energy_in_mask_3d',
        'bbox_iou3d_03', 'bbox_iou3d_05',
    ]

    per_volume_path = out_dir / 'per_volume_metrics.csv'
    df.to_csv(per_volume_path, index=False, float_format='%.6f')

    global_df = _summarize(
        df=df,
        by_cols=['method'],
        metric_cols=metric_cols,
        rng=np.random.default_rng(args.seed + 1),
        boot_iters=args.bootstrap_iters,
        ci_alpha=args.ci_alpha,
        extended_stats=args.extended_stats,
    ).sort_values(f'{args.primary_metric}_mean', ascending=False)

    by_mod_df = _summarize(
        df=df,
        by_cols=['method', 'mod'],
        metric_cols=metric_cols,
        rng=np.random.default_rng(args.seed + 2),
        boot_iters=args.bootstrap_iters,
        ci_alpha=args.ci_alpha,
        extended_stats=args.extended_stats,
    ).sort_values(['mod', f'{args.primary_metric}_mean'], ascending=[True, False])

    by_type_df = _summarize(
        df=df,
        by_cols=['method', 'type'],
        metric_cols=metric_cols,
        rng=np.random.default_rng(args.seed + 3),
        boot_iters=args.bootstrap_iters,
        ci_alpha=args.ci_alpha,
        extended_stats=args.extended_stats,
    ).sort_values(['type', f'{args.primary_metric}_mean'], ascending=[True, False])

    summary_global_path = out_dir / 'summary_global.csv'
    summary_mod_path = out_dir / 'summary_by_mod.csv'
    summary_type_path = out_dir / 'summary_by_type.csv'
    global_df.to_csv(summary_global_path, index=False, float_format='%.6f')
    by_mod_df.to_csv(summary_mod_path, index=False, float_format='%.6f')
    by_type_df.to_csv(summary_type_path, index=False, float_format='%.6f')

    saved_extra: list[Path] = []

    if args.save_pairwise_tests:
        pw = _pairwise_tests(df=df, metric_cols=metric_cols, sample_id_cols=['img_id'])
        pairwise_path = out_dir / 'pairwise_tests.csv'
        pw.to_csv(pairwise_path, index=False, float_format='%.6f')
        saved_extra.append(pairwise_path)

    if args.save_figures:
        fig_paths = _save_paper_figures(
            df=df,
            summary_global=global_df,
            summary_by_mod=by_mod_df,
            summary_by_type=by_type_df,
            metric=args.primary_metric,
            out_dir=out_dir,
            formats=args.figure_formats,
            dpi=args.figure_dpi,
        )
        saved_extra.extend(fig_paths)

    print('\nSaved:')
    print(f'  - {per_volume_path}')
    print(f'  - {summary_global_path}')
    print(f'  - {summary_mod_path}')
    print(f'  - {summary_type_path}')
    for p in saved_extra:
        print(f'  - {p}')
    for p in vis_saved:
        print(f'  - {p}')

    print(f'\nTop methods by {args.primary_metric}_mean:')
    for _, r in global_df.head(6).iterrows():
        print(
            f"  {r['method']:<18s} "
            f"{args.primary_metric}={r[f'{args.primary_metric}_mean']:.4f}  "
            f"iou3d_05={r['iou3d_05_mean']:.4f}  "
            f"pointing3d={r['pointing_game_3d_mean']:.4f}"
        )


if __name__ == '__main__':
    main()
