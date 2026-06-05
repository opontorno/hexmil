"""
xai_utils.py
------------
Shared utilities for eval_xai.py and eval_xai_3d.py.

Provides: normalisation helpers, patch-CAM reconstruction, Grad-CAM hooks,
ABMIL/Grad-CAM heatmap extractors, bootstrap CI, FDR correction,
pairwise statistical tests, and figure generation.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Type detection
# ---------------------------------------------------------------------------

def _get_ty(img_id: str) -> str:
    return 'removal' if str(img_id).startswith('rem_') else 'injection'


# ---------------------------------------------------------------------------
# Normalization and smoothing
# ---------------------------------------------------------------------------

def _normalize01(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + 1e-8)


def _smooth(a: np.ndarray, sigma_frac: float) -> np.ndarray:
    sigma = max(1.0, max(a.shape) * max(1e-6, sigma_frac))
    a = gaussian_filter(a.astype(np.float32), sigma=sigma)
    return _normalize01(a)


# ---------------------------------------------------------------------------
# Patch-CAM reconstruction
# ---------------------------------------------------------------------------

def _patch_cams_to_heatmap(
    cam_patches: np.ndarray,
    slice_hw: tuple[int, int],
    patch_size: int,
    stride: int,
) -> np.ndarray:
    """Reconstruct full-res heatmap by averaging overlapping per-patch CAM maps."""
    H, W = slice_hw
    half  = patch_size // 2
    accum = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)

    ys = list(range(half, H - half + 1, stride))
    xs = list(range(half, W - half + 1, stride))
    if not ys or ys[-1] < H - half:
        ys.append(H - half)
    if not xs or xs[-1] < W - half:
        xs.append(W - half)
    ys = sorted(set(ys))
    xs = sorted(set(xs))

    idx = 0
    for cy in ys:
        for cx in xs:
            y0, y1 = max(cy - half, 0), min(cy + half, H)
            x0, x1 = max(cx - half, 0), min(cx + half, W)
            ph, pw  = y1 - y0, x1 - x0
            accum[y0:y1, x0:x1] += cam_patches[idx, :ph, :pw]
            count[y0:y1, x0:x1] += 1.0
            idx += 1

    return accum / np.maximum(count, 1e-6)


# ---------------------------------------------------------------------------
# Grad-CAM hook infrastructure
# ---------------------------------------------------------------------------

class _FeatureStore:
    feat: torch.Tensor | None = None

    def clear(self) -> None:
        self.feat = None


def _register_gradcam_hook(model) -> tuple[_FeatureStore, list]:
    store = _FeatureStore()

    def _fwd_hook(module, inp, out):
        feat = out[-1] if isinstance(out, (list, tuple)) else out
        store.feat = feat
        store.feat.retain_grad()

    handle = model.encoder.backbone.register_forward_hook(_fwd_hook)
    return store, [handle]


# ---------------------------------------------------------------------------
# Heatmap extractors
# ---------------------------------------------------------------------------

@torch.no_grad()
def _abmil_heatmap(
    model,
    patches: torch.Tensor,
    grid_hw: tuple[int, int],
    slice_hw: tuple[int, int],
    patch_size: int,
    stride: int,
    device: torch.device,
    sigma_frac: float,
    reconstruct_heatmap_fn,
) -> np.ndarray:
    """Native ABMIL attention heatmap (ante-hoc)."""
    _, attn_w = model(patches.unsqueeze(0).to(device), return_attn=True)
    hm = reconstruct_heatmap_fn(
        attn_w[0].detach().cpu().numpy(), grid_hw, slice_hw, patch_size, stride
    )
    return _smooth(hm, sigma_frac)


def _gradcam_heatmap(
    model,
    patches: torch.Tensor,
    slice_hw: tuple[int, int],
    patch_size: int,
    stride: int,
    device: torch.device,
    sigma_frac: float,
    plusplus: bool = False,
) -> np.ndarray:
    """Grad-CAM or Grad-CAM++ (plusplus=True) heatmap."""
    store, handles = _register_gradcam_hook(model)

    model.eval()
    inp    = patches.unsqueeze(0).to(device)
    logits = model(inp)
    model.zero_grad(set_to_none=True)
    logits[0, 0].backward()

    if store.feat is None or store.feat.grad is None:
        for h in handles:
            h.remove()
        raise RuntimeError('Grad-CAM hook did not capture feature gradients.')

    grad = store.feat.grad.detach()
    act  = store.feat.detach()

    for h in handles:
        h.remove()
    store.clear()

    N, C, _, _  = act.shape
    cam_patches = np.zeros((N, patch_size, patch_size), dtype=np.float32)

    for n in range(N):
        g = grad[n]
        a = act[n]

        if not plusplus:
            w = g.mean((-2, -1))
        else:
            g2        = g ** 2
            g3        = g ** 3
            alpha_num = g2
            alpha_den = 2.0 * g2 + (a * g3).sum((-2, -1), keepdim=True)
            alpha     = alpha_num / (alpha_den + 1e-8)
            w         = (alpha * F.relu(g)).sum((-2, -1))

        cam_n  = F.relu((w[:, None, None] * a).sum(0))
        cam_up = F.interpolate(
            cam_n[None, None],
            size=(patch_size, patch_size),
            mode='bilinear',
            align_corners=False,
        )[0, 0].detach().cpu().numpy()
        cam_patches[n] = cam_up

    hm = _patch_cams_to_heatmap(cam_patches, slice_hw, patch_size, stride)
    return _smooth(hm, sigma_frac)


# ---------------------------------------------------------------------------
# Statistical functions
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    iters: int,
    alpha: float,
) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    n      = values.size
    if n == 0:
        return float('nan'), float('nan')
    if n == 1 or iters <= 0:
        x = float(values.mean())
        return x, x
    means = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        idx      = rng.integers(0, n, size=n)
        means[i] = values[idx].mean()
    q_lo = (1.0 - alpha) / 2.0
    q_hi = 1.0 - q_lo
    return float(np.quantile(means, q_lo)), float(np.quantile(means, q_hi))


def _extended_stats(arr: np.ndarray) -> dict[str, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {k: float('nan') for k in ('median', 'p25', 'p75', 'min', 'max', 'sem')}
    std = float(np.std(finite))
    return {
        'median': float(np.median(finite)),
        'p25':    float(np.quantile(finite, 0.25)),
        'p75':    float(np.quantile(finite, 0.75)),
        'min':    float(np.min(finite)),
        'max':    float(np.max(finite)),
        'sem':    std / float(np.sqrt(max(1, finite.size))),
    }


def _bh_fdr(pvals: list[float]) -> list[float]:
    if not pvals:
        return []
    n       = len(pvals)
    order   = np.argsort(pvals)
    ordered = np.array([pvals[i] for i in order], dtype=np.float64)
    q       = np.empty(n, dtype=np.float64)
    prev    = 1.0
    for i in range(n - 1, -1, -1):
        val    = min(prev, ordered[i] * n / (i + 1))
        q[i]   = val
        prev   = val
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
        pivot = df.pivot_table(
            index=sample_id_cols, columns='method', values=metric, aggfunc='mean'
        )
        for a, b in combinations(methods, 2):
            if a not in pivot.columns or b not in pivot.columns:
                continue
            sub  = pivot[[a, b]].dropna()
            n    = int(len(sub))
            base = {'metric': metric, 'method_a': a, 'method_b': b, 'n': n}
            if n < 5:
                rows.append({**base,
                              'mean_diff_a_minus_b': float('nan'),
                              'wilcoxon_stat': float('nan'),
                              'p_value': float('nan')})
                continue
            diff      = sub[a].to_numpy(dtype=np.float64) - sub[b].to_numpy(dtype=np.float64)
            mean_diff = float(np.mean(diff))
            try:
                stat, pval = wilcoxon(sub[a], sub[b], zero_method='wilcox', correction=False)
            except Exception:
                stat, pval = float('nan'), float('nan')
            rows.append({**base,
                         'mean_diff_a_minus_b': mean_diff,
                         'wilcoxon_stat': float(stat),
                         'p_value': float(pval)})

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    qvals: dict[int, float] = {}
    for _, sub in out.groupby('metric', dropna=False):
        ps = sub['p_value'].fillna(1.0).to_list()
        q  = _bh_fdr(ps)
        qvals.update(zip(sub.index.tolist(), q))
    out['p_value_fdr_bh'] = out.index.map(lambda i: qvals.get(i, float('nan')))
    return out


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
            arr    = sub[m].to_numpy(dtype=np.float64)
            finite = arr[np.isfinite(arr)]
            row[f'{m}_mean'] = float(np.mean(finite)) if finite.size else float('nan')
            row[f'{m}_std']  = float(np.std(finite))  if finite.size else float('nan')
            if extended_stats:
                for k, v in _extended_stats(finite).items():
                    row[f'{m}_{k}'] = v
            lo, hi = _bootstrap_ci(finite, rng, boot_iters, ci_alpha)
            row[f'{m}_ci_low']  = lo
            row[f'{m}_ci_high'] = hi
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Paper figures
# ---------------------------------------------------------------------------

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

    g      = summary_global.sort_values(f'{metric}_mean', ascending=False).reset_index(drop=True)
    vals   = g[f'{metric}_mean'].to_numpy(dtype=np.float64)
    lo_arr = g[f'{metric}_ci_low'].to_numpy(dtype=np.float64)
    hi_arr = g[f'{metric}_ci_high'].to_numpy(dtype=np.float64)
    yerr   = np.vstack([np.maximum(0.0, vals - lo_arr), np.maximum(0.0, hi_arr - vals)])
    methods = g['method'].tolist()

    # 1) Global ranking with CI
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(len(g))
    ax.bar(x, vals, yerr=yerr, capsize=4)
    ax.set_xticks(x);  ax.set_xticklabels(methods, rotation=25, ha='right')
    ax.set_ylabel(metric);  ax.set_title(f'Global Ranking by {metric}')
    ax.grid(True, axis='y', alpha=0.25);  fig.tight_layout()
    for ext in formats:
        p = fig_dir / f'global_ranking_{metric}.{ext}'
        fig.savefig(p, dpi=dpi, bbox_inches='tight');  saved.append(p)
    plt.close(fig)

    # 2) Per-sample boxplot
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    data = [df.loc[df['method'] == m, metric].dropna().to_numpy(dtype=np.float64) for m in methods]
    ax.boxplot(data, labels=methods, showfliers=False)
    ax.set_ylabel(metric);  ax.set_title(f'Distribution of {metric} (per sample)')
    ax.tick_params(axis='x', rotation=25);  ax.grid(True, axis='y', alpha=0.25);  fig.tight_layout()
    for ext in formats:
        p = fig_dir / f'boxplot_{metric}.{ext}'
        fig.savefig(p, dpi=dpi, bbox_inches='tight');  saved.append(p)
    plt.close(fig)

    # 3) Heatmap by modality
    hm_mod = summary_by_mod.pivot(index='method', columns='mod', values=f'{metric}_mean')
    hm_mod = hm_mod.reindex(index=methods)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    im = ax.imshow(hm_mod.to_numpy(dtype=np.float64), aspect='auto')
    ax.set_yticks(np.arange(hm_mod.shape[0]));  ax.set_yticklabels(hm_mod.index.tolist())
    ax.set_xticks(np.arange(hm_mod.shape[1]));  ax.set_xticklabels(hm_mod.columns.tolist(), rotation=20, ha='right')
    ax.set_title(f'{metric} by Modality')
    for i in range(hm_mod.shape[0]):
        for j in range(hm_mod.shape[1]):
            v = hm_mod.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f'{float(v):.3f}', ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04);  fig.tight_layout()
    for ext in formats:
        p = fig_dir / f'heatmap_by_mod_{metric}.{ext}'
        fig.savefig(p, dpi=dpi, bbox_inches='tight');  saved.append(p)
    plt.close(fig)

    # 4) Heatmap by type (removal/injection)
    hm_ty = summary_by_type.pivot(index='method', columns='type', values=f'{metric}_mean')
    hm_ty = hm_ty.reindex(index=methods)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    im = ax.imshow(hm_ty.to_numpy(dtype=np.float64), aspect='auto')
    ax.set_yticks(np.arange(hm_ty.shape[0]));  ax.set_yticklabels(hm_ty.index.tolist())
    ax.set_xticks(np.arange(hm_ty.shape[1]));  ax.set_xticklabels(hm_ty.columns.tolist())
    ax.set_title(f'{metric} by Type')
    for i in range(hm_ty.shape[0]):
        for j in range(hm_ty.shape[1]):
            v = hm_ty.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f'{float(v):.3f}', ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04);  fig.tight_layout()
    for ext in formats:
        p = fig_dir / f'heatmap_by_type_{metric}.{ext}'
        fig.savefig(p, dpi=dpi, bbox_inches='tight');  saved.append(p)
    plt.close(fig)

    return saved
