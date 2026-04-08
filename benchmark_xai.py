#!/usr/bin/env python3
"""
benchmark_xai.py
================
Quantitative XAI benchmark for ABMIL slice classifier checkpoints.

Compares multiple explanation methods on the same selected fake slices:
  - ABMIL native attention (ante-hoc)
  - Grad-CAM
  - GradCAM++
  - Random map (sanity baseline)
  - Center prior (sanity baseline)

Default slice selection strategy mirrors compare_xai.py:
for each (modality, volume-id), use the slice with maximum GT mask coverage.

Outputs (in out_dir):
  - per_sample_metrics.csv
  - summary_global.csv
  - summary_by_mod.csv
  - summary_by_type.csv
	- pairwise_tests.csv
	- figures/*.{png,pdf,svg}

Usage example:
  python experiments/ABMIL/benchmark_xai.py \
	  --run_dir experiments/ABMIL/runs/slice-cls_resnet50_p128_s64/trained_on_all \
	  --n_samples_per_mod 80 --bootstrap_iters 1000
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
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm


# Import train_slice-cls.py via importlib so we reuse exactly the same dataset/model stack.
_TRAIN_PY = Path(__file__).resolve().parent / 'train_slice.py'
if not _TRAIN_PY.exists():
	sys.exit(f'[ERROR] train_slice.py not found at {_TRAIN_PY}')

_spec = importlib.util.spec_from_file_location('train_slice_cls_bench', _TRAIN_PY)
_tm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tm)

DATA_DIR = _tm.DATA_DIR
ALL_FAKES = _tm.ALL_FAKES
SliceDataset = _tm.SliceDataset
load_split_table = _tm.load_split_table
build_abmil_classifier_scratch = _tm.build_abmil_classifier_scratch
build_fabmil_classifier_scratch = _tm.build_fabmil_classifier_scratch
reconstruct_heatmap = _tm.reconstruct_heatmap
_assemble_slice_preview = _tm._assemble_slice_preview


def get_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description='Quantitative XAI benchmark for ABMIL slice checkpoints')
	p.add_argument('--run_dir', type=str, required=True,
				   help='Path to slice run directory containing best_model.pt and args.json')
	p.add_argument('--out_dir', type=str, default=None,
				   help='Output directory (default: <run_dir>/evaluation/xai_benchmark)')
	p.add_argument('--batch_size', type=int, default=8)
	p.add_argument('--num_workers', type=int, default=4)
	p.add_argument('--gpu_id', type=int, default=None)
	p.add_argument('--seed', type=int, default=42)

	p.add_argument('--mods', nargs='+', default=None,
				   help='Fake modalities to include (default: all fake modalities)')
	p.add_argument('--n_samples_per_mod', type=int, default=0,
				   help='Max slices per modality after selection (0 = use all)')
	p.add_argument('--selection', choices=['best_mask', 'all_slices'], default='best_mask',
				   help='Slice selection strategy')

	p.add_argument('--methods', nargs='+',
				   default=['abmil', 'gradcam', 'gradcampp', 'random', 'center_prior'],
				   choices=['abmil', 'gradcam', 'gradcampp', 'random', 'center_prior'])
	p.add_argument('--smooth_sigma_frac', type=float, default=0.03,
				   help='Gaussian sigma as fraction of max(H, W) for all heatmaps')

	# Enabled by default: full statistical reporting + paper figures.
	p.add_argument('--bootstrap_iters', type=int, default=1000,
				   help='Bootstrap iterations for confidence intervals (0 disables CI)')
	p.add_argument('--ci_alpha', type=float, default=0.95,
				   help='Confidence level for bootstrap CI')
	p.add_argument('--extended_stats', action=argparse.BooleanOptionalAction, default=True,
				   help='Compute extended descriptive statistics in summaries')
	p.add_argument('--save_pairwise_tests', action=argparse.BooleanOptionalAction, default=True,
				   help='Save pairwise Wilcoxon tests across methods')

	p.add_argument('--save_figures', action=argparse.BooleanOptionalAction, default=True,
				   help='Generate paper-ready figures')
	p.add_argument('--figure_dpi', type=int, default=400)
	p.add_argument('--figure_formats', nargs='+', default=['png'],
				   choices=['png', 'pdf', 'svg'],
				   help='Output figure formats')
	p.add_argument('--primary_metric', type=str, default='iou_07',
				   choices=['pixel_auc', 'iou_03', 'iou_05', 'iou_07', 'iou_7', 'pointing_game', 'energy_in_mask'],
				   help='Primary metric used for ranking plots')

	# Qualitative visual examples (enabled by default).
	p.add_argument('--save_visual_examples', action=argparse.BooleanOptionalAction, default=True,
				   help='Save per-slice visual comparisons across XAI methods')
	p.add_argument('--n_visual_examples_per_mod', type=int, default=4,
				   help='How many qualitative examples to save per modality')
	p.add_argument('--visual_overlay_alpha', type=float, default=0.50,
				   help='Heatmap overlay alpha for qualitative figures')
	p.add_argument('--qualitative_only', action=argparse.BooleanOptionalAction, default=False,
				   help='Generate only qualitative visual examples (skip metrics/statistics)')
	return p.parse_args()


def _get_ty(img_id: str) -> str:
	return 'removal' if str(img_id).startswith('rem_') else 'injection'


def _normalize01(a: np.ndarray) -> np.ndarray:
	a = a.astype(np.float32)
	lo, hi = float(a.min()), float(a.max())
	return (a - lo) / (hi - lo + 1e-8)


def _smooth(a: np.ndarray, sigma_frac: float) -> np.ndarray:
	sigma = max(1.0, max(a.shape) * max(1e-6, sigma_frac))
	a = gaussian_filter(a.astype(np.float32), sigma=sigma)
	return _normalize01(a)


def _patch_cams_to_heatmap(
	cam_patches: np.ndarray,
	slice_hw: tuple[int, int],
	patch_size: int,
	stride: int,
) -> np.ndarray:
	"""Reconstruct full-res heatmap by averaging overlapping per-patch CAM maps."""
	H, W = slice_hw
	half = patch_size // 2
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
			ph, pw = y1 - y0, x1 - x0
			accum[y0:y1, x0:x1] += cam_patches[idx, :ph, :pw]
			count[y0:y1, x0:x1] += 1.0
			idx += 1

	return accum / np.maximum(count, 1e-6)


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
) -> np.ndarray:
	_, attn_w = model(patches.unsqueeze(0).to(device), return_attn=True)
	hm = reconstruct_heatmap(attn_w[0].detach().cpu().numpy(), grid_hw, slice_hw, patch_size, stride)
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
	store, handles = _register_gradcam_hook(model)

	model.eval()
	inp = patches.unsqueeze(0).to(device)
	logits = model(inp)
	model.zero_grad(set_to_none=True)
	logits[0, 0].backward()

	if store.feat is None or store.feat.grad is None:
		for h in handles:
			h.remove()
		raise RuntimeError('Grad-CAM hook did not capture feature gradients.')

	grad = store.feat.grad.detach()
	act = store.feat.detach()

	for h in handles:
		h.remove()
	store.clear()

	N, C, _, _ = act.shape
	cam_patches = np.zeros((N, patch_size, patch_size), dtype=np.float32)

	for n in range(N):
		g = grad[n]
		a = act[n]

		if not plusplus:
			w = g.mean((-2, -1))
		else:
			g2 = g ** 2
			g3 = g ** 3
			alpha_num = g2
			alpha_den = 2.0 * g2 + (a * g3).sum((-2, -1), keepdim=True)
			alpha = alpha_num / (alpha_den + 1e-8)
			w = (alpha * F.relu(g)).sum((-2, -1))

		cam_n = F.relu((w[:, None, None] * a).sum(0))
		cam_up = F.interpolate(
			cam_n[None, None],
			size=(patch_size, patch_size),
			mode='bilinear',
			align_corners=False,
		)[0, 0].detach().cpu().numpy()
		cam_patches[n] = cam_up

	hm = _patch_cams_to_heatmap(cam_patches, slice_hw, patch_size, stride)
	return _smooth(hm, sigma_frac)


def _random_heatmap(shape: tuple[int, int], rng: np.random.Generator, sigma_frac: float) -> np.ndarray:
	return _smooth(rng.random(shape, dtype=np.float32), sigma_frac)


def _center_prior_heatmap(shape: tuple[int, int], sigma_frac: float) -> np.ndarray:
	H, W = shape
	yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
	cy = (H - 1) / 2.0
	cx = (W - 1) / 2.0
	sy = max(2.0, H * 0.2)
	sx = max(2.0, W * 0.2)
	g = np.exp(-(((yy - cy) ** 2) / (2 * sy * sy) + ((xx - cx) ** 2) / (2 * sx * sx)))
	return _smooth(g, sigma_frac)


def _xai_metrics(heatmap: np.ndarray, mask: np.ndarray) -> dict[str, float]:
	m = (mask > 0.5)
	hm = np.clip(heatmap.astype(np.float32), 0.0, 1.0)
	out: dict[str, float] = {}

	if m.sum() > 0 and (~m).sum() > 0:
		out['pixel_auc'] = float(roc_auc_score(m.ravel().astype(np.uint8), hm.ravel()))
	else:
		out['pixel_auc'] = float('nan')

	for thr in (0.3, 0.5, 0.7):
		h_bin = hm >= thr
		inter = float((h_bin & m).sum())
		union = float((h_bin | m).sum())
		out[f'iou_{int(thr * 10):02d}'] = inter / union if union > 0 else 0.0

	py, px = np.unravel_index(np.argmax(hm), hm.shape)
	out['pointing_game'] = float(m[py, px])

	e_tot = float(hm.sum())
	e_in = float((hm * m.astype(np.float32)).sum())
	out['energy_in_mask'] = e_in / e_tot if e_tot > 0 else float('nan')

	return out


def _bootstrap_ci(
	values: np.ndarray,
	rng: np.random.Generator,
	iters: int,
	alpha: float,
) -> tuple[float, float]:
	values = values[np.isfinite(values)]
	n = values.size
	if n == 0:
		return float('nan'), float('nan')
	if n == 1 or iters <= 0:
		x = float(values.mean())
		return x, x

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

	# 1) Global ranking with CI.
	g = summary_global.sort_values(f'{metric}_mean', ascending=False).reset_index(drop=True)
	vals = g[f'{metric}_mean'].to_numpy(dtype=np.float64)
	lo = g[f'{metric}_ci_low'].to_numpy(dtype=np.float64)
	hi = g[f'{metric}_ci_high'].to_numpy(dtype=np.float64)
	yerr = np.vstack([np.maximum(0.0, vals - lo), np.maximum(0.0, hi - vals)])

	fig, ax = plt.subplots(figsize=(8.8, 4.8))
	x = np.arange(len(g))
	ax.bar(x, vals, yerr=yerr, capsize=4)
	ax.set_xticks(x)
	ax.set_xticklabels(g['method'].tolist(), rotation=25, ha='right')
	ax.set_ylabel(metric)
	ax.set_title(f'Global Ranking by {metric}')
	ax.grid(True, axis='y', alpha=0.25)
	fig.tight_layout()
	for ext in formats:
		p = fig_dir / f'global_ranking_{metric}.{ext}'
		fig.savefig(p, dpi=dpi, bbox_inches='tight')
		saved.append(p)
	plt.close(fig)

	# 2) Per-sample boxplot.
	fig, ax = plt.subplots(figsize=(8.8, 4.8))
	methods = g['method'].tolist()
	data = [df.loc[df['method'] == m, metric].dropna().to_numpy(dtype=np.float64) for m in methods]
	ax.boxplot(data, labels=methods, showfliers=False)
	ax.set_ylabel(metric)
	ax.set_title(f'Distribution of {metric} (per sample)')
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
	fig, ax = plt.subplots(figsize=(7.6, 4.8))
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

	# 4) Heatmap by type (removal/injection).
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


def _select_visual_example_keys(selected: list[dict], n_per_mod: int) -> set[tuple[str, str, int]]:
	"""Pick up to n_per_mod examples per mod, prioritizing highest mask_sum slices."""
	if n_per_mod <= 0:
		return set()

	by_mod: dict[str, list[dict]] = {}
	for s in selected:
		by_mod.setdefault(str(s['mod']), []).append(s)

	keys: set[tuple[str, str, int]] = set()
	for mod, items in by_mod.items():
		items_sorted = sorted(items, key=lambda x: float(x.get('mask_sum', 0.0)), reverse=True)
		for s in items_sorted[:n_per_mod]:
			keys.add((str(mod), str(s['img_id']), int(s['coord_z'])))
	return keys


def _save_visual_comparison(
	vis_dir: Path,
	sample: dict,
	grid: np.ndarray,
	heatmaps: dict[str, np.ndarray],
	methods: list[str],
	overlay_alpha: float,
) -> Path | None:
	"""Save one compact side-by-side comparison figure with minimal spacing."""
	try:
		import matplotlib
		matplotlib.use('Agg')
		import matplotlib.pyplot as plt
		import matplotlib.patches as mpatches
		import matplotlib.patheffects as pe
	except Exception:
		return None

	img_id = str(sample['img_id'])
	mod = str(sample['mod'])
	coord_z = int(sample['coord_z'])
	mask = sample['mask'].astype(np.float32)
	prob = float(sample['prob'])

	n_cols = 1 + len(methods)
	fig, axes = plt.subplots(1, n_cols, figsize=(2.6 * n_cols, 2.8), facecolor='none')
	title_kws = {
		'fontsize': 9,
		'fontweight': 'semibold',
		'pad': 2,
		'color': 'white',
		'bbox': {'facecolor': 'none', 'edgecolor': 'none', 'pad': 0.0},
	}
	title_map = {
		'abmil': 'ABMIL',
		'gradcam': 'Grad-CAM',
		'gradcampp': 'Grad-CAM++',
		'random': 'Random',
		'center_prior': 'Center Prior',
	}

	def _bbox_from_mask(m: np.ndarray) -> tuple[int, int, int, int] | None:
		r = np.any(m > 0.5, axis=1)
		c = np.any(m > 0.5, axis=0)
		if not r.any() or not c.any():
			return None
		r0, r1 = np.where(r)[0][[0, -1]]
		c0, c1 = np.where(c)[0][[0, -1]]
		return int(r0), int(r1), int(c0), int(c1)

	axes[0].imshow(grid, cmap='gray')
	bb = _bbox_from_mask(mask)
	if bb is not None:
		r0, r1, c0, c1 = bb
		axes[0].add_patch(mpatches.Rectangle(
			(c0, r0),
			max(1, c1 - c0),
			max(1, r1 - r0),
			linewidth=1.2,
			edgecolor='lime',
			facecolor='none',
		))
	t0 = axes[0].set_title('CT + GT BBox', **title_kws)
	t0.set_path_effects([pe.withStroke(linewidth=2.0, foreground='black')])
	axes[0].axis('off')

	for j, m in enumerate(methods, start=1):
		hm = heatmaps[m]
		axes[j].imshow(grid, cmap='gray')
		axes[j].imshow(hm, cmap='turbo', alpha=overlay_alpha, vmin=0, vmax=1)
		if mask.sum() > 0:
			axes[j].contour(mask, levels=[0.5], colors='lime', linewidths=1.0)
		tj = axes[j].set_title(title_map.get(m, m.replace('_', ' ').title()), **title_kws)
		tj.set_path_effects([pe.withStroke(linewidth=2.0, foreground='black')])
		axes[j].axis('off')

	# Keep layout intentionally dense for paper crops.
	fig.subplots_adjust(left=0.0, right=1.0, top=0.94, bottom=0.0, wspace=0.005, hspace=0.0)

	vis_dir.mkdir(parents=True, exist_ok=True)
	img_tag = img_id.replace('/', '_')
	out_path = vis_dir / f'xai_visual_{mod}_{img_tag}_z{coord_z}.png'
	fig.savefig(out_path, dpi=220, bbox_inches='tight', pad_inches=0, transparent=True)
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
	grouped = df.groupby(by_cols, dropna=False)

	for keys, sub in grouped:
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


def _load_model(run_dir: Path, train_args: dict, device: torch.device):
	ckpt = torch.load(run_dir / 'best_model.pt', map_location='cpu', weights_only=False)
	use_fourier = bool(train_args.get('use_fourier', False))

	if use_fourier:
		model = build_fabmil_classifier_scratch(
			backbone=train_args.get('backbone', 'resnet50'),
			pretrained=False,
			patch_size=int(train_args['patch_size']),
			fourier_feat_dim=int(train_args.get('fourier_feat_dim', 256)),
			proj_dim=int(train_args.get('proj_dim', 512)),
			attn_dim=int(train_args.get('attn_dim', 128)),
			dropout=float(train_args.get('dropout', 0.25)),
		)
	else:
		model = build_abmil_classifier_scratch(
			backbone=train_args.get('backbone', 'resnet50'),
			pretrained=False,
			proj_dim=int(train_args.get('proj_dim', 512)),
			attn_dim=int(train_args.get('attn_dim', 128)),
			dropout=float(train_args.get('dropout', 0.25)),
		)

	model.load_state_dict(ckpt['model_state_dict'])
	model = model.to(device)
	model.eval()
	return model, ckpt


def _select_samples(
	dl_test: DataLoader,
	selection: str,
	mods_to_keep: set[str],
	n_samples_per_mod: int,
	seed: int,
) -> list[dict]:
	rng = np.random.default_rng(seed)

	if selection == 'best_mask':
		best: dict[str, dict[str, dict]] = {m: {} for m in mods_to_keep}
		for batch in tqdm(dl_test, desc='Selecting slices', unit='batch', dynamic_ncols=True):
			B = len(batch['label'])
			for i in range(B):
				label = int(batch['label'][i])
				if label == 0:
					continue
				mod = batch['mod'][i]
				if mod not in mods_to_keep:
					continue
				img_id = str(batch['img_id'][i])
				mask = batch['mask'][i, 0].numpy()
				msum = float(mask.sum())
				if msum <= 0:
					continue

				prev = best[mod].get(img_id)
				if prev is None or msum > prev['mask_sum']:
					best[mod][img_id] = {
						'patches': batch['patches'][i].clone(),
						'mask': mask.astype(np.float32),
						'grid_hw': (int(batch['grid_hw'][i, 0]), int(batch['grid_hw'][i, 1])),
						'slice_hw': (int(batch['slice_hw'][i, 0]), int(batch['slice_hw'][i, 1])),
						'coord_z': int(batch['coord_z'][i]),
						'label': label,
						'img_id': img_id,
						'mod': mod,
						'mask_sum': msum,
					}

		out: list[dict] = []
		for mod in sorted(mods_to_keep):
			cands = sorted(best[mod].values(), key=lambda x: x['mask_sum'], reverse=True)
			if n_samples_per_mod > 0 and len(cands) > n_samples_per_mod:
				idx = rng.choice(len(cands), size=n_samples_per_mod, replace=False)
				cands = [cands[k] for k in sorted(idx)]
			out.extend(cands)
		return out

	# all_slices mode
	out = []
	for batch in tqdm(dl_test, desc='Selecting slices', unit='batch', dynamic_ncols=True):
		B = len(batch['label'])
		for i in range(B):
			label = int(batch['label'][i])
			if label == 0:
				continue
			mod = batch['mod'][i]
			if mod not in mods_to_keep:
				continue
			mask = batch['mask'][i, 0].numpy().astype(np.float32)
			if mask.sum() <= 0:
				continue
			out.append({
				'patches': batch['patches'][i].clone(),
				'mask': mask,
				'grid_hw': (int(batch['grid_hw'][i, 0]), int(batch['grid_hw'][i, 1])),
				'slice_hw': (int(batch['slice_hw'][i, 0]), int(batch['slice_hw'][i, 1])),
				'coord_z': int(batch['coord_z'][i]),
				'label': label,
				'img_id': str(batch['img_id'][i]),
				'mod': mod,
				'mask_sum': float(mask.sum()),
			})

	if n_samples_per_mod > 0:
		filt = []
		for mod in sorted(mods_to_keep):
			mod_s = [x for x in out if x['mod'] == mod]
			if len(mod_s) > n_samples_per_mod:
				idx = rng.choice(len(mod_s), size=n_samples_per_mod, replace=False)
				mod_s = [mod_s[k] for k in sorted(idx)]
			filt.extend(mod_s)
		return filt
	return out


def _select_visual_samples_fast(
	dl_test: DataLoader,
	mods_to_keep: set[str],
	n_per_mod: int,
) -> list[dict]:
	"""
	Fast path for qualitative-only mode.
Collect first non-empty-mask fake slices per modality and stop early.
	"""
	if n_per_mod <= 0:
		return []

	need = {m: int(n_per_mod) for m in sorted(mods_to_keep)}
	selected: list[dict] = []

	for batch in tqdm(dl_test, desc='Selecting visual examples (fast)', unit='batch', dynamic_ncols=True):
		if all(v <= 0 for v in need.values()):
			break

		B = len(batch['label'])
		for i in range(B):
			label = int(batch['label'][i])
			if label == 0:
				continue
			mod = str(batch['mod'][i])
			if mod not in mods_to_keep or need.get(mod, 0) <= 0:
				continue
			mask = batch['mask'][i, 0].numpy().astype(np.float32)
			msum = float(mask.sum())
			if msum <= 0:
				continue

			selected.append({
				'patches': batch['patches'][i].clone(),
				'mask': mask,
				'grid_hw': (int(batch['grid_hw'][i, 0]), int(batch['grid_hw'][i, 1])),
				'slice_hw': (int(batch['slice_hw'][i, 0]), int(batch['slice_hw'][i, 1])),
				'coord_z': int(batch['coord_z'][i]),
				'label': label,
				'img_id': str(batch['img_id'][i]),
				'mod': mod,
				'mask_sum': msum,
			})
			need[mod] = max(0, need[mod] - 1)
			if all(v <= 0 for v in need.values()):
				break

	return selected


def main() -> None:
	args = get_args()
	if args.primary_metric == 'iou_7':
		args.primary_metric = 'iou_07'
	rng = np.random.default_rng(args.seed)

	run_dir = Path(args.run_dir)
	ckpt_path = run_dir / 'best_model.pt'
	args_path = run_dir / 'args.json'
	if not ckpt_path.exists():
		sys.exit(f'[ERROR] Missing checkpoint: {ckpt_path}')
	if not args_path.exists():
		sys.exit(f'[ERROR] Missing args file: {args_path}')

	with open(args_path) as f:
		train_args = json.load(f)

	patch_size = int(train_args['patch_size'])
	stride = train_args.get('stride') or (patch_size // 2)

	if args.gpu_id is not None:
		device = torch.device(f'cuda:{args.gpu_id}')
	elif torch.cuda.is_available():
		device = torch.device('cuda:0')
	else:
		device = torch.device('cpu')

	out_dir = Path(args.out_dir) if args.out_dir else run_dir / 'evaluation' / 'xai_benchmark'
	out_dir.mkdir(parents=True, exist_ok=True)

	mods_to_keep = set(args.mods) if args.mods else set(ALL_FAKES)
	mods_to_keep = {m for m in mods_to_keep if m in set(ALL_FAKES)}
	if not mods_to_keep:
		sys.exit('[ERROR] No valid fake modalities selected. Use one of: pix2pix cycle diffusion')

	print('\n' + '=' * 70)
	print('XAI Benchmark (slice-level)')
	print('=' * 70)
	print(f'run_dir         : {run_dir}')
	print(f'device          : {device}')
	print(f'patch/stride    : {patch_size}/{stride}')
	print(f'methods         : {args.methods}')
	print(f'selection       : {args.selection}')
	print(f'mods            : {sorted(mods_to_keep)}')
	print(f'output          : {out_dir}')

	model, ckpt = _load_model(run_dir, train_args, device)
	print(f'loaded epoch    : {ckpt.get("epoch", "?")}')

	tab_test = load_split_table(DATA_DIR, 'test', ['real'] + ALL_FAKES)
	ds_test = SliceDataset(DATA_DIR, tab_test, patch_size=patch_size, stride=stride, augment=False)
	dl_test = DataLoader(
		ds_test,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=True,
	)

	if args.qualitative_only:
		selected = _select_visual_samples_fast(
			dl_test=dl_test,
			mods_to_keep=mods_to_keep,
			n_per_mod=max(1, int(args.n_visual_examples_per_mod)),
		)
	else:
		selected = _select_samples(
			dl_test=dl_test,
			selection=args.selection,
			mods_to_keep=mods_to_keep,
			n_samples_per_mod=args.n_samples_per_mod,
			seed=args.seed,
		)
	if not selected:
		sys.exit('[ERROR] No slices selected. Check --mods or dataset content.')

	print(f'selected slices  : {len(selected)}')

	if args.qualitative_only:
		if not args.save_visual_examples:
			args.save_visual_examples = True
		if args.n_visual_examples_per_mod <= 0:
			args.n_visual_examples_per_mod = 4
		print('[INFO] qualitative_only=True -> metrics/statistics are skipped.')

	rows: list[dict] = []
	visual_keys = _select_visual_example_keys(selected, args.n_visual_examples_per_mod)
	visual_saved: list[Path] = []
	for s in tqdm(selected, desc='Benchmarking methods', unit='slice', dynamic_ncols=True):
		patches = s['patches']
		mask = s['mask']
		grid_hw = s['grid_hw']
		slice_hw = s['slice_hw']
		mod = s['mod']
		img_id = s['img_id']
		coord_z = int(s['coord_z'])
		key = (str(mod), str(img_id), int(coord_z))
		need_visual = args.save_visual_examples and (key in visual_keys)
		if args.qualitative_only and not need_visual:
			continue

		with torch.no_grad():
			logit = model(patches.unsqueeze(0).to(device))
			prob = float(torch.sigmoid(logit[0, 0]).detach().cpu())
		pred = int(prob >= 0.5)

		heatmaps: dict[str, np.ndarray] = {}
		for m in args.methods:
			if m == 'abmil':
				heatmaps[m] = _abmil_heatmap(
					model, patches, grid_hw, slice_hw, patch_size, stride, device, args.smooth_sigma_frac,
				)
			elif m == 'gradcam':
				heatmaps[m] = _gradcam_heatmap(
					model, patches, slice_hw, patch_size, stride, device, args.smooth_sigma_frac, plusplus=False,
				)
			elif m == 'gradcampp':
				heatmaps[m] = _gradcam_heatmap(
					model, patches, slice_hw, patch_size, stride, device, args.smooth_sigma_frac, plusplus=True,
				)
			elif m == 'random':
				heatmaps[m] = _random_heatmap(slice_hw, rng, args.smooth_sigma_frac)
			elif m == 'center_prior':
				heatmaps[m] = _center_prior_heatmap(slice_hw, args.smooth_sigma_frac)
			else:
				raise ValueError(f'Unsupported method: {m}')

		if not args.qualitative_only:
			for method, hm in heatmaps.items():
				x = _xai_metrics(hm, mask)
				rows.append({
					'method': method,
					'mod': mod,
					'type': _get_ty(img_id),
					'img_id': img_id,
					'coord_z': coord_z,
					'prob': prob,
					'pred': pred,
					'mask_sum': float(mask.sum()),
					**x,
				})

		if args.save_visual_examples:
			if key in visual_keys:
				grid = _assemble_slice_preview(patches, grid_hw, slice_hw, patch_size, stride)
				grid = _normalize01(grid)
				p_vis = _save_visual_comparison(
					vis_dir=out_dir / 'visual_examples',
					sample={
						'img_id': img_id,
						'mod': mod,
						'coord_z': coord_z,
						'mask': mask,
						'prob': prob,
					},
					grid=grid,
					heatmaps=heatmaps,
					methods=args.methods,
					overlay_alpha=args.visual_overlay_alpha,
				)
				if p_vis is not None:
					visual_saved.append(p_vis)
				# Save one time per selected key.
				visual_keys.discard(key)

	if args.qualitative_only:
		print('\nSaved qualitative visual examples:')
		if visual_saved:
			for p in visual_saved:
				print(f'  - {p}')
		else:
			print('  [WARN] No visual examples were generated.')
		return

	df = pd.DataFrame(rows)
	metric_cols = ['pixel_auc', 'iou_03', 'iou_05', 'iou_07', 'pointing_game', 'energy_in_mask']

	per_sample_path = out_dir / 'per_sample_metrics.csv'
	df.to_csv(per_sample_path, index=False, float_format='%.6f')

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

	global_path = out_dir / 'summary_global.csv'
	by_mod_path = out_dir / 'summary_by_mod.csv'
	by_type_path = out_dir / 'summary_by_type.csv'
	global_df.to_csv(global_path, index=False, float_format='%.6f')
	by_mod_df.to_csv(by_mod_path, index=False, float_format='%.6f')
	by_type_df.to_csv(by_type_path, index=False, float_format='%.6f')

	saved_extra = []

	if args.save_pairwise_tests:
		pw = _pairwise_tests(df=df, metric_cols=metric_cols, sample_id_cols=['img_id', 'coord_z'])
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
	print(f'  - {per_sample_path}')
	print(f'  - {global_path}')
	print(f'  - {by_mod_path}')
	print(f'  - {by_type_path}')
	for p in saved_extra:
		print(f'  - {p}')
	for p in visual_saved:
		print(f'  - {p}')

	print(f'\nTop methods by {args.primary_metric}_mean:')
	for _, r in global_df.head(5).iterrows():
		print(
			f"  {r['method']:<12s} "
			f"{args.primary_metric}={r[f'{args.primary_metric}_mean']:.4f}  "
			f"iou_05={r['iou_05_mean']:.4f}  pointing={r['pointing_game_mean']:.4f}"
		)


if __name__ == '__main__':
	main()
