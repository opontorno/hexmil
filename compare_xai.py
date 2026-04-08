#!/usr/bin/env python3
"""
compare_xai.py  [ABMIL]
========================
Qualitative comparison of HEXMIL (ante-hoc) attention vs post-hoc attribution
methods (Grad-CAM, GradCAM++) on the central manipulation slice of each volume.

For each unique volume in the test set, we select the slice with the highest
ground-truth mask coverage (most manipulated) and render a 5-row comparison:

    Row 0 — CT slice (assembled from patches)
    Row 1 — GT manipulation mask
    Row 2 — ABMIL native attention  [ANTE-HOC — the model's decision mechanism]
    Row 3 — Grad-CAM               [post-hoc backpropagation approximation]
    Row 4 — GradCAM++              [post-hoc, improved weighting]

Produces one multi-column figure per output type:
    out_dir/xai_comparison_{mod}.png   — per-modality grid (up to n_samples cols)
    out_dir/xai_sample_{img_id}.png    — individual sample overlays

Usage:
    python experiments/ABMIL/compare_xai.py \\
        --run_dir experiments/ABMIL/runs/sa-slice-cls_resnet50_p128_s64/trained_on_pix2pix \\
        [--n_samples 5] [--out_dir evaluation/xai_comparison] [--gpu_id 0]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader

# ── Import train module helpers (same importlib pattern as eval_slice-cls.py) ─
_TRAIN_PY = Path(__file__).resolve().parent / 'train_slice.py'
if not _TRAIN_PY.exists():
    sys.exit(f'[ERROR] train_slice.py not found at {_TRAIN_PY}')

_spec = importlib.util.spec_from_file_location('train_slice_cls_abmil', _TRAIN_PY)
_tm   = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tm)

DATA_DIR                    = _tm.DATA_DIR
ALL_FAKES                   = _tm.ALL_FAKES
SliceDataset                = _tm.SliceDataset
load_split_table            = _tm.load_split_table
build_abmil_classifier_scratch = _tm.build_abmil_classifier_scratch
reconstruct_heatmap         = _tm.reconstruct_heatmap
_assemble_slice_preview     = _tm._assemble_slice_preview


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='HEXMIL XAI comparison: ABMIL attention vs Grad-CAM vs GradCAM++'
    )
    parser.add_argument('--run_dir',     type=str, required=True,
                        help='Path to a slice-level run directory '
                             '(must contain best_model.pt and args.json)')
    parser.add_argument('--n_samples',   type=int, default=5,
                        help='Max samples per modality in the comparison grid (default: 5)')
    parser.add_argument('--out_dir',     type=str, default=None,
                        help='Output directory (default: <run_dir>/evaluation/xai_comparison)')
    parser.add_argument('--batch_size',  type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu_id',      type=int, default=None)
    return parser.parse_args()

# =============================================================================
#  Attention / CAM utilities
# =============================================================================

def _smooth(a: np.ndarray) -> np.ndarray:
    """Gaussian smooth + min-max normalise to [0, 1]."""
    sigma = max(1.0, max(a.shape) * 0.03)
    a = gaussian_filter(a.astype(np.float32), sigma=sigma)
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo + 1e-8)


def _patch_cams_to_heatmap(
    cam_patches: np.ndarray,   # (N, P, P) — one CAM map per patch
    grid_hw:     tuple[int, int],
    slice_hw:    tuple[int, int],
    patch_size:  int,
    stride:      int,
) -> np.ndarray:
    """
    Reconstruct a full-resolution heatmap from per-patch CAM maps by
    averaging overlapping contributions — mirrors reconstruct_heatmap().
    """
    H, W  = slice_hw
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


# ── Grad-CAM store ────────────────────────────────────────────────────────────

class _FeatureStore:
    """Captures backbone feature maps and their gradients via hooks."""
    feat: torch.Tensor | None = None

    def clear(self) -> None:
        self.feat = None


def _register_gradcam_hook(model) -> tuple[_FeatureStore, list]:
    """
    Registers a forward hook on model.encoder.backbone to capture the last
    feature map (timm features_only returns a list; we take [-1]).

    The tensor is kept in the computational graph via retain_grad() so that
    .backward() fills its .grad attribute.

    Returns:
        store   — object holding .feat after the forward pass
        handles — list of hook handles to remove after use
    """
    store = _FeatureStore()

    def _fwd_hook(module, inp, out):
        store.feat = out[-1]        # (B*N, C, h, w) — last conv feature
        store.feat.retain_grad()

    handle = model.encoder.backbone.register_forward_hook(_fwd_hook)
    return store, [handle]


@torch.no_grad()
def _abmil_heatmap(
    model,
    patches:    torch.Tensor,   # (N, 1, P, P) CPU
    grid_hw:    tuple[int, int],
    slice_hw:   tuple[int, int],
    patch_size: int,
    stride:     int,
    device:     torch.device,
) -> np.ndarray:
    """Native ABMIL attention heatmap (ante-hoc)."""
    _, attn_w = model(patches.unsqueeze(0).to(device), return_attn=True)
    hm = reconstruct_heatmap(
        attn_w[0].cpu().numpy(), grid_hw, slice_hw, patch_size, stride
    )
    return _smooth(hm)


def _gradcam_heatmap(
    model,
    patches:    torch.Tensor,   # (N, 1, P, P) CPU
    grid_hw:    tuple[int, int],
    slice_hw:   tuple[int, int],
    patch_size: int,
    stride:     int,
    device:     torch.device,
    plusplus:   bool = False,
) -> np.ndarray:
    """
    Grad-CAM (or GradCAM++ when plusplus=True) heatmap.

    Backpropagates the classification logit through the full ABMIL stack to
    the last backbone feature map.  Per-patch CAMs are reconstructed into a
    full-resolution heatmap with the same spatial averaging as ABMIL attention.
    """
    store, handles = _register_gradcam_hook(model)

    model.eval()
    inp = patches.unsqueeze(0).to(device)   # (1, N, 1, P, P)

    # Forward (with grad)
    logits = model(inp)                     # (1, 1)
    model.zero_grad()
    logits[0, 0].backward()

    grad = store.feat.grad.detach()         # (N, C, h, w)
    act  = store.feat.detach()              # (N, C, h, w)

    for h in handles:
        h.remove()
    store.clear()

    N, C, fh, fw = act.shape
    cam_patches = np.zeros((N, patch_size, patch_size), dtype=np.float32)

    for n in range(N):
        g = grad[n]    # (C, h, w)
        a = act[n]     # (C, h, w)

        if not plusplus:
            # Standard Grad-CAM: weight = global average of gradients
            w = g.mean((-2, -1))                           # (C,)
        else:
            # GradCAM++: alpha weighting (Chattopadhyay et al. 2018)
            g2 = g ** 2
            g3 = g ** 3
            alpha_num = g2
            alpha_den = 2.0 * g2 + (a * g3).sum((-2, -1), keepdim=True)
            alpha = alpha_num / (alpha_den + 1e-8)         # (C, h, w)
            w = (alpha * F.relu(g)).sum((-2, -1))          # (C,)

        cam_n = F.relu((w[:, None, None] * a).sum(0))     # (h, w)

        # Upsample CAM from feature-map resolution to patch resolution
        cam_up = F.interpolate(
            cam_n[None, None],
            size=(patch_size, patch_size),
            mode='bilinear',
            align_corners=False,
        )[0, 0].cpu().numpy()

        cam_patches[n] = cam_up

    hm = _patch_cams_to_heatmap(cam_patches, grid_hw, slice_hw, patch_size, stride)
    return _smooth(hm)


# =============================================================================
#  Figure saving
# =============================================================================

def _save_comparison_figure(
    samples:    list[dict],
    out_path:   Path,
    mod:        str,
) -> None:
    """
    Save a 5-row × len(samples)-column comparison figure.

    Each sample dict must contain:
        grid, mask, hm_abmil, hm_gradcam, hm_gradcampp,
        prob, label, img_id, coord_z
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print('[WARN] matplotlib not available — skipping figure.')
        return

    N_COLS = len(samples)
    N_ROWS = 5
    row_labels = [
        'CT Slice',
        'GT Mask',
        'ABMIL Attention\n(ante-hoc)',
        'Grad-CAM\n(post-hoc)',
        'GradCAM++\n(post-hoc)',
    ]
    row_cmaps = ['gray', 'gray', 'turbo', 'turbo', 'turbo']

    fig, axes = plt.subplots(
        N_ROWS, N_COLS,
        figsize=(3.5 * N_COLS, 3.0 * N_ROWS),
        squeeze=False,
        facecolor='#111111',
    )
    fig.suptitle(
        f'XAI Comparison — {mod.upper()}   '
        r'[$\hat{\mathcal{H}}_{3D}$: ABMIL attention = ante-hoc | '
        r'Grad-CAM, GradCAM++ = post-hoc]',
        fontsize=9, fontweight='bold', color='white', y=1.01,
    )

    for col, s in enumerate(samples):
        gt_str   = 'real' if s['label'] == 0 else mod
        ok       = (s['label'] == 0) == (s['prob'] <= 0.5)
        col_title = (
            f"{s['img_id']} | z={s['coord_z']}\n"
            f"GT:{gt_str}  p={s['prob']:.2f}  {'✓' if ok else '✗'}"
        )
        axes[0, col].set_title(col_title, fontsize=7, color='white')

        data_rows = [
            s['grid'],
            s['mask'],
            s['hm_abmil'],
            s['hm_gradcam'],
            s['hm_gradcampp'],
        ]
        for row, (data, cmap) in enumerate(zip(data_rows, row_cmaps)):
            ax = axes[row, col]
            ax.imshow(data, cmap=cmap, vmin=0, vmax=1 if cmap == 'turbo' else None)
            # Green mask contour overlay on rows 2–4
            if row >= 2 and s['label'] > 0 and s['mask'].sum() > 0:
                ax.contour(s['mask'], levels=[0.5], colors='lime', linewidths=0.8)
            ax.axis('off')

    # Row labels on the left
    for row, label in enumerate(row_labels):
        color = '#00e5ff' if row == 2 else '#ff7043'  # cyan=ante-hoc, orange=post-hoc
        if row == 0 or row == 1:
            color = '#cccccc'
        axes[row, 0].set_ylabel(
            label, fontsize=8, fontweight='bold',
            color=color, rotation=90, labelpad=4,
        )

    # Ante-hoc / post-hoc separator between row 2 and row 3
    if N_COLS > 0:
        try:
            fig.canvas.draw()
            y_sep = (
                axes[2, 0].get_position().y0 +
                axes[3, 0].get_position().y1
            ) / 2
            fig.add_artist(plt.Line2D(
                [0.05, 0.98], [y_sep, y_sep],
                transform=fig.transFigure,
                color='#e65100', linewidth=1.5, linestyle='--',
            ))
            fig.text(0.5, y_sep + 0.005, '── post-hoc ──',
                     ha='center', va='bottom', fontsize=7,
                     color='#e65100', fontweight='bold',
                     transform=fig.transFigure)
        except Exception:
            pass

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches='tight', facecolor='#111111')
    plt.close(fig)
    print(f'  Saved → {out_path}')


def _save_overlay_figure(s: dict, out_path: Path, mod: str) -> None:
    """
    Save a compact 1-row × 5-column overlay figure for a single sample.
    Columns: CT | GT mask | ABMIL attn overlay | Grad-CAM overlay | GradCAM++ overlay
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    gt_str = 'real' if s['label'] == 0 else mod
    fig, axes = plt.subplots(1, 5, figsize=(17.5, 3.5), facecolor='#111111')

    axes[0].imshow(s['grid'], cmap='gray')
    axes[0].set_title('CT Slice', fontsize=8, color='white')

    axes[1].imshow(s['mask'], cmap='gray')
    axes[1].set_title('GT Mask', fontsize=8, color='white')

    for ax, hm, label, color in zip(
        axes[2:],
        [s['hm_abmil'], s['hm_gradcam'], s['hm_gradcampp']],
        ['ABMIL (ante-hoc)', 'Grad-CAM', 'GradCAM++'],
        ['#00e5ff', '#ff7043', '#ff7043'],
    ):
        ax.imshow(s['grid'], cmap='gray')
        ax.imshow(hm, cmap='turbo', alpha=0.55, vmin=0, vmax=1)
        if s['label'] > 0 and s['mask'].sum() > 0:
            ax.contour(s['mask'], levels=[0.5], colors='lime', linewidths=1.0)
        ax.set_title(label, fontsize=8, color=color, fontweight='bold')

    for ax in axes:
        ax.axis('off')

    fig.suptitle(
        f"{s['img_id']} | z={s['coord_z']} | {mod.upper()} | "
        f"GT:{gt_str}  p={s['prob']:.2f}",
        fontsize=8, color='white',
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110, bbox_inches='tight', facecolor='#111111')
    plt.close(fig)


# =============================================================================
#  Main
# =============================================================================


def main() -> None:
    args    = get_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / 'evaluation' / 'xai_comparison'
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = run_dir / 'best_model.pt'
    args_path = run_dir / 'args.json'
    if not ckpt_path.exists():
        sys.exit(f'[ERROR] best_model.pt not found in {run_dir}')
    if not args_path.exists():
        sys.exit(f'[ERROR] args.json not found in {run_dir}')

    with open(args_path) as f:
        train_args = json.load(f)

    patch_size  = int(train_args['patch_size'])
    stride      = train_args.get('stride') or (patch_size // 2)
    seed        = int(train_args.get('seed', 42))
    raw_mods    = train_args.get('mods', ALL_FAKES + ['real'])
    fake_mods   = [m for m in raw_mods if m != 'real']

    # ── Device ───────────────────────────────────────────────────────────────
    if args.gpu_id is not None:
        device = torch.device(f'cuda:{args.gpu_id}')
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    print(f"\n{'='*60}")
    print(f"  HEXMIL XAI Comparison")
    print(f"{'='*60}")
    print(f"  Run dir    : {run_dir}")
    print(f"  Device     : {device}")
    print(f"  Patch size : {patch_size}   Stride: {stride}")
    print(f"  Fakes      : {fake_mods}")
    print(f"  Out dir    : {out_dir}")

    # ── Load model ───────────────────────────────────────────────────────────
    ckpt  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = build_abmil_classifier_scratch(
        backbone   = train_args.get('backbone', 'resnet50'),
        pretrained = False,
        proj_dim   = train_args.get('proj_dim', 512),
        attn_dim   = train_args.get('attn_dim', 128),
        dropout    = train_args.get('dropout', 0.25),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"  Loaded epoch {ckpt['epoch']}  "
          f"val_loss={ckpt['best_val_loss']:.4f}  "
          f"val_auc={ckpt['best_val_auc']:.4f}")

    # ── Build test dataloader ─────────────────────────────────────────────────
    tab_test = load_split_table(DATA_DIR, 'test', ['real'] + ALL_FAKES)
    ds_test  = SliceDataset(DATA_DIR, tab_test, patch_size=patch_size,
                            stride=stride, augment=False)
    dl_test  = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)
    print(f"  Test slices: {len(ds_test)}")

    # ── Collect fake samples: best slice per (img_id, mod) ────────────────────
    # best = highest GT mask coverage (= central manipulation slice)
    print('\nSelecting central manipulation slices...')
    best: dict[str, dict[str, dict]] = {m: {} for m in ALL_FAKES}

    from scipy.special import expit as sigmoid_np

    with torch.no_grad():
        for batch in dl_test:
            for i in range(len(batch['label'])):
                label = int(batch['label'][i])
                if label == 0:
                    continue                          # skip real
                mod    = batch['mod'][i]
                img_id = batch['img_id'][i]
                mask_sum = float(batch['mask'][i, 0].sum())

                if (img_id not in best[mod] or
                        mask_sum > best[mod][img_id]['mask_sum']):
                    best[mod][img_id] = {
                        'mask_sum':  mask_sum,
                        'patches':   batch['patches'][i].clone(),
                        'mask':      batch['mask'][i, 0].numpy(),
                        'grid_hw':   (int(batch['grid_hw'][i, 0]),
                                      int(batch['grid_hw'][i, 1])),
                        'slice_hw':  (int(batch['slice_hw'][i, 0]),
                                      int(batch['slice_hw'][i, 1])),
                        'coord_z':   int(batch['coord_z'][i]),
                        'label':     label,
                        'img_id':    img_id,
                        'mod':       mod,
                    }

    # ── Compute heatmaps & build figures ─────────────────────────────────────
    print('\nComputing heatmaps...')
    rng = np.random.default_rng(seed)

    for mod in ALL_FAKES:
        candidates = sorted(best[mod].values(),
                            key=lambda x: x['mask_sum'], reverse=True)
        # deterministic subsample
        if len(candidates) > args.n_samples:
            idx = rng.choice(len(candidates), args.n_samples, replace=False)
            candidates = [candidates[k] for k in sorted(idx)]
        if not candidates:
            print(f'  [WARN] No fake samples found for mod={mod}')
            continue

        print(f'\n  Modality: {mod}  ({len(candidates)} samples)')
        processed = []

        for s in candidates:
            patches   = s['patches']     # (N, 1, P, P) CPU
            grid_hw   = s['grid_hw']
            slice_hw  = s['slice_hw']

            # ── ABMIL native attention (ante-hoc, no grad needed) ──────────
            hm_abmil = _abmil_heatmap(
                model, patches, grid_hw, slice_hw, patch_size, stride, device
            )

            # ── Grad-CAM ──────────────────────────────────────────────────
            hm_gradcam = _gradcam_heatmap(
                model, patches, grid_hw, slice_hw, patch_size, stride, device,
                plusplus=False,
            )

            # ── GradCAM++ ─────────────────────────────────────────────────
            hm_gradcampp = _gradcam_heatmap(
                model, patches, grid_hw, slice_hw, patch_size, stride, device,
                plusplus=True,
            )

            # ── Grid preview of the CT slice ───────────────────────────────
            grid = _assemble_slice_preview(patches, grid_hw, slice_hw, patch_size, stride)
            # normalise to [0, 1] for display
            lo, hi = grid.min(), grid.max()
            grid = (grid - lo) / (hi - lo + 1e-8)

            # ── Compute prob from model ────────────────────────────────────
            with torch.no_grad():
                logit = model(patches.unsqueeze(0).to(device))
            prob = float(torch.sigmoid(logit[0, 0]).cpu())

            entry = {
                **s,
                'grid':        grid,
                'hm_abmil':    hm_abmil,
                'hm_gradcam':  hm_gradcam,
                'hm_gradcampp': hm_gradcampp,
                'prob':         prob,
            }
            processed.append(entry)

            # Individual overlay figure
            _save_overlay_figure(
                entry,
                out_dir / f'xai_sample_{s["img_id"]}_z{s["coord_z"]}.png',
                mod,
            )
            print(f'    {s["img_id"]}  z={s["coord_z"]}  '
                  f'mask_sum={s["mask_sum"]:.0f}  p={prob:.3f}')

        # Multi-sample grid figure for this modality
        _save_comparison_figure(
            processed,
            out_dir / f'xai_comparison_{mod}.png',
            mod,
        )

    print(f'\nDone.  Outputs → {out_dir}')


if __name__ == '__main__':
    main()
