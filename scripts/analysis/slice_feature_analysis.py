#!/usr/bin/env python3
"""
slice_feature_analysis.py
=========================
Analisi delle feature estratte dalle CT slice intere (Phase B).

Per ogni tipo di feature vengono addestrati 4 classificatori ML base:
  Logistic Regression, KNN, SVM (RBF), Random Forest.

Tipi di feature analizzate:
  1. raw        – pixel dell'intera slice ridimensionata (+ PCA)
  2. fourier    – magnitudine dello spettro FFT 2D (+ PCA)
  3. dct        – coefficienti DCT 2D in zig-zag (+ PCA)
  4. wavelet    – coefficienti Wavelet (Haar, 2 livelli, + PCA)

Le slice vengono ridimensionate a `--resize` × `--resize` prima
dell'analisi nel dominio spaziale (raw) e frequenziale.

Si usa esattamente la stessa divisione train/valid/test del training DL.

Output:
  scripts/results/slice_feature_analysis.json
  scripts/results/slice_feature_analysis.csv

Usage:
    python scripts/slice_feature_analysis.py [--resize 128] [--pca_dim 128]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from scipy.fft import fft2, fftshift
from scipy.fft import dctn
from scipy.ndimage import zoom
import pywt

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── project path
sys.path.insert(0, str(WORK_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from config import DATA_DIR
from hexmil.data.patch_dataset import load_split_table
from hexmil.utils.tiff_utils import (
    get_shape_tiff_scan,
    load_slice_tiff_scan,
    get_percentile_tiff_scan,
    apply_percentile,
)

RESULTS_DIR = Path(__file__).resolve().parent / 'results'

# =============================================================================
#  CLI
# =============================================================================

def get_args():
    p = argparse.ArgumentParser(description='Slice feature analysis — ML classifiers')
    p.add_argument('--resize',    type=int, default=128,
                   help='Resize slice to NxN before feature extraction (default 128)')
    p.add_argument('--pca_dim',  type=int, default=128,
                   help='PCA components before ML classifiers (default 128)')
    p.add_argument('--max_train', type=int, default=None,
                   help='Cap training samples (None = all, useful for quick tests)')
    p.add_argument('--n_jobs',   type=int, default=-1,
                   help='Parallel jobs for sklearn (default -1 = all CPUs)')
    return p.parse_args()

# =============================================================================
#  Feature extraction  (all operate on a resized (H, W) float32 slice)
# =============================================================================

def extract_raw(img: np.ndarray) -> np.ndarray:
    """Flatten pixel values. img: (H, W) float32 in [0,1]."""
    return img.ravel()

def extract_fourier(img: np.ndarray) -> np.ndarray:
    """
    2D FFT magnitude spectrum, log-scaled and flattened.
    GAN artifacts often manifest as periodic patterns in the frequency domain.
    """
    spec = np.abs(fftshift(fft2(img)))
    spec = np.log1p(spec)
    return spec.ravel().astype(np.float32)

def extract_dct(img: np.ndarray) -> np.ndarray:
    """
    2D DCT-II coefficients in zig-zag order (low-frequency first).
    GAN upsampling artifacts tend to appear at specific DCT frequencies.
    """
    coeff = dctn(img, norm='ortho')
    H, W  = coeff.shape
    zigzag = []
    for s in range(H + W - 1):
        if s % 2 == 0:
            row = min(s, H - 1)
            col = s - row
            while row >= 0 and col < W:
                zigzag.append(coeff[row, col])
                row -= 1
                col += 1
        else:
            col = min(s, W - 1)
            row = s - col
            while col >= 0 and row < H:
                zigzag.append(coeff[row, col])
                row += 1
                col -= 1
    return np.array(zigzag, dtype=np.float32)

def extract_wavelet(img: np.ndarray, wavelet: str = 'haar', level: int = 3) -> np.ndarray:
    """
    2D DWT (3 levels on full slice), all sub-bands concatenated.
    Captures multi-scale structural differences between real and fake CT.
    """
    coeffs = pywt.wavedec2(img, wavelet=wavelet, level=level)
    parts  = [coeffs[0].ravel()]
    for detail in coeffs[1:]:
        for sub in detail:
            parts.append(sub.ravel())
    return np.concatenate(parts).astype(np.float32)

FEATURE_EXTRACTORS = {
    'raw':     extract_raw,
    'fourier': extract_fourier,
    'dct':     extract_dct,
    'wavelet': extract_wavelet,
}

# =============================================================================
#  Data loading
# =============================================================================

def load_slice_np(data_dir: str, mod: str, img_id: str, cz: int) -> np.ndarray:
    """Load and normalise a single axial CT slice. Returns (H, W) float32 in [0,1]."""
    scan_dir  = os.path.join(data_dir, mod, 'scan', img_id)
    shape     = get_shape_tiff_scan(scan_dir)
    low, high = get_percentile_tiff_scan(scan_dir, np.uint16)
    sl = load_slice_tiff_scan(scan_dir, shape, np.uint16, cz, cz + 1)[0]
    return apply_percentile(sl.astype(np.float32), low, high)

def resize_slice(img: np.ndarray, target: int) -> np.ndarray:
    """Resize (H, W) slice to (target, target) using bilinear zoom."""
    H, W = img.shape
    if H == target and W == target:
        return img
    return zoom(img, (target / H, target / W), order=1).astype(np.float32)

def load_split(
    split: str,
    resize_to: int,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Load all slices for a split (resized to resize_to × resize_to).
    Returns:
        images:  (N, resize_to, resize_to) float32
        labels:  (N,) int   0=real  1=fake (binary)
        mods:    list of modality strings
        img_ids: list of img_id strings
    """
    mods = ['real', 'pix2pix', 'cycle', 'diffusion']
    tab  = load_split_table(DATA_DIR, split, mods)

    if max_samples is not None and len(tab) > max_samples:
        tab = tab.sample(n=max_samples, random_state=42).reset_index(drop=True)

    print(f"  [{split:>5}] Loading {len(tab)} slices (resize→{resize_to}×{resize_to})...",
          end='', flush=True)
    t0 = time()

    images   = np.empty((len(tab), resize_to, resize_to), dtype=np.float32)
    labels   = np.empty(len(tab), dtype=np.int32)
    mod_list = []
    id_list  = []

    for i, row in tab.iterrows():
        idx = int(row.name) if hasattr(row, 'name') else i
        pos = len(mod_list)   # position in output arrays

        try:
            sl = load_slice_np(DATA_DIR, row['mod'], str(row['img_id']), int(row['coord_z']))
            images[pos]  = resize_slice(sl, resize_to)
        except Exception as e:
            images[pos]  = np.zeros((resize_to, resize_to), dtype=np.float32)

        labels[pos] = 0 if row['mod'] == 'real' else 1
        mod_list.append(row['mod'])
        id_list.append(str(row['img_id']))

        if (pos + 1) % 500 == 0:
            print(f"\r  [{split:>5}] {pos+1}/{len(tab)} slices loaded...", end='', flush=True)

    print(f"\r  [{split:>5}] {len(tab)} slices loaded in {time()-t0:.1f}s  "
          f"(real={np.sum(labels==0)}, fake={np.sum(labels==1)})")

    return images, labels, mod_list, id_list

# =============================================================================
#  Classifiers
# =============================================================================

def build_classifiers(pca_dim: int, n_jobs: int) -> dict:
    """Return dict of sklearn Pipelines: StandardScaler → PCA → Classifier."""
    pca = ('pca', PCA(n_components=pca_dim, random_state=42))
    return {
        'logistic': Pipeline([
            ('scaler', StandardScaler()), pca,
            ('clf', LogisticRegression(max_iter=1000, n_jobs=n_jobs, random_state=42)),
        ]),
        'knn': Pipeline([
            ('scaler', StandardScaler()), pca,
            ('clf', KNeighborsClassifier(n_neighbors=5, n_jobs=n_jobs)),
        ]),
        'svm': Pipeline([
            ('scaler', StandardScaler()), pca,
            ('clf', SVC(kernel='rbf', probability=True, random_state=42)),
        ]),
        'random_forest': Pipeline([
            ('scaler', StandardScaler()), pca,
            ('clf', RandomForestClassifier(n_estimators=200, n_jobs=n_jobs, random_state=42)),
        ]),
    }

def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    mods: list[str],
) -> dict:
    """Compute overall + per-modality classification metrics."""
    y_pred = (y_prob >= 0.5).astype(int)
    out = {
        'auc':       float(roc_auc_score(y_true, y_prob)),
        'accuracy':  float(accuracy_score(y_true, y_pred)),
        'f1':        float(f1_score(y_true, y_pred, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall':    float(recall_score(y_true, y_pred, zero_division=0)),
    }
    mods_arr = np.array(mods)
    real_idx = mods_arr == 'real'
    for mod in ['pix2pix', 'cycle', 'diffusion']:
        fake_idx = mods_arr == mod
        sel = real_idx | fake_idx
        if sel.sum() > 0 and fake_idx.sum() > 0:
            out[f'auc_{mod}'] = float(roc_auc_score(y_true[sel], y_prob[sel]))
    return out

# =============================================================================
#  Main
# =============================================================================

def main():
    args = get_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  Slice Feature Analysis  (resize={args.resize}  pca_dim={args.pca_dim})")
    print(f"{'='*65}\n")

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading data splits...")
    imgs_tr,  labs_tr,  mods_tr,  ids_tr  = load_split('train', args.resize, args.max_train)
    imgs_val, labs_val, mods_val, ids_val  = load_split('valid', args.resize)
    imgs_te,  labs_te,  mods_te,  ids_te   = load_split('test',  args.resize)
    print()

    all_results = {}
    rows = []

    for feat_name, extractor in FEATURE_EXTRACTORS.items():
        print(f"─── Feature: {feat_name.upper()} ───────────────────────────")

        # ── Extract features ───────────────────────────────────────────
        print(f"  Extracting features...", end='', flush=True)
        t0 = time()
        X_tr  = np.vstack([extractor(img) for img in imgs_tr])
        X_val = np.vstack([extractor(img) for img in imgs_val])
        X_te  = np.vstack([extractor(img) for img in imgs_te])
        print(f"  done in {time()-t0:.1f}s   shape={X_tr.shape}")

        clfs = build_classifiers(
            pca_dim=min(args.pca_dim, X_tr.shape[1], X_tr.shape[0]),
            n_jobs=args.n_jobs,
        )

        all_results[feat_name] = {}

        for clf_name, pipeline in clfs.items():
            t0 = time()
            print(f"  [{clf_name:<15}] training...", end='', flush=True)

            pipeline.fit(X_tr, labs_tr)

            # Validate
            prob_val = pipeline.predict_proba(X_val)[:, 1]
            val_auc  = roc_auc_score(labs_val, prob_val)

            # Test
            prob_te = pipeline.predict_proba(X_te)[:, 1]
            metrics = compute_metrics(labs_te, prob_te, mods_te)

            elapsed = time() - t0
            print(f"  val_auc={val_auc:.3f}  test_auc={metrics['auc']:.3f}  "
                  f"acc={metrics['accuracy']:.3f}  ({elapsed:.1f}s)")

            all_results[feat_name][clf_name] = {
                'val_auc': round(val_auc, 4),
                **{k: round(v, 4) for k, v in metrics.items()},
            }

            rows.append({
                'feature':    feat_name,
                'classifier': clf_name,
                'val_auc':    round(val_auc, 4),
                **{k: round(v, 4) for k, v in metrics.items()},
            })

        print()

    # ── Save results ───────────────────────────────────────────────────────
    json_path = RESULTS_DIR / f'slice_feature_analysis_r{args.resize}.json'
    csv_path  = RESULTS_DIR / f'slice_feature_analysis_r{args.resize}.csv'

    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    # ── Print summary table ────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  SUMMARY TABLE  (test set, resize={args.resize})")
    print(f"{'='*65}")
    pivot = df.pivot_table(
        index='classifier', columns='feature', values='auc', aggfunc='first'
    ).round(4)
    print(pivot.to_string())
    print()
    print(f"  Results saved to:\n    {json_path}\n    {csv_path}")

if __name__ == '__main__':
    main()
