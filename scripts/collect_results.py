#!/usr/bin/env python3
"""
collect_results.py
==================
Walks all experiments/*/runs/ directories, reads every test_metrics.json,
and produces a tidy CSV sorted by:
  phase → model_family → backbone → patch_size → stride → trained_on

Usage:
    python scripts/collect_results.py                     # prints + saves CSV
    python scripts/collect_results.py --out results.csv   # custom output path
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / 'experiments'

# ── Canonical column order ───────────────────────────────────────────────────
#  Identity columns first, then overall test metrics, then per-modality AUC.
ID_COLS = [
    'phase', 'model_family', 'backbone',
    'patch_size', 'stride', 'K',
    'trained_on',
]

OVERALL_COLS = [
    'auc', 'accuracy', 'ap', 'f1',
    'ood_test_auc', 'ood_test_accuracy', 'ood_test_ap', 'ood_test_f1',
]

MOD_COLS_TEMPLATE = [
    '{m}_auc', '{m}_acc', '{m}_ap', '{m}_f1',
    '{m}_rem_auc', '{m}_rem_acc',
    '{m}_inj_auc', '{m}_inj_acc',
]
ALL_MODS = ['pix2pix', 'cycle', 'diffusion']

MOD_COLS = [t.format(m=m) for m in ALL_MODS for t in MOD_COLS_TEMPLATE]

# Some older runs used 'auc_cycle' / 'auc_pix2pix' keys instead of 'cycle_auc'.
_LEGACY_KEY_MAP: dict[str, str] = {
    f'auc_{m}': f'{m}_auc' for m in ALL_MODS
}


def _normalise_keys(d: dict) -> dict:
    """Rename legacy keys to the current naming convention."""
    out = {}
    for k, v in d.items():
        out[_LEGACY_KEY_MAP.get(k, k)] = v
    return out


def _parse_run_dir_name(run_dir_name: str) -> dict:
    """
    Parse a run directory name like:
        slice-cls_resnet50_p128_s64
        sa-slice-cls_resnet50_p64_s32
        volume-cls_resnet50_p64_s32_K16
        patch-cls_resnet50_p128
    Returns a dict with model_family, backbone, patch_size, stride, K.
    """
    name = run_dir_name

    # K (volume only)
    m_k = re.search(r'_K(\d+)', name)
    K = int(m_k.group(1)) if m_k else None

    # patch_size and stride
    m_p = re.search(r'_p(\d+)', name)
    m_s = re.search(r'_s(\d+)', name)
    patch_size = int(m_p.group(1)) if m_p else None
    stride     = int(m_s.group(1)) if m_s else None

    # backbone: everything between the first underscore after the family prefix and _p{N}
    # e.g. sa-slice-cls_resnet50_p64_s32  →  backbone = resnet50
    backbone_match = re.search(r'(?:patch-cls|sa-slice-cls|slice-cls|volume-cls)_([^_]+(?:_[^_]+)*?)_p\d+', name)
    if backbone_match:
        backbone = backbone_match.group(1)
    else:
        # patch-cls without stride: patch-cls_resnet50_p128
        backbone_match2 = re.search(r'(?:patch-cls)_(.+)_p\d+', name)
        backbone = backbone_match2.group(1) if backbone_match2 else 'unknown'

    # model family
    if name.startswith('sa-slice-cls'):
        family = 'SA-ABMIL-slice'
    elif name.startswith('slice-cls'):
        family = 'ABMIL-slice'
    elif name.startswith('sa-volume-cls'):
        family = 'SA-ABMIL-volume'
    elif name.startswith('volume-cls'):
        family = 'ABMIL-volume'
    elif name.startswith('patch-cls'):
        family = 'PatchCLS'
    else:
        family = name.split('_')[0]

    return {
        'model_family': family,
        'backbone':     backbone,
        'patch_size':   patch_size,
        'stride':       stride,
        'K':            K,
    }


def _parse_phase(experiment_folder: str) -> str:
    mapping = {
        'analysis':      'A-patch',
        'ABMIL':         'B-slice / C-volume',
        'SelfAttention': 'B-slice (SA) / C-volume (SA)',
    }
    return mapping.get(experiment_folder, experiment_folder)


def _trained_on(sub_dir_name: str) -> str:
    """Extract the 'trained_on_*' part from the sub-directory name."""
    m = re.search(r'trained_on_([^_/]+(?:\+[^_/]+)*)', sub_dir_name)
    if m:
        return m.group(1)
    # fallback: use the raw dir name
    return sub_dir_name


def collect() -> pd.DataFrame:
    rows = []

    for metrics_path in sorted(EXPERIMENTS_DIR.rglob('test_metrics.json')):
        parts = metrics_path.relative_to(EXPERIMENTS_DIR).parts
        # Expected structure:  <experiment>/ runs/ <run_dir>/ <sub_dir>/ test_metrics.json
        # or (analysis):       analysis/ runs/ <run_dir>/ test_metrics.json  (no sub_dir)
        if len(parts) < 3:
            continue

        experiment_folder = parts[0]   # ABMIL / SelfAttention / analysis
        run_dir_name      = parts[2]   # e.g. slice-cls_resnet50_p128_s64

        if len(parts) == 4:
            # experiment / runs / run_dir / test_metrics.json
            sub_dir_name = run_dir_name   # no trained_on subfolder
            trained_on   = 'all'
        else:
            # experiment / runs / run_dir / sub_dir / test_metrics.json
            sub_dir_name = parts[3]
            trained_on   = _trained_on(sub_dir_name)

        parsed = _parse_run_dir_name(run_dir_name)
        phase  = _parse_phase(experiment_folder)

        raw = json.loads(metrics_path.read_text())
        raw = _normalise_keys(raw)

        row: dict = {
            'phase':        phase,
            'experiment':   experiment_folder,
            **parsed,
            'trained_on':   trained_on,
        }

        # Overall metrics
        for col in OVERALL_COLS:
            row[col] = raw.get(col, float('nan'))

        # Per-modality metrics
        for col in MOD_COLS:
            # handle both 'cycle_acc' and legacy 'cycle_accuracy' keys
            val = raw.get(col, float('nan'))
            if pd.isna(val):
                # try the 'accuracy' variant for older keys
                alt = col.replace('_acc', '_accuracy')
                val = raw.get(alt, float('nan'))
            row[col] = val

        rows.append(row)

    df = pd.DataFrame(rows)

    # ── Sort ─────────────────────────────────────────────────────────────────
    phase_order  = {'A-patch': 0, 'B-slice / C-volume': 1, 'B-slice (SA) / C-volume (SA)': 2}
    family_order = {'PatchCLS': 0, 'ABMIL-slice': 1, 'ABMIL-volume': 2,
                    'SA-ABMIL-slice': 3, 'SA-ABMIL-volume': 4}
    trained_order = {'all': 0, 'pix2pix': 1, 'cycle': 2, 'diffusion': 3}

    df['_phase_ord']   = df['phase'].map(phase_order).fillna(99)
    df['_family_ord']  = df['model_family'].map(family_order).fillna(99)
    df['_trained_ord'] = df['trained_on'].map(trained_order).fillna(99)

    df = df.sort_values(
        ['_phase_ord', '_family_ord', 'backbone', 'patch_size', 'stride', '_trained_ord'],
        ascending=True,
        na_position='last',
    ).drop(columns=['_phase_ord', '_family_ord', '_trained_ord'])

    # Round all floats to 4 decimal places
    float_cols = df.select_dtypes(include='float').columns
    df[float_cols] = df[float_cols].round(4)

    # Final column order
    extra_id = ['experiment']
    all_metric_cols = OVERALL_COLS + MOD_COLS
    present_metric_cols = [c for c in all_metric_cols if c in df.columns]
    remaining = [c for c in df.columns if c not in ID_COLS + extra_id + present_metric_cols]
    final_cols = ID_COLS + extra_id + present_metric_cols + remaining
    df = df[[c for c in final_cols if c in df.columns]]

    return df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description='Collect all test_metrics.json into a CSV')
    parser.add_argument('--out', type=str, default=None,
                        help='Output CSV path (default: scripts/results/all_results.csv)')
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else (
        Path(__file__).resolve().parent / 'results' / 'all_results.csv'
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = collect()
    df.to_csv(out_path, index=False)

    # Pretty-print a compact view
    view_cols = ['phase', 'model_family', 'backbone', 'patch_size', 'stride', 'K',
                 'trained_on', 'auc', 'accuracy', 'ood_test_auc', 'ood_test_accuracy']
    view_cols = [c for c in view_cols if c in df.columns]
    print(df[view_cols].to_string(index=True, na_rep='—'))
    print(f"\n✓  Full CSV ({len(df)} rows × {len(df.columns)} cols) → {out_path}")


if __name__ == '__main__':
    main()
