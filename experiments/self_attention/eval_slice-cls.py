#!/usr/bin/env python3
"""
eval_slice-cls.py  [SA-ABMIL variant]
======================================
Phase B standalone evaluation: replicates exactly the end-of-training test
phase from train_slice-cls.py (SA variant).

Loads best_model.pt from a run directory, rebuilds the OOD test dataset
(same logic as training), and produces identical outputs:
  - evaluation/metrics.json
  - evaluation/per_sample_metrics.csv
  - evaluation/grid_removal.png / grid_injection.png
  - evaluation/vis/*.png  (if --save_vis)

Identical results are guaranteed because this script imports and calls
the exact same functions from train_slice-cls.py via importlib.

Usage:
    python experiments/SelfAttention/eval_slice-cls.py \\
        --run_dir experiments/SelfAttention/runs/sa-slice-cls_resnet50_p128_s64/trained_on_pix2pix_bce_bs16_lr1e-04
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# ── Import train module to guarantee identical code paths ────────────────────
# (importlib handles the hyphen in the filename)
_TRAIN_PY = Path(__file__).resolve().parent / 'train_slice-cls.py'
if not _TRAIN_PY.exists():
    sys.exit(f'[ERROR] train_slice-cls.py not found at {_TRAIN_PY}')

_spec = importlib.util.spec_from_file_location('train_slice_cls_sa', _TRAIN_PY)
_tm   = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tm)

# Aliases to train-module symbols used below
DATA_DIR                      = _tm.DATA_DIR
ALL_FAKES                     = _tm.ALL_FAKES
SliceDataset                  = _tm.SliceDataset
load_split_table              = _tm.load_split_table
run_test_evaluation_slice     = _tm.run_test_evaluation_slice
build_sa_classifier_scratch   = _tm.build_sa_classifier_scratch

# =============================================================================
#  CLI
# =============================================================================

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Phase B eval (SA-ABMIL) — identical to end-of-training test evaluation'
    )
    parser.add_argument('--run_dir',     type=str, required=True,
                        help='Path to a Phase B SA run directory (must contain '
                             'best_model.pt and args.json)')
    parser.add_argument('--batch_size',  type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu_id',      type=int, default=None,
                        help='GPU id to use (default: cuda:0 if available)')
    parser.add_argument('--save_vis',    action='store_true', default=True,
                        help='Save per-sample visualisations in evaluation/vis/')
    parser.add_argument('--max_vis',     type=int, default=20,
                        help='Maximum number of individual vis to save')
    return parser.parse_args()

# =============================================================================
#  Main
# =============================================================================

def main() -> None:
    args = get_args()

    run_dir   = Path(args.run_dir)
    ckpt_path = run_dir / 'best_model.pt'
    args_path = run_dir / 'args.json'

    if not ckpt_path.exists():
        sys.exit(f'[ERROR] No best_model.pt found in {run_dir}')
    if not args_path.exists():
        sys.exit(f'[ERROR] No args.json found in {run_dir}')

    with open(args_path) as f:
        train_args = json.load(f)

    # ── Reconstruct modality sets (mirrors main() in train script) ────────────
    raw_mods   = train_args.get('mods')
    fake_mods  = [m for m in raw_mods if m != 'real'] if raw_mods else ALL_FAKES
    ood_fakes  = [m for m in ALL_FAKES if m not in set(fake_mods)]
    seed       = int(train_args.get('seed', 42))
    patch_size = int(train_args['patch_size'])
    stride     = train_args.get('stride') or (patch_size // 2)
    sa_n_heads = int(train_args.get('sa_n_heads', 8))
    sa_n_layers= int(train_args.get('sa_n_layers', 2))

    # ── Device ───────────────────────────────────────────────────────────────
    if args.gpu_id is not None:
        device = torch.device(f'cuda:{args.gpu_id}')
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    print(f"\n{'='*60}")
    print(f"  Phase B Evaluation — SA-ABMIL slice classifier")
    print(f"{'='*60}")
    print(f"  Run dir       : {run_dir}")
    print(f"  Device        : {device}")
    print(f"  Patch size    : {patch_size}   Stride: {stride}")
    print(f"  SA heads/layers: {sa_n_heads}/{sa_n_layers}")
    print(f"  In-domain     : {fake_mods}")
    print(f"  OOD fakes     : {ood_fakes}")
    print(f"  Test fakes    : {ALL_FAKES}")

    # ── Load model ───────────────────────────────────────────────────────────
    print("\nLoading model...")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    model = build_sa_classifier_scratch(
        backbone    = train_args.get('backbone', 'resnet50'),
        pretrained  = False,
        proj_dim    = train_args.get('proj_dim',   512),
        attn_dim    = train_args.get('attn_dim',   128),
        dropout     = train_args.get('dropout',   0.25),
        sa_n_heads  = sa_n_heads,
        sa_n_layers = sa_n_layers,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"  Loaded epoch {ckpt['epoch']}  |  "
          f"best_val_loss={ckpt['best_val_loss']:.4f}  "
          f"best_val_auc={ckpt['best_val_auc']:.4f}")

    # ── Build test dataloader (mirrors build_dataloaders in train script) ──────
    print(f"\nBuilding test dataset (fakes: {ALL_FAKES})...")
    tab_test = load_split_table(DATA_DIR, 'test', ['real'] + ALL_FAKES)
    ds_test  = SliceDataset(DATA_DIR, tab_test, patch_size=patch_size,
                             stride=stride, augment=False)
    dl_test  = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)
    print(f"  {len(ds_test):6d} slices  ({len(dl_test)} batches)")

    # ── Run test evaluation (exact same function as end of training) ──────────
    print("\nEvaluating...")
    eval_dir = run_dir / 'evaluation'
    results  = run_test_evaluation_slice(
        model           = model,
        dl_test         = dl_test,
        device          = device,
        eval_dir        = eval_dir,
        epoch           = int(ckpt['epoch']),
        run_dir         = run_dir,
        patch_size      = patch_size,
        stride          = stride,
        in_domain_fakes = set(fake_mods),
        seed            = seed,
        save_vis        = args.save_vis,
        max_vis         = args.max_vis,
    )
    cls_m = results['classification']

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  Classification metrics")
    print(f"{'─'*60}")
    print(f"  Overall    : AUC {cls_m['auc']:.4f}  ACC {cls_m['accuracy']:.4f}"
          f"  AP {cls_m['ap']:.4f}  F1 {cls_m['f1']:.4f}")
    print(f"  {'─'*56}")
    for m in sorted(ALL_FAKES):
        a  = cls_m.get(f'{m}_auc', float('nan'))
        ac = cls_m.get(f'{m}_acc', float('nan'))
        ap = cls_m.get(f'{m}_ap',  float('nan'))
        f1 = cls_m.get(f'{m}_f1',  float('nan'))
        print(f"  {m:<12s}  AUC {a:.4f}  ACC {ac:.4f}  AP {ap:.4f}  F1 {f1:.4f}")

    print(f"\n  Outputs saved to: {eval_dir}")
    print("Done.")

if __name__ == '__main__':
    main()
