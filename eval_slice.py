#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# (importlib handles the hyphen in the filename)
_TRAIN_PY = Path(__file__).resolve().parent / 'train_slicemil.py'
if not _TRAIN_PY.exists():
    sys.exit(f'[ERROR] train_slicemil.py not found at {_TRAIN_PY}')

_spec = importlib.util.spec_from_file_location('train_slicemil', _TRAIN_PY)
_tm   = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tm)

# Aliases to train-module symbols used below
DATA_DIR                        = _tm.DATA_DIR
ALL_FAKES                       = _tm.ALL_FAKES
SliceDataset                    = _tm.SliceDataset
load_split_table                = _tm.load_split_table
run_test_evaluation_slice       = _tm.run_test_evaluation_slice
build_slicemil  = _tm.build_slicemil


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Stage 1 eval — identical to end-of-training test evaluation'
    )
    parser.add_argument('--run_dir',     type=str, required=True,
                        help='Path to a Stage 1 run directory (must contain '
                             'best_model.pt and args.json)')
    parser.add_argument('--batch_size',  type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu_id',      type=int, default=None,
                        help='GPU id to use (default: cuda:0 if available)')
    return parser.parse_args()


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

    raw_mods       = train_args.get('mods')          # list or None
    fake_mods      = [m for m in raw_mods if m != 'real'] if raw_mods else ALL_FAKES
    ood_fakes      = [m for m in ALL_FAKES if m not in set(fake_mods)]
    seed           = int(train_args.get('seed', 42))
    patch_size     = int(train_args['patch_size'])
    stride         = train_args.get('stride') or (patch_size // 2)

    if args.gpu_id is not None:
        device = torch.device(f'cuda:{args.gpu_id}')
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    print(f"\n{'='*60}")
    print(f"  Stage 1 Evaluation — ABMIL slice classifier")
    print(f"{'='*60}")
    print(f"  Run dir       : {run_dir}")
    print(f"  Device        : {device}")
    print(f"  Patch size    : {patch_size}   Stride: {stride}")
    print(f"  In-domain     : {fake_mods}")
    print(f"  OOD fakes     : {ood_fakes}")
    print(f"  Test fakes    : {ALL_FAKES}")

    print("\nLoading model...")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    model = build_slicemil(
        backbone   = train_args.get('backbone',   'resnet50'),
        pretrained = False,
        patch_size = patch_size,
        proj_dim   = train_args.get('proj_dim',   512),
        attn_dim   = train_args.get('attn_dim',   128),
        dropout    = train_args.get('dropout',   0.25),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"  Loaded epoch {ckpt['epoch']}  |  "
          f"best_val_loss={ckpt['best_val_loss']:.4f}  "
          f"best_val_auc={ckpt['best_val_auc']:.4f}")

    print(f"\nBuilding test dataset (fakes: {ALL_FAKES})...")
    tab_test = load_split_table(DATA_DIR, 'test', ['real'] + ALL_FAKES)
    ds_test  = SliceDataset(DATA_DIR, tab_test, patch_size=patch_size,
                             stride=stride, augment=False)
    dl_test  = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)
    print(f"  {len(ds_test):6d} slices  ({len(dl_test)} batches)")

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
    )
    cls_m = results['classification']

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
