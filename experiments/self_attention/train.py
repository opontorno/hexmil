#!/usr/bin/env python
"""
SelfAttention/train.py  —  Full pipeline: sa-slice-cls → volume-cls in one command.

Arguments that differ between stages are prefixed:
  s_*  → forwarded to train_slice-cls.py   (e.g. --s_epochs, --s_lr)
  v_*  → forwarded to train_volume-cls.py  (e.g. --v_epochs, --v_K)

Arguments shared across both stages (backbone, patch_size, mods, …)
are passed once and forwarded unchanged.

The two stages run fully independently:
  • separate output directories  (sa-slice-cls_* / volume-cls_*)
  • separate WandB runs
  • separate champion/challenger promotion
  • separate checkpoints

For single-stage training use the original scripts directly:
  python train_slice-cls.py  ...
  python train_volume-cls.py ...
"""

import argparse
import subprocess
import sys
from pathlib import Path

ALL_FAKES = ["pix2pix", "cycle", "diffusion"]
HERE      = Path(__file__).parent        # experiments/SelfAttention/

# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SA-ABMIL full pipeline: sa-slice-cls then volume-cls.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Shared ────────────────────────────────────────────────────────────────
    sh = p.add_argument_group("Shared  (forwarded to both stages)")
    sh.add_argument("--backbone",    default="resnet50",
                    choices=["resnet50", "resnet34",
                             "efficientnet_b0", "efficientnet_b4", "densenet121"])
    sh.add_argument("--mods",        nargs="+", default=None,
                    help="Fake modalities to train on (default: all three)")
    sh.add_argument("--patch_size",  type=int,  default=64)
    sh.add_argument("--stride",      type=int,  default=None,
                    help="Sliding-window stride (default: patch_size // 2)")
    sh.add_argument("--proj_dim",    type=int,  default=512)
    sh.add_argument("--seed",        type=int,  default=42)
    sh.add_argument("--num_workers", type=int,  default=4)
    sh.add_argument("--wandb_mode",  default="online",
                    choices=["online", "offline", "disabled"])
    sh.add_argument("--debug",       action="store_true",
                    help="Disable WandB logging")
    sh.add_argument("--gpu_id",      type=int,  default=None,
                    help="Force a specific GPU (default: auto-select)")

    # ── Slice stage ───────────────────────────────────────────────────────────
    sl = p.add_argument_group("Slice stage  [s_ prefix → train_slice-cls.py]")
    sl.add_argument("--s_attn_dim",       type=int,   default=128)
    sl.add_argument("--s_dropout",        type=float, default=0.25)
    sl.add_argument("--s_epochs",         type=int,   default=50)
    sl.add_argument("--s_batch_size",     type=int,   default=32)
    sl.add_argument("--s_lr",             type=float, default=1e-4)
    sl.add_argument("--s_weight_decay",   type=float, default=1e-4)
    sl.add_argument("--s_patience",       type=int,   default=10)
    sl.add_argument("--s_warmup_epochs",  type=int,   default=0)
    sl.add_argument("--s_scheduler",      action="store_true", default=False,
                    help="Enable ReduceLROnPlateau for slice stage")
    sl.add_argument("--s_sa_n_heads",     type=int,   default=8,
                    help="Self-Attention heads in the slice Transformer")
    sl.add_argument("--s_sa_n_layers",    type=int,   default=2,
                    help="Transformer encoder layers in the slice stage")
    sl.add_argument("--s_vis_every",      type=int,   default=5)
    sl.add_argument("--s_vis_n_samples",  type=int,   default=8)
    sl.add_argument("--s_run_name",       default=None,
                    help="Custom run-name for the slice stage")
    sl.add_argument("--aux_attn_loss",    action="store_true", default=False,
                    help="Guide ABMIL α with GT mask coverage (KL loss)")
    sl.add_argument("--aux_weight",       type=float, default=0.3,
                    help="Weight λ of the auxiliary KL attention loss")

    # ── Volume stage ──────────────────────────────────────────────────────────
    vl = p.add_argument_group("Volume stage [v_ prefix → train_volume-cls.py]")
    vl.add_argument("--v_K",             type=int,   default=32,
                    help="Slice-window size K")
    vl.add_argument("--v_attn_dim",      type=int,   default=256)
    vl.add_argument("--v_dropout",       type=float, default=0.25)
    vl.add_argument("--v_epochs",        type=int,   default=60)
    vl.add_argument("--v_batch_size",    type=int,   default=8)
    vl.add_argument("--v_lr",            type=float, default=1e-4)
    vl.add_argument("--v_weight_decay",  type=float, default=1e-4)
    vl.add_argument("--v_patience",      type=int,   default=15)
    vl.add_argument("--v_warmup_epochs", type=int,   default=3)
    vl.add_argument("--v_sa_n_heads",    type=int,   default=8,
                    help="Self-Attention heads in the volume Transformer")
    vl.add_argument("--v_sa_n_layers",   type=int,   default=2,
                    help="Transformer encoder layers in the volume stage")
    vl.add_argument("--v_vis_every",     type=int,   default=5)
    vl.add_argument("--v_vis_n_samples", type=int,   default=4)
    vl.add_argument("--v_run_name",      default=None,
                    help="Custom run-name for the volume stage")

    return p

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _infer_slice_ckpt_dir(args) -> Path:
    """Reconstruct the out_dir that train_slice-cls.py will create.
    Mirrors the naming logic at lines 1199-1214 of train_slice-cls.py exactly.
    """
    fake_mods  = [m for m in (args.mods or ALL_FAKES) if m != "real"]
    stride_eff = args.stride or (args.patch_size // 2)
    mods_tag   = "all" if set(fake_mods) == set(ALL_FAKES) else "+".join(sorted(fake_mods))
    base_dir   = (Path(WORK_DIR) / "experiments" / "SelfAttention" / "runs"
                  / f"sa-slice-cls_{args.backbone}_p{args.patch_size}_s{stride_eff}")
    if args.s_run_name is not None:
        run_tag = args.s_run_name
    else:
        loss_tag = "bce+attn" if args.aux_attn_loss else "bce"
        run_tag  = f"trained_on_{mods_tag}_{loss_tag}_bs{args.s_batch_size}_lr{args.s_lr}"
    return base_dir / run_tag

def _build_slice_argv(args) -> list:
    a = []
    # shared
    a += ["--backbone",    args.backbone]
    a += ["--patch_size",  str(args.patch_size)]
    a += ["--proj_dim",    str(args.proj_dim)]
    a += ["--seed",        str(args.seed)]
    a += ["--num_workers", str(args.num_workers)]
    a += ["--wandb_mode",  args.wandb_mode]
    a += ["--pretrained", "--amp", "--balance_classes"]   # always True in originals
    if args.stride is not None:
        a += ["--stride", str(args.stride)]
    if args.mods:
        a += ["--mods"] + args.mods
    if args.debug:
        a += ["--debug"]
    if args.gpu_id is not None:
        a += ["--gpu_id", str(args.gpu_id)]
    # slice-specific
    a += ["--attn_dim",      str(args.s_attn_dim)]
    a += ["--dropout",       str(args.s_dropout)]
    a += ["--epochs",        str(args.s_epochs)]
    a += ["--batch_size",    str(args.s_batch_size)]
    a += ["--lr",            str(args.s_lr)]
    a += ["--weight_decay",  str(args.s_weight_decay)]
    a += ["--patience",      str(args.s_patience)]
    a += ["--warmup_epochs", str(args.s_warmup_epochs)]
    a += ["--sa_n_heads",    str(args.s_sa_n_heads)]
    a += ["--sa_n_layers",   str(args.s_sa_n_layers)]
    a += ["--vis_every",     str(args.s_vis_every)]
    a += ["--vis_n_samples", str(args.s_vis_n_samples)]
    if args.s_scheduler:
        a += ["--scheduler"]
    if args.s_run_name is not None:
        a += ["--run_name", args.s_run_name]
    if args.aux_attn_loss:
        a += ["--aux_attn_loss"]
    a += ["--aux_weight", str(args.aux_weight)]
    return a

def _build_volume_argv(args, slice_ckpt_dir: Path) -> list:
    a = []
    a += ["--slice_ckpt_dir", str(slice_ckpt_dir)]
    # shared
    a += ["--seed",        str(args.seed)]
    a += ["--num_workers", str(args.num_workers)]
    a += ["--wandb_mode",  args.wandb_mode]
    a += ["--amp", "--balance_classes", "--scheduler"]   # always True in volume originals
    if args.mods:
        a += ["--mods"] + args.mods
    if args.debug:
        a += ["--debug"]
    if args.gpu_id is not None:
        a += ["--gpu_id", str(args.gpu_id)]
    # volume-specific
    a += ["--K",             str(args.v_K)]
    a += ["--attn_dim",      str(args.v_attn_dim)]
    a += ["--dropout",       str(args.v_dropout)]
    a += ["--epochs",        str(args.v_epochs)]
    a += ["--batch_size",    str(args.v_batch_size)]
    a += ["--lr",            str(args.v_lr)]
    a += ["--weight_decay",  str(args.v_weight_decay)]
    a += ["--patience",      str(args.v_patience)]
    a += ["--warmup_epochs", str(args.v_warmup_epochs)]
    a += ["--sa_n_heads",    str(args.v_sa_n_heads)]
    a += ["--sa_n_layers",   str(args.v_sa_n_layers)]
    a += ["--vis_every",     str(args.v_vis_every)]
    a += ["--vis_n_samples", str(args.v_vis_n_samples)]
    if args.v_run_name is not None:
        a += ["--run_name", args.v_run_name]
    return a

def _run(script: Path, extra_argv: list):
    cmd = [sys.executable, str(script)] + extra_argv
    print(f"\n{'=' * 70}")
    print(f"  STAGE: {script.name}")
    print(f"  CMD:   {' '.join(cmd[:4])} ...")
    print(f"{'=' * 70}\n")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n[train.py] ERROR: {script.name} exited with code {result.returncode}.")
        sys.exit(result.returncode)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = build_parser().parse_args()

    # ── Phase 1: slice ────────────────────────────────────────────────────────
    _run(HERE / "train_slice-cls.py", _build_slice_argv(args))

    # ── Locate the checkpoint created by Phase 1 ─────────────────────────────
    slice_ckpt_dir = _infer_slice_ckpt_dir(args)
    ckpt_file      = slice_ckpt_dir / "best_model.pt"
    if not ckpt_file.exists():
        print(f"\n[train.py] ERROR: expected slice checkpoint not found at:\n  {ckpt_file}")
        print("  If you used --s_run_name, make sure it matches exactly.")
        sys.exit(1)
    print(f"\n[train.py] Slice checkpoint confirmed:\n  {slice_ckpt_dir}\n")

    # ── Phase 2: volume ───────────────────────────────────────────────────────
    _run(HERE / "train_volume-cls.py", _build_volume_argv(args, slice_ckpt_dir))

if __name__ == "__main__":
    main()
