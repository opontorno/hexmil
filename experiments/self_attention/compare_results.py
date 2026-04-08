#!/usr/bin/env python3
"""
compare_results.py — Separate slice / volume results tables (SA-ABMIL variant).

Run from the SelfAttention experiment folder:

    python compare_results.py [--runs runs/] [--out_dir results/]

Produces two CSVs:
  results/slices.csv   — one row per slice run
  results/volumes.csv  — one row per volume run

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
slices.csv columns
  Meta       : patch_size  stride  aux_attn_loss
               backbone  trained_on  epoch
  Hyperparams: proj_dim  attn_dim   (from args.json)
  Global cls : auc  acc  precision  recall  f1  ap
  Per-mod cls: {pix2pix,cycle,diffusion}_{auc,acc,f1,ap}

volumes.csv columns
  Meta       : patch_size  stride  K  aux_attn_loss
               backbone  trained_on  epoch
  Hyperparams: vol_attn_dim  slice_attn_dim  sa_n_heads  sa_n_layers
               (from args.json)
  Global cls : auc  acc  ap  f1  pd_at_1
  Per-mod cls: {pix2pix,cycle,diffusion}_{auc,acc,f1,ap,pd_at_1}
  Global XAI : pixel_auc  iou_03  iou_05  iou_07  pointing_game
  Per-mod XAI: same 5 keys × {pix2pix,cycle,diffusion}

Sort order:
  slices  : patch_size → stride → aux_attn_loss → backbone → trained_on
  volumes : patch_size → stride → K → aux_attn_loss → sa_n_heads → sa_n_layers
            → backbone → trained_on
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────
_MODALITIES = ["pix2pix", "cycle", "diffusion"]
_XAI_MODS   = ["pix2pix", "cycle", "diffusion"]
_XAI_KEYS   = ["pixel_auc", "iou_03", "iou_05", "iou_07",
                "pointing_game"]

_TO_ORDER   = {"all": 0, "pix2pix": 1, "cycle": 2, "diffusion": 3}

# ── Run-directory name parser ──────────────────────────────────────────────────
# SA slice dirs have the 'sa-' prefix:  sa-slice-cls_{bb}_p{ps}_s{st}[_attn]
# SA volume dirs do NOT:                volume-cls_{bb}_p{ps}_s{st}_K{K}[_attn]
_RUN_RE = re.compile(
    r"^(?:sa-)?(?P<kind>slice|volume)-cls_"
    r"(?P<backbone>.+?)_"
    r"p(?P<ps>\d+)_s(?P<st>\d+)"
    r"(?:_K(?P<K>\d+))?"
    r"(?P<attn>_attn)?$"
)

def _parse_run(name: str) -> dict | None:
    m = _RUN_RE.match(name)
    if not m:
        return None
    return dict(
        kind       = m.group("kind"),
        backbone   = m.group("backbone"),
        patch_size = int(m.group("ps")),
        stride     = int(m.group("st")),
        K          = int(m.group("K")) if m.group("K") else None,
        _attn_flag = bool(m.group("attn")),
    )

def _trained_on(sub_name: str) -> str:
    m = re.match(r"trained_on_(.+)", sub_name)
    return m.group(1) if m else sub_name

def _nan() -> float:
    return float("nan")

# ── Metrics loading ────────────────────────────────────────────────────────────

def _load_metrics(path: Path) -> tuple[dict, dict, int | None]:
    """Return (cls_dict, xai_dict, epoch).
    cls_dict keys are normalised to {mod}_{metric} format."""
    d     = json.loads(path.read_text())
    epoch = d.get("epoch")
    # cls block: volume → 'cls', slice → 'classification'
    raw   = d.get("cls", d.get("classification", {}))
    cls   = {}
    for k, v in raw.items():
        # Normalise legacy format: auc_cycle → cycle_auc
        mm = re.match(r"^(auc|acc|accuracy|precision|recall|ap|f1)_(\w+)$", k)
        if mm:
            cls[f"{mm.group(2)}_{mm.group(1)}"] = v
        else:
            cls[k] = v
    xai = dict(d.get("xai", {}))
    return cls, xai, epoch

def _g(d: dict, *keys) -> float:
    """Get first existing key from dict, else nan."""
    for k in keys:
        if k in d:
            return d[k]
    return _nan()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SLICE collector
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def collect_slices(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        info = _parse_run(run_dir.name)
        if info is None or info["kind"] != "slice":
            continue

        for sub in sorted(run_dir.iterdir()):
            if not sub.is_dir():
                continue
            mp = sub / "evaluation" / "metrics.json"
            if not mp.exists():
                mp = sub / "test_metrics.json"
                if not mp.exists():
                    continue

            args     = json.loads((sub / "args.json").read_text()) if (sub / "args.json").exists() else {}
            aux_attn = bool(args.get("aux_attn_loss", info["_attn_flag"]))
            proj_dim = args.get("proj_dim", None)
            attn_dim = args.get("attn_dim", None)

            cls, _, epoch = _load_metrics(mp)

            row = dict(
                patch_size    = info["patch_size"],
                stride        = info["stride"],
                aux_attn_loss = aux_attn,
                trained_on    = _trained_on(sub.name),
                backbone      = info["backbone"],
                epoch         = epoch,
                proj_dim      = proj_dim,
                attn_dim      = attn_dim,
                # Global cls
                auc           = _g(cls, "auc"),
                acc           = _g(cls, "accuracy", "acc"),
                precision     = _g(cls, "precision"),
                recall        = _g(cls, "recall"),
                f1            = _g(cls, "f1"),
                ap            = _g(cls, "ap"),
            )
            # Per-modality cls
            for mod in _MODALITIES:
                row[f"{mod}_auc"] = _g(cls, f"{mod}_auc")
                row[f"{mod}_acc"] = _g(cls, f"{mod}_acc", f"{mod}_accuracy")
                row[f"{mod}_f1"]  = _g(cls, f"{mod}_f1")
                row[f"{mod}_ap"]  = _g(cls, f"{mod}_ap")
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["_to_ord"] = df["trained_on"].map(lambda x: _TO_ORDER.get(x, 9))
    df = (df.sort_values(["patch_size", "stride", "aux_attn_loss", "backbone",
                          "_to_ord"], na_position="first")
            .drop(columns=["_to_ord"])
            .reset_index(drop=True))
    return df

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VOLUME collector
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def collect_volumes(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        info = _parse_run(run_dir.name)
        if info is None or info["kind"] != "volume":
            continue

        for sub in sorted(run_dir.iterdir()):
            if not sub.is_dir():
                continue
            mp = sub / "evaluation" / "metrics.json"
            if not mp.exists():
                mp = sub / "test_metrics.json"
                if not mp.exists():
                    continue

            args           = json.loads((sub / "args.json").read_text()) if (sub / "args.json").exists() else {}
            slice_args     = args.get("slice_args", {})
            aux_attn       = bool(slice_args.get("aux_attn_loss", info["_attn_flag"]))
            vol_attn_dim   = args.get("attn_dim", None)
            slice_attn_dim = slice_args.get("attn_dim", None)
            sa_n_heads     = args.get("sa_n_heads", None)
            sa_n_layers    = args.get("sa_n_layers", None)

            cls, xai, epoch = _load_metrics(mp)

            row = dict(
                patch_size     = info["patch_size"],
                stride         = info["stride"],
                K              = info["K"],
                aux_attn_loss  = aux_attn,
                backbone       = info["backbone"],
                trained_on     = _trained_on(sub.name),
                epoch          = epoch,
                vol_attn_dim   = vol_attn_dim,
                slice_attn_dim = slice_attn_dim,
                sa_n_heads     = sa_n_heads,
                sa_n_layers    = sa_n_layers,
                # Global cls
                auc            = _g(cls, "auc"),
                acc            = _g(cls, "accuracy", "acc"),
                ap             = _g(cls, "ap"),
                f1             = _g(cls, "f1"),
                pd_at_1        = _g(cls, "pd_at_1"),
            )
            # Per-modality cls
            for mod in _MODALITIES:
                row[f"{mod}_auc"] = _g(cls, f"{mod}_auc")
                row[f"{mod}_acc"] = _g(cls, f"{mod}_acc", f"{mod}_accuracy")
                row[f"{mod}_f1"]  = _g(cls, f"{mod}_f1")
                row[f"{mod}_ap"]  = _g(cls, f"{mod}_ap")
            # Global XAI
            for k in _XAI_KEYS:
                row[f"xai_{k}"] = _g(xai, k)
            # Per-modality XAI
            for mod in _XAI_MODS:
                for k in _XAI_KEYS:
                    row[f"xai_{k}_{mod}"] = _g(xai, f"{k}_{mod}")
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["_to_ord"] = df["trained_on"].map(lambda x: _TO_ORDER.get(x, 9))
    df = (df.sort_values(["patch_size", "stride", "K", "aux_attn_loss",
                          "sa_n_heads", "sa_n_layers", "backbone", "_to_ord"],
                         na_position="first")
            .drop(columns=["_to_ord"])
            .reset_index(drop=True))
    return df

# ── Pretty printer ─────────────────────────────────────────────────────────────

def _pretty(df: pd.DataFrame) -> str:
    disp = df.copy()
    _int_cols = {"proj_dim", "attn_dim", "vol_attn_dim", "slice_attn_dim",
                 "sa_n_heads", "sa_n_layers", "K", "patch_size", "stride", "epoch"}
    float_cols = [c for c in disp.columns
                  if disp[c].dtype == float and c not in _int_cols]
    for c in float_cols:
        disp[c] = disp[c].apply(
            lambda x: f"{x:.4f}" if isinstance(x, float) and not np.isnan(x) else "—"
        )
    for c in _int_cols:
        if c in disp.columns:
            disp[c] = disp[c].apply(
                lambda x: str(int(x)) if pd.notna(x) and x == x else "—"
            )
    if "aux_attn_loss" in disp.columns:
        disp["aux_attn_loss"] = disp["aux_attn_loss"].apply(lambda x: "yes" if x else "no")
    return disp.to_string(index=False)

# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    here         = Path(__file__).resolve().parent
    default_runs = here / "runs"
    default_out  = here / "results"

    parser = argparse.ArgumentParser(
        description="Produce slices.csv and volumes.csv from SA-ABMIL runs/"
    )
    parser.add_argument("--runs",    default=str(default_runs),
                        help="Path to runs/ directory (default: ./runs/)")
    parser.add_argument("--out_dir", default=str(default_out),
                        help=f"Output directory (default: {default_out})")
    args = parser.parse_args()

    runs_dir = Path(args.runs)
    if not runs_dir.exists():
        print(f"[ERROR] runs/ not found at: {runs_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Slices ─────────────────────────────────────────────────────────────────
    df_sl = collect_slices(runs_dir)
    if df_sl.empty:
        print("[WARN] No slice runs found.", file=sys.stderr)
    else:
        print("━━  SLICE RESULTS  ━━")
        print(_pretty(df_sl))
        out_sl = out_dir / "slices.csv"
        df_sl.to_csv(out_sl, index=False, float_format="%.6f")
        print(f"\n✓  slices.csv  ({len(df_sl)} rows × {len(df_sl.columns)} cols) → {out_sl}")

    print()

    # ── Volumes ────────────────────────────────────────────────────────────────
    df_vol = collect_volumes(runs_dir)
    if df_vol.empty:
        print("[WARN] No volume runs found.", file=sys.stderr)
    else:
        print("━━  VOLUME RESULTS  ━━")
        print(_pretty(df_vol))
        out_vol = out_dir / "volumes.csv"
        df_vol.to_csv(out_vol, index=False, float_format="%.6f")
        print(f"\n✓  volumes.csv ({len(df_vol)} rows × {len(df_vol.columns)} cols) → {out_vol}")

if __name__ == "__main__":
    main()
