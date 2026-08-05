#!/usr/bin/env python3

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_MODS = ["pix2pix", "cycle", "diffusion"]

_TO_ORDER = {"all": 0, "pix2pix": 1, "cycle": 2, "diffusion": 3}
_MODEL_ORDER = {
    "flat_cnn":         0,
    "d3":               1,
    "r3d":              2,
    "3d_resnet":        3,
    "3d_densenet":      4,
    "3d_vit":           5,
    "vivit_f":          6,
    "3d_mvit":          7,
    "hp_fcn":           8,
    "trufor":           9,
    "trufor_mitb2":    10,
    "trufor_full":     11,
    "mvssnet":         12,
    "mvssnet_full":    13,
    "mantranet":       14,
}

_FLAT_RE          = re.compile(r"^flat_cnn_(?P<backbone>[^_]+(?:_[^_K][^_]*)*)_K(?P<K>\d+)$")
_R3D_RE           = re.compile(r"^(?P<arch>r3d\d+|mc3_\d+)_K(?P<K>\d+)$")
_D3_RE            = re.compile(r"^d3_K(?P<K>\d+)$")
_3D_RESNET_RE     = re.compile(r"^3d_resnet$")
_3D_DENSENET_RE   = re.compile(r"^3d_densenet$")
_3D_VIT_RE        = re.compile(r"^3d_vit(?:_.+)?$")
_3D_MVIT_RE       = re.compile(r"^3d_mvit(?:_.+)?$")
_HPFCN_RE         = re.compile(r"^hp_fcn_K(?P<K>\d+)$")
_TRUFOR_RE        = re.compile(r"^trufor_K(?P<K>\d+)$")
_TRUFOR_MITB2_RE  = re.compile(r"^trufor_mitb2_K(?P<K>\d+)$")
_TRUFOR_FULL_RE   = re.compile(r"^trufor_full_K(?P<K>\d+)$")
_MVSSNET_RE       = re.compile(r"^mvssnet_K(?P<K>\d+)$")
_MVSSNET_FULL_RE  = re.compile(r"^mvssnet_full_K(?P<K>\d+)$")
_MANTRANET_RE     = re.compile(r"^mantranet_K(?P<K>\d+)$")


def _nan() -> float:
    return float("nan")


def _g(d: dict, *keys) -> float:
    for k in keys:
        if k in d:
            return float(d[k])
    return _nan()


def _trained_on(sub_name: str) -> str:
    m = re.match(r"trained_on_(.+)", sub_name)
    return m.group(1) if m else sub_name


def _find_metrics(sub: Path) -> Path | None:
    for p in [
        sub / "evaluation" / "metrics.json",
        sub / "test_metrics.json",
        sub / "results.json",
    ]:
        if p.exists():
            return p
    return None


def _load_args(sub: Path) -> dict:
    p = sub / "args.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _per_mod_cls(per_mod: dict) -> dict:
    out = {}
    for mod in _MODS:
        md = per_mod.get(mod, {})
        out[f"{mod}_auc"] = _g(md, "auc")
        out[f"{mod}_acc"] = _g(md, "acc", "accuracy")
        out[f"{mod}_f1"]  = _g(md, "f1")
        out[f"{mod}_ap"]  = _g(md, "ap")
    return out


def _cls_row(block: dict) -> dict:
    row = dict(
        auc = _g(block, "auc"),
        acc = _g(block, "accuracy", "acc"),
        f1  = _g(block, "f1"),
        ap  = _g(block, "ap"),
    )
    row.update(_per_mod_cls(block.get("per_mod", {})))
    return row


def _per_mod_loc(per_mod: dict) -> dict:
    out = {}
    for mod in _MODS:
        md = per_mod.get(mod, {})
        out[f"{mod}_loc_pixel_auc"]     = _g(md, "pixel_auc")
        out[f"{mod}_loc_pointing_game"] = _g(md, "pointing_game")
        out[f"{mod}_loc_energy_on_mask"]= _g(md, "energy_on_mask")
        out[f"{mod}_loc_iou_03"]        = _g(md, "iou_0.3")
        out[f"{mod}_loc_iou_05"]        = _g(md, "iou_0.5")
        out[f"{mod}_loc_iou_07"]        = _g(md, "iou_0.7")
    return out


def _loc_row(loc: dict) -> dict:
    row = dict(
        loc_pixel_auc      = _g(loc, "pixel_auc"),
        loc_pointing_game  = _g(loc, "pointing_game"),
        loc_energy_on_mask = _g(loc, "energy_on_mask"),
        loc_iou_03         = _g(loc, "iou_0.3"),
        loc_iou_05         = _g(loc, "iou_0.5"),
        loc_iou_07         = _g(loc, "iou_0.7"),
    )
    row.update(_per_mod_loc(loc.get("per_mod", {})))
    return row


# Generic helpers to avoid duplicating collector bodies

def _collect_2stage(run_dir: Path, regex: re.Pattern,
                    model: str, arch: str, backbone: str,
                    vol_agg: str = "max") -> list[dict]:
    m = regex.match(run_dir.name)
    if not m:
        return []
    K = int(m.group("K")) if "K" in m.groupdict() and m.group("K") else None
    rows = []
    for sub in sorted(run_dir.iterdir()):
        if not sub.is_dir():
            continue
        mp = _find_metrics(sub)
        if mp is None:
            continue
        d    = json.loads(mp.read_text())
        args = _load_args(sub)
        if K is None:
            K = args.get("K")
        base = dict(
            model      = model,
            arch       = arch,
            K          = K,
            pool_mode  = None,
            vol_agg    = vol_agg,
            backbone   = backbone,
            trained_on = _trained_on(sub.name),
        )
        for stage in ("slice", "volume"):
            if stage not in d:
                continue
            row = dict(**base, stage=stage)
            row.update(_cls_row(d[stage]))
            if "localization" in d and stage == "volume":
                row.update(_loc_row(d["localization"]))
            rows.append(row)
    return rows


def _collect_volume_flat(run_dir: Path, regex: re.Pattern,
                         model: str, arch_fn,
                         backbone: str | None = None) -> list[dict]:
    m = regex.match(run_dir.name)
    if not m:
        return []
    rows = []
    for sub in sorted(run_dir.iterdir()):
        if not sub.is_dir():
            continue
        mp = _find_metrics(sub)
        if mp is None:
            continue
        d    = json.loads(mp.read_text())
        args = _load_args(sub)
        K    = args.get("K", int(m.group("K")) if "K" in m.groupdict() and m.group("K") else None)
        # Skip if the file has a stage wrapper (wrong helper called)
        block = d
        if "volume" in d and isinstance(d["volume"], dict) and "auc" in d.get("volume", {}):
            block = d["volume"]
        row = dict(
            model      = model,
            arch       = arch_fn(args),
            stage      = "volume",
            K          = K,
            pool_mode  = None,
            vol_agg    = None,
            backbone   = backbone,
            trained_on = _trained_on(sub.name),
        )
        row.update(_cls_row(block))
        if "localization" in d:
            row.update(_loc_row(d["localization"]))
        rows.append(row)
    return rows


def _collect_flat_cnn(run_dir: Path) -> list[dict]:
    m = _FLAT_RE.match(run_dir.name)
    if not m:
        return []
    backbone = m.group("backbone")
    K        = int(m.group("K"))
    rows = []
    for sub in sorted(run_dir.iterdir()):
        if not sub.is_dir():
            continue
        mp = _find_metrics(sub)
        if mp is None:
            continue
        d    = json.loads(mp.read_text())
        args = _load_args(sub)
        base = dict(
            model      = "flat_cnn",
            arch       = "ResNet-50",
            K          = K,
            pool_mode  = None,
            vol_agg    = args.get("vol_agg", "max"),
            backbone   = backbone,
            trained_on = _trained_on(sub.name),
        )
        for stage in ("slice", "volume"):
            if stage not in d:
                continue
            row = dict(**base, stage=stage)
            row.update(_cls_row(d[stage]))
            rows.append(row)
    return rows


def _collect_r3d(run_dir: Path) -> list[dict]:
    return _collect_volume_flat(
        run_dir, _R3D_RE,
        model    = "r3d",
        arch_fn  = lambda a: a.get("arch", run_dir.name.split("_K")[0]),
        backbone = None,
    )


def _collect_d3(run_dir: Path) -> list[dict]:
    return _collect_2stage(run_dir, _D3_RE, "d3", "D3-ResNet50-LPF", "resnet50")


def _collect_3d_resnet(run_dir: Path) -> list[dict]:
    return _collect_volume_flat(
        run_dir, _3D_RESNET_RE,
        model    = "3d_resnet",
        arch_fn  = lambda a: a.get("arch", "r3d_18"),
        backbone = None,
    )


def _collect_3d_densenet(run_dir: Path) -> list[dict]:
    return _collect_volume_flat(
        run_dir, _3D_DENSENET_RE,
        model    = "3d_densenet",
        arch_fn  = lambda _: "DenseNet121-3D",
        backbone = None,
    )


def _collect_3d_vit(run_dir: Path) -> list[dict]:
    if not _3D_VIT_RE.match(run_dir.name):
        return []
    rows = []
    for sub in sorted(run_dir.iterdir()):
        if not sub.is_dir():
            continue
        mp = _find_metrics(sub)
        if mp is None:
            continue
        d    = json.loads(mp.read_text())
        args = _load_args(sub)
        variant  = args.get("variant", "plain")
        arch_key = args.get("arch", "vit3d_base")
        K        = args.get("K")
        if variant == "factorised":
            model    = "vivit_f"
            arch_str = f"ViViT-F-{arch_key}"
        else:
            model    = "3d_vit"
            arch_str = f"ViT3D-{arch_key}"
        block = d
        if "volume" in d and isinstance(d["volume"], dict) and "auc" in d.get("volume", {}):
            block = d["volume"]
        row = dict(
            model      = model,
            arch       = arch_str,
            stage      = "volume",
            K          = K,
            pool_mode  = None,
            vol_agg    = None,
            backbone   = None,
            trained_on = _trained_on(sub.name),
        )
        row.update(_cls_row(block))
        rows.append(row)
    return rows


def _collect_3d_mvit(run_dir: Path) -> list[dict]:
    return _collect_volume_flat(
        run_dir, _3D_MVIT_RE,
        model    = "3d_mvit",
        arch_fn  = lambda a: a.get("arch", "mvit_v2_s"),
        backbone = None,
    )


def _collect_hp_fcn(run_dir: Path) -> list[dict]:
    m = _HPFCN_RE.match(run_dir.name)
    if not m:
        return []
    K = int(m.group("K"))
    rows = []
    for sub in sorted(run_dir.iterdir()):
        if not sub.is_dir():
            continue
        mp = _find_metrics(sub)
        if mp is None:
            continue
        d     = json.loads(mp.read_text())
        block = d.get("volume", d)
        row   = dict(
            model      = "hp_fcn",
            arch       = "HP-FCN",
            stage      = "volume",
            K          = K,
            pool_mode  = "max",
            vol_agg    = "max",
            backbone   = "resnet-small",
            trained_on = _trained_on(sub.name),
        )
        row.update(_cls_row(block))
        if "localization" in d:
            row.update(_loc_row(d["localization"]))
        rows.append(row)
    return rows


def _collect_trufor(run_dir: Path) -> list[dict]:
    m = _TRUFOR_RE.match(run_dir.name)
    if not m:
        return []
    K = int(m.group("K"))
    rows = []
    for sub in sorted(run_dir.iterdir()):
        if not sub.is_dir():
            continue
        mp = _find_metrics(sub)
        if mp is None:
            continue
        d     = json.loads(mp.read_text())
        block = d.get("volume", d)
        row   = dict(
            model      = "trufor",
            arch       = "TruFor-B2",
            stage      = "volume",
            K          = K,
            pool_mode  = "max",
            vol_agg    = "max",
            backbone   = "resnet50",
            trained_on = _trained_on(sub.name),
        )
        row.update(_cls_row(block))
        if "localization" in d:
            row.update(_loc_row(d["localization"]))
        rows.append(row)
    return rows


def _collect_trufor_mitb2(run_dir: Path) -> list[dict]:
    m = _TRUFOR_MITB2_RE.match(run_dir.name)
    if not m:
        return []
    K = int(m.group("K"))
    rows = []
    for sub in sorted(run_dir.iterdir()):
        if not sub.is_dir():
            continue
        mp = _find_metrics(sub)
        if mp is None:
            continue
        d     = json.loads(mp.read_text())
        block = d.get("volume", d)
        row   = dict(
            model      = "trufor_mitb2",
            arch       = "TruFor-MiT-B2",
            stage      = "volume",
            K          = K,
            pool_mode  = "max",
            vol_agg    = "max",
            backbone   = "mit_b2",
            trained_on = _trained_on(sub.name),
        )
        row.update(_cls_row(block))
        if "localization" in d:
            row.update(_loc_row(d["localization"]))
        rows.append(row)
    return rows


def _collect_trufor_full(run_dir: Path) -> list[dict]:
    return _collect_2stage(
        run_dir, _TRUFOR_FULL_RE,
        model    = "trufor_full",
        arch     = "TruFor-Full",
        backbone = "mit_b2",
    )


def _collect_mvssnet(run_dir: Path) -> list[dict]:
    m = _MVSSNET_RE.match(run_dir.name)
    if not m:
        return []
    K = int(m.group("K"))
    rows = []
    for sub in sorted(run_dir.iterdir()):
        if not sub.is_dir():
            continue
        mp = _find_metrics(sub)
        if mp is None:
            continue
        d     = json.loads(mp.read_text())
        block = d.get("volume", d)
        row   = dict(
            model      = "mvssnet",
            arch       = "ResFCN",
            stage      = "volume",
            K          = K,
            pool_mode  = "max",
            vol_agg    = "max",
            backbone   = "resnet50",
            trained_on = _trained_on(sub.name),
        )
        row.update(_cls_row(block))
        if "localization" in d:
            row.update(_loc_row(d["localization"]))
        rows.append(row)
    return rows


def _collect_mvssnet_full(run_dir: Path) -> list[dict]:
    m = _MVSSNET_FULL_RE.match(run_dir.name)
    if not m:
        return []
    K = int(m.group("K"))
    rows = []
    for sub in sorted(run_dir.iterdir()):
        if not sub.is_dir():
            continue
        mp = _find_metrics(sub)
        if mp is None:
            continue
        d     = json.loads(mp.read_text())
        block = d.get("volume", d)
        row   = dict(
            model      = "mvssnet_full",
            arch       = "MVSSNet-Sobel",
            stage      = "volume",
            K          = K,
            pool_mode  = "max",
            vol_agg    = "max",
            backbone   = "resnet50",
            trained_on = _trained_on(sub.name),
        )
        row.update(_cls_row(block))
        if "localization" in d:
            row.update(_loc_row(d["localization"]))
        rows.append(row)
    return rows


def _collect_mantranet(run_dir: Path) -> list[dict]:
    m = _MANTRANET_RE.match(run_dir.name)
    if not m:
        return []
    K = int(m.group("K"))
    rows = []
    for sub in sorted(run_dir.iterdir()):
        if not sub.is_dir():
            continue
        mp = _find_metrics(sub)
        if mp is None:
            continue
        d     = json.loads(mp.read_text())
        block = d.get("volume", d)
        row   = dict(
            model      = "mantranet",
            arch       = "ManTraNet-PT4",
            stage      = "volume",
            K          = K,
            pool_mode  = "max",
            vol_agg    = "max",
            backbone   = "vgg-imc",
            trained_on = _trained_on(sub.name),
        )
        row.update(_cls_row(block))
        if "localization" in d:
            row.update(_loc_row(d["localization"]))
        rows.append(row)
    return rows


_COLLECTORS = [
    _collect_flat_cnn,
    _collect_d3,
    _collect_r3d,
    _collect_3d_resnet,
    _collect_3d_densenet,
    _collect_3d_vit,
    _collect_3d_mvit,
    _collect_hp_fcn,
    _collect_trufor,
    _collect_trufor_mitb2,
    _collect_trufor_full,
    _collect_mvssnet,
    _collect_mvssnet_full,
    _collect_mantranet,
]


def collect_all(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        for fn in _COLLECTORS:
            found = fn(run_dir)
            if found:
                rows.extend(found)
                break

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["_mod_ord"]   = df["trained_on"].map(lambda x: _TO_ORDER.get(x, 9))
    df["_model_ord"] = df["model"].map(lambda x: _MODEL_ORDER.get(x, 99))
    df["_stage_ord"] = df["stage"].map({"slice": 0, "volume": 1})
    df = (df
          .sort_values(["_model_ord", "_stage_ord", "K", "_mod_ord"],
                       na_position="first")
          .drop(columns=["_mod_ord", "_model_ord", "_stage_ord"])
          .reset_index(drop=True))
    return df


def _pretty(df: pd.DataFrame) -> str:
    disp = df.copy()
    skip = {"K"}
    float_cols = [
        c for c in disp.columns
        if disp[c].dtype == float and c not in skip
    ]
    for c in float_cols:
        disp[c] = disp[c].apply(
            lambda x: f"{x:.4f}" if isinstance(x, float) and not np.isnan(x) else "—"
        )
    for c in ("K",):
        if c in disp.columns:
            disp[c] = disp[c].apply(
                lambda x: str(int(x)) if pd.notna(x) and x == x else "—"
            )
    for c in ("pool_mode", "vol_agg", "backbone"):
        if c in disp.columns:
            disp[c] = disp[c].fillna("—")
    return disp.to_string(index=False)


def main():
    here         = Path(__file__).resolve().parent
    default_runs = here / "runs"
    default_out  = here / "results"

    parser = argparse.ArgumentParser(
        description="Aggregate baseline results from runs/ into baselines.csv"
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

    df = collect_all(runs_dir)
    if df.empty:
        print("[WARN] No baseline runs found.", file=sys.stderr)
        sys.exit(0)

    df_slice  = df[df["stage"] == "slice" ].drop(columns=["stage"]).reset_index(drop=True)
    df_volume = df[df["stage"] == "volume"].drop(columns=["stage"]).reset_index(drop=True)

    # Drop loc_* columns from slices (localization is volume-level only)
    loc_cols = [c for c in df_slice.columns if c.startswith("loc_") or "_loc_" in c]
    df_slice = df_slice.drop(columns=loc_cols, errors="ignore")

    print("━━  SLICE RESULTS  ━━")
    print(_pretty(df_slice))
    print("\n━━  VOLUME RESULTS  ━━")
    print(_pretty(df_volume))

    slices_path = out_dir / "slices.csv"
    volume_path = out_dir / "volumes.csv"
    df_slice.to_csv(slices_path,  index=False, float_format="%.6f")
    df_volume.to_csv(volume_path, index=False, float_format="%.6f")
    print(f"\n✓  slices.csv   ({len(df_slice)} rows × {len(df_slice.columns)} cols)  → {slices_path}")
    print(f"✓  volumes.csv  ({len(df_volume)} rows × {len(df_volume.columns)} cols) → {volume_path}")


if __name__ == "__main__":
    main()
