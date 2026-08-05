<div align="center">

# HexMIL

### Hierarchical Attention MIL for Ante-Hoc Explainable Detection of AI-Manipulated CT Volumes

<p>
  <a href="#"><img src="https://img.shields.io/badge/ACM%20MM-2026-0d9488.svg" alt="Venue"></a>
  <a href="#"><img src="https://img.shields.io/badge/Paper-PDF-b31b1b.svg" alt="Paper"></a>
  <a href="https://opontorno.github.io/hexmil/"><img src="https://img.shields.io/badge/Project-Page-0f766e.svg" alt="Project Page"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python">
</p>

<img src="docs/static/images/hexmil_main.png" alt="HexMIL overview" width="100%">

</div>

---

## Overview

**HexMIL** is a **mask-free** detector for *medical deepfakes* — CT volumes in which a small
sub-region has been synthetically injected or removed by a deep generative model. It addresses two
limitations of existing detectors at once:

- **Generalization** to unseen generative architectures (cross-generator, out-of-distribution setting).
- **Interpretability**: it *localizes* the manipulated 3D sub-region **without any pixel-level supervision**.

The framework is a two-level **Multiple Instance Learning** hierarchy trained with binary
volume-level labels only. Two independent **Gated Attention** modules — *Patch Gated Attention*
(within a slice) and *Volume Gated Attention* (across slices) — are directly combined into a
full-resolution 3D attention volume. Because these attention weights are the *exact forward
computation* that produces the classification score, the resulting spatial attribution is
**ante-hoc** and structurally faithful — unlike post-hoc approximations such as Grad-CAM.

---

## Method

HexMIL is trained in two sequential stages (see figure above):

1. **SliceMIL (Stage 1)** — a 2D slice is tiled into overlapping patches (`P=64`, stride `S=32`),
   each encoded by a shared ResNet-50. A *Patch Gated Attention* module pools the patch bag into a
   slice representation and a binary real/fake head is trained. The intra-slice attention map
   `α` marks *where* within a slice the model looks.
2. **HexMIL (Stage 2)** — SliceMIL is **frozen** and used as a slice encoder. A window of `K=32`
   slices is encoded, augmented with a sinusoidal positional encoding, and aggregated by a second,
   independent *Volume Gated Attention* module into a volume-level decision. The inter-slice
   attention `β` marks *which* slices carry the manipulation.
3. **Inference & XAI** — a full scan is split into `K`-slice sub-volumes; the highest sub-volume
   score is the final prediction. The XAI Module combines `β × α` into a 3D attention heatmap and an
   axis-aligned 3D bounding box.

---

## Installation

```bash
git clone https://github.com/opontorno/hexmil.git
cd hexmil

conda create -n hexmil python=3.11 -y
conda activate hexmil

# (optional) install a CUDA-matched PyTorch build first — see https://pytorch.org
pip install -e .                 # installs HexMIL + all dependencies (from pyproject.toml)
pip install -e ".[nifti]"        # optional: NIfTI export of 3D attention volumes
```

All dependencies are declared once in `pyproject.toml`; `requirements.txt` simply installs the
package, so `pip install -r requirements.txt` and `pip install -e .` are equivalent.

---

## Configuration

All filesystem paths live in a single file: [`config.py`](./config.py). **The only variable you
must set is `DATA_DIR`:**

```python
DATA_DIR = "/path/to/M3DSynth"   # root holding data.csv + sets.csv
```

`WORK_DIR` and `RUNS_DIR` are derived automatically from the repository location. Every value can
also be overridden via environment variables (`HEXMIL_DATA_DIR`, `HEXMIL_WORK_DIR`,
`HEXMIL_RUNS_DIR`). Training/evaluation hyper-parameters are **not** configured here — they are
command-line arguments of each script.

---

## Data

HexMIL is evaluated on two public benchmarks built on **LIDC-IDRI**:

- **[M3DSynth](https://github.com/grip-unina/M3DSynth)** — nodule injection/removal by Pix2Pix,
  CycleGAN and a Diffusion Model.
- **CT-GAN** — conditional-GAN nodule injection/removal.

Both datasets are publicly released by their respective authors and are **not redistributed here**.
Point `DATA_DIR` to the M3DSynth root, which is expected to contain:

```
data/
├── data.csv                    # per-volume metadata (img_id, mod, ty, coord_z/y/x, …)
├── sets.csv                    # patient-level train/valid/test split
├── ctgan_data.csv              # (optional) CT-GAN metadata
├── ctgan_sets.csv              # (optional) CT-GAN split
├── real/      scan/<img_id>/   # multi-page TIFF stacks (one page = one axial slice)
├── pix2pix/   scan/…  label/…  # `label/` = binary manipulation masks (eval only)
├── cycle/     scan/…  label/…
├── diffusion/ scan/…  label/…
└── ctgan/     scan/…  label/…
```

Modalities are selected per run via `--mods`. Training on a **single** modality (e.g. `pix2pix`)
turns the remaining ones into the out-of-distribution test set; omitting `--mods` trains on all.

---

## Training

**Full two-stage pipeline** (trains SliceMIL then HexMIL with a single command):

```bash
python train.py --mods pix2pix          # cross-generator: train on pix2pix, test on the rest
python train.py                         # in-domain: train on all modalities
```

**Or run each stage individually:**

```bash
# Stage 1 — SliceMIL  →  runs/slicemil_resnet50_p64_s32/trained_on_pix2pix/
python train_slicemil.py --backbone resnet50 --patch_size 64 --mods pix2pix

# Stage 2 — HexMIL     →  runs/hexmil_resnet50_p64_s32_K32/trained_on_pix2pix/
python train_hexmil.py \
    --slice_ckpt_dir runs/slicemil_resnet50_p64_s32/trained_on_pix2pix \
    --K 32
```

To reproduce the full **cross-generator protocol**, repeat with each
`--mods {pix2pix, cycle, diffusion, ctgan}`; every held-out generator becomes the OOD test set.

Each run directory contains `best_model.pt`, `args.json` (full reproducibility record),
per-split metrics, and periodic visualizations. Useful flags: `--gpu_id`,
and `--wandb_mode` to enable logging (disabled by default).

---

## Evaluation & Explainability

```bash
# Slice-level metrics (Stage 1)
python eval_slice.py  --run_dir runs/slicemil_resnet50_p64_s32/trained_on_pix2pix

# Volume-level metrics + 3D attention heatmaps (Stage 2) — full volume by default
python eval_volume.py --run_dir runs/hexmil_resnet50_p64_s32_K32/trained_on_pix2pix --save_3d

# XAI benchmark: HexMIL attention vs. Grad-CAM / Grad-CAM++ (pixel-AUC, IoU, Pointing Game)
python eval_xai.py    --run_dir runs/slicemil_resnet50_p64_s32/trained_on_pix2pix
```

---

## Inference

Run a trained HexMIL model on a single scan (TIFF stack) — no ground-truth labels required. The
whole volume is swept with sliding `K`-slice windows and the output includes the tampered/pristine
prediction plus the 3D attention heatmap and bounding box:

```bash
python inference.py --run_dir runs/hexmil_resnet50_p64_s32_K32/trained_on_pix2pix \
                     --scan_dir /path/to/scan --save_3d
```

Add `--label_dir /path/to/masks` if ground-truth manipulation masks are available (for visual
comparison only), and `--save_nifti` to also export the volume and heatmap as `.nii.gz`.

---

## Repository structure

```
hexmil/
├── config.py                 # single source of truth for all paths
├── train.py                  # full SliceMIL → HexMIL pipeline
├── train_slicemil.py         # Stage 1
├── train_hexmil.py           # Stage 2
├── eval_slice.py             # Stage 1 evaluation
├── eval_volume.py            # Stage 2 evaluation + 3D heatmaps
├── eval_xai.py               # XAI benchmark (attention vs. Grad-CAM)
├── inference.py              # single-scan inference
├── src/hexmil/               # installable package (data / models / utils)
├── baselines/                # SOTA comparison suite (+ clone_repos.sh)
└── docs/                     # project page (GitHub Pages)
```

---

## Citation

```bibtex
@inproceedings{pontorno2026hexmil,
  title     = {{HexMIL: Hierarchical Attention MIL for Ante-Hoc Explainable Detection of AI-Manipulated CT Volumes}},
  author={Pontorno, Orazio and Guarnera, Luca and Akhtar, Zahid and Battiato, Sebastiano},
  booktitle={Proceedings of the 34th ACM International Conference on Multimedia},
  year      = {2026},
}
```

---

## License

Released under the [MIT License](./LICENSE). Baseline code retains the license of its original
repository.
