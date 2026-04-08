# MedForensics

> Detection and localization of synthetic manipulations in 3D medical CT scans using attention-based deep learning.

---

## Overview

Medical imaging is increasingly vulnerable to AI-generated forgeries. Generative models can convincingly inject or remove pathological findings — such as pulmonary nodules — in CT scans, creating false evidence that could mislead clinical decisions or corrupt research datasets.

**MedForensics** addresses this threat by developing a detection system that can:

1. **Classify** a CT scan as real or synthetically manipulated
2. **Localize** the manipulated region without ever training a segmentation model

The core principle is to use **attention-based explainability (XAI)** as a byproduct of classification: the same attention weights that drive the binary decision also highlight *where* in the volume the manipulation likely occurred. This eliminates the need for pixel-level annotation during training — we only need image-level labels.

---

## Dataset

The project uses **M3DSynth**, a benchmark dataset of lung CT volumes with synthetically manipulated nodule regions. Each volume is available in four variants:

| Modality | Description |
|----------|-------------|
| `real` | Original, unmodified CT scan |
| `pix2pix` | Nodule injected/removed by a Pix2Pix GAN |
| `cycle` | Nodule injected/removed by a CycleGAN |
| `diffusion` | Nodule injected/removed by a Diffusion model |

Ground-truth pixel-level manipulation masks are provided for all fake samples, enabling quantitative evaluation of localization quality.

---

## Approach

The detection pipeline is hierarchical, processing CT data at progressively coarser scales:

```
Single patch → Full axial slice → 3D volume window
   (Phase A)      (Phase B)          (Phase C)
```

**Phase A — Patch-level feasibility**  
A CNN is trained on individual 2D patches cropped around the known manipulation coordinates. This confirms that forgery traces are visually detectable at the patch level before scaling up.

**Phase B — Slice-level detection with spatial attention**  
A full axial slice is divided into a grid of overlapping patches. A CNN backbone extracts per-patch features, and an attention mechanism aggregates them into a slice-level decision. The resulting attention map over the patch grid naturally highlights which spatial regions the model finds most suspicious — providing interpretable localization without segmentation supervision.

**Phase C — Volume-level detection**  
A window of K consecutive axial slices centered on the annotated region is processed by the frozen Phase B encoder. A second attention mechanism aggregates slice-level features along the Z-axis, producing a volume-level binary decision and a 3D attention heatmap (patch-attention within slice × slice-attention across volume).

---

## Key Features

- **No segmentation supervision** — localization emerges from classification attention
- **Quantitative XAI evaluation** — attention heatmaps are benchmarked against ground-truth masks (IoU, Pointing Game, pixel-level AUC)
- **Cross-domain generalization** — models are evaluated on unseen GAN architectures to measure robustness
- **Modular, reproducible** — each phase is independently trainable and evaluable

---

## Repository Structure

```
MedForensics/
├── src/medforensics/
│   ├── data/              # Dataset classes and preprocessing utilities
│   │   ├── patch_dataset.py
│   │   ├── slice_dataset.py
│   │   └── volume_dataset.py
│   ├── models/            # Model architectures
│   └── utils/
├── experiments/
│   ├── analysis/          # Phase A — patch-level CNN baseline
│   ├── ABMIL/             # Phase B & C — ABMIL baseline (standard)
│   └── SelfAttention/     # Phase B & C — SA-ABMIL variant (experimental)
├── scripts/               # Analysis notebooks and exploratory scripts
│   ├── dataset_analysis.ipynb
│   ├── patch_analysis.ipynb
│   └── slice_analysis.ipynb
└── IDEA_REPORT.md         # Full research design document
```

---

## Status

The project is in active experimentation. Multiple backbone architectures (ResNet-50, EfficientNet-B0, DenseNet-121), patch sizes (32, 64, 128 px) and aggregation strategies are being compared. The final model architecture will be selected based on the results of these experiments — with emphasis on both in-domain accuracy and out-of-distribution (OOD) generalization across GAN architectures.

---

## Author

**Orazio Pontorno**  
PhD student — University of Catania  
`orazio.pontorno@phd.unict.it`
