# MedForensics — Claude Internal Context File

> Only I (Claude) read this. Written for fast, accurate context retrieval.
> Last updated: 2026-03-13 (session 2)

---

## PROJECT SNAPSHOT

**Goal**: Detect and localize synthetic manipulations (nodule injection/removal) in 3D CT scans using hierarchical attention-based MIL — weakly supervised (volume labels only), natively interpretable (3D heatmaps from attention weights). Medical CT forensics / deepfake detection.

**Owner**: Orazio Pontorno, PhD student, University of Catania. Email: orazio.pontorno@phd.unict.it

**Codebase root**: `/home/opontorno/projects/MedForensics`
**Python package**: `src/medforensics` — installed editable via `pip install -e .`
**Conda env**: `medfor`
**Data root**: `/mnt/lguarnera_group/opontorno/med_datasets/M3DSynth`
**GPU scheduling**: `tsp` (task-spooler), multi-GPU auto-select via GPUtil (max memoryFree)

---

## DATASET: M3DSynth

```
Modalities: real(0), pix2pix(1), cycle(2), diffusion(3)
Format: TIFF multi-page (slide0000.tiff, ...), uint16
Normalization: percentile [1,99] on non-zero pixels → [0,1]
Metadata:
  data.csv  → img_id, mod, ty, coord_z, coord_y, coord_x
  sets.csv  → patient-level train/val/test split
ty:    'injection' (nodule added) | 'removal' (nodule removed)
img_id prefix: 'inj_' | 'rem_'
GT mask: available for fakes only, pixel-level
```

**Label convention**: `VolumeDataset`/`SliceDataset` return multiclass (0-3). BCE training always applies `(label > 0).float()` to get binary real/fake. The multiclass label is preserved for `_balanced_mod_metrics()`.

---

## ARCHITECTURE — 3-PHASE PIPELINE

### Phase A — Patch POC (complete, not actively developed)
- `NodulePatchDataset`: 128×128 crop centered at `(coord_y, coord_x)` on `coord_z` slice
- `CNNPatchClassifier`: timm backbone + SpatialAttention + GAP + head
- **Result**: ResNet-50 p=128 → AUC 0.973, Acc 92.2%
- Scripts: `experiments/patches_poc/`

### Phase B — Slice ABMIL (MAIN METHOD)
- **`SliceDataset`**: sliding-window tiling → bag of N patches
  - `build_patch_grid(arr, patch_size, stride)` → `(N, P, P)`, positions, `(n_rows, n_cols)`
  - `reconstruct_heatmap(weights, grid_hw, slice_hw, patch_size, stride)` → `(H,W)` heatmap
- **`ABMILSliceClassifier`**: encoder → projector → GatedAttention → head
  - `encode_patches(patches)`: `backbone[-1]` → `spatial_attn` → weighted GAP → `(B,N,feat_dim)`
  - `GatedAttention`: Ilse 2018 — `a = softmax(w^T(tanh(Vh) ⊙ σ(Uh)))`
- **`SABMILSliceClassifier`** (SA variant): inserts `PatchTransformer` between projector and ABMIL
  - Pre-LN, batch_first=True, sa_n_heads=8, sa_n_layers=2
- **Defaults**: resnet50, proj_dim=512, attn_dim=128, dropout=0.25, patch_size=128, stride=64
- **Augment**: random H/V flip on whole slice before tiling
- **Champion metric**: `ood_test_accuracy` (or `accuracy` if trained_on_all)
- Scripts: `experiments/ABMIL/`, `experiments/SelfAttention/`

### Phase C — Volume Classifier
- **`VolumeDataset`**: K-slice window centered at `coord_z` + random Z-jitter
  - Fake: `coord_z` placed at random position in `[0, K-1]`; Real: random window
  - Zero-padding + `valid_mask` for out-of-volume slices
- **`VolumeClassifier`**: Phase B encoder FROZEN → SinPosEncoding → GatedAttention(Z) → head
  - 3D heatmap: `β_k × α_k[y,x]` (β = slice-level Z-attention, α = patch-level spatial attention)
- **`SAVolumeClassifier`**: adds `SliceTransformer` after PE, before Z-aggregator
  - `src_key_padding_mask` for padded slices (PyTorch: True = padded = ignored)
- **Factory**: `build_volume_classifier` / `build_sa_volume_classifier` — reads Phase B `args.json`
  - Auto-detects `use_sa` from `args.json`
- **Champion metric**: `test_accuracy` (global, real+ALL_FAKES)
- Scripts: `experiments/ABMIL/train_volume-cls.py`, `eval_volume-cls.py`

---

## FILE MAP

```
src/medforensics/
├── data/
│   ├── patch_dataset.py               # NodulePatchDataset, MOD_LABEL, load_split_table
│   ├── slice_dataset.py               # SliceDataset, build_patch_grid, reconstruct_heatmap
│   └── volume_dataset.py              # VolumeDataset  [CHANGED: multiclass labels 0-3]
├── models/
│   ├── cnn_patch_classifier.py        # CNNPatchClassifier, SpatialAttention, build_cnn_classifier
│   ├── abmil_slice_classifier.py      # ABMILSliceClassifier, GatedAttention, build_abmil_classifier_scratch
│   ├── abmil_slice_classifier_sa.py   # SABMILSliceClassifier, PatchTransformer, build_sa_classifier_scratch
│   ├── abmil_slice_classifier_fourier.py  # FA-ABMIL: dual-branch spatial+freq  [EXISTS, standalone variant]
│   ├── volume_classifier.py           # VolumeClassifier, SinPosEncoding, build_volume_classifier
│   └── volume_classifier_sa.py        # SAVolumeClassifier, SliceTransformer, build_sa_volume_classifier
└── utils/tiff_utils.py                # get_shape, load_slice, get_percentile, apply_percentile, save

experiments/
├── ABMIL/
│   ├── train_slice-cls.py    [stable]
│   ├── eval_slice-cls.py     [stable]
│   ├── train_volume-cls.py   [REWRITTEN — session 2026-03-13]
│   ├── eval_volume-cls.py    [REWRITTEN — session 2026-03-13]
│   ├── compare_results.py
│   └── inference.py
├── SelfAttention/
│   ├── train_slice-cls.py
│   ├── eval_slice-cls.py
│   ├── train_volume-cls.py   [REWRITTEN — session 2026-03-13 s2]
│   ├── eval_volume-cls.py    [REWRITTEN — session 2026-03-13 s2]
│   └── compare_results.py
├── baselines/
│   ├── train_3d_resnet.py
│   ├── train_vit_abmil.py
│   ├── eval_robustness.py
│   └── models/  (resnet3d_classifier, swin3d_classifier, vit_volume_classifier)
└── patches_poc/
```

---

## RUN DIR CONVENTION

```
# Phase B
slice-cls_{backbone}_p{patch_size}_s{stride}[_attn][_fourier{mode}]/
  trained_on_{mods}_{loss}_bs{bs}_lr{lr}/
    best_model.pt       # state_dict + epoch + metrics
    args.json           # all hyperparams (use_sa included if SA variant)
    test_metrics.json
    vis/epoch_{N}/

# Phase C
volume-cls_{backbone}_p{patch_size}_s{stride}_K{K}/
  trained_on_{mods}/
    best_model.pt / args.json / test_metrics.json
```

Optional dir name suffixes (append only when non-default):
- `_attn` → aux_attn_loss active
- `_fourier{mode}` → fourier_mode != 'none' (planned, not yet implemented)

---

## KEY FUNCTIONS TO KNOW

### In `train_volume-cls.py`

```python
build_dataloaders()
# → dl_train (real+mods), dl_in_valid (real+mods), dl_valid (real+ALL_FAKES), dl_test (real+ALL_FAKES)
# WeightedRandomSampler for class balance in dl_train

_balanced_mod_metrics(labels_np, scores_np, preds_np, mods_arr, mod, seed=42)
# Equal real vs each fake arch sampling. Returns {acc, auc, f1} for a single mod.
# Called per-arch: for mod in ALL_FAKES → cls_metrics[f'{mod}_{k}']

run_test_evaluation_volume(model, dl_test, device, vis_dir, args, tab)
# Full test eval: cls_metrics + XAI metrics. IMPORTABLE by eval_volume-cls.py via importlib.

promote_if_better(model, metrics, best_metrics, ckpt_path)
# Champion based on metrics['test_accuracy'] (global, not OOD)

save_epoch_vis(model, dl_real, dl_fakes, epoch, vis_dir, device, n_samples)
# NOTE: signature has NO ood_dataloader param (removed)
```

### In `eval_volume-cls.py` (local only, not in train)

```python
load_full_volume_windows(row, data_dir, K, patch_size, stride)
_run_window(model, window_tensor, valid_mask, device)
run_full_volume_eval(model, tab, data_dir, device, args)
compute_bbox_iou_3d(...)
save_volume_3d_projection(...)
save_as_nifti(...)
```

### Importlib pattern (used in all eval scripts)
```python
import importlib.util
_TRAIN_SCRIPT = Path(__file__).parent / 'train_volume-cls.py'
_spec = importlib.util.spec_from_file_location('_train_vol', _TRAIN_SCRIPT)
_tm   = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tm)
# Then: _tm.run_test_evaluation_volume, _tm._balanced_mod_metrics, etc.
# Required because filenames contain hyphens (not importable directly)
```

---

## CHANGES LOG

### 2026-03-13 (session 2) — SelfAttention symmetric rewrite + cleanup

**`SelfAttention/train_volume-cls.py`** — rewritten to mirror `ABMIL/train_volume-cls.py`:
- Added `ALL_FAKES = ['pix2pix', 'cycle', 'diffusion']` (was missing)
- `build_dataloaders()` → 4-loader pattern, removed OOD loaders
- `evaluate()` uses `_balanced_mod_metrics()`; removed old `_mod_metrics()` helper
- `train_one_epoch()`: multiclass labels + `labels_bin = (label > 0).float()`
- `save_epoch_vis()`: removed `ood_dataloader` param
- `promote_if_better()`: champion = `test_accuracy`
- `main()`: removed OOD block; SA-specific: `build_sa_volume_classifier(..., sa_n_heads=..., sa_n_layers=...)`
- `--test_mods` arg removed; `--sa_n_heads`, `--sa_n_layers` kept

**`SelfAttention/eval_volume-cls.py`** — full rewrite as importlib-based:
- Mirrors ABMIL eval exactly; SA-specific: `SAVolumeClassifier`/`build_sa_volume_classifier` with `sa_n_heads`/`sa_n_layers` from `saved_args`

**Cleanup**:
- Deleted all 44 `test_metrics.json` and `metrics.json` files from `runs/` subdirs



### 2026-03-13 — Volume pipeline rewrite

**`volume_dataset.py`**
- Label now multiclass: `label = MOD_LABEL.get(mod, 0)` → 0=real,1=pix2pix,2=cycle,3=diffusion
- BCE callers apply `(label > 0).float()` for binary training

**`train_volume-cls.py`** — complete rewrite/alignment:
- Added `build_dataloaders()` with 4 loaders (train/in_valid/valid/test)
- Added `_balanced_mod_metrics()` for per-arch eval with balanced sampling
- `evaluate()` returns flat dict with per-arch metrics
- `train_one_epoch()`: AMP + WeightedRandomSampler
- Added `run_test_evaluation_volume()` as importable helper
- `promote_if_better()` uses `test_accuracy` (was: OOD-based)
- WandB logging: `train/`, `val/`, `test/xai/` namespaces
- Removed `ood_dataloader` param from `save_epoch_vis()` entirely
- Removed OOD test block from `main()`; replaced with:
  ```python
  wandb.log({f'test/{k}': v for k, v in cls_m.items()})
  wandb.log({f'test/xai/{k}': v for k, v in results.get('xai', {}).items()})
  ```

**`eval_volume-cls.py`** — full rewrite as importlib-based script:
- Imports all helpers from `train_volume-cls.py` via importlib
- `get_args()`: removed `--split`, `--test_mods`; kept `--full_volume`, `--beta_thresh`, `--save_3d`, `--save_nifti`
- Normal mode: delegates entirely to `_tm.run_test_evaluation_volume()`
- `--full_volume` mode: sliding-window inference (non-overlapping K-slice windows, max-score aggregation), defined locally
- Always loads: `tab = load_split_table(DATA_DIR, 'test', ['real'] + ALL_FAKES)`

---

## PENDING: FOURIER MODE INTEGRATION

**Status**: Plan exists (`/home/opontorno/.claude/plans/parsed-weaving-llama.md`), NOT yet implemented.

**Two competing approaches — clarify with Orazio before implementing**:
1. **`fourier_mode` param** (the plan): modifies input to existing backbone (`'concat'` = 2-channel input, `'freq_only'` = log|FFT| replaces spatial). Stateless preprocessing. Backward-compatible.
2. **`abmil_slice_classifier_fourier.py`** (already exists): full dual-branch FA-ABMIL — spatial CNN + FourierMagnitudeNet (3-layer CNN on rfft2 magnitude) in parallel, features concatenated before projector. Has learnable frequency encoder.

The plan's `FourierMagnitudeModule` is stateless (no params). The existing `FourierMagnitudeNet` is learnable (3 conv blocks). These are architecturally distinct.

**Critical fix in the plan** (apply if implementing `fourier_mode`):
`encode_patches()` in both `abmil_slice_classifier.py` and `abmil_slice_classifier_sa.py` calls `self.encoder.backbone(flat)[-1]` **directly**, bypassing `forward()`. Must add:
```python
flat = self.encoder._apply_fourier(flat)  # before backbone call
```

**Backward compat**: `sargs.get("fourier_mode", "none")` in `volume_classifier.py` handles old `args.json` without the key.

---

## XAI METRICS

**2D** (on `coord_z` slice, computed by `run_test_evaluation_volume`):
- `pixel_auc` — ROC-AUC of heatmap vs GT mask, pixel-level
- `iou_03/05/07` — IoU at threshold 0.3/0.5/0.7
- `pointing_game` — argmax of heatmap hits GT mask?
- `energy_in_mask` — fraction of attention energy inside mask
- `pd_at_1` — precision-at-1 (top prediction hit rate)

**3D** (full-volume mode only, computed locally in `eval_volume-cls.py`):
- `bbox_iou_3d` — IoU of predicted 3D bounding box vs GT
- `pd_at_1_3d` — 3D pointing game

---

## RESULTS SUMMARY

| Phase | Backbone | p | AUC | Acc | Notes |
|---|---|---|---|---|---|
| B (ABMIL) | ResNet-50 | 128 | 0.982 | 95.6% | trained_on_all |
| B (ABMIL) | ResNet-50 | 64 | 0.972 | 93.9% | trained_on_all |
| B (ABMIL) | EfficientNet-B0 | 128 | 0.977 | 94.4% | trained_on_all |
| B (ABMIL) | DenseNet-121 | 128 | 0.959 | 91.0% | trained_on_all |
| C (Volume) | ResNet-50 | 64, K=16 | 0.965 | 89.7% | pix2pix, OOD +10pp vs Phase B |
| B (SA-ABMIL) | ResNet-50 | 128 | TBD | TBD | first run in progress |

Cross-domain gap (Phase B, ResNet-50 p=128): OOD acc ~79-82% (~10-15pp below in-domain)

---

## OPEN TASKS

- [ ] SA-ABMIL Phase B results (resnet50, p=128, pix2pix first run)
- [ ] Decide: `fourier_mode` vs `abmil_slice_classifier_fourier.py` — ask Orazio
- [ ] Implement sliding-window inference without `coord_z` (unsupervised localization)
- [ ] Systematic XAI metrics sweep across all trained models
- [ ] Ablation: K (num slices) in Phase C
- [ ] Final comparison: ABMIL vs SA-ABMIL vs 3D baselines
- [ ] Coronal/sagittal planes (future, low priority)

---

## INFRASTRUCTURE NOTES

- AMP: `torch.amp.autocast` + `GradScaler` — both enabled everywhere
- WandB: `--wandb_mode` arg controls logging; GIF Z-scroll + PNG grids logged per epoch
- WeightedRandomSampler: class balance during training (not validation/test)
- Early stopping: on `val_loss`
- Champion: saved to `best_model.pt` when champion metric improves
- `compare_results.py`: reads `test_metrics.json` from all run subdirs, builds paired Phase B↔C table
  ⚠️ All `test_metrics.json`/`metrics.json` files were deleted on 2026-03-13; re-run eval scripts to regenerate
