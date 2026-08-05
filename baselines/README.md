# Baselines

Baseline methods evaluated against HexMIL on the M3DSynth dataset.

## Directory structure

```
baselines/
├── git_repo/                     # Official repos (fetch with clone_repos.sh)
│   ├── D3/                              # D3 (Corvi et al., ICASSP 2023)
│   ├── Deep_inpainting_localization/   # HP-FCN (Li & Huang, ICCV 2019)
│   ├── ManTraNet/                      # ManTraNet (Wu et al., CVPR 2019)
│   ├── MVSS-Net/                       # MVSS-Net (Chen et al., CVPR 2022)
│   └── TruFor/                         # TruFor (Guillaro et al., CVPR 2023)
├── models/                       # Custom model definitions (ours)
│   ├── resnet3d_classifier.py
│   ├── densenet3d_classifier.py
│   ├── vit3d_classifier.py
│   └── mvit_classifier.py
├── runs/                         # Training outputs (checkpoints, logs)
├── results/                      # Aggregated evaluation results
├── train_trufor.py               # TruFor — full model from repo (PyTorch)
├── train_mvssnet.py              # MVSS-Net — model from repo (PyTorch)
├── train_d3.py                   # D3 — model from repo (PyTorch)
├── train_hp_fcn.py               # HP-FCN — model from repo (TensorFlow)
├── train_mantranet.py            # ManTraNet — model from repo (Keras/TF)
├── train_3d_resnet.py            # R3D-18 3D CNN (PyTorch)
├── train_3d_densenet.py          # DenseNet-121 3D (PyTorch)
├── train_3d_vit.py               # ViT-3D (plain + factorised ViViT) (PyTorch)
├── train_3d_mvit.py              # MViT-V2-S (PyTorch)
├── train_flat_cnn.py             # Flat ResNet-50 2D (PyTorch)
├── clone_repos.sh                # Fetch the official repos at pinned commits
├── compare_results.py            # Aggregate metrics across all baselines
└── README.md
```

## Reproducing results

### 1. Clone the official repositories

The repo-based baselines are imported verbatim from their authors' code. Fetch
every repository at the exact pinned commit with the helper script:

```bash
bash baselines/clone_repos.sh
```

This clones the following upstream repositories into `baselines/git_repo/`:

| Repo | URL | Commit |
|------|-----|--------|
| `D3` | github.com/BigAandSmallq/D3 | `14f21ad` |
| `MVSS-Net` | github.com/dong03/MVSS-Net | `cc2aed7` |
| `TruFor` | github.com/grip-unina/TruFor | `ae54475` |
| `ManTraNet` | github.com/ISICV/ManTraNet | `59436db` |
| `Deep_inpainting_localization` | github.com/lihaod/Deep_inpainting_localization | `d33468d` |

### 2. Install dependencies

**PyTorch baselines** (TruFor, MVSS-Net, D3, 3D CNNs/Transformers, Flat CNN):
```bash
conda run -n medsota pip install -r baselines/requirements_pytorch.txt
```

**TensorFlow baselines** (HP-FCN, ManTraNet):
```bash
/path/to/miniconda3/envs/medsota-tf-cu122/bin/pip install -r baselines/requirements_tensorflow.txt
```

### 3. Run training

Each script follows the same interface (see `run_baselines.txt` for the full
queue of commands):

```bash
# Train on a single modality (the others become the OOD test set)
python baselines/train_trufor.py --train_mods pix2pix --epochs 50

# Train on all modalities
python baselines/train_trufor.py --epochs 50

# Evaluate only (requires prior training)
python baselines/train_trufor.py --eval_only --out_dir baselines/runs/trufor_full_K16/trained_on_...
```

Common arguments:
| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | config.DATA_DIR | Path to M3DSynth dataset |
| `--train_mods` | all | Fake modalities to train on |
| `--K` | 16 | Number of slices per volume |
| `--epochs` | 50 | Max training epochs |
| `--batch_size` | 4-8 | Batch size |
| `--lr` | 1e-4 to 5e-5 | Learning rate |
| `--patience` | 12 | Early stopping patience |
| `--gpu_id` | auto | GPU to use |
| `--eval_only` | false | Skip training, evaluate only |

### 4. Aggregate results

```bash
cd baselines
python compare_results.py --runs runs/ --out_dir results/
```

## Baseline categories

### From official repositories (imported models)

| Script | Paper | Framework | Model source |
|--------|-------|-----------|-------------|
| `train_trufor.py` | Guillaro et al., CVPR 2023 | PyTorch | Full EncoderDecoder (NP++ + CMX + MLP decoder) |
| `train_mvssnet.py` | Chen et al., CVPR 2022 | PyTorch | MVSSNet via importlib |
| `train_d3.py` | Corvi et al., ICASSP 2023 | PyTorch | ResNet-50 + BlurPool (LPF) anti-aliasing |
| `train_hp_fcn.py` | Li & Huang, ICCV 2019 | TensorFlow | Architecture from hp_fcn.py + tf_slim |
| `train_mantranet.py` | Wu et al., CVPR 2019 | Keras/TF | create_model() + pretrained .h5 weights |

### Custom implementations (ours)

| Script | Description |
|--------|-------------|
| `train_3d_resnet.py` | R3D-18 3D CNN on full CT volumes |
| `train_3d_densenet.py` | DenseNet-121 adapted for 3D |
| `train_3d_vit.py` | ViT-3D (plain joint attention and factorised ViViT variants) |
| `train_3d_mvit.py` | MViT-V2-S multiscale 3D transformer |
| `train_flat_cnn.py` | ResNet-50 on 2D slices (no MIL) |

## CT adaptation notes

All baselines are adapted for 1-channel CT input:
- **1ch → 3ch**: grayscale CT slices are repeated to 3 channels for models expecting RGB
- **Percentile normalization**: CT HU values are mapped to [0, 1] via dataset-specific percentiles
- **Volume score**: `max(pixel_score)` over K slices per volume
- **No pixel-level supervision**: only volume-level binary labels; pixel maps are used for evaluation only
```
