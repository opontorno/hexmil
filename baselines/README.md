# Baselines

Baseline methods evaluated against HexMIL on the M3DSynth dataset.

## Directory structure

```
baselines/
├── git_repo/                     # Official repos (clone these)
│   ├── Deep_inpainting_localization/   # HP-FCN (Li & Huang, ICCV 2019)
│   ├── ManTraNet/                      # ManTraNet (Wu et al., CVPR 2019)
│   ├── MVSS-Net/                       # MVSS-Net (Chen et al., CVPR 2022)
│   └── TruFor/                         # TruFor (Guillaro et al., CVPR 2023)
├── models/                       # Custom model definitions (ours)
│   ├── resnet3d_classifier.py
│   ├── densenet3d_classifier.py
│   ├── efficientnet3d_classifier.py
│   ├── swin3d_classifier.py
│   └── vit_volume_classifier.py
├── runs/                         # Training outputs (checkpoints, logs)
├── results/                      # Aggregated evaluation results
├── train_trufor.py               # TruFor — full model from repo (PyTorch)
├── train_mvssnet.py              # MVSS-Net — model from repo (PyTorch)
├── train_hp_fcn.py               # HP-FCN — model from repo (TensorFlow)
├── train_mantranet.py            # ManTraNet — model from repo (Keras/TF)
├── train_3d_resnet.py            # R3D-18 3D CNN (PyTorch)
├── train_3d_densenet.py          # DenseNet-121 3D (PyTorch)
├── train_3d_efficientnet.py      # EfficientNet-B0 3D (PyTorch)
├── train_flat_cnn.py             # Flat ResNet-50 2D (PyTorch)
├── train_pool_mil.py             # Pool-MIL ablation (PyTorch)
├── train_vit_abmil.py            # ViT-ABMIL (PyTorch)
├── compare_results.py            # Aggregate metrics across all baselines
└── README.md
```

## Reproducing results

### 1. Clone the official repositories

```bash
cd baselines/git_repo

git clone https://github.com/lihaod/Deep_inpainting_localization.git
# commit: d33468d

git clone https://github.com/ISICV/ManTraNet.git
# commit: 59436db

git clone https://github.com/dong03/MVSS-Net.git
# commit: cc2aed7

git clone https://github.com/grip-unina/TruFor.git
# commit: ae54475
```

### 2. Install dependencies

**PyTorch baselines** (TruFor, MVSS-Net, 3D CNNs, Pool-MIL, ViT-ABMIL, Flat CNN):
```bash
# Same environment as HexMIL
conda activate medfor
pip install torch torchvision timm scikit-learn tqdm wandb
# TruFor also requires:
pip install yacs
```

**TensorFlow baselines** (HP-FCN, ManTraNet):
```bash
pip install tensorflow tf-slim h5py scikit-image scikit-learn
```

### 3. Run training

Each script follows the same interface:

```bash
# Train on a single modality
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
| `train_hp_fcn.py` | Li & Huang, ICCV 2019 | TensorFlow | Architecture from hp_fcn.py + tf_slim |
| `train_mantranet.py` | Wu et al., CVPR 2019 | Keras/TF | create_model() + pretrained .h5 weights |

### Custom implementations (ours)

| Script | Description |
|--------|-------------|
| `train_3d_resnet.py` | R3D-18 3D CNN on full CT volumes |
| `train_3d_densenet.py` | DenseNet-121 adapted for 3D |
| `train_3d_efficientnet.py` | EfficientNet-B0 adapted for 3D |
| `train_flat_cnn.py` | ResNet-50 on 2D slices (no MIL) |
| `train_pool_mil.py` | Same pipeline as HexMIL but with mean/max pooling instead of attention |
| `train_vit_abmil.py` | ViT encoder + gated attention (end-to-end) |

## CT adaptation notes

All baselines are adapted for 1-channel CT input:
- **1ch → 3ch**: grayscale CT slices are repeated to 3 channels for models expecting RGB
- **Percentile normalization**: CT HU values are mapped to [0, 1] via dataset-specific percentiles
- **Volume score**: `max(pixel_score)` over K slices per volume
- **No pixel-level supervision**: only volume-level binary labels; pixel maps are used for evaluation only
