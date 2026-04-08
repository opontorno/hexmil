# SelfAttention — Phase B & C with Transformer contextualization

SA-ABMIL variant: a lightweight Transformer encoder is inserted between the patch/slice embedder and the Gated ABMIL aggregator, allowing each token to attend to all others before pooling.

**Motivation:** Standard ABMIL scores patches/slices independently → learns distribution-specific artefacts.  Self-attention lets the model detect *spatial inconsistency* between patches (or between slices) — a more domain-agnostic forgery cue that should improve OOD generalization.

## Architecture

### Phase B — SA slice classifier (`abmil_slice_classifier_sa.py`)

```
CNN encoder → projector → PatchTransformer (Pre-LN, sa_n_layers × sa_n_heads)
           → Gated ABMIL → head → logit
```

The `PatchTransformer` is a `nn.TransformerEncoder` (`norm_first=True`, `batch_first=True`).  
Each patch attends to **all other patches** in the same slice before ABMIL pools them.  
ABMIL α weights remain valid for patch-level heatmaps (now over *contextualised* representations).

### Phase C — SA volume classifier (`volume_classifier_sa.py`)

```
Frozen Phase B encoder (ABMIL or SA-ABMIL) → sinusoidal z-pos-enc
→ SliceTransformer (Pre-LN, sa_n_layers × sa_n_heads)
→ Gated Volume Aggregator → head → logit
```

Each slice attends to **all other slices** in the window. The β weights still drive the 3-D heatmap.  
Supports both ABMIL and SA-ABMIL Phase B checkpoints — auto-detected via `use_sa` in `args.json`.

## Key hyper-parameters

| Arg | Default | Notes |
|-----|---------|-------|
| `--sa_n_heads` | 8 | Transformer attention heads; `proj_dim % sa_n_heads == 0` |
| `--sa_n_layers` | 2 | Transformer encoder depth |

## Scripts

| Script | Purpose |
|--------|---------|
| `train_slice-cls.py` | Train SA Phase B |
| `eval_slice-cls.py` | Evaluate SA Phase B (classification + XAI; auto-detects `use_sa`) |
| `train_volume-cls.py` | Train SA Phase C |
| `eval_volume-cls.py` | Evaluate SA Phase C (auto-detects `use_sa`) |

## Run naming

```
runs/
  sa-slice-cls_{backbone}_p{patch_size}_s{stride}/
    trained_on_{mods}_{loss}_bs{bs}_lr{lr}/
      best_model.pt
      args.json         ← contains use_sa=True, sa_n_heads, sa_n_layers
      metrics/
      vis/
  sa-volume-cls_{backbone}_p{patch_size}_s{stride}_K{K}/
    trained_on_{mods}_bce_bs{bs}_lr{lr}/
```

## Quick start

```bash
# Phase B with SA
python experiments/SelfAttention/train_slice-cls.py \
    --mods real pix2pix cycle diffusion \
    --backbone resnet50 --patch_size 128 \
    --sa_n_heads 8 --sa_n_layers 2 --epochs 40

# Phase C with SA (accepts both ABMIL and SA Phase B checkpoints)
python experiments/SelfAttention/train_volume-cls.py \
    --slice_ckpt_dir experiments/SelfAttention/runs/sa-slice-cls_resnet50_p128_s64/trained_on_all_bce_bs8_lr0.0001 \
    --K 16 --sa_n_heads 8 --sa_n_layers 2 --epochs 60
```

## Differences from ABMIL baseline

| | ABMIL | SA-ABMIL |
|--|-------|----------|
| Patch interaction | None (independent) | Full self-attention |
| Slice interaction | None (independent) | Full self-attention |
| OOD signal | Artefact-specific | Inconsistency-based |
| #Params (extra) | 0 | ~2-3 M (Transformer layers) |
| `args.json` flag | — | `use_sa: true` |
