# analysis — Phase A (patch-level classifier)

Patch-level binary classifier used for analysis and ablation.  
Not the primary detection model — this is used to study patch-level discriminability before ABMIL pooling.

## Architecture

Standard CNN binary classifier trained on individual patches (real / fake).

## Scripts

| Script | Purpose |
|--------|---------|
| `train_patch-cls.py` | Train a patch-level CNN classifier |
| `eval_patch-cls.py` | Evaluate patch classifier |

## Run naming

```
runs/
  patch-cls_{backbone}_p{patch_size}_s{stride}/
    trained_on_{mods}_{loss}_bs{bs}_lr{lr}/
```
