"""Central path configuration for HexMIL.

This is the ONLY file a user needs to edit to run the method on their machine.
It defines the filesystem locations used across training, evaluation and
inference. All training/evaluation hyper-parameters are exposed as command-line
arguments in the individual scripts (`train_slicemil.py`, `train_hexmil.py`,
`eval_*.py`, `inference.py`) and are not configured here.

Only `DATA_DIR` normally requires editing; the remaining paths are derived from
the repository location and can be left untouched. Every value may also be
overridden through an environment variable (shown in brackets) without editing
this file.
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# Dataset location (EDIT THIS)                               [HEXMIL_DATA_DIR]
# ─────────────────────────────────────────────────────────────────────────────
# Root of the M3DSynth benchmark. It must contain `data.csv` and `sets.csv`.
# The optional CT-GAN split (`ctgan_data.csv` / `ctgan_sets.csv`) is read from
# this same directory when the `ctgan` modality is requested.
DATA_DIR = os.environ.get(
    "HEXMIL_DATA_DIR",
    "/mnt/lguarnera_group/opontorno/med_datasets/M3DSynth",
)

# ─────────────────────────────────────────────────────────────────────────────
# Repository and output locations (usually left as-is)
# ─────────────────────────────────────────────────────────────────────────────
# Repository root — auto-derived from this file's location.        [HEXMIL_WORK_DIR]
WORK_DIR = os.environ.get(
    "HEXMIL_WORK_DIR",
    os.path.dirname(os.path.abspath(__file__)),
)

# Directory where training runs (checkpoints, metrics, visualizations) live. [HEXMIL_RUNS_DIR]
RUNS_DIR = os.environ.get("HEXMIL_RUNS_DIR", os.path.join(WORK_DIR, "runs"))
