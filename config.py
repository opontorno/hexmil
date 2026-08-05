import os

# Dataset root (must contain data.csv and sets.csv). Set it here or export
# HEXMIL_DATA_DIR; the environment variable takes precedence.
_DATA_DIR_PLACEHOLDER = ""
DATA_DIR = os.environ.get("HEXMIL_DATA_DIR", _DATA_DIR_PLACEHOLDER)

WORK_DIR = os.environ.get("HEXMIL_WORK_DIR", os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.environ.get("HEXMIL_RUNS_DIR", os.path.join(WORK_DIR, "runs"))


def require_data_dir() -> str:
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(
            f"DATA_DIR is not configured (got {DATA_DIR!r}). Set it in config.py "
            "or export HEXMIL_DATA_DIR=/path/to/your/data (must contain data.csv, sets.csv)."
        )
    return DATA_DIR
