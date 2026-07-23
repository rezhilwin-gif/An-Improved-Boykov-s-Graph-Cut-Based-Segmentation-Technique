"""
Central configuration for the IBGC-CRF-SPSST pipeline.

Every other module imports paths, the random seed, the device, and the
class list from here so that all scripts (inspect_data.py, train.py,
tune_hyperparams.py, final_eval.py, visualize_preds.py, run_pipeline.py)
stay in agreement about where things live on disk.
"""

from pathlib import Path
import random

import numpy as np
import torch

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
# Herlev Pap-smear class folders. Drop the dataset's class folders directly
# inside Data/ (e.g. Data/normal_superficiel/, Data/light_dysplastic/, ...).
EXPECTED_CLASSES = [
    "normal_superficiel",
    "normal_intermediate",
    "normal_columnar",
    "light_dysplastic",
    "moderate_dysplastic",
    "severe_dysplastic",
    "carcinoma_in_situ",
]

IMG_SIZE = 224

# --------------------------------------------------------------------------
# Ground-truth color scheme (verified against martin2003.pdf)
# --------------------------------------------------------------------------
GT_COLORS_BGR = {
    "background": np.array([0, 0, 255]),
    "cytoplasm": np.array([128, 0, 0]),
    "nucleus": np.array([255, 0, 0]),
}
GT_LABELS = list(GT_COLORS_BGR.keys())
GT_COLOR_ARRAY = np.stack([GT_COLORS_BGR[k] for k in GT_LABELS])

# --------------------------------------------------------------------------
# Project paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "Data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
GRAPHS_DIR = PROJECT_ROOT / "graphs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

for _dir in (CHECKPOINT_DIR, GRAPHS_DIR, OUTPUTS_DIR, DATA_ROOT):
    _dir.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pt"
TRAIN_CONFIG_PATH = CHECKPOINT_DIR / "training_config.json"
SWEEP_CONFIG_PATH = CHECKPOINT_DIR / "best_sweep_config.json"
SWEEP_LOG_PATH = CHECKPOINT_DIR / "sweep_log.csv"
FINAL_METRICS_PATH = OUTPUTS_DIR / "final_metrics.json"


def require_data_root():
    """Raise a clear, actionable error if Data/ doesn't contain the expected
    class folders yet (i.e. the user hasn't pasted the dataset in)."""
    missing = [c for c in EXPECTED_CLASSES if not (DATA_ROOT / c).exists()]
    if missing:
        raise FileNotFoundError(
            "Dataset not found under '{}'.\n"
            "Copy the Herlev Pap-smear class folders into the Data/ directory, e.g.\n"
            "  Data/normal_superficiel/...\n"
            "  Data/light_dysplastic/...\n"
            "Missing class folders: {}".format(DATA_ROOT, missing)
        )
    return DATA_ROOT
