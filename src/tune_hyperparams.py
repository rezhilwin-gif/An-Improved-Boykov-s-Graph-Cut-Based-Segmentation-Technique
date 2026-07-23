"""
Sweeps Boykov graph-cut / DenseCRF hyperparameters on a validation subset and
picks the combination with the best mean accuracy.

Requires checkpoints/best_model.pt and checkpoints/train_val_split.json,
both produced by train.py.

Usage:
    python src/tune_hyperparams.py

Writes:
    checkpoints/best_sweep_config.json
    checkpoints/sweep_log.csv
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import BEST_MODEL_PATH, CHECKPOINT_DIR, SWEEP_CONFIG_PATH, SWEEP_LOG_PATH
from model import load_model
from postprocess import run_full_pipeline

# Reduced from the full paper-order 3x3x3x3=81-combination grid so a first
# pass actually completes. Widen SLIC_SEGMENT_CANDIDATES /
# CRF_ITERATION_CANDIDATES back to the full ranges once a fast pass confirms
# the loop runs cleanly end-to-end.
GRAPH_CUT_LAMBDAS = [40.0, 50.0, 60.0]
GRAPH_CUT_SIGMAS = [20.0, 25.0, 30.0]
SLIC_SEGMENT_CANDIDATES = [200]  # widen to [200, 250, 300] after a first pass works
CRF_ITERATION_CANDIDATES = [10]  # widen to [5, 10, 15] after a first pass works

VAL_SUBSET_N = 30  # small subset for the search loop itself; widen once confirmed fast/stable


def evaluate_configuration(model, val_imgs, val_masks, k_value, sigma_value, n_segments, crf_iters, max_n=VAL_SUBSET_N):
    metrics = []
    for img_path, mask_path in list(zip(val_imgs, val_masks))[:max_n]:
        result = run_full_pipeline(
            model,
            img_path,
            mask_path,
            n_segments=n_segments,
            compactness=10,
            k=k_value,
            sigma=sigma_value,
            crf_iters=crf_iters,
        )
        if result["metrics"] is not None:
            metrics.append(result["metrics"])
    if not metrics:
        return None
    return {
        "accuracy": float(np.mean([m["accuracy"] for m in metrics])),
        "precision": float(np.mean([m["precision"] for m in metrics])),
        "recall": float(np.mean([m["recall"] for m in metrics])),
        "specificity": float(np.mean([m["specificity"] for m in metrics])),
        "samples": len(metrics),
    }


def main():
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"{BEST_MODEL_PATH} not found. Run src/train.py first.")

    split_path = CHECKPOINT_DIR / "train_val_split.json"
    if not split_path.exists():
        raise FileNotFoundError(f"{split_path} not found. Run src/train.py first.")

    with open(split_path) as f:
        split = json.load(f)
    val_imgs, val_masks = split["val_imgs"], split["val_masks"]

    model = load_model(checkpoint_path=BEST_MODEL_PATH)

    best_sweep = None
    sweep_rows = []
    for k_value in GRAPH_CUT_LAMBDAS:
        for sigma_value in GRAPH_CUT_SIGMAS:
            for n_segments in SLIC_SEGMENT_CANDIDATES:
                for crf_iters in CRF_ITERATION_CANDIDATES:
                    print(f"Sweep: lambda={k_value}, sigma={sigma_value}, segments={n_segments}, crf={crf_iters}")
                    metrics = evaluate_configuration(model, val_imgs, val_masks, k_value, sigma_value, n_segments, crf_iters)
                    if metrics is None:
                        continue
                    row = {"k": k_value, "sigma": sigma_value, "segments": n_segments, "crf_iters": crf_iters, **metrics}
                    sweep_rows.append(row)
                    if best_sweep is None or row["accuracy"] > best_sweep["accuracy"]:
                        best_sweep = row

    print("Best sweep config:", best_sweep)

    if best_sweep is None:
        best_sweep = {"k": 50.0, "sigma": 20.0, "segments": 200, "crf_iters": 10}

    with open(SWEEP_CONFIG_PATH, "w") as f:
        json.dump(best_sweep, f, indent=2)
    print(f"Saved best sweep config -> {SWEEP_CONFIG_PATH}")

    if sweep_rows:
        with open(SWEEP_LOG_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sweep_rows)
        print(f"Saved full sweep log -> {SWEEP_LOG_PATH}")


if __name__ == "__main__":
    main()
