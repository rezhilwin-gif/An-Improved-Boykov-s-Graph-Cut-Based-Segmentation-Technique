"""
Full held-out evaluation: runs the trained model + best graph-cut/CRF config
over the ENTIRE validation split saved by train.py, and reports aggregate
accuracy / precision / recall / specificity.

Usage:
    python src/final_eval.py

Reads:
    checkpoints/best_model.pt
    checkpoints/best_sweep_config.json  (falls back to defaults if missing)
    checkpoints/train_val_split.json

Writes:
    outputs/final_metrics.json
    outputs/final_metrics.csv
    outputs/final_metrics_bar.png
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    BEST_MODEL_PATH,
    CHECKPOINT_DIR,
    FINAL_METRICS_PATH,
    OUTPUTS_DIR,
    SWEEP_CONFIG_PATH,
)
from model import load_model
from postprocess import run_full_pipeline

DEFAULT_SWEEP = {"k": 50.0, "sigma": 20.0, "segments": 200, "crf_iters": 10}


def load_sweep_config():
    if SWEEP_CONFIG_PATH.exists():
        with open(SWEEP_CONFIG_PATH) as f:
            return json.load(f)
    print(f"[WARN] {SWEEP_CONFIG_PATH} not found, using default graph-cut/CRF params.")
    return DEFAULT_SWEEP


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
    cfg = load_sweep_config()

    per_image_rows = []
    for idx, (img_path, mask_path) in enumerate(zip(val_imgs, val_masks)):
        result = run_full_pipeline(
            model,
            img_path,
            mask_path,
            n_segments=cfg["segments"],
            compactness=10,
            k=cfg["k"],
            sigma=cfg["sigma"],
            crf_iters=cfg["crf_iters"],
        )
        if result["metrics"] is None:
            continue
        row = {"image_path": img_path, "mask_path": mask_path, "time_sec": result["time"], **result["metrics"]}
        per_image_rows.append(row)
        print(f"[{idx + 1}/{len(val_imgs)}] {Path(img_path).name} -> " + ", ".join(f"{k}={v:.4f}" for k, v in result["metrics"].items()))

    if not per_image_rows:
        raise RuntimeError("No metrics were produced on the held-out validation set.")

    agg = {
        "accuracy": float(np.mean([r["accuracy"] for r in per_image_rows])),
        "precision": float(np.mean([r["precision"] for r in per_image_rows])),
        "recall": float(np.mean([r["recall"] for r in per_image_rows])),
        "specificity": float(np.mean([r["specificity"] for r in per_image_rows])),
        "n_samples": len(per_image_rows),
        "graph_cut_crf_config": cfg,
    }
    print("\n=== Final aggregate metrics (held-out validation set) ===")
    print(json.dumps(agg, indent=2))

    with open(FINAL_METRICS_PATH, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\nSaved aggregate metrics -> {FINAL_METRICS_PATH}")

    csv_path = OUTPUTS_DIR / "final_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_image_rows)
    print(f"Saved per-image metrics -> {csv_path}")

    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ["accuracy", "precision", "recall", "specificity"]
    values = [agg[k] for k in labels]
    ax.bar(labels, values, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(f"Held-out evaluation (n={agg['n_samples']})")
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
    plt.tight_layout()
    bar_path = OUTPUTS_DIR / "final_metrics_bar.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved metrics bar chart -> {bar_path}")


if __name__ == "__main__":
    main()
