"""
Runs the full pipeline (model -> graph cut -> CRF) on demo samples and saves
the 4-panel stage visualization: preprocessed image / foreground probability
/ Boykov graph cut / CRF refined mask.

Usage:
    python src/visualize_preds.py [--n N]

Reads checkpoints/best_model.pt and checkpoints/best_sweep_config.json
(falls back to default graph-cut/CRF parameters if the sweep hasn't been
run yet). Writes figures to graphs/pred_sample_<i>.png
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import BEST_MODEL_PATH, GRAPHS_DIR, SWEEP_CONFIG_PATH, require_data_root
from dataset import scan_dataset
from model import load_model
from postprocess import run_full_pipeline

DEFAULT_SWEEP = {"k": 50.0, "sigma": 20.0, "segments": 200, "crf_iters": 10}


def load_sweep_config():
    if SWEEP_CONFIG_PATH.exists():
        with open(SWEEP_CONFIG_PATH) as f:
            return json.load(f)
    print(f"[WARN] {SWEEP_CONFIG_PATH} not found, using default graph-cut/CRF params.")
    return DEFAULT_SWEEP


def visualize_sample(model, sample, cfg, out_path):
    result = run_full_pipeline(
        model,
        sample["image_path"],
        sample["mask_path"],
        n_segments=cfg["segments"],
        compactness=10,
        k=cfg["k"],
        sigma=cfg["sigma"],
        crf_iters=cfg["crf_iters"],
    )
    print(f"Evaluated {sample['image_path']}")
    if result["metrics"] is not None:
        print({k: round(float(v), 4) for k, v in result["metrics"].items()})

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(cv2.cvtColor(result["image"], cv2.COLOR_BGR2RGB))
    axes[0].set_title("Preprocessed image")
    axes[0].axis("off")
    axes[1].imshow(result["prob_fg"], cmap="viridis")
    axes[1].set_title("Foreground probability")
    axes[1].axis("off")
    axes[2].imshow(result["coarse_mask"], cmap="gray")
    axes[2].set_title("Boykov graph cut")
    axes[2].axis("off")
    axes[3].imshow(result["final_mask"], cmap="gray")
    axes[3].set_title("CRF refined mask")
    axes[3].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")
    return result["metrics"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1, help="number of demo samples to visualize")
    args = parser.parse_args()

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"{BEST_MODEL_PATH} not found. Run src/train.py first.")

    samples = scan_dataset(require_data_root())
    if not samples:
        raise RuntimeError("No image/mask pairs were found under Data/.")

    model = load_model(checkpoint_path=BEST_MODEL_PATH)
    cfg = load_sweep_config()

    for i, sample in enumerate(samples[: args.n]):
        out_path = GRAPHS_DIR / f"pred_sample_{i}.png"
        visualize_sample(model, sample, cfg, out_path)


if __name__ == "__main__":
    main()
