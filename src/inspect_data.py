"""
Environment sanity checks + dataset sanity checks.

Surfaces concrete risk points up front (missing GPU, wrong scikit-image
version, package import failures) and verifies the local Data/ mirror
actually uses the documented 3-color ground-truth scheme before anything
downstream trusts parse_gt_mask() on the whole dataset.

Usage:
    python src/inspect_data.py

Writes a verification figure to graphs/gt_color_check.png
"""

import sys
from collections import Counter
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEVICE, GRAPHS_DIR, GT_COLORS_BGR, require_data_root
from dataset import imread_unicode, nucleus_binary, scan_dataset


def environment_checks():
    print("=== Environment sanity checks ===")

    if torch.cuda.is_available():
        print(f"[PASS] GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARN] No GPU detected - training will be much slower on CPU.")

    try:
        import skimage

        ver = tuple(int(x) for x in skimage.__version__.split(".")[:2])
        if ver >= (0, 19):
            print(f"[PASS] scikit-image {skimage.__version__} supports channel_axis=")
        else:
            print(
                f"[WARN] scikit-image {skimage.__version__} is older than 0.19 - "
                "slic(..., channel_axis=2) may raise TypeError; postprocess.py auto-falls back "
                "to the legacy multichannel=True argument."
            )
    except Exception as e:
        print(f"[FAIL] Could not check scikit-image version: {e}")

    try:
        import pydensecrf.densecrf  # noqa: F401

        print("[PASS] pydensecrf imported successfully")
    except Exception as e:
        print(f"[FAIL] pydensecrf failed to import ({e}). pip install git+https://github.com/lucasb-eyer/pydensecrf.git")

    try:
        import maxflow  # noqa: F401

        print("[PASS] PyMaxflow imported successfully")
    except Exception as e:
        print(f"[FAIL] PyMaxflow failed to import ({e}). pip install PyMaxflow")

    print("=== End sanity checks ===\n")


def dataset_checks():
    data_root = require_data_root()
    print("Dataset root:", data_root)

    samples = scan_dataset(data_root)
    print("Total matched image/mask pairs:", len(samples))
    counts = Counter(row["class_name"] for row in samples)
    for class_name, count in counts.items():
        print(f"{class_name:22s} {count:4d}")

    if not samples:
        raise RuntimeError("No image/mask pairs were found. Check the Data/ folder layout.")

    sample = samples[0]
    image_bgr = imread_unicode(sample["image_path"], cv2.IMREAD_COLOR)
    mask_bgr = imread_unicode(sample["mask_path"], cv2.IMREAD_COLOR)
    if image_bgr is None or mask_bgr is None:
        raise ValueError("Could not read a sample image or mask from the dataset.")

    # Verify the color scheme on this actual copy of the dataset before trusting it dataset-wide
    pix = mask_bgr.reshape(-1, 3).astype(np.int32)
    print("\nSample mask color check (should be close to background/cytoplasm/nucleus BGR triplets):")
    uniq, cnts = np.unique(pix, axis=0, return_counts=True)
    order = np.argsort(-cnts)
    for i in order[:6]:
        print("  BGR", uniq[i], "count", cnts[i])
    print("Expected (from martin2003.pdf):", GT_COLORS_BGR)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
    nuc = nucleus_binary(mask_bgr)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_rgb)
    axes[0].set_title(f"Image: {sample['class_name']}")
    axes[0].axis("off")
    axes[1].imshow(mask_rgb)
    axes[1].set_title("Raw color-coded mask")
    axes[1].axis("off")
    axes[2].imshow(nuc, cmap="gray")
    axes[2].set_title("Parsed nucleus binary")
    axes[2].axis("off")
    plt.tight_layout()

    out_path = GRAPHS_DIR / "gt_color_check.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved ground-truth color verification figure -> {out_path}")


def main():
    environment_checks()
    dataset_checks()


if __name__ == "__main__":
    main()
