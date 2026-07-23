"""
Dataset loading and preprocessing for the Herlev Pap-smear dataset.

Handles:
  * Unicode-safe image reading
  * Color-coded ground-truth mask parsing (red=background, dark blue=cytoplasm,
    light blue=nucleus)
  * Illumination bias correction
  * A single shared crop bounding box applied to BOTH the image and the mask,
    so they stay pixel-aligned
  * Scanning Data/<class_name>/ for raw-image / mask pairs
  * The PyTorch Dataset used for training
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from config import EXPECTED_CLASSES, GT_COLOR_ARRAY, IMG_SIZE

# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


# --------------------------------------------------------------------------
# Ground-truth mask parsing
# --------------------------------------------------------------------------


def parse_gt_mask(mask_bgr):
    """(H, W) int array: 0=background, 1=cytoplasm, 2=nucleus, via nearest-color
    classification (robust to antialiasing at region borders)."""
    H, W = mask_bgr.shape[:2]
    pix = mask_bgr.reshape(-1, 3).astype(np.int32)
    dists = np.linalg.norm(pix[:, None, :] - GT_COLOR_ARRAY[None, :, :], axis=2)
    labels = np.argmin(dists, axis=1).reshape(H, W)
    return labels.astype(np.uint8)


def nucleus_binary(mask_bgr):
    return (parse_gt_mask(mask_bgr) == 2).astype(np.uint8)


# --------------------------------------------------------------------------
# Bias correction + shared crop
# --------------------------------------------------------------------------


def bias_correct(img_bgr, sigma=25):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) + 1.0
    log_img = np.log(gray)
    blur = cv2.GaussianBlur(log_img, (0, 0), sigma)
    high_pass = log_img - blur
    corrected = np.exp(high_pass)
    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = corrected
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def compute_crop_bbox(img_bgr, pad=10):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = img_bgr.shape[:2]
    if not contours:
        return 0, 0, W, H
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    return x0, y0, x1, y1


def crop_to_cell(img_bgr, pad=10):
    """Used at inference time on its own (no paired mask)."""
    x0, y0, x1, y1 = compute_crop_bbox(img_bgr, pad=pad)
    return img_bgr[y0:y1, x0:x1]


def load_pair(image_path, mask_path):
    """Loads image+mask, bias-corrects the image, computes ONE crop bbox, applies
    it to BOTH image and mask so they stay pixel-aligned. Used by training so
    train/inference preprocessing match exactly."""
    img_bgr = imread_unicode(image_path, cv2.IMREAD_COLOR)
    mask_bgr = imread_unicode(mask_path, cv2.IMREAD_COLOR)
    if img_bgr is None or mask_bgr is None:
        raise ValueError(f"Could not read pair: {image_path} / {mask_path}")
    corrected = bias_correct(img_bgr)
    x0, y0, x1, y1 = compute_crop_bbox(corrected)
    cropped_img = corrected[y0:y1, x0:x1]
    cropped_mask = mask_bgr[y0:y1, x0:x1]
    gt_nucleus = nucleus_binary(cropped_mask)
    return cropped_img, gt_nucleus


# --------------------------------------------------------------------------
# Dataset scanning
# --------------------------------------------------------------------------


def resolve_raw_for_mask(mask_path):
    name = mask_path.name
    if not name.lower().endswith("-d.bmp"):
        return None
    base = name[:-6]
    candidates = [
        mask_path.with_name(base + ".BMP"),
        mask_path.with_name(base + "_2.BMP"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def scan_dataset(root_dir):
    rows = []
    for class_name in EXPECTED_CLASSES:
        class_dir = Path(root_dir) / class_name
        if not class_dir.exists():
            continue
        mask_candidates = sorted(class_dir.glob("*-d.bmp")) + sorted(class_dir.glob("*-d.BMP"))
        for mask_path in mask_candidates:
            raw_path = resolve_raw_for_mask(mask_path)
            if raw_path is None:
                continue
            rows.append(
                {
                    "class_name": class_name,
                    "image_path": str(raw_path),
                    "mask_path": str(mask_path),
                }
            )
    return rows


# --------------------------------------------------------------------------
# Preprocessing transform + PyTorch Dataset
# --------------------------------------------------------------------------

preprocess = T.Compose(
    [
        T.ToPILImage(),
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class HerlevSegDataset(Dataset):
    """Loads a Herlev image + its color-coded mask using bias-correction + crop."""

    def __init__(self, img_paths, mask_paths, size=IMG_SIZE):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.size = size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        cropped_img, gt_nucleus = load_pair(self.img_paths[i], self.mask_paths[i])
        rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        img_t = preprocess(rgb)
        mask_resized = cv2.resize(
            gt_nucleus, (self.size, self.size), interpolation=cv2.INTER_NEAREST
        )
        return img_t, torch.from_numpy(mask_resized).float()
