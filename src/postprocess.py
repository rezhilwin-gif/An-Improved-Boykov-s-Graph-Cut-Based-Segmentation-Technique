"""
Postprocessing pipeline: SLIC superpixel smoothing -> Boykov graph cut ->
DenseCRF boundary refinement, wired together in run_full_pipeline().

PyMaxflow, pydensecrf and scikit-image are hard imports: if any of them fail
to install, this module fails loudly at import time instead of silently
falling back to a stub while still labeling the output "graph cut" / "CRF".
"""

import time

import cv2
import numpy as np

try:
    import maxflow
except ImportError as e:
    raise ImportError(
        "PyMaxflow is required for the Boykov graph-cut step. Install with:\n"
        "  pip install PyMaxflow"
    ) from e

try:
    from skimage.segmentation import slic
except ImportError as e:
    raise ImportError(
        "scikit-image is required for SLIC superpixels. Install with:\n"
        "  pip install scikit-image"
    ) from e

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels
except ImportError as e:
    raise ImportError(
        "pydensecrf is required for CRF boundary refinement. Install with:\n"
        "  pip install git+https://github.com/lucasb-eyer/pydensecrf.git"
    ) from e

from dataset import bias_correct, crop_to_cell, imread_unicode, load_pair
from model import extract_features


def superpixel_and_edges(img_bgr, n_segments=200, compactness=10):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    try:
        # scikit-image >= 0.19
        segments = slic(
            rgb, n_segments=n_segments, compactness=compactness, start_label=0, channel_axis=2
        )
    except TypeError:
        # scikit-image < 0.19: channel_axis doesn't exist yet, use the legacy argument
        segments = slic(
            rgb, n_segments=n_segments, compactness=compactness, start_label=0, multichannel=True
        )
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    zero_cross = np.zeros_like(lap, dtype=np.uint8)
    sign = np.sign(lap)
    zero_cross[:-1, :] |= sign[:-1, :] != sign[1:, :]
    zero_cross[:, :-1] |= sign[:, :-1] != sign[:, 1:]
    return segments, zero_cross.astype(np.float32)


def smooth_probabilities(prob_fg, segments):
    smoothed = prob_fg.copy()
    for label in np.unique(segments):
        mask = segments == label
        smoothed[mask] = prob_fg[mask].mean()
    return np.clip(smoothed, 1e-6, 1.0 - 1e-6)


def boykov_graph_cut(prob_fg, img_gray, k=50.0, sigma=20.0):
    """
    PyMaxflow's get_grid_segments() returns True for nodes on the SINK side of
    the min-cut. Given add_grid_tedges(nodes, source_cap, sink_cap) with
    source_cap=k*bg_cost and sink_cap=k*fg_cost: a node ends up on the SINK
    side (True) when source_cap < sink_cap, i.e. when bg_cost < fg_cost -
    which is precisely when the pixel is likely BACKGROUND. So the raw output
    has True/1 = background and False/0 = foreground - the opposite of what
    every downstream step treats label 1 as. The returned labels are inverted
    to correct this (1 = foreground/nucleus, 0 = background).
    """
    g = maxflow.Graph[float]()
    h, w = prob_fg.shape
    nodes = g.add_grid_nodes((h, w))
    eps = 1e-6
    fg_cost = -np.log(prob_fg + eps)
    bg_cost = -np.log(1.0 - prob_fg + eps)
    g.add_grid_tedges(nodes, k * bg_cost, k * fg_cost)

    right = np.exp(-((np.diff(img_gray.astype(np.float32), axis=1) ** 2) / (2.0 * sigma ** 2)))
    down = np.exp(-((np.diff(img_gray.astype(np.float32), axis=0) ** 2) / (2.0 * sigma ** 2)))
    g.add_grid_edges(nodes[:, :-1], weights=right, symmetric=True)
    g.add_grid_edges(nodes[:-1, :], weights=down, symmetric=True)

    g.maxflow()
    sink_side = g.get_grid_segments(nodes)  # True = SINK side = background
    labels = (~sink_side).astype(np.uint8)  # invert: 1 = foreground/nucleus
    return labels


def crf_boundary_recovery(img_bgr, unary_labels, max_iters=10):
    h, w = unary_labels.shape
    d = dcrf.DenseCRF2D(w, h, 2)
    unary = unary_from_labels(unary_labels.astype(np.uint8), 2, gt_prob=0.9, zero_unsure=False)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=20, srgb=13, rgbim=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), compat=10)

    labels = unary_labels
    prev = None
    for _ in range(max_iters):
        q = d.inference(1)
        labels = np.argmax(np.array(q), axis=0).reshape(h, w)
        if prev is not None and np.array_equal(labels, prev):
            break
        prev = labels
    return labels.astype(np.uint8)


def run_full_pipeline(
    model,
    img_path,
    mask_path=None,
    n_segments=200,
    compactness=10,
    k=50.0,
    sigma=20.0,
    crf_iters=10,
    warn_on_implausible_fg=True,
):
    """mask_path, if given, is used with the SAME crop bbox as the image (via
    load_pair) so the ground truth stays aligned with the prediction."""
    t0 = time.time()
    if mask_path is not None:
        cropped, gt = load_pair(img_path, mask_path)
    else:
        img_bgr = imread_unicode(img_path, cv2.IMREAD_COLOR)
        corrected = bias_correct(img_bgr)
        cropped = crop_to_cell(corrected)
        gt = None

    prob_fg = extract_features(model, cropped)
    segments, boundary_map = superpixel_and_edges(cropped, n_segments=n_segments, compactness=compactness)
    prob_fg = smooth_probabilities(prob_fg, segments)

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    coarse = boykov_graph_cut(prob_fg, gray, k=k, sigma=sigma)
    final_mask = crf_boundary_recovery(cropped, coarse, max_iters=crf_iters)
    elapsed = time.time() - t0

    # Nucleus is normally a MINORITY of a cropped cell image. If a majority of
    # pixels get called foreground, that's a strong signal of a labeling bug
    # rather than a real segmentation result.
    if warn_on_implausible_fg:
        fg_fraction = final_mask.mean()
        if fg_fraction > 0.6:
            print(
                f"  [WARN] predicted foreground fraction is {fg_fraction:.2f} (>60% of the "
                f"image) - nucleus is normally a minority region. This may indicate a label "
                f"inversion or other bug rather than a real result."
            )

    metrics = None
    if gt is not None:
        tp = np.sum((final_mask == 1) & (gt == 1))
        tn = np.sum((final_mask == 0) & (gt == 0))
        fp = np.sum((final_mask == 1) & (gt == 0))
        fn = np.sum((final_mask == 0) & (gt == 1))
        metrics = {
            "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1),
        }

    return {
        "image": cropped,
        "gt": gt,
        "prob_fg": prob_fg,
        "segments": segments,
        "boundary_map": boundary_map,
        "coarse_mask": coarse,
        "final_mask": final_mask,
        "metrics": metrics,
        "time": elapsed,
    }
