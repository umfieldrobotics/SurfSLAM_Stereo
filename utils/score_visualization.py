"""Per-frame region-error panels for score_predictions.py.

One 2x3 figure per frame: ground truth, prediction, and the mask overlay on top;
the on_geometry / water_column / combined error maps below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from utils.metrics import region_masks  # noqa: E402


def save_region_visualization(gt_disparity: np.ndarray, prediction: np.ndarray,
                              gt_annotated: np.ndarray, pred_valid: np.ndarray,
                              fg_mask: Optional[np.ndarray],
                              max_error: Optional[float],
                              output_path: Path) -> None:
    finite_gt = np.isfinite(gt_disparity)
    regions = region_masks(gt_disparity, gt_annotated, pred_valid, fg_mask)
    combined = regions["combined"]

    error = np.abs(prediction - gt_disparity)

    # A shared disparity range across GT and prediction.
    shown = np.concatenate([gt_disparity[finite_gt], prediction[pred_valid]])
    if shown.size:
        disp_min, disp_max = float(shown.min()), float(shown.max())
        if disp_max <= disp_min:
            disp_max = disp_min + 1.0
    else:
        disp_min, disp_max = 0.0, 1.0

    disparity_cmap = plt.get_cmap("viridis")

    gt_vis = np.where(finite_gt, gt_disparity, np.nan).astype(np.float32)
    pred_vis = np.where(pred_valid, prediction, np.nan).astype(np.float32)

    # GT disparity with the foreground mask (or, without one, the scored region)
    # overlaid in red.
    normalized = np.clip((gt_vis - disp_min) / (disp_max - disp_min), 0.0, 1.0)
    overlay = disparity_cmap(normalized)
    overlay[~finite_gt] = (0.0, 0.0, 0.0, 1.0)
    overlaid = fg_mask if fg_mask is not None else combined
    overlay[..., :3][overlaid] = (0.5 * overlay[..., :3][overlaid]
                                  + 0.5 * np.array([1.0, 0.0, 0.0]))

    if max_error is not None and max_error > 0.0:
        error_max = float(max_error)
    else:
        error_max = float(error[combined].max()) if combined.any() else 1.0
        if not np.isfinite(error_max) or error_max <= 0.0:
            error_max = 1.0

    error_cmap = plt.get_cmap("magma").copy()
    error_cmap.set_bad(color="black")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    panels = [
        (axes[0, 0], gt_vis, "GT disparity", disparity_cmap, disp_min, disp_max),
        (axes[0, 1], pred_vis, "Predicted disparity", disparity_cmap, disp_min, disp_max),
    ]
    for axis, image, title, cmap, vmin, vmax in panels:
        im = axis.imshow(image, vmin=vmin, vmax=vmax, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
        fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)

    axes[0, 2].imshow(overlay[..., :3])
    axes[0, 2].set_title("GT disparity + mask overlay")
    axes[0, 2].axis("off")

    for axis, (name, title) in zip(
            axes[1], [("on_geometry", "On-geometry error"),
                      ("water_column", "Water-column error"),
                      ("combined", "Combined error")]):
        masked_error = np.where(regions[name], error, np.nan).astype(np.float32)
        im = axis.imshow(masked_error, vmin=0.0, vmax=error_max, cmap=error_cmap)
        axis.set_title(title)
        axis.axis("off")
        fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
