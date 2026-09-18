"""Disparity metrics for evaluation.

Kept separate from ``utils/losses.py`` on purpose. The metrics that come out of
``FoundationStereoLoss.__call__`` are training diagnostics: they need a full batch
dict, and they apply the bpX sample-filtering that stabilises training but has no
place in a reported number. These are plain functions over ``(pred, gt, valid)``.

Everything is in disparity pixels. Aggregation is **pixel-weighted**: a frame with
twice the valid pixels counts twice, which is what you want when scenes have very
different amounts of ground truth. :meth:`MetricAccumulator.frame_averaged` gives
the other convention if you need it.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

#: Thresholds, in pixels, for the bad-N rate. Reported as bad1/bad2/bad3.
DEFAULT_BAD_THRESHOLDS = (1.0, 2.0, 3.0)

#: KITTI D1: an outlier is off by more than 3 px *and* more than 5% of the true
#: disparity, so far-field pixels are not judged on the same absolute scale as
#: near ones.
D1_ABS_THRESHOLD = 3.0
D1_REL_THRESHOLD = 0.05


def bad_key(threshold: float) -> str:
    """Metric name for a bad-pixel threshold: 2.0 -> 'bad2', 0.5 -> 'bad0.5'."""
    return f"bad{threshold:g}"


def metric_names(bad_thresholds: Sequence[float] = DEFAULT_BAD_THRESHOLDS) -> List[str]:
    """Every metric key, in report order."""
    return ["epe", "rmse", *(bad_key(t) for t in bad_thresholds), "d1"]


def disparity_metrics(pred, gt, valid=None,
                      bad_thresholds: Sequence[float] = DEFAULT_BAD_THRESHOLDS,
                      max_gt_disparity: Optional[float] = None) -> Dict[str, float]:
    """Metrics for one frame.

    Args:
        pred, gt: disparity maps of the same shape, in pixels. Anything
            array-like; torch tensors are accepted and moved to numpy.
        valid: bool mask of pixels to score. Defaults to finite, positive ``gt``.
            Non-finite ``pred`` pixels are always excluded on top of this.
        bad_thresholds: thresholds in pixels for the bad-N rates.
        max_gt_disparity: drop pixels whose true disparity exceeds this.

    Returns:
        ``{"epe", "rmse", "bad1", ..., "d1", "n_valid"}``. Rates are fractions in
        [0, 1], not percentages. When nothing is valid every metric is NaN and
        ``n_valid`` is 0 -- callers should skip such frames rather than average
        the NaNs in (:class:`MetricAccumulator` does).
    """
    pred = _to_numpy(pred).astype(np.float32, copy=False)
    gt = _to_numpy(gt).astype(np.float32, copy=False)
    if pred.shape != gt.shape:
        raise ValueError(f"prediction is {pred.shape} but ground truth is {gt.shape}")

    if valid is None:
        mask = np.isfinite(gt) & (gt > 0)
    else:
        mask = _to_numpy(valid).astype(bool, copy=False)
        if mask.shape != gt.shape:
            raise ValueError(f"valid mask is {mask.shape} but ground truth is {gt.shape}")
        mask = mask & np.isfinite(gt)

    mask = mask & np.isfinite(pred)
    if max_gt_disparity is not None:
        mask = mask & (gt <= float(max_gt_disparity))

    n_valid = int(mask.sum())
    if n_valid == 0:
        return {**{name: float("nan") for name in metric_names(bad_thresholds)},
                "n_valid": 0}

    error = np.abs(pred[mask] - gt[mask])
    gt_valid = gt[mask]

    mse = float(np.mean(error ** 2))
    out = {
        "epe": float(error.mean()),
        "rmse": math.sqrt(mse),
        # Carried so aggregation can pool the squared error and take one root at
        # the end; averaging per-frame RMSEs is not the RMSE of the pool.
        "mse": mse,
    }
    for threshold in bad_thresholds:
        out[bad_key(threshold)] = float((error > float(threshold)).mean())
    out["d1"] = float(((error > D1_ABS_THRESHOLD) &
                       (error > D1_REL_THRESHOLD * np.abs(gt_valid))).mean())
    out["n_valid"] = n_valid
    return out


def region_masks(gt_disparity: np.ndarray, gt_valid_annotated: np.ndarray,
                 pred_valid: np.ndarray,
                 fg_mask: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """The three scoring regions for masked evaluation, as bool masks.

    ``on_geometry`` is annotated ground truth on the foreground, ``water_column``
    is everything the annotator marked as open water (any finite ground truth off
    the foreground), and ``combined`` is their union. Without a foreground mask,
    ``on_geometry`` degrades to all annotated ground truth and ``water_column``
    is empty.
    """
    if fg_mask is not None:
        on_geometry = gt_valid_annotated & pred_valid & fg_mask
        water_column = ~fg_mask & pred_valid & np.isfinite(gt_disparity)
    else:
        on_geometry = gt_valid_annotated & pred_valid
        water_column = np.zeros_like(pred_valid, dtype=bool)
    return {"on_geometry": on_geometry,
            "water_column": water_column,
            "combined": on_geometry | water_column}


class MetricAccumulator:
    """Pixel-weighted running totals, grouped by sequence.

    Frames with no valid ground truth contribute nothing but are counted, so
    ``n_frames`` and ``n_scored`` together say how much of the split was usable.
    NaN and inf metric values are treated as unusable rather than poisoning the
    mean -- the training-time ``evaluate()`` gets this wrong in the other
    direction (it adds 0 but still increments the denominator).
    """

    def __init__(self, bad_thresholds: Sequence[float] = DEFAULT_BAD_THRESHOLDS):
        self.bad_thresholds = tuple(bad_thresholds)
        self.names = metric_names(self.bad_thresholds)
        #: What is actually summed. "mse" is pooled and rooted at the end to give
        #: rmse; "rmse" itself is per-frame only and never averaged directly.
        self._summed = [n for n in self.names if n != "rmse"] + ["mse"]
        self._weighted = defaultdict(lambda: defaultdict(float))  # seq -> metric -> sum
        self._per_frame = defaultdict(lambda: defaultdict(float))
        self._pixels = defaultdict(float)
        self._frames = defaultdict(int)
        self._scored = defaultdict(int)

    def add(self, sequence: str, metrics: Dict[str, float]) -> None:
        """Fold in one frame's metrics, as returned by :func:`disparity_metrics`."""
        self._frames[sequence] += 1
        n_valid = float(metrics.get("n_valid", 0) or 0)
        if n_valid <= 0:
            return

        usable = {name: float(metrics[name]) for name in self._summed
                  if name in metrics and math.isfinite(float(metrics[name]))}
        if len(usable) != len(self._summed):
            return

        self._scored[sequence] += 1
        self._pixels[sequence] += n_valid
        for name, value in usable.items():
            self._weighted[sequence][name] += value * n_valid
            self._per_frame[sequence][name] += value

    def extend(self, rows: Iterable[Dict[str, float]], sequence_key: str = "sequence") -> None:
        """Fold in an iterable of per-frame rows that carry their own sequence name."""
        for row in rows:
            self.add(row[sequence_key], row)

    @property
    def sequences(self) -> List[str]:
        return sorted(self._frames)

    def per_sequence(self) -> Dict[str, Dict[str, float]]:
        """One pixel-weighted row per sequence."""
        return {sequence: self._summarize([sequence]) for sequence in self.sequences}

    def overall(self) -> Dict[str, float]:
        """One pixel-weighted row over every sequence."""
        return self._summarize(self.sequences)

    def frame_averaged(self) -> Dict[str, float]:
        """Overall row where every scored frame counts equally."""
        scored = sum(self._scored.values())
        row = {"n_frames": sum(self._frames.values()), "n_scored": scored,
               "n_valid": int(sum(self._pixels.values()))}
        for name in self._summed:
            total = sum(self._per_frame[s][name] for s in self.sequences)
            row[name] = total / scored if scored else float("nan")
        return self._finish(row)

    def frame_averaged_per_sequence(self) -> Dict[str, Dict[str, float]]:
        """One row per sequence where every scored frame counts equally."""
        out = {}
        for sequence in self.sequences:
            scored = self._scored[sequence]
            row = {"n_frames": self._frames[sequence], "n_scored": scored,
                   "n_valid": int(self._pixels[sequence])}
            for name in self._summed:
                row[name] = (self._per_frame[sequence][name] / scored
                             if scored else float("nan"))
            out[sequence] = self._finish(row)
        return out

    def _summarize(self, sequences: Sequence[str]) -> Dict[str, float]:
        pixels = sum(self._pixels[s] for s in sequences)
        row = {"n_frames": sum(self._frames[s] for s in sequences),
               "n_scored": sum(self._scored[s] for s in sequences),
               "n_valid": int(pixels)}
        for name in self._summed:
            total = sum(self._weighted[s][name] for s in sequences)
            row[name] = total / pixels if pixels else float("nan")
        return self._finish(row)

    @staticmethod
    def _finish(row: Dict[str, float]) -> Dict[str, float]:
        """Turn the pooled mean squared error into an rmse column and drop it."""
        mse = row.pop("mse", float("nan"))
        row["rmse"] = math.sqrt(mse) if math.isfinite(mse) and mse >= 0 else float("nan")
        return row


def format_table(rows: Dict[str, Dict[str, float]], names: Sequence[str],
                 label: str = "sequence") -> str:
    """Fixed-width table of ``{row_name: metrics}``, for printing to a terminal."""
    columns = [label, "frames", "px", *names]
    width = {label: max(len(label), *(len(str(k)) for k in rows)) if rows else len(label)}
    lines = []

    def cell(name, value):
        if name in ("frames", "px"):
            return f"{int(value):>{max(len(name), 8)}d}"
        return f"{value:>{max(len(name), 8)}.4f}"

    header = f"{label:<{width[label]}}  " + "  ".join(
        f"{c:>{max(len(c), 8)}}" for c in columns[1:])
    lines.append(header)
    lines.append("-" * len(header))
    for key, row in rows.items():
        cells = [cell("frames", row.get("n_scored", 0)), cell("px", row.get("n_valid", 0))]
        cells += [cell(name, row.get(name, float("nan"))) for name in names]
        lines.append(f"{key:<{width[label]}}  " + "  ".join(cells))
    return "\n".join(lines)


def _to_numpy(x):
    """Array-like -> numpy, without importing torch just to check for it."""
    if isinstance(x, np.ndarray):
        return np.squeeze(x)
    detach = getattr(x, "detach", None)
    if detach is not None:               # a torch tensor
        return np.squeeze(detach().cpu().numpy())
    return np.squeeze(np.asarray(x))
