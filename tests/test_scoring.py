"""Pins score_predictions.py to the metric definitions the CVPR tables used.

The reference implementations inside these tests are transcribed from the
original standalone evaluation scripts; the assertions prove the repo code
reproduces them exactly. Synthetic data only -- no datasets, no GPU.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from score_predictions import (LegacyGroundTruth, frame_rows,  # noqa: E402
                               load_prediction, unpad_to)
from utils.metrics import (MetricAccumulator, bad_key, disparity_metrics,  # noqa: E402
                           region_masks)

BP_THRESHOLDS = [0.5, 1.0, 3.0, 5.0, 10.0, 15.0]


# --------------------------------------------------------------------------
# Reference formulas (transcribed, kept independent of the code under test)
# --------------------------------------------------------------------------

def reference_metrics(err, gt_vals, thresholds):
    epe = float(err.mean())
    bp = {str(t): float((err > t).mean() * 100.0) for t in thresholds}
    bad = (err > 3.0) & (err > 0.05 * np.abs(gt_vals))
    d1 = float(bad.sum() / err.size * 100.0) if err.size > 0 else float("nan")
    return epe, bp, d1


def reference_regions(disp_gt, valid_gt_masked, valid_pred, fg_mask_bool):
    if fg_mask_bool is not None:
        on_geometry = valid_gt_masked & valid_pred & fg_mask_bool
        water_column = (~fg_mask_bool) & valid_pred & np.isfinite(disp_gt)
    else:
        on_geometry = valid_gt_masked & valid_pred
        water_column = np.zeros_like(valid_pred, dtype=bool)
    return {"on_geometry": on_geometry, "water_column": water_column,
            "combined": on_geometry | water_column}


def reference_pad(arr, divis_by=32):
    """Centered ('sintel') padding to the next multiple of divis_by."""
    ht, wd = arr.shape
    pad_ht = (((ht // divis_by) + 1) * divis_by - ht) % divis_by
    pad_wd = (((wd // divis_by) + 1) * divis_by - wd) % divis_by
    pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
    return torch.nn.functional.pad(
        torch.from_numpy(arr)[None, None], pad, mode="replicate"
    ).squeeze().numpy()


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def synthetic_frame(seed=0, shape=(64, 96)):
    rng = np.random.default_rng(seed)
    gt = rng.uniform(1.0, 60.0, size=shape).astype(np.float32)
    gt[5:9, :] = np.nan                        # unannotatable ground truth
    gt[12:14, :] = 0.0                          # finite but unannotated

    pred = gt + rng.normal(0.0, 2.0, size=shape).astype(np.float32)
    pred[np.abs(pred) < 0.5] += 20.0            # some large errors
    pred[20:22, :] = np.nan                     # invalid prediction

    # Foreground mask with all three release classes {0, 1, 255}.
    mask_u8 = np.zeros(shape, dtype=np.uint8)
    mask_u8[:, :40] = 255
    mask_u8[:, 40:60] = 1

    annotated = np.isfinite(gt) & (gt != 0)
    pred_valid = np.isfinite(pred)
    return gt, pred, annotated, pred_valid, mask_u8


def assert_row_matches_reference(row, err, gt_vals):
    epe, bp, d1 = reference_metrics(err, gt_vals, BP_THRESHOLDS)
    assert row["epe"] == pytest.approx(epe, abs=1e-6)
    assert row["d1"] * 100.0 == pytest.approx(d1, abs=1e-6)
    for t in BP_THRESHOLDS:
        assert row[bad_key(t)] * 100.0 == pytest.approx(bp[str(t)], abs=1e-6)


# --------------------------------------------------------------------------
# Region metrics
# --------------------------------------------------------------------------

def test_masked_regions_match_reference():
    gt, pred, annotated, pred_valid, mask_u8 = synthetic_frame()
    fg = mask_u8 > 127

    ours = region_masks(gt, annotated, pred_valid, fg)
    reference = reference_regions(gt, annotated, pred_valid, fg)
    for name in ("on_geometry", "water_column", "combined"):
        np.testing.assert_array_equal(ours[name], reference[name])
        assert ours[name].sum() > 0
        err = np.abs(pred[reference[name]] - gt[reference[name]])
        row = disparity_metrics(pred, gt, ours[name], bad_thresholds=BP_THRESHOLDS)
        assert_row_matches_reference(row, err, gt[reference[name]])
        assert row["n_valid"] == int(reference[name].sum())


def test_no_mask_fallback_matches_reference():
    gt, pred, annotated, pred_valid, _ = synthetic_frame()
    ours = region_masks(gt, annotated, pred_valid, None)
    reference = reference_regions(gt, annotated, pred_valid, None)
    for name in reference:
        np.testing.assert_array_equal(ours[name], reference[name])
    assert not ours["water_column"].any()


def test_empty_region_is_nan_not_zero():
    gt, pred, annotated, pred_valid, _ = synthetic_frame()
    empty = np.zeros_like(pred_valid)
    row = disparity_metrics(pred, gt, empty, bad_thresholds=BP_THRESHOLDS)
    assert row["n_valid"] == 0
    assert np.isnan(row["epe"]) and np.isnan(row["d1"])


def test_unmasked_variant_regions():
    gt, pred, annotated, pred_valid, _ = synthetic_frame()

    class Gt:
        disparity, fg_mask = gt, None
    Gt.annotated = annotated

    rows = frame_rows("scene", "0", pred, pred_valid, Gt, ["unmasked"],
                      require_masks=True, thresholds=BP_THRESHOLDS)
    by_region = {row["region"]: row for row in rows}
    assert set(by_region) == {"annotated", "full"}

    region = annotated & pred_valid
    err = np.abs(pred[region] - gt[region])
    assert_row_matches_reference(by_region["annotated"], err, gt[region])

    # The full region scores every finite prediction pixel with finite GT.
    region = pred_valid & np.isfinite(gt)
    err = np.abs(pred[region] - gt[region])
    assert_row_matches_reference(by_region["full"], err, gt[region])


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def test_frame_averaged_matches_running_totals():
    accumulator = MetricAccumulator(BP_THRESHOLDS)
    totals = {"epe": 0.0, "d1": 0.0,
              "bp": {str(t): 0.0 for t in BP_THRESHOLDS}, "n": 0}

    for seed in range(4):
        gt, pred, annotated, pred_valid, mask_u8 = synthetic_frame(seed=seed)
        region = reference_regions(gt, annotated, pred_valid,
                                   mask_u8 > 127)["on_geometry"]
        if seed == 3:
            region[:] = False           # an empty frame: skipped, not averaged in
        row = disparity_metrics(pred, gt, region, bad_thresholds=BP_THRESHOLDS)
        accumulator.add("scene", row)

        err = np.abs(pred[region] - gt[region])
        if region.sum() == 0:
            continue
        epe, bp, d1 = reference_metrics(err, gt[region], BP_THRESHOLDS)
        totals["epe"] += epe
        totals["d1"] += d1
        totals["n"] += 1
        for t in BP_THRESHOLDS:
            totals["bp"][str(t)] += bp[str(t)]

    result = accumulator.frame_averaged_per_sequence()["scene"]
    assert result["n_scored"] == totals["n"] == 3
    assert result["n_frames"] == 4
    assert result["epe"] == pytest.approx(totals["epe"] / totals["n"], abs=1e-6)
    assert result["d1"] * 100.0 == pytest.approx(totals["d1"] / totals["n"], abs=1e-6)
    for t in BP_THRESHOLDS:
        assert result[bad_key(t)] * 100.0 == pytest.approx(
            totals["bp"][str(t)] / totals["n"], abs=1e-6)
    assert result == accumulator.frame_averaged() | {
        k: result[k] for k in ("n_frames", "n_scored", "n_valid")}


# --------------------------------------------------------------------------
# Padding
# --------------------------------------------------------------------------

def test_unpad_inverts_model_padding():
    rng = np.random.default_rng(1)
    original = rng.uniform(0, 100, size=(1080, 1920)).astype(np.float32)
    padded = reference_pad(original)
    assert padded.shape == (1088, 1920)
    np.testing.assert_array_equal(unpad_to(padded, original.shape), original)

    original = rng.uniform(0, 100, size=(100, 150)).astype(np.float32)
    padded = reference_pad(original)
    assert padded.shape == (128, 160)
    np.testing.assert_array_equal(unpad_to(padded, original.shape), original)


def test_unpad_rejects_unexplained_shapes():
    array = np.zeros((1080, 1920), dtype=np.float32)
    with pytest.raises(ValueError):
        unpad_to(array, (1088, 1920))
    with pytest.raises(ValueError):
        unpad_to(np.zeros((1090, 1920), np.float32), (1080, 1920))


# --------------------------------------------------------------------------
# Legacy ground truth and prediction loading
# --------------------------------------------------------------------------

def test_legacy_ground_truth_mask_rule(tmp_path):
    cv2 = pytest.importorskip("cv2")
    gt, _, annotated, _, mask_u8 = synthetic_frame()
    depth_mask = annotated.copy()

    scene_dir = tmp_path / "gt" / "scene" / "disparity_maps"
    scene_dir.mkdir(parents=True)
    torch.save({"disparity": torch.from_numpy(gt),
                "depth_mask": torch.from_numpy(depth_mask)},
               scene_dir / "123.pt")
    mask_dir = tmp_path / "masks" / "scene" / "masks_fg"
    mask_dir.mkdir(parents=True)
    cv2.imwrite(str(mask_dir / "123_left_mask.png"), mask_u8)

    source = LegacyGroundTruth(str(tmp_path / "gt"), str(tmp_path / "masks"))
    frame = source.read("scene", "123")
    # Value 1 is background under the legacy > 127 rule.
    np.testing.assert_array_equal(frame.fg_mask, mask_u8 > 127)
    np.testing.assert_array_equal(frame.annotated,
                                  depth_mask & np.isfinite(gt))
    assert source.read("scene", "999") is None

    # Half-size masks are nearest-neighbour resized to the GT shape.
    small = mask_u8[::2, ::2]
    cv2.imwrite(str(mask_dir / "123_left_mask.png"), small)
    frame = source.read("scene", "123")
    assert frame.fg_mask.shape == gt.shape


def test_load_prediction_unsqueezes_and_ands_depth_valid(tmp_path):
    _, pred, _, _, _ = synthetic_frame()
    depth_valid = np.ones_like(pred, dtype=bool)
    depth_valid[0, :] = False
    torch.save({"disp_pred": torch.from_numpy(pred)[None],
                "depth_valid": torch.from_numpy(depth_valid)},
               tmp_path / "p.pt")
    disparity, valid = load_prediction(tmp_path / "p.pt")
    assert disparity.shape == pred.shape
    np.testing.assert_array_equal(valid, np.isfinite(pred) & depth_valid)
