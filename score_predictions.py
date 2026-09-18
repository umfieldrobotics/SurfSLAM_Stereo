"""Score precomputed disparity predictions against ground truth.

    ./eval_scripts/score.sh tbnms scoring.predictions_dir=eval/defom_stereo_suds_test/ours

Reads the per-frame ``<scene>/data/<frame>.pt`` payloads an evaluation run wrote
(no model is loaded) and scores them per region, writing to ``scoring.output_dir``
(default ``<predictions_dir>/scores``):

    metrics.jsonl                   one row per (frame, variant, region), resumable
    metrics.csv                     the same rows, merged and deduplicated
    summary.json                    per-region, per-scene and overall
    latex_rows.txt                  tab-separated paste rows per scene

Two variants:

    masked      three regions from the foreground masks: on_geometry (annotated
                GT on the foreground), water_column (finite GT off the
                foreground), and combined (their union).
    unmasked    two regions with no foreground mask: annotated (all annotated
                GT) and full (everywhere the prediction is finite).

Predictions saved from padded inputs (e.g. 1088 rows against 1080-row ground
truth) are cropped back with the same centered padding the models used.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.arguments import EVAL_CONFIG_DIR, Config, get_args  # noqa: E402
from utils.eval_io import append_rows, json_safe, read_rows, write_csv  # noqa: E402
from utils.metrics import (MetricAccumulator, bad_key, disparity_metrics,  # noqa: E402
                           format_table, metric_names, region_masks)

VARIANTS = ("masked", "unmasked")
MASKED_REGIONS = ("on_geometry", "water_column", "combined")
UNMASKED_REGIONS = ("annotated", "full")

PAD_DIVISOR = 32


def to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.squeeze(np.asarray(x))


def unpad_to(array: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Crop off the centered padding that inflated ``shape`` to the next multiple
    of 32 (how every model here pads its input). Anything else is an error."""
    ht, wd = shape
    pad_ht = (((ht // PAD_DIVISOR) + 1) * PAD_DIVISOR - ht) % PAD_DIVISOR
    pad_wd = (((wd // PAD_DIVISOR) + 1) * PAD_DIVISOR - wd) % PAD_DIVISOR
    if array.shape != (ht + pad_ht, wd + pad_wd):
        raise ValueError(
            f"prediction is {array.shape} but ground truth is {(ht, wd)}, which "
            f"pads to {(ht + pad_ht, wd + pad_wd)}; this is not a padding "
            f"mismatch this tool knows how to undo.")
    top, left = pad_ht // 2, pad_wd // 2
    return array[top:top + ht, left:left + wd]


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------

class GroundTruthFrame:
    """One frame's ground truth: raw disparity (NaNs intact), which pixels are
    annotated, and the foreground mask if the frame has one."""

    def __init__(self, disparity: np.ndarray, annotated: np.ndarray,
                 fg_mask: Optional[np.ndarray]):
        self.disparity = disparity
        self.annotated = annotated
        self.fg_mask = fg_mask


class SudsGroundTruth:
    """The SUDS release layout, located via the dataset's own path helpers."""

    def __init__(self, args: Config):
        from data.tbnms_dataset import TBNMSDataset
        self.dataset = TBNMSDataset(
            split="all", root=args.io.suds_stereo_dir, max_frames=0,
            load_ground_truth=False, transform=None, device="cpu",
            check_ground_truth_rectification=args.evaluation.check_gt_rectification)
        self.check_rectification = args.evaluation.check_gt_rectification

    def describe(self) -> Dict:
        return {"gt_source": "suds", "root": str(self.dataset.root),
                "mask_rule": "annotated where mask > 0 ({0,1,255} masks)"}

    def mask_pattern(self, scene: str) -> str:
        return str(self.dataset.gt_mask_dir / scene / "masks" / "<frame>_left_mask.png")

    def read(self, scene: str, stem: str) -> Optional[GroundTruthFrame]:
        import imageio.v3 as iio
        disparity_path, mask_path = self.dataset.ground_truth_paths(scene, stem)
        if not disparity_path.exists():
            return None

        payload = torch.load(disparity_path.as_posix(), map_location="cpu",
                             weights_only=False)
        if isinstance(payload, dict):
            if "disparity" not in payload:
                raise KeyError(f"{disparity_path} has no 'disparity' key "
                               f"(found: {sorted(payload)})")
            if self.check_rectification and "K" in payload:
                self.dataset._check_rectification(payload["K"], scene, disparity_path)
            disparity = payload["disparity"]
        else:
            disparity = payload
        disparity = to_numpy(disparity).astype(np.float32)
        annotated = np.isfinite(disparity) & (disparity != 0)

        fg_mask = None
        if mask_path.exists():
            mask = to_numpy(iio.imread(mask_path.as_posix()))
            if mask.ndim == 3:
                mask = mask[..., 0]
            if mask.shape != disparity.shape:
                raise ValueError(f"{mask_path} is {mask.shape} but its disparity "
                                 f"is {disparity.shape}")
            # {0, 1, 255}: both non-zero classes are inside the annotation.
            fg_mask = mask > 0
        return GroundTruthFrame(disparity, annotated, fg_mask)


class LegacyGroundTruth:
    """The pre-release TBNMS_EVALUATION_FINAL layout:
    ``<gt_dir>/<scene>/disparity_maps/<frame>.pt`` with a ``depth_mask`` key, and
    masks at ``<mask_dir>/<scene>/masks_fg/<frame>_left_mask.png``."""

    def __init__(self, gt_dir: str, mask_dir: Optional[str]):
        self.gt_dir = Path(gt_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        if not self.gt_dir.is_dir():
            raise FileNotFoundError(f"scoring.legacy_gt_dir: {self.gt_dir} is not "
                                    f"a directory")

    def describe(self) -> Dict:
        return {"gt_source": "legacy", "gt_dir": str(self.gt_dir),
                "mask_dir": str(self.mask_dir),
                "mask_rule": "foreground where mask > 127, nearest-neighbour "
                             "resized to the GT shape"}

    def mask_pattern(self, scene: str) -> str:
        base = self.mask_dir if self.mask_dir else Path("<scoring.legacy_mask_dir>")
        return str(base / scene / "masks_fg" / "<frame>_left_mask.png")

    def read(self, scene: str, stem: str) -> Optional[GroundTruthFrame]:
        import cv2
        gt_path = self.gt_dir / scene / "disparity_maps" / f"{stem}.pt"
        if not gt_path.exists():
            return None

        payload = torch.load(gt_path.as_posix(), map_location="cpu",
                             weights_only=False)
        disparity = to_numpy(payload["disparity"]).astype(np.float32)
        depth_mask = to_numpy(payload["depth_mask"]).astype(bool)
        annotated = depth_mask & np.isfinite(disparity)

        fg_mask = None
        if self.mask_dir is not None:
            mask_path = self.mask_dir / scene / "masks_fg" / f"{stem}_left_mask.png"
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    fg_mask = mask > 127
                    if fg_mask.shape != disparity.shape:
                        fg_mask = cv2.resize(
                            fg_mask.astype(np.uint8),
                            (disparity.shape[1], disparity.shape[0]),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
        return GroundTruthFrame(disparity, annotated, fg_mask)


def build_gt_source(args: Config):
    scoring = args.scoring
    if scoring.gt_source == "suds":
        return SudsGroundTruth(args)
    if scoring.gt_source == "legacy":
        if not scoring.legacy_gt_dir:
            raise ValueError("scoring.gt_source=legacy needs scoring.legacy_gt_dir "
                             "(and scoring.legacy_mask_dir for the masked variant)")
        return LegacyGroundTruth(scoring.legacy_gt_dir, scoring.legacy_mask_dir)
    raise ValueError(f"scoring.gt_source must be 'suds' or 'legacy', "
                     f"got {scoring.gt_source!r}")


# --------------------------------------------------------------------------
# Predictions
# --------------------------------------------------------------------------

def load_prediction(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Disparity and validity from one prediction payload. ``depth_valid``, when
    present and the same shape, is ANDed into the validity."""
    payload = torch.load(path.as_posix(), map_location="cpu", weights_only=False)
    disparity = to_numpy(payload["disp_pred"]).astype(np.float32)
    if disparity.ndim == 3 and disparity.shape[-1] == 1:
        disparity = disparity[..., 0]
    valid = np.isfinite(disparity)
    if "depth_valid" in payload:
        depth_valid = to_numpy(payload["depth_valid"]).astype(bool)
        if depth_valid.ndim == 3 and depth_valid.shape[-1] == 1:
            depth_valid = depth_valid[..., 0]
        if depth_valid.shape == disparity.shape:
            valid &= depth_valid
    return disparity, valid


def discover_scenes(predictions_dir: Path,
                    requested: Optional[List[str]]) -> List[str]:
    found = sorted(d.name for d in predictions_dir.iterdir()
                   if d.is_dir() and any((d / "data").glob("*.pt")))
    if not found:
        raise FileNotFoundError(
            f"{predictions_dir} contains no <scene>/data/*.pt predictions. Point "
            f"scoring.predictions_dir at an evaluation stage directory.")
    if not requested:
        return found
    missing = sorted(set(requested) - set(found))
    if missing:
        raise FileNotFoundError(
            f"scoring.scenes names {', '.join(missing)}, but {predictions_dir} "
            f"only has: {', '.join(found)}")
    return [s for s in found if s in set(requested)]


def resolve_targets(scoring) -> List[Tuple[Path, Path]]:
    """(predictions_dir, output_dir) pairs to score."""
    if scoring.predictions_dir and scoring.run:
        raise ValueError("set scoring.predictions_dir or scoring.run, not both")
    if scoring.predictions_dir:
        predictions_dir = Path(scoring.predictions_dir)
        out_dir = Path(scoring.output_dir) if scoring.output_dir \
            else predictions_dir / "scores"
        return [(predictions_dir, out_dir)]
    if scoring.run:
        run_dir = Path(scoring.run)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"scoring.run: {run_dir} is not a directory")
        stages = sorted(
            stage for stage in run_dir.iterdir() if stage.is_dir()
            and any(scene.is_dir() and any((scene / "data").glob("*.pt"))
                    for scene in stage.iterdir()))
        if not stages:
            raise FileNotFoundError(
                f"{run_dir} has no stage directories with <scene>/data/*.pt "
                f"predictions; was the run made with evaluation.save_predictions?")
        base = Path(scoring.output_dir) if scoring.output_dir else None
        return [(stage, (base / stage.name) if base else stage / "scores")
                for stage in stages]
    raise ValueError("set scoring.predictions_dir (a stage directory) or "
                     "scoring.run (a run directory of stages)")


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def frame_rows(scene: str, stem: str, prediction: np.ndarray,
               pred_valid: np.ndarray, gt: GroundTruthFrame, variants: List[str],
               require_masks: bool, thresholds: List[float]) -> List[Dict]:
    rows = []

    def score(variant: str, region: str, mask: np.ndarray) -> None:
        metrics = disparity_metrics(prediction, gt.disparity, mask,
                                    bad_thresholds=thresholds)
        rows.append({"sequence": scene, "frame_key": stem,
                     "variant": variant, "region": region, **metrics})

    if "masked" in variants and (gt.fg_mask is not None or not require_masks):
        for region, mask in region_masks(gt.disparity, gt.annotated,
                                         pred_valid, gt.fg_mask).items():
            score("masked", region, mask)

    if "unmasked" in variants:
        score("unmasked", "annotated", gt.annotated & pred_valid)
        score("unmasked", "full", pred_valid)

    return rows


def latex_row(row: Dict[str, float], thresholds: List[float]) -> str:
    """EPE, then each bad-pixel rate and D1 as percentages, tab-separated."""
    def fmt(value: float, decimals: int) -> str:
        return f"{value:.{decimals}f}" if np.isfinite(value) else "nan"
    cells = [fmt(row.get("epe", float("nan")), 4)]
    cells += [fmt(row.get(bad_key(t), float("nan")) * 100.0, 2) for t in thresholds]
    cells.append(fmt(row.get("d1", float("nan")) * 100.0, 2))
    return "\t".join(cells)


def write_latex_rows(path: Path, accumulators: Dict[Tuple[str, str], MetricAccumulator],
                     variants: List[str], thresholds: List[float]) -> None:
    """Per-scene paste rows: the masked variant reports its three regions side by
    side, frame-averaged; the unmasked variant one region per line, pixel-weighted."""
    lines = []
    if "masked" in variants:
        per_scene = {region: accumulators[("masked", region)].frame_averaged_per_sequence()
                     for region in MASKED_REGIONS}
        overall = {region: accumulators[("masked", region)].frame_averaged()
                   for region in MASKED_REGIONS}
        scenes = sorted(set().union(*(per_scene[r].keys() for r in MASKED_REGIONS)))
        lines.append(f"# masked (frame-averaged): "
                     + " | ".join(MASKED_REGIONS)
                     + f", each as EPE, BP@{{{','.join(f'{t:g}' for t in thresholds)}}}%, D1%")
        for scene in scenes:
            row = "\t".join(latex_row(per_scene[r].get(scene, {}), thresholds)
                            for r in MASKED_REGIONS)
            lines.append(f'Scene: "{scene}"')
            lines.append(row)
        lines.append("OVERALL")
        lines.append("\t".join(latex_row(overall[r], thresholds)
                               for r in MASKED_REGIONS))
    if "unmasked" in variants:
        lines.append("# unmasked (pixel-weighted): one line per region "
                     f"({', '.join(UNMASKED_REGIONS)}), each as EPE, BP%, D1%")
        per_scene = {region: accumulators[("unmasked", region)].per_sequence()
                     for region in UNMASKED_REGIONS}
        scenes = sorted(set().union(*(per_scene[r].keys() for r in UNMASKED_REGIONS)))
        for scene in scenes:
            lines.append(f'Scene: "{scene}"')
            for region in UNMASKED_REGIONS:
                lines.append(latex_row(per_scene[region].get(scene, {}), thresholds))
        lines.append("OVERALL")
        for region in UNMASKED_REGIONS:
            lines.append(latex_row(accumulators[("unmasked", region)].overall(),
                                   thresholds))
    path.write_text("\n".join(lines) + "\n")


def score_target(args: Config, gt_source, predictions_dir: Path,
                 out_dir: Path) -> None:
    scoring = args.scoring
    variants = list(scoring.variants)
    thresholds = list(scoring.bad_thresholds)
    scenes = discover_scenes(predictions_dir, scoring.scenes)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_path = out_dir / "metrics.jsonl"
    done = set()
    if scoring.skip_existing:
        done = {(row["variant"], row["sequence"], row["frame_key"])
                for row in read_rows(rows_path)}
    elif rows_path.exists():
        rows_path.unlink()

    print(f"scoring {predictions_dir} -> {out_dir}")
    print(f"  scenes:   {len(scenes)} ({', '.join(scenes)})")
    print(f"  variants: {', '.join(variants)}")

    n_no_gt = 0
    frames_without_mask: List[str] = []
    scene_mask_counts: Dict[str, int] = {}

    for scene in scenes:
        pred_paths = sorted((predictions_dir / scene / "data").glob("*.pt"))
        if scoring.max_frames is not None and 0 <= scoring.max_frames < len(pred_paths):
            indices = np.linspace(0, len(pred_paths) - 1,
                                  num=scoring.max_frames, dtype=int)
            pred_paths = [pred_paths[i] for i in sorted(set(indices.tolist()))]

        scene_mask_counts[scene] = 0
        n_examined = 0
        for pred_path in tqdm(pred_paths, desc=scene, dynamic_ncols=True):
            stem = pred_path.stem
            if all((variant, scene, stem) in done for variant in variants):
                continue

            gt = gt_source.read(scene, stem)
            if gt is None:
                n_no_gt += 1
                continue
            n_examined += 1
            if gt.fg_mask is not None:
                scene_mask_counts[scene] += 1
            elif "masked" in variants:
                frames_without_mask.append(f"{scene}/{stem}")

            prediction, pred_valid = load_prediction(pred_path)
            if prediction.shape != gt.disparity.shape:
                prediction = unpad_to(prediction, gt.disparity.shape)
                pred_valid = unpad_to(pred_valid, gt.disparity.shape)

            rows = frame_rows(scene, stem, prediction, pred_valid, gt, variants,
                              scoring.require_masks, thresholds)
            append_rows(rows_path, rows)

            if scoring.save_visualizations and "masked" in variants:
                from utils.score_visualization import save_region_visualization
                save_region_visualization(
                    gt.disparity, prediction, gt.annotated, pred_valid, gt.fg_mask,
                    scoring.max_error_for_vis,
                    out_dir / scene / "viz_error" / f"{stem}_vis.png")

        # A scene with predictions and GT but not one mask means the masks are
        # not where the configuration says they are.
        if ("masked" in variants and scoring.require_masks
                and n_examined and scene_mask_counts[scene] == 0):
            raise FileNotFoundError(
                f"scene '{scene}' has no foreground masks at "
                f"{gt_source.mask_pattern(scene)}. Fix the mask location, or set "
                f"scoring.require_masks=false to score without foreground masks "
                f"(on_geometry becomes all annotated GT, water_column empty).")

    # ---------- aggregate -----------------------------------------------------
    merged: Dict[Tuple[str, str, str, str], Dict] = {}
    for row in read_rows(rows_path):
        merged[(row["variant"], row["region"], row["sequence"], row["frame_key"])] = row
    all_rows = [merged[key] for key in sorted(merged)]
    if not all_rows:
        print("no frames scored: no predictions matched any ground truth.")
        return

    names = metric_names(thresholds)
    write_csv(out_dir / "metrics.csv", all_rows,
              ["variant", "region", "sequence", "frame_key", "n_valid", *names])

    regions = {"masked": MASKED_REGIONS, "unmasked": UNMASKED_REGIONS}
    accumulators = {(variant, region): MetricAccumulator(thresholds)
                    for variant in variants for region in regions[variant]}
    for row in all_rows:
        key = (row["variant"], row["region"])
        if key in accumulators:
            accumulators[key].add(row["sequence"], row)

    summary = {
        **gt_source.describe(),
        "predictions_dir": str(predictions_dir),
        "variants": list(variants),
        "bad_thresholds": thresholds,
        "require_masks": scoring.require_masks,
        "frames_without_ground_truth": n_no_gt,
        "frames_without_mask": len(frames_without_mask),
        "aggregation": {
            "overall": "pixel-weighted; rmse pooled over squared error",
            "overall_frame_averaged": "every scored frame counts equally",
        },
        "results": {
            variant: {
                region: {
                    "overall": accumulators[(variant, region)].overall(),
                    "overall_frame_averaged":
                        accumulators[(variant, region)].frame_averaged(),
                    "per_sequence": accumulators[(variant, region)].per_sequence(),
                    "per_sequence_frame_averaged":
                        accumulators[(variant, region)].frame_averaged_per_sequence(),
                } for region in regions[variant]
            } for variant in variants
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2) + "\n")

    write_latex_rows(out_dir / "latex_rows.txt", accumulators, variants, thresholds)

    if n_no_gt:
        print(f"\n{n_no_gt} prediction frames had no ground truth and were skipped.")
    if frames_without_mask:
        print(f"{len(frames_without_mask)} frames had ground truth but no "
              f"foreground mask and were "
              + ("skipped for the masked variant."
                 if scoring.require_masks else
                 "scored without one (on_geometry only, empty water_column)."))
    for (variant, region), accumulator in accumulators.items():
        print(f"\n===== {variant} / {region} =====")
        print(format_table(accumulator.per_sequence(), names))
        print(format_table({"ALL": accumulator.overall()}, names, label="overall"))
    print(f"\nwrote {out_dir / 'metrics.csv'}, summary.json, latex_rows.txt")


def main() -> int:
    args: Config = get_args(config_root=EVAL_CONFIG_DIR)
    scoring = args.scoring

    unknown = sorted(set(scoring.variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"scoring.variants contains {', '.join(unknown)}; "
                         f"choose from: {', '.join(VARIANTS)}")

    targets = resolve_targets(scoring)
    gt_source = build_gt_source(args)
    for predictions_dir, out_dir in targets:
        score_target(args, gt_source, predictions_dir, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
