"""Evaluate a stereo model on real underwater data.

    ./eval_scripts/inference.sh defom_stereo/suds_test 0

For each checkpoint in ``evaluation.checkpoints`` this writes, under
``<evaluation.output_dir>/<stage>/``:

    <scene>/data/<frame>.pt         per-frame prediction, ground truth and calibration
    <scene>/viz/<frame>.png         colormapped disparity
    metrics_rank<k>.jsonl           one row per frame, appended as it is produced
    metrics.csv                     the same rows, merged and deduplicated
    summary.json                    per-scene and overall

and ``comparison.csv`` at the run root, one row per checkpoint.

Datasets are built directly (not via ``create_dataset_manager``, which resizes
unconditionally) so evaluation runs at native resolution with no transform. The
forward pass and checkpoint resolution are shared with demo/.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent

# Append, never insert: the repo's own `utils` package must keep priority over
# models/DEFOM-Stereo/utils.
for _p in ("models", "models/DEFOM-Stereo", "models/FoundationStereo", "models/IGEV-plusplus"):
    _path = str(REPO_ROOT / _p)
    if _path not in sys.path:
        sys.path.append(_path)

# The repo root must be importable for models.Underwater_Stereo.Nets.bgnet.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import imageio.v3 as iio  # noqa: E402

from FoundationStereo.core.utils.utils import InputPadder  # noqa: E402

from demo.core import Checkpoint, colorize, forward_disparity, robust_limits  # noqa: E402
from demo.core import default_iters as configured_iters  # noqa: E402
from utils.arguments import EVAL_CONFIG_DIR, Config, get_args  # noqa: E402
from utils.dataset_paths import resolve_checkpoint  # noqa: E402
from utils.eval_io import append_rows, json_safe, read_rows, write_csv  # noqa: E402
from utils.metrics import (MetricAccumulator, disparity_metrics, format_table,  # noqa: E402
                           metric_names)
from utils.utils import cleanup_ddp, setup_ddp  # noqa: E402

autocast = torch.amp.autocast

#: Percentile range for the visualizations; fixed so pictures are comparable.
VIZ_PERCENTILES = (2.0, 98.0)


# --------------------------------------------------------------------------
# The forward pass
# --------------------------------------------------------------------------

#: Required input-size multiple. BGNet downsamples further than the rest and needs 64.
PAD_DIVISOR = {"underwater_stereo": 64}
DEFAULT_PAD_DIVISOR = 32

#: Largest disparity a model's architecture can represent (BGNet: 97 half-res
#: samples doubled), reported in summary.json.
ARCHITECTURAL_MAX_DISPARITY = {"underwater_stereo": 192.0}

#: Baselines handled here rather than in demo.core.forward_disparity. All are
#: single feed-forward passes, hence uses_iters below.
EVAL_ONLY_MODELS = ("underwater_stereo",)


def uses_iters(cfg: Config) -> bool:
    """Whether this model refines iteratively. BGNet is a single feed-forward pass."""
    return cfg.model not in EVAL_ONLY_MODELS


def forward_eval(module: torch.nn.Module, cfg: Config,
                 left: torch.Tensor, right: torch.Tensor,
                 iters: Optional[int] = None) -> torch.Tensor:
    """Disparity as ``[B, 1, H, W]``, for any model evaluation supports."""
    padder = InputPadder(left.shape,
                         divis_by=PAD_DIVISOR.get(cfg.model, DEFAULT_PAD_DIVISOR),
                         force_square=False)
    left, right = padder.pad(left, right)

    if cfg.model == "underwater_stereo":
        # BGNet wants 0..1 input and returns (disparity, disparity) as [B, H, W].
        out, _ = module(left / 255.0, right / 255.0)
        disparity = out.unsqueeze(1).float()
    else:
        disparity = forward_disparity(module, cfg, left, right, iters)

    return padder.unpad(disparity)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def cap_frames(dataset, max_frames: Optional[int]):
    """Take ``max_frames`` evenly spaced samples, for loaders that can't cap themselves."""
    if max_frames is None or max_frames < 0 or max_frames >= len(dataset):
        return dataset
    indices = np.linspace(0, len(dataset) - 1, num=max_frames, dtype=int)
    return torch.utils.data.Subset(dataset, sorted(set(indices.tolist())))


def build_eval_dataset(name: str, args: Config, device):
    """One dataset at native resolution. Only ``tbnms`` (SUDS) has ground truth."""
    evaluation = args.evaluation

    if name == "tbnms":
        from data.tbnms_dataset import TBNMSDataset
        return TBNMSDataset(
            device=device,
            split=evaluation.split,
            root=args.io.suds_stereo_dir,
            scene_filter=list(evaluation.scene_filter) if evaluation.scene_filter else None,
            max_frames=evaluation.max_frames,
            seed=args.optimization.seed,
            load_ground_truth=True,
            use_foreground_mask=evaluation.use_foreground_mask,
            check_ground_truth_rectification=evaluation.check_gt_rectification,
            transform=None,
        )
    if name == "svin2":
        from data.svin2_dataset import SVIN2Dataset
        return cap_frames(
            SVIN2Dataset(device=device, root=args.io.svin2_dir, transform=None),
            evaluation.max_frames)
    if name == "lizard_island":
        from data.lizard_island_dataset import LizardIslandDataset
        return cap_frames(
            LizardIslandDataset(device=device, root=args.io.lizard_island_dir,
                                transform=None),
            evaluation.max_frames)

    raise ValueError(
        f"evaluation.datasets contains '{name}', which has no evaluation loader. "
        f"Supported: tbnms (SUDS, with ground truth), svin2, lizard_island. The "
        f"simulated sets (tartanair, oceansim) need the water augmentation pipeline "
        f"to produce left_uw/right_uw at all and are not wired up here.")


def build_eval_loader(args: Config, device, world_size: int, rank: int):
    """A loader over every requested dataset, sharded across ranks."""
    datasets = [build_eval_dataset(name, args, device)
                for name in args.evaluation.datasets]
    dataset = datasets[0] if len(datasets) == 1 else torch.utils.data.ConcatDataset(datasets)

    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                     shuffle=False)
    loader = DataLoader(
        dataset,
        batch_size=args.optimization.batch_size_per_gpu,
        sampler=sampler,
        shuffle=False,
        num_workers=args.optimization.num_workers,
        pin_memory=False,
    )
    return loader, len(dataset)


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def frame_keys(batch) -> List[str]:
    """Per-frame filename keys: SUDS's ``frame_stem`` when present, else ``frame_id``."""
    if "frame_stem" in batch:
        return [str(stem) for stem in batch["frame_stem"]]
    return [str(int(frame_id)) for frame_id in batch["frame_id"]]


def frame_paths(out_dir: Path, sequence: str, frame_key: str) -> Tuple[Path, Path]:
    """Where this frame's prediction and picture go."""
    return (out_dir / sequence / "data" / f"{frame_key}.pt",
            out_dir / sequence / "viz" / f"{frame_key}.png")


def already_done(out_dir: Path, sequence: str, frame_key: str,
                 want_prediction: bool, want_viz: bool) -> bool:
    """True if everything this run would write for the frame is already there."""
    prediction, viz = frame_paths(out_dir, sequence, frame_key)
    return ((prediction.exists() or not want_prediction)
            and (viz.exists() or not want_viz))


def write_frame(out_dir: Path, sequence: str, frame_key: str,
                payload: Dict[str, torch.Tensor], disparity: np.ndarray,
                save_prediction: bool, save_viz: bool) -> None:
    prediction_path, viz_path = frame_paths(out_dir, sequence, frame_key)

    if save_prediction:
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, prediction_path.as_posix())

    if save_viz:
        viz_path.parent.mkdir(parents=True, exist_ok=True)
        finite = np.isfinite(disparity)
        vmin, vmax = robust_limits(disparity, mask=finite,
                                   lo=VIZ_PERCENTILES[0], hi=VIZ_PERCENTILES[1])
        iio.imwrite(viz_path.as_posix(),
                    colorize(disparity, vmin=vmin, vmax=vmax, invalid=~finite))


def merge_rows(out_dir: Path) -> List[Dict]:
    """Every rank's rows, deduplicated on (sequence, frame_key) and sorted.

    Duplicates come from DistributedSampler's padding and from resumed runs.
    """
    seen: "OrderedDict[Tuple[str, str], Dict]" = OrderedDict()
    for shard in sorted(out_dir.glob("metrics_rank*.jsonl")):
        for row in read_rows(shard):
            seen[(row["sequence"], row["frame_key"])] = row
    return [seen[key] for key in sorted(seen)]


def aggregate(out_dir: Path, args: Config, label: str) -> Optional[Dict[str, float]]:
    """Merge the per-rank rows into metrics.csv + summary.json. Returns the overall
    row, or None when nothing had ground truth (normal for the qualitative suite)."""
    rows = merge_rows(out_dir)
    if not rows:
        print(f"[{label}] predictions written to {out_dir}; no ground truth in "
              f"{', '.join(args.evaluation.datasets)}, so no metrics.")
        return None

    names = metric_names(args.evaluation.bad_thresholds)
    write_csv(out_dir / "metrics.csv", rows,
              ["sequence", "frame_key", "n_valid", *names])

    accumulator = MetricAccumulator(args.evaluation.bad_thresholds)
    accumulator.extend(rows)

    per_sequence = accumulator.per_sequence()
    overall = accumulator.overall()
    summary = {
        "checkpoint": label,
        "model": args.model,
        "datasets": list(args.evaluation.datasets),
        "split": args.evaluation.split,
        "iters": (args.evaluation.iters or configured_iters(args)) if uses_iters(args) else None,
        "architectural_max_disparity": ARCHITECTURAL_MAX_DISPARITY.get(args.model),
        "aggregation": "pixel-weighted; rmse pooled over squared error",
        "bad_thresholds": list(args.evaluation.bad_thresholds),
        "use_foreground_mask": args.evaluation.use_foreground_mask,
        "max_gt_disparity": args.evaluation.max_gt_disparity,
        "checked_gt_rectification": args.evaluation.check_gt_rectification,
        "overall": overall,
        "overall_frame_averaged": accumulator.frame_averaged(),
        "per_sequence": per_sequence,
    }
    (out_dir / "summary.json").write_text(json.dumps(json_safe(summary), indent=2) + "\n")

    if overall["n_scored"] == 0:
        print(f"[{label}] {overall['n_frames']} frames, none with ground truth "
              f"-- predictions written, no metrics.")
        return None

    print(f"\n[{label}] {overall['n_scored']}/{overall['n_frames']} frames scored")
    print(format_table(per_sequence, names))
    print(format_table({"ALL": overall}, names, label="overall"))
    return overall


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

@torch.no_grad()
def evaluate_checkpoint(args: Config, checkpoint: Checkpoint, out_dir: Path,
                        loader: DataLoader, n_total: int,
                        device, world_size: int, rank: int) -> None:
    """Run one checkpoint over the whole suite, writing predictions and metrics."""
    from utils.train_utils import load_model

    args.io.restore_checkpoints = checkpoint.path
    module = load_model(args, device, world_size, rank)
    module.eval()  # load_model leaves the model in train mode

    iters = args.evaluation.iters or configured_iters(args)
    save_prediction = args.evaluation.save_predictions
    save_viz = args.evaluation.save_visualizations
    skip_existing = args.evaluation.skip_existing
    thresholds = list(args.evaluation.bad_thresholds)

    rows_path = out_dir / f"metrics_rank{rank}.jsonl"

    # A frame counts as done when its artifacts are on disk and, if it has ground
    # truth, this rank's shard already has its metric row.
    scored_already = set()
    if skip_existing:
        scored_already = {(row["sequence"], row["frame_key"])
                          for row in read_rows(rows_path)}
    elif rows_path.exists():
        rows_path.unlink()

    if rank == 0:
        where = ", ".join(args.evaluation.datasets)
        if "tbnms" in args.evaluation.datasets:
            where += f" (SUDS split: {args.evaluation.split})"
        print(f"\n===== {checkpoint.label} =====")
        print(f"  checkpoint: {checkpoint.path}")
        print(f"  output:     {out_dir}")
        print(f"  frames:     {n_total} on {where}")
        print(f"  iters:      {iters if uses_iters(args) else 'n/a (feed-forward)'}")
        ceiling = ARCHITECTURAL_MAX_DISPARITY.get(args.model)
        if ceiling is not None:
            print(f"  note:       this architecture cannot represent disparity above "
                  f"{ceiling:g} px; anything beyond that is clipped by construction.")

    n_skipped = 0
    for batch in tqdm(loader, disable=rank != 0, dynamic_ncols=True,
                      desc=checkpoint.stage):
        sequences = list(batch["sequence"])
        keys = frame_keys(batch)
        has_gt = "disparity_valid" in batch

        if skip_existing and all(
                already_done(out_dir, s, k, save_prediction, save_viz)
                and (not has_gt or (s, k) in scored_already)
                for s, k in zip(sequences, keys)):
            n_skipped += len(keys)
            continue

        left = batch["left_uw"].to(device)
        right = batch["right_uw"].to(device)

        with autocast("cuda", enabled=args.optimization.mixed_precision):
            disparity = forward_eval(module, args, left, right, iters)
        disparity = disparity.float().cpu()

        gt = batch["disparity"].cpu() if has_gt else None
        gt_valid = batch["disparity_valid"].cpu() if has_gt else None

        rows = []
        for i, (sequence, key) in enumerate(zip(sequences, keys)):
            prediction = disparity[i]
            if prediction.ndim == 3:            # [1, H, W] -> [H, W]
                prediction = prediction.squeeze(0)

            if has_gt:
                metrics = disparity_metrics(
                    prediction, gt[i], gt_valid[i],
                    bad_thresholds=thresholds,
                    max_gt_disparity=args.evaluation.max_gt_disparity)
                rows.append({"sequence": sequence, "frame_key": key, **metrics})

            if save_prediction or save_viz:
                payload = {
                    "sequence": sequence,
                    "frame_key": key,
                    "disp_pred": prediction,
                    "intrinsics": batch["intrinsics"][i].cpu(),
                    "right_intrinsics": batch["right_intrinsics"][i].cpu(),
                    "baseline": batch["baseline"][i].cpu(),
                }
                if has_gt:
                    payload["disparity"] = gt[i]
                    payload["disparity_valid"] = gt_valid[i]
                write_frame(out_dir, sequence, key, payload, prediction.numpy(),
                            save_prediction, save_viz)

        append_rows(rows_path, rows)

    if n_skipped and rank == 0:
        print(f"  skipped {n_skipped} frames that were already complete")

    del module
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


#: Filenames that identify a run only by its directory.
GENERIC_CHECKPOINT_STEMS = frozenset(
    {"best", "latest", "final", "model_best", "checkpoint", "model"})

#: `label=<path-or-stage>` in evaluation.checkpoints.
_LABELLED = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.\-]*)=(.+)$")


def stage_name_for_path(path: str) -> str:
    """A directory-safe name: the parent directory when the stem is generic
    (`.../run_xyz/best.pth` -> `run_xyz`), else the stem."""
    p = Path(path)
    if p.stem.lower() in GENERIC_CHECKPOINT_STEMS and p.parent.name:
        return p.parent.name
    return p.stem


def resolve_checkpoints(args: Config) -> List[Checkpoint]:
    """Turn ``evaluation.checkpoints`` into resolved checkpoints, failing early.

    Resolved against ``args.checkpoints`` rather than by model name: defom_stereo
    and defom_stereo_vits both set ``model: defom_stereo`` but have different
    weights.
    """
    resolved: List[Checkpoint] = []
    for raw in args.evaluation.checkpoints:
        spec = str(raw).strip()

        # `label=<path or stage>` names the run explicitly.
        label = None
        labelled = _LABELLED.match(spec)
        if labelled and not os.path.isfile(spec):
            label, spec = labelled.group(1), labelled.group(2).strip()

        if os.path.isfile(spec):
            # A path that exists as given is taken as-is; resolve_checkpoint would
            # join a relative path onto model_weights.
            path = os.path.abspath(spec)
            stage = label or stage_name_for_path(path)
        elif spec.endswith((".pth", ".ckpt")):
            path = resolve_checkpoint(spec, args.io.model_weights_dir)
            stage = label or stage_name_for_path(path)
        else:
            if spec not in args.checkpoints:
                raise FileNotFoundError(
                    f"evaluation.checkpoints names '{spec}', which this model does not "
                    f"declare. It has: {', '.join(sorted(args.checkpoints))}. Add the "
                    f"stage to config/train/models/<model>/<model>.yaml, or pass a path "
                    f"to a .pth directly.")
            if args.checkpoints[spec] is None:
                have = sorted(k for k, v in args.checkpoints.items() if v)
                raise FileNotFoundError(
                    f"'{spec}' is null for this model: that stage was never run, or its "
                    f"weights are not part of the release. "
                    + (f"Stages with weights: {', '.join(have)}." if have else
                       "This model declares no stage with weights -- download them and "
                       "register the path in config/train/models/<model>/<model>.yaml, "
                       "or pass a path to a .pth directly."))
            path = resolve_checkpoint(args.checkpoints[spec], args.io.model_weights_dir)
            stage = label or spec

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"'{spec}' resolves to {path}, which does not exist. Download the "
                f"released weights and point `model_weights` in config/dataset_paths.yaml at "
                f"them (`python utils/dataset_paths.py` shows what resolves).")
        resolved.append(Checkpoint(model=args.model, stage=stage, path=path, exists=True))

    # Two checkpoints resolving to the same stage would share an output directory
    # and, with skip_existing, report the first one's numbers under both names.
    by_stage: Dict[str, List[str]] = {}
    for checkpoint in resolved:
        by_stage.setdefault(checkpoint.stage, []).append(checkpoint.path)
    collisions = {s: p for s, p in by_stage.items() if len(set(p)) > 1}
    if collisions:
        detail = "\n".join(f"  {s}:\n" + "\n".join(f"    {q}" for q in sorted(set(p)))
                           for s, p in sorted(collisions.items()))
        raise ValueError(
            f"two or more checkpoints in this sweep resolve to the same name, so they "
            f"would share an output directory and report the same numbers:\n{detail}\n"
            f"Name them explicitly, e.g. "
            f"evaluation.checkpoints=[haze_only=/path/a/best.pth,no_warp=/path/b/best.pth]")

    duplicates = [s for s, p in by_stage.items() if len(p) > 1 and len(set(p)) == 1]
    if duplicates:
        print(f"warning: evaluation.checkpoints lists {', '.join(sorted(duplicates))} "
              f"more than once; each is evaluated only once.")
        seen, unique = set(), []
        for checkpoint in resolved:
            if checkpoint.stage not in seen:
                seen.add(checkpoint.stage)
                unique.append(checkpoint)
        resolved = unique

    return resolved


def main() -> int:
    args: Config = get_args(config_root=EVAL_CONFIG_DIR)

    seed = args.optimization.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        setup_ddp(rank, world_size)
        assert dist.is_initialized(), f"[rank {rank}]: DDP not initialised"
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    run_dir = Path(args.evaluation.output_dir
                   or os.path.join(args.io.output_dir_base, args.name))

    checkpoints = resolve_checkpoints(args)
    if rank == 0:
        print(f"evaluating {args.model} on {list(args.evaluation.datasets)} "
              f"({len(checkpoints)} checkpoint(s)) -> {run_dir}")

    loader, n_total = build_eval_loader(args, device, world_size, rank)

    comparison = []
    for checkpoint in checkpoints:
        out_dir = run_dir / checkpoint.stage
        out_dir.mkdir(parents=True, exist_ok=True)
        evaluate_checkpoint(args, checkpoint, out_dir, loader, n_total,
                            device, world_size, rank)

        if world_size > 1:
            dist.barrier()  # all rows on disk before merging
        if rank == 0:
            overall = aggregate(out_dir, args, checkpoint.label)
            if overall is not None:
                comparison.append({"checkpoint": checkpoint.label,
                                   "stage": checkpoint.stage, **overall})

    if rank == 0 and comparison:
        names = metric_names(args.evaluation.bad_thresholds)
        write_csv(run_dir / "comparison.csv", comparison,
                  ["checkpoint", "stage", "n_frames", "n_scored", "n_valid", *names])
        print("\n===== comparison =====")
        print(format_table({row["checkpoint"]: row for row in comparison}, names,
                           label="checkpoint"))
        print(f"\nwrote {run_dir / 'comparison.csv'}")

    cleanup_ddp()
    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    raise SystemExit(main())
