"""Evaluate a 3D reconstruction against a ground-truth point cloud.

    python3 evaluate_reconstruction.py config_path=reconstruction/mesh \\
        reconstruction.mesh=recon.obj reconstruction.gt_cloud=fused.ply \\
        reconstruction.output_dir=results/

Two modes (``reconstruction.mode``):

    mesh    one reconstructed mesh, sampled uniformly and compared to the GT
            cloud: Chamfer L1/L2, accuracy/completeness (mean/median/p95),
            F-score at ``fscore_radius``, Hausdorff. Writes
            ``<mesh_stem>_reconstruction_metrics.{json,md,txt}``.
    frames  per-frame depth maps (``*.npy``) projected to point clouds and
            compared to the GT cloud: per-frame accuracy, completeness and
            Chamfer, averaged over frames. Writes ``depth_eval_metrics.{json,md}``.
            Point ``reconstruction.depth_dir`` at one directory, or
            ``reconstruction.results_root`` at a ``<method>/<scene>/depths``
            tree to sweep everything in it.

Needs open3d and kaolin (mesh mode also point_cloud_utils, frames mode also
kornia); each is imported on first use so the rest of the repo does not
depend on them.
"""

from __future__ import annotations

import importlib
import json
import logging
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.arguments import EVAL_CONFIG_DIR, Config, get_args  # noqa: E402
from utils.eval_io import json_safe  # noqa: E402

log = logging.getLogger("evaluate_reconstruction")


def _require(module: str, pip_hint: Optional[str] = None):
    try:
        return importlib.import_module(module)
    except ImportError as err:
        raise ImportError(
            f"evaluate_reconstruction.py needs '{module}': "
            f"pip install {pip_hint or module}") from err


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def load_gt_cloud(path: Path) -> torch.Tensor:
    o3d = _require("open3d")
    cloud = o3d.io.read_point_cloud(str(path))
    if len(cloud.points) == 0:
        raise ValueError(f"ground-truth point cloud {path} is empty")
    points = torch.from_numpy(np.asarray(cloud.points, dtype=np.float32))
    return points.unsqueeze(0).to(_device())


# --------------------------------------------------------------------------
# mesh mode
# --------------------------------------------------------------------------

@dataclass
class MeshMetrics:
    accuracy_mean: float
    accuracy_median: float
    accuracy_p95: float
    completeness_mean: float
    completeness_median: float
    completeness_p95: float
    chamferl1: float
    chamferl2: float
    fscore_radius: float
    hausdorff: float


def evaluate_mesh(mesh_path: Path, gt_path: Path, n_samples: int,
                  fscore_radius: float, cross_check: bool,
                  seed: int) -> MeshMetrics:
    o3d = _require("open3d")
    pcu = _require("point_cloud_utils")
    kaolin_pc = _require("kaolin.metrics.pointcloud", pip_hint="kaolin")

    seed_everything(seed)
    o3d.utility.random.seed(seed)

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.triangles) == 0:
        raise ValueError(f"mesh {mesh_path} has no triangles")
    mesh.compute_vertex_normals()
    gt_pts = load_gt_cloud(gt_path)

    pred_pts = torch.from_numpy(np.asarray(
        mesh.sample_points_uniformly(n_samples).points, dtype=np.float32
    )).unsqueeze(0).to(_device())

    chamferl1 = float(kaolin_pc.chamfer_distance(pred_pts, gt_pts,
                                                 squared=False).item())
    chamferl2 = float(kaolin_pc.chamfer_distance(pred_pts, gt_pts,
                                                 squared=True).item())
    if cross_check:
        pcu_cd = pcu.chamfer_distance(pred_pts.squeeze(0).cpu().numpy(),
                                      gt_pts.squeeze(0).cpu().numpy())
        if not np.isclose(pcu_cd, chamferl1, rtol=1e-5, atol=1e-5):
            log.warning("point_cloud_utils and kaolin disagree on the Chamfer "
                        "distance: %.6f vs %.6f", pcu_cd, chamferl1)

    accuracy = kaolin_pc.sided_distance(pred_pts, gt_pts)[0].cpu().numpy()
    completeness = kaolin_pc.sided_distance(gt_pts, pred_pts)[0].cpu().numpy()
    fscore = float(kaolin_pc.f_score(gt_pts, pred_pts,
                                     radius=fscore_radius).item())
    hausdorff = float(pcu.hausdorff_distance(pred_pts.squeeze(0).cpu().numpy(),
                                             gt_pts.squeeze(0).cpu().numpy()))

    return MeshMetrics(
        accuracy_mean=float(accuracy.mean()),
        accuracy_median=float(np.median(accuracy)),
        accuracy_p95=float(np.percentile(accuracy, 95)),
        completeness_mean=float(completeness.mean()),
        completeness_median=float(np.median(completeness)),
        completeness_p95=float(np.percentile(completeness, 95)),
        chamferl1=chamferl1,
        chamferl2=chamferl2,
        fscore_radius=fscore,
        hausdorff=hausdorff,
    )


def write_mesh_report(out_dir: Path, mesh_path: Path, fscore_radius: float,
                      metrics: MeshMetrics) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = mesh_path.stem
    (out_dir / f"{stem}_reconstruction_metrics.json").write_text(
        json.dumps(json_safe(asdict(metrics)), indent=2) + "\n")

    rows = [
        ("Accuracy (mean) [m]", metrics.accuracy_mean),
        ("Accuracy (median) [m]", metrics.accuracy_median),
        ("Accuracy 95-pct [m]", metrics.accuracy_p95),
        ("Completeness (mean) [m]", metrics.completeness_mean),
        ("Completeness (median) [m]", metrics.completeness_median),
        ("Completeness 95-pct [m]", metrics.completeness_p95),
        ("Chamfer Distance L1 [m]", metrics.chamferl1),
        ("Chamfer Distance L2 [m]", metrics.chamferl2),
        (f"F-score @ {fscore_radius * 100:.0f} cm", metrics.fscore_radius),
        ("Hausdorff Distance [m]", metrics.hausdorff),
    ]
    with (out_dir / f"{stem}_reconstruction_metrics.md").open("w") as f:
        f.write(f"# Metrics for {stem}\n\n")
        f.write(f"Evaluated {datetime.now().isoformat(timespec='seconds')}\n\n")
        f.write("| Metric | Value |\n|---|---|\n")
        for name, value in rows:
            f.write(f"| {name} | {value:.4f} |\n")
    (out_dir / f"{stem}_reconstruction_metrics.txt").write_text(
        "\t".join(f"{value:.4f}" for _, value in rows))
    print(f"wrote {out_dir / f'{stem}_reconstruction_metrics.json'} (+ .md, .txt)")


# --------------------------------------------------------------------------
# frames mode
# --------------------------------------------------------------------------

def load_intrinsics(path: Path) -> torch.Tensor:
    """3x3 K from a text file holding fx fy cx cy."""
    fx, fy, cx, cy = np.loadtxt(path).reshape(-1)[:4]
    return torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                        dtype=torch.float32)


def evaluate_depth_frames(depth_dir: Path, intrinsics: torch.Tensor,
                          gt_pts: torch.Tensor, max_points: int,
                          max_frames: Optional[int], seed: int) -> Dict:
    kornia_depth = _require("kornia.geometry.depth", pip_hint="kornia")
    kaolin_pc = _require("kaolin.metrics.pointcloud", pip_hint="kaolin")

    seed_everything(seed)
    device = _device()
    K = intrinsics.to(device).unsqueeze(0)

    depth_files = sorted(depth_dir.glob("*.npy"))
    if not depth_files:
        raise FileNotFoundError(f"no *.npy depth maps in {depth_dir}")
    if max_frames is not None and 0 <= max_frames < len(depth_files):
        depth_files = depth_files[:max_frames]

    per_frame = []
    n_empty = 0
    for depth_file in tqdm(depth_files, desc=depth_dir.parent.name,
                           dynamic_ncols=True):
        depth = np.load(depth_file)
        mask = depth > 0
        if not mask.any():
            n_empty += 1
            continue

        depth_torch = torch.from_numpy(depth).to(device)
        points = kornia_depth.depth_to_3d_v2(depth_torch.unsqueeze(0),
                                             K).squeeze(0)
        points = points[mask]
        if points.shape[0] == 0:
            n_empty += 1
            continue
        if points.shape[0] > max_points:
            indices = torch.randperm(points.shape[0])[:max_points]
            points = points[indices]

        pred = points.reshape(-1, 3).unsqueeze(0)
        accuracy = kaolin_pc.sided_distance(pred, gt_pts)[0]
        completeness = kaolin_pc.sided_distance(gt_pts, pred)[0]
        chamfer = accuracy.mean(dim=-1) + completeness.mean(dim=-1)
        per_frame.append({
            "frame": depth_file.name,
            "accuracy_mean": float(accuracy.mean().item()),
            "completeness_mean": float(completeness.mean().item()),
            "chamfer_distance": float(chamfer.item()),
        })

    if not per_frame:
        raise ValueError(f"{depth_dir}: every depth map was empty")
    return {
        "average_accuracy": float(np.mean([f["accuracy_mean"] for f in per_frame])),
        "average_completeness": float(np.mean([f["completeness_mean"]
                                               for f in per_frame])),
        "average_chamfer_distance": float(np.mean([f["chamfer_distance"]
                                                   for f in per_frame])),
        "n_frames": len(per_frame),
        "n_empty_frames": n_empty,
        "per_frame": per_frame,
    }


def write_frames_report(out_dir: Path, summary: Dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "depth_eval_metrics.json").write_text(
        json.dumps(json_safe(summary), indent=2) + "\n")
    with (out_dir / "depth_eval_metrics.md").open("w") as f:
        f.write("# Depth-based Point-cloud Accuracy\n\n")
        f.write(f"Evaluated {datetime.now().isoformat(timespec='seconds')}\n\n")
        f.write(f"Average accuracy (mean) [m]: **{summary['average_accuracy']:.4f}**\n\n")
        f.write(f"Average completeness (mean) [m]: "
                f"**{summary['average_completeness']:.4f}**\n\n")
        f.write(f"Average Chamfer distance [m]: "
                f"**{summary['average_chamfer_distance']:.4f}**\n\n")
        f.write("| Frame | Accuracy (mean) [m] |\n|---|---|\n")
        for frame in summary["per_frame"]:
            f.write(f"| {frame['frame']} | {frame['accuracy_mean']:.4f} |\n")
    print(f"wrote {out_dir / 'depth_eval_metrics.json'} (+ .md)")


def frames_targets(cfg) -> List[Tuple[Path, Path, Path]]:
    """(depth_dir, gt_cloud, out_dir) jobs for frames mode."""
    if cfg.depth_dir:
        if not cfg.gt_cloud:
            raise ValueError("reconstruction.depth_dir needs reconstruction.gt_cloud")
        out_dir = Path(cfg.output_dir) if cfg.output_dir else Path(cfg.depth_dir).parent
        return [(Path(cfg.depth_dir), Path(cfg.gt_cloud), out_dir)]

    if not (cfg.results_root and cfg.gt_root):
        raise ValueError("frames mode needs reconstruction.depth_dir + gt_cloud, "
                         "or reconstruction.results_root + gt_root")
    results_root, gt_root = Path(cfg.results_root), Path(cfg.gt_root)
    out_root = Path(cfg.output_dir) if cfg.output_dir else results_root

    jobs = []
    for method_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        for scene_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
            depth_dir = scene_dir / "depths"
            if not depth_dir.is_dir():
                print(f"warning: no depths in {scene_dir}, skipped")
                continue
            gt_path = gt_root / scene_dir.name / "COLMAP" / "dense" / "fused.ply"
            if not gt_path.is_file():
                print(f"warning: missing GT {gt_path}, skipped")
                continue
            jobs.append((depth_dir, gt_path,
                         out_root / method_dir.name / scene_dir.name))
    if not jobs:
        raise FileNotFoundError(
            f"{results_root} has no <method>/<scene>/depths directories with "
            f"ground truth under {gt_root}/<scene>/COLMAP/dense/fused.ply")
    return jobs


# --------------------------------------------------------------------------

def main() -> int:
    args: Config = get_args(config_root=EVAL_CONFIG_DIR)
    cfg = args.reconstruction
    logging.basicConfig(level=logging.INFO if cfg.verbose else logging.WARNING,
                        format="[%(levelname)s] %(message)s")

    if cfg.mode == "mesh":
        if not (cfg.mesh and cfg.gt_cloud):
            raise ValueError("mesh mode needs reconstruction.mesh and "
                             "reconstruction.gt_cloud")
        out_dir = Path(cfg.output_dir) if cfg.output_dir else Path(cfg.mesh).parent
        metrics = evaluate_mesh(Path(cfg.mesh), Path(cfg.gt_cloud),
                                cfg.n_samples, cfg.fscore_radius,
                                cfg.chamfer_cross_check, cfg.seed)
        write_mesh_report(out_dir, Path(cfg.mesh), cfg.fscore_radius, metrics)
        return 0

    if cfg.mode == "frames":
        if not cfg.intrinsics:
            raise ValueError("frames mode needs reconstruction.intrinsics "
                             "(text file: fx fy cx cy)")
        intrinsics = load_intrinsics(Path(cfg.intrinsics))
        for depth_dir, gt_path, out_dir in frames_targets(cfg):
            if cfg.skip_existing and (out_dir / "depth_eval_metrics.json").exists():
                print(f"{out_dir} already evaluated, skipping")
                continue
            gt_pts = load_gt_cloud(gt_path)
            summary = evaluate_depth_frames(depth_dir, intrinsics, gt_pts,
                                            cfg.max_points, cfg.max_frames,
                                            cfg.seed)
            write_frames_report(out_dir, summary)
        return 0

    raise ValueError(f"reconstruction.mode must be 'mesh' or 'frames', "
                     f"got {cfg.mode!r}")


if __name__ == "__main__":
    raise SystemExit(main())
