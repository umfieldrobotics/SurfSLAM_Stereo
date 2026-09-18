"""Command-line demo: one stereo pair in, disparity and depth out.

    python -m demo --list-scenes
    python -m demo --scene monohansett_hull
    python -m demo --left L.png --right R.png --calib my_kalibr.yaml

Bundled scenes are raw camera frames and are rectified automatically. Your own
images are assumed to be rectified already unless you pass ``--calib``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from demo import core


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m demo",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See docs/demo.md for the full walkthrough.")

    source = parser.add_argument_group("input")
    source.add_argument("--scene", metavar="NAME",
                        help="bundled example from demo/data (see --list-scenes)")
    source.add_argument("--left", metavar="PATH", help="left image of a stereo pair")
    source.add_argument("--right", metavar="PATH", help="right image of a stereo pair")

    model = parser.add_argument_group("model")
    model.add_argument("--checkpoint", default=core.DEFAULT_CHECKPOINT, metavar="SPEC",
                       help="'<model>/<stage>' from the registry, or a path to a .pth "
                            f"(default: {core.DEFAULT_CHECKPOINT})")
    model.add_argument("--model", default=core.DEFAULT_MODEL, metavar="NAME",
                       help="architecture to assume when --checkpoint is a bare path "
                            f"(default: {core.DEFAULT_MODEL})")
    model.add_argument("--iters", type=int, metavar="N",
                       help="refinement iterations (default: whatever the model config says)")
    model.add_argument("--scale", type=float, default=core.DEFAULT_SCALE, metavar="S",
                       help="run the network at this fraction of the input resolution "
                            f"(default: {core.DEFAULT_SCALE})")
    model.add_argument("--device", default="cuda", help="torch device (default: cuda)")

    geometry = parser.add_argument_group("geometry")
    geometry.add_argument("--calib", metavar="PATH",
                          help="Kalibr stereo yaml; rectifies the pair and supplies "
                               "the focal length and baseline for metric depth")
    geometry.add_argument("--no-rectify", action="store_true",
                          help="use a bundled scene's images as they are, unrectified "
                               "(a demonstration of why rectification matters)")
    geometry.add_argument("--focal-px", type=float, metavar="F",
                          help="rectified focal length in pixels, for metric depth "
                               "on an already-rectified pair")
    geometry.add_argument("--baseline-m", type=float, metavar="B",
                          help="stereo baseline in metres, for metric depth")
    geometry.add_argument("--max-depth", type=float, metavar="M",
                          help="clip the depth colormap at this many metres")
    geometry.add_argument("--min-disparity", type=float, default=core.MIN_DISPARITY_PX,
                          metavar="PX",
                          help="disparity below this many pixels is open water, not a "
                               f"distance, and gets no depth (default: {core.MIN_DISPARITY_PX:g})")

    output = parser.add_argument_group("output")
    output.add_argument("--out", metavar="DIR",
                        help="where to write results (default: demo/outputs/<name>)")

    listing = parser.add_argument_group("listings")
    listing.add_argument("--list-scenes", action="store_true",
                         help="print the bundled scenes and exit")
    listing.add_argument("--list-checkpoints", action="store_true",
                         help="print the checkpoints available on this machine and exit")
    return parser


def print_scenes() -> int:
    scenes = core.list_scenes()
    if not scenes:
        print(f"no scenes found in {core.DATA_DIR}", file=sys.stderr)
        return 1
    for scene in scenes:
        print(f"  {scene.name}")
    return 0


def print_checkpoints() -> int:
    """Every registry entry, resolved or not -- the weights analogue of
    `python utils/dataset_paths.py`."""
    checkpoints = core.list_checkpoints(only_existing=False)
    if not checkpoints:
        print("no checkpoints declared by any model config", file=sys.stderr)
        return 1

    width = max(len(c.label) for c in checkpoints)
    for ckpt in checkpoints:
        status = "ok" if ckpt.exists else "missing"
        print(f"  {ckpt.label:<{width}}  {status:<8}  {ckpt.path or '-'}")

    if not any(c.exists for c in checkpoints):
        print("\nNone of them are on disk. Download the released weights and set "
              "`model_weights` in config/dataset_paths.yaml.", file=sys.stderr)
        return 1
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_scenes:
        return print_scenes()
    if args.list_checkpoints:
        return print_checkpoints()

    if args.scene and (args.left or args.right):
        parser.error("--scene and --left/--right are alternatives; pass one or the other")
    if not args.scene and not (args.left and args.right):
        parser.error("give either --scene NAME or both --left and --right "
                     "(--list-scenes shows what is bundled)")

    # Bundled scenes are raw frames, so they come with the rig that shot them.
    # Your own images are assumed rectified unless you say otherwise.
    if args.scene:
        scene = core.find_scene(args.scene)
        left_path, right_path = scene.left, scene.right
        calib_path = None if args.no_rectify else (args.calib or scene.calib)
        name = args.scene
    else:
        left_path, right_path = Path(args.left), Path(args.right)
        calib_path = None if args.no_rectify else args.calib
        name = left_path.stem

    left = core.read_image(left_path)
    right = core.read_image(right_path)
    rectification = core.load_kalibr_calibration(calib_path) if calib_path else None

    checkpoint = core.find_checkpoint(args.checkpoint, model=args.model)
    print(f"checkpoint  {checkpoint.label}  ({checkpoint.path})")
    print(f"input       {left_path.parent if args.scene else left_path}  "
          f"{left.shape[1]}x{left.shape[0]}, "
          + (f"rectified with {calib_path}" if rectification else "assumed rectified"))

    result = core.predict_pair(
        left, right,
        checkpoint=args.checkpoint,
        model=args.model,
        rectification=rectification,
        focal_px=args.focal_px,
        baseline_m=args.baseline_m,
        scale=args.scale,
        iters=args.iters,
        min_disparity=args.min_disparity,
        device=args.device,
    )

    scaled = "" if result.scale == 1.0 else f" (network ran at {result.scale:g}x)"
    print(f"inference   {result.width}x{result.height}, {result.iters} iters, "
          f"{result.runtime_s:.1f} s on {args.device}{scaled}")
    print(result.summary())

    out_dir = Path(args.out) if args.out else core.DEFAULT_OUTPUT_DIR / name
    written = core.write_outputs(result, out_dir, max_depth=args.max_depth)
    print(f"wrote       {out_dir}/{{{', '.join(sorted(p.name for p in written.values()))}}}")
    return 0


def run() -> int:
    """Entry point that turns expected failures into one-line errors."""
    try:
        return main()
    except (FileNotFoundError, KeyError, ValueError, RuntimeError, ImportError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(run())
