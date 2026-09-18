"""Turn a stereo pair and a checkpoint into disparity, depth, and pictures.

Both front ends (``demo/cli.py``, ``demo/web.py``) are thin shells over
:func:`predict_pair`, so there is exactly one implementation of the inference
path and they cannot drift apart.

Two things are worth knowing before reading further:

* **The bundled example images are raw.** ``demo/data/<scene>/{left,right}.png``
  are unrectified camera frames; disparity on an unrectified pair is
  meaningless. Pass a :class:`~utils.rectification.StereoRectification` (the
  scene loaders do this for you) and the pair is undistorted and rectified
  first, exactly the way training does it.
* **Everything comes back at the resolution you passed in.** ``scale`` only
  controls what the network sees; disparity is resampled and rescaled back to
  the input resolution, so the numbers are always in pixels of the images you
  provided.

Model construction is delegated to ``utils.train_utils.load_model``, the same
loader training uses, imported lazily so that listing scenes and checkpoints
does not pay for importing torch's whole training stack.
"""

from __future__ import annotations

import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import imageio.v3 as iio
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")  # the demo never has a display; web and CLI both write files
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# The submodules are not installed as packages, so put them on the path the way
# train.py and data/tbnms_dataset.py do. Append, never insert: the repo's own
# `utils` package must keep priority over models/DEFOM-Stereo/utils.
for _p in ("models", "models/DEFOM-Stereo", "models/FoundationStereo", "models/IGEV-plusplus"):
    _path = str(REPO_ROOT / _p)
    if _path not in sys.path:
        sys.path.append(_path)

from FoundationStereo.core.utils.utils import InputPadder  # noqa: E402

from utils.arguments import Config, list_ablations, list_models, model_config_file  # noqa: E402
from utils.dataset_paths import DatasetNotRegisteredError, resolve_checkpoint  # noqa: E402
from utils.rectification import StereoRectification, load_kalibr_calibration  # noqa: E402

#: Bundled example scenes, and the calibration of the rig that captured them.
DATA_DIR = REPO_ROOT / "demo" / "data"
DEMO_CALIB_FILE = DATA_DIR / "stereo_calib.yaml"

#: Where results go when the caller does not say. Gitignored by the `outputs/`
#: rule in .gitignore, which matches at any depth.
DEFAULT_OUTPUT_DIR = REPO_ROOT / "demo" / "outputs"

#: What ``--checkpoint`` defaults to: the model from the paper.
DEFAULT_CHECKPOINT = "defom_stereo/ours"

#: Architecture assumed when a bare ``.pth`` path is given.
DEFAULT_MODEL = "defom_stereo"

#: Fraction of the input resolution the network runs at. 1.0 (full resolution)
#: is both the most accurate and the sharpest on the release images; see
#: docs/demo.md for the measurement.
DEFAULT_SCALE = 1.0

#: Architectures the demo knows how to run, by their `model:` config value. The
#: config tree also describes evaluation-only baselines; those are evaluate.py's
#: business, and listing a checkpoint here that forward_disparity would refuse
#: only wastes the reader's time.
SUPPORTED_MODELS = ("defom_stereo", "foundation_stereo", "igev_plusplus")

#: Disparity below this many pixels is treated as unmeasurable rather than
#: very far away -- see disparity_to_depth().
MIN_DISPARITY_PX = 1.0


# --------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Checkpoint:
    """One entry of one model's ``checkpoints:`` registry."""

    model: str
    stage: str
    path: Optional[str]   # resolved absolute path, or None if unresolvable
    exists: bool

    @property
    def label(self) -> str:
        return f"{self.model}/{self.stage}"


@lru_cache(maxsize=None)
def model_config(model: str) -> Config:
    """The Config a training run of ``model`` would start from.

    Goes through the normal overlay system so the demo builds the architecture
    with the same settings training used.
    """
    config_file = model_config_file(model)
    if config_file is None:
        raise ValueError(f"no model '{model}'; available: {', '.join(list_models())}")
    return get_args_no_cli(config_file)


def get_args_no_cli(config_file: str) -> Config:
    """``get_args`` without letting OmegaConf near our argparse flags."""
    from utils.arguments import get_args
    return get_args(config_path=config_file, use_cli=False)


def _stage_order(stage: str) -> Tuple[int, str]:
    """Sort key: the paper model first, then ablations, then anything else."""
    if stage == "ours":
        return (0, "")
    ablations = list_ablations()
    if stage in ablations:
        return (1, f"{ablations.index(stage):03d}")
    return (2, stage)


def list_checkpoints(only_existing: bool = True) -> List[Checkpoint]:
    """Every checkpoint the demo can run, resolved on this machine.

    Registry entries are relative to the ``model_weights`` root from
    config/dataset_paths.yaml. Entries a model never ran are null and are skipped, as
    are entries whose file is not present unless ``only_existing`` is False.

    Models :func:`forward_disparity` cannot run are left out entirely -- the
    config tree also carries evaluation-only baselines, which belong in
    ``evaluate.py`` and not in a demo of our own model.
    """
    found: List[Checkpoint] = []
    for model in list_models():
        try:
            cfg = model_config(model)
        except Exception:
            continue  # a malformed config should not take the whole list down

        if cfg.model not in SUPPORTED_MODELS:
            continue

        for stage, path in sorted(cfg.checkpoints.items(), key=lambda kv: _stage_order(kv[0])):
            if path is None:
                continue
            try:
                resolved = resolve_checkpoint(path, cfg.io.model_weights_dir)
            except DatasetNotRegisteredError:
                resolved = None
            exists = resolved is not None and os.path.isfile(resolved)
            if exists or not only_existing:
                found.append(Checkpoint(model=model, stage=stage, path=resolved, exists=exists))
    return found


def find_checkpoint(spec: str, model: str = DEFAULT_MODEL) -> Checkpoint:
    """Resolve ``<model>/<stage>`` from the registry, or a path to a ``.pth``.

    Raises:
        FileNotFoundError: if the registry entry is unknown or the file is missing.
    """
    spec = str(spec).strip()

    if spec.endswith(".pth") or os.path.isfile(spec):
        path = resolve_checkpoint(spec)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"no checkpoint at {path}")
        return Checkpoint(model=model, stage=Path(path).stem, path=path, exists=True)

    model_name, _, stage = spec.partition("/")
    if not stage:
        model_name, stage = model, spec

    cfg = model_config(model_name)
    if stage not in cfg.checkpoints:
        raise FileNotFoundError(
            f"model '{model_name}' declares no checkpoint '{stage}'; it has: "
            f"{', '.join(sorted(cfg.checkpoints))}")
    if cfg.checkpoints[stage] is None:
        raise FileNotFoundError(
            f"'{spec}' is null: {model_name} never ran that stage. "
            f"Run `--list-checkpoints` to see what is available.")

    path = resolve_checkpoint(cfg.checkpoints[stage], cfg.io.model_weights_dir)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"'{spec}' resolves to {path}, which does not exist. Download the released "
            f"weights and point `model_weights` in config/dataset_paths.yaml at them.")
    return Checkpoint(model=model_name, stage=stage, path=path, exists=True)


# --------------------------------------------------------------------------
# Bundled scenes
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Scene:
    """One bundled example: a raw stereo pair plus the rig that took it."""

    name: str
    left: Path
    right: Path
    calib: Path


def list_scenes() -> List[Scene]:
    """Example scenes shipped in ``demo/data``."""
    scenes = []
    for entry in sorted(DATA_DIR.iterdir()) if DATA_DIR.is_dir() else []:
        if not entry.is_dir():
            continue
        left = next(iter(sorted(entry.glob("left.*"))), None)
        right = next(iter(sorted(entry.glob("right.*"))), None)
        if left and right:
            scenes.append(Scene(name=entry.name, left=left, right=right, calib=DEMO_CALIB_FILE))
    return scenes


def find_scene(name: str) -> Scene:
    for scene in list_scenes():
        if scene.name == name:
            return scene
    known = ", ".join(s.name for s in list_scenes()) or "none found"
    # ValueError, not KeyError: the CLI prints this straight to the user, and
    # KeyError's repr would wrap the whole sentence in quotes.
    raise ValueError(f"no scene '{name}' in {DATA_DIR}. Available: {known}")


def read_image(path: Union[str, Path]) -> np.ndarray:
    """Read an image as RGB uint8, dropping alpha and widening greyscale."""
    img = iio.imread(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------

def default_iters(cfg: Config) -> int:
    """Refinement iterations this model's config asks for at test time.

    Every model has its own `valid_iters`, all defaulting to 32, which is what the
    checkpoints were evaluated with. `train_iters` is a training-time number and is
    deliberately not the fallback here -- reading it at test time silently ran
    FoundationStereo for 22 iterations instead of 32.
    """
    if cfg.model == "defom_stereo":
        return cfg.defom_stereo.valid_iters
    if cfg.model == "igev_plusplus":
        return cfg.igev_pp.valid_iters
    if cfg.model == "foundation_stereo":
        return cfg.foundation_stereo.valid_iters
    return cfg.train_iters


def forward_disparity(module: torch.nn.Module, cfg: Config,
                      left: torch.Tensor, right: torch.Tensor,
                      iters: Optional[int] = None) -> torch.Tensor:
    """Run one stereo forward pass and return disparity as ``[B, 1, H, W]``.

    The models disagree about their call signature and about what they return, and
    this is the one place that knows the differences. ``evaluate.py`` shares it, so
    the demo and the reported numbers cannot come from different forward passes.

    Args:
        module: a model built by ``utils.train_utils.load_model``, in eval mode.
        cfg: the Config it was built from; ``cfg.model`` selects the call.
        left, right: padded ``[B, 3, H, W]`` RGB in 0..255. Every model in this
            repo normalizes internally.
        iters: refinement iterations, or None for the model's configured value.
    """
    if iters is None:
        iters = default_iters(cfg)

    if cfg.model == "defom_stereo":
        out = module(left, right, iters=iters,
                     scale_iters=cfg.defom_stereo.scale_iters, test_mode=True)
    elif cfg.model in ("foundation_stereo", "igev_plusplus"):
        out = module(left.contiguous(), right.contiguous(), iters=iters, test_mode=True)
    else:
        raise ValueError(
            f"forward_disparity does not know how to call model '{cfg.model}'. "
            f"Supported: {', '.join(SUPPORTED_MODELS)}. Evaluation-only baselines are "
            f"handled by evaluate.py:forward_eval.")

    # DEFOM-Stereo returns a tuple in test mode; upstream FoundationStereo and
    # IGEV-plusplus return a bare tensor. Take the first element either way.
    disparity = out[0] if isinstance(out, (tuple, list)) else out
    return disparity.float()


class StereoPredictor:
    """A loaded stereo network, ready to run on pairs of numpy images."""

    def __init__(self, cfg: Config, module: torch.nn.Module,
                 device: torch.device, checkpoint: Checkpoint):
        self.cfg = cfg
        self.module = module
        self.device = device
        self.checkpoint = checkpoint

    @classmethod
    def load(cls, checkpoint: Checkpoint, device: str = "cuda") -> "StereoPredictor":
        # A copy: model_config is cached, and load_model writes resolved paths
        # back into the config it is given.
        cfg = deepcopy(model_config(checkpoint.model))
        cfg.io.restore_checkpoints = checkpoint.path

        torch_device = resolve_device(device)

        # Imported here, not at module scope: this pulls in the training stack
        # (wandb, the dataset loaders, flash-attn via FoundationStereo), which
        # listing scenes and checkpoints has no business paying for.
        try:
            from utils.train_utils import load_model
        except ImportError as err:
            raise ImportError(
                f"could not import the model loader ({err}). The demo runs in the "
                f"project's docker image; see docs/docker.md.") from err

        module = load_model(cfg, torch_device, world_size=1, rank=0)
        # load_model leaves the model in train mode. DEFOM's context encoder uses
        # batch norm, so predictions are wrong (not just noisy) without this.
        module.eval()
        return cls(cfg=cfg, module=module, device=torch_device, checkpoint=checkpoint)

    @property
    def default_iters(self) -> int:
        """Refinement iterations this model's config asks for at test time."""
        return default_iters(self.cfg)

    def _forward(self, left: torch.Tensor, right: torch.Tensor, iters: int) -> torch.Tensor:
        return forward_disparity(self.module, self.cfg, left, right, iters)

    def disparity(self, left: np.ndarray, right: np.ndarray,
                  scale: float = 1.0, iters: Optional[int] = None) -> np.ndarray:
        """Disparity in pixels of the *input* images, as float32 ``[H, W]``.

        Args:
            left, right: rectified RGB uint8 images of identical shape.
            scale: run the network at this fraction of the input resolution.
            iters: refinement iterations; the model's configured value if None.
        """
        if left.shape != right.shape:
            raise ValueError(
                f"left is {left.shape[1]}x{left.shape[0]} but right is "
                f"{right.shape[1]}x{right.shape[0]}; a stereo pair must match.")

        full_h, full_w = left.shape[:2]
        if scale != 1.0:
            size = (max(32, int(round(full_w * scale))), max(32, int(round(full_h * scale))))
            left = cv2.resize(left, size, interpolation=cv2.INTER_AREA)
            right = cv2.resize(right, size, interpolation=cv2.INTER_AREA)

        # RGB in 0..255: every model in this repo normalizes internally.
        def to_tensor(img):
            return (torch.as_tensor(img).to(self.device).float()[None]
                    .permute(0, 3, 1, 2).contiguous())

        left_t, right_t = to_tensor(left), to_tensor(right)
        padder = InputPadder(left_t.shape, divis_by=32, force_square=False)
        left_t, right_t = padder.pad(left_t, right_t)

        with torch.inference_mode():
            disp = self._forward(left_t, right_t, iters or self.default_iters)

        disp = padder.unpad(disp.float())[0, 0].cpu().numpy().astype(np.float32)

        if disp.shape != (full_h, full_w):
            # Disparity is a length in pixels, so it scales with the image.
            disp = cv2.resize(disp, (full_w, full_h), interpolation=cv2.INTER_LINEAR)
            disp /= scale
        return disp


def resolve_device(device: str) -> torch.device:
    """Turn a device string into a torch.device, failing loudly if it is not usable."""
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Start the container with `docker/run.sh` (it passes "
            "`--gpus all`), or pass `--device cpu` -- but note a ViT-L model on a 1080p "
            "pair takes minutes on CPU.")
    return torch.device(device)


@lru_cache(maxsize=2)
def get_predictor(model: str, path: str, device: str = "cuda") -> StereoPredictor:
    """Load a predictor, reusing it across calls.

    Keyed on the resolved checkpoint path so the web app keeps the model warm
    between requests and only reloads when the user picks a different one.
    """
    checkpoint = Checkpoint(model=model, stage=Path(path).stem, path=path, exists=True)
    return StereoPredictor.load(checkpoint, device=device)


# --------------------------------------------------------------------------
# Geometry and pictures
# --------------------------------------------------------------------------

def disparity_to_depth(disparity: np.ndarray, focal_px: float, baseline_m: float,
                       min_disparity: float = MIN_DISPARITY_PX) -> np.ndarray:
    """Metric depth in metres. Pixels below ``min_disparity`` come back as 0.

    Underwater scenes are mostly water: the model reports near-zero disparity
    for the open water column, which divides out to depths of hundreds of
    kilometres and would swamp both the colormap and the percentiles. Sub-pixel
    disparity is not a measurement, so it is marked invalid instead.
    """
    depth = np.zeros_like(disparity, dtype=np.float32)
    valid = disparity >= max(min_disparity, np.finfo(np.float32).tiny)
    depth[valid] = float(focal_px) * float(baseline_m) / disparity[valid]
    return depth


def robust_limits(values: np.ndarray, mask: Optional[np.ndarray] = None,
                  lo: float = 2.0, hi: float = 98.0) -> Tuple[float, float]:
    """Percentile display range, so one bad pixel cannot flatten the colormap."""
    data = values[mask] if mask is not None else values.reshape(-1)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return 0.0, 1.0
    vmin, vmax = float(np.percentile(data, lo)), float(np.percentile(data, hi))
    return (vmin, vmax) if vmax > vmin else (vmin, vmin + 1.0)


def _colormap(name: str):
    try:
        return matplotlib.colormaps[name]        # matplotlib >= 3.6
    except (AttributeError, KeyError):
        return plt.get_cmap(name)


def colorize(values: np.ndarray, cmap: str = "magma",
             vmin: Optional[float] = None, vmax: Optional[float] = None,
             invalid: Optional[np.ndarray] = None) -> np.ndarray:
    """Map a float array to an RGB uint8 image; invalid pixels come out black."""
    if vmin is None or vmax is None:
        auto_min, auto_max = robust_limits(values, None if invalid is None else ~invalid)
        vmin = auto_min if vmin is None else vmin
        vmax = auto_max if vmax is None else vmax

    normed = np.clip((values - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0)
    rgb = (_colormap(cmap)(normed)[:, :, :3] * 255).astype(np.uint8)
    if invalid is not None:
        rgb[invalid] = 0
    return rgb


def make_panel(result: "StereoResult") -> np.ndarray:
    """Left image, disparity and (when known) depth, side by side with colorbars."""
    has_depth = result.depth is not None
    fig, axes = plt.subplots(1, 3 if has_depth else 2,
                             figsize=(16 if has_depth else 11, 3.6), dpi=140)

    axes[0].imshow(result.left)
    axes[0].set_title("Left (rectified)" if result.rectified else "Left")

    disp_min, disp_max = robust_limits(result.disparity, result.disparity > 0)
    im = axes[1].imshow(result.disparity, cmap="magma", vmin=disp_min, vmax=disp_max)
    axes[1].set_title("Disparity (px)")
    fig.colorbar(im, ax=axes[1], fraction=0.035, pad=0.02)

    if has_depth:
        valid = result.depth > 0
        depth_min, depth_max = robust_limits(result.depth, valid)
        shown = np.where(valid, result.depth, np.nan)
        # Water (no depth) reads as black here, the same as it does in depth.png.
        cmap = _colormap("turbo").with_extremes(bad="black")
        im = axes[2].imshow(shown, cmap=cmap, vmin=depth_min, vmax=depth_max)
        axes[2].set_title("Depth (m)")
        fig.colorbar(im, ax=axes[2], fraction=0.035, pad=0.02)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{result.checkpoint}   |   {result.width}x{result.height}   |   "
                 f"{result.iters} iters   |   {result.runtime_s:.1f} s", fontsize=9)
    fig.tight_layout()

    fig.canvas.draw()
    panel = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return panel


# --------------------------------------------------------------------------
# The one call both front ends make
# --------------------------------------------------------------------------

@dataclass
class StereoResult:
    """What the demo produces for one stereo pair."""

    disparity: np.ndarray              # float32 [H, W], pixels, left frame
    depth: Optional[np.ndarray]        # float32 [H, W], metres; None without calibration
    left: np.ndarray                   # uint8 RGB, what the network actually saw
    right: np.ndarray
    rectified: bool
    focal_px: Optional[float]
    baseline_m: Optional[float]
    checkpoint: str
    scale: float
    iters: int
    runtime_s: float
    min_disparity: float = MIN_DISPARITY_PX

    @property
    def height(self) -> int:
        return self.disparity.shape[0]

    @property
    def width(self) -> int:
        return self.disparity.shape[1]

    def disparity_image(self) -> np.ndarray:
        return colorize(self.disparity, "magma", invalid=self.disparity <= 0)

    def depth_image(self, max_depth: Optional[float] = None) -> Optional[np.ndarray]:
        if self.depth is None:
            return None
        invalid = self.depth <= 0
        vmin, vmax = robust_limits(self.depth, ~invalid)
        return colorize(self.depth, "turbo", vmin=vmin,
                        vmax=min(vmax, max_depth) if max_depth else vmax, invalid=invalid)

    def summary(self) -> str:
        """One line per quantity, with percentiles rather than min/max."""
        valid = self.disparity > 0
        lines = []
        if valid.any():
            p2, p50, p98 = np.percentile(self.disparity[valid], [2, 50, 98])
            lines.append(f"disparity   p2 {p2:.1f} px   median {p50:.1f} px   p98 {p98:.1f} px")
        if self.depth is not None:
            dvalid = self.depth > 0
            if dvalid.any():
                p2, p50, p98 = np.percentile(self.depth[dvalid], [2, 50, 98])
                beyond = 100.0 * (~dvalid).mean()
                lines.append(f"depth       p2 {p2:.2f} m   median {p50:.2f} m   p98 {p98:.2f} m"
                             f"   (f={self.focal_px:.1f} px, b={self.baseline_m:.4f} m)")
        else:
            lines.append("depth       not computed: no calibration, focal length or baseline given")
        return "\n".join(lines)


def predict_pair(left: np.ndarray,
                 right: np.ndarray,
                 checkpoint: str = DEFAULT_CHECKPOINT,
                 model: str = DEFAULT_MODEL,
                 rectification: Optional[StereoRectification] = None,
                 focal_px: Optional[float] = None,
                 baseline_m: Optional[float] = None,
                 scale: float = DEFAULT_SCALE,
                 iters: Optional[int] = None,
                 min_disparity: float = MIN_DISPARITY_PX,
                 device: str = "cuda") -> StereoResult:
    """Run a released model on one stereo pair.

    Args:
        left, right: RGB uint8 images of the same size.
        checkpoint: ``<model>/<stage>`` from the registry, or a path to a .pth.
        model: architecture to assume when ``checkpoint`` is a bare path.
        rectification: applied to the pair first, and used for the depth
            conversion unless focal/baseline are given explicitly. Pass None
            when the images are already rectified.
        focal_px, baseline_m: overrides for the depth conversion. Depth is
            skipped when neither these nor a rectification are available.
        scale: fraction of the input resolution the network runs at.
        iters: refinement iterations; the model's configured value if None.
        min_disparity: disparity below this many pixels gets no depth.
    """
    ckpt = find_checkpoint(checkpoint, model=model)

    if rectification is not None:
        left, right = rectification.rectify(left, right)
        if focal_px is None:
            focal_px = rectification.focal_px
        if baseline_m is None:
            baseline_m = rectification.baseline_m

    predictor = get_predictor(ckpt.model, ckpt.path, device)

    start = time.perf_counter()
    disparity = predictor.disparity(left, right, scale=scale, iters=iters)
    if torch.cuda.is_available() and predictor.device.type == "cuda":
        torch.cuda.synchronize()
    runtime = time.perf_counter() - start

    depth = None
    if focal_px and baseline_m:
        depth = disparity_to_depth(disparity, focal_px, baseline_m, min_disparity)

    return StereoResult(
        disparity=disparity,
        depth=depth,
        left=left,
        right=right,
        rectified=rectification is not None,
        focal_px=focal_px,
        baseline_m=baseline_m,
        checkpoint=ckpt.label,
        scale=scale,
        iters=iters or predictor.default_iters,
        runtime_s=runtime,
        min_disparity=min_disparity,
    )


def write_outputs(result: StereoResult, out_dir: Union[str, Path],
                  max_depth: Optional[float] = None) -> Dict[str, Path]:
    """Write every product to ``out_dir`` and return what was written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    def save_png(key: str, image: np.ndarray):
        path = out_dir / f"{key}.png"
        iio.imwrite(path, image)
        written[key] = path

    save_png("panel", make_panel(result))
    save_png("disparity", result.disparity_image())
    save_png("left_rectified" if result.rectified else "left", result.left)

    depth_image = result.depth_image(max_depth)
    if depth_image is not None:
        save_png("depth", depth_image)

    np.save(out_dir / "disparity.npy", result.disparity)
    written["disparity_npy"] = out_dir / "disparity.npy"

    arrays = {"disparity_px": result.disparity}
    if result.depth is not None:
        np.save(out_dir / "depth.npy", result.depth)
        written["depth_npy"] = out_dir / "depth.npy"
        arrays["depth_m"] = result.depth
        arrays["focal_px"] = np.array(result.focal_px)
        arrays["baseline_m"] = np.array(result.baseline_m)

    np.savez_compressed(out_dir / "arrays.npz", **arrays)
    written["arrays"] = out_dir / "arrays.npz"
    return written


