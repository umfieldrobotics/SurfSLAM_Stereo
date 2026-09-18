"""Where the datasets live on this machine.

Copy ``config/dataset_paths.example.yaml`` to ``config/dataset_paths.yaml`` and
fill in the paths you have.

Every dataset class in ``data/`` calls :func:`get_dataset_root` instead of
hardcoding an absolute path. Resolution order for a key (first hit wins):

1. an explicit ``override`` argument (``args.io.<key>_dir`` when set),
2. the environment variable ``SURF_<KEY>_DIR`` (handy for containers and cluster jobs),
3. ``config/dataset_paths.yaml`` (or ``$SURF_DATASET_PATHS``).

Inside Docker, host paths from ``config/dataset_paths.yaml`` do not exist. ``docker/run.sh``
bind-mounts every registered dataset at ``/data/<key>``, so a path read from the yaml file
is re-pointed there when we are running in a container and that mount is present. Explicit
overrides (1) and ``SURF_<KEY>_DIR`` (2) are container-aware by assumption and never remapped.

Released checkpoints use the same machinery, under the ``model_weights`` key.

Run ``python utils/dataset_paths.py`` to print what currently resolves.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Default location of the paths file. Override with ``$SURF_DATASET_PATHS``.
DEFAULT_PATHS_FILE = REPO_ROOT / "config" / "dataset_paths.yaml"

EXAMPLE_PATHS_FILE = REPO_ROOT / "config" / "dataset_paths.example.yaml"

ENV_PATHS_FILE = "SURF_DATASET_PATHS"

#: Set by ``docker/run.sh``; forces the in-container branch even if the heuristics below miss.
ENV_IN_DOCKER = "SURF_IN_DOCKER"

#: Where ``docker/run.sh`` mounts the datasets. Must match CONTAINER_DATA_ROOT in that script.
ENV_DOCKER_DATA_ROOT = "SURF_DOCKER_DATA_ROOT"
DEFAULT_DOCKER_DATA_ROOT = "/data"


@dataclass(frozen=True)
class DatasetSpec:
    """One dataset key, and how to tell whether a path really points at it."""

    key: str
    description: str
    #: Relative path that must exist under the root for it to look valid.
    sentinel: Optional[str] = None
    #: Glob (relative to the root) used when a single sentinel file isn't meaningful.
    sentinel_glob: Optional[str] = None
    #: How to run without this dataset, shown when no path is configured for it.
    skip_hint: Optional[str] = None


DATASET_SPECS: Dict[str, DatasetSpec] = {
    spec.key: spec
    for spec in [
        DatasetSpec(
            key="suds_stereo",
            description="SUDS real-world stereo release (formerly TBNMS)",
            sentinel="data/calibration/stereo_calib.yaml",
            skip_hint="remove 'tbnms' from optimization.train_datasets / "
                      "evaluation.datasets and set logging.tbnms_samples.sequences=[]",
        ),
        DatasetSpec(
            key="uwsim",
            description="UWSim simulated stereo release (formerly OceanSim)",
            sentinel="metadata_files.json",
            skip_hint="remove 'oceansim' from optimization.train_datasets",
        ),
        DatasetSpec(
            key="svin2",
            description="SVIn2 underwater stereo sequences (third party)",
            sentinel_glob="*/calib.yaml",
            skip_hint="remove 'svin2' from optimization.train_datasets / "
                      "evaluation.datasets",
        ),
        DatasetSpec(
            key="lizard_island",
            description="Lizard Island COLMAP stereo reconstruction (third party)",
            sentinel="images/left",
            skip_hint="remove 'lizard_island' from optimization.train_datasets / "
                      "evaluation.datasets",
        ),
        DatasetSpec(
            key="tartanair",
            description="TartanAir stereo dataset (third party)",
            sentinel_glob="*/*/*/image_left",
            skip_hint="remove 'tartanair' from optimization.train_datasets "
                      "(e.g. optimization.train_datasets=[oceansim,flyingthings])",
        ),
        DatasetSpec(
            key="flyingthings",
            description="SceneFlow: FlyingThings3D / Driving / Monkaa (third party)",
            sentinel_glob="*/frames_finalpass",
            skip_hint="remove 'flyingthings' from optimization.train_datasets",
        ),
        DatasetSpec(
            key="model_weights",
            description="Released model checkpoints (DATA_RELEASE/weights)",
            sentinel_glob="*",
            skip_hint="use absolute paths for the checkpoint entries the run "
                      "loads (e.g. io.restore_checkpoints)",
        ),
    ]
}


class DatasetNotRegisteredError(RuntimeError):
    """Raised when a dataset root cannot be resolved from any source."""


def _env_var_for(name: str) -> str:
    return f"SURF_{name.upper()}_DIR"


def paths_file() -> Path:
    """Path to ``config/dataset_paths.yaml``, honouring ``$SURF_DATASET_PATHS``."""
    override = os.environ.get(ENV_PATHS_FILE)
    return Path(override).expanduser() if override else DEFAULT_PATHS_FILE


def in_docker() -> bool:
    """True if this process looks like it is running inside a container."""
    flag = os.environ.get(ENV_IN_DOCKER, "")
    if flag:
        return flag.lower() not in ("0", "false", "no")
    if Path("/.dockerenv").exists():
        return True
    try:
        with open("/proc/1/cgroup", "r") as f:
            return any(marker in f.read() for marker in ("docker", "containerd", "kubepods"))
    except OSError:
        return False


def docker_data_root() -> Path:
    """Directory under which ``docker/run.sh`` mounts the registered datasets."""
    return Path(os.environ.get(ENV_DOCKER_DATA_ROOT) or DEFAULT_DOCKER_DATA_ROOT)


def docker_path_for(name: str) -> Optional[Path]:
    """The in-container mount for ``name``, or ``None`` if it isn't there.

    Returns a path only when we are in a container *and* the mount actually exists, so a
    container started without ``docker/run.sh`` falls back to whatever the yaml file says.
    """
    if not in_docker():
        return None
    candidate = docker_data_root() / name
    return candidate if candidate.is_dir() else None


def _check_known(name: str) -> DatasetSpec:
    if name not in DATASET_SPECS:
        known = ", ".join(sorted(DATASET_SPECS))
        raise KeyError(f"Unknown dataset key {name!r}. Known keys: {known}")
    return DATASET_SPECS[name]


def load_paths() -> Dict[str, str]:
    """Read ``config/dataset_paths.yaml``. Missing file or unfilled (null) keys are fine."""
    path = paths_file()
    if not path.exists():
        return {}
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping of dataset name -> path")

    unknown = sorted(set(data) - set(DATASET_SPECS))
    if unknown:
        print(
            f"warning: {path} has unrecognized keys: {', '.join(unknown)}. "
            f"Known keys: {', '.join(sorted(DATASET_SPECS))}",
            file=sys.stderr,
        )
    return {str(k): str(v) for k, v in data.items() if v is not None}


def looks_valid(name: str, path: Path) -> bool:
    """True if ``path`` plausibly contains the dataset identified by ``name``."""
    spec = _check_known(name)
    if not path.is_dir():
        return False
    if spec.sentinel is not None:
        return (path / spec.sentinel).exists()
    if spec.sentinel_glob is not None:
        return next(path.glob(spec.sentinel_glob), None) is not None
    return True


_WARNED: set = set()


def get_dataset_root(name: str, override: Optional[str] = None) -> Path:
    """Resolve the root directory for ``name``.

    Args:
        name: a key from :data:`DATASET_SPECS`.
        override: explicit path that wins over the environment and the paths file.
            ``None`` or an empty string means "not set".

    Raises:
        DatasetNotRegisteredError: if no source provides a path.
    """
    spec = _check_known(name)

    if override:
        raw, source = str(override), "override"
    elif os.environ.get(_env_var_for(name)):
        raw, source = os.environ[_env_var_for(name)], f"${_env_var_for(name)}"
    else:
        raw, source = load_paths().get(name), str(paths_file())
        mounted = docker_path_for(name)
        if mounted is not None:
            # The yaml path is a host path; use the bind mount docker/run.sh made from it.
            raw, source = str(mounted), f"docker mount for {paths_file()}"

    if raw is None:
        msg = (
            f"No path set for '{name}' ({spec.description}).\n"
            f"  To use it, add it to {paths_file()}:  {name}: /path/to/{name}\n"
            f"  (copy {EXAMPLE_PATHS_FILE.relative_to(REPO_ROOT)} if that file doesn't exist yet)\n"
        )
        if spec.skip_hint:
            msg += (
                f"  To run without it, {spec.skip_hint},\n"
                f"  in the run's yaml or as a command-line override.\n"
            )
        msg += "  See docs/data.md for details."
        raise DatasetNotRegisteredError(msg)

    path = Path(raw).expanduser().resolve()

    if not path.is_dir():
        raise DatasetNotRegisteredError(
            f"'{name}' points at {path} (from {source}), which is not a directory.\n"
            f"  Fix the '{name}' entry in {paths_file()}."
        )

    if name not in _WARNED and not looks_valid(name, path):
        _WARNED.add(name)
        expected = spec.sentinel or spec.sentinel_glob
        print(
            f"warning: {path} (from {source}) does not look like '{name}' "
            f"- expected to find {expected!r} inside it. Continuing anyway.",
            file=sys.stderr,
        )

    return path


MODEL_WEIGHTS_KEY = "model_weights"


def resolve_checkpoint(path: Optional[str], override: Optional[str] = None) -> Optional[str]:
    """Resolve a config checkpoint entry against the model_weights root.

    None and absolute paths pass through. Call it on the checkpoint a run loads, not on
    the whole registry, so the root is only needed when a released checkpoint is used.
    """
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return str(get_dataset_root(MODEL_WEIGHTS_KEY, override) / path)


def _main() -> int:
    """Print every key and what it currently resolves to."""
    entries = load_paths()
    exists = paths_file().exists()
    print(f"paths file: {paths_file()}" + ("" if exists else "  (does not exist yet)"))
    if not exists:
        print(f"            copy {EXAMPLE_PATHS_FILE.relative_to(REPO_ROOT)} to it and fill it in")
    if in_docker():
        print(f"in docker: yes, yaml paths re-pointed under {docker_data_root()}/<key> when mounted")

    width = max(len(k) for k in DATASET_SPECS)
    for name in sorted(DATASET_SPECS):
        env_val = os.environ.get(_env_var_for(name))
        mounted = None if env_val else docker_path_for(name)
        raw = env_val or (str(mounted) if mounted else entries.get(name))
        if not raw:
            status = "unset"
        elif not Path(raw).expanduser().is_dir():
            status = "missing"
        elif not looks_valid(name, Path(raw).expanduser().resolve()):
            status = "suspect"
        else:
            status = "ok"
        if env_val:
            suffix = f"  [{_env_var_for(name)}]"
        elif mounted:
            suffix = "  [docker mount]"
        else:
            suffix = ""
        print(f"  {name:<{width}}  {status:<8}  {raw or '-'}{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
