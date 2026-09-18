"""On-disk cache for dataset file indices.

Scanning a dataset means globbing and stat-ing up to hundreds of thousands of
files, which takes minutes on network storage. The scan result depends only on
the dataset root's contents, and these datasets are static, so each index is
built once and reused:

- in memory for the lifetime of the process (the train and test dataset
  objects share one scan), and
- on disk under ``cache/dataset_index/`` in the repo, across runs.

Delete a cache file, or set ``SURF_RESCAN_DATASETS=1``, to force a fresh scan
(needed only if the dataset's files changed on disk). Cache files carry a
version number; bump a dataset's ``_INDEX_VERSION`` whenever its scan logic
changes so stale caches are ignored automatically.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from utils.progress import main_rank_print

REPO_ROOT = Path(__file__).resolve().parent.parent

ENV_CACHE_DIR = "SURF_INDEX_CACHE_DIR"
ENV_RESCAN = "SURF_RESCAN_DATASETS"

_MEMO: Dict[Tuple[str, str], List[dict]] = {}


def cache_dir() -> Path:
    override = os.environ.get(ENV_CACHE_DIR)
    return Path(override).expanduser() if override else REPO_ROOT / "cache" / "dataset_index"


def cache_file(key: str, root: Path) -> Path:
    digest = hashlib.sha1(str(root).encode()).hexdigest()[:12]
    return cache_dir() / f"{key}_{digest}.json"


def _rescan_forced() -> bool:
    return os.environ.get(ENV_RESCAN, "").lower() not in ("", "0", "false", "no")


def cached_index(key: str, root, version: int,
                 scan: Callable[[], List[dict]]) -> List[dict]:
    """The sample index for dataset ``key`` at ``root``, scanning only when needed.

    ``scan`` must return the complete, split-independent index as a list of
    JSON-serializable dicts. Callers filter it per split and must copy entries
    before mutating them -- the returned list is shared.
    """
    root = Path(root)
    memo_key = (key, str(root))
    if not _rescan_forced() and memo_key in _MEMO:
        main_rank_print(f"[{key}] Reusing the index scanned earlier in this run.")
        return _MEMO[memo_key]

    path = cache_file(key, root)
    if not _rescan_forced() and path.exists():
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if data.get("version") == version and data.get("root") == str(root):
                entries = data["entries"]
                main_rank_print(f"[{key}] Loaded cached index of {root} "
                                f"({len(entries)} entries) from {path}.")
                _MEMO[memo_key] = entries
                return entries
            main_rank_print(f"[{key}] Index cache {path} is stale "
                            "(scan version or root changed); rescanning.")
        except (OSError, ValueError, KeyError) as e:
            main_rank_print(f"[{key}] Ignoring unreadable index cache {path}: {e}")

    main_rank_print(
        f"[{key}] Indexing {root}. This is slow on network storage, but a "
        f"one-time cost: results are cached at {path} for later runs "
        f"(delete it or set {ENV_RESCAN}=1 to rescan)."
    )
    entries = scan()
    _MEMO[memo_key] = entries

    # An empty index means a misconfigured root; don't immortalize it.
    if not entries:
        return entries

    path.parent.mkdir(parents=True, exist_ok=True)
    # Concurrent DDP ranks may scan and save at the same time; write through a
    # unique temp file and atomically replace so readers never see a partial file.
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump({"version": version, "root": str(root), "entries": entries}, f)
        os.replace(tmp, path)
    except OSError as e:
        main_rank_print(f"[{key}] Could not write index cache {path}: {e}")
        try:
            os.remove(tmp)
        except OSError:
            pass
    return entries
