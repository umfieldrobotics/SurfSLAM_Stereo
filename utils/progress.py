"""Rank-aware progress reporting for dataset scans.

Under DDP every rank builds the datasets, so unguarded prints and tqdm bars
appear once per rank and interleave. These helpers render on the main rank
only (RANK unset or 0, which covers single-process runs and torchrun rank 0).

Kept free of heavy imports so any dataset module can use it.
"""

import os

from tqdm import tqdm


def is_main_rank() -> bool:
    return os.environ.get("RANK", "0") in ("", "0")


def main_rank_print(*args, **kwargs):
    if is_main_rank():
        print(*args, **kwargs, flush=True)


def progress(iterable, desc: str, **kwargs):
    """tqdm over ``iterable`` that only renders on the main rank."""
    return tqdm(iterable, desc=desc, disable=not is_main_rank(), **kwargs)
