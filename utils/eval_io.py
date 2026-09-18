"""Shared I/O for evaluation tools: per-frame metric rows as JSON lines, merged
into CSV, and JSON-safe summaries. Used by evaluate.py and score_predictions.py."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List


def append_rows(path: Path, rows: List[Dict]) -> None:
    """Append per-frame metric rows as JSON lines (one file per rank, resumable)."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_rows(path: Path) -> List[Dict]:
    """Rows from one shard. Missing file is empty; a torn final line is dropped."""
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"warning: ignoring an incomplete final row in {path}")
    return rows


def json_safe(value):
    """Recursively turn NaN/inf into null, so summary.json is strict JSON."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_csv(path: Path, rows: List[Dict], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
