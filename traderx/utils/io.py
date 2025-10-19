"""IO helpers for safe file operations."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class AtomicWriter:
    """Context manager for atomic file writes using temporary files."""

    def __init__(self, path: str | Path, suffix: str = ".tmp") -> None:
        self.path = Path(path)
        self.tmp_path = self.path.with_suffix(self.path.suffix + suffix)

    def __enter__(self):
        self.tmp_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.tmp_path.open("w", encoding="utf-8")
        return self.handle

    def __exit__(self, exc_type, exc, tb) -> None:
        self.handle.close()
        if exc is None:
            self.tmp_path.replace(self.path)
        else:
            self.tmp_path.unlink(missing_ok=True)


def atomic_to_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Write a DataFrame to CSV atomically."""
    writer = AtomicWriter(path)
    with writer as fh:
        df.to_csv(fh, index=False)


def append_jsonl(record: dict[str, Any], path: str | Path) -> None:
    """Append a JSON record to a JSONL file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
