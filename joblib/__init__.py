"""Tiny subset of :mod:`joblib` used for persistence in the tests."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def dump(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    with path.open("wb") as fh:
        pickle.dump(obj, fh)
    return path


def load(path: str | Path) -> Any:
    path = Path(path)
    with path.open("rb") as fh:
        return pickle.load(fh)


__all__ = ["dump", "load"]
