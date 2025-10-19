"""Caching wrapper for market data vendors."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd


class VendorCache:
    """Simple file-based cache."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def load_or_fetch(self, key: str, fetcher: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        path = self.root / f"{key}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        df = fetcher()
        df.to_parquet(path)
        return df
