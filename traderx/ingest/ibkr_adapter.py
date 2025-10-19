"""IBKR market data adapter placeholder."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import pandas as pd


@dataclass
class HistoricalDataRequest:
    symbol: str
    bar_size: str = "1 day"
    duration: str = "1 M"


class IBKRAdapter:
    """Fetch historical data using ib_insync (placeholder)."""

    def fetch(self, requests: Iterable[HistoricalDataRequest]) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for req in requests:
            idx = pd.date_range("2023-01-01", periods=5, freq="D")
            frames.append(
                pd.DataFrame(
                    {
                        "symbol": req.symbol,
                        "close": 100.0,
                        "high": 101.0,
                        "low": 99.0,
                        "volume": 1_000_000,
                    },
                    index=idx,
                )
            )
        return pd.concat(frames)
