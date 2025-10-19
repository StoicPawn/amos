"""Execution simulators."""
from __future__ import annotations

from typing import Dict, List


def simulate_vwap(prices: Dict[str, List[float]], participation: float = 0.1) -> List[float]:
    """Simple VWAP execution price estimate."""
    high = prices["high"]
    low = prices["low"]
    close = prices["close"]
    fills = []
    for h, l, c in zip(high, low, close):
        vwap = (h + l + c) / 3
        impact = participation * (h - l) / 2
        fills.append(vwap + impact)
    return fills
