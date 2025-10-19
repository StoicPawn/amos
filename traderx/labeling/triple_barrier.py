"""Triple barrier labeling implemented without pandas."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import pandas as pd


@dataclass
class TripleBarrierResult:
    label: int
    t_exit: int
    ret: float


def apply_triple_barrier(
    close: Sequence[float],
    pt_mult: float,
    sl_mult: float,
    max_h: int,
    volatility: Iterable[float] | None = None,
) -> List[TripleBarrierResult]:
    """Apply the triple barrier method to generate labels."""
    prices = list(close)
    if volatility is None:
        volatility = [0.01 for _ in prices]
    vol = list(volatility)
    results: List[TripleBarrierResult] = []
    n = len(prices)
    for idx, price in enumerate(prices):
        if idx == n - 1:
            results.append(TripleBarrierResult(0, idx, 0.0))
            continue
        pt = price * (1 + pt_mult * vol[idx])
        sl = price * (1 - sl_mult * vol[idx])
        horizon = min(n - 1, idx + max_h)
        label = 0
        exit_idx = horizon
        for forward in range(idx + 1, horizon + 1):
            next_price = prices[forward]
            if next_price >= pt:
                label = 1
                exit_idx = forward
                break
            if next_price <= sl:
                label = -1
                exit_idx = forward
                break
        ret = prices[exit_idx] / price - 1 if price else 0.0
        results.append(TripleBarrierResult(label, exit_idx, ret))
    return results


def results_to_frame(
    results: Sequence[TripleBarrierResult],
    index: Sequence[object] | None = None,
) -> pd.DataFrame:
    """Convert a sequence of :class:`TripleBarrierResult` to a DataFrame."""

    if index is None:
        idx = range(len(results))
    else:
        idx = list(index)
        if len(idx) != len(results):
            raise ValueError("Index length does not match number of results")
    data = {
        "label": [res.label for res in results],
        "t_exit": [res.t_exit for res in results],
        "ret": [res.ret for res in results],
    }
    return pd.DataFrame(data, index=idx)
