"""Portfolio reconciliation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable


@dataclass
class Position:
    symbol: str
    quantity: int


def reconcile_targets(
    current_positions: Dict[str, int],
    target_weights: Dict[str, float] | Iterable[tuple[str, float]],
    prices: Dict[str, float],
    nav: float,
) -> Dict[str, int]:
    """Return delta quantities required to reach target weights."""
    if isinstance(target_weights, dict):
        target_dict = target_weights
    else:
        target_dict = dict(target_weights)
    deltas: Dict[str, int] = {}
    for symbol, target_weight in target_dict.items():
        price = prices[symbol]
        target_qty = round(target_weight * nav / price)
        current_qty = current_positions.get(symbol, 0)
        deltas[symbol] = int(target_qty - current_qty)
    return deltas
