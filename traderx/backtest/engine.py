"""Simple backtest engine using lists."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


def _pct_change(series: List[float]) -> List[float]:
    out = [0.0]
    for i in range(1, len(series)):
        prev = series[i - 1]
        out.append((series[i] - prev) / prev if prev else 0.0)
    return out


@dataclass
class BacktestResult:
    pnl: List[float]
    turnover: List[float]
    equity: List[float]


class BacktestEngine:
    """Vectorised backtest using simple price lists."""

    def __init__(self, prices: Dict[str, List[float]], costs_bps: float = 2.0) -> None:
        self.prices = prices
        self.costs_bps = costs_bps

    def run(self, weights: Dict[str, List[float]]) -> BacktestResult:
        num_points = len(next(iter(self.prices.values())))
        pnl = [0.0] * num_points
        turnover = [0.0] * num_points
        equity = [1.0]
        for t in range(1, num_points):
            day_pnl = 0.0
            day_turnover = 0.0
            for symbol, price_series in self.prices.items():
                returns = _pct_change(price_series)
                prev_weight = weights[symbol][t - 1]
                day_pnl += prev_weight * returns[t]
                day_turnover += abs(weights[symbol][t] - weights[symbol][t - 1])
            cost = day_turnover * (self.costs_bps / 1e4)
            pnl[t] = day_pnl - cost
            turnover[t] = day_turnover
            equity.append(equity[-1] * (1 + pnl[t]))
        while len(equity) < num_points:
            equity.append(equity[-1])
        return BacktestResult(pnl=pnl, turnover=turnover, equity=equity)
