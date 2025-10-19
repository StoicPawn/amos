"""Backtest metric utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series) -> float:
    if returns.std() == 0:
        return 0.0
    return returns.mean() / returns.std() * np.sqrt(252)


def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1
    return drawdown.min()
