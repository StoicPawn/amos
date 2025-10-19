"""Risk management helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd


def apply_vol_target(weights: pd.Series, returns: pd.Series, target_vol: float) -> pd.Series:
    """Scale weights to hit a target annualised volatility."""
    realized_vol = returns.std() * np.sqrt(252)
    if realized_vol == 0 or np.isnan(realized_vol):
        return weights
    scale = target_vol / realized_vol
    return weights * scale


def enforce_gross_limits(weights: pd.Series, max_gross: float) -> pd.Series:
    gross = weights.abs().sum()
    if gross <= max_gross:
        return weights
    return weights * (max_gross / gross)
