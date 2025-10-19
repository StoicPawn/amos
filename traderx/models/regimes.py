"""Regime detection utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd


def detect_vol_regime(vol_series: pd.Series, threshold: float) -> pd.Series:
    """Return regime labels based on volatility threshold."""
    return (vol_series > threshold).astype(int)


def rolling_zscore(series: pd.Series, window: int = 60) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std().replace(0, np.nan)
    return (series - mean) / std
