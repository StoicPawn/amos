"""Time alignment utilities to avoid look-ahead bias."""
from __future__ import annotations

import pandas as pd


def lag_features(df: pd.DataFrame, columns: list[str], periods: int = 1) -> pd.DataFrame:
    """Lag selected columns by a number of periods."""
    shifted = df.copy()
    for col in columns:
        shifted[col] = shifted[col].shift(periods)
    return shifted


def align_on_close(prices: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe aligned on close timestamp using forward fill."""
    aligned = prices.sort_index().ffill()
    return aligned
