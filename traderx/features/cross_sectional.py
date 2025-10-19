"""Cross-sectional feature calculations."""
from __future__ import annotations

import pandas as pd


def rank_pct(series: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank."""
    return series.rank(pct=True)


def demean_cross_section(df: pd.DataFrame) -> pd.DataFrame:
    """Demean each row in a cross-section."""
    return df.sub(df.mean(axis=1), axis=0)
