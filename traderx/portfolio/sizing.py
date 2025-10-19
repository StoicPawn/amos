"""Position sizing utilities."""
from __future__ import annotations

import pandas as pd


def weights_to_quantity(weights: pd.Series, prices: pd.Series, nav: float) -> pd.Series:
    """Convert weights into share quantities."""
    dollar_exposure = weights * nav
    quantity = (dollar_exposure / prices).round().astype(int)
    return quantity
