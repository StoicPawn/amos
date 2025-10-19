"""Market data helpers using IBKR client."""
from __future__ import annotations

import pandas as pd

from traderx_ibkr.ibkr_client import IBKRClient


def download_history(client: IBKRClient, symbol: str) -> pd.DataFrame:
    """Return placeholder historical data."""
    if not client.is_connected():
        raise RuntimeError("Client not connected")
    idx = pd.date_range("2023-01-01", periods=5, freq="D")
    return pd.DataFrame({"close": 100.0}, index=idx)
