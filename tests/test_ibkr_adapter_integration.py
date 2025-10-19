"""Integration smoke tests for the IBKR adapter."""
from __future__ import annotations

import os

import pandas as pd
import pytest

pytest.importorskip("ib_insync")

from traderx.ingest.ibkr_adapter import HistoricalDataRequest, IBKRAdapter

if not os.getenv("IBKR_INTEGRATION_ENABLED"):
    pytest.skip("IBKR integration tests disabled", allow_module_level=True)


def test_ibkr_adapter_fetch_smoke() -> None:
    adapter = IBKRAdapter()
    requests = [
        HistoricalDataRequest(symbol="AAPL", duration="5 D", bar_size="5 mins"),
        HistoricalDataRequest(symbol="MSFT", duration="5 D", bar_size="5 mins"),
    ]

    frame = pd.DataFrame()
    try:
        frame = adapter.fetch(requests)
    finally:
        adapter.disconnect()

    assert not frame.empty
    assert {"open", "high", "low", "close", "volume"}.issubset(frame.columns)
    assert {"AAPL", "MSFT"} <= set(frame["symbol"].unique())
