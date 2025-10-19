from __future__ import annotations

import numpy as np
import pandas as pd

from traderx.pipeline import TradingSystem


def _make_price_panel() -> dict[str, pd.DataFrame]:
    idx = pd.date_range("2023-01-01", periods=90, freq="B")
    grid = np.linspace(0, 6 * np.pi, len(idx))
    panel: dict[str, pd.DataFrame] = {}
    for i, symbol in enumerate(["AAPL", "MSFT"]):
        base = 100 + i * 5 + np.sin(grid + i) * 2 + np.cos(grid / 3 + i)
        drift = np.linspace(0, 3 + i, len(idx))
        close = base + drift
        high = close * (1 + 0.005 + 0.002 * np.sin(grid))
        low = close * (1 - 0.005 - 0.002 * np.cos(grid))
        panel[symbol] = pd.DataFrame({"close": close, "high": high, "low": low}, index=idx)
    return panel


def test_trading_system_generates_weights_and_equity():
    prices = _make_price_panel()
    system = TradingSystem(
        prices,
        risk_cfg={"max_symbol_weight": 0.2, "max_gross": 1.0, "target_vol": 0.15},
        costs_bps=1.5,
        barrier_cfg={"pt_mult": 0.8, "sl_mult": 1.2, "max_h": 7, "vol_lookback": 5},
    )
    result = system.run()

    assert set(result.weights.columns) == {"AAPL", "MSFT"}
    assert len(result.weights) == len(next(iter(prices.values())))
    assert len(result.backtest.equity) == len(result.weights)

    # Risk constraints should be enforced
    assert (result.weights.abs().max(axis=1) <= 0.2 + 1e-6).all()
    assert (result.weights.abs().sum(axis=1) <= 1.0 + 1e-6).all()

    # Artifacts should include feature engineering outputs and probabilities
    apple_artifacts = result.artifacts["AAPL"]
    assert not apple_artifacts.features.empty
    assert "rsi14" in apple_artifacts.features.columns
    assert apple_artifacts.probabilities.between(0.0, 1.0).all()
