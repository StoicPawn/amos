from __future__ import annotations

from traderx.features.tech import atr, rsi


def test_rsi_basic():
    prices = [100, 101, 102, 103, 102, 104]
    values = rsi(prices, window=3)
    assert len(set(round(v, 2) for v in values)) > 1
    assert all(0 <= v <= 100 for v in values)


def test_atr_returns_positive():
    high = [101, 102, 103, 104]
    low = [99, 100, 101, 102]
    close = [100, 101, 102, 103]
    values = atr(high, low, close, window=2)
    assert values[-1] >= 0
    assert all(v >= 0 for v in values)
