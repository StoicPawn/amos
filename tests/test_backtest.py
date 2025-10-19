from __future__ import annotations

from datetime import timedelta

from traderx.backtest.cv import PurgedKFold
from traderx.backtest.engine import BacktestEngine
from traderx.exec.simulate_exec import simulate_vwap


def test_purged_kfold_respects_embargo():
    idx = list(range(10))
    X = {i: val for i, val in enumerate(idx)}
    pkf = PurgedKFold(n_splits=3, embargo_td=timedelta(days=1))
    splits = list(pkf.split(idx))
    assert len(splits) == 3
    for train_idx, test_idx in splits:
        if not train_idx:
            continue
        assert max(train_idx) < min(test_idx) or min(train_idx) > max(test_idx)


def test_backtest_engine_generates_equity():
    prices = {"AAPL": [100, 101, 102, 103]}
    weights = {"AAPL": [0.0, 0.1, 0.1, 0.1]}
    engine = BacktestEngine(prices)
    result = engine.run(weights)
    assert len(result.equity) == 4
    assert result.equity[-1] > 0


def test_simulate_vwap_returns_fill_price():
    prices = {
        "high": [101, 102, 103],
        "low": [99, 100, 101],
        "close": [100, 101, 102],
    }
    fill = simulate_vwap(prices, participation=0.05)
    assert all(fill[i] > prices["low"][i] for i in range(len(fill)))
    assert all(fill[i] < prices["high"][i] * 1.1 for i in range(len(fill)))
