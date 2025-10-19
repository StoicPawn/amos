from __future__ import annotations

from traderx_ibkr.portfolio import reconcile_targets


def test_reconcile_targets_delta():
    current = {"AAPL": 10, "MSFT": -5}
    targets = {"AAPL": 0.05, "MSFT": -0.02}
    prices = {"AAPL": 100.0, "MSFT": 200.0}
    nav = 100000
    delta = reconcile_targets(current, targets, prices, nav)
    assert delta["AAPL"] > 0
    assert delta["MSFT"] < 0
    assert isinstance(delta["AAPL"], int)
