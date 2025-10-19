"""Microstructure feature approximations."""
from __future__ import annotations

import pandas as pd


def bid_ask_spread(bid: pd.Series, ask: pd.Series) -> pd.Series:
    """Compute relative bid/ask spread in basis points."""
    mid = (bid + ask) / 2
    spread = (ask - bid) / mid
    return (spread * 1e4).fillna(0)


def order_imbalance(bid_volume: pd.Series, ask_volume: pd.Series) -> pd.Series:
    """Order imbalance metric between -1 and 1."""
    total = bid_volume + ask_volume
    imbalance = (bid_volume - ask_volume) / total.replace(0, pd.NA)
    return imbalance.fillna(0)
