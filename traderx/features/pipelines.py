"""Feature pipeline orchestration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import pandas as pd

from traderx.features import tech


@dataclass
class FeaturePipeline:
    """A simple feature pipeline builder."""

    price_data: pd.DataFrame

    def compute(self) -> pd.DataFrame:
        close = self.price_data["close"]
        high = self.price_data["high"]
        low = self.price_data["low"]
        features: Dict[str, pd.Series] = {
            "rsi14": tech.rsi(close, 14),
            "mom_5": tech.momentum(close, 5),
            "atr14": tech.atr(high, low, close, 14),
            "rv20": tech.rolling_volatility(close, 20),
            "vol_norm": tech.normalise_volatility(close, 20),
        }
        return pd.DataFrame(features)


def registry() -> Dict[str, Callable[[pd.DataFrame], pd.Series]]:
    """Return a registry of supported feature functions."""
    return {
        "rsi14": lambda df: tech.rsi(df["close"], 14),
        "mom_5": lambda df: tech.momentum(df["close"], 5),
        "atr14": lambda df: tech.atr(df["high"], df["low"], df["close"], 14),
        "rv20": lambda df: tech.rolling_volatility(df["close"], 20),
        "vol_norm": lambda df: tech.normalise_volatility(df["close"], 20),
    }
