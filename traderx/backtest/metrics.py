"""Backtest metric utilities."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


def _sanitize_series(values: Iterable[float] | pd.Series) -> pd.Series:
    """Return a clean :class:`pandas.Series` without NaNs or infinities."""

    if isinstance(values, pd.Series):
        series = values.copy()
    else:
        series = pd.Series(list(values))
    return series.replace([np.inf, -np.inf], np.nan).dropna()


def sharpe_ratio(returns: Iterable[float] | pd.Series) -> float:
    clean = _sanitize_series(returns)
    if clean.empty:
        return 0.0
    std = clean.std(ddof=1)
    if std == 0:
        return 0.0
    return clean.mean() / std * np.sqrt(252)


def sortino_ratio(
    returns: Iterable[float] | pd.Series,
    target: float = 0.0,
) -> float:
    clean = _sanitize_series(returns)
    if clean.empty:
        return 0.0
    downside = clean[clean < target]
    if downside.empty:
        return 0.0
    downside_diff = downside - target
    downside_std = np.sqrt((downside_diff**2).mean())
    if downside_std == 0:
        return 0.0
    excess_mean = clean.mean() - target
    return excess_mean / downside_std * np.sqrt(252)


def probabilistic_sharpe_ratio(
    returns: Iterable[float] | pd.Series,
    sr_target: float = 0.0,
) -> float:
    clean = _sanitize_series(returns)
    n_obs = len(clean)
    if n_obs < 2:
        return 0.0
    std = clean.std(ddof=1)
    if std == 0:
        return 0.0
    sr = clean.mean() / std * np.sqrt(252)
    demeaned = clean - clean.mean()
    skew = (demeaned**3).mean() / (std**3) if std else 0.0
    kurt = (demeaned**4).mean() / (std**4) if std else 3.0
    var_sr = (1 - skew * sr + ((kurt - 1) / 4) * (sr**2)) / (n_obs - 1)
    if var_sr <= 0:
        return 0.0
    z_score = (sr - sr_target) / math.sqrt(var_sr)
    return 0.5 * (1 + math.erf(z_score / math.sqrt(2)))


def expected_shortfall(
    returns: Iterable[float] | pd.Series,
    alpha: float = 0.95,
) -> float:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    clean = _sanitize_series(returns)
    if clean.empty:
        return 0.0
    threshold = clean.quantile(1 - alpha)
    tail = clean[clean <= threshold]
    if tail.empty:
        return threshold
    return tail.mean()


def portfolio_turnover(weights: Iterable[float] | pd.Series) -> float:
    if isinstance(weights, pd.Series):
        series = weights.fillna(0.0).to_numpy()
    else:
        series = np.asarray(list(weights), dtype=float)
    if series.size == 0:
        return 0.0
    diffs = np.diff(series, prepend=0.0)
    return float(np.abs(diffs).sum())


def estimate_slippage(turnover: float, costs_bps: float) -> float:
    return float(turnover) * (costs_bps / 10_000)


def max_drawdown(equity: Iterable[float] | pd.Series) -> float:
    clean = _sanitize_series(equity)
    if clean.empty:
        return 0.0
    roll_max = clean.cummax()
    drawdown = clean / roll_max - 1
    return float(drawdown.min())
