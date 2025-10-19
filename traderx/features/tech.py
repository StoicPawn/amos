"""Technical indicator utilities implemented with pure Python lists."""
from __future__ import annotations

from math import sqrt
from typing import Iterable, List


def _to_list(series: Iterable[float]) -> List[float]:
    return list(series)


def rolling_volatility(series: Iterable[float], window: int) -> List[float]:
    """Calculate annualised rolling volatility."""
    values = _to_list(series)
    returns = [0.0]
    for i in range(1, len(values)):
        prev = values[i - 1]
        returns.append((values[i] - prev) / prev if prev else 0.0)
    out: List[float] = []
    for i in range(len(values)):
        if i + 1 < window:
            out.append(0.0)
            continue
        window_slice = returns[i + 1 - window : i + 1]
        mean = sum(window_slice) / window
        variance = sum((x - mean) ** 2 for x in window_slice) / window if window else 0.0
        out.append(sqrt(variance) * sqrt(252))
    return out


def momentum(series: Iterable[float], window: int) -> List[float]:
    values = _to_list(series)
    out: List[float] = []
    for i in range(len(values)):
        if i < window:
            out.append(0.0)
        else:
            prev = values[i - window]
            out.append((values[i] - prev) / prev if prev else 0.0)
    return out


def rsi(series: Iterable[float], window: int = 14) -> List[float]:
    values = _to_list(series)
    gains = [0.0]
    losses = [0.0]
    for i in range(1, len(values)):
        change = values[i] - values[i - 1]
        gains.append(max(change, 0.0))
        losses.append(abs(min(change, 0.0)))
    avg_gain = 0.0
    avg_loss = 0.0
    result: List[float] = []
    alpha = 1 / window if window else 1.0
    for i in range(len(values)):
        avg_gain = (1 - alpha) * avg_gain + alpha * gains[i]
        avg_loss = (1 - alpha) * avg_loss + alpha * losses[i]
        if avg_loss == 0:
            result.append(100.0)
        else:
            rs = avg_gain / avg_loss
            result.append(100 - (100 / (1 + rs)))
    if result:
        first = min(max(result[0], 0.0), 100.0)
        result[0] = first
    return [min(max(v, 0.0), 100.0) for v in result]


def atr(high: Iterable[float], low: Iterable[float], close: Iterable[float], window: int = 14) -> List[float]:
    highs = _to_list(high)
    lows = _to_list(low)
    closes = _to_list(close)
    tr: List[float] = []
    for i in range(len(highs)):
        prev_close = closes[i - 1] if i > 0 else closes[i]
        ranges = [highs[i] - lows[i], abs(highs[i] - prev_close), abs(lows[i] - prev_close)]
        tr.append(max(ranges))
    out: List[float] = []
    for i in range(len(tr)):
        if i + 1 < window:
            out.append(tr[i])
        else:
            window_slice = tr[i + 1 - window : i + 1]
            out.append(sum(window_slice) / window)
    return out


def normalise_volatility(series: Iterable[float], window: int = 20) -> List[float]:
    values = _to_list(series)
    abs_returns = [0.0]
    for i in range(1, len(values)):
        prev = values[i - 1]
        abs_returns.append(abs((values[i] - prev) / prev) if prev else 0.0)
    out: List[float] = []
    for i in range(len(values)):
        if i + 1 < window:
            out.append(0.0)
        else:
            window_slice = abs_returns[i + 1 - window : i + 1]
            mean = sum(window_slice) / window if window else 0.0
            out.append(abs_returns[i] / mean if mean else 0.0)
    return out
