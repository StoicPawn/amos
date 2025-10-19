"""Utilities for ad-hoc data analysis and historical downloads."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from traderx.utils.io import AtomicWriter


@dataclass
class DownloadRequest:
    """Parameters describing a historical data download."""

    symbol: str
    market: str = "SMART"
    timeframe: str = "1 day"
    start: datetime | str | None = None
    end: datetime | str | None = None


class HistoricalDownloader:
    """Generate simple OHLCV data for ad-hoc analysis pipelines."""

    def __init__(self, default_market: str = "SMART") -> None:
        self.default_market = default_market

    def download(self, request: DownloadRequest) -> pd.DataFrame:
        now = datetime.now(timezone.utc)
        start = _normalize_datetime(request.start, now - timedelta(days=30))
        end = _normalize_datetime(request.end, now)
        if end <= start:
            raise ValueError("end must be after start")

        step, freq = _timeframe_to_step(request.timeframe)
        periods = int(((end - start).total_seconds() // step.total_seconds()) + 1)
        index = pd.date_range(start, periods=periods, freq=freq)
        index = [ts for ts in index if ts <= end]

        base_price = 100.0
        close = []
        high = []
        low = []
        volume = []
        for i, ts in enumerate(index):
            drift = 0.05 * i
            seasonal = 2.0 * __import__("math").sin(i / 5)
            price = base_price + drift + seasonal
            close.append(price)
            high.append(price * 1.002)
            low.append(price * 0.998)
            volume.append(1_000_000 + i * 10)

        frame = pd.DataFrame(
            {
                "symbol": [request.symbol] * len(index),
                "market": [request.market or self.default_market] * len(index),
                "close": close,
                "high": high,
                "low": low,
                "volume": volume,
            },
            index=index,
        )
        return frame

    def download_to_csv(self, request: DownloadRequest, destination: Path | str, frame: pd.DataFrame | None = None) -> Path:
        frame = frame if frame is not None else self.download(request)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with AtomicWriter(destination) as handle:
            frame.to_csv(handle)
        return destination


def _normalize_datetime(value: datetime | str | None, default: datetime) -> datetime:
    if value is None:
        return default
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed


def _timeframe_to_step(timeframe: str) -> tuple[timedelta, str]:
    cleaned = timeframe.strip().lower()
    if cleaned.endswith("min"):
        value = int(cleaned.split()[0]) if " " in cleaned else int(cleaned.rstrip("min"))
        return timedelta(minutes=max(value, 1)), "1min"
    if cleaned.endswith("m") and cleaned[:-1].isdigit():
        return timedelta(minutes=max(int(cleaned[:-1]), 1)), "1min"
    if cleaned.endswith("hour") or cleaned.endswith("h"):
        digits = "".join(ch for ch in cleaned if ch.isdigit())
        value = int(digits) if digits else 1
        return timedelta(hours=max(value, 1)), "1h"
    digits = "".join(ch for ch in cleaned if ch.isdigit())
    value = int(digits) if digits else 1
    return timedelta(days=max(value, 1)), "1d"


__all__ = ["DownloadRequest", "HistoricalDownloader"]
