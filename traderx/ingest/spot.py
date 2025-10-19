"""Utilities for ad-hoc data analysis and historical downloads."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from traderx.ingest.ibkr_adapter import (
    HistoricalDataRequest,
    IBKRAdapter,
    IBKRAdapterError,
)
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
    """Download OHLCV market data from IBKR for ad-hoc analysis."""

    def __init__(self, default_market: str = "SMART", adapter: IBKRAdapter | None = None) -> None:
        self.default_market = default_market
        self.adapter = adapter or IBKRAdapter()

    def download(self, request: DownloadRequest) -> pd.DataFrame:
        now = datetime.now(timezone.utc)
        end = _normalize_datetime(request.end, now)
        start_default = end - timedelta(days=30)
        start = _normalize_datetime(request.start, start_default)
        if end <= start:
            raise ValueError("end must be after start")

        bar_size = _normalize_bar_size(request.timeframe)
        duration = _duration_from_range(start, end)
        primary_exchange = _normalize_primary_exchange(request.market)
        hist_request = HistoricalDataRequest(
            symbol=request.symbol,
            duration=duration,
            bar_size=bar_size,
            what_to_show="TRADES",
            use_rth=True,
            end_datetime=end,
            exchange=request.market or self.default_market,
            primary_exchange=primary_exchange,
        )

        try:
            frame = self.adapter.fetch([hist_request])
        except IBKRAdapterError as exc:
            raise RuntimeError(f"IBKR download failed: {exc}") from exc

        if frame.empty:
            return frame
        frame["market"] = request.market or self.default_market
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


def _normalize_bar_size(timeframe: str) -> str:
    cleaned = timeframe.strip().lower().replace("-", " ")
    if not cleaned:
        return "1 day"
    if cleaned.endswith("min"):
        value = cleaned.replace("min", "").strip()
        value = int(value) if value else 1
        value = max(value, 1)
        return "1 min" if value == 1 else f"{value} mins"
    if cleaned.endswith("m") and cleaned[:-1].isdigit():
        value = max(int(cleaned[:-1]), 1)
        return "1 min" if value == 1 else f"{value} mins"
    if cleaned.endswith("hour"):
        digits = "".join(ch for ch in cleaned if ch.isdigit())
        value = max(int(digits), 1) if digits else 1
        return "1 hour" if value == 1 else f"{value} hours"
    if cleaned.endswith("h") and cleaned[:-1].isdigit():
        value = max(int(cleaned[:-1]), 1)
        return "1 hour" if value == 1 else f"{value} hours"
    if cleaned.endswith("day") or cleaned.endswith("d"):
        digits = "".join(ch for ch in cleaned if ch.isdigit())
        value = max(int(digits), 1) if digits else 1
        return "1 day" if value == 1 else f"{value} days"
    if cleaned.endswith("week") or cleaned.endswith("w"):
        digits = "".join(ch for ch in cleaned if ch.isdigit())
        value = max(int(digits), 1) if digits else 1
        return "1 week" if value == 1 else f"{value} weeks"
    if cleaned.endswith("month") or cleaned.endswith("mo"):
        digits = "".join(ch for ch in cleaned if ch.isdigit())
        value = max(int(digits), 1) if digits else 1
        return "1 month" if value == 1 else f"{value} months"
    return timeframe


def _duration_from_range(start: datetime, end: datetime) -> str:
    delta = end - start
    seconds = max(int(delta.total_seconds()), 1)
    if seconds < 86400:
        return f"{seconds} S"
    days = seconds / 86400
    if days < 7:
        return f"{math.ceil(days)} D"
    weeks = days / 7
    if weeks < 4:
        return f"{math.ceil(weeks)} W"
    months = days / 30
    if months < 12:
        return f"{math.ceil(months)} M"
    years = days / 365
    return f"{max(1, math.ceil(years))} Y"


def _normalize_primary_exchange(market: str | None) -> str | None:
    if not market:
        return None
    upper = market.upper()
    if upper in {"NASDAQ", "NYSE"}:
        return upper
    return None


__all__ = ["DownloadRequest", "HistoricalDownloader"]
