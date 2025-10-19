"""Interactive Brokers market data adapter using :mod:`ib_insync`."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from ib_insync import IB, Stock

from traderx.utils.config import load_config


_LOGGER = logging.getLogger(__name__)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "ibkr.yaml"
_DEFAULT_ENV_PATH = _REPO_ROOT / ".env"
_DEFAULT_COLUMNS = [
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "bar_count",
    "wap",
]
_RETRYABLE_ERROR_CODES = {162, 366, 420, 504}  # pacing, data farm issues
_CONTRACT_ERROR_CODES = {200, 321}  # contract definition related errors
_ENV_LOADED = False
_ENV_LOCK = threading.Lock()


class IBKRAdapterError(RuntimeError):
    """Adapter level exception with extra context."""

    def __init__(
        self,
        message: str,
        *,
        code: int | None = None,
        request: "HistoricalDataRequest" | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.request = request
        self.retryable = retryable


def _ensure_env_loaded() -> None:
    """Populate ``os.environ`` with key/value pairs from ``.env`` if present."""

    global _ENV_LOADED
    if _ENV_LOADED:
        return
    with _ENV_LOCK:
        if _ENV_LOADED or not _DEFAULT_ENV_PATH.exists():
            _ENV_LOADED = True
            return
        for line in _DEFAULT_ENV_PATH.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)
        _ENV_LOADED = True


def _load_connection_defaults() -> dict[str, str | int | float]:
    """Return IBKR connection defaults from configuration and environment."""

    _ensure_env_loaded()
    cfg: dict[str, str | int | float] = {}
    if _DEFAULT_CONFIG_PATH.exists():
        bundle = load_config(_DEFAULT_CONFIG_PATH)
        cfg.update(bundle.get("connection", {}))
    env_mappings = {
        "host": os.getenv("IBKR_HOST"),
        "port": os.getenv("IBKR_PORT"),
        "clientId": os.getenv("IBKR_CLIENT_ID"),
        "connectTimeout": os.getenv("IBKR_CONNECT_TIMEOUT"),
        "readonly": os.getenv("IBKR_READONLY"),
    }
    for key, value in env_mappings.items():
        if value is not None:
            cfg[key] = value
    return cfg


def _parse_bool(value: str | int | bool | None, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _to_int(value: str | int | float | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return default


def _to_float(value: str | int | float | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return default


@dataclass(slots=True)
class HistoricalDataRequest:
    """Parameters for a historical market data download."""

    symbol: str
    duration: str = "1 D"
    bar_size: str = "1 day"
    what_to_show: str = "TRADES"
    use_rth: bool = True
    end_datetime: datetime | str | None = None
    exchange: str = "SMART"
    currency: str = "USD"
    primary_exchange: str | None = None

    def formatted_end_datetime(self) -> str:
        if self.end_datetime is None:
            return ""
        if isinstance(self.end_datetime, datetime):
            dt = self.end_datetime.astimezone(timezone.utc)
            return dt.strftime("%Y%m%d %H:%M:%S")
        return str(self.end_datetime)


@dataclass
class IBKRAdapter:
    """Adapter responsible for downloading historical bars from IBKR."""

    host: str | None = None
    port: int | None = None
    client_id: int | None = None
    readonly: bool = True
    connect_timeout: float = 10.0
    request_timeout: float = 30.0
    max_retries: int = 3
    retry_backoff: float = 1.5
    initial_retry_delay: float = 1.0
    logger: logging.Logger = field(default_factory=lambda: _LOGGER)

    def __post_init__(self) -> None:
        defaults = _load_connection_defaults()
        self.host = self.host or str(defaults.get("host", "127.0.0.1"))
        self.port = self.port or _to_int(defaults.get("port"), 7497)
        self.client_id = self.client_id or _to_int(defaults.get("clientId"), 42)
        self.connect_timeout = _to_float(defaults.get("connectTimeout"), self.connect_timeout)
        self.readonly = _parse_bool(defaults.get("readonly"), self.readonly)
        self._ib = IB()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------
    def connect(self) -> None:
        """Establish the connection with the IBKR Gateway/TWS."""

        with self._lock:
            if self._ib.isConnected():
                return
            self.logger.info(
                "Connecting to IBKR", extra={"host": self.host, "port": self.port, "client_id": self.client_id}
            )
            connected = self._ib.connect(
                host=self.host,
                port=self.port,
                clientId=self.client_id,
                timeout=self.connect_timeout,
                readonly=self.readonly,
            )
            if not connected:
                raise ConnectionError(
                    f"Unable to connect to IBKR at {self.host}:{self.port} (client_id={self.client_id})"
                )

    def disconnect(self) -> None:
        """Close the underlying IBKR connection."""

        with self._lock:
            if self._ib.isConnected():
                self._ib.disconnect()

    def ensure_connected(self) -> None:
        try:
            self.connect()
        except (ConnectionError, asyncio.TimeoutError, OSError) as exc:  # pragma: no cover - network dependent
            raise IBKRAdapterError(str(exc), request=None, retryable=True) from exc

    # ------------------------------------------------------------------
    # Historical data download
    # ------------------------------------------------------------------
    def fetch(self, requests: Iterable[HistoricalDataRequest]) -> pd.DataFrame:
        """Fetch and concatenate historical data for ``requests``."""

        req_list = list(requests)
        if not req_list:
            return pd.DataFrame(columns=_DEFAULT_COLUMNS)

        frames: List[pd.DataFrame] = []
        for req in req_list:
            frame = self._fetch_with_retry(req)
            frames.append(frame)

        if not frames:
            return pd.DataFrame(columns=_DEFAULT_COLUMNS)

        combined = pd.concat(frames)
        combined = combined.sort_index()
        return combined

    def _fetch_with_retry(self, req: HistoricalDataRequest) -> pd.DataFrame:
        delay = self.initial_retry_delay
        for attempt in range(self.max_retries + 1):
            start_time = time.monotonic()
            try:
                frame = self._fetch_single(req)
            except IBKRAdapterError as exc:
                latency = time.monotonic() - start_time
                self._log_request(req, success=False, latency=latency, rows=0, error=str(exc))
                if exc.retryable and attempt < self.max_retries:
                    self.logger.warning(
                        "Retrying IBKR historical request", extra={"symbol": req.symbol, "attempt": attempt + 1}
                    )
                    time.sleep(delay)
                    delay *= self.retry_backoff
                    self.ensure_connected()
                    continue
                raise
            else:
                latency = time.monotonic() - start_time
                self._log_request(req, success=True, latency=latency, rows=len(frame), error=None)
                return frame

        raise IBKRAdapterError("Exceeded maximum retries for IBKR request", request=req, retryable=False)

    def _fetch_single(self, req: HistoricalDataRequest) -> pd.DataFrame:
        self.ensure_connected()
        contract = Stock(
            req.symbol,
            exchange=req.exchange,
            currency=req.currency,
            primaryExchange=req.primary_exchange,
        )
        try:
            bars = self._ib.reqHistoricalData(
                contract,
                endDateTime=req.formatted_end_datetime(),
                durationStr=req.duration,
                barSizeSetting=req.bar_size,
                whatToShow=req.what_to_show,
                useRTH=1 if req.use_rth else 0,
                formatDate=1,
                keepUpToDate=False,
                timeout=self.request_timeout,
            )
        except (ConnectionError, OSError) as exc:
            raise IBKRAdapterError("IBKR session disconnected", request=req, retryable=True) from exc
        except asyncio.TimeoutError as exc:
            raise IBKRAdapterError("IBKR historical data request timed out", request=req, retryable=True) from exc
        except Exception as exc:  # pragma: no cover - defensive
            message, code, retryable = self._interpret_ib_exception(exc, req)
            raise IBKRAdapterError(message, code=code, request=req, retryable=retryable) from exc

        frame = self._bars_to_frame(bars, req.symbol)
        return frame

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _bars_to_frame(self, bars, symbol: str) -> pd.DataFrame:
        if not bars:
            return pd.DataFrame(columns=_DEFAULT_COLUMNS)

        records = []
        for bar in bars:
            ts = pd.Timestamp(bar.date)
            if ts.tzinfo is None:
                ts = ts.tz_localize(timezone.utc)
            else:
                ts = ts.tz_convert(timezone.utc)
            records.append(
                {
                    "timestamp": ts,
                    "symbol": symbol,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "bar_count": getattr(bar, "barCount", None),
                    "wap": getattr(bar, "WAP", getattr(bar, "wap", None)),
                }
            )

        frame = pd.DataFrame.from_records(records)
        frame = frame.set_index("timestamp")
        frame = frame.sort_index()
        frame = frame[_DEFAULT_COLUMNS]
        return frame

    def _interpret_ib_exception(
        self, exc: Exception, req: HistoricalDataRequest
    ) -> tuple[str, int | None, bool]:
        error_code = getattr(exc, "errorCode", getattr(exc, "code", None))
        error_msg = getattr(exc, "errorMsg", str(exc))

        message = self._format_error_message(error_code, error_msg, req)
        retryable = bool(error_code in _RETRYABLE_ERROR_CODES) if error_code is not None else False
        return message, error_code, retryable

    def _format_error_message(
        self, error_code: int | None, error_msg: str, req: HistoricalDataRequest
    ) -> str:
        if error_code is None:
            return f"IBKR error while fetching {req.symbol}: {error_msg}"

        prefix = f"IBKR error {error_code} while fetching {req.symbol}"
        if error_code in _RETRYABLE_ERROR_CODES:
            return f"{prefix}: pacing/data farm violation ({error_msg})"
        if error_code in _CONTRACT_ERROR_CODES:
            return f"{prefix}: contract not found ({error_msg})"
        if error_code == 354:
            return f"{prefix}: no data returned ({error_msg})"
        return f"{prefix}: {error_msg}"

    def _log_request(
        self,
        req: HistoricalDataRequest,
        *,
        success: bool,
        latency: float,
        rows: int,
        error: str | None,
    ) -> None:
        payload = {
            "symbol": req.symbol,
            "duration": req.duration,
            "barSize": req.bar_size,
            "success": success,
            "latencyMs": round(latency * 1000, 3),
            "rows": rows,
        }
        if error:
            payload["error"] = error
        self.logger.info(json.dumps(payload, default=str))

    # ------------------------------------------------------------------
    # Context manager helpers
    # ------------------------------------------------------------------
    def __enter__(self) -> "IBKRAdapter":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        self.disconnect()


__all__ = ["HistoricalDataRequest", "IBKRAdapter", "IBKRAdapterError"]

