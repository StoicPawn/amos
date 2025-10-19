"""Real IBKR client wrapper built on top of ``ibapi``.

This module exposes :class:`IBKRClient`, a small convenience wrapper that hides
most of the boilerplate required to interact with Interactive Brokers via the
``ibapi`` package.  The client handles the connection life-cycle, request id
management and turns the asynchronous ``EWrapper`` callbacks into blocking
helpers that return ``pandas`` data frames.
"""
from __future__ import annotations

import itertools
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pandas as pd
from ibapi.client import EClient
from ibapi.common import BarData
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper


_LOGGER = logging.getLogger(__name__)


class _IBKRApp(EWrapper, EClient):
    """Combined ``EWrapper``/``EClient`` with a minimal synchronous facade."""

    def __init__(self, logger: logging.Logger) -> None:
        EWrapper.__init__(self)
        EClient.__init__(self, wrapper=self)
        self._logger = logger
        self.connected_event = threading.Event()
        self._req_id_counter = itertools.count(1)
        self._lock = threading.Lock()
        self._historical_events: Dict[int, threading.Event] = {}
        self._historical_data: Dict[int, List[BarData]] = {}
        self._historical_errors: Dict[int, Exception] = {}

    # ------------------------------------------------------------------
    # Connection / error handling callbacks
    # ------------------------------------------------------------------
    def nextValidId(self, orderId: int) -> None:  # noqa: N802 (IB API signature)
        self._logger.debug("Received next valid order id %s", orderId)
        self.connected_event.set()
        super().nextValidId(orderId)

    def error(  # noqa: D401,N802 (IB API signature)
        self,
        reqId: int,
        errorCode: int,
        errorString: str,
        advancedOrderRejectJson: str = "",
    ) -> None:
        """Handle IBKR errors and wake up waiting requests."""

        message = f"IBKR error {errorCode} (reqId={reqId}): {errorString}"
        # Do not treat noisy status updates as failures.
        if errorCode not in {2104, 2106, 2158, 2159, 10167}:
            self._logger.error(message)
        else:
            self._logger.info(message)
        with self._lock:
            event = self._historical_events.get(reqId)
            if event is not None:
                self._historical_errors[reqId] = RuntimeError(message)
                event.set()
        super().error(reqId, errorCode, errorString, advancedOrderRejectJson)

    # ------------------------------------------------------------------
    # Historical market data callbacks
    # ------------------------------------------------------------------
    def historicalData(self, reqId: int, bar: BarData) -> None:  # noqa: N802
        with self._lock:
            self._historical_data.setdefault(reqId, []).append(bar)
        super().historicalData(reqId, bar)

    def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:  # noqa: N802
        with self._lock:
            event = self._historical_events.get(reqId)
        if event is not None:
            event.set()
        super().historicalDataEnd(reqId, start, end)

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def next_request_id(self) -> int:
        return next(self._req_id_counter)

    def register_historical_request(self, req_id: int) -> threading.Event:
        event = threading.Event()
        with self._lock:
            self._historical_events[req_id] = event
            self._historical_data[req_id] = []
            self._historical_errors.pop(req_id, None)
        return event

    def pop_historical_response(self, req_id: int) -> Tuple[List[BarData], Exception | None]:
        with self._lock:
            data = self._historical_data.pop(req_id, [])
            self._historical_events.pop(req_id, None)
            error = self._historical_errors.pop(req_id, None)
        return data, error


@dataclass
class IBKRClient:
    """Synchronous helper built on top of ``ibapi``'s asynchronous client."""

    host: str
    port: int
    client_id: int
    connect_timeout: float = 10.0
    request_timeout: float = 30.0
    logger: logging.Logger = field(default_factory=lambda: _LOGGER)

    def __post_init__(self) -> None:
        self._app = _IBKRApp(self.logger)
        self._reader_thread: threading.Thread | None = None
        self._thread_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------
    def connect(self) -> None:
        """Open the socket to IBKR's Gateway/TWS and start the network thread."""

        with self._thread_lock:
            if self.is_connected():
                self.logger.debug("IBKR client already connected")
                return
            self.logger.info(
                "Connecting to IBKR at %s:%s with client id %s",
                self.host,
                self.port,
                self.client_id,
            )
            self._app.connected_event.clear()
            success = self._app.connect(self.host, self.port, self.client_id)
            if not success:
                raise ConnectionError("Unable to initiate connection to IBKR")
            self._reader_thread = threading.Thread(target=self._app.run, daemon=True)
            self._reader_thread.start()
            # Trigger a request id so that ``nextValidId`` fires.
            self._app.reqIds(-1)
        if not self._app.connected_event.wait(self.connect_timeout):
            self.disconnect()
            raise TimeoutError(
                "Timed out while waiting for the IBKR session to become available"
            )
        self.logger.info("Connected to IBKR")

    def disconnect(self) -> None:
        """Disconnect from IBKR and stop the reader thread."""

        with self._thread_lock:
            if self._app.isConnected():
                self.logger.info("Disconnecting from IBKR")
                self._app.disconnect()
            if self._reader_thread and self._reader_thread.is_alive():
                self._reader_thread.join(timeout=2.0)
            self._reader_thread = None

    def is_connected(self) -> bool:
        return bool(self._app.isConnected())

    # ------------------------------------------------------------------
    # Market data helpers
    # ------------------------------------------------------------------
    def create_stock_contract(
        self,
        symbol: str,
        exchange: str = "SMART",
        currency: str = "USD",
        primary_exchange: str | None = None,
    ) -> Contract:
        """Create a stock contract suitable for most US equities."""

        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = exchange
        contract.currency = currency
        if primary_exchange:
            contract.primaryExchange = primary_exchange
        return contract

    def request_historical_data(
        self,
        symbol: str,
        *,
        end_datetime: str = "",
        duration: str = "1 D",
        bar_size: str = "5 mins",
        what_to_show: str = "TRADES",
        use_rth: bool = True,
        exchange: str = "SMART",
        currency: str = "USD",
        primary_exchange: str | None = None,
    ) -> pd.DataFrame:
        """Fetch historical bars for ``symbol`` as a :class:`pandas.DataFrame`."""

        if not self.is_connected():
            raise RuntimeError("IBKR client is not connected")

        contract = self.create_stock_contract(
            symbol,
            exchange=exchange,
            currency=currency,
            primary_exchange=primary_exchange,
        )
        req_id = self._app.next_request_id()
        event = self._app.register_historical_request(req_id)

        self.logger.debug(
            "Requesting historical data for %s (req_id=%s, duration=%s, bar_size=%s)",
            symbol,
            req_id,
            duration,
            bar_size,
        )
        self._app.reqHistoricalData(
            reqId=req_id,
            contract=contract,
            endDateTime=end_datetime,
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=1 if use_rth else 0,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[],
        )

        if not event.wait(self.request_timeout):
            raise TimeoutError(
                f"Timed out waiting for historical data for {symbol} (req_id={req_id})"
            )

        bars, error = self._app.pop_historical_response(req_id)
        if error is not None:
            raise error
        if not bars:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume", "bar_count", "wap"]
            )

        records = []
        for bar in bars:
            timestamp = pd.to_datetime(bar.date)
            records.append(
                {
                    "timestamp": timestamp,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "bar_count": bar.barCount,
                    "wap": bar.WAP,
                }
            )
        df = pd.DataFrame.from_records(records).set_index("timestamp").sort_index()
        return df

    # Convenience context manager helpers
    def __enter__(self) -> "IBKRClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        self.disconnect()
