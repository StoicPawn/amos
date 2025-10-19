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
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from ibapi.client import EClient
from ibapi.common import BarData
from ibapi.contract import Contract, ContractDetails
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
        self._request_events: Dict[int, threading.Event] = {}
        self._historical_data: Dict[int, List[BarData]] = {}
        self._contract_details: Dict[int, List[ContractDetails]] = {}
        self._request_errors: Dict[int, Exception] = {}

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
            event = self._request_events.get(reqId)
            if event is not None:
                self._request_errors[reqId] = RuntimeError(message)
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
            event = self._request_events.get(reqId)
        if event is not None:
            event.set()
        super().historicalDataEnd(reqId, start, end)

    def contractDetails(  # noqa: N802 (IB API signature)
        self, reqId: int, contractDetails: ContractDetails
    ) -> None:
        with self._lock:
            self._contract_details.setdefault(reqId, []).append(contractDetails)
        super().contractDetails(reqId, contractDetails)

    def contractDetailsEnd(self, reqId: int) -> None:  # noqa: N802
        with self._lock:
            event = self._request_events.get(reqId)
        if event is not None:
            event.set()
        super().contractDetailsEnd(reqId)

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def next_request_id(self) -> int:
        return next(self._req_id_counter)

    def register_historical_request(self, req_id: int) -> threading.Event:
        event = threading.Event()
        with self._lock:
            self._request_events[req_id] = event
            self._historical_data[req_id] = []
            self._contract_details.pop(req_id, None)
            self._request_errors.pop(req_id, None)
        return event

    def pop_historical_response(self, req_id: int) -> Tuple[List[BarData], Exception | None]:
        with self._lock:
            data = self._historical_data.pop(req_id, [])
            self._request_events.pop(req_id, None)
            error = self._request_errors.pop(req_id, None)
        return data, error

    def register_contract_request(self, req_id: int) -> threading.Event:
        event = threading.Event()
        with self._lock:
            self._request_events[req_id] = event
            self._contract_details[req_id] = []
            self._historical_data.pop(req_id, None)
            self._request_errors.pop(req_id, None)
        return event

    def pop_contract_response(
        self, req_id: int
    ) -> Tuple[List[ContractDetails], Exception | None]:
        with self._lock:
            details = self._contract_details.pop(req_id, [])
            self._request_events.pop(req_id, None)
            error = self._request_errors.pop(req_id, None)
        return details, error


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
    def _as_timestamp(
        self, value: str | datetime | pd.Timestamp | None, *, default: pd.Timestamp
    ) -> pd.Timestamp:
        if value is None or value == "":
            ts = default
        elif isinstance(value, pd.Timestamp):
            ts = value
        else:
            ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.tz_localize(None)

    def _ib_date_str(self, ts: pd.Timestamp) -> str:
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        return ts.strftime("%Y%m%d %H:%M:%S")

    def _duration_to_ib(self, delta: pd.Timedelta) -> str:
        total_seconds = int(delta.total_seconds())
        if total_seconds <= 0:
            raise ValueError("Duration must be positive")
        seconds_per_day = 24 * 60 * 60
        seconds_per_hour = 60 * 60
        seconds_per_minute = 60

        if total_seconds % seconds_per_day == 0:
            days = total_seconds // seconds_per_day
            if days % 7 == 0:
                weeks = days // 7
                return f"{weeks} W"
            return f"{days} D"
        if total_seconds % seconds_per_hour == 0:
            hours = total_seconds // seconds_per_hour
            return f"{hours} H"
        if total_seconds % seconds_per_minute == 0:
            minutes = total_seconds // seconds_per_minute
            return f"{minutes} M"
        return f"{total_seconds} S"

    def _iter_historical_chunks(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        chunk: pd.Timedelta,
    ) -> Iterable[tuple[str, str]]:
        current_end = end
        while current_end > start:
            chunk_start = max(start, current_end - chunk)
            duration = current_end - chunk_start
            yield self._ib_date_str(current_end), self._duration_to_ib(duration)
            current_end = chunk_start

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

    def qualify_stock_contract(
        self,
        symbol: str,
        exchange: str = "SMART",
        currency: str = "USD",
        primary_exchange: str | None = None,
    ) -> Contract:
        if not self.is_connected():
            raise RuntimeError("IBKR client is not connected")

        base_contract = self.create_stock_contract(
            symbol,
            exchange=exchange,
            currency=currency,
            primary_exchange=primary_exchange,
        )

        details = self.request_contract_details(base_contract)
        if not details:
            raise ValueError(f"No contract details returned for {symbol}")

        qualified = details[0].contract
        qualified.exchange = exchange
        if primary_exchange:
            qualified.primaryExchange = primary_exchange
        elif details[0].contract.primaryExchange:
            qualified.primaryExchange = details[0].contract.primaryExchange
        qualified.currency = currency
        self.logger.debug(
            "Qualified contract for %s: conId=%s, primaryExchange=%s",
            symbol,
            qualified.conId,
            qualified.primaryExchange or "",
        )
        return qualified

    def request_contract_details(self, contract: Contract) -> List[ContractDetails]:
        if not self.is_connected():
            raise RuntimeError("IBKR client is not connected")

        req_id = self._app.next_request_id()
        event = self._app.register_contract_request(req_id)
        self.logger.debug(
            "Requesting raw contract details for %s (req_id=%s)", contract.symbol, req_id
        )
        self._app.reqContractDetails(req_id, contract)

        if not event.wait(self.request_timeout):
            raise TimeoutError(
                "Timed out waiting for contract details "
                f"for {contract.symbol} (req_id={req_id})"
            )

        details, error = self._app.pop_contract_response(req_id)
        if error is not None:
            raise error
        return details

    def request_historical_data(
        self,
        symbol: str,
        *,
        start_datetime: str | datetime | pd.Timestamp | None = None,
        end_datetime: str | datetime | pd.Timestamp | None = None,
        duration: str = "1 D",
        max_chunk_duration: str = "30 D",
        bar_size: str = "5 mins",
        what_to_show: str = "TRADES",
        use_rth: bool = True,
        exchange: str = "SMART",
        currency: str = "USD",
        primary_exchange: str | None = None,
    ) -> pd.DataFrame:
        """Fetch historical bars for ``symbol`` as a :class:`pandas.DataFrame`.

        Parameters
        ----------
        symbol:
            The ticker to query.
        start_datetime, end_datetime:
            Optional datetime bounds.  When ``start_datetime`` is supplied the
            method derives the required ``duration`` and breaks long intervals
            into ``max_chunk_duration`` windows to comply with IB's limits.
        duration:
            Duration string understood by IB (e.g. ``"1 D"``).  Used when
            ``start_datetime`` is not provided.
        max_chunk_duration:
            Maximum window size to use when chunking long requests.
        bar_size, what_to_show, use_rth, exchange, currency, primary_exchange:
            Parameters passed to ``reqHistoricalData``.
        """

        if not self.is_connected():
            raise RuntimeError("IBKR client is not connected")

        contract = self.qualify_stock_contract(
            symbol,
            exchange=exchange,
            currency=currency,
            primary_exchange=primary_exchange,
        )

        chunks: Iterable[tuple[str, str]]
        if start_datetime is not None:
            end_ts = self._as_timestamp(end_datetime, default=pd.Timestamp.utcnow())
            start_ts = self._as_timestamp(
                start_datetime,
                default=end_ts - pd.Timedelta(duration.replace(" ", "")),
            )
            if end_ts <= start_ts:
                raise ValueError("end_datetime must be after start_datetime")
            chunk_td = pd.Timedelta(max_chunk_duration.replace(" ", ""))
            if chunk_td <= pd.Timedelta(0):
                raise ValueError("max_chunk_duration must be positive")
            chunks = self._iter_historical_chunks(start_ts, end_ts, chunk_td)
        else:
            if end_datetime in (None, ""):
                chunk_end = ""
            else:
                end_ts = self._as_timestamp(end_datetime, default=pd.Timestamp.utcnow())
                chunk_end = self._ib_date_str(end_ts)
            chunks = [(chunk_end, duration)]

        frames: List[pd.DataFrame] = []
        for chunk_end, chunk_duration in chunks:
            req_id = self._app.next_request_id()
            event = self._app.register_historical_request(req_id)
            self.logger.debug(
                "Requesting historical data for %s (req_id=%s, duration=%s, bar_size=%s, end=%s)",
                symbol,
                req_id,
                chunk_duration,
                bar_size,
                chunk_end,
            )
            self._app.reqHistoricalData(
                reqId=req_id,
                contract=contract,
                endDateTime=chunk_end,
                durationStr=chunk_duration,
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
                continue

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
            frame = pd.DataFrame.from_records(records).set_index("timestamp").sort_index()
            frames.append(frame)

        if not frames:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume", "bar_count", "wap"]
            )

        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df

    # Convenience context manager helpers
    def __enter__(self) -> "IBKRClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        self.disconnect()
