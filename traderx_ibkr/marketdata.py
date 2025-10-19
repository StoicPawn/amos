"""Market data helpers using the :class:`~traderx_ibkr.ibkr_client.IBKRClient`."""
from __future__ import annotations

import pandas as pd

from traderx_ibkr.ibkr_client import IBKRClient


def download_history(
    client: IBKRClient,
    symbol: str,
    *,
    start_datetime: str | pd.Timestamp | None = None,
    end_datetime: str | pd.Timestamp | None = None,
    duration: str = "1 D",
    max_chunk_duration: str = "30 D",
    bar_size: str = "5 mins",
    what_to_show: str = "TRADES",
    use_rth: bool = True,
    exchange: str = "SMART",
    currency: str = "USD",
    primary_exchange: str | None = None,
) -> pd.DataFrame:
    """Return historical price bars for ``symbol``.

    Parameters
    ----------
    client:
        Connected :class:`IBKRClient` instance.
    symbol:
        Ticker symbol to query.
    start_datetime, end_datetime:
        Optional timestamps delimiting the history to download.  If ``start_datetime``
        is supplied, the method automatically derives the required ``duration`` and
        fetches the data in ``max_chunk_duration`` windows.
    duration:
        Duration string as required by ``ibapi`` (e.g. ``"1 D"``, ``"1 W"``).  Ignored
        when ``start_datetime`` is provided.
    max_chunk_duration:
        Maximum chunk size (e.g. ``"30 D"``) when splitting large requests.
    bar_size:
        Bar size string (e.g. ``"5 mins"``, ``"1 hour"``).
    what_to_show:
        Data type to request (``"TRADES"``, ``"MIDPOINT"``, etc.).
    use_rth:
        Whether to restrict data to regular trading hours.
    exchange, currency, primary_exchange:
        Contract routing parameters.
    """

    if not client.is_connected():
        raise RuntimeError("Client not connected to IBKR")

    return client.request_historical_data(
        symbol,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        duration=duration,
        max_chunk_duration=max_chunk_duration,
        bar_size=bar_size,
        what_to_show=what_to_show,
        use_rth=use_rth,
        exchange=exchange,
        currency=currency,
        primary_exchange=primary_exchange,
    )
