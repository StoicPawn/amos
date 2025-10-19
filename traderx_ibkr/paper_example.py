"""Paper trading example script."""
from __future__ import annotations

from traderx_ibkr.ibkr_client import IBKRClient
from traderx_ibkr.marketdata import download_history


def main() -> None:
    with IBKRClient("127.0.0.1", 7497, 1) as client:
        print("Connected:", client.is_connected())
        history = download_history(client, "AAPL", duration="1 D", bar_size="15 mins")
        print(history.tail())


if __name__ == "__main__":
    main()
