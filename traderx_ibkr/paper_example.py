"""Paper trading example script."""
from __future__ import annotations

from traderx_ibkr.ibkr_client import IBKRClient


def main() -> None:
    client = IBKRClient("127.0.0.1", 7497, 1)
    client.connect()
    print("Connected:", client.is_connected())


if __name__ == "__main__":
    main()
