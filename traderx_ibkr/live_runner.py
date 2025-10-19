"""Live runner reading targets and submitting orders."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd

from traderx_ibkr.execution import Order, submit_order
from traderx_ibkr.ibkr_client import IBKRClient
from traderx_ibkr.portfolio import reconcile_targets


TARGETS_PATH = Path("data/targets/targets.csv")
POSITIONS_PATH = Path("data/processed/positions.json")
LOG_PATH = Path("data/logs/orders.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TraderX live runner")
    parser.add_argument("--targets", default=str(TARGETS_PATH))
    parser.add_argument("--mode", default="vwap")
    return parser.parse_args()


def load_positions() -> Dict[str, int]:
    if not POSITIONS_PATH.exists():
        return {}
    with POSITIONS_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_positions(positions: Dict[str, int]) -> None:
    POSITIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with POSITIONS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(positions, fh)


def log_order(payload: Dict[str, object]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")


def main() -> None:
    args = parse_args()
    targets = pd.read_csv(args.targets)
    targets.set_index("symbol", inplace=True)

    client = IBKRClient(host="127.0.0.1", port=7497, client_id=42)
    client.connect()

    current_positions = load_positions()
    prices = pd.Series(100.0, index=targets.index)
    nav = 100000

    delta = reconcile_targets(current_positions, targets["target_weight"], prices, nav)
    for symbol, qty in delta.items():
        if qty == 0:
            continue
        order = Order(symbol=symbol, quantity=int(qty))
        response = submit_order(order)
        log_order({"symbol": symbol, "quantity": int(qty), "response": response})
        current_positions[symbol] = current_positions.get(symbol, 0) + int(qty)
    save_positions(current_positions)


if __name__ == "__main__":
    main()
