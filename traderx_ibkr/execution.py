"""Execution utilities for IBKR."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Order:
    symbol: str
    quantity: int
    order_type: str = "MKT"


def submit_order(order: Order) -> dict:
    """Simulate order submission."""
    return {"symbol": order.symbol, "quantity": order.quantity, "status": "submitted"}
