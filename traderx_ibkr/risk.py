"""Live risk guards."""
from __future__ import annotations


def check_drawdown(current_dd: float, limit: float) -> bool:
    """Return True if drawdown is within limit."""
    return current_dd <= limit
