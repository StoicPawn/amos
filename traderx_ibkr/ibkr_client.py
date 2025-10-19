"""Thin wrapper around IBKR connection (placeholder)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IBKRClient:
    host: str
    port: int
    client_id: int

    def connect(self) -> None:
        """Simulate connection."""
        self.connected = True

    def is_connected(self) -> bool:
        return getattr(self, "connected", False)
