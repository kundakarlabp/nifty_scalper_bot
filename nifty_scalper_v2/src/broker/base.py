"""BrokerGateway ABC — all broker implementations must implement this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BrokerGateway(ABC):
    """Abstract broker interface. All methods are async."""

    @abstractmethod
    async def place_order(self, symbol: str, side: str, quantity: int, order_type: str, price: float = 0.0) -> dict[str, Any]:
        """Place an order. Returns dict with order_id, average_price, status."""
        ...

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an open order. Returns True on success."""
        ...

    @abstractmethod
    async def get_orders(self) -> list[dict[str, Any]]:
        """Return all orders for the current session."""
        ...

    @abstractmethod
    async def get_positions(self) -> list[dict[str, Any]]:
        """Return all open positions."""
        ...

    @abstractmethod
    async def get_quote(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        """Return LTP/bid/ask/OI for given symbols."""
        ...

    @abstractmethod
    async def get_profile(self) -> dict[str, Any]:
        """Return account profile. Used for auth verification."""
        ...

    @abstractmethod
    async def get_instruments(self, exchange: str = "NFO") -> list[dict[str, Any]]:
        """Return full instrument list for the exchange."""
        ...
