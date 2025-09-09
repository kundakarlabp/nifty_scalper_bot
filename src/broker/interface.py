"""Common broker protocol and data transfer objects."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence


class Side(Enum):
    """Order side."""

    BUY = auto()
    SELL = auto()


class OrderType(Enum):
    """Order type."""

    MARKET = auto()
    LIMIT = auto()


class TimeInForce(Enum):
    """Time-in-force for an order."""

    DAY = auto()
    IOC = auto()
    GTC = auto()


class OrderStatus(Enum):
    """High-level order status."""

    OPEN = auto()
    PARTIALLY_FILLED = auto()
    FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()


@dataclass(frozen=True)
class Tick:
    """Market tick event."""

    instrument_id: int
    ts: float
    ltp: Decimal
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    volume: Optional[int] = None
    raw: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class OrderRequest:
    """Parameters needed to submit an order."""

    instrument_id: int
    side: Side
    qty: int
    order_type: OrderType = OrderType.MARKET
    price: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    target: Optional[Decimal] = None
    tif: TimeInForce = TimeInForce.DAY
    client_order_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class Order:
    """Executed order information."""

    order_id: str
    status: OrderStatus
    filled_qty: int
    avg_fill_price: Optional[Decimal] = None
    client_order_id: Optional[str] = None
    tif: Optional[TimeInForce] = None
    raw: Optional[Dict[str, Any]] = None


class BrokerError(Exception):
    """Base class for broker errors."""


class AuthError(BrokerError):
    """Authentication failure."""


class NetworkError(BrokerError):
    """Network communication issue."""


class RateLimitError(BrokerError):
    """API rate limit encountered."""


class OrderRejected(BrokerError):
    """Order rejected by broker."""


class OrderNotFound(BrokerError):
    """Order ID not recognised."""


class InsufficientMargin(BrokerError):
    """Not enough margin to place order."""


class Broker(Protocol):
    """Protocol that concrete broker adapters must implement."""

    def connect(self) -> None:
        """Establish connection to the broker."""

    def is_connected(self) -> bool:
        """Return ``True`` if a connection is active."""

    def subscribe_ticks(
        self, instruments: Sequence[int], on_tick: Callable[[Tick], None]
    ) -> None:
        """Subscribe to tick data for instruments."""

    def ltp(self, instrument_id: int) -> Decimal:
        """Return the last traded price for an instrument."""

    def place_order(self, req: OrderRequest) -> str:
        """Submit a new order and return its ID."""

    def modify_order(self, order_id: str, **kwargs: Any) -> None:
        """Modify an existing order."""

    def cancel_order(self, order_id: str) -> None:
        """Cancel an existing order."""

    def get_order(self, order_id: str) -> Order:
        """Fetch details for a specific order."""

    def get_positions(self) -> List[Dict[str, Any]]:
        """Return a list of open positions."""
