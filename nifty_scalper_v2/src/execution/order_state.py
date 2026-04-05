"""Order and Position state machines."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from typing import Optional


class OrderStatus(str, Enum):
    PENDING          = "PENDING"
    SUBMITTED        = "SUBMITTED"
    OPEN             = "OPEN"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED           = "FILLED"
    CANCELLED        = "CANCELLED"
    REJECTED         = "REJECTED"


class OrderSide(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT  = "LIMIT"
    SL     = "SL"
    SL_M   = "SL-M"


class PositionSide(str, Enum):
    LONG  = "LONG"
    SHORT = "SHORT"


_VALID_TRANSITIONS: dict[OrderStatus, set[OrderStatus]] = {
    OrderStatus.PENDING:          {OrderStatus.SUBMITTED},
    OrderStatus.SUBMITTED:        {OrderStatus.OPEN, OrderStatus.REJECTED},
    OrderStatus.OPEN:             {OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED, OrderStatus.CANCELLED},
    OrderStatus.PARTIALLY_FILLED: {OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED, OrderStatus.CANCELLED},
    OrderStatus.FILLED:           set(),
    OrderStatus.CANCELLED:        set(),
    OrderStatus.REJECTED:         set(),
}


class InvalidTransitionError(Exception):
    pass


@dataclass(slots=True, frozen=True)
class Fill:
    fill_id: str
    order_id: str
    quantity: int
    price: float
    timestamp: datetime
    is_partial: bool


@dataclass
class Order:
    order_id: str
    broker_order_id: Optional[str]
    symbol: str
    side: OrderSide
    quantity: int
    filled_quantity: int
    price: float
    trigger_price: float
    order_type: OrderType
    status: OrderStatus
    strategy_name: str
    position_id: str
    average_price: float
    created_at: datetime
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    reject_reason: Optional[str] = None
    fills: list[Fill] = field(default_factory=list)


def transition(order: Order, new_status: OrderStatus) -> Order:
    """Validate and apply status transition. Raises InvalidTransitionError."""
    allowed = _VALID_TRANSITIONS.get(order.status, set())
    if new_status not in allowed:
        raise InvalidTransitionError(
            f"Cannot transition {order.status} → {new_status} for order {order.order_id}"
        )
    order.status = new_status
    now = datetime.now(timezone.utc)
    if new_status == OrderStatus.SUBMITTED:
        order.submitted_at = now
    elif new_status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
        order.filled_at = now
    elif new_status == OrderStatus.CANCELLED:
        order.cancelled_at = now
    return order


@dataclass
class Position:
    position_id: str
    symbol: str
    underlying: str
    option_type: str       # "CE" or "PE"
    strike: float
    expiry: date
    side: PositionSide     # Always LONG in v2
    quantity: int
    initial_quantity: int
    entry_price: float
    current_price: float
    pnl_unrealized: float
    pnl_realized: float
    strategy_name: str
    regime_at_entry: str
    opened_at: datetime
    last_updated_at: datetime
    is_closed: bool = False

    def update_price(self, price: float) -> None:
        self.current_price = price
        self.pnl_unrealized = (price - self.entry_price) * self.quantity * 50  # LOT_SIZE
        self.last_updated_at = datetime.now(timezone.utc)
