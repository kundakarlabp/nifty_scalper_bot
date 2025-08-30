from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class OrderState(Enum):
    """Lifecycle states for an order leg."""

    NEW = "NEW"
    PENDING = "PENDING"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class OrderSide(Enum):
    """Buy or sell side."""

    BUY = "BUY"
    SELL = "SELL"


class LegType(Enum):
    """Role of a leg in a trade."""

    ENTRY = "ENTRY"
    TP1 = "TP1"
    TP2 = "TP2"
    SL = "SL"
    TRAIL = "TRAIL"


TERMINAL_STATES = {
    OrderState.FILLED,
    OrderState.CANCELLED,
    OrderState.REJECTED,
    OrderState.EXPIRED,
}


@dataclass
class OrderLeg:
    """State for a single broker order/leg."""

    trade_id: str
    leg_id: str
    leg_type: LegType
    side: OrderSide
    symbol: str
    qty: int
    limit_price: Optional[float]
    state: OrderState
    filled_qty: int = 0
    avg_price: float = 0.0
    broker_order_id: Optional[str] = None
    idempotency_key: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    reason: Optional[str] = None

    def mark_acked(self, broker_id: str) -> None:
        """Transition NEW→PENDING on broker acknowledgement."""
        if self.state is OrderState.NEW:
            self.state = OrderState.PENDING
            self.broker_order_id = broker_id

    def on_partial(self, qty: int, avg: float) -> None:
        """Apply a partial fill update."""
        if self.state in (OrderState.PENDING, OrderState.PARTIAL):
            self.state = OrderState.PARTIAL
            self.filled_qty = qty
            self.avg_price = avg

    def on_fill(self, avg: float) -> None:
        """Mark the leg as fully filled."""
        if self.state in (OrderState.PENDING, OrderState.PARTIAL):
            self.state = OrderState.FILLED
            self.filled_qty = self.qty
            self.avg_price = avg

    def on_cancel(self, reason: str) -> None:
        """Mark the leg cancelled."""
        if self.state not in TERMINAL_STATES:
            self.state = OrderState.CANCELLED
            self.reason = reason

    def on_reject(self, reason: str) -> None:
        """Mark the leg rejected."""
        if self.state not in TERMINAL_STATES:
            self.state = OrderState.REJECTED
            self.reason = reason

    def expired(self, now: datetime) -> bool:
        """Return True if the leg has expired."""
        if self.state in TERMINAL_STATES:
            return False
        return bool(self.expires_at and now > self.expires_at)

    def to_dict(self) -> Dict[str, object]:
        """Dictionary representation for journalling/diagnostics."""
        data = asdict(self)
        data["state"] = self.state.name
        data["leg_type"] = self.leg_type.name
        data["side"] = self.side.name
        return data


@dataclass
class TradeFSM:
    """Finite state machine tracking a trade and its legs."""

    trade_id: str
    legs: Dict[str, OrderLeg] = field(default_factory=dict)
    status: str = "OPEN"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def add_leg(self, leg: OrderLeg) -> None:
        self.legs[leg.leg_id] = leg
        self.updated_at = datetime.utcnow()

    def open_legs(self) -> List[OrderLeg]:
        return [leg for leg in self.legs.values() if leg.state not in TERMINAL_STATES]

    def is_done(self) -> bool:
        return all(leg.state in TERMINAL_STATES for leg in self.legs.values())

    def close_if_done(self) -> None:
        if self.status == "OPEN" and self.is_done():
            self.status = "CLOSED"
            self.updated_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, object]:
        return {
            "trade_id": self.trade_id,
            "status": self.status,
            "legs": [leg.to_dict() for leg in self.legs.values()],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
