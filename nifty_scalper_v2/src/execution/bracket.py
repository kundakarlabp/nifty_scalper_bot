"""Bracket order management: SL, TP1, TP2, trailing stop."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .order_state import Order
    from ..strategies.signal import Signal


@dataclass
class Bracket:
    bracket_id: str
    position_id: str
    symbol: str
    side: str               # "BUY" (long option)
    initial_quantity: int
    remaining_quantity: int
    entry_price: float
    sl_price: float
    initial_sl_price: float
    tp1_price: float
    tp2_price: float
    tp1_hit: bool = False
    tp2_hit: bool = False
    trail_active: bool = False
    trail_price: Optional[float] = None
    peak_price: Optional[float] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_evaluated_at: Optional[datetime] = None

    def initial_risk(self) -> float:
        return abs(self.entry_price - self.initial_sl_price)

    def current_r_multiple(self, current_price: float) -> float:
        risk = self.initial_risk()
        if risk <= 0:
            return 0.0
        return (current_price - self.entry_price) / risk


@dataclass
class ExitSignal:
    position_id: str
    bracket_id: str
    symbol: str
    quantity: int
    reason: str  # SL, TP1, TP2, TRAIL, TIME, MANUAL


class BracketManager:
    def __init__(self) -> None:
        self._brackets: dict[str, Bracket] = {}  # bracket_id → Bracket
        self._position_map: dict[str, str] = {}  # position_id → bracket_id

    def create_bracket(self, position_id: str, signal: "Signal", entry_price: float) -> Bracket:
        b = Bracket(
            bracket_id=str(uuid.uuid4()),
            position_id=position_id,
            symbol=signal.symbol,
            side="BUY",
            initial_quantity=signal.quantity,
            remaining_quantity=signal.quantity,
            entry_price=entry_price,
            sl_price=signal.sl_price,
            initial_sl_price=signal.sl_price,
            tp1_price=signal.tp1_price,
            tp2_price=signal.tp2_price,
        )
        self._brackets[b.bracket_id] = b
        self._position_map[position_id] = b.bracket_id
        return b

    def get_by_position(self, position_id: str) -> Bracket | None:
        bid = self._position_map.get(position_id)
        return self._brackets.get(bid) if bid else None

    def get(self, bracket_id: str) -> Bracket | None:
        return self._brackets.get(bracket_id)

    def remove(self, bracket_id: str) -> None:
        b = self._brackets.pop(bracket_id, None)
        if b:
            self._position_map.pop(b.position_id, None)

    def get_all(self) -> list[Bracket]:
        return list(self._brackets.values())
