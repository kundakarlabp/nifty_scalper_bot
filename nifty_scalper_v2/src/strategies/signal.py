"""Canonical Signal dataclass and related enums."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum

from ..config.constants import LOT_SIZE


class Direction(str, Enum):
    BUY_CALL = "BUY_CALL"
    BUY_PUT  = "BUY_PUT"


class SignalStrength(str, Enum):
    WEAK     = "WEAK"      # confidence 0.5–0.65
    MODERATE = "MODERATE"  # confidence 0.65–0.80
    STRONG   = "STRONG"    # confidence > 0.80


@dataclass(slots=True)
class Signal:
    strategy_name: str
    symbol: str         # Underlying, e.g. "NIFTY"
    direction: Direction
    confidence: float   # 0.0–1.0
    entry_price: float  # Estimated option premium at entry
    sl_price: float
    tp1_price: float
    tp2_price: float
    quantity: int       # Lots
    strike: float
    expiry: date
    option_type: str    # "CE" or "PE"
    regime: str
    timestamp: datetime
    metadata: dict

    @property
    def strength(self) -> SignalStrength:
        if self.confidence >= 0.80:
            return SignalStrength.STRONG
        elif self.confidence >= 0.65:
            return SignalStrength.MODERATE
        return SignalStrength.WEAK

    @property
    def risk_per_lot(self) -> float:
        return abs(self.entry_price - self.sl_price) * LOT_SIZE

    @property
    def r_multiple_tp1(self) -> float:
        risk = abs(self.entry_price - self.sl_price)
        if risk > 0:
            return abs(self.tp1_price - self.entry_price) / risk
        return 0.0
