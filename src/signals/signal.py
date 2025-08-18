from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Signal:
    """A typed dataclass to represent a trading signal."""
    signal: str  # "BUY" or "SELL"
    score: int
    confidence: float
    entry_price: float
    stop_loss: float
    target: float
    reasons: List[str]
    market_volatility: Optional[float] = None
    hash: Optional[str] = None
