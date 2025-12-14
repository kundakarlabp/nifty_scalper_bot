"""Volatility-adaptive trailing stop using ATR with regime detection."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.execution.trailing_stop import TrailingSpec, TrailingStopController

log = get_logger(__name__)

@dataclass(slots=True)
class ATRSnapshot:
    """Current ATR state for a symbol."""
    current_atr: float
    avg_atr_20: float  # 20-bar SMA of ATR
    atr_ratio: float = 0.0  # current_atr / avg_atr_20
    timestamp: float = field(default_factory=time.time)
    
    def is_volatile(self) -> bool:
        """Return True if volatility above normal (>1.3x average)."""
        return self.atr_ratio > 1.3

    def is_calm(self) -> bool:
        """Return True if volatility below normal (<0.7x average)."""
        return self.atr_ratio < 0.7
    
    def get_trail_multiplier(self) -> float:
        """Return SL trail adjustment multiplier (0.8x to 1.3x)."""
        if self.is_volatile():
            return 1.3  # Widen trail in high volatility to avoid whipsaw
        elif self.is_calm():
            return 0.8  # Tighten trail in calm markets to lock profit
        return 1.0

class AdaptiveTrailingController(TrailingStopController):
    """Volatility-aware trailing SL controller."""
    
    def __init__(
        self,
        *,
        symbol: str,
        side: str,
        entry: float,
        sl_order_id: str,
        variety: str,
        spec: TrailingSpec,
        get_ltp: Callable[[str], Optional[float]],
        modify_order: Callable[[str, str, Optional[int], Optional[float]], dict],
        get_atr: Callable[[str], Optional[ATRSnapshot]],
        journal=None,
    ) -> None:
        super().__init__(
            symbol=symbol,
            side=side,
            entry=entry,
            sl_order_id=sl_order_id,
            variety=variety,
            spec=spec,
            get_ltp=get_ltp,
            modify_order=modify_order,
            journal=journal,
        )
        self._get_atr = get_atr
        self._last_atr_ratio = 1.0
        
        log.info(
            f"✅ AdaptiveTrailingController initialized for {symbol}",
            extra={"symbol": symbol, "side": side}
        )

    def _compute_trigger(self, ltp: float) -> float | None:
        """Compute trigger with ATR scaling."""
        atr_snap = self._get_atr(self.symbol)
        
        # Fallback to static logic if ATR missing
        if atr_snap is None:
            return super()._compute_trigger(ltp)

        self._last_atr_ratio = atr_snap.atr_ratio
        
        # Dynamic calculation
        base_trail = float(self.spec.trail_by)
        multiplier = atr_snap.get_trail_multiplier()
        adaptive_trail = base_trail * multiplier
        
        min_gap = float(self.spec.min_gap)
        
        if self.side == "LONG":
            # Trail below price
            # High Watermark Logic: max(current_trail_level, historical_max)
            anchor = max(ltp - adaptive_trail, self.entry - adaptive_trail)
            if self._last_trigger is not None:
                anchor = max(anchor, self._last_trigger)
            # Ensure we don't violate min_gap (broker limit)
            trigger = min(anchor, ltp - min_gap)
            
        else:  # SHORT
            # Trail above price
            anchor = min(ltp + adaptive_trail, self.entry + adaptive_trail)
            if self._last_trigger is not None:
                anchor = min(anchor, self._last_trigger)
            trigger = max(anchor, ltp + min_gap)
            
        return float(trigger)
