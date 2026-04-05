"""Straddle Theta / IV mean-reversion strategy."""
from __future__ import annotations
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING
from ..base import Strategy
from ..signal import Direction, Signal
from ...regime.detector import RegimeType
if TYPE_CHECKING:
    from ...indicators.engine import IndicatorSnapshot
    from ...regime.detector import MarketRegime

class StraddleThetaStrategy(Strategy):
    name = "straddle_theta"
    def generate_signal(self, symbol, indicators: "IndicatorSnapshot", regime: "MarketRegime") -> Signal | None:
        if not self._check_cooldown() or not self._check_daily_limit():
            return None
        if regime.regime not in (RegimeType.RANGE, RegimeType.CALM):
            return None
        iv, ltp, atr = indicators.iv, indicators.ltp, indicators.atr_14
        if not iv or iv < 0.20 or not atr or atr <= 0:
            return None
        # Direction from delta
        delta = indicators.delta or 0.0
        direction = Direction.BUY_CALL if delta >= 0 else Direction.BUY_PUT
        option_type = "CE" if direction == Direction.BUY_CALL else "PE"
        risk = atr * 1.0
        if direction == Direction.BUY_CALL:
            sl, tp1, tp2 = max(1.0, ltp - risk), ltp + risk, ltp + risk*1.4
        else:
            sl, tp1, tp2 = ltp + risk, max(1.0, ltp - risk), max(1.0, ltp - risk*1.4)
        self._record_signal()
        return Signal(strategy_name=self.name, symbol=symbol, direction=direction, confidence=0.63,
            entry_price=ltp, sl_price=sl, tp1_price=tp1, tp2_price=tp2, quantity=1,
            strike=round(ltp/50)*50, expiry=date.today(), option_type=option_type,
            regime=regime.regime.value, timestamp=datetime.now(timezone.utc),
            metadata={"iv": iv})
