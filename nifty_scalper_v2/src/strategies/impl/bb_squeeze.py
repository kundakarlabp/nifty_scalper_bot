"""Bollinger Band Squeeze Breakout strategy."""
from __future__ import annotations
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING
from ..base import Strategy
from ..signal import Direction, Signal
if TYPE_CHECKING:
    from ...indicators.engine import IndicatorSnapshot
    from ...regime.detector import MarketRegime

class BBSqueezeStrategy(Strategy):
    name = "bb_squeeze"
    def generate_signal(self, symbol, indicators: "IndicatorSnapshot", regime: "MarketRegime") -> Signal | None:
        if not self._check_cooldown() or not self._check_daily_limit():
            return None
        if not indicators.bb_squeeze:
            return None
        ltp, bb_upper, bb_lower, atr = indicators.ltp, indicators.bb_upper, indicators.bb_lower, indicators.atr_14
        if not bb_upper or not bb_lower or not atr or atr <= 0:
            return None
        bullish = ltp > bb_upper
        bearish = ltp < bb_lower
        if not bullish and not bearish:
            return None
        direction = Direction.BUY_CALL if bullish else Direction.BUY_PUT
        option_type = "CE" if bullish else "PE"
        risk = atr * 1.3
        if direction == Direction.BUY_CALL:
            sl, tp1, tp2 = max(1.0, ltp - risk), ltp + risk, ltp + risk * 1.8
        else:
            sl, tp1, tp2 = ltp + risk, max(1.0, ltp - risk), max(1.0, ltp - risk * 1.8)
        self._record_signal()
        return Signal(strategy_name=self.name, symbol=symbol, direction=direction, confidence=0.68,
            entry_price=ltp, sl_price=sl, tp1_price=tp1, tp2_price=tp2, quantity=1,
            strike=round(ltp/50)*50, expiry=date.today(), option_type=option_type,
            regime=regime.regime.value, timestamp=datetime.now(timezone.utc),
            metadata={"bb_upper": bb_upper, "bb_lower": bb_lower})
