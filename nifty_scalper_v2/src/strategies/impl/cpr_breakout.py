"""Central Pivot Range Breakout strategy."""
from __future__ import annotations
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING
from ..base import Strategy
from ..signal import Direction, Signal
if TYPE_CHECKING:
    from ...indicators.engine import IndicatorSnapshot
    from ...regime.detector import MarketRegime

class CPRBreakoutStrategy(Strategy):
    name = "cpr_breakout"
    def generate_signal(self, symbol, indicators: "IndicatorSnapshot", regime: "MarketRegime") -> Signal | None:
        if not self._check_cooldown() or not self._check_daily_limit():
            return None
        ltp, cpr_top, cpr_bot, atr = indicators.ltp, indicators.cpr_top, indicators.cpr_bottom, indicators.atr_14
        if not cpr_top or not cpr_bot or not atr or atr <= 0:
            return None
        bullish = ltp > cpr_top
        bearish = ltp < cpr_bot
        if not bullish and not bearish:
            return None
        direction = Direction.BUY_CALL if bullish else Direction.BUY_PUT
        option_type = "CE" if bullish else "PE"
        risk = atr * 1.2
        if direction == Direction.BUY_CALL:
            sl, tp1, tp2 = max(1.0, cpr_top - atr*0.5), ltp + risk, ltp + risk*1.7
        else:
            sl, tp1, tp2 = cpr_bot + atr*0.5, max(1.0, ltp - risk), max(1.0, ltp - risk*1.7)
        self._record_signal()
        return Signal(strategy_name=self.name, symbol=symbol, direction=direction, confidence=0.67,
            entry_price=ltp, sl_price=sl, tp1_price=tp1, tp2_price=tp2, quantity=1,
            strike=round(ltp/50)*50, expiry=date.today(), option_type=option_type,
            regime=regime.regime.value, timestamp=datetime.now(timezone.utc),
            metadata={"cpr_top": cpr_top, "cpr_bot": cpr_bot})
