"""ADX Trend Momentum strategy."""
from __future__ import annotations
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING
from ..base import Strategy
from ..signal import Direction, Signal
if TYPE_CHECKING:
    from ...indicators.engine import IndicatorSnapshot
    from ...regime.detector import MarketRegime

class TrendMomentumStrategy(Strategy):
    name = "trend_momentum"
    def generate_signal(self, symbol, indicators: "IndicatorSnapshot", regime: "MarketRegime") -> Signal | None:
        if not self._check_cooldown() or not self._check_daily_limit():
            return None
        adx, ema9, ema21, ema50, atr, ltp = (
            indicators.adx, indicators.ema_9, indicators.ema_21,
            indicators.ema_50, indicators.atr_14, indicators.ltp
        )
        if not adx or adx < 25 or not ema9 or not ema21 or not ema50 or not atr or atr <= 0:
            return None
        bullish = ema9 > ema21 > ema50 and ltp > ema9
        bearish = ema9 < ema21 < ema50 and ltp < ema9
        if not bullish and not bearish:
            return None
        direction = Direction.BUY_CALL if bullish else Direction.BUY_PUT
        option_type = "CE" if bullish else "PE"
        confidence = 0.72 if adx >= 35 else 0.68
        risk = atr * 1.5
        if direction == Direction.BUY_CALL:
            sl, tp1, tp2 = max(1.0, ltp - risk), ltp + risk, ltp + risk*1.8
        else:
            sl, tp1, tp2 = ltp + risk, max(1.0, ltp - risk), max(1.0, ltp - risk*1.8)
        self._record_signal()
        return Signal(strategy_name=self.name, symbol=symbol, direction=direction, confidence=confidence,
            entry_price=ltp, sl_price=sl, tp1_price=tp1, tp2_price=tp2, quantity=1,
            strike=round(ltp/50)*50, expiry=date.today(), option_type=option_type,
            regime=regime.regime.value, timestamp=datetime.now(timezone.utc),
            metadata={"adx": adx, "ema9": ema9, "ema21": ema21, "ema50": ema50})
