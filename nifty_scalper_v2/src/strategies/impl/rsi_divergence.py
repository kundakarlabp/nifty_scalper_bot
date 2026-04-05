"""RSI Divergence strategy."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import TYPE_CHECKING

from ..base import Strategy
from ..signal import Direction, Signal

if TYPE_CHECKING:
    from ...indicators.engine import IndicatorSnapshot
    from ...regime.detector import MarketRegime


class RSIDivergenceStrategy(Strategy):
    name = "rsi_divergence"

    def generate_signal(self, symbol, indicators: "IndicatorSnapshot", regime: "MarketRegime") -> Signal | None:
        if not self._check_cooldown() or not self._check_daily_limit():
            return None

        rsi = indicators.rsi_14
        atr = indicators.atr_14
        ltp = indicators.ltp

        if rsi is None or atr is None or atr <= 0:
            return None

        # Bullish: oversold RSI bouncing
        bullish_div = rsi < 32 and (indicators.macd_hist or 0) > 0
        # Bearish: overbought RSI topping
        bearish_div = rsi > 68 and (indicators.macd_hist or 0) < 0

        if not bullish_div and not bearish_div:
            return None

        direction = Direction.BUY_CALL if bullish_div else Direction.BUY_PUT
        option_type = "CE" if bullish_div else "PE"
        risk = atr * 1.3

        if direction == Direction.BUY_CALL:
            sl = max(1.0, ltp - risk)
            tp1, tp2 = ltp + risk, ltp + risk * 1.6
        else:
            sl = ltp + risk
            tp1, tp2 = max(1.0, ltp - risk), max(1.0, ltp - risk * 1.6)

        self._record_signal()
        return Signal(
            strategy_name=self.name, symbol=symbol, direction=direction,
            confidence=0.68, entry_price=ltp, sl_price=sl, tp1_price=tp1, tp2_price=tp2,
            quantity=1, strike=round(ltp / 50) * 50, expiry=date.today(),
            option_type=option_type, regime=regime.regime.value,
            timestamp=datetime.now(timezone.utc),
            metadata={"rsi": rsi, "macd_hist": indicators.macd_hist},
        )
