"""Gamma Scalping strategy — near-expiry gamma acceleration."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import TYPE_CHECKING

from ..base import Strategy
from ..signal import Direction, Signal

if TYPE_CHECKING:
    from ...indicators.engine import IndicatorSnapshot
    from ...regime.detector import MarketRegime


class GammaScalpingStrategy(Strategy):
    name = "gamma_scalping"

    def generate_signal(self, symbol, indicators: "IndicatorSnapshot", regime: "MarketRegime") -> Signal | None:
        if not self._check_cooldown() or not self._check_daily_limit():
            return None

        gamma = indicators.gamma
        delta = indicators.delta
        atr = indicators.atr_14
        ltp = indicators.ltp
        ema9 = indicators.ema_9
        ema21 = indicators.ema_21

        if gamma is None or delta is None or atr is None or atr <= 0:
            return None
        if not (gamma > 0.002 and 0.3 <= abs(delta) <= 0.65):
            return None

        if ema9 is None or ema21 is None:
            return None

        bullish = ema9 > ema21
        direction = Direction.BUY_CALL if bullish else Direction.BUY_PUT
        option_type = "CE" if bullish else "PE"
        risk = atr

        if direction == Direction.BUY_CALL:
            sl = max(1.0, ltp - risk)
            tp1, tp2 = ltp + risk, ltp + risk * 1.5
        else:
            sl = ltp + risk
            tp1, tp2 = max(1.0, ltp - risk), max(1.0, ltp - risk * 1.5)

        self._record_signal()
        return Signal(
            strategy_name=self.name, symbol=symbol, direction=direction,
            confidence=0.70, entry_price=ltp, sl_price=sl, tp1_price=tp1, tp2_price=tp2,
            quantity=1, strike=round(ltp / 50) * 50, expiry=date.today(),
            option_type=option_type, regime=regime.regime.value,
            timestamp=datetime.now(timezone.utc),
            metadata={"gamma": gamma, "delta": delta},
        )
