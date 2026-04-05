"""OI Max Pain attraction strategy."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import TYPE_CHECKING

from ..base import Strategy
from ..signal import Direction, Signal
from ...utils.time_utils import now_ist
from ...config.constants import OI_MIN_BUILD_TIME

if TYPE_CHECKING:
    from ...indicators.engine import IndicatorSnapshot
    from ...regime.detector import MarketRegime


class OIMaxPainStrategy(Strategy):
    name = "oi_max_pain"

    def generate_signal(self, symbol, indicators: "IndicatorSnapshot", regime: "MarketRegime") -> Signal | None:
        if not self._check_cooldown() or not self._check_daily_limit():
            return None

        # Only valid after 11:00 IST
        if now_ist().time() < OI_MIN_BUILD_TIME:
            return None

        max_pain = indicators.max_pain_strike
        ltp = indicators.ltp
        atr = indicators.atr_14

        if max_pain is None or atr is None or atr <= 0:
            return None

        distance = abs(ltp - max_pain)
        if distance < 100:  # Too close to max pain
            return None

        # Price attracted toward max pain
        direction = Direction.BUY_PUT if ltp > max_pain else Direction.BUY_CALL
        option_type = "PE" if ltp > max_pain else "CE"
        risk = atr * 1.2

        if direction == Direction.BUY_CALL:
            sl = max(1.0, ltp - risk)
            tp1, tp2 = ltp + risk, ltp + risk * 1.4
        else:
            sl = ltp + risk
            tp1, tp2 = max(1.0, ltp - risk), max(1.0, ltp - risk * 1.4)

        self._record_signal()
        return Signal(
            strategy_name=self.name, symbol=symbol, direction=direction,
            confidence=0.65, entry_price=ltp, sl_price=sl, tp1_price=tp1, tp2_price=tp2,
            quantity=1, strike=round(ltp / 50) * 50, expiry=date.today(),
            option_type=option_type, regime=regime.regime.value,
            timestamp=datetime.now(timezone.utc),
            metadata={"max_pain": max_pain, "distance": distance},
        )
