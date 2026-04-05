"""Opening Range Breakout (15-min) strategy."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import TYPE_CHECKING

from ..base import Strategy
from ..signal import Direction, Signal
from ...utils.time_utils import now_ist
from ...config.constants import ORB_CLOSE_TIME, ORB_VALID_UNTIL

if TYPE_CHECKING:
    from ...indicators.engine import IndicatorSnapshot
    from ...regime.detector import MarketRegime


class ORBProStrategy(Strategy):
    name = "orb_pro"

    def __init__(self, settings) -> None:
        super().__init__(settings)
        self._orb_triggered: dict[str, str] = {}  # symbol → direction triggered

    def generate_signal(self, symbol, indicators: "IndicatorSnapshot", regime: "MarketRegime") -> Signal | None:
        if not self._check_cooldown() or not self._check_daily_limit():
            return None

        ist_now = now_ist().time()
        # ORB only valid after range forms and before 13:00
        if ist_now < ORB_CLOSE_TIME or ist_now > ORB_VALID_UNTIL:
            return None

        orb_high = indicators.opening_range_high
        orb_low = indicators.opening_range_low
        ltp = indicators.ltp
        vol_ratio = indicators.volume_ratio or 0.0
        atr = indicators.atr_14

        if not orb_high or not orb_low or not atr or atr <= 0:
            return None

        # Check if already triggered for this direction
        triggered = self._orb_triggered.get(symbol, "")

        bullish = ltp > orb_high and triggered != "BULL"
        bearish = ltp < orb_low and triggered != "BEAR"

        if not bullish and not bearish:
            return None

        direction = Direction.BUY_CALL if bullish else Direction.BUY_PUT
        option_type = "CE" if bullish else "PE"
        confidence = 0.75 if vol_ratio >= 1.5 else 0.65

        risk = atr * 1.5
        if direction == Direction.BUY_CALL:
            sl = max(1.0, orb_high - atr * 0.5)
            tp1, tp2 = ltp + risk, ltp + risk * 2.0
        else:
            sl = orb_low + atr * 0.5
            tp1, tp2 = max(1.0, ltp - risk), max(1.0, ltp - risk * 2.0)

        self._orb_triggered[symbol] = "BULL" if bullish else "BEAR"
        self._record_signal()
        return Signal(
            strategy_name=self.name, symbol=symbol, direction=direction,
            confidence=confidence, entry_price=ltp, sl_price=sl, tp1_price=tp1, tp2_price=tp2,
            quantity=1, strike=round(ltp / 50) * 50, expiry=date.today(),
            option_type=option_type, regime=regime.regime.value,
            timestamp=datetime.now(timezone.utc),
            metadata={"orb_high": orb_high, "orb_low": orb_low, "vol_ratio": vol_ratio},
        )

    def reset_daily(self) -> None:
        super().reset_daily()
        self._orb_triggered.clear()
