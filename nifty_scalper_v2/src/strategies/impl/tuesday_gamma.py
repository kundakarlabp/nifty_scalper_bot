"""Tuesday Gamma Ramp strategy — 2 days before Thursday expiry."""
from __future__ import annotations
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING
from ..base import Strategy
from ..signal import Direction, Signal
from ...utils.time_utils import now_ist
from ...config.constants import TUESDAY_GAMMA_START, TUESDAY_WEEKDAY
if TYPE_CHECKING:
    from ...indicators.engine import IndicatorSnapshot
    from ...regime.detector import MarketRegime

class TuesdayGammaStrategy(Strategy):
    name = "tuesday_gamma"
    def generate_signal(self, symbol, indicators: "IndicatorSnapshot", regime: "MarketRegime") -> Signal | None:
        if not self._check_cooldown() or not self._check_daily_limit():
            return None
        ist_now = now_ist()
        if ist_now.weekday() != TUESDAY_WEEKDAY or ist_now.time() < TUESDAY_GAMMA_START:
            return None
        ema9, ema21, vwap, ltp, atr = (
            indicators.ema_9, indicators.ema_21, indicators.vwap,
            indicators.ltp, indicators.atr_14
        )
        if not ema9 or not ema21 or not vwap or not atr or atr <= 0:
            return None
        bullish = ema9 > ema21 and ltp > vwap
        bearish = ema9 < ema21 and ltp < vwap
        if not bullish and not bearish:
            return None
        direction = Direction.BUY_CALL if bullish else Direction.BUY_PUT
        option_type = "CE" if bullish else "PE"
        risk = atr * 1.2
        if direction == Direction.BUY_CALL:
            sl, tp1, tp2 = max(1.0, ltp - risk), ltp + risk, ltp + risk*1.7
        else:
            sl, tp1, tp2 = ltp + risk, max(1.0, ltp - risk), max(1.0, ltp - risk*1.7)
        self._record_signal()
        return Signal(strategy_name=self.name, symbol=symbol, direction=direction, confidence=0.70,
            entry_price=ltp, sl_price=sl, tp1_price=tp1, tp2_price=tp2, quantity=1,
            strike=round(ltp/50)*50, expiry=date.today(), option_type=option_type,
            regime=regime.regime.value, timestamp=datetime.now(timezone.utc),
            metadata={"weekday": ist_now.weekday(), "time": str(ist_now.time())})
