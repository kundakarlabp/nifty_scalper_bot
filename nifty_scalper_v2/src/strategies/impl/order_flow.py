"""Volume Imbalance / Order Flow strategy."""
from __future__ import annotations
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING
from ..base import Strategy
from ..signal import Direction, Signal
if TYPE_CHECKING:
    from ...indicators.engine import IndicatorSnapshot
    from ...regime.detector import MarketRegime

class OrderFlowStrategy(Strategy):
    name = "order_flow"
    def generate_signal(self, symbol, indicators: "IndicatorSnapshot", regime: "MarketRegime") -> Signal | None:
        if not self._check_cooldown() or not self._check_daily_limit():
            return None
        vol_ratio = indicators.volume_ratio or 0.0
        ema9, ema21, atr, ltp = indicators.ema_9, indicators.ema_21, indicators.atr_14, indicators.ltp
        if vol_ratio < 2.0 or not ema9 or not ema21 or not atr or atr <= 0:
            return None
        bullish = ema9 > ema21 and ltp > (indicators.vwap or 0)
        bearish = ema9 < ema21 and ltp < (indicators.vwap or float("inf"))
        if not bullish and not bearish:
            return None
        direction = Direction.BUY_CALL if bullish else Direction.BUY_PUT
        option_type = "CE" if bullish else "PE"
        risk = atr * 1.2
        if direction == Direction.BUY_CALL:
            sl, tp1, tp2 = max(1.0, ltp - risk), ltp + risk, ltp + risk*1.6
        else:
            sl, tp1, tp2 = ltp + risk, max(1.0, ltp - risk), max(1.0, ltp - risk*1.6)
        self._record_signal()
        return Signal(strategy_name=self.name, symbol=symbol, direction=direction, confidence=0.66,
            entry_price=ltp, sl_price=sl, tp1_price=tp1, tp2_price=tp2, quantity=1,
            strike=round(ltp/50)*50, expiry=date.today(), option_type=option_type,
            regime=regime.regime.value, timestamp=datetime.now(timezone.utc),
            metadata={"vol_ratio": vol_ratio})
