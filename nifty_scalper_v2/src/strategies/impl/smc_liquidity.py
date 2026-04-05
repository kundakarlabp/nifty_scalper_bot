"""SMC Liquidity Sweep strategy."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import TYPE_CHECKING

from ..base import Strategy
from ..signal import Direction, Signal

if TYPE_CHECKING:
    from ...indicators.engine import IndicatorSnapshot
    from ...regime.detector import MarketRegime
    from ...config.settings import StrategySettings


class SMCLiquidityStrategy(Strategy):
    name = "smc_liquidity"

    def generate_signal(self, symbol, indicators: "IndicatorSnapshot", regime: "MarketRegime") -> Signal | None:
        if not self._check_cooldown() or not self._check_daily_limit():
            return None

        ltp = indicators.ltp
        day_high = indicators.day_high
        day_low = indicators.day_low
        atr = indicators.atr_14

        if not day_high or not day_low or not atr or atr <= 0:
            return None

        # Sweep detection: price pierced level then reversed
        bullish_sweep = ltp < (day_low + atr * 0.3) and indicators.rsi_14 and indicators.rsi_14 < 35
        bearish_sweep = ltp > (day_high - atr * 0.3) and indicators.rsi_14 and indicators.rsi_14 > 65

        if not bullish_sweep and not bearish_sweep:
            return None

        direction = Direction.BUY_CALL if bullish_sweep else Direction.BUY_PUT
        option_type = "CE" if bullish_sweep else "PE"

        risk = atr * 1.2
        if direction == Direction.BUY_CALL:
            sl = max(1.0, ltp - risk)
            tp1, tp2 = ltp + risk, ltp + risk * 1.8
        else:
            sl = ltp + risk
            tp1, tp2 = max(1.0, ltp - risk), max(1.0, ltp - risk * 1.8)

        self._record_signal()
        return Signal(
            strategy_name=self.name, symbol=symbol, direction=direction,
            confidence=0.72, entry_price=ltp, sl_price=sl, tp1_price=tp1, tp2_price=tp2,
            quantity=1, strike=round(ltp / 50) * 50, expiry=date.today(),
            option_type=option_type, regime=regime.regime.value,
            timestamp=datetime.now(timezone.utc),
            metadata={"day_high": day_high, "day_low": day_low, "atr": atr},
        )
