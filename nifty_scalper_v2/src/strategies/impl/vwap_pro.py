"""VWAP Pro strategy — production-hardened v2."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import TYPE_CHECKING

from ..base import Strategy
from ..signal import Direction, Signal
from ...options.universe import OptionUniverse

if TYPE_CHECKING:
    from ...indicators.engine import IndicatorSnapshot
    from ...regime.detector import MarketRegime
    from ...config.settings import StrategySettings


class VWAPProStrategy(Strategy):
    """BUY_CALL/BUY_PUT based on VWAP cross + volume + RSI filters."""

    name = "vwap_pro"

    def __init__(self, settings: "StrategySettings", expiry_resolver: OptionUniverse | None = None) -> None:
        super().__init__(settings)
        self._expiry_resolver = expiry_resolver

    def generate_signal(
        self, symbol: str, indicators: "IndicatorSnapshot", regime: "MarketRegime"
    ) -> Signal | None:
        if not self._check_cooldown() or not self._check_daily_limit():
            return None

        ltp = indicators.ltp
        vwap = indicators.vwap
        rsi = indicators.rsi_14
        atr = indicators.atr_14
        vol_ratio = indicators.volume_ratio or 0.0
        adx = indicators.adx or 0.0

        if not vwap or vwap <= 0 or not atr or atr <= 0:
            return None
        if rsi is None:
            return None

        # VWAP cross signals
        bullish = ltp > vwap and 40 <= rsi <= 65 and vol_ratio >= 0.9
        bearish = ltp < vwap and 35 <= rsi <= 60 and vol_ratio >= 0.9

        if not bullish and not bearish:
            return None

        direction = Direction.BUY_CALL if bullish else Direction.BUY_PUT
        option_type = "CE" if bullish else "PE"

        # Confidence
        confidence = 0.65
        if vol_ratio >= 1.2:
            confidence += 0.08
        if adx >= 20:
            confidence += 0.07

        # ATR-based SL/TP
        risk = atr * 1.5
        if direction == Direction.BUY_CALL:
            sl = max(1.0, ltp - risk)
            tp1 = ltp + risk * 1.0
            tp2 = ltp + risk * (1.8 if regime.regime.value == "TREND" else 1.4)
        else:
            sl = ltp + risk
            tp1 = max(1.0, ltp - risk * 1.0)
            tp2 = max(1.0, ltp - risk * (1.8 if regime.regime.value == "TREND" else 1.4))

        expiry = self._expiry_resolver.get_weekly_expiry() if self._expiry_resolver else date.today()
        strike = round(ltp / 50) * 50  # ATM approximation

        self._record_signal()
        return Signal(
            strategy_name=self.name,
            symbol=symbol,
            direction=direction,
            confidence=min(0.95, confidence),
            entry_price=ltp,
            sl_price=sl,
            tp1_price=tp1,
            tp2_price=tp2,
            quantity=1,
            strike=strike,
            expiry=expiry,
            option_type=option_type,
            regime=regime.regime.value,
            timestamp=datetime.now(timezone.utc),
            metadata={"vwap": vwap, "rsi": rsi, "vol_ratio": vol_ratio, "adx": adx},
        )
