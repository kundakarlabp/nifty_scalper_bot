"""
Opening Range Breakout (ORB) Pro Strategy.
World-Class implementation with VWAP Filtering and Greeks Validation.
"""

from __future__ import annotations

from typing import Any, Mapping

from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    ORBProStrategyConfig,
)
from nifty_scalper_bot.utils.logging import get_logger

# Initialize structured logger
LOGGER = get_logger(__name__)


class ORBProStrategy(EliteStrategy):
    """
    Trade validated Opening Range Breakouts (ORB) with volume confirmation.
    Includes VWAP filtering to avoid false breakouts.
    """

    def __init__(self, config: ORBProStrategyConfig, indicator_engine: Any) -> None:
        """
        Initialize strategy with configuration and engine.
        
        Args:
            config: Strategy configuration dataclass.
            indicator_engine: Data provider.
        """
        # CRITICAL FIX: Correct init signature
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._orb_config = config
        self._cached_orb: dict[str, dict[str, float]] = {}

    def _evaluate_signal(self) -> EliteSignal | None:
        """
        Core Logic:
        1. Establish Range (High/Low of first X mins).
        2. Check Breakout (Price leaves range).
        3. Check VWAP (Trend Confirmation).
        4. Validate Physics (Greeks/Liquidity).
        """
        symbol = self._orb_config.symbol
        
        # 1. Fetch Indicators
        required_indicators = {
            "ltp", "volume", "avg_volume", 
            "orb_high", "orb_low", "vwap",
            "minutes_since_open"
        }
        
        indicators = self._indicator_engine.get_indicators(symbol, required_indicators)
        
        try:
            ltp = float(indicators.get("ltp") or 0)
            vol = float(indicators.get("volume") or 0)
            avg_vol = float(indicators.get("avg_volume") or 1)
            vwap = float(indicators.get("vwap") or 0)
            mins_open = float(indicators.get("minutes_since_open") or 0)
            
            # ORB Levels (Assume computed by engine or bar builder)
            orb_high = float(indicators.get("orb_high") or 0)
            orb_low = float(indicators.get("orb_low") or 0)
            
        except (ValueError, TypeError):
            return None

        if ltp == 0 or orb_high == 0:
            return None

        # 2. Time Window Check
        # ORB logic applies after the range is formed (e.g., >15 mins)
        # And usually valid only until 11:00 AM
        if mins_open < 15:
            return None # Range still building
        
        if mins_open > 120: 
            return None # Late day breakouts are often fake

        # 3. Cache Check (Ensure consistency)
        # If we haven't locked the range yet, do it now
        if symbol not in self._cached_orb:
            self._cached_orb[symbol] = {"high": orb_high, "low": orb_low}
        
        range_high = self._cached_orb[symbol]["high"]
        range_low = self._cached_orb[symbol]["low"]
        range_width = range_high - range_low

        # 4. Breakout Logic
        side: str | None = None
        
        # Bullish: Price > Range High AND Price > VWAP (Trend Filter)
        if ltp > range_high and ltp > vwap:
            side = "BUY"
            
        # Bearish: Price < Range Low AND Price < VWAP
        elif ltp < range_low and ltp < vwap:
            side = "SELL" # BaseStrategy handles PE mapping

        if not side:
            return None

        # 5. Volume Confirmation
        # Breakout volume should be > 1.2x average
        vol_ratio = vol / avg_vol if avg_vol > 0 else 0
        if vol_ratio < 1.2:
            return None # Low energy breakout

        # 6. 🛡️ SAFETY GATE (Physics Check)
        if not self.validate_option_health(symbol, side):
            LOGGER.info(f"⛔ Rejected {symbol}: Failed Greeks/Liquidity Check")
            return None

        # 7. Risk Management
        # Stop Loss: Mid-point of the range (aggressive) or Low of range (conservative)
        # We use Mid-point to keep RR healthy
        mid_point = (range_high + range_low) / 2
        
        if side == "BUY":
            stop_loss = mid_point
            tp1 = ltp + range_width
            tp2 = ltp + (range_width * 2.0)
        else:
            stop_loss = mid_point
            tp1 = ltp - range_width
            tp2 = ltp - (range_width * 2.0)

        # 8. Confidence Calculation
        # High confidence for VWAP-aligned volume breakouts
        confidence = 0.80
        if vol_ratio > 2.0: confidence += 0.10
        
        # 9. Construct Signal
        LOGGER.info(
            f"🚀 ORB Pro Signal: {symbol} {side} | Vol: {vol_ratio:.1f}x | Range: {range_width:.2f}",
            extra={
                "event": "orb_pro_signal",
                "symbol": symbol,
                "range_high": range_high,
                "range_low": range_low
            }
        )

        return EliteSignal(
            symbol=symbol,
            side=side,
            confidence=min(confidence, 0.99),
            entry_price=ltp,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            quantity=self._orb_config.quantity or 1,
            strategy_name="ORB_Pro",
            metadata={
                "orb_high": range_high,
                "orb_low": range_low,
                "vwap": vwap,
                "volume_ratio": vol_ratio
            }
        )


__all__ = ["ORBProStrategy"]
