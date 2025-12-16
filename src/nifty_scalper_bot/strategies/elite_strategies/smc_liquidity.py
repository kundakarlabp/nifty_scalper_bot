"""
Smart Money Concepts (SMC) Liquidity Sweep Strategy.
World-Class implementation with Sweep Detection, Volume Absorption, and Greeks Validation.
"""

from __future__ import annotations

from typing import Any, Mapping

from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    SMCStrategyConfig,
)
from nifty_scalper_bot.utils.logging import get_logger

# Initialize structured logger
LOGGER = get_logger(__name__)


class SMCStrategy(EliteStrategy):
    """
    Detects Liquidity Sweeps (Stop Hunts).
    Enters on Rejection Candles where price pierces a level but closes back inside.
    """

    def __init__(self, config: SMCStrategyConfig, indicator_engine: Any) -> None:
        """
        Initialize strategy with configuration and engine.
        
        Args:
            config: Strategy configuration dataclass.
            indicator_engine: Data provider.
        """
        # CRITICAL FIX: Correct init signature
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._smc_config = config

    def _evaluate_signal(self) -> EliteSignal | None:
        """
        Core Logic:
        1. Check Extremes (Bollinger Bands / Recent Highs).
        2. Detect Sweep (High > Band but Close < Band).
        3. Check Absorption (High Volume on Rejection).
        4. Validate Physics (Greeks/Liquidity).
        """
        symbol = self._smc_config.symbol
        
        # 1. Fetch Indicators
        required_indicators = {
            "open", "high", "low", "close", "ltp",
            "bb_upper", "bb_lower", "vwap",
            "volume", "avg_volume", "atr"
        }
        
        indicators = self._indicator_engine.get_indicators(symbol, required_indicators)
        
        try:
            # Candle Data
            close = float(indicators.get("close") or 0)
            high = float(indicators.get("high") or 0)
            low = float(indicators.get("low") or 0)
            
            # Context Data
            upper = float(indicators.get("bb_upper") or 0)
            lower = float(indicators.get("bb_lower") or 0)
            vwap = float(indicators.get("vwap") or 0)
            vol = float(indicators.get("volume") or 0)
            avg_vol = float(indicators.get("avg_volume") or 1)
            atr = float(indicators.get("atr") or 0)
            
        except (ValueError, TypeError):
            return None

        if close == 0 or upper == 0:
            return None

        # 2. Sweep Detection (Turtle Soup Pattern)
        # We look for a candle that poked OUTSIDE the bands but closed INSIDE.
        side: str | None = None
        sweep_level: float = 0.0
        
        # Bearish Sweep (Liquidity Grab at Highs)
        # Price went above Upper Band, but Close is below Upper Band
        # Ideally Close is also below Open (Red Candle) for stronger signal
        if high > upper and close < upper:
            side = "SELL" # Reversal Down
            sweep_level = high
            
        # Bullish Sweep (Liquidity Grab at Lows)
        # Price went below Lower Band, but Close is above Lower Band
        elif low < lower and close > lower:
            side = "BUY" # Reversal Up
            sweep_level = low

        if not side:
            return None

        # 3. Absorption Confirmation (Volume)
        # A sweep needs effort. Volume should be significant (>1.2x Avg)
        # This confirms "Smart Money" absorbed the stops.
        vol_ratio = vol / avg_vol if avg_vol > 0 else 0
        if vol_ratio < 1.2:
            return None # Weak rejection

        # 4. Trend Context (Optional but recommended)
        # Don't fade a super strong trend. 
        # If we are selling, price should be extended far from VWAP.
        # Simple Check: Is the reversion target (VWAP) worth it?
        dist_to_vwap = abs(close - vwap)
        if dist_to_vwap < (atr * 2):
            return None # Not enough room to profit

        # 5. 🛡️ SAFETY GATE (Physics Check)
        if not self.validate_option_health(symbol, side):
            LOGGER.info(f"⛔ Rejected {symbol}: Failed Greeks/Liquidity Check")
            return None

        # 6. Risk Management (SMC Style)
        # Stop Loss: Just beyond the Sweep Wick (The Liquidity Pool)
        # Take Profit: VWAP (Liquidity Equilibrium) and Opposite Band
        
        buffer = atr * 0.2 # Tiny buffer above wick
        
        if side == "BUY":
            stop_loss = low - buffer
            tp1 = vwap
            tp2 = upper
        else:
            stop_loss = high + buffer
            tp1 = vwap
            tp2 = lower

        # 7. Confidence Calculation
        # High confidence for high volume rejections
        confidence = 0.75
        if vol_ratio > 2.0: confidence += 0.15
        
        # 8. Construct Signal
        LOGGER.info(
            f"🚀 SMC Sweep Detected: {symbol} {side} | Vol: {vol_ratio:.1f}x | Wick: {sweep_level}",
            extra={
                "event": "smc_sweep_signal",
                "symbol": symbol,
                "sweep_level": sweep_level,
                "vwap_target": vwap
            }
        )

        return EliteSignal(
            symbol=symbol,
            side=side,
            confidence=min(confidence, 0.99),
            entry_price=close,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            quantity=self._smc_config.quantity or 1,
            strategy_name="SMC_Liquidity_Pro",
            metadata={
                "sweep_type": "Bollinger_Rejection",
                "volume_ratio": vol_ratio,
                "wick_size": high - close if side == "SELL" else close - low
            }
        )


__all__ = ["SMCStrategy"]
