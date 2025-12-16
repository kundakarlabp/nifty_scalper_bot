"""
CPR Breakout Strategy.
World-Class implementation with Greeks validation, NR7 Detection, and ATR Risk.
"""

from __future__ import annotations

from typing import Any, Mapping

from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    CPRBreakoutStrategyConfig,
)
from nifty_scalper_bot.utils.logging import get_logger

# Initialize structured logger
LOGGER = get_logger(__name__)


class CPRBreakoutStrategy(EliteStrategy):
    """
    Engage when Opening Range (ORB) or NR7 compression resolves with a volume-backed break.
    """

    def __init__(self, config: CPRBreakoutStrategyConfig, indicator_engine: Any) -> None:
        """
        Initialize strategy with configuration and engine.
        
        Args:
            config: Strategy configuration dataclass.
            indicator_engine: Data provider.
        """
        # CRITICAL FIX: Removed 'name' arg, added 'indicator_engine'
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._cpr_config = config

    def _evaluate_signal(self) -> EliteSignal | None:
        """
        Core Logic:
        1. Check Compression (NR7 / Tight Orbit).
        2. Check Breakout (Price > Range High/Low).
        3. Check Volume (Expansion > Average).
        4. Validate Physics (Greeks/Liquidity).
        """
        symbol = self._cpr_config.symbol
        
        # 1. Fetch Indicators
        # We need Open/High/Low/Close, Volume, ATR, and Greeks
        required_indicators = {
            "open", "high", "low", "close", "ltp",
            "volume", "avg_volume", "atr",
            "orb_high", "orb_low"
        }
        
        indicators = self._indicator_engine.get_indicators(symbol, required_indicators)
        
        try:
            ltp = float(indicators.get("ltp") or 0)
            orb_high = float(indicators.get("orb_high") or 0)
            orb_low = float(indicators.get("orb_low") or 0)
            vol = float(indicators.get("volume") or 0)
            avg_vol = float(indicators.get("avg_volume") or 1)
            atr = float(indicators.get("atr") or 0)
            
            # Helper: Check for NR7 (Narrowest range in 7 bars) if available
            # Assuming 'nr7' boolean is computed by engine, else default to False
            is_nr7 = bool(indicators.get("nr7", False))
            
        except (ValueError, TypeError):
            return None # Data incomplete

        if ltp == 0 or orb_high == 0:
            return None

        # 2. Logic: Breakout Detection
        # We look for price breaking the Opening Range (ORB) OR a CPR/NR7 level
        side: str | None = None
        
        buffer = atr * 0.1 # Small buffer to avoid fakeouts
        
        # Bullish Breakout
        if ltp > (orb_high + buffer):
            side = "BUY"
            
        # Bearish Breakout
        elif ltp < (orb_low - buffer):
            side = "SELL" # For Options, BaseStrategy handles PE mapping if needed

        if not side:
            return None

        # 3. Volume Confirmation
        # Breakouts need power. Vol > 1.0x Avg Vol
        vol_ratio = vol / avg_vol if avg_vol > 0 else 0
        if vol_ratio < 1.0:
            return None # Weak breakout

        # 4. Compression Bonus (NR7)
        # If breakout follows an NR7 day/candle, it's explosive.
        confidence = 0.75
        if is_nr7:
            confidence += 0.15 # Boost confidence for NR7 expansion

        # 5. 🛡️ SAFETY GATE (Physics Check)
        # Critical: Don't buy breakout options if they are illiquid or deep OTM
        if not self.validate_option_health(symbol, side):
            LOGGER.info(f"⛔ Rejected {symbol}: Failed Greeks/Liquidity Check")
            return None

        # 6. Risk Management (ATR Based)
        # Stop Loss: For Buy, below breakout point - 1 ATR
        # Take Profit: 1.5 ATR and 2.5 ATR
        if atr == 0: atr = ltp * 0.01 # Fallback 1%
        
        if side == "BUY":
            # SL is slightly below the breakout level (Orbit High) to survive retests
            stop_loss = orb_high - (atr * 0.5) 
            tp1 = ltp + (atr * 2.0)
            tp2 = ltp + (atr * 4.0)
        else:
            stop_loss = orb_low + (atr * 0.5)
            tp1 = ltp - (atr * 2.0)
            tp2 = ltp - (atr * 4.0)

        # 7. Construct Signal
        LOGGER.info(
            f"🚀 CPR Breakout: {symbol} {side} | Vol: {vol_ratio:.1f}x | NR7: {is_nr7}",
            extra={
                "event": "cpr_breakout_signal",
                "symbol": symbol,
                "orb_range": f"{orb_low}-{orb_high}",
                "atr": atr
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
            quantity=self._cpr_config.quantity or 1,
            strategy_name="CPR_Breakout_Pro",
            metadata={
                "orb_high": orb_high,
                "orb_low": orb_low,
                "volume_ratio": vol_ratio,
                "is_nr7": is_nr7,
                "atr": atr
            }
        )


__all__ = ["CPRBreakoutStrategy"]
