"""
Opening Range Breakout (ORB) Pro Strategy.
World-Class implementation with VWAP Filtering, Volume Confirmation, and Range Caching.
Refactored for Push-Based Architecture (Zero-Latency).
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from datetime import datetime, time

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
    MIN_BARS_REQUIRED = 5

    # ✅ OPTIMIZATION: Use slots for memory efficiency
    __slots__ = ("_orb_config", "_orb_cache")

    def __init__(self, config: ORBProStrategyConfig, indicator_engine: Any) -> None:
        """
        Initialize strategy with configuration and engine.
        
        Args:
            config: Strategy configuration dataclass.
            indicator_engine: Data provider.
        """
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._orb_config = config
        
        # Cache to store the High/Low of the opening range per symbol
        # Key: Symbol -> Value: {'high': float, 'low': float, 'date': str}
        self._orb_cache: Dict[str, Dict[str, Any]] = {}

    def get_required_indicators(self) -> set[str]:
        """
        Declare which indicators this strategy needs pre-calculated.
        The StrategyManager will inject these into _evaluate_signal.
        """
        return {
            "high",
            "low",
            "ltp",
            "volume",
            "average_volume",
            "vwap",
            "atr",
            "timestamp" # Needed to check time of day
        }

    def _evaluate_signal(
        self, 
        symbol: str, 
        indicators: Dict[str, Any], 
        current_price: float, 
        position: Any | None = None
    ) -> EliteSignal | None:
        """
        Modern Signature: Evaluates signal using injected data.
        
        Args:
            symbol: Ticker symbol.
            indicators: Dictionary containing pre-fetched indicators.
            current_price: Latest LTP.
            position: Current open position (if any).
        """
        try:
            # 1. Safe Data Extraction (Fast Path)
            current_high = float(indicators.get("high") or current_price)
            current_low = float(indicators.get("low") or current_price)
            vol = float(indicators.get("volume") or 0.0)
            avg_vol = float(indicators.get("average_volume") or 1.0)
            vwap = float(indicators.get("vwap") or 0.0)
            atr = float(indicators.get("atr") or 0.0)
            
            # Timestamp handling
            ts = indicators.get("timestamp")
            if not ts:
                return None
                
            # Convert timestamp to datetime if it's a string/int
            # Assuming 'timestamp' is a datetime object or parseable string
            # Simplified for robustness: use system time if object isn't datetime
            now = datetime.now() 
            if isinstance(ts, datetime):
                now = ts
            
            # 2. Define ORB Window (e.g., 09:15 to 09:30)
            # Default to 15-minute ORB
            orb_minutes = getattr(self._orb_config, "orb_minutes", 15)
            market_open = time(9, 15)
            
            # Calculate end time of ORB
            # (Simplified minute math)
            orb_end_minute = 15 + orb_minutes
            orb_end_time = time(9, orb_end_minute if orb_end_minute < 60 else 30)

            current_time = now.time()
            today_str = now.strftime("%Y-%m-%d")

            # 3. Logic: Range Formation Phase
            # If we are INSIDE the ORB window, we just track High/Low
            if current_time < orb_end_time:
                # Update/Create cache entry
                if symbol not in self._orb_cache or self._orb_cache[symbol].get("date") != today_str:
                    self._orb_cache[symbol] = {
                        "high": current_high,
                        "low": current_low,
                        "date": today_str,
                        "complete": False
                    }
                else:
                    # Update High/Low
                    cache = self._orb_cache[symbol]
                    cache["high"] = max(cache["high"], current_high)
                    cache["low"] = min(cache["low"], current_low)
                
                return None # No trades during formation phase

            # 4. Logic: Range Breakout Phase
            # If we are AFTER the window, check for breakout
            
            # Retrieve cached range
            cache = self._orb_cache.get(symbol)
            if not cache or cache.get("date") != today_str:
                # We missed the opening range data (bot started late?)
                return None
            
            range_high = cache["high"]
            range_low = cache["low"]
            
            # Mark range as complete
            cache["complete"] = True

            # Sanity Check: If range is too tiny (flatline data), skip
            range_width = range_high - range_low
            if range_width < (current_price * 0.001): # < 0.1% range is suspicious
                return None

            # 5. Logic: Signal Detection
            side = ""
            stop_loss = 0.0
            tp1 = 0.0
            tp2 = 0.0
            
            # Fallback ATR
            if atr == 0: atr = current_price * 0.005

            # --- Bullish Breakout ---
            # Price breaks Range High + VWAP Confirmation
            if current_price > range_high:
                # VWAP Filter: Price must be > VWAP (Trend alignment)
                if current_price < vwap:
                    return None # False breakout into resistance
                
                # Volume Filter: Breakout needs power
                if (vol / avg_vol) < 1.0:
                    return None
                
                side = "BUY"
                # Stop Loss: Midpoint of range or VWAP (whichever is closer)
                # Conservative: Low of Range
                # Aggressive: VWAP
                stop_loss = range_low
                tp1 = current_price + range_width
                tp2 = current_price + (range_width * 2.0)

            # --- Bearish Breakout ---
            # Price breaks Range Low + VWAP Confirmation
            elif current_price < range_low:
                # VWAP Filter: Price must be < VWAP
                if current_price > vwap:
                    return None # False breakdown into support
                
                if (vol / avg_vol) < 1.0:
                    return None
                
                side = "SELL"
                stop_loss = range_high
                tp1 = current_price - range_width
                tp2 = current_price - (range_width * 2.0)

            if not side:
                return None

            # 6. Confidence Scoring
            # Base 80%. Boost if volume is massive (>2x)
            confidence = 0.80
            if (vol / avg_vol) > 2.0:
                confidence += 0.10

            LOGGER.info(
                f"🚀 ORB Breakout: {symbol} {side} | Range: {range_low}-{range_high} | Vol: {(vol/avg_vol):.1f}x",
                extra={
                    "event": "orb_breakout_signal",
                    "symbol": symbol,
                    "breakout_level": range_high if side == "BUY" else range_low,
                    "vwap": vwap
                }
            )

            return EliteSignal(
                symbol=symbol,
                signal=side,
                confidence=min(confidence, 0.99),
                entry_price=current_price,
                stop_loss=stop_loss,
                target=tp1,
                quantity=self._orb_config.quantity or 1,
                strategy_name="ORB_Pro_Breakout",
                metadata={
                    "type": "Opening_Range_Breakout",
                    "range_width": round(range_width, 2),
                    "vwap_confirmation": True,
                    "orb_time": f"{orb_minutes}m"
                }
            )

        except Exception as e:
            LOGGER.error(f"ORB Strategy Error on {symbol}: {e}", exc_info=True)
            return None


__all__ = ["ORBProStrategy"]
