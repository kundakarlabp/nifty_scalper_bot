"""
VWAP Pro Strategy.
World-Class implementation with Trend Pullbacks, Volume Validation, and Greeks Safety.
Refactored for Push-Based Architecture (Zero-Latency).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    VWAPProStrategyConfig,
)
from nifty_scalper_bot.utils.logging import get_logger

# Initialize structured logger
LOGGER = get_logger(__name__)


class VWAPProStrategy(EliteStrategy):
    """
    Institutions trade at VWAP.
    We look for 'Trend Pullbacks': 
    1. Strong Trend (Price vs EMA).
    2. Retrace to VWAP.
    3. Bounce/Rejection at VWAP with Volume.
    """
    MIN_BARS_REQUIRED = 1

    # ✅ OPTIMIZATION: Use slots for memory efficiency
    __slots__ = ("_vwap_config",)

    def __init__(self, config: VWAPProStrategyConfig, indicator_engine: Any) -> None:
        """
        Initialize strategy with configuration and engine.
        
        Args:
            config: Strategy configuration dataclass.
            indicator_engine: Data provider.
        """
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._vwap_config = config

    def get_required_indicators(self) -> set[str]:
        """
        Declare which indicators this strategy needs pre-calculated.
        The StrategyManager will inject these into _evaluate_signal.
        """
        return {
            "vwap", 
            "ema",      # Trend Filter (usually 50 or 200 EMA)
            "atr",      # Volatility for stops
            "volume", 
            "average_volume", 
            "high",     # To detect wicks touching VWAP
            "low", 
            "close",
            "open"
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
        
        ✅ FIX: Added time guard at source to prevent signal generation
                outside market hours.
        """
        # ═══════════════════════════════════════════════════════════
        # 🛡️ FIX: EARLIEST TIME GUARD (Stop signals at source)
        # ═══════════════════════════════════════════════════════════
        try:
            from nifty_scalper_bot.utils.market_hours import is_market_hours_cached
            
            if not is_market_hours_cached():
                return None  # Don't even evaluate - market is closed
        except ImportError:
            # Fallback if module not available yet
            import os
            from datetime import datetime
            from zoneinfo import ZoneInfo
            
            if os.getenv("SESSION_ALLOW_OUT_OF_HOURS", "").lower() != "true":
                ist_now = datetime.now(ZoneInfo("Asia/Kolkata"))
                if not (9 <= ist_now.hour < 16):
                    return None
        # ═══════════════════════════════════════════════════════════
        
        try:
            # 1. Safe Data Extraction (Fast Path)
            vwap = float(indicators.get("vwap") or 0.0)
            ema = float(indicators.get("ema") or 0.0)
            atr = float(indicators.get("atr") or 0.0)
            vol = float(indicators.get("volume") or 0.0)
            avg_vol = float(indicators.get("average_volume") or 1.0)  # Avoid div/0
            
            # Candle OHLC for Interaction Checks
            high = float(indicators.get("high") or current_price)
            low = float(indicators.get("low") or current_price)
            close = float(indicators.get("close") or current_price)

            # Sanity Check: If VWAP is missing or flat, skip
            if vwap <= 0 or ema <= 0:
                return None

            # 2. Logic: Define Trend Regime
            trend_bullish = current_price > ema
            trend_bearish = current_price < ema

            if not trend_bullish and not trend_bearish:
                return None

            # 3. Logic: Detect VWAP Interaction (The Pullback)
            proximity = current_price * 0.0015
            touched_vwap = (low <= vwap <= high)
            near_vwap = abs(current_price - vwap) <= proximity

            if not (touched_vwap or near_vwap):
                return None

            # 4. Logic: Volume Confirmation
            vol_ratio = vol / avg_vol
            if vol_ratio < 0.8:
                return None

            # 5. Construct Signal
            side = ""
            stop_loss = 0.0
            tp1 = 0.0
            
            # ✅ FIX: Better ATR fallback
            if atr <= 0:
                atr = current_price * 0.01  # 1% of price as fallback

            if trend_bullish:
                if close < vwap: 
                    return None
                
                side = "BUY"
                stop_loss = vwap - (atr * 1.5)
                tp1 = current_price + (atr * 3.0)

            elif trend_bearish:
                if close > vwap:
                    return None
                
                side = "SELL"
                stop_loss = vwap + (atr * 1.5)
                tp1 = current_price - (atr * 3.0)

            # 6. Confidence Scoring
            confidence = 80.0
            if vol_ratio > 1.5:
                confidence += 10.0
            
            if trend_bullish and ema < vwap and abs(vwap - ema) < (atr * 5):
                confidence += 5.0
            elif trend_bearish and ema > vwap and abs(ema - vwap) < (atr * 5):
                confidence += 5.0

            LOGGER.info(
                f"🚀 VWAP Pro Signal: {symbol} {side} | Trend: {'Bull' if trend_bullish else 'Bear'} | Vol: {vol_ratio:.1f}x",
                extra={
                    "event": "vwap_pro_signal",
                    "symbol": symbol,
                    "vwap": vwap,
                    "ema": ema,
                    "proximity": abs(current_price - vwap)
                }
            )

            return EliteSignal(
                symbol=symbol,
                signal=side,
                confidence=min(confidence, 99.0),
                entry_price=current_price,
                stop_loss=stop_loss,
                target=tp1,
                quantity=self._vwap_config.quantity or 1,
                strategy_name="VWAP_Pro_Trend",
                metadata={
                    "type": "Trend_Pullback",
                    "vwap": vwap,
                    "ema_trend": "Bullish" if trend_bullish else "Bearish",
                    "vol_ratio": round(vol_ratio, 2),
                    "atr": round(atr, 2),  # ✅ Include ATR in metadata
                }
            )

        except Exception as e:
            LOGGER.error(
                f"🔴 VWAP Pro evaluation error: {e}",
                extra={"event": "vwap_pro_error", "symbol": symbol}
            )
            return None


__all__ = ["VWAPProStrategy"]
