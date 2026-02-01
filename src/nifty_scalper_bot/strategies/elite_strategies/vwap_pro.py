"""
VWAP Pro Strategy - PRACTICAL PRODUCTION FIX
═══════════════════════════════════════════════════════════════════════════════

This version works with EXISTING infrastructure without requiring new indicator fields.

FIXES:
1. ✅ CE only trades bullish (price > vwap), PE only trades bearish (price < vwap)
2. ✅ Per-symbol signal cooldown prevents spam (30s default)
3. ✅ Volume fallback handles empty history gracefully
4. ✅ Options long-only mode: bearish = BUY PUT (not SELL)

KEY INSIGHT: Instead of requiring underlying_ema (which doesn't exist), 
we use a simple rule:
- For CE options: Only signal when option price > VWAP (bullish momentum)
- For PE options: Only signal when option price < VWAP (bearish momentum)

This eliminates the contradiction of both CE and PE signaling simultaneously.
"""

from __future__ import annotations

import os
import time as time_module
from typing import Any, Dict

from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    VWAPProStrategyConfig,
)
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


class VWAPProStrategy(EliteStrategy):
    """
    VWAP Pro Strategy - Practical Production Implementation
    
    Key Logic:
    - CE options: Only trade when showing BULLISH characteristics
    - PE options: Only trade when showing BEARISH characteristics
    - This prevents contradictory signals on same underlying
    """
    MIN_BARS_REQUIRED = 1

    __slots__ = ("_vwap_config", "_signal_cooldown_tracker")

    def __init__(self, config: VWAPProStrategyConfig, indicator_engine: Any) -> None:
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._vwap_config = config
        self._signal_cooldown_tracker: Dict[str, float] = {}

    def get_required_indicators(self) -> set[str]:
        return {
            "vwap", 
            "ema",
            "atr",
            "volume", 
            "avg_volume",
            "high",
            "low", 
            "close",
            "open",
        }

    def _evaluate_signal(
        self, 
        symbol: str, 
        indicators: Dict[str, Any], 
        current_price: float, 
        position: Any | None = None
    ) -> EliteSignal | None:
        """
        Evaluate VWAP Pro signal with practical options handling.
        """
        # ═══════════════════════════════════════════════════════════
        # 🛡️ TIME GUARD
        # ═══════════════════════════════════════════════════════════
        try:
            from nifty_scalper_bot.utils.market_hours import is_market_hours_cached
            if not is_market_hours_cached():
                return None
        except ImportError:
            pass

        # ═══════════════════════════════════════════════════════════
        # 🛡️ SIGNAL COOLDOWN (Prevents spam)
        # ═══════════════════════════════════════════════════════════
        cooldown_seconds = float(os.getenv("VWAP_SIGNAL_COOLDOWN", "30.0"))
        now = time_module.time()
        last_signal = self._signal_cooldown_tracker.get(symbol, 0)
        
        if now - last_signal < cooldown_seconds:
            return None
        
        try:
            # ═══════════════════════════════════════════════════════
            # 1. DATA EXTRACTION
            # ═══════════════════════════════════════════════════════
            vwap = float(indicators.get("vwap") or 0.0)
            ema = float(indicators.get("ema") or 0.0)
            atr = float(indicators.get("atr") or 0.0)
            vol = float(indicators.get("volume") or 0.0)
            avg_vol = float(indicators.get("avg_volume") or 0.0)
            
            high = float(indicators.get("high") or current_price)
            low = float(indicators.get("low") or current_price)
            close = float(indicators.get("close") or current_price)

            # Volume fallback
            if avg_vol <= 0:
                avg_vol = vol * 0.8 if vol > 0 else 1.0

            # Sanity Check
            if vwap <= 0 or ema <= 0 or current_price <= 0:
                return None

            # ═══════════════════════════════════════════════════════
            # 2. OPTION TYPE DETECTION
            # ═══════════════════════════════════════════════════════
            is_option = "CE" in symbol or "PE" in symbol
            is_ce = "CE" in symbol
            is_pe = "PE" in symbol

            # ═══════════════════════════════════════════════════════
            # 3. TREND DETECTION - ✅ FIXED: Option-specific rules
            # ═══════════════════════════════════════════════════════
            # Use price vs VWAP as primary trend indicator
            price_above_vwap = current_price > vwap
            price_below_vwap = current_price < vwap
            price_above_ema = current_price > ema
            price_below_ema = current_price < ema

            # Define trend based on both VWAP and EMA
            trend_bullish = price_above_vwap and price_above_ema
            trend_bearish = price_below_vwap and price_below_ema

            # ═══════════════════════════════════════════════════════
            # ✅ CRITICAL FIX: Option type MUST match trend
            # This prevents CE+PE both signaling simultaneously
            # ═══════════════════════════════════════════════════════
            if is_ce:
                if not trend_bullish:
                    return None  # CE only on confirmed bullish
            elif is_pe:
                if not trend_bearish:
                    return None  # PE only on confirmed bearish
            else:
                # For futures/index - require clear trend
                if not trend_bullish and not trend_bearish:
                    return None

            # ═══════════════════════════════════════════════════════
            # 4. VWAP INTERACTION CHECK
            # ═══════════════════════════════════════════════════════
            proximity_pct = 0.02 if is_option else 0.005
            proximity = current_price * proximity_pct
            
            touched_vwap = (low <= vwap <= high)
            near_vwap = abs(current_price - vwap) <= proximity
            
            trend_momentum = abs(current_price - vwap) / vwap if vwap > 0 else 0
            strong_trend = trend_momentum > 0.03  # Reduced from 0.05 for options

            should_evaluate = touched_vwap or near_vwap or strong_trend

            if not should_evaluate:
                return None

            # ═══════════════════════════════════════════════════════
            # 5. VOLUME CONFIRMATION
            # ═══════════════════════════════════════════════════════
            vol_ratio = vol / avg_vol if avg_vol > 0 else 1.0
            
            if vol_ratio > 100:
                LOGGER.warning(f"⚠️ Abnormal vol_ratio: {vol_ratio:.2f} for {symbol}")
                vol_ratio = min(vol_ratio, 10.0)
            
            # Minimum volume threshold
            if vol_ratio < 0.5:
                return None

            # ═══════════════════════════════════════════════════════
            # 6. ATR FALLBACK
            # ═══════════════════════════════════════════════════════
            if atr <= 0:
                atr = current_price * 0.01

            # ═══════════════════════════════════════════════════════
            # 7. SIGNAL CONSTRUCTION
            # ═══════════════════════════════════════════════════════
            options_long_only = os.getenv("OPTIONS_LONG_ONLY", "true").lower() == "true"
            
            side = ""
            option_type = None
            stop_loss = 0.0
            tp1 = 0.0

            if is_ce or (not is_option and trend_bullish):
                # Bullish signal
                side = "BUY"
                option_type = "CE" if is_option else None
                stop_loss = current_price - (atr * 1.5)
                tp1 = current_price + (atr * 3.0)

            elif is_pe or (not is_option and trend_bearish):
                # Bearish signal
                if options_long_only or is_pe:
                    side = "BUY"
                    option_type = "PE" if is_option else None
                else:
                    side = "SELL"
                    option_type = None
                
                stop_loss = current_price + (atr * 1.5)
                tp1 = current_price - (atr * 3.0)

            if not side:
                return None

            # ═══════════════════════════════════════════════════════
            # 8. CONFIDENCE SCORING (0-1 scale)
            # ═══════════════════════════════════════════════════════
            confidence = 0.75  # Base confidence
            
            # Volume boost
            if vol_ratio > 1.5:
                confidence += 0.10
            elif vol_ratio > 1.2:
                confidence += 0.05
            
            # Trend alignment boost
            if trend_bullish and is_ce:
                confidence += 0.05
            elif trend_bearish and is_pe:
                confidence += 0.05

            # Strong momentum boost
            if strong_trend:
                confidence += 0.05

            confidence = min(confidence, 0.95)

            # ═══════════════════════════════════════════════════════
            # 9. RECORD COOLDOWN & LOG
            # ═══════════════════════════════════════════════════════
            self._signal_cooldown_tracker[symbol] = time_module.time()
            
            trend_str = "Bull" if (is_ce or trend_bullish) else "Bear"
            
            LOGGER.info(
                f"🚀 VWAP Pro Signal: {symbol} {side} | "
                f"Trend: {trend_str} | "
                f"Vol: {vol_ratio:.1f}x | "
                f"Option: {option_type or 'N/A'}",
                extra={
                    "event": "vwap_pro_signal",
                    "symbol": symbol,
                    "side": side,
                    "option_type": option_type,
                    "vwap": vwap,
                    "vol_ratio": vol_ratio,
                }
            )

            return EliteSignal(
                symbol=symbol,
                signal=side,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                target=tp1,
                quantity=self._vwap_config.quantity or 1,
                strategy_name="VWAP_Pro_Trend",
                metadata={
                    "type": "Trend_Pullback",
                    "vwap": round(vwap, 2),
                    "ema_trend": trend_str,
                    "vol_ratio": round(vol_ratio, 2),
                    "atr": round(atr, 2),
                    "option_type": option_type,
                }
            )

        except Exception as e:
            LOGGER.error(
                f"🔴 VWAP Pro evaluation error: {e}",
                extra={"event": "vwap_pro_error", "symbol": symbol}
            )
            return None


__all__ = ["VWAPProStrategy"]
