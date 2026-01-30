"""
VWAP Pro Strategy - PRODUCTION FIXED VERSION
═══════════════════════════════════════════════════════════════════════════════

FIXES APPLIED:
1. ✅ Volume indicator: "average_volume" → "avg_volume"
2. ✅ SELL signal: In options long-only mode, bearish = BUY PUT (not SELL PUT)
3. ✅ Confidence scale: 80.0 → 0.80 (normalized 0-1)
4. ✅ Added option_type to metadata for strike selector guidance

"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

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
    Institutions trade at VWAP.
    We look for 'Trend Pullbacks': 
    1. Strong Trend (Price vs EMA).
    2. Retrace to VWAP.
    3. Bounce/Rejection at VWAP with Volume.
    
    ✅ PRODUCTION FIX: Proper options signal handling
    """
    MIN_BARS_REQUIRED = 1

    __slots__ = ("_vwap_config", "_last_signal_time", "_signal_cooldown")

    def __init__(self, config: VWAPProStrategyConfig, indicator_engine: Any) -> None:
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._vwap_config = config
        self._last_signal_time: dict[str, float] = {}
        self._signal_cooldown = float(os.getenv("VWAP_SIGNAL_COOLDOWN", "30.0"))

    def get_required_indicators(self) -> set[str]:
        return {
            "vwap", 
            "ema",
            "atr",
            "volume", 
            "avg_volume",  # ✅ FIXED: Was "average_volume"
            "high",
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
        Evaluate VWAP Pro signal with proper options handling.
        
        ✅ PRODUCTION FIXES:
        - Volume indicator name corrected
        - Bearish signals properly handled for options buying
        - Confidence normalized to 0-1 scale
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
        
        try:
            # ═══════════════════════════════════════════════════════
            # 1. DATA EXTRACTION
            # ═══════════════════════════════════════════════════════
            vwap = float(indicators.get("vwap") or 0.0)
            ema = float(indicators.get("ema") or 0.0)
            atr = float(indicators.get("atr") or 0.0)
            vol = float(indicators.get("volume") or 0.0)
            
            # ✅ FIXED: Use "avg_volume" not "average_volume"
            avg_vol = float(indicators.get("avg_volume") or 0.0)
            
            # Fallback if avg_volume is still 0 (defensive)
            if avg_vol <= 0:
                avg_vol = vol * 0.8 if vol > 0 else 1.0
            
            high = float(indicators.get("high") or current_price)
            low = float(indicators.get("low") or current_price)
            close = float(indicators.get("close") or current_price)

            # Sanity Check
            if vwap <= 0 or ema <= 0 or current_price <= 0:
                return None

            # ═══════════════════════════════════════════════════════
            # 2. TREND REGIME
            # ═══════════════════════════════════════════════════════
            trend_bullish = current_price > ema
            trend_bearish = current_price < ema

            if not trend_bullish and not trend_bearish:
                return None

            # ═══════════════════════════════════════════════════════
            # 3. VWAP INTERACTION CHECK
            # ═══════════════════════════════════════════════════════
            is_option = "CE" in symbol or "PE" in symbol
            proximity_pct = 0.02 if is_option else 0.005  # ✅ Relaxed for options
            proximity = current_price * proximity_pct
            
            touched_vwap = (low <= vwap <= high)
            near_vwap = abs(current_price - vwap) <= proximity
            
            trend_momentum = abs(current_price - vwap) / vwap
            strong_trend = trend_momentum > 0.05

            should_evaluate = touched_vwap or near_vwap or (strong_trend and (trend_bullish or trend_bearish))

            if not should_evaluate:
                return None

            # ═══════════════════════════════════════════════════════
            # 4. VOLUME CONFIRMATION
            # ═══════════════════════════════════════════════════════
            vol_ratio = vol / avg_vol if avg_vol > 0 else 1.0
            
            # ✅ SANITY CHECK: vol_ratio should be reasonable (0.1 to 100)
            if vol_ratio > 100:
                LOGGER.warning(
                    f"⚠️ Abnormal vol_ratio: {vol_ratio:.2f} for {symbol} | "
                    f"vol={vol}, avg_vol={avg_vol}"
                )
                vol_ratio = min(vol_ratio, 10.0)  # Cap at 10x
            
            if vol_ratio < 0.8:
                return None

            # ═══════════════════════════════════════════════════════
            # 5. ATR FALLBACK
            # ═══════════════════════════════════════════════════════
            if atr <= 0:
                atr = current_price * 0.01

            # ═══════════════════════════════════════════════════════
            # 6. SIGNAL CONSTRUCTION (✅ PRODUCTION FIX)
            # ═══════════════════════════════════════════════════════
            options_long_only = os.getenv("OPTIONS_LONG_ONLY", "true").lower() == "true"
            
            side = ""
            option_type = None
            stop_loss = 0.0
            tp1 = 0.0

            if trend_bullish:
                if close < vwap: 
                    return None
                
                side = "BUY"
                option_type = "CE"  # Bullish = Call
                stop_loss = vwap - (atr * 1.5)
                tp1 = current_price + (atr * 3.0)

            elif trend_bearish:
                if close > vwap:
                    return None
                
                # ═══════════════════════════════════════════════════
                # ✅ CRITICAL FIX: Options Long-Only Mode
                # In options buying strategy:
                # - Bearish view = BUY PUT (not SELL PUT)
                # - SELL would mean writing/shorting the option (unlimited risk!)
                # ═══════════════════════════════════════════════════
                if options_long_only:
                    side = "BUY"      # BUY the PUT option
                    option_type = "PE"  # Bearish = Put
                else:
                    side = "SELL"     # Only for futures/short-selling mode
                    option_type = None
                
                stop_loss = vwap + (atr * 1.5)
                tp1 = current_price - (atr * 3.0)

            if not side:
                return None

            # ═══════════════════════════════════════════════════════
            # 7. CONFIDENCE SCORING (✅ NORMALIZED 0-1)
            # ═══════════════════════════════════════════════════════
            confidence = 0.80  # ✅ FIXED: Was 80.0
            
            if vol_ratio > 1.5:
                confidence += 0.10  # ✅ FIXED: Was 10.0
            
            if trend_bullish and ema < vwap and abs(vwap - ema) < (atr * 5):
                confidence += 0.05  # ✅ FIXED: Was 5.0
            elif trend_bearish and ema > vwap and abs(ema - vwap) < (atr * 5):
                confidence += 0.05  # ✅ FIXED: Was 5.0

            # ═══════════════════════════════════════════════════════
            # 8. LOG AND RETURN
            # ═══════════════════════════════════════════════════════
            LOGGER.info(
                f"🚀 VWAP Pro Signal: {symbol} {side} | "
                f"Trend: {'Bull' if trend_bullish else 'Bear'} | "
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
                confidence=min(confidence, 0.99),  # ✅ FIXED: Was 99.0
                entry_price=current_price,
                stop_loss=stop_loss,
                target=tp1,
                quantity=self._vwap_config.quantity or 1,
                strategy_name="VWAP_Pro_Trend",
                metadata={
                    "type": "Trend_Pullback",
                    "vwap": round(vwap, 2),
                    "ema_trend": "Bullish" if trend_bullish else "Bearish",
                    "vol_ratio": round(vol_ratio, 2),
                    "atr": round(atr, 2),
                    "option_type": option_type,  # ✅ NEW: Guide strike selector
                }
            )

        except Exception as e:
            LOGGER.error(
                f"🔴 VWAP Pro evaluation error: {e}",
                extra={"event": "vwap_pro_error", "symbol": symbol}
            )
            return None


__all__ = ["VWAPProStrategy"]
