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
        
        Args:
            symbol: Ticker symbol.
            indicators: Dictionary containing pre-fetched indicators.
            current_price: Latest LTP.
            position: Current open position (if any).
        """
        try:
            # 1. Safe Data Extraction (Fast Path)
            vwap = float(indicators.get("vwap") or 0.0)
            ema = float(indicators.get("ema") or 0.0)
            atr = float(indicators.get("atr") or 0.0)
            vol = float(indicators.get("volume") or 0.0)
            avg_vol = float(indicators.get("average_volume") or 1.0) # Avoid div/0
            
            # Candle OHLC for Interaction Checks
            high = float(indicators.get("high") or current_price)
            low = float(indicators.get("low") or current_price)
            close = float(indicators.get("close") or current_price)

            # Sanity Check: If VWAP is missing or flat, skip
            if vwap <= 0 or ema <= 0:
                return None

            # 2. Logic: Define Trend Regime
            # Bullish Trend: Price > EMA
            # Bearish Trend: Price < EMA
            trend_bullish = current_price > ema
            trend_bearish = current_price < ema

            if not trend_bullish and not trend_bearish:
                # Price is exactly at EMA (Rare/Choppy)
                return None

            # 3. Logic: Detect VWAP Interaction (The Pullback)
            # We want to enter when price touches or gets very close to VWAP
            # Proximity threshold: 0.15% of price
            proximity = current_price * 0.0015
            
            # Check if current candle wick touched VWAP
            touched_vwap = (low <= vwap <= high)
            
            # Check if price is within "Magnet Zone" of VWAP
            near_vwap = abs(current_price - vwap) <= proximity

            if not (touched_vwap or near_vwap):
                return None

            # 4. Logic: Volume Confirmation
            # We want institutional activity at VWAP.
            # Volume should be at least average or higher to confirm support/resistance.
            vol_ratio = vol / avg_vol
            if vol_ratio < 0.8: # Allow slightly below avg, but not dead silent
                return None

            # 5. Construct Signal
            side = ""
            stop_loss = 0.0
            tp1 = 0.0
            tp2 = 0.0
            
            # Fallback ATR if missing
            if atr == 0: atr = current_price * 0.005 

            if trend_bullish:
                # Setup: Uptrend -> Pullback to VWAP -> Buy
                # Only buy if price is currently ABOVE VWAP (Bounce confirmation)
                if close < vwap: 
                    return None # Failed support, broke below VWAP
                
                side = "BUY"
                stop_loss = vwap - (atr * 1.5) # Stop below VWAP support
                tp1 = current_price + (atr * 3.0)
                tp2 = current_price + (atr * 6.0)

            elif trend_bearish:
                # Setup: Downtrend -> Rally to VWAP -> Sell
                # Only sell if price is currently BELOW VWAP (Rejection confirmation)
                if close > vwap:
                    return None # Failed resistance, broke above VWAP
                
                side = "SELL"
                stop_loss = vwap + (atr * 1.5) # Stop above VWAP resistance
                tp1 = current_price - (atr * 3.0)
                tp2 = current_price - (atr * 6.0)

            # 6. Confidence Scoring
            # Base 80%. Boost if volume is high (strong hand defense)
            confidence = 80.0
            if vol_ratio > 1.5:
                confidence += 10.0
            
            # Boost if EMA and VWAP are aligned (Confluence)
            # e.g. In uptrend, EMA is below VWAP, providing double support
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
                    "vol_ratio": round(vol_ratio, 2)
                }
            )

        except Exception as e:
            LOGGER.error(f"VWAP Strategy Error on {symbol}: {e}", exc_info=True)
            return None


__all__ = ["VWAPProStrategy"]
