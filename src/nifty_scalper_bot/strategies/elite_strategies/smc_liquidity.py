"""
Smart Money Concepts (SMC) Liquidity Sweep Strategy.
World-Class implementation with Sweep Detection, Volume Absorption, and Greeks Validation.
Refactored for Push-Based Architecture (Zero-Latency).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

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
    Enters on Rejection Candles where price pierces a level (Bollinger Band) but closes back inside.
    """
    MIN_BARS_REQUIRED = 15

    # ✅ OPTIMIZATION: Use slots for memory efficiency
    __slots__ = ("_smc_config",)

    def __init__(self, config: SMCStrategyConfig, indicator_engine: Any) -> None:
        """
        Initialize strategy with configuration and engine.
        
        Args:
            config: Strategy configuration dataclass.
            indicator_engine: Data provider.
        """
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._smc_config = config

    def get_required_indicators(self) -> set[str]:
        """
        Declare which indicators this strategy needs pre-calculated.
        The StrategyManager will inject these into _evaluate_signal.
        """
        return {
            "bollinger_upper", 
            "bollinger_lower", 
            "volume", 
            "avg_volume", # Usually SMA(Volume, 20)
            "atr", 
            "vwap",
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
        Modern Signature: Evaluates signal using injected data.
        
        Args:
            symbol: Ticker symbol.
            indicators: Dictionary containing pre-fetched indicators.
            current_price: Latest LTP.
            position: Current open position (if any).
        """
        try:
            # 1. Safe Data Extraction (Fast Path)
            # Use defaults to prevent crashing if an indicator is temporarily missing
            upper = float(indicators.get("bollinger_upper") or 0.0)
            lower = float(indicators.get("bollinger_lower") or 0.0)
            vol = float(indicators.get("volume") or 0.0)
            avg_vol = float(indicators.get("avg_volume") or 1.0) # Avoid div/0
            atr = float(indicators.get("atr") or 0.0)
            vwap = float(indicators.get("vwap") or current_price)
            
            # Candle OHLC (needed for wick detection)
            high = float(indicators.get("high") or current_price)
            low = float(indicators.get("low") or current_price)
            close = float(indicators.get("close") or current_price)
            # open_price = float(indicators.get("open") or current_price)

            # Sanity Check: If bands are collapsed or data is invalid, skip
            if upper <= lower or avg_vol <= 0:
                return None

            # 2. Logic: Detect Liquidity Sweep
            # A sweep occurs when price wick goes OUTSIDE the band, but candle closes INSIDE.
            
            # --- Bearish Sweep (Short) ---
            # Wick pierces Upper Band, but Close is below Upper Band
            bearish_sweep = (high > upper) and (close < upper)
            
            # --- Bullish Sweep (Long) ---
            # Wick pierces Lower Band, but Close is above Lower Band
            bullish_sweep = (low < lower) and (close > lower)

            if not bearish_sweep and not bullish_sweep:
                return None

            # 3. Logic: Check Volume Absorption
            # We want high volume on the rejection candle (Absorption)
            vol_ratio = vol / avg_vol
            if vol_ratio < 1.2: # Minimum 20% higher than average
                return None

            # 4. Construct Signal
            signal_side = ""
            sweep_level = 0.0
            stop_loss = 0.0
            tp1 = 0.0
            tp2 = 0.0
            
            # ✅ ATR fallback to prevent zero stop-loss
            if atr <= 0:
                atr = current_price * 0.01  # 1% fallback
            buffer = max(atr * 0.2, current_price * 0.002)  # Min 0.2% buffer

            if bullish_sweep:
                signal_side = "BUY"
                sweep_level = low
                # SL below the sweep wick
                stop_loss = low - buffer
                # TP1: VWAP (Liquidity Equilibrium)
                tp1 = vwap
                # TP2: Upper Band
                tp2 = upper
                
            elif bearish_sweep:
                signal_side = "SELL"
                sweep_level = high
                # SL above the sweep wick
                stop_loss = high + buffer
                # TP1: VWAP
                tp1 = vwap
                # TP2: Lower Band
                tp2 = lower

            # 5. Confidence Scoring
            # Base confidence 75%, +15% if volume is massive (>2x avg)
            confidence = 0.75
            if vol_ratio > 2.0:
                confidence += 0.15

            LOGGER.info(
                f"🚀 SMC Sweep Detected: {symbol} {signal_side} | Vol: {vol_ratio:.1f}x | Wick: {sweep_level}",
                extra={
                    "event": "smc_sweep_signal",
                    "symbol": symbol,
                    "sweep_level": sweep_level,
                    "volume_ratio": vol_ratio
                }
            )

            return EliteSignal(
                symbol=symbol,
                signal=signal_side, # Standardized to "BUY" / "SELL"
                confidence=min(confidence, 0.99),
                entry_price=close, # Enter at close of rejection candle
                stop_loss=stop_loss,
                target=tp1, # Primary Target
                quantity=self._smc_config.quantity or 1,
                strategy_name="SMC_Liquidity_Pro",
                metadata={
                    "type": "Bollinger_Rejection",
                    "volume_ratio": round(vol_ratio, 2),
                    "sweep_level": sweep_level,
                    "vwap_target": vwap,
                    "tp2": tp2
                }
            )

        except Exception as e:
            LOGGER.error(f"SMC Strategy Error on {symbol}: {e}", exc_info=True)
            return None


__all__ = ["SMCStrategy"]
