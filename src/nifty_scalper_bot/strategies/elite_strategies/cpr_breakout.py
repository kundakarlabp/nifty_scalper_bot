"""
CPR Breakout Strategy.
World-Class implementation with Narrow CPR, NR7 Detection, and ATR Risk.
Refactored for Push-Based Architecture (Zero-Latency).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Deque
from collections import deque

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
    Engage when Central Pivot Range (CPR) is Narrow (Trending) or NR7 compression resolves.
    Entry: Price breaks CPR/R1/S1 with Volume.
    """
    MIN_BARS_REQUIRED = 2

    # ✅ OPTIMIZATION: Use slots for memory efficiency
    __slots__ = ("_cpr_config", "_range_history")

    def __init__(self, config: CPRBreakoutStrategyConfig, indicator_engine: Any) -> None:
        """
        Initialize strategy with configuration and engine.
        
        Args:
            config: Strategy configuration dataclass.
            indicator_engine: Data provider.
        """
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._cpr_config = config
        
        # Store last 7 daily ranges for NR7 calculation
        # Key: Symbol -> Deque of floats (High - Low)
        self._range_history: Dict[str, Deque[float]] = {}

    def get_required_indicators(self) -> set[str]:
        """
        Declare which indicators this strategy needs pre-calculated.
        The StrategyManager will inject these into _evaluate_signal.
        """
        return {
            "pivot",        # Central Pivot
            "bc",           # Bottom Central
            "tc",           # Top Central
            "r1", "s1",     # Resistance 1, Support 1
            "high",         # Day High (for range calc)
            "low",          # Day Low
            "close",        # Previous Close (for gaps)
            "ltp",
            "volume",
            "average_volume",
            "atr",
            "futures_vwap",
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
            pivot = float(indicators.get("pivot") or 0.0)
            bc = float(indicators.get("bc") or 0.0)
            tc = float(indicators.get("tc") or 0.0)
            r1 = float(indicators.get("r1") or 0.0)
            s1 = float(indicators.get("s1") or 0.0)
            
            day_high = float(indicators.get("high") or current_price)
            day_low = float(indicators.get("low") or current_price)
            
            vol = float(indicators.get("volume") or 0.0)
            avg_vol = float(indicators.get("average_volume") or 1.0)
            atr = float(indicators.get("atr") or 0.0)
            futures_vwap = float(
                indicators.get("futures_vwap")
                or indicators.get("nifty_fut_vwap")
                or indicators.get("nifty_index_vwap")
                or 0.0
            )

            # Sanity Check
            if pivot == 0 or tc == 0 or bc == 0 or futures_vwap <= 0:
                return None

            # 2. Update Range History (For NR7)
            # We track the Daily Range (High - Low)
            current_range = day_high - day_low
            
            if symbol not in self._range_history:
                self._range_history[symbol] = deque(maxlen=7)
            
            # Only append if this is a "completed" day or significant snapshot
            # For real-time scalping, we might check 'previous_day_range' indicator instead.
            # Assuming 'current_range' updates live, we use it for intraday volatility check.
            
            # Logic Note: NR7 technically requires COMPLETED days. 
            # Ideally, the indicator engine provides "prev_day_range_1", "prev_day_range_2"...
            # Here we simplify: if current range is extremely small relative to ATR, we flag compression.
            is_compressed = current_range < (atr * 0.5)

            # 3. Analyze CPR Width (Trend Potential)
            # Narrow CPR = (Abs(TC - BC) < 0.25% of price) -> High Trend Probability
            cpr_width = abs(tc - bc)
            is_narrow_cpr = cpr_width < (current_price * 0.0025)

            if not is_narrow_cpr and not is_compressed:
                # If CPR is Wide and Price isn't compressed, likely choppy range day. Skip breakout.
                return None

            # 4. Logic: Breakout Detection
            # We trade breakouts of the CPR Zone or R1/S1
            
            side = ""
            breakout_level = 0.0
            
            # --- Bullish Breakout ---
            # Condition: Price breaks above TC (Top Central) or R1
            # Volume: Must be expanding (> 1.2x avg)
            if current_price > tc and (vol / avg_vol) > 1.2:
                # Filter: Don't buy if right under R1 resistance
                if current_price < r1 and (r1 - current_price) < (atr * 0.5):
                    return None 
                
                side = "BUY"
                breakout_level = tc
                # Stop Loss: Below BC (Bottom Central)
                stop_loss = bc - (atr * 0.5)
                # Targets: R1, R2
                tp1 = r1 if r1 > current_price else (current_price + atr * 2)
                tp2 = tp1 + (atr * 2)

            # --- Bearish Breakout — SKIPPED (long-only options buying) ---
            # A bearish CPR break cannot be traded without writing options.
            # Previously emitted SELL which conflicted with VWAPPro/SMC BUY signals
            # in _combine_signals → "weak consensus" → trade cancelled. Now skipped.

            if not side:
                return None

            # 5. Risk Management Checks
            # Ensure Stop Loss isn't too wide (> 1% risk)
            risk_pct = abs(current_price - stop_loss) / current_price * 100
            if risk_pct > 1.0:
                return None # Too risky for a scalp

            # 6. Confidence Scoring (0.0–1.0 scale — was wrongly 75.0/15.0/5.0 → passed as 75x)
            confidence = 0.75
            if is_narrow_cpr:
                confidence += 0.15  # Very high probability on trend day
            if is_compressed:
                confidence += 0.05  # NR7-like compression breakout

            LOGGER.info(
                f"🚀 CPR Breakout: {symbol} {side} | Narrow: {is_narrow_cpr} | CPR Width: {cpr_width:.2f}",
                extra={
                    "event": "cpr_breakout_signal",
                    "symbol": symbol,
                    "cpr_width": cpr_width,
                    "is_narrow": is_narrow_cpr
                }
            )

            return EliteSignal(
                symbol=symbol,
                signal=side,
                confidence=min(confidence, 0.99),
                entry_price=current_price,
                stop_loss=stop_loss,
                target=tp1,
                quantity=self._cpr_config.quantity or 1,
                strategy_name="CPR_Breakout_Pro",
                metadata={
                    "type": "CPR_Breakout",
                    "cpr_state": "Narrow" if is_narrow_cpr else "Normal",
                    "breakout_level": breakout_level,
                    "risk_pct": round(risk_pct, 2)
                }
            )

        except Exception as e:
            LOGGER.error(f"CPR Strategy Error on {symbol}: {e}", exc_info=True)
            return None


__all__ = ["CPRBreakoutStrategy"]
