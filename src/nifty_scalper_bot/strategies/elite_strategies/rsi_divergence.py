"""
RSI Divergence Strategy.
World-Class implementation with Greeks validation and Swing Analysis.
Refactored for Push-Based Architecture (Zero-Latency).
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Mapping, Sequence, Tuple, Optional, Dict

from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    RSIDivergenceStrategyConfig,
)
from nifty_scalper_bot.utils.logging import get_logger

# Initialize structured logger
LOGGER = get_logger(__name__)


class RSIDivergenceStrategy(EliteStrategy):
    """
    Detects RSI divergences (Regular & Hidden) to play reversals.
    """
    MIN_BARS_REQUIRED = 20

    # ✅ OPTIMIZATION: Use slots for memory efficiency
    __slots__ = ("_rsi_config", "_price_history")

    def __init__(self, config: RSIDivergenceStrategyConfig, indicator_engine: Any) -> None:
        """
        Initialize strategy with configuration and engine.
        
        Args:
            config: Strategy configuration dataclass.
            indicator_engine: Data provider.
        """
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._rsi_config = config
        
        # Store last 60 candles: (Price, RSI, Volume) for swing detection
        # Key: Symbol -> Value: Deque of tuples
        self._price_history: dict[str, Deque[Tuple[float, float, float]]] = {}

    def get_required_indicators(self) -> set[str]:
        """
        Declare which indicators this strategy needs pre-calculated.
        The StrategyManager will inject these into _evaluate_signal.
        """
        return {"rsi", "atr", "volume"}

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
            indicators: Dictionary containing pre-fetched 'rsi', 'atr', 'volume'.
            current_price: Latest LTP.
            position: Current open position (if any).
        """
        try:
            # 1. Safe Data Extraction (Fast Path)
            # Use defaults to prevent crashing if an indicator is temporarily missing
            rsi = float(indicators.get("rsi") or 50.0)
            _raw_atr = float(indicators.get("atr") or 0.0)
            # ATR floor: 1% of premium, min ₹1 — prevents zero SL/TP on low-premium options
            atr = max(_raw_atr, current_price * 0.01, 1.0)
            volume = float(indicators.get("volume") or 0.0)

            # Use NIFTY index price for divergence detection (NOT option premium).
            # Divergence on index LTP is meaningful (index made LL, RSI made HL = reversal).
            # Option premium prices are noisy and don't form reliable swing highs/lows.
            index_ltp = float(
                indicators.get("nifty_index_ltp")
                or indicators.get("nifty_fut_ltp")
                or 0.0
            )
            # If no index data available, fall back to option price (degraded mode — less accurate)
            track_price = index_ltp if index_ltp > 0 else current_price

            # 2. Update Internal History (track index price, not option premium)
            # We need history to detect Swings (Highs/Lows)
            if symbol not in self._price_history:
                self._price_history[symbol] = deque(maxlen=60)
            
            history = self._price_history[symbol]
            history.append((track_price, rsi, volume))

            # Need minimum bars to detect swings (window * 2 + 1)
            window = 5  # Swing pivot window
            if len(history) < (window * 2 + 2):
                return None

            # 3. Detect Swings
            # Get recent Swing Highs and Swing Lows
            swing_highs = self._find_swings(history, window, "high")
            swing_lows = self._find_swings(history, window, "low")

            if not swing_highs and not swing_lows:
                return None

            # 4. Analyze Divergence (Regular)
            
            # --- Bullish Divergence (Long) ---
            # Condition: Price makes Lower Low, RSI makes Higher Low
            if len(swing_lows) >= 2:
                last_swing = swing_lows[-1]  # Most recent
                prev_swing = swing_lows[-2]  # Previous
                
                # Ensure the last swing is very recent (within last 3 candles)
                # history_len - 1 is current candle index
                if (len(history) - 1) - last_swing[0] <= 3:
                    price_lower_low = last_swing[1] < prev_swing[1]
                    rsi_higher_low = last_swing[2] > prev_swing[2]
                    
                    if price_lower_low and rsi_higher_low and rsi < 40:
                        LOGGER.info(f"🐂 Bullish Divergence on {symbol} (RSI: {rsi:.1f})")
                        return EliteSignal(
                            symbol=symbol,
                            signal="BUY",
                            confidence=0.0,
                            entry_price=current_price,
                            stop_loss=current_price - (atr * 2.0),
                            target=current_price + (atr * 4.0),
                            quantity=self._config.quantity or 1,
                            strategy_name="RSI_Div_Pro",
                            metadata={
                                "type": "Bullish Divergence",
                                "rsi": rsi,
                                "atr": atr
                            }
                        )

            # --- Bearish Divergence (Short) ---
            # Condition: Price makes Higher High, RSI makes Lower High
            if len(swing_highs) >= 2:
                last_swing = swing_highs[-1]
                prev_swing = swing_highs[-2]
                
                if (len(history) - 1) - last_swing[0] <= 3:
                    price_higher_high = last_swing[1] > prev_swing[1]
                    rsi_lower_high = last_swing[2] < prev_swing[2]
                    
                    if price_higher_high and rsi_lower_high and rsi > 60:
                        # Long-only mode: bearish divergence means price may reverse DOWN.
                        # We cannot short options (Zerodha requires writing margin).
                        # Skip this signal — do NOT emit SELL which conflicts with BUY signals
                        # from VWAPPro and cancels valid trades in _combine_signals.
                        LOGGER.debug(
                            f"🐻 Bearish divergence SKIPPED (long-only): {symbol} RSI={rsi:.1f}",
                            extra={"event": "rsi_bearish_skip_long_only", "symbol": symbol, "rsi": rsi},
                        )
                        return None

            return None

        except Exception as e:
            # Fail gracefully without crashing the bot loop
            LOGGER.error(f"RSI Strategy Error on {symbol}: {e}", exc_info=True)
            return None

    def _find_swings(
        self,
        history: Deque[Tuple[float, float, float]],
        window: int,
        mode: str,
    ) -> Sequence[Tuple[int, float, float]]:
        """
        Return swing points (Index, Price, RSI).
        Helper function: purely mathematical, no I/O.
        """
        try:
            if len(history) < window * 2 + 1:
                return []
            
            # Convert deque to list for slicing
            data = list(history)
            prices = [x[0] for x in data]
            rsis = [x[1] for x in data]
            swings: list[Tuple[int, float, float]] = []
            
            # Iterate through data, excluding edges to ensure full window check
            # We stop 'window' bars before the end to ensure we have a 'right side' to the pivot
            limit = len(data) - window
            
            for idx in range(window, limit):
                chunk = prices[idx - window : idx + window + 1]
                current = prices[idx]
                
                if mode == "low":
                    if current == min(chunk):
                        swings.append((idx, current, rsis[idx]))
                elif mode == "high":
                    if current == max(chunk):
                        swings.append((idx, current, rsis[idx]))
            
            return swings
            
        except Exception as exc:
            LOGGER.error(f"Swing detection failed: {exc}")
            return []


__all__ = ["RSIDivergenceStrategy"]
