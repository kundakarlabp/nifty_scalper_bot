"""
RSI Divergence Strategy.
World-Class implementation with Greeks validation and Swing Analysis.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Mapping, Sequence, Tuple

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

    def __init__(self, config: RSIDivergenceStrategyConfig, indicator_engine: Any) -> None:
        """
        Initialize strategy with configuration and engine.
        
        Args:
            config: Strategy configuration dataclass.
            indicator_engine: Data provider.
        """
        # CRITICAL FIX: Correct init signature
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._rsi_config = config
        # Store last 60 candles: (Price, RSI, Volume)
        self._price_history: dict[str, Deque[Tuple[float, float, float]]] = {}

    def _evaluate_signal(self) -> EliteSignal | None:
        """
        Core Logic:
        1. Accumulate History (Price, RSI).
        2. Detect Swings (Highs/Lows).
        3. Identify Divergence (Price vs RSI).
        4. Validate Physics (Greeks/Liquidity).
        """
        symbol = self._rsi_config.symbol
        
        # 1. Fetch Indicators
        required_indicators = {
            "ltp", "rsi", "volume", "atr", "minutes_since_open"
        }
        
        indicators = self._indicator_engine.get_indicators(symbol, required_indicators)
        
        try:
            ltp = float(indicators.get("ltp") or 0)
            rsi = float(indicators.get("rsi") or 50)
            vol = float(indicators.get("volume") or 0)
            atr = float(indicators.get("atr") or 0)
        except (ValueError, TypeError):
            return None

        if ltp == 0:
            return None

        # Initialize history buffer
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=60) # Keep 1 hour of 1-min bars
        
        history = self._price_history[symbol]
        history.append((ltp, rsi, vol))
        
        # Need enough data to find swings (e.g., at least 20 bars)
        if len(history) < 20:
            return None

        # 2. Find Swings
        # Lookback window for local extrema (e.g., 5 bars left/right)
        window = 5
        swing_lows = self._find_swings(history, window, "low")
        swing_highs = self._find_swings(history, window, "high")
        
        if not swing_lows and not swing_highs:
            return None

        side: str | None = None
        confidence = 0.60
        
        # 3. Detect Divergence
        # Bullish Divergence: Price Lower Low, RSI Higher Low
        if len(swing_lows) >= 2:
            last_low = swing_lows[-1]
            prev_low = swing_lows[-2]
            
            # Check Price Lower Low
            if last_low[1] < prev_low[1]:
                # Check RSI Higher Low
                if last_low[2] > prev_low[2]:
                    # Filter: RSI must be somewhat oversold (<40) to matter
                    if last_low[2] < 40:
                        side = "BUY"
                        confidence += 0.20

        # Bearish Divergence: Price Higher High, RSI Lower High
        if len(swing_highs) >= 2:
            last_high = swing_highs[-1]
            prev_high = swing_highs[-2]
            
            # Check Price Higher High
            if last_high[1] > prev_high[1]:
                # Check RSI Lower High
                if last_high[2] < prev_high[2]:
                    # Filter: RSI must be somewhat overbought (>60)
                    if last_high[2] > 60:
                        side = "SELL" # BaseStrategy handles PE mapping
                        confidence += 0.20

        if not side:
            return None

        # 4. 🛡️ SAFETY GATE (Physics Check)
        if not self.validate_option_health(symbol, side):
            LOGGER.info(f"⛔ Rejected {symbol}: Failed Greeks/Liquidity Check")
            return None

        # 5. Risk Management (ATR Based)
        if atr == 0: atr = ltp * 0.01
        
        stop_buffer = atr * 2.0
        
        if side == "BUY":
            stop_loss = ltp - stop_buffer
            tp1 = ltp + (stop_buffer * 1.5)
            tp2 = ltp + (stop_buffer * 3.0)
        else:
            stop_loss = ltp + stop_buffer
            tp1 = ltp - (stop_buffer * 1.5)
            tp2 = ltp - (stop_buffer * 3.0)

        # 6. Construct Signal
        LOGGER.info(
            f"🚀 RSI Divergence: {symbol} {side} | RSI: {rsi:.1f} | ATR: {atr:.2f}",
            extra={
                "event": "rsi_divergence_signal",
                "symbol": symbol,
                "rsi_current": rsi,
                "confidence": confidence
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
            quantity=self._rsi_config.quantity or 1,
            strategy_name="RSI_Div_Pro",
            metadata={
                "rsi": rsi,
                "atr": atr,
                "history_len": len(history)
            }
        )

    def _find_swings(
        self,
        history: Deque[Tuple[float, float, float]],
        window: int,
        mode: str,
    ) -> Sequence[Tuple[int, float, float]]:
        """Return swing points (Index, Price, RSI)."""
        try:
            if len(history) < window * 2 + 1:
                return []
            
            # Convert deque to list for slicing
            data = list(history)
            prices = [x[0] for x in data]
            rsis = [x[1] for x in data]
            swings: list[Tuple[int, float, float]] = []
            
            # Iterate through data, excluding edges
            for idx in range(window, len(data) - window):
                chunk = prices[idx - window : idx + window + 1]
                current = prices[idx]
                
                if mode == "low":
                    if current == min(chunk):
                        swings.append((idx, current, rsis[idx]))
                elif mode == "high":
                    if current == max(chunk):
                        swings.append((idx, current, rsis[idx]))
            
            return swings
            
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(f"Swing detection failed: {exc}")
            return []


__all__ = ["RSIDivergenceStrategy"]
