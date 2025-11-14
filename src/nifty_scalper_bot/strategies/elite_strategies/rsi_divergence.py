"""RSI divergence strategy."""

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


class RSIDivergenceStrategy(EliteStrategy):
    """Look for RSI diverging from price direction as a reversal cue."""

    def __init__(self, config: RSIDivergenceStrategyConfig) -> None:
        """Initialise strategy with configuration.

        Args:
            config: Strategy configuration dataclass.

        Returns:
            None.

        Raises:
            None.
        """

        super().__init__(name="RSI Divergence", config=config)
        self._rsi_config = config
        self._price_history: dict[str, Deque[Tuple[float, float, float]]] = {}

    def _evaluate_signal(
        self,
        symbol: str,
        indicators: Mapping[str, Any],
        current_price: float,
        position: Any | None,
    ) -> EliteSignal | None:
        """Generate signal when RSI diverges from price swings with volume.

        Args:
            symbol: Trading symbol evaluated.
            indicators: Indicator snapshot for symbol.
            current_price: Latest traded price.
            position: Existing open position when present.

        Returns:
            EliteSignal | None: Signal when setup detected else ``None``.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered RSIDivergenceStrategy._evaluate_signal",
            extra={"event": "rsi_div_evaluate", "symbol": symbol},
        )
        try:
            rsi = float(indicators.get("rsi") or 50.0)
            volume = float(indicators.get("volume") or 0.0)
            avg_volume = float(indicators.get("avg_volume") or 0.0)
            atr = float(indicators.get("atr") or 0.0)
            if avg_volume <= 0:
                return None

            history = self._price_history.setdefault(
                symbol,
                deque(maxlen=max(self._rsi_config.swing_lookback * 3, 30)),
            )
            history.append((current_price, rsi, volume))
            if len(history) < self._rsi_config.swing_lookback * 2:
                return None

            swing_window = max(2, self._rsi_config.swing_lookback // 2)
            swing_lows = self._find_swings(history, swing_window, mode="low")
            swing_highs = self._find_swings(history, swing_window, mode="high")

            signal_side = ""
            signal_rsi = 0.0
            pivot_index = -1
            if len(swing_lows) >= 2:
                (_, price_one, rsi_one), (_, price_two, rsi_two) = swing_lows[-2:]
                if (
                    price_two < price_one
                    and rsi_two > rsi_one
                    and rsi_two < self._rsi_config.oversold
                ):
                    signal_side = "BUY"
                    signal_rsi = rsi_two
                    pivot_index = len(history) - 1
            if not signal_side and len(swing_highs) >= 2:
                (_, price_one, rsi_one), (_, price_two, rsi_two) = swing_highs[-2:]
                if (
                    price_two > price_one
                    and rsi_two < rsi_one
                    and rsi_two > self._rsi_config.overbought
                ):
                    signal_side = "SELL"
                    signal_rsi = rsi_two
                    pivot_index = len(history) - 1
            if not signal_side:
                return None

            if volume < avg_volume * self._rsi_config.confirmation_volume:
                return None

            if position and getattr(position, "side", "").upper() == (
                "LONG" if signal_side == "BUY" else "SHORT"
            ):
                return None

            confidence = self._rsi_config.min_confidence + 10.0
            stop_offset = max(atr, 25.0)
            if signal_side == "BUY":
                stop_loss = current_price - stop_offset
                tp1 = current_price + max(40.0, stop_offset * 1.6)
                tp2 = current_price + max(60.0, stop_offset * 2.4)
            else:
                stop_loss = current_price + stop_offset
                tp1 = current_price - max(40.0, stop_offset * 1.6)
                tp2 = current_price - max(60.0, stop_offset * 2.4)

            self._logger.info(
                "Condition met: rsi_divergence_signal",
                extra={
                    "event": "rsi_divergence_signal",
                    "symbol": symbol,
                    "side": signal_side,
                    "confidence": confidence,
                },
            )
            return EliteSignal(
                symbol=symbol,
                side=signal_side,
                confidence=min(confidence, 100.0),
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit_1=tp1,
                take_profit_2=tp2,
                quantity=1,
                strategy_name=self.name,
                metadata={
                    "pivot_index": pivot_index,
                    "signal_rsi": signal_rsi,
                    "volume": volume,
                    "avg_volume": avg_volume,
                },
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in RSIDivergenceStrategy._evaluate_signal: %s",
                exc,
                exc_info=exc,
                extra={"event": "rsi_div_evaluate_error", "symbol": symbol},
            )
            return None

    def _find_swings(
        self,
        history: Deque[Tuple[float, float, float]],
        window: int,
        mode: str,
    ) -> Sequence[Tuple[int, float, float]]:
        """Return swing points with RSI context for provided *history*.

        Args:
            history: Rolling buffer of (price, RSI, volume) tuples.
            window: Neighbourhood window used for extrema detection.
            mode: Either ``"low"`` or ``"high"`` to locate swings.

        Returns:
            Sequence[Tuple[int, float, float]]: Identified swing tuples.

        Raises:
            None.
        """

        try:
            if len(history) < window * 2 + 1:
                return []
            prices = [sample[0] for sample in history]
            rsis = [sample[1] for sample in history]
            swings: list[Tuple[int, float, float]] = []
            for idx in range(window, len(history) - window):
                span = prices[idx - window : idx + window + 1]
                if mode == "low" and prices[idx] <= min(span):
                    swings.append((idx, prices[idx], rsis[idx]))
                elif mode == "high" and prices[idx] >= max(span):
                    swings.append((idx, prices[idx], rsis[idx]))
            return swings
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in RSIDivergenceStrategy._find_swings: %s",
                exc,
                exc_info=exc,
                extra={"event": "rsi_div_find_swings_error"},
            )
            return []


__all__ = ["RSIDivergenceStrategy"]
