"""Base abstractions and helpers for elite strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Mapping

from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    EliteStrategyConfig,
)
from nifty_scalper_bot.strategies.signal_generator import Signal, Strategy
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class EliteSignal:
    """Container for elite strategy signal output."""

    symbol: str
    side: str
    confidence: float
    entry_price: float
    stop_loss: float | None
    take_profit_1: float | None
    take_profit_2: float | None
    quantity: int = 1
    strategy_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_payload(self) -> dict[str, Any]:
        """Return serialisable representation of the signal.

        Args:
            None.

        Returns:
            dict[str, Any]: Payload describing the signal attributes.

        Raises:
            None.
        """

        return {
            "symbol": self.symbol,
            "side": self.side,
            "confidence": self.confidence,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit_1": self.take_profit_1,
            "take_profit_2": self.take_profit_2,
            "quantity": self.quantity,
            "strategy_name": self.strategy_name,
            "metadata": dict(self.metadata),
            "timestamp": self.timestamp.isoformat(),
        }


class EliteStrategy(Strategy):
    """Adapter bridging elite strategy logic with core manager contract."""

    def __init__(self, name: str, config: EliteStrategyConfig) -> None:
        """Initialise strategy with configuration.

        Args:
            name: Strategy identifier used in logs and telemetry.
            config: Strongly typed configuration for the strategy.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered EliteStrategy.__init__",
            extra={"event": "elite_strategy_init", "strategy": name},
        )
        self._config = config
        self._last_signal_at: datetime | None = None
        self._signals_generated = 0
        self._last_signal: EliteSignal | None = None
        self._logger = LOGGER
        parameters = {
            "min_confidence": config.min_confidence,
            "enabled": config.enabled,
            "cooldown_seconds": config.cooldown_seconds,
        }
        super().__init__(name=name, parameters=parameters)

    def get_required_indicators(self) -> list[str]:
        """Return default indicator dependencies.

        Args:
            None.

        Returns:
            list[str]: Indicator names required for evaluation.

        Raises:
            None.
        """

        return [
            "rsi",
            "atr",
            "vwap",
            "bollinger_upper",
            "bollinger_middle",
            "bollinger_lower",
            "ema",
            "sma",
            "macd",
            "macd_signal",
            "macd_histogram",
            "volume",
            "avg_volume",
            "volume_spike_ratio",
            "minutes_since_open",
            "minutes_until_close",
            "orb_high",
            "orb_low",
            "orb_ready",
            "nr7",
            "nr7_range",
            "nr7_min_range",
            "bar_range",
        ]

    def generate_signal(
        self,
        symbol: str,
        indicators: Mapping[str, Any],
        current_price: float,
        position: Any | None,
    ) -> Signal | None:
        """Generate trading signal when conditions are satisfied.

        Args:
            symbol: Trading symbol under evaluation.
            indicators: Indicator snapshot provided by engine.
            current_price: Latest traded price.
            position: Existing open position for the symbol, if any.

        Returns:
            Signal | None: Strategy signal when criteria met else ``None``.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered EliteStrategy.generate_signal",
            extra={"event": "elite_generate", "strategy": self.name, "symbol": symbol},
        )
        try:
            if not self._config.enabled:
                return None
            if not self._respect_cooldown():
                return None
            signal = self._evaluate_signal(symbol, indicators, current_price, position)
            if signal is None:
                return None
            if signal.confidence < self._config.min_confidence:
                return None
            converted = self._convert_signal(signal)
            self._update_state(signal)
            self._logger.info(
                "Condition met: elite_signal_generated",
                extra={
                    "event": "elite_signal_generated",
                    "strategy": self.name,
                    "symbol": symbol,
                    "side": signal.side,
                    "confidence": signal.confidence,
                },
            )
            return converted
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in EliteStrategy.generate_signal: %s",
                exc,
                exc_info=exc,
                extra={"event": "elite_generate_error", "strategy": self.name},
            )
            return None

    def _evaluate_signal(
        self,
        symbol: str,
        indicators: Mapping[str, Any],
        current_price: float,
        position: Any | None,
    ) -> EliteSignal | None:
        """Evaluate and return raw elite signal.

        Args:
            symbol: Trading symbol under evaluation.
            indicators: Indicator snapshot provided by engine.
            current_price: Latest traded price.
            position: Existing open position for the symbol, if any.

        Returns:
            EliteSignal | None: Raw signal when detected else ``None``.

        Raises:
            NotImplementedError: When subclass omits implementation.
        """

        raise NotImplementedError

    def _respect_cooldown(self) -> bool:
        """Return ``True`` when cooldown has elapsed.

        Args:
            None.

        Returns:
            bool: ``True`` if cooldown window satisfied.

        Raises:
            None.
        """

        if self._last_signal_at is None:
            return True
        delta = datetime.now(timezone.utc) - self._last_signal_at
        return delta.total_seconds() >= self._config.cooldown_seconds

    def _convert_signal(self, signal: EliteSignal) -> Signal:
        """Convert :class:`EliteSignal` into core :class:`Signal` type.

        Args:
            signal: Elite signal emitted by subclass implementation.

        Returns:
            Signal: Converted signal compatible with strategy manager.

        Raises:
            None.
        """

        confidence = max(0.0, min(signal.confidence / 100.0, 1.0))
        metadata = dict(signal.metadata)
        metadata.update(
            {
                "strategy": self.name,
                "elite_timestamp": signal.timestamp.isoformat(),
                "elite_confidence": signal.confidence,
            }
        )
        action: Literal["BUY", "SELL"] = (
            "BUY" if signal.side.upper() == "BUY" else "SELL"
        )
        return Signal(
            action=action,
            symbol=signal.symbol,
            quantity=max(1, signal.quantity),
            confidence=confidence,
            reason=f"{self.name} elite setup",
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit_1,
            metadata=metadata,
        )

    def _update_state(self, signal: EliteSignal) -> None:
        """Persist bookkeeping fields for telemetry.

        Args:
            signal: Signal that triggered state update.

        Returns:
            None.

        Raises:
            None.
        """

        self._last_signal_at = signal.timestamp
        self._last_signal = signal
        self._signals_generated += 1

    def get_stats(self) -> dict[str, Any]:
        """Return diagnostic statistics for the strategy.

        Args:
            None.

        Returns:
            dict[str, Any]: Statistics describing recent activity.

        Raises:
            None.
        """

        last_payload: dict[str, Any] | None = None
        if self._last_signal is not None:
            last_payload = self._last_signal.to_payload()
        return {
            "strategy": self.name,
            "enabled": self._config.enabled,
            "signals": self._signals_generated,
            "last_signal": last_payload,
        }

    @property
    def config(self) -> EliteStrategyConfig:
        """Return strategy configuration reference.

        Args:
            None.

        Returns:
            EliteStrategyConfig: Configuration structure for strategy.

        Raises:
            None.
        """

        return self._config


__all__ = ["EliteSignal", "EliteStrategy"]
