"""Opening range breakout pro strategy."""

from __future__ import annotations

from typing import Any, Mapping

from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    ORBProStrategyConfig,
)


class ORBProStrategy(EliteStrategy):
    """Trade validated opening range breakouts with volume confirmation."""

    def __init__(self, config: ORBProStrategyConfig) -> None:
        """Initialise strategy with configuration.

        Args:
            config: Strategy configuration dataclass.

        Returns:
            None.

        Raises:
            None.
        """

        super().__init__(name="ORB Pro", config=config)
        self._orb_config = config
        self._cached_orb: dict[str, dict[str, float]] = {}

    def _evaluate_signal(
        self,
        symbol: str,
        indicators: Mapping[str, Any],
        current_price: float,
        position: Any | None,
    ) -> EliteSignal | None:
        """Return signal when a narrow opening range breaks with volume expansion.

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
            "Entered ORBProStrategy._evaluate_signal",
            extra={"event": "orb_pro_evaluate", "symbol": symbol},
        )
        try:
            minutes_since_open = float(indicators.get("minutes_since_open") or 0.0)
            if minutes_since_open < self._orb_config.duration_min:
                self._cached_orb.pop(symbol, None)
                return None

            if symbol not in self._cached_orb:
                orb_ready = bool(indicators.get("orb_ready"))
                orb_high = float(indicators.get("orb_high") or 0.0)
                orb_low = float(indicators.get("orb_low") or 0.0)
                if orb_ready and orb_high > 0 and orb_low > 0:
                    orb_range = orb_high - orb_low
                    self._cached_orb[symbol] = {
                        "high": orb_high,
                        "low": orb_low,
                        "range": orb_range,
                    }
                    self._logger.info(
                        "ORB cached for symbol",  # intentionally short for logs
                        extra={
                            "event": "orb_cached",
                            "symbol": symbol,
                            "orb_high": orb_high,
                            "orb_low": orb_low,
                            "range": orb_range,
                        },
                    )

            orb_snapshot = self._cached_orb.get(symbol)
            if not orb_snapshot:
                return None
            if orb_snapshot["range"] <= 0:
                return None
            if orb_snapshot["range"] > self._orb_config.narrow_threshold_points:
                return None

            volume = float(indicators.get("volume") or 0.0)
            avg_volume = float(indicators.get("avg_volume") or 0.0)
            if avg_volume <= 0 or volume < avg_volume * self._orb_config.volume_mult:
                return None

            side = ""
            if current_price > orb_snapshot["high"]:
                side = "BUY"
            elif current_price < orb_snapshot["low"]:
                side = "SELL"
            if not side:
                return None

            if position and getattr(position, "side", "").upper() == (
                "LONG" if side == "BUY" else "SHORT"
            ):
                return None

            confidence = self._orb_config.min_confidence + 5.0
            range_width = orb_snapshot["range"]
            if side == "BUY":
                stop_loss = orb_snapshot["low"]
                tp1 = current_price + range_width
                tp2 = current_price + range_width * 2.0
            else:
                stop_loss = orb_snapshot["high"]
                tp1 = current_price - range_width
                tp2 = current_price - range_width * 2.0

            self._logger.info(
                "Condition met: orb_pro_signal",
                extra={
                    "event": "orb_pro_signal",
                    "symbol": symbol,
                    "side": side,
                    "confidence": confidence,
                    "range": range_width,
                },
            )
            return EliteSignal(
                symbol=symbol,
                side=side,
                confidence=min(confidence, 100.0),
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit_1=tp1,
                take_profit_2=tp2,
                quantity=1,
                strategy_name=self.name,
                metadata={
                    "orb_high": orb_snapshot["high"],
                    "orb_low": orb_snapshot["low"],
                    "range": range_width,
                    "volume": volume,
                    "avg_volume": avg_volume,
                },
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in ORBProStrategy._evaluate_signal: %s",
                exc,
                exc_info=exc,
                extra={"event": "orb_pro_evaluate_error", "symbol": symbol},
            )
            return None


__all__ = ["ORBProStrategy"]
