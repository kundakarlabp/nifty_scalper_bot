"""CPR breakout strategy built on opening range compression."""

from __future__ import annotations

from typing import Any, Mapping

from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    CPRBreakoutStrategyConfig,
)


class CPRBreakoutStrategy(EliteStrategy):
    """Engage when opening range compresses followed by a volume backed break."""

    def __init__(self, config: CPRBreakoutStrategyConfig) -> None:
        """Initialise strategy with configuration.

        Args:
            config: Strategy configuration dataclass.

        Returns:
            None.

        Raises:
            None.
        """

        super().__init__(name="CPR Breakout", config=config)
        self._cpr_config = config

    def _evaluate_signal(
        self,
        symbol: str,
        indicators: Mapping[str, Any],
        current_price: float,
        position: Any | None,
    ) -> EliteSignal | None:
        """Generate signal when NR7 compression resolves with range expansion.

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
            "Entered CPRBreakoutStrategy._evaluate_signal",
            extra={"event": "cpr_evaluate", "symbol": symbol},
        )
        try:
            minutes_since_open = float(indicators.get("minutes_since_open") or 0.0)
            if not (
                self._cpr_config.breakout_window_start
                <= minutes_since_open
                <= self._cpr_config.breakout_window_end
            ):
                return None
            orb_ready = bool(indicators.get("orb_ready"))
            if not orb_ready:
                return None
            orb_high = float(indicators.get("orb_high") or 0.0)
            orb_low = float(indicators.get("orb_low") or 0.0)
            nr7_flag = bool(indicators.get("nr7"))
            volume_ratio = float(indicators.get("volume_spike_ratio") or 0.0)
            if orb_high <= 0 or orb_low <= 0:
                return None
            if not nr7_flag and volume_ratio < self._cpr_config.volume_mult:
                return None
            buffer = self._cpr_config.gap_threshold_points
            side = ""
            confidence = self._cpr_config.min_confidence
            if current_price > orb_high + buffer:
                side = "BUY"
            elif current_price < orb_low - buffer:
                side = "SELL"
            if not side:
                return None
            if position and getattr(position, "side", "").upper() == (
                "LONG" if side == "BUY" else "SHORT"
            ):
                return None
            atr = float(indicators.get("atr") or 0.0)
            if atr <= 0:
                return None
            confidence += min(15.0, volume_ratio * 4.0)
            if side == "BUY":
                stop_loss = orb_low - buffer
                tp1 = current_price + atr * 1.8
                tp2 = current_price + atr * 2.5
            else:
                stop_loss = orb_high + buffer
                tp1 = current_price - atr * 1.8
                tp2 = current_price - atr * 2.5
            self._logger.info(
                "Condition met: cpr_breakout_signal",
                extra={
                    "event": "cpr_breakout_signal",
                    "symbol": symbol,
                    "side": side,
                    "confidence": confidence,
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
                    "orb_high": orb_high,
                    "orb_low": orb_low,
                    "volume_ratio": volume_ratio,
                    "nr7": nr7_flag,
                },
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in CPRBreakoutStrategy._evaluate_signal: %s",
                exc,
                exc_info=exc,
                extra={"event": "cpr_evaluate_error", "symbol": symbol},
            )
            return None


__all__ = ["CPRBreakoutStrategy"]
