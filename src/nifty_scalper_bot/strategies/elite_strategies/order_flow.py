"""Order flow imbalance strategy."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    OrderFlowStrategyConfig,
)


class OrderFlowStrategy(EliteStrategy):
    """Use top-of-book imbalance and large orders for confirmation."""

    def __init__(self, config: OrderFlowStrategyConfig) -> None:
        """Initialise strategy with configuration.

        Args:
            config: Strategy configuration dataclass.

        Returns:
            None.

        Raises:
            None.
        """

        super().__init__(name="Order Flow Imbalance", config=config)
        self._of_config = config
        self._last_signal_epoch: dict[str, float] = {}

    def _evaluate_signal(
        self,
        symbol: str,
        indicators: Mapping[str, Any],
        current_price: float,
        position: Any | None,
    ) -> EliteSignal | None:
        """Generate signal when depth imbalance persists with large orders.

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
            "Entered OrderFlowStrategy._evaluate_signal",
            extra={"event": "order_flow_evaluate", "symbol": symbol},
        )
        try:
            now = datetime.now(timezone.utc).timestamp()
            last_epoch = self._last_signal_epoch.get(symbol, 0.0)
            if now - last_epoch < self._of_config.persistence_seconds:
                return None

            depth_snapshot = indicators.get("market_depth")
            imbalance = self._calculate_imbalance(depth_snapshot)
            if imbalance is None:
                return None
            side, ratio = imbalance

            if side == "BUY" and ratio < self._of_config.imbalance_ratio_buy:
                return None
            if side == "SELL" and ratio < self._of_config.imbalance_ratio_sell:
                return None

            if not self._has_large_order(depth_snapshot, side):
                return None

            if position and getattr(position, "side", "").upper() == (
                "LONG" if side == "BUY" else "SHORT"
            ):
                return None

            confidence = self._of_config.min_confidence
            confidence += min(20.0, max(ratio - 1.0, 0.0) * 10.0)
            stop_offset = 10.0
            target_points = self._of_config.target_points
            if side == "BUY":
                stop_loss = current_price - stop_offset
                tp1 = current_price + target_points
                tp2 = current_price + target_points * 1.5
            else:
                stop_loss = current_price + stop_offset
                tp1 = current_price - target_points
                tp2 = current_price - target_points * 1.5

            self._last_signal_epoch[symbol] = now
            self._logger.info(
                "Condition met: order_flow_signal",
                extra={
                    "event": "order_flow_signal",
                    "symbol": symbol,
                    "side": side,
                    "confidence": confidence,
                    "imbalance_ratio": ratio,
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
                    "imbalance_ratio": ratio,
                    "large_order_threshold": self._of_config.large_order_pct,
                },
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in OrderFlowStrategy._evaluate_signal: %s",
                exc,
                exc_info=exc,
                extra={"event": "order_flow_evaluate_error", "symbol": symbol},
            )
            return None

    def _calculate_imbalance(self, depth_snapshot: Any) -> tuple[str, float] | None:
        """Return dominant side and ratio from *depth_snapshot*.

        Args:
            depth_snapshot: Market depth structure containing bids and asks.

        Returns:
            tuple[str, float] | None: Imbalance side and ratio when available.

        Raises:
            None.
        """

        try:
            if not isinstance(depth_snapshot, Mapping):
                return None
            bids: Sequence[Sequence[float]] | None = depth_snapshot.get("bids")
            asks: Sequence[Sequence[float]] | None = depth_snapshot.get("asks")
            if not bids or not asks:
                return None
            bid_total = sum(float(level[1]) for level in bids[:5])
            ask_total = sum(float(level[1]) for level in asks[:5])
            if bid_total <= 0 or ask_total <= 0:
                return None
            if bid_total >= ask_total:
                return "BUY", bid_total / ask_total
            return "SELL", ask_total / bid_total
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in OrderFlowStrategy._calculate_imbalance: %s",
                exc,
                exc_info=exc,
                extra={"event": "order_flow_imbalance_error"},
            )
            return None

    def _has_large_order(self, depth_snapshot: Any, side: str) -> bool:
        """Return ``True`` when side of book holds a large resting order.

        Args:
            depth_snapshot: Market depth structure containing bids and asks.
            side: Trade side under evaluation (``BUY`` or ``SELL``).

        Returns:
            bool: ``True`` if large order detected else ``False``.

        Raises:
            None.
        """

        try:
            if not isinstance(depth_snapshot, Mapping):
                return False
            book_key = "bids" if side == "BUY" else "asks"
            levels: Sequence[Sequence[float]] | None = depth_snapshot.get(book_key)
            if not levels:
                return False
            total_quantity = sum(float(level[1]) for level in levels[:5])
            if total_quantity <= 0:
                return False
            threshold = self._of_config.large_order_pct
            return any(
                (float(level[1]) / total_quantity) * 100.0 >= threshold
                for level in levels[:5]
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in OrderFlowStrategy._has_large_order: %s",
                exc,
                exc_info=exc,
                extra={"event": "order_flow_large_order_error"},
            )
            return False

    def get_required_indicators(self) -> list[str]:
        """Return indicator dependencies including market depth.

        Args:
            None.

        Returns:
            list[str]: Indicator names required for evaluation.

        Raises:
            None.
        """

        base_indicators = super().get_required_indicators()
        if "market_depth" not in base_indicators:
            base_indicators.append("market_depth")
        return base_indicators


__all__ = ["OrderFlowStrategy"]
