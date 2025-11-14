"""Strategy orchestrator coordinating entries and risk gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Protocol, Sequence

from nifty_scalper_bot.utils.logging import get_logger

logger = get_logger(__name__)


class OrderManagerProtocol(Protocol):
    """Protocol capturing the subset of order manager behaviour used here."""

    def execute_bracket_trade(
        self,
        *,
        symbol: str,
        side: str,
        quantity: int,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        tag: str | None = None,
    ) -> tuple[str, str, str]:
        """Place bracket trade and return order identifiers."""


class BrokerProtocol(Protocol):
    """Protocol capturing the broker operations required by the orchestrator."""

    def get_positions(self) -> Sequence[Mapping[str, object]]:
        """Return iterable of live positions with realised P&L."""


@dataclass(slots=True)
class StrategyOrchestratorConfig:
    """Configuration for :class:`StrategyOrchestrator`."""

    daily_loss_limit: float
    daily_profit_target: float
    starting_equity: float


class StrategyOrchestrator:
    """Coordinate strategy execution with risk-based guardrails."""

    def __init__(
        self,
        *,
        order_manager: OrderManagerProtocol,
        broker: BrokerProtocol,
        config: StrategyOrchestratorConfig,
        signal_handler: Callable[[], None] | None = None,
    ) -> None:
        """Initialise orchestrator with dependencies and config."""

        self._order_manager = order_manager
        self._broker = broker
        self._config = config
        self._signal_handler = signal_handler

    def run_strategy_cycle(self) -> None:
        """Execute one strategy cycle while enforcing P&L gates.

        Args:
            None

        Returns:
            None

        Raises:
            None.
        """

        logger.debug(
            "Entered StrategyOrchestrator.run_strategy_cycle",
            extra={"event": "strategy_cycle_enter"},
        )
        daily_pnl = self._calculate_daily_pnl()
        daily_pnl_pct = self._resolve_daily_pnl_pct(daily_pnl)

        loss_limit = float(self._config.daily_loss_limit)
        if loss_limit > 0 and daily_pnl_pct <= -abs(loss_limit):
            logger.info(
                "daily_loss_limit_hit",
                extra={
                    "event": "daily_loss_limit_hit",
                    "daily_pnl_pct": daily_pnl_pct,
                },
            )
            self._close_all_positions()
            return

        profit_target = float(self._config.daily_profit_target)
        if profit_target > 0 and daily_pnl_pct >= profit_target:
            logger.info(
                "daily_profit_target_reached",
                extra={
                    "event": "daily_profit_target_reached",
                    "daily_pnl_pct": daily_pnl_pct,
                },
            )
            self._close_all_positions()
            return

        self._process_signals()

    def _resolve_daily_pnl_pct(self, daily_pnl: float) -> float:
        """Convert absolute P&L to percent using starting equity.

        Args:
            daily_pnl: Absolute realised P&L for the trading day.

        Returns:
            Realised P&L as a percentage of starting equity.

        Raises:
            None.
        """

        equity = max(float(self._config.starting_equity), 0.0)
        if equity <= 0:
            return 0.0
        return (daily_pnl / equity) * 100.0

    def _calculate_daily_pnl(self) -> float:
        """Sum realised P&L across broker positions for the session.

        Args:
            None

        Returns:
            Aggregate realised P&L for active positions.

        Raises:
            None.
        """

        logger.debug(
            "Entered StrategyOrchestrator._calculate_daily_pnl",
            extra={"event": "calculate_daily_pnl_enter"},
        )
        try:
            positions = self._broker.get_positions()
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failure in StrategyOrchestrator._calculate_daily_pnl: %s",
                exc,
                extra={"event": "calculate_daily_pnl_error"},
            )
            return 0.0

        total = 0.0
        for entry in self._iter_positions(positions):
            realised = self._extract_realised(entry)
            total += realised
        return total

    def _process_signals(self) -> None:
        """Run downstream signal handling callback when configured.

        Args:
            None

        Returns:
            None

        Raises:
            None.
        """

        logger.debug(
            "Entered StrategyOrchestrator._process_signals",
            extra={"event": "process_signals_enter"},
        )
        if self._signal_handler is None:
            return
        try:
            self._signal_handler()
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failure in StrategyOrchestrator._process_signals: %s",
                exc,
                extra={"event": "process_signals_error"},
            )

    def _close_all_positions(self) -> None:
        """Close all positions using the configured order manager.

        Args:
            None

        Returns:
            None

        Raises:
            None.
        """

        logger.debug(
            "Entered StrategyOrchestrator._close_all_positions",
            extra={"event": "close_all_positions_enter"},
        )
        closer = getattr(self._order_manager, "close_all_positions", None)
        if callable(closer):
            try:
                closer(reason="risk_gate")
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Failure in StrategyOrchestrator._close_all_positions: %s",
                    exc,
                    extra={"event": "close_all_positions_error"},
                )
            return

        broker_closer = getattr(self._broker, "close_all_positions", None)
        if callable(broker_closer):
            try:
                broker_closer(reason="risk_gate")
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Failure in StrategyOrchestrator._close_all_positions: %s",
                    exc,
                    extra={"event": "close_all_positions_error"},
                )

    def _iter_positions(
        self, entries: Iterable[Mapping[str, object]]
    ) -> Iterable[Mapping[str, object]]:
        """Yield mapping entries from broker payload defensively.

        Args:
            entries: Raw broker payload.

        Returns:
            Iterable of mapping entries.

        Raises:
            None.
        """

        for entry in entries:
            if isinstance(entry, Mapping):
                yield entry

    @staticmethod
    def _extract_realised(entry: Mapping[str, object]) -> float:
        """Extract realised P&L value from a position entry.

        Args:
            entry: Position mapping from broker response.

        Returns:
            Realised P&L as float.

        Raises:
            None.
        """

        for key in ("realized", "realised", "pnl", "day_pnl", "pnl_realized"):
            value = entry.get(key)
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    continue
        return 0.0

    def _execute_entry(
        self,
        *,
        symbol: str,
        side: str,
        quantity: int,
        entry_price: float,
        spot_price: float,
        sl_spot: float,
        delta: float,
        tag: str | None = None,
    ) -> tuple[str, str, str]:
        """Submit bracket order converting spot SL/TP to option premiums.

        Args:
            symbol: Instrument identifier for the option leg.
            side: Trade direction ('BUY' or 'SELL').
            quantity: Order quantity.
            entry_price: Option entry price.
            spot_price: Underlying spot price at entry time.
            sl_spot: Spot level for protective stop.
            delta: Option delta estimate for conversion.
            tag: Optional audit tag propagated to order manager.

        Returns:
            Tuple of entry, stop, and target order identifiers.

        Raises:
            None.
        """

        logger.debug(
            "Entered StrategyOrchestrator._execute_entry",
            extra={"event": "execute_entry_enter", "symbol": symbol},
        )
        option_move = abs(float(spot_price) - float(sl_spot)) * abs(float(delta))
        stop_loss_price = float(entry_price) - option_move
        take_profit_price = float(entry_price) + option_move * 2.0

        try:
            orders = self._order_manager.execute_bracket_trade(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=entry_price,
                stop_loss=stop_loss_price,
                take_profit=take_profit_price,
                tag=tag,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failure in StrategyOrchestrator._execute_entry: %s",
                exc,
                extra={"event": "execute_entry_error", "symbol": symbol},
            )
            raise

        logger.info(
            "Condition met: bracket_trade_submitted",
            extra={
                "event": "execute_entry_success",
                "symbol": symbol,
                "stop_loss_price": stop_loss_price,
                "take_profit_price": take_profit_price,
            },
        )
        return orders


__all__ = ["StrategyOrchestrator", "StrategyOrchestratorConfig"]
