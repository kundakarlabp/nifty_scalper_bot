"""Thread-safe bracket order manager with atomic OCO guarantees."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from nifty_scalper_bot.utils.logging import get_logger

if TYPE_CHECKING:
    from nifty_scalper_bot.infra.metrics import MetricsCollector

try:
    from nifty_scalper_bot.infra.metrics import METRICS as GLOBAL_METRICS

    METRICS_AVAILABLE = True
    METRICS = cast("MetricsCollector | None", GLOBAL_METRICS)
except ImportError:  # pragma: no cover - optional metrics package
    METRICS_AVAILABLE = False
    METRICS = cast("MetricsCollector | None", None)

LOGGER = get_logger(__name__)
_FILLED_STATUSES = {"FILLED", "COMPLETE", "COMPLETED"}


class SupportsCancelOrder(Protocol):
    """Protocol representing broker cancel capability."""

    def cancel_order(self, order_id: str, *args: Any, **kwargs: Any) -> Any:
        """Cancel an order via broker API."""


@runtime_checkable
class SupportsModifyOrder(Protocol):
    """Protocol representing broker order modification capability."""

    def modify_order(self, order_id: str, **kwargs: Any) -> Any:
        """Modify an existing order."""


@dataclass
class BracketState:
    """Thread-safe bracket order state with optimistic locking."""

    entry_order_id: str
    stop_loss_order_id: str | None
    target_order_id: str | None
    tp2_order_id: str | None
    filled_leg: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    version: int = 0
    entry_quantity: int = 0
    active_quantity: int = 0
    cumulative_tp_filled: int = 0
    created_at: float = field(default_factory=time.time)


class BracketManager:
    """Manage bracket orders and enforce atomic OCO cancellation."""

    def __init__(self, broker_client: SupportsCancelOrder, logger=None) -> None:
        """Initialize bracket manager.

        Args:
            broker_client: Broker API interface for order operations.
            logger: Optional logger instance to override module logger.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered BracketManager.__init__",
            extra={"event": "bracket.manager.init.enter"},
        )
        self.broker: SupportsCancelOrder = broker_client
        self.logger = logger or LOGGER
        self.brackets: dict[str, BracketState] = {}
        self._global_lock = threading.Lock()
        self._order_to_entry: dict[str, str] = {}
        self._auto_reduce_sl = True
        self.logger.info(
            "BracketManager initialized",
            extra={"event": "bracket.manager.init"},
        )

    def register_bracket(
        self,
        entry_order_id: str,
        stop_loss_order_id: str | None,
        target_order_id: str | None,
        tp2_order_id: str | None = None,
        entry_quantity: int = 0,
    ) -> None:
        """Register a bracket order set for monitoring.

        Args:
            entry_order_id: Entry order identifier.
            stop_loss_order_id: Stop-loss order identifier.
            target_order_id: Primary target order identifier.
            tp2_order_id: Optional secondary target identifier.
            entry_quantity: Total filled quantity for partial exit tracking.

        Returns:
            None.

        Raises:
            Exception: Propagated on unexpected registration failure.
        """

        self.logger.debug(
            "Entered BracketManager.register_bracket",
            extra={
                "event": "bracket.register.enter",
                "entry_id": entry_order_id,
            },
        )
        try:
            with self._global_lock:
                state = BracketState(
                    entry_order_id=entry_order_id,
                    stop_loss_order_id=stop_loss_order_id,
                    target_order_id=target_order_id,
                    tp2_order_id=tp2_order_id,
                    entry_quantity=max(entry_quantity, 0),
                    active_quantity=max(entry_quantity, 0),
                )
                self.brackets[entry_order_id] = state
                if stop_loss_order_id:
                    self._order_to_entry[stop_loss_order_id] = entry_order_id
                if target_order_id:
                    self._order_to_entry[target_order_id] = entry_order_id
                if tp2_order_id:
                    self._order_to_entry[tp2_order_id] = entry_order_id

            self.logger.info(
                "Bracket registered",
                extra={
                    "event": "bracket.register",
                    "entry_id": entry_order_id,
                    "sl_id": stop_loss_order_id,
                    "tp_id": target_order_id,
                    "tp2_id": tp2_order_id,
                    "quantity": entry_quantity,
                },
            )

            if METRICS_AVAILABLE and METRICS is not None:
                METRICS.bracket_registrations.inc()
                METRICS.bracket_active.inc()
        except Exception as exc:  # noqa: BLE001
            self.logger.error(
                "Failure in BracketManager.register_bracket: %s",
                exc,
                extra={
                    "event": "bracket.register.error",
                    "entry_id": entry_order_id,
                },
            )
            raise

    def handle_bracket_update(
        self, order_id: str, status: str, filled_quantity: int
    ) -> None:
        """Handle an order status update with atomic OCO cancellation.

        Args:
            order_id: Order identifier receiving the update.
            status: Updated order status.
            filled_quantity: Quantity filled in the update.

        Returns:
            None.

        Raises:
            Exception: Propagated on unexpected coordination failure.
        """

        self.logger.debug(
            "Entered BracketManager.handle_bracket_update",
            extra={
                "event": "bracket.update.enter",
                "order_id": order_id,
                "status": status,
            },
        )
        try:
            bracket_state: BracketState | None = None
            entry_id: str | None = None
            with self._global_lock:
                entry_id = self._order_to_entry.get(order_id)
                if entry_id is not None:
                    bracket_state = self.brackets.get(entry_id)

            if bracket_state is None or entry_id is None:
                self.logger.debug(
                    "No bracket state found for order update",
                    extra={
                        "event": "bracket.update.miss",
                        "order_id": order_id,
                    },
                )
                return

            normalized_status = status.strip().upper()
            if normalized_status not in _FILLED_STATUSES:
                self.logger.debug(
                    "Bracket update ignored due to non-filled status",
                    extra={
                        "event": "bracket.update.noop",
                        "order_id": order_id,
                        "status": status,
                    },
                )
                return

            with bracket_state.lock:
                current_version = bracket_state.version
                filled_leg = None
                if order_id == bracket_state.stop_loss_order_id:
                    filled_leg = "SL"
                elif order_id == bracket_state.target_order_id:
                    filled_leg = "TARGET"
                elif order_id == bracket_state.tp2_order_id:
                    filled_leg = "TP2"

                if filled_leg is None:
                    self.logger.debug(
                        "No matching leg for order update",
                        extra={
                            "event": "bracket.update.unmatched",
                            "order_id": order_id,
                        },
                    )
                    return

                if bracket_state.filled_leg is not None:
                    self.logger.warning(
                        "Bracket leg already filled; skipping OCO cancel",
                        extra={
                            "event": "bracket.oco.skip",
                            "entry_id": entry_id,
                            "already_filled": bracket_state.filled_leg,
                            "current_fill": filled_leg,
                            "version": current_version,
                        },
                    )
                    return

                bracket_state.filled_leg = filled_leg
                bracket_state.version += 1
                self.logger.info(
                    "Bracket leg filled; canceling opposing orders",
                    extra={
                        "event": "bracket.oco.trigger",
                        "entry_id": entry_id,
                        "filled_leg": filled_leg,
                        "version": bracket_state.version,
                        "filled_qty": filled_quantity,
                    },
                )

                if filled_leg == "SL":
                    self._safe_cancel(bracket_state.target_order_id, "TARGET")
                    if METRICS_AVAILABLE and METRICS is not None:
                        METRICS.bracket_oco_cancellations.labels(leg_type="SL").inc()
                    if bracket_state.tp2_order_id:
                        self._safe_cancel(bracket_state.tp2_order_id, "TP2")

                    self.unregister_bracket(entry_id)
                    return

                bracket_state.cumulative_tp_filled += max(filled_quantity, 0)
                is_partial = (
                    bracket_state.cumulative_tp_filled < bracket_state.entry_quantity
                )
                if is_partial:
                    self.logger.info(
                        "Partial TP fill; reducing SL quantity",
                        extra={
                            "event": "bracket.partial.tp",
                            "entry_id": entry_id,
                            "filled_qty": filled_quantity,
                            "total_qty": bracket_state.entry_quantity,
                            "cumulative_tp_filled": bracket_state.cumulative_tp_filled,
                        },
                    )
                    if self._auto_reduce_sl:
                        self._reduce_sl_quantity(
                            entry_id,
                            bracket_state,
                            filled_quantity,
                        )
                else:
                    all_filled = (
                        bracket_state.cumulative_tp_filled
                        >= bracket_state.entry_quantity
                        and bracket_state.entry_quantity > 0
                    )
                    if all_filled:
                        self._safe_cancel(bracket_state.stop_loss_order_id, "SL")
                        if METRICS_AVAILABLE and METRICS is not None:
                            METRICS.bracket_oco_cancellations.labels(
                                leg_type="TARGET"
                            ).inc()
                        self.unregister_bracket(entry_id)
                    else:
                        self.logger.info(
                            "TP filled but more targets remain; keeping SL active",
                            extra={
                                "event": "bracket.tp.partial_cumulative",
                                "entry_id": entry_id,
                                "cumulative_filled": bracket_state.cumulative_tp_filled,
                                "total_qty": bracket_state.entry_quantity,
                            },
                        )

                if filled_leg == 'TARGET' and bracket_state.tp2_order_id:
                    self.logger.info(
                        'TP1 filled; TP2 remains active',
                        extra={
                            'event': 'bracket.tp1.filled',
                            'entry_id': entry_id,
                            'tp2_id': bracket_state.tp2_order_id,
                        },
                    )
        except Exception as exc:  # noqa: BLE001
            self.logger.error(
                "Failure in BracketManager.handle_bracket_update: %s",
                exc,
                extra={
                    "event": "bracket.update.error",
                    "order_id": order_id,
                },
                exc_info=exc,
            )
            raise

    def _reduce_sl_quantity(
        self,
        entry_id: str,
        bracket_state: BracketState,
        filled_qty: int,
    ) -> None:
        """Reduce SL order quantity after partial TP fill.

        Args:
            entry_id: Entry order identifier.
            bracket_state: Bracket state containing SL order reference.
            filled_qty: Quantity filled at TP level.

        Returns:
            None.

        Raises:
            None.
        """

        self.logger.debug(
            'Entered BracketManager._reduce_sl_quantity',
            extra={
                'event': 'bracket.sl.reduce.enter',
                'entry_id': entry_id,
                'filled_qty': filled_qty,
            },
        )
        if not bracket_state.stop_loss_order_id:
            return

        try:
            remaining_qty = max(
                bracket_state.entry_quantity - bracket_state.cumulative_tp_filled,
                0,
            )

            if remaining_qty <= 0:
                self.logger.warning(
                    'No remaining quantity for SL reduction',
                    extra={
                        'event': 'bracket.sl.reduce.no_remaining',
                        'entry_id': entry_id,
                        'sl_order_id': bracket_state.stop_loss_order_id,
                    },
                )
                return

            if not isinstance(self.broker, SupportsModifyOrder):
                self.logger.warning(
                    'Broker does not support order modification',
                    extra={
                        'event': 'bracket.sl.reduce.unsupported',
                        'entry_id': entry_id,
                    },
                )
                return

            cast(SupportsModifyOrder, self.broker).modify_order(
                order_id=bracket_state.stop_loss_order_id,
                quantity=remaining_qty,
            )
            bracket_state.active_quantity = remaining_qty

            self.logger.info(
                'SL quantity reduced after partial TP',
                extra={
                    'event': 'bracket.sl.reduced',
                    'entry_id': entry_id,
                    'sl_order_id': bracket_state.stop_loss_order_id,
                    'filled_qty': filled_qty,
                    'remaining_qty': remaining_qty,
                    'cumulative_tp': bracket_state.cumulative_tp_filled,
                },
            )

            if (
                METRICS_AVAILABLE
                and METRICS is not None
                and hasattr(METRICS, 'bracket_sl_reductions')
            ):
                METRICS.bracket_sl_reductions.inc()

        except Exception as exc:  # noqa: BLE001
            self.logger.error(
                'Failed to reduce SL quantity: %s',
                exc,
                extra={
                    'event': 'bracket.sl.reduce.failed',
                    'entry_id': entry_id,
                    'sl_order_id': bracket_state.stop_loss_order_id,
                    'error': str(exc),
                },
                exc_info=exc,
            )

    def _safe_cancel(self, order_id: str | None, leg_name: str) -> None:
        """Cancel an order leg in an idempotent manner.

        Args:
            order_id: Order identifier to cancel.
            leg_name: Human-readable leg name for logging.

        Returns:
            None.

        Raises:
            Exception: Propagated when the broker reports unexpected failure.
        """

        self.logger.debug(
            "Entered BracketManager._safe_cancel",
            extra={
                "event": "bracket.cancel.enter",
                "order_id": order_id,
                "leg": leg_name,
            },
        )
        if not order_id:
            self.logger.debug(
                "Safe cancel skipped due to missing order id",
                extra={"event": "bracket.cancel.skip", "leg": leg_name},
            )
            return

        try:
            self.broker.cancel_order(order_id)
            self.logger.info(
                "Canceled %s order",
                leg_name,
                extra={
                    "event": "bracket.oco.canceled",
                    "order_id": order_id,
                    "leg": leg_name,
                },
            )
        except Exception as exc:  # noqa: BLE001
            error_msg = str(exc).lower()
            if "cancelled" in error_msg or "complete" in error_msg:
                self.logger.debug(
                    "%s already cancelled/filled",
                    leg_name,
                    extra={
                        "event": "bracket.oco.already_done",
                        "order_id": order_id,
                        "leg": leg_name,
                    },
                )
                return

            if METRICS_AVAILABLE and METRICS is not None:
                METRICS.bracket_oco_failures.inc()

            self.logger.error(
                "Failure in BracketManager._safe_cancel: %s",
                exc,
                extra={
                    "event": "bracket.oco.error",
                    "order_id": order_id,
                    "leg": leg_name,
                },
                exc_info=exc,
            )
            raise

    def unregister_bracket(self, entry_order_id: str) -> None:
        """Remove a bracket from tracking once closed.

        Args:
            entry_order_id: Entry order identifier to remove.

        Returns:
            None.

        Raises:
            Exception: Propagated when cleanup fails unexpectedly.
        """

        self.logger.debug(
            "Entered BracketManager.unregister_bracket",
            extra={
                "event": "bracket.unregister.enter",
                "entry_id": entry_order_id,
            },
        )
        try:
            with self._global_lock:
                state = self.brackets.pop(entry_order_id, None)
                if state:
                    if state.stop_loss_order_id:
                        self._order_to_entry.pop(state.stop_loss_order_id, None)
                    if state.target_order_id:
                        self._order_to_entry.pop(state.target_order_id, None)
                    if state.tp2_order_id:
                        self._order_to_entry.pop(state.tp2_order_id, None)

            if state:
                self.logger.info(
                    "Bracket unregistered",
                    extra={
                        "event": "bracket.unregister",
                        "entry_id": entry_order_id,
                        "lifetime_seconds": time.time() - state.created_at,
                    },
                )
                if METRICS_AVAILABLE and METRICS is not None:
                    METRICS.bracket_active.dec()
            else:
                self.logger.debug(
                    "No bracket state found during unregister",
                    extra={
                        "event": "bracket.unregister.miss",
                        "entry_id": entry_order_id,
                    },
                )
        except Exception as exc:  # noqa: BLE001
            self.logger.error(
                "Failure in BracketManager.unregister_bracket: %s",
                exc,
                extra={
                    "event": "bracket.unregister.error",
                    "entry_id": entry_order_id,
                },
            )
            raise

    def cleanup_stale_brackets(self, max_age_seconds: float = 86400) -> int:
        """Remove brackets older than ``max_age_seconds``.

        Args:
            max_age_seconds: Maximum age in seconds before cleanup.

        Returns:
            int: Number of brackets cleaned up.

        Raises:
            Exception: Propagated if cleanup fails unexpectedly.
        """

        self.logger.debug(
            "Entered BracketManager.cleanup_stale_brackets",
            extra={
                "event": "bracket.cleanup.enter",
                "max_age": max_age_seconds,
            },
        )
        now = time.time()
        to_remove: list[tuple[str, float]] = []

        try:
            with self._global_lock:
                for entry_id, state in self.brackets.items():
                    age = now - state.created_at
                    if age > max_age_seconds:
                        to_remove.append((entry_id, age))

            for entry_id, age in to_remove:
                self.logger.warning(
                    "Removing stale bracket",
                    extra={
                        "event": "bracket.cleanup.stale",
                        "entry_id": entry_id,
                        "age_seconds": age,
                    },
                )
                self.unregister_bracket(entry_id)

            if to_remove and METRICS_AVAILABLE and METRICS is not None:
                METRICS.bracket_stale_cleanups.inc(len(to_remove))

            return len(to_remove)
        except Exception as exc:  # noqa: BLE001
            self.logger.error(
                "Failure in BracketManager.cleanup_stale_brackets: %s",
                exc,
                extra={
                    "event": "bracket.cleanup.error",
                    "max_age": max_age_seconds,
                },
                exc_info=exc,
            )
            raise

    def get_stats(self) -> dict[str, Any]:
        """Get current bracket manager statistics.

        Args:
            None.

        Returns:
            dict[str, Any]: Dictionary of statistics.

        Raises:
            None.
        """

        self.logger.debug(
            "Entered BracketManager.get_stats",
            extra={"event": "bracket.stats.enter"},
        )
        with self._global_lock:
            return {
                "active_brackets": len(self.brackets),
                "reverse_index_size": len(self._order_to_entry),
                "auto_reduce_sl": self._auto_reduce_sl,
            }
