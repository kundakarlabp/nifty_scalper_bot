"""Background reconciliation monitor for execution state alignment."""

from __future__ import annotations

import asyncio
from typing import Any

from nifty_scalper_bot.utils.logging import get_logger


class PostFillMonitor:
    """Monitor and reconcile broker positions with local state."""

    def __init__(
        self,
        *,
        broker_client: Any,
        state_tracker: Any,
        interval_sec: int = 30,
        alert_on_mismatch: bool = True,
    ) -> None:
        """Initialise the monitor with broker and state dependencies.

        Args:
            broker_client: Broker API client used for position fetches.
            state_tracker: Local execution state tracker implementation.
            interval_sec: Interval between reconciliation cycles.
            alert_on_mismatch: Flag to emit warnings when mismatches appear.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger = get_logger(__name__)
        self._logger.debug(
            "Entered PostFillMonitor.__init__",
            extra={"event": "post_fill_monitor_init", "interval": interval_sec},
        )
        self._broker = broker_client
        self._state_tracker = state_tracker
        self._interval = max(10, int(interval_sec))
        self._alert_on_mismatch = alert_on_mismatch
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._mismatch_count = 0

    async def start(self) -> None:
        """Start the background reconciliation loop.

        Args:
            None.

        Returns:
            None.

        Raises:
            None. Errors are logged when the loop cannot start.
        """

        self._logger.debug(
            "Entered PostFillMonitor.start",
            extra={"event": "post_fill_monitor_start"},
        )
        if self._task is not None and not self._task.done():
            return
        self._stop_event.clear()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in PostFillMonitor.start: %s",
                exc,
                extra={"event": "post_fill_monitor_loop_missing"},
                exc_info=exc,
            )
            return
        self._task = loop.create_task(self._reconciliation_loop())

    async def stop(self) -> None:
        """Stop the background reconciliation loop.

        Args:
            None.

        Returns:
            None.

        Raises:
            None. Cancellation errors are suppressed.
        """

        self._logger.debug(
            "Entered PostFillMonitor.stop",
            extra={"event": "post_fill_monitor_stop"},
        )
        self._stop_event.set()
        task = self._task
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def reconcile(self) -> None:
        """Perform a single reconciliation cycle.

        Args:
            None.

        Returns:
            None.

        Raises:
            None. Exceptions are logged for diagnostics.
        """

        self._logger.debug(
            "Entered PostFillMonitor.reconcile",
            extra={"event": "post_fill_monitor_reconcile"},
        )
        try:
            broker_positions = await asyncio.to_thread(self._fetch_broker_positions)
            broker_map = self._normalise_positions(broker_positions)
            local_snapshot = list(await self._state_tracker.get_open_positions())
            local_positions = self._normalise_positions(local_snapshot)
            mismatches = self._diff_positions(broker_map, local_positions)
            self._mismatch_count = len(mismatches)
            if mismatches and self._alert_on_mismatch:
                self._logger.warning(
                    "post_fill_reconcile_mismatch",
                    extra={
                        "event": "post_fill_reconcile_mismatch",
                        "count": len(mismatches),
                        "mismatches": mismatches,
                    },
                )
            elif not mismatches:
                self._logger.debug(
                    "Condition met: reconciliation clean",
                    extra={"event": "post_fill_reconcile_clean"},
                )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in PostFillMonitor.reconcile: %s",
                exc,
                extra={"event": "post_fill_monitor_reconcile_error"},
                exc_info=exc,
            )

    async def _reconciliation_loop(self) -> None:
        """Run reconciliation periodically until stopped.

        Args:
            None.

        Returns:
            None.

        Raises:
            None. Exceptions are logged for observability.
        """

        self._logger.debug(
            "Entered PostFillMonitor._reconciliation_loop",
            extra={"event": "post_fill_monitor_loop_start"},
        )
        try:
            while not self._stop_event.is_set():
                await asyncio.sleep(float(self._interval))
                await self.reconcile()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in PostFillMonitor._reconciliation_loop: %s",
                exc,
                extra={"event": "post_fill_monitor_loop_error"},
                exc_info=exc,
            )

    def get_stats(self) -> dict[str, int]:
        """Return reconciliation statistics for observability.

        Args:
            None.

        Returns:
            dict[str, int]: Summary with mismatch count and interval.

        Raises:
            None.
        """

        return {
            "total_mismatches": int(self._mismatch_count),
            "interval_sec": int(self._interval),
        }

    # ------------------------------------------------------------------
    def _fetch_broker_positions(self) -> list[dict[str, Any]]:
        """Fetch positions from the broker client defensively.

        Args:
            None.

        Returns:
            list[dict[str, Any]]: Raw broker position payloads.

        Raises:
            None. Failures are logged and an empty list returned.
        """

        self._logger.debug(
            "Entered PostFillMonitor._fetch_broker_positions",
            extra={"event": "post_fill_monitor_fetch"},
        )
        try:
            if hasattr(self._broker, "get_positions"):
                payload = self._broker.get_positions()
            elif hasattr(self._broker, "positions"):
                payload = self._broker.positions()
            else:
                payload = []
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in PostFillMonitor._fetch_broker_positions: %s",
                exc,
                extra={"event": "post_fill_monitor_fetch_error"},
                exc_info=exc,
            )
            return []
        return list(payload or [])

    def _normalise_positions(
        self, positions: list[dict[str, Any]] | list[Any]
    ) -> dict[str, dict[str, float | int]]:
        """Normalise position payloads into a symbol-indexed mapping.

        Args:
            positions: Iterable of broker or local position payloads.

        Returns:
            dict[str, dict[str, float | int]]: Mapping keyed by symbol.

        Raises:
            None.
        """

        normalised: dict[str, dict[str, float | int]] = {}
        for entry in positions:
            if not isinstance(entry, dict):
                continue
            symbol_raw = entry.get("tradingsymbol") or entry.get("symbol")
            if not symbol_raw:
                continue
            symbol = str(symbol_raw).strip().upper()
            if not symbol:
                continue
            quantity_raw = entry.get("net_qty") or entry.get("quantity")
            quantity = self._coerce_int(quantity_raw)
            if quantity == 0:
                continue
            price_raw = entry.get("average_price") or entry.get("avg_price")
            avg_price = self._coerce_float(price_raw)
            normalised[symbol] = {"quantity": quantity, "avg_price": avg_price}
        return normalised

    def _diff_positions(
        self,
        broker_map: dict[str, dict[str, float | int]],
        local_map: dict[str, dict[str, float | int]],
    ) -> list[dict[str, Any]]:
        """Return mismatch descriptions and synchronise local state.

        Args:
            broker_map: Broker positions keyed by symbol.
            local_map: Locally tracked positions keyed by symbol.

        Returns:
            list[dict[str, Any]]: Detected mismatch descriptors.

        Raises:
            None.
        """

        mismatches: list[dict[str, Any]] = []
        symbols = set(broker_map).union(local_map)
        for symbol in symbols:
            broker_pos = broker_map.get(symbol)
            local_pos = local_map.get(symbol)
            if broker_pos and not local_pos:
                mismatches.append({"symbol": symbol, "type": "missing_local"})
                self._safe_update(
                    symbol,
                    int(broker_pos["quantity"]),
                    float(broker_pos["avg_price"]),
                )
                continue
            if local_pos and not broker_pos:
                mismatches.append({"symbol": symbol, "type": "missing_broker"})
                self._safe_update(symbol, 0, 0.0)
                continue
            if not broker_pos or not local_pos:
                continue
            broker_qty = int(broker_pos["quantity"])
            local_qty = int(self._coerce_int(local_pos.get("quantity")))
            if broker_qty != local_qty:
                mismatches.append(
                    {
                        "symbol": symbol,
                        "type": "quantity_mismatch",
                        "broker": broker_pos,
                        "local": local_pos,
                    }
                )
                self._safe_update(
                    symbol,
                    broker_qty,
                    float(broker_pos["avg_price"]),
                )
        return mismatches

    def _safe_update(self, symbol: str, quantity: int, avg_price: float) -> None:
        """Update the local state tracker with defensive logging.

        Args:
            symbol: Trading symbol for the position update.
            quantity: Quantity to store in the tracker.
            avg_price: Average price associated with the position.

        Returns:
            None.

        Raises:
            None. Errors are logged when updates fail.
        """

        try:
            self._state_tracker.update_position(
                symbol,
                {"quantity": quantity, "entry_price": avg_price},
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in PostFillMonitor._safe_update: %s",
                exc,
                extra={"event": "post_fill_monitor_update_error", "symbol": symbol},
                exc_info=exc,
            )

    @staticmethod
    def _coerce_int(value: Any) -> int:
        """Return ``value`` coerced to ``int`` when possible.

        Args:
            value: Value to coerce into an integer.

        Returns:
            int: Coerced integer or ``0`` on failure.

        Raises:
            None.
        """

        try:
            return int(float(value))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return 0

    @staticmethod
    def _coerce_float(value: Any) -> float:
        """Return ``value`` coerced to ``float`` when possible.

        Args:
            value: Value to coerce into a float.

        Returns:
            float: Coerced float or ``0.0`` on failure.

        Raises:
            None.
        """

        try:
            return float(value)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return 0.0


__all__ = ["PostFillMonitor"]
