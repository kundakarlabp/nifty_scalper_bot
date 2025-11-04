"""Central coordination hub for order execution."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Literal, Mapping, cast

from nifty_scalper_bot.execution.execution_router import (
    ExecutionResult,
    ExecutionRouter,
)
from nifty_scalper_bot.execution.lifecycle_manager import LifecycleManager
from nifty_scalper_bot.execution.metrics import VALIDATION_FAILURES
from nifty_scalper_bot.execution.order_queue import (
    OrderIntent,
    OrderQueue,
    OrderRequest,
)
from nifty_scalper_bot.execution.post_fill_monitor import PostFillMonitor
from nifty_scalper_bot.execution.preflight_validator import (
    PreFlightValidator,
    ValidationResult,
)
from nifty_scalper_bot.execution.state_tracker import StateTracker
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


class OrderExecutionHub:
    """Coordinate order intake, validation, routing, and reconciliation."""

    def __init__(
        self,
        *,
        state_tracker: StateTracker,
        preflight_validator: PreFlightValidator,
        lifecycle_manager: LifecycleManager,
        order_queue: OrderQueue,
        execution_router: ExecutionRouter,
        post_fill_monitor: PostFillMonitor,
        data_hub: Any | None = None,
        regime_manager: Any | None = None,
        risk_manager: Any | None = None,
    ) -> None:
        """Store dependencies and initialise bookkeeping.

        Args:
            state_tracker: Tracker maintaining execution state and positions.
            preflight_validator: Validator gating order requests.
            lifecycle_manager: Manager handling lifecycle transitions.
            order_queue: Shared priority queue for execution requests.
            execution_router: Router dispatching orders to executors.
            post_fill_monitor: Monitor reconciling broker state.
            data_hub: Optional data hub reference for diagnostics.
            regime_manager: Optional regime manager used for gating context.
            risk_manager: Optional risk manager providing circuit breaker state.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered OrderExecutionHub.__init__",
            extra={"event": "order_execution_hub_init"},
        )
        self._state_tracker = state_tracker
        self._preflight_validator = preflight_validator
        self._lifecycle_manager = lifecycle_manager
        self._order_queue = order_queue
        self._execution_router = execution_router
        self._post_fill_monitor = post_fill_monitor
        self._data_hub = data_hub
        self._regime_manager = (
            regime_manager
            if regime_manager is not None
            else getattr(preflight_validator, "_regime_manager", None)
        )
        self._risk_manager = (
            risk_manager
            if risk_manager is not None
            else getattr(preflight_validator, "_risk_manager", None)
        )
        self._worker_task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._stats: dict[str, int] = {
            "submitted": 0,
            "validated": 0,
            "rejected": 0,
            "executed": 0,
            "failed": 0,
            "circuit_breaker_pauses": 0,
        }
        self._circuit_pause_interval = 5.0
        self._last_circuit_log = 0.0
        self._order_log = Path("data/order_submissions.jsonl")
        self._restore_pending_orders()

    async def start(self) -> None:
        """Start lifecycle dependencies and queue worker.

        Args:
            None.

        Returns:
            None.

        Raises:
            None. Errors are logged and the worker is not started.
        """

        LOGGER.debug(
            "Entered OrderExecutionHub.start",
            extra={"event": "order_execution_hub_start"},
        )
        if self._worker_task is not None and not self._worker_task.done():
            return
        await self._lifecycle_manager.start()
        await self._post_fill_monitor.start()
        self._stop_event.clear()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in OrderExecutionHub.start: %s",
                exc,
                extra={"event": "order_execution_hub_loop_missing"},
                exc_info=exc,
            )
            return
        self._worker_task = loop.create_task(self._worker_loop())

    async def shutdown(self) -> None:
        """Shutdown worker and dependent subsystems.

        Args:
            None.

        Returns:
            None.

        Raises:
            None. Exceptions are logged during teardown.
        """

        LOGGER.debug(
            "Entered OrderExecutionHub.shutdown",
            extra={"event": "order_execution_hub_shutdown"},
        )
        self._stop_event.set()
        task = self._worker_task
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        await self._lifecycle_manager.shutdown()
        await self._post_fill_monitor.stop()

    def submit_order_request(self, request: OrderRequest) -> str:
        """Submit ``request`` to the shared queue for processing.

        Args:
            request: OrderRequest instance produced by strategies.

        Returns:
            str: Generated identifier for tracing the submission.

        Raises:
            RuntimeError: When enqueueing fails unexpectedly.
        """

        LOGGER.debug(
            "Entered OrderExecutionHub.submit_order_request",
            extra={
                "event": "order_execution_hub_submit",
                "symbol": request.symbol,
                "intent": request.intent,
            },
        )
        self._stats["submitted"] += 1
        try:
            self._order_queue.submit_order_request(request)
            self._persist_order(request, status="pending")
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in OrderExecutionHub.submit_order_request: %s",
                exc,
                extra={"event": "order_execution_hub_submit_error"},
                exc_info=exc,
            )
            self._stats["failed"] += 1
            raise
        return f"req_{int(request.created_ts * 1000)}"

    def _persist_order(self, request: OrderRequest, status: str) -> None:
        """Persist order submission metadata for recovery.

        Args:
            request: Order request being persisted.
            status: Lifecycle status persisted alongside the submission.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered OrderExecutionHub._persist_order",
            extra={
                "event": "order_execution_hub_persist_order",
                "symbol": request.symbol,
                "status": status,
            },
        )
        request_id = f"req_{int(request.created_ts * 1000)}"
        try:
            self._order_log.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "timestamp": time.time(),
                "created_ts": request.created_ts,
                "request_id": request_id,
                "symbol": request.symbol,
                "side": request.side,
                "quantity": request.quantity,
                "intent": request.intent,
                "status": status,
            }
            with self._order_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry))
                handle.write("\n")
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in OrderExecutionHub._persist_order: %s",
                exc,
                extra={
                    "event": "order_execution_hub_persist_error",
                    "symbol": request.symbol,
                },
                exc_info=exc,
            )

    def _restore_pending_orders(self) -> None:
        """Reload pending order submissions from the persistence log.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered OrderExecutionHub._restore_pending_orders",
            extra={"event": "order_execution_hub_restore_enter"},
        )
        if not self._order_log.exists():
            return
        try:
            entries: dict[str, dict[str, Any]] = {}
            with self._order_log.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    payload = line.strip()
                    if not payload:
                        continue
                    try:
                        entry = json.loads(payload)
                    except json.JSONDecodeError as exc:
                        LOGGER.error(
                            (
                                "Failure in OrderExecutionHub._restore_pending_orders "
                                "parse: %s"
                            ),
                            exc,
                            extra={
                                "event": "order_execution_hub_restore_parse_error",
                                "line": line_number,
                            },
                            exc_info=exc,
                        )
                        continue
                    request_id = str(entry.get("request_id") or "").strip()
                    if not request_id:
                        request_id = f"legacy-{line_number}"
                        entry["request_id"] = request_id
                    entries[request_id] = entry
            restored = 0
            for entry in entries.values():
                if entry.get("status") != "pending":
                    continue
                try:
                    symbol = str(entry.get("symbol") or "").strip()
                    side = str(entry.get("side") or "").strip()
                    quantity = int(entry.get("quantity") or 0)
                    intent = str(entry.get("intent") or "ENTRY").strip() or "ENTRY"
                    if not symbol or not side or quantity <= 0:
                        LOGGER.warning(
                            "order_restore_skipped_invalid_entry",
                            extra={
                                "event": "order_execution_hub_restore_skipped",
                                "request_id": entry.get("request_id"),
                            },
                        )
                        continue
                    if side not in {"BUY", "SELL"}:
                        LOGGER.warning(
                            "order_restore_skipped_invalid_side",
                            extra={
                                "event": "order_execution_hub_restore_invalid_side",
                                "side": side,
                                "request_id": entry.get("request_id"),
                            },
                        )
                        continue
                    valid_intents: set[OrderIntent] = {
                        "ENTRY",
                        "EXIT_SL",
                        "EXIT_TP1",
                        "EXIT_TP2",
                        "ADJUST_TRAIL",
                    }
                    if intent not in valid_intents:
                        LOGGER.warning(
                            "order_restore_skipped_invalid_intent",
                            extra={
                                "event": "order_execution_hub_restore_invalid_intent",
                                "intent": intent,
                                "request_id": entry.get("request_id"),
                            },
                        )
                        continue
                    request = OrderRequest(
                        symbol=symbol,
                        side=cast(Literal["BUY", "SELL"], side),
                        quantity=quantity,
                        intent=cast(OrderIntent, intent),
                        source="restore",
                    )
                    self._order_queue.submit_order_request(request)
                    restored += 1
                    LOGGER.info(
                        "Condition met: restored pending order",
                        extra={
                            "event": "order_execution_hub_restored_order",
                            "symbol": request.symbol,
                            "intent": request.intent,
                            "request_id": entry.get("request_id"),
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    LOGGER.error(
                        (
                            "Failure in OrderExecutionHub._restore_pending_orders "
                            "enqueue: %s"
                        ),
                        exc,
                        extra={
                            "event": "order_execution_hub_restore_enqueue_error",
                            "request_id": entry.get("request_id"),
                        },
                        exc_info=exc,
                    )
            if restored:
                LOGGER.info(
                    "Condition met: restored_pending_orders",
                    extra={
                        "event": "order_execution_hub_restore_complete",
                        "restored": restored,
                    },
                )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in OrderExecutionHub._restore_pending_orders: %s",
                exc,
                extra={"event": "order_execution_hub_restore_error"},
                exc_info=exc,
            )

    def get_position_state(self, symbol: str) -> dict[str, Any] | None:
        """Return position state stored for ``symbol``.

        Args:
            symbol: Trading symbol to query.

        Returns:
            Optional dictionary describing tracked position state.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered OrderExecutionHub.get_position_state",
            extra={"event": "order_execution_hub_get_position", "symbol": symbol},
        )
        return self._state_tracker.get_position_state(symbol)

    def emergency_stop(self) -> dict[str, Any]:
        """Pause queue processing and submit market exits for open positions.

        Args:
            None.

        Returns:
            dict[str, Any]: Results including paused flag and closed symbols.

        Raises:
            None. Failures are logged per position.
        """

        LOGGER.debug(
            "Entered OrderExecutionHub.emergency_stop",
            extra={"event": "order_execution_hub_emergency_stop"},
        )
        snapshot: dict[str, Any] = {
            "timestamp": time.monotonic(),
            "positions_closed": [],
            "queue_paused": False,
        }
        try:
            self._order_queue.pause()
            snapshot["queue_paused"] = True
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in OrderExecutionHub.emergency_stop pause: %s",
                exc,
                extra={"event": "order_execution_hub_emergency_pause_error"},
                exc_info=exc,
            )
        try:
            for position in self._state_tracker.get_open_positions():
                symbol = str(position.get("symbol") or "").strip()
                if not symbol:
                    continue
                try:
                    self._lifecycle_manager.exit_at_market(symbol, "EMERGENCY_STOP")
                    positions_closed = snapshot.setdefault("positions_closed", [])
                    if isinstance(positions_closed, list):
                        positions_closed.append(symbol)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.error(
                        "Failure in OrderExecutionHub.emergency_stop exit: %s",
                        exc,
                        extra={
                            "event": "order_execution_hub_emergency_exit_error",
                            "symbol": symbol,
                        },
                        exc_info=exc,
                    )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in OrderExecutionHub.emergency_stop iterate: %s",
                exc,
                extra={"event": "order_execution_hub_emergency_iter_error"},
                exc_info=exc,
            )
        return snapshot

    def get_stats(self) -> dict[str, Any]:
        """Return execution statistics.

        Args:
            None.

        Returns:
            dict[str, Any]: Aggregated execution and reconciliation metrics.

        Raises:
            None.
        """

        queue_depth = len(self._order_queue.get_queue_snapshot())
        router_stats = self._execution_router.get_stats()
        monitor_stats = self._post_fill_monitor.get_stats()
        return {
            **self._stats,
            "queue_depth": queue_depth,
            "router": router_stats,
            "reconciliation": monitor_stats,
        }

    async def _worker_loop(self) -> None:
        """Process queued requests until shutdown signal.

        Args:
            None.

        Returns:
            None.

        Raises:
            None. Exceptions are logged and the loop exits gracefully.
        """

        LOGGER.debug(
            "Entered OrderExecutionHub._worker_loop",
            extra={"event": "order_execution_hub_worker_start"},
        )
        try:
            while not self._stop_event.is_set():
                if self._should_halt_processing():
                    await asyncio.sleep(self._circuit_pause_interval)
                    continue
                request = await asyncio.to_thread(
                    self._order_queue.get_next_request, 0.5
                )
                if request is None:
                    continue
                validation = self._run_preflight(request)
                if validation is None:
                    continue
                await self._dispatch_request(request)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in OrderExecutionHub._worker_loop: %s",
                exc,
                extra={"event": "order_execution_hub_worker_error"},
                exc_info=exc,
            )

    def _run_preflight(self, request: OrderRequest) -> ValidationResult | None:
        """Run preflight validation returning the result when allowed.

        Args:
            request: OrderRequest pending validation.

        Returns:
            ValidationResult when allowed; otherwise ``None``.

        Raises:
            None. Errors are logged and ``None`` is returned.
        """

        LOGGER.debug(
            "Entered OrderExecutionHub._run_preflight",
            extra={
                "event": "order_execution_hub_preflight",
                "symbol": request.symbol,
            },
        )
        try:
            outcome = self._preflight_validator.validate(
                request.symbol,
                context={"intent": request.intent, "quantity": request.quantity},
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in OrderExecutionHub._run_preflight: %s",
                exc,
                extra={"event": "order_execution_hub_preflight_error"},
                exc_info=exc,
            )
            VALIDATION_FAILURES.labels(
                symbol=request.symbol, gate="preflight", level="ERROR"
            ).inc()
            self._stats["rejected"] += 1
            self._persist_order(request, status="rejected")
            return None
        if not outcome.allowed:
            LOGGER.info(
                "Condition met: order rejected by preflight",
                extra={
                    "event": "order_execution_hub_preflight_block",
                    "symbol": request.symbol,
                    "reasons": outcome.reasons,
                },
            )
            for reason in outcome.reasons:
                gate_name = str(reason.get("gate") or "unknown")
                VALIDATION_FAILURES.labels(
                    symbol=request.symbol,
                    gate=gate_name,
                    level=str(outcome.blocking_level or "UNKNOWN"),
                ).inc()
            self._stats["rejected"] += 1
            self._persist_order(request, status="rejected")
            return None
        self._stats["validated"] += 1
        return outcome

    async def _dispatch_request(self, request: OrderRequest) -> None:
        """Route ``request`` through the execution router.

        Args:
            request: OrderRequest to dispatch via the router.

        Returns:
            None.

        Raises:
            None. Errors are logged when execution fails.
        """

        LOGGER.debug(
            "Entered OrderExecutionHub._dispatch_request",
            extra={
                "event": "order_execution_hub_dispatch",
                "symbol": request.symbol,
            },
        )
        try:
            result = await asyncio.to_thread(self._execution_router.execute, request)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in OrderExecutionHub._dispatch_request: %s",
                exc,
                extra={"event": "order_execution_hub_dispatch_error"},
                exc_info=exc,
            )
            self._stats["failed"] += 1
            self._persist_order(request, status="failed")
            return
        self._handle_execution_result(request, result)

    def _handle_execution_result(
        self, request: OrderRequest, result: ExecutionResult
    ) -> None:
        """Record execution results and trigger lifecycle transitions.

        Args:
            request: OrderRequest that produced the result.
            result: ExecutionResult returned by the router.

        Returns:
            None.

        Raises:
            None. Failures are logged without raising.
        """

        LOGGER.debug(
            "Entered OrderExecutionHub._handle_execution_result",
            extra={
                "event": "order_execution_hub_handle_result",
                "symbol": request.symbol,
                "status": result.status,
            },
        )
        final_status = (result.status or "unknown").lower()
        if result.status in {"FILLED", "SUBMITTED"} and result.order_id:
            self._stats["executed"] += 1
            self._record_order(request, result)
            if request.intent == "ENTRY":
                self._trigger_lifecycle_on_entry(request, result)
            LOGGER.info(
                "Condition met: order execution recorded",
                extra={
                    "event": "order_execution_hub_executed",
                    "symbol": request.symbol,
                    "status": result.status,
                },
            )
            self._persist_order(request, status=final_status or "completed")
            return
        self._stats["failed"] += 1
        LOGGER.warning(
            "Order execution failed",
            extra={
                "event": "order_execution_hub_failed",
                "symbol": request.symbol,
                "status": result.status,
                "reason": result.rejection_reason,
            },
        )
        self._persist_order(request, status=final_status or "failed")

    def _record_order(self, request: OrderRequest, result: ExecutionResult) -> None:
        """Persist execution details to the state tracker.

        Args:
            request: OrderRequest that was executed.
            result: ExecutionResult describing fills and status.

        Returns:
            None.

        Raises:
            None. Errors are logged for observability.
        """

        fill_price = self._safe_float(
            result.fill_price, self._safe_float(request.price)
        )
        payload = {
            "order_id": result.order_id,
            "symbol": request.symbol,
            "side": request.side,
            "quantity": request.quantity,
            "status": result.status.lower(),
            "fill_price": fill_price,
            "intent": request.intent,
            "timestamp": time.time(),
            "parent_id": request.parent_id,
        }
        try:
            self._state_tracker.add_order(payload)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in OrderExecutionHub._record_order: %s",
                exc,
                extra={"event": "order_execution_hub_record_order_error"},
                exc_info=exc,
            )
        quantity_signed = int(request.quantity)
        if request.side.upper() == "SELL":
            quantity_signed *= -1
        try:
            existing = self._state_tracker.get_position_state(request.symbol) or {}
            current_qty = int(existing.get("quantity", 0))
        except Exception:  # pragma: no cover - defensive fallback
            existing = {}
            current_qty = 0
        new_quantity = current_qty + quantity_signed
        intent_upper = request.intent.upper()
        try:
            if intent_upper.startswith("EXIT") and new_quantity == 0:
                self._state_tracker.update_position(request.symbol, {"delete": True})
                return
            updates: dict[str, Any] = {"quantity": new_quantity}
            if intent_upper == "ENTRY" and current_qty == 0:
                updates.update(
                    {
                        "entry_price": fill_price,
                        "entry_time": time.time(),
                        "lifecycle_stage": "ENTRY",
                    }
                )
            elif intent_upper.startswith("EXIT"):
                updates["lifecycle_stage"] = "EXIT"
            self._state_tracker.update_position(request.symbol, updates)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in OrderExecutionHub._record_order update: %s",
                exc,
                extra={
                    "event": "order_execution_hub_record_position_error",
                    "symbol": request.symbol,
                },
                exc_info=exc,
            )

    def _trigger_lifecycle_on_entry(
        self, request: OrderRequest, result: ExecutionResult
    ) -> None:
        """Initialise lifecycle manager for a filled entry.

        Args:
            request: Original entry order request.
            result: ExecutionResult providing fill information.

        Returns:
            None.

        Raises:
            None. Failures are logged for diagnostics.
        """

        metadata = request.metadata or {}
        entry_price = result.fill_price or request.price or 0.0
        quantity = result.fill_quantity or request.quantity
        atr_value = self._get_atr_for_symbol(request.symbol, metadata)
        regime = self._get_current_regime(request.symbol, metadata)
        iv_value = metadata.get("iv")
        try:
            self._lifecycle_manager.on_fill(
                symbol=request.symbol,
                entry_price=float(entry_price),
                quantity=int(quantity),
                atr=atr_value,
                regime=regime,
                iv=float(iv_value) if iv_value is not None else None,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in OrderExecutionHub._trigger_lifecycle_on_entry: %s",
                exc,
                extra={
                    "event": "order_execution_hub_lifecycle_error",
                    "symbol": request.symbol,
                },
                exc_info=exc,
            )

    def _should_halt_processing(self) -> bool:
        """Return ``True`` when queue processing should pause.

        Args:
            None.

        Returns:
            bool: ``True`` when a circuit breaker or pause condition is active.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered OrderExecutionHub._should_halt_processing",
            extra={"event": "order_execution_hub_should_halt"},
        )
        try:
            risk_manager = self._risk_manager
            if risk_manager is None:
                risk_manager = getattr(self._preflight_validator, "_risk_manager", None)
            if risk_manager is None:
                return False
            active_check = getattr(risk_manager, "is_circuit_breaker_active", None)
            breaker_check = getattr(risk_manager, "is_circuit_breaker_tripped", None)
            breaker_active = bool(active_check()) if callable(active_check) else False
            reason = ""
            if callable(breaker_check):
                breaker_result = breaker_check()
                if isinstance(breaker_result, tuple):
                    tripped = bool(breaker_result[0])
                    detail = (
                        str(breaker_result[1])
                        if len(breaker_result) > 1 and breaker_result[1] is not None
                        else ""
                    )
                else:
                    tripped = bool(breaker_result)
                    detail = ""
                if tripped:
                    breaker_active = True
                    reason = detail or "circuit_breaker"
                elif breaker_active and detail:
                    reason = detail
            if breaker_active:
                now = time.monotonic()
                if now - self._last_circuit_log >= self._circuit_pause_interval:
                    LOGGER.info(
                        "Condition met: circuit breaker active",
                        extra={
                            "event": "order_execution_hub_circuit_open",
                            "reason": reason or "circuit_breaker",
                        },
                    )
                    self._last_circuit_log = now
                self._stats["circuit_breaker_pauses"] += 1
                return True
            return False
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in OrderExecutionHub._should_halt_processing: %s",
                exc,
                extra={"event": "order_execution_hub_circuit_check_error"},
                exc_info=exc,
            )
            return False

    def _get_atr_for_symbol(
        self, symbol: str, metadata: Mapping[str, Any] | None = None
    ) -> float:
        """Return the ATR value for ``symbol`` from metadata or trackers.

        Args:
            symbol: Instrument identifier for lookup.
            metadata: Optional execution metadata payload.

        Returns:
            float: Discovered ATR value or symbol-specific default.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered OrderExecutionHub._get_atr_for_symbol",
            extra={"event": "order_execution_hub_resolve_atr", "symbol": symbol},
        )
        default_atr = 10.0
        symbol_upper = symbol.upper()
        if "NIFTY" in symbol_upper and "BANK" not in symbol_upper:
            default_atr = 50.0
        elif "BANKNIFTY" in symbol_upper or "FINNIFTY" in symbol_upper:
            default_atr = 150.0
        try:
            if metadata:
                for key in ("atr", "avg_true_range", "atr_value"):
                    if key in metadata and metadata[key] is not None:
                        return self._safe_float(metadata[key], default_atr)
            position_state = self._state_tracker.get_position_state(symbol)
            if position_state is not None:
                atr_state = position_state.get("atr")
                if atr_state is not None:
                    return self._safe_float(atr_state, default_atr)
            data_hub = self._data_hub
            if data_hub is not None:
                indicator_fn = getattr(data_hub, "get_indicator", None)
                if callable(indicator_fn):
                    atr_value = indicator_fn(symbol, "atr")
                    if atr_value is not None:
                        return self._safe_float(atr_value, default_atr)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in OrderExecutionHub._get_atr_for_symbol: %s",
                exc,
                extra={
                    "event": "order_execution_hub_resolve_atr_error",
                    "symbol": symbol,
                },
                exc_info=exc,
            )
        LOGGER.info(
            "Condition met: default ATR applied",
            extra={
                "event": "order_execution_hub_default_atr",
                "symbol": symbol,
                "atr": default_atr,
            },
        )
        return default_atr

    def _get_current_regime(
        self, symbol: str, metadata: Mapping[str, Any] | None = None
    ) -> str:
        """Return the most recent regime label with safe fallbacks.

        Args:
            symbol: Instrument identifier for lookup.
            metadata: Optional execution metadata payload.

        Returns:
            str: Regime label when available; defaults to ``"NEUTRAL"``.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered OrderExecutionHub._get_current_regime",
            extra={"event": "order_execution_hub_resolve_regime"},
        )
        try:
            if metadata and metadata.get("regime"):
                return str(metadata.get("regime"))
            regime_manager = self._regime_manager
            if regime_manager is None:
                regime_manager = getattr(
                    self._preflight_validator, "_regime_manager", None
                )
            if regime_manager is not None:
                getter = getattr(regime_manager, "get_current_regime", None)
                if callable(getter):
                    regime_value = getter()
                    if regime_value:
                        return str(regime_value)
            state_snapshot = self._state_tracker.get_position_state(symbol)
            if state_snapshot is not None and state_snapshot.get("regime"):
                return str(state_snapshot["regime"])
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in OrderExecutionHub._get_current_regime: %s",
                exc,
                extra={"event": "order_execution_hub_resolve_regime_error"},
                exc_info=exc,
            )
        LOGGER.info(
            "Condition met: default regime applied",
            extra={
                "event": "order_execution_hub_default_regime",
                "symbol": symbol,
                "regime": "NEUTRAL",
            },
        )
        return "NEUTRAL"

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """Return ``value`` as float with ``default`` fallback.

        Args:
            value: Potential float-like input.
            default: Fallback value when coercion fails.

        Returns:
            float: Coerced float or the provided default.

        Raises:
            None.
        """

        try:
            return float(value)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return float(default)


__all__ = ["OrderExecutionHub"]
