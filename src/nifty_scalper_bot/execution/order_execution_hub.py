"""Central coordination hub for order execution."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Mapping, cast

from nifty_scalper_bot.core.trade_manager import TradeManager
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
from nifty_scalper_bot.strategies.signal_generator import Signal
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


class TradeState(str, Enum):
    """Explicit lifecycle states for a trade managed by ExecutionEngine."""
    VALIDATING = "VALIDATING"
    ORDER_PLACED = "ORDER_PLACED"
    FILLED = "FILLED"
    SL_PLACED = "SL_PLACED"
    TP_PLACED = "TP_PLACED"
    EXITED = "EXITED"
    FAILED = "FAILED"
    REJECTED = "REJECTED"


class ExecutionError(RuntimeError):
    """Raised when deterministic execution flow fails."""


class ExecutionEngine:
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
        """Store dependencies and initialise bookkeeping."""

        LOGGER.debug(
            "Entered ExecutionEngine.__init__",
            extra={"event": "execution_engine_init"},
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
        self._reconcile_task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._stats: dict[str, int] = {
            "submitted": 0,
            "validated": 0,
            "rejected": 0,
            "executed": 0,
            "failed": 0,
            "circuit_breaker_pauses": 0,
            "stale_rejects": 0,
        }
        self._circuit_pause_interval = 5.0
        self._last_circuit_log = 0.0
        self._order_log = Path("data/order_submissions.jsonl")
        self._trades_log = Path("data/active_trades.json")
        
        # ✅ SINGLE AUTHORITY: Track active trades and their lifecycle
        self.active_trades: dict[str, dict[str, Any]] = {}
        self._trade_id_counter = int(time.time())
        self._lock = asyncio.Lock()
        
        # Load persisted state before starting
        self._load_trades()
        self._restore_pending_orders()

    async def submit_signal(self, signal: Signal) -> str | None:
        """Unified entry point for all strategy trades (Single Authority)."""
        symbol = signal.symbol
        action = signal.action
        
        # Extract Strategy ID for granular tracking
        strategy_id = str(
            signal.metadata.get("strategy_id") or 
            signal.metadata.get("strategy_name") or 
            "UNKNOWN"
        )

        async with self._lock:
            # Granular Duplicate check:
            for tid, trade in self.active_trades.items():
                if trade["status"] in {TradeState.VALIDATING, TradeState.ORDER_PLACED, TradeState.FILLED, TradeState.SL_PLACED, TradeState.TP_PLACED}:
                    if trade["symbol"] == symbol and trade["action"] == action:
                        LOGGER.warning(f"🛡️ SINGLE AUTHORITY: Duplicate entry rejected for {symbol} {action}")
                        return None

            # 1. Assign Trade ID
            self._trade_id_counter += 1
            trade_id = f"TRD_{self._trade_id_counter}"
            
            # 2. Basic Validation
            if not getattr(signal, "tradable", True):
                return None
                
            # 3. Create Initial Trade State (VALIDATING)
            now = time.time()
            self.active_trades[trade_id] = {
                "trade_id": trade_id,
                "symbol": symbol,
                "action": action,
                "strategy_id": strategy_id,
                "status": TradeState.VALIDATING,
                "timestamps": {TradeState.VALIDATING: now},
                "signal": signal,
                "metadata": dict(signal.metadata or {}),
            }
            self._save_trades()

        LOGGER.info(f"🚀 ExecutionEngine: Processing Signal {trade_id} | {symbol} {action}")
        
        # 4. Authority over Risk & Execution
        # Pass trade_id through metadata to enable state transitions down the pipeline
        signal.metadata["execution_trade_id"] = trade_id
        order_id = await self.execute(signal)
        
        async with self._lock:
            if not order_id and self.active_trades[trade_id]["status"] == TradeState.VALIDATING:
                self._transition_trade_state_inner(trade_id, TradeState.REJECTED)
                
        return order_id

    async def _transition_trade_state(self, trade_id: str, new_state: TradeState) -> None:
        """Atomically update trade state and record timestamp."""
        async with self._lock:
            self._transition_trade_state_inner(trade_id, new_state)

    def _transition_trade_state_inner(self, trade_id: str, new_state: TradeState) -> None:
        """Internal synchronous helper for state transition. MUST BE CALLED UNDER LOCK."""
        if trade_id not in self.active_trades:
            return
        trade = self.active_trades[trade_id]
        trade["status"] = new_state
        trade["timestamps"][new_state] = time.time()
        LOGGER.info(f"🔁 Trade {trade_id} transitioned to {new_state}")
        self._save_trades()

    async def reconcile_trades(self) -> None:
        """Synchronize local active_trades with broker positions and orders."""
        LOGGER.debug("🚀 ExecutionEngine: Starting Trade Reconciliation...")
        
        try:
            # 1. Fetch Broker Ground Truth
            # Use to_thread for blocking broker calls
            broker_positions = await asyncio.to_thread(self._execution_router._live_executor.broker.positions)
            broker_orders = await asyncio.to_thread(self._execution_router._live_executor.broker.orders)
            
            # Extract net positions (qty > 0)
            live_positions = {}
            if isinstance(broker_positions, dict):
                for p in broker_positions.get("net", []):
                    if abs(p.get("quantity", 0)) > 0:
                        live_positions[p["tradingsymbol"]] = p
            
            # Extract active exit orders (SL/TP)
            active_orders = {}
            for o in broker_orders:
                if o.get("status") in {"OPEN", "TRIGGER PENDING"}:
                    active_orders[o["tradingsymbol"]] = active_orders.get(o["tradingsymbol"], [])
                    active_orders[o["tradingsymbol"]].append(o)

            async with self._lock:
                # Case A: Local trade exists but broker is flat -> Mark EXITED
                for tid, trade in list(self.active_trades.items()):
                    if trade["status"] in {TradeState.FILLED, TradeState.SL_PLACED, TradeState.TP_PLACED}:
                        symbol = trade["symbol"]
                        if symbol not in live_positions:
                            LOGGER.warning(f"⚠️ Reconcile: Trade {tid} ({symbol}) missing at broker. Marking EXITED.")
                            self._transition_trade_state_inner(tid, TradeState.EXITED)

                # Case B: Broker has position but local missing -> Recreate Trade
                for symbol, pos in live_positions.items():
                    local_active = any(t["symbol"] == symbol and t["status"] in {TradeState.FILLED, TradeState.SL_PLACED, TradeState.TP_PLACED} for t in self.active_trades.values())
                    if not local_active:
                        LOGGER.info(f"✨ Reconcile: Found orphan position {symbol}. Recreating local trade state.")
                        self._trade_id_counter += 1
                        tid = f"TRD_{self._trade_id_counter}"
                        self.active_trades[tid] = {
                            "trade_id": tid,
                            "symbol": symbol,
                            "action": "BUY" if pos.get("quantity", 0) > 0 else "SELL",
                            "status": TradeState.FILLED,
                            "timestamps": {TradeState.FILLED: time.time()},
                            "metadata": {"reconciled": True},
                        }
                        self._save_trades()

                # Case C: SL missing for active trade -> Trigger SL placement
                for tid, trade in self.active_trades.items():
                    if trade["status"] == TradeState.FILLED:
                        symbol = trade["symbol"]
                        symbol_orders = active_orders.get(symbol, [])
                        has_sl = any(o.get("transaction_type") != trade["action"] for o in symbol_orders)
                        if not has_sl:
                            LOGGER.warning(f"🛡️ Reconcile: SL missing for {symbol}. Triggering protective exit placement.")
                            # Note: We trigger trigger_lifecycle_on_entry to let LifecycleManager handle it
                            # We create a dummy result based on position price
                            pos = live_positions.get(symbol, {})
                            request = OrderRequest(symbol=symbol, side=trade["action"], quantity=abs(pos.get("quantity", 0)), intent="ENTRY", metadata=trade.get("metadata", {}))
                            request.metadata["execution_trade_id"] = tid
                            result = ExecutionResult(order_id=trade.get("order_id", "RECON"), status="FILLED", fill_price=pos.get("average_price", 0.0), fill_quantity=abs(pos.get("quantity", 0)))
                            await self.trigger_lifecycle_on_entry(request, result)

            LOGGER.debug("✅ ExecutionEngine: Trade Reconciliation Complete.")
        except Exception as exc:
            LOGGER.error(f"❌ Reconcile: Reconciliation failed: {exc}", exc_info=True)

    async def _reconciliation_loop(self) -> None:
        """Run reconciliation every 5 seconds."""
        while not self._stop_event.is_set():
            try:
                await self.reconcile_trades()
            except Exception as exc:
                LOGGER.error(f"Reconciliation loop error: {exc}")
            await asyncio.sleep(5.0)

    def _save_trades(self) -> None:
        """Persist active trades to disk."""
        try:
            serializable = {}
            for tid, trade in self.active_trades.items():
                t_copy = dict(trade)
                if isinstance(t_copy.get("signal"), Signal):
                    t_copy["signal"] = dataclasses.asdict(t_copy["signal"])
                # Ensure TradeState is string
                t_copy["status"] = str(t_copy["status"])
                serializable[tid] = t_copy
            
            self._trades_log.parent.mkdir(parents=True, exist_ok=True)
            with self._trades_log.open("w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2)
        except Exception as exc:
            LOGGER.error(f"Failed to save active trades: {exc}", exc_info=True)

    def _load_trades(self) -> None:
        """Load active trades from disk on startup."""
        if not self._trades_log.exists():
            return
        try:
            with self._trades_log.open("r", encoding="utf-8") as f:
                data = json.load(f)
            for tid, trade in data.items():
                if "status" in trade:
                    trade["status"] = TradeState(trade["status"])
                
                # Reconstruct Signal object
                if "signal" in trade and isinstance(trade["signal"], dict):
                    s_data = trade["signal"]
                    trade["signal"] = Signal(
                        action=s_data.get("action", "HOLD"),
                        symbol=s_data.get("symbol", ""),
                        quantity=s_data.get("quantity", 0),
                        confidence=s_data.get("confidence", 0.0),
                        reason=s_data.get("reason", ""),
                        stop_loss=s_data.get("stop_loss"),
                        take_profit=s_data.get("take_profit"),
                        metadata=s_data.get("metadata", {}),
                        tradable=s_data.get("tradable", True),
                        source=s_data.get("source", "runner")
                    )
                self.active_trades[tid] = trade
            
            # Synchronize ID counter to avoid collisions
            if self.active_trades:
                max_id = 0
                for tid in self.active_trades:
                    if tid.startswith("TRD_"):
                        try:
                            val = int(tid.split("_")[1])
                            max_id = max(max_id, val)
                        except (IndexError, ValueError):
                            continue
                self._trade_id_counter = max(self._trade_id_counter, max_id)
            
            LOGGER.info(f"📂 Restored {len(self.active_trades)} trades from persistence.")
        except Exception as exc:
            LOGGER.error(f"Failed to load active trades: {exc}", exc_info=True)

    async def execute(self, signal: Signal) -> str | None:
        """Execute a validated strategy signal via the single deterministic path."""
        if not hasattr(signal, "source") or getattr(signal, "source", "") != "runner":
            LOGGER.warning("Blocked non-runner execution call")
            return None
        if not getattr(signal, "tradable", True):
            LOGGER.info(
                {
                    "event": "SIGNAL_REJECTED",
                    "symbol": signal.symbol,
                    "reason": "signal_not_tradable",
                }
            )
            return None
        LOGGER.info(
            {
                "event": "SIGNAL_GENERATED",
                "symbol": signal.symbol,
                "action": signal.action,
                "quantity": signal.quantity,
                "reason": signal.reason,
            }
        )
        try:
            if signal.action not in {"BUY", "SELL"}:
                LOGGER.info(
                    {
                        "event": "SIGNAL_REJECTED",
                        "symbol": signal.symbol,
                        "reason": "non_entry_action",
                        "action": signal.action,
                    }
                )
                return None
            risk_manager = self._risk_manager or getattr(
                self._preflight_validator, "_risk_manager", None
            )
            if risk_manager is not None and hasattr(risk_manager, "allow_trade"):
                risk_allowed = risk_manager.allow_trade(signal)
                if risk_allowed is False or (
                    isinstance(risk_allowed, tuple) and not risk_allowed[0]
                ):
                    reason = (
                        risk_allowed[1]
                        if isinstance(risk_allowed, tuple) and len(risk_allowed) > 1
                        else "risk_blocked"
                    )
                    LOGGER.info(
                        {
                            "event": "SIGNAL_REJECTED",
                            "symbol": signal.symbol,
                            "reason": reason,
                        }
                    )
                    return None
            trade_manager = getattr(self, "_trade_manager", None)
            if isinstance(trade_manager, TradeManager) and trade_manager.has_open_trade(
                signal.symbol
            ):
                LOGGER.info(
                    {
                        "event": "SIGNAL_REJECTED",
                        "symbol": signal.symbol,
                        "reason": "open_trade_exists",
                    }
                )
                return None

            request = OrderRequest(
                symbol=signal.symbol,
                side=cast(Literal["BUY", "SELL"], signal.action),
                quantity=int(signal.quantity),
                intent="ENTRY",
                price=signal.metadata.get("signal_price"),
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                metadata=dict(signal.metadata or {}),
            )
            validation = await self._run_preflight(request)
            if validation is None:
                LOGGER.info(
                    {
                        "event": "SIGNAL_REJECTED",
                        "symbol": signal.symbol,
                        "reason": "preflight_rejected",
                    }
                )
                return None
            result = await asyncio.to_thread(self._execution_router.execute, request)
            if not result.order_id:
                raise ExecutionError("Order placement failed")
            await self._handle_execution_result(request, result)
            LOGGER.info(
                {
                    "event": "SIGNAL_EXECUTED",
                    "symbol": signal.symbol,
                    "order_id": result.order_id,
                }
            )
            return result.order_id
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception(
                "[CRITICAL FAILURE]",
                extra={"event": "execution_engine_execute_error"},
                exc_info=True,
            )
            raise

    async def start(self) -> None:
        """Start lifecycle dependencies and queue worker."""

        LOGGER.debug(
            "Entered ExecutionEngine.start",
            extra={"event": "execution_engine_start"},
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
                "Failure in ExecutionEngine.start: %s",
                exc,
                extra={"event": "execution_engine_loop_missing"},
                exc_info=exc,
            )
            return
        self._worker_task = loop.create_task(self._worker_loop())
        # ✅ Start reconciliation loop
        self._reconcile_task = loop.create_task(self._reconciliation_loop())

    async def shutdown(self) -> None:
        """Shutdown worker and dependent subsystems."""

        LOGGER.debug(
            "Entered ExecutionEngine.shutdown",
            extra={"event": "execution_engine_shutdown"},
        )
        self._stop_event.set()
        
        # Stop reconciliation
        if self._reconcile_task:
            self._reconcile_task.cancel()
            with asyncio.suppress(asyncio.CancelledError):
                await self._reconcile_task
            self._reconcile_task = None

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
        """Submit ``request`` to the shared queue for processing."""

        LOGGER.debug(
            "Entered ExecutionEngine.submit_order_request",
            extra={
                "event": "execution_engine_submit",
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
                "Failure in ExecutionEngine.submit_order_request: %s",
                exc,
                extra={"event": "execution_engine_submit_error"},
                exc_info=exc,
            )
            self._stats["failed"] += 1
            raise
        return f"req_{int(request.created_ts * 1000)}"

    def _persist_order(self, request: OrderRequest, status: str) -> None:
        """Persist order submission metadata for recovery."""

        LOGGER.debug(
            "Entered ExecutionEngine._persist_order",
            extra={
                "event": "execution_engine_persist_order",
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
                "Failure in ExecutionEngine._persist_order: %s",
                exc,
                extra={
                    "event": "execution_engine_persist_error",
                    "symbol": request.symbol,
                },
                exc_info=exc,
            )

    def _restore_pending_orders(self) -> None:
        """Reload pending order submissions from the persistence log."""

        LOGGER.debug(
            "Entered ExecutionEngine._restore_pending_orders",
            extra={"event": "execution_engine_restore_enter"},
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
                                "Failure in ExecutionEngine._restore_pending_orders "
                                "parse: %s"
                            ),
                            exc,
                            extra={
                                "event": "execution_engine_restore_parse_error",
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
                                "event": "execution_engine_restore_skipped",
                                "request_id": entry.get("request_id"),
                            },
                        )
                        continue
                    if side not in {"BUY", "SELL"}:
                        LOGGER.warning(
                            "order_restore_skipped_invalid_side",
                            extra={
                                "event": "execution_engine_restore_invalid_side",
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
                                "event": "execution_engine_restore_invalid_intent",
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
                            "event": "execution_engine_restored_order",
                            "symbol": request.symbol,
                            "intent": request.intent,
                            "request_id": entry.get("request_id"),
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    LOGGER.error(
                        (
                            "Failure in ExecutionEngine._restore_pending_orders "
                            "enqueue: %s"
                        ),
                        exc,
                        extra={
                            "event": "execution_engine_restore_enqueue_error",
                            "request_id": entry.get("request_id"),
                        },
                        exc_info=exc,
                    )
            if restored:
                LOGGER.info(
                    "Condition met: restored_pending_orders",
                    extra={
                        "event": "execution_engine_restore_complete",
                        "restored": restored,
                    },
                )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in ExecutionEngine._restore_pending_orders: %s",
                exc,
                extra={"event": "execution_engine_restore_error"},
                exc_info=exc,
            )

    async def get_position_state(self, symbol: str) -> dict[str, Any] | None:
        """Return position state stored for ``symbol``."""

        LOGGER.debug(
            "Entered ExecutionEngine.get_position_state",
            extra={"event": "execution_engine_get_position", "symbol": symbol},
        )
        return self._state_tracker.get_position_state(symbol)

    def emergency_stop(self) -> dict[str, Any]:
        """Pause queue processing and submit market exits for open positions."""

        LOGGER.debug(
            "Entered ExecutionEngine.emergency_stop",
            extra={"event": "execution_engine_emergency_stop"},
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
                "Failure in ExecutionEngine.emergency_stop pause: %s",
                exc,
                extra={"event": "execution_engine_emergency_pause_error"},
                exc_info=exc,
            )
        try:
            for position in self._state_tracker.get_open_positions():
                symbol = str(position.get("symbol") or "").strip()
                if not symbol:
                    continue
                try:
                    # Note: LifecycleManager.exit_at_market handles submitting an EXIT_MARKET request
                    self._lifecycle_manager.exit_at_market(symbol, "EMERGENCY_STOP")
                    positions_closed = snapshot.setdefault("positions_closed", [])
                    if isinstance(positions_closed, list):
                        positions_closed.append(symbol)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.error(
                        "Failure in ExecutionEngine.emergency_stop exit: %s",
                        exc,
                        extra={
                            "event": "execution_engine_emergency_exit_error",
                            "symbol": symbol,
                        },
                        exc_info=exc,
                    )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in ExecutionEngine.emergency_stop iterate: %s",
                exc,
                extra={"event": "execution_engine_emergency_iter_error"},
                exc_info=exc,
            )
        return snapshot

    def get_stats(self) -> dict[str, Any]:
        """Return execution statistics."""

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
        """Process queued requests until shutdown signal."""

        LOGGER.debug(
            "Entered ExecutionEngine._worker_loop",
            extra={"event": "execution_engine_worker_start"},
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
                validation = await self._run_preflight(request)
                if validation is None:
                    continue
                await self._dispatch_request(request)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in ExecutionEngine._worker_loop: %s",
                exc,
                extra={"event": "execution_engine_worker_error"},
                exc_info=exc,
            )

    async def _run_preflight(self, request: OrderRequest) -> ValidationResult | None:
        """Run preflight validation returning the result when allowed."""

        LOGGER.debug(
            "Entered ExecutionEngine._run_preflight",
            extra={
                "event": "execution_engine_preflight",
                "symbol": request.symbol,
            },
        )
        try:
            # Note: PreflightValidator.validate is currently synchronous.
            # If it performs I/O in the future, we should await it.
            outcome = self._preflight_validator.validate(
                request.symbol,
                context={"intent": request.intent, "quantity": request.quantity},
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception(
                "[CRITICAL FAILURE]",
                extra={"event": "execution_engine_preflight_error"},
                exc_info=True,
            )
            VALIDATION_FAILURES.labels(
                symbol=request.symbol, gate="preflight", level="ERROR"
            ).inc()
            async with self._lock:
                self._stats["rejected"] += 1
            self._persist_order(request, status="rejected")
            raise
        if not outcome.allowed:
            LOGGER.info(
                "Condition met: order rejected by preflight",
                extra={
                    "event": "execution_engine_preflight_block",
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
            async with self._lock:
                self._stats["rejected"] += 1
            self._persist_order(request, status="rejected")
            return None
        async with self._lock:
            self._stats["validated"] += 1
        return outcome

    async def _dispatch_request(self, request: OrderRequest) -> None:
        """Route ``request`` through the execution router."""

        stale_threshold_ms = 3000
        if self._data_hub is not None:
            try:
                freshness_check = getattr(self._data_hub, 'is_quote_fresh', None)
                if callable(freshness_check):
                    is_fresh = bool(
                        freshness_check(request.symbol, stale_threshold_ms)
                    )
                    if not is_fresh:
                        LOGGER.info(
                            'Condition met: stale quote rejected',
                            extra={
                                'event': 'execution_engine_stale_quote_reject',
                                'symbol': request.symbol,
                                'threshold_ms': stale_threshold_ms,
                            },
                        )
                        async with self._lock:
                            self._stats['stale_rejects'] += 1
                            self._stats['rejected'] += 1
                        self._persist_order(request, status='rejected')
                        return
                else:
                    LOGGER.debug(
                        'execution_engine_stale_check_unavailable',
                        extra={
                            'event': 'execution_engine_stale_check_unavailable',
                            'symbol': request.symbol,
                        },
                    )
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception(
                    '[CRITICAL FAILURE]',
                    extra={
                        'event': 'execution_engine_stale_check_error',
                        'symbol': request.symbol,
                    },
                    exc_info=True,
                )
                raise

        LOGGER.debug(
            "Entered ExecutionEngine._dispatch_request",
            extra={
                "event": "execution_engine_dispatch",
                "symbol": request.symbol,
            },
        )
        try:
            self._enrich_request_metadata(request)
            # Use asyncio.to_thread for blocking execution to keep the main event loop responsive
            result = await asyncio.to_thread(self._execution_router.execute, request)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception(
                "[CRITICAL FAILURE]",
                extra={"event": "execution_engine_dispatch_error"},
                exc_info=True,
            )
            async with self._lock:
                self._stats["failed"] += 1
            self._persist_order(request, status="failed")
            raise
        await self._handle_execution_result(request, result)

    def _enrich_request_metadata(self, request: OrderRequest) -> None:
        """Populate resolver metadata on ``request`` when available."""

        LOGGER.debug(
            "Entered ExecutionEngine._enrich_request_metadata",
            extra={
                "event": "execution_engine_enrich_metadata",
                "symbol": request.symbol,
            },
        )
        try:
            metadata = dict(request.metadata or {})
            resolver = None
            if self._data_hub is not None:
                resolver = getattr(self._data_hub, "_resolver", None) or getattr(
                    self._data_hub, "resolver", None
                )
            if resolver is None:
                live_executor = getattr(self._execution_router, "_live_executor", None)
                if live_executor is not None:
                    order_manager = getattr(live_executor, "order_manager", None)
                    resolver = getattr(order_manager, "_resolver", None)
                    if resolver is None:
                        market_data = getattr(order_manager, "_market_data", None)
                        resolver = getattr(market_data, "resolver", None) or getattr(
                            market_data, "_resolver", None
                        )
            exchange = metadata.get("exchange")
            tradingsymbol = metadata.get("tradingsymbol")
            instrument_token = metadata.get("instrument_token")
            if resolver is not None:
                if not tradingsymbol and hasattr(resolver, "tradingsymbol_for_order"):
                    try:
                        tradingsymbol = resolver.tradingsymbol_for_order(  # type: ignore[attr-defined]
                            request.symbol
                        )
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.debug(
                            "execution_engine_enrich_tradingsymbol_failed",
                            extra={
                                "event": (
                                    "execution_engine_enrich_tradingsymbol_failed"
                                ),
                                "symbol": request.symbol,
                            },
                            exc_info=exc,
                        )
                if not exchange and hasattr(resolver, "exchange_for_symbol"):
                    try:
                        exchange = resolver.exchange_for_symbol(request.symbol)  # type: ignore[attr-defined]
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.debug(
                            "execution_engine_enrich_exchange_failed",
                            extra={
                                "event": "execution_engine_enrich_exchange_failed",
                                "symbol": request.symbol,
                            },
                            exc_info=exc,
                        )
                if instrument_token is None and hasattr(
                    resolver, "resolve_symbol_to_token"
                ):
                    try:
                        resolved_token = resolver.resolve_symbol_to_token(  # type: ignore[attr-defined]
                            request.symbol
                        )
                        if resolved_token is not None:
                            instrument_token = int(resolved_token)
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.debug(
                            "execution_engine_enrich_token_failed",
                            extra={
                                "event": "execution_engine_enrich_token_failed",
                                "symbol": request.symbol,
                            },
                            exc_info=exc,
                        )
            updates: dict[str, Any] = {}
            if exchange:
                updates["exchange"] = exchange
            if tradingsymbol:
                updates["tradingsymbol"] = tradingsymbol
            if instrument_token is not None:
                updates["instrument_token"] = instrument_token
            if updates:
                metadata.update(updates)
                request.metadata = metadata
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in ExecutionEngine._enrich_request_metadata: %s",
                exc,
                extra={
                    "event": "execution_engine_enrich_metadata_error",
                    "symbol": request.symbol,
                },
                exc_info=exc,
            )

    async def _handle_execution_result(
        self, request: OrderRequest, result: ExecutionResult
    ) -> None:
        """Record execution results and trigger lifecycle transitions."""

        LOGGER.debug(
            "Entered ExecutionEngine._handle_execution_result",
            extra={
                "event": "execution_engine_handle_result",
                "symbol": request.symbol,
                "status": result.status,
            },
        )
        
        # ✅ Extract trade_id for state tracking
        trade_id = request.metadata.get("execution_trade_id")
        
        final_status = (result.status or "unknown").lower()
        if result.status in {"FILLED", "SUBMITTED"} and result.order_id:
            async with self._lock:
                self._stats["executed"] += 1
            await self._record_order(request, result)
            
            # Transition State
            if trade_id:
                new_state = TradeState.FILLED if result.status == "FILLED" else TradeState.ORDER_PLACED
                await self._transition_trade_state(trade_id, new_state)
                async with self._lock:
                    if trade_id in self.active_trades:
                        self.active_trades[trade_id]["order_id"] = result.order_id
                        self._save_trades()

            # Trigger Lifecycle
            if request.intent == "ENTRY":
                await self.trigger_lifecycle_on_entry(request, result)
            
            LOGGER.info(
                "Condition met: order execution recorded",
                extra={
                    "event": "execution_engine_executed",
                    "symbol": request.symbol,
                    "status": result.status,
                },
            )
            self._persist_order(request, status=final_status or "completed")
            return
            
        # Failed/Rejected
        async with self._lock:
            self._stats["failed"] += 1
        if trade_id:
            await self._transition_trade_state(trade_id, TradeState.FAILED)
            
        LOGGER.warning(
            "Order execution failed",
            extra={
                "event": "execution_engine_failed",
                "symbol": request.symbol,
                "status": result.status,
                "reason": result.rejection_reason,
            },
        )
        self._persist_order(request, status=final_status or "failed")

    async def _execute_entry_exits_safely(
        self, request: OrderRequest, result: ExecutionResult
    ) -> tuple[str | None, str | None]:
        """Register entry fills with the lifecycle manager for exit automation."""

        entry_id = result.order_id or f"entry_{int(time.time() * 1000)}"
        symbol = request.symbol
        
        # ✅ Extract trade_id
        trade_id = request.metadata.get("execution_trade_id")
        
        filled_qty = result.fill_quantity
        if filled_qty <= 0:
            filled_qty = request.quantity
        fill_price = self._safe_float(
            result.fill_price, self._safe_float(request.price)
        )
        metadata = request.metadata or {}
        raw_atr = self._get_atr_for_symbol(symbol, metadata)
        signal_price = self._safe_float(metadata.get("signal_price"), fill_price)
        fill_delta = fill_price - signal_price
        bid_value = metadata.get("bid")
        ask_value = metadata.get("ask")
        bid = self._safe_float(bid_value, 0.0) if bid_value is not None else None
        ask = self._safe_float(ask_value, 0.0) if ask_value is not None else None
        atr = self._resolve_spread_aware_atr(
            raw_atr=raw_atr,
            execution_price=fill_price,
            bid=bid,
            ask=ask,
        )
        regime = self._get_current_regime(symbol, metadata)
        iv_value = metadata.get("iv") or metadata.get("implied_volatility")

        LOGGER.info(
            "Condition met: entry fill registered for lifecycle",
            extra={
                "event": "execution_engine_entry_fill",
                "symbol": symbol,
                "order_id": entry_id,
                "fill_price": fill_price,
                "fill_quantity": filled_qty,
                "atr": atr,
                "regime": regime,
                "signal_price": signal_price,
                "fill_delta": fill_delta,
            },
        )
        try:
            # LifecycleManager.on_fill is synchronous
            self._lifecycle_manager.on_fill(
                symbol=symbol,
                entry_price=fill_price,
                quantity=filled_qty,
                atr=atr,
                regime=regime,
                iv=iv_value,
            )
            # Record SL/TP Placement
            if trade_id:
                await self._transition_trade_state(trade_id, TradeState.SL_PLACED)
                await self._transition_trade_state(trade_id, TradeState.TP_PLACED)
                
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in ExecutionEngine._execute_entry_exits_safely: %s",
                exc,
                extra={
                    "event": "execution_engine_entry_lifecycle_error",
                    "symbol": symbol,
                    "order_id": entry_id,
                },
                exc_info=exc,
            )
            raise
        return None, None

    async def _record_order(self, request: OrderRequest, result: ExecutionResult) -> None:
        """Persist execution details to the state tracker."""

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
                "Failure in ExecutionEngine._record_order: %s",
                exc,
                extra={"event": "execution_engine_record_order_error"},
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
                "Failure in ExecutionEngine._record_order update: %s",
                exc,
                extra={
                    "event": "execution_engine_record_position_error",
                    "symbol": request.symbol,
                },
                exc_info=exc,
            )

    async def trigger_lifecycle_on_entry(
        self, request: OrderRequest, result: ExecutionResult
    ) -> None:
        """Initialise lifecycle manager and place protective exits safely."""

        try:
            await self._execute_entry_exits_safely(request, result)
            LOGGER.info(
                "Condition met: entry protection activated",
                extra={
                    "event": "execution_engine_entry_protection",
                    "symbol": request.symbol,
                    "order_id": result.order_id,
                },
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.critical(
                "Failure in ExecutionEngine.trigger_lifecycle_on_entry: %s",
                exc,
                extra={
                    "event": "execution_engine_entry_protection_failed",
                    "symbol": request.symbol,
                    "order_id": result.order_id,
                },
                exc_info=exc,
            )
            raise

    def _should_halt_processing(self) -> bool:
        """Return ``True`` when queue processing should pause."""

        LOGGER.debug(
            "Entered ExecutionEngine._should_halt_processing",
            extra={"event": "execution_engine_should_halt"},
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
                            "event": "execution_engine_circuit_open",
                            "reason": reason or "circuit_breaker",
                        },
                    )
                    self._last_circuit_log = now
                asyncio.run_coroutine_threadsafe(self._inc_circuit_pauses(), asyncio.get_event_loop())
                return True
            return False
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in ExecutionEngine._should_halt_processing: %s",
                exc,
                extra={"event": "execution_engine_circuit_check_error"},
                exc_info=exc,
            )
            return False

    async def _inc_circuit_pauses(self) -> None:
        async with self._lock:
            self._stats["circuit_breaker_pauses"] += 1

    def _get_atr_for_symbol(
        self, symbol: str, metadata: Mapping[str, Any] | None = None
    ) -> float:
        """Return the ATR value for ``symbol`` from metadata or trackers."""

        LOGGER.debug(
            "Entered ExecutionEngine._get_atr_for_symbol",
            extra={"event": "execution_engine_resolve_atr", "symbol": symbol},
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
                "Failure in ExecutionEngine._get_atr_for_symbol: %s",
                exc,
                extra={
                    "event": "execution_engine_resolve_atr_error",
                    "symbol": symbol,
                },
                exc_info=exc,
            )
        LOGGER.info(
            "Condition met: default ATR applied",
            extra={
                "event": "execution_engine_default_atr",
                "symbol": symbol,
                "atr": default_atr,
            },
        )
        return default_atr

    def _resolve_spread_aware_atr(
        self,
        *,
        raw_atr: float,
        execution_price: float,
        bid: float | None,
        ask: float | None,
    ) -> float:
        """Return spread-aware ATR floor for execution safety."""
        spread = 0.0
        if bid is not None and ask is not None and ask >= bid:
            spread = float(ask - bid)
        min_atr = max(spread * 1.5, 1.0)  # at least ₹1 regardless of spread
        return max(float(raw_atr), min_atr)

    def _get_current_regime(
        self, symbol: str, metadata: Mapping[str, Any] | None = None
    ) -> str:
        """Return the most recent regime label with safe fallbacks."""

        LOGGER.debug(
            "Entered ExecutionEngine._get_current_regime",
            extra={"event": "execution_engine_resolve_regime"},
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
                "Failure in ExecutionEngine._get_current_regime: %s",
                exc,
                extra={"event": "execution_engine_resolve_regime_error"},
                exc_info=exc,
            )
        LOGGER.info(
            "Condition met: default regime applied",
            extra={
                "event": "execution_engine_default_regime",
                "symbol": symbol,
                "regime": "NEUTRAL",
            },
        )
        return "NEUTRAL"

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """Return ``value`` as float with ``default`` fallback."""

        try:
            return float(value)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return float(default)


__all__ = ["ExecutionEngine"]
