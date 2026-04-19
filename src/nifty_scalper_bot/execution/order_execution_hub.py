"""Central coordination hub for order execution."""

from __future__ import annotations

import asyncio
import os
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
    EXIT_PENDING = "EXIT_PENDING"
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
        
        # ✅ SINGLE AUTHORITY: Discipline & Risk Tracking
        self._last_trade_time: float = 0.0
        self._trades_today_count: int = 0
        self._last_reset_date: str = time.strftime("%Y-%m-%d")
        
        # Configuration (Defaults)
        self._trade_cooldown_sec = float(os.getenv("TRADE_COOLDOWN_SECONDS", "10.0"))
        self._max_trades_per_day = int(os.getenv("MAX_TRADES_PER_DAY", "50"))
        self._max_drawdown_pct = float(os.getenv("MAX_DRAWDOWN_PCT", "2.0"))

        # ✅ SINGLE AUTHORITY: Track active trades and their lifecycle
        self.active_trades: dict[str, dict[str, Any]] = {}
        self._trade_id_counter = int(time.time())
        self._lock = asyncio.Lock()
        
        # Load persisted state before starting
        self._load_trades()
        self._restore_pending_orders()

    async def can_take_trade(self, symbol: str, action: str) -> tuple[bool, str]:
        """Verify execution discipline: Cooldown, Max Trades, and Drawdown."""
        now = time.time()
        today = time.strftime("%Y-%m-%d")
        
        async with self._lock:
            # 1. Reset daily counter if needed
            if today != self._last_reset_date:
                self._trades_today_count = 0
                self._last_reset_date = today

            # 2. Check Cooldown
            elapsed = now - self._last_trade_time
            if elapsed < self._trade_cooldown_sec:
                return False, f"Cooldown active ({self._trade_cooldown_sec - elapsed:.1f}s remaining)"

            # 3. Check Max Trades per Day
            if self._trades_today_count >= self._max_trades_per_day:
                return False, f"Max daily trades reached ({self._max_trades_per_day})"

            # 4. Check Drawdown (via RiskManager)
            if self._risk_manager:
                # Assuming RiskManager has a method or property for current drawdown
                # If not, we use a placeholder check or skip
                if hasattr(self._risk_manager, "is_drawdown_limit_hit") and self._risk_manager.is_drawdown_limit_hit():
                    return False, "Max drawdown limit hit"

            # 5. Prevent Duplicate Active Directional Trades
            for tid, trade in self.active_trades.items():
                if trade["status"] in {TradeState.VALIDATING, TradeState.ORDER_PLACED, TradeState.FILLED, TradeState.SL_PLACED, TradeState.TP_PLACED}:
                    if trade["symbol"] == symbol and trade["action"] == action:
                        return False, f"Duplicate active {action} trade for {symbol}"

        return True, "OK"

    async def submit_signal(self, signal: Signal) -> str | None:
        """Unified entry point for all strategy signals (Single Authority)."""
        symbol = signal.symbol
        action = signal.action
        
        # 🛡️ ENFORCE DISCIPLINE (Entry only for now, or all?)
        # Requirements said "Strategy cannot bypass execution discipline"
        if action in {"BUY", "SELL"}:
            allowed, reason = await self.can_take_trade(symbol, action)
            if not allowed:
                LOGGER.warning(f"🛡️ EXECUTION DISCIPLINE: Entry Signal for {symbol} rejected: {reason}")
                return None

        # Extract Strategy ID for granular tracking
        strategy_id = str(
            signal.metadata.get("strategy_id") or 
            signal.metadata.get("strategy_name") or 
            "UNKNOWN"
        )

        async with self._lock:
            # 1. Assign Trade ID
            self._trade_id_counter += 1
            trade_id = f"TRD_{self._trade_id_counter}"
            
            # 2. Basic Validation
            if not getattr(signal, "tradable", True):
                return None
                
            # 3. Create Initial Signal State
            now = time.time()
            # If it's an exit, we might want to link it to an active trade, 
            # but for now we track it as a new "Event" in the engine.
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
            
            # Record submission for discipline (cooldown/counts)
            if action in {"BUY", "SELL"}:
                self._last_trade_time = now
                self._trades_today_count += 1
            
            self._save_trades()

        LOGGER.info(f"🚀 ExecutionEngine: Processing Signal {trade_id} | {symbol} {action}")
        
        # 4. Atomic Execution Flow
        return await self._execute_trade(trade_id)

    async def _execute_trade(self, trade_id: str) -> str | None:
        """AUTHORITATIVE EXECUTION FLOW: Handles validation, routing, and lifecycle."""
        async with self._lock:
            trade = self.active_trades.get(trade_id)
            if not trade:
                return None
            signal: Signal = trade["signal"]
            symbol = trade["symbol"]
            action = trade["action"]

        try:
            # 1. Source Check
            if not hasattr(signal, "source") or getattr(signal, "source", "") != "runner":
                LOGGER.warning(f"Blocked non-runner execution call for {trade_id}")
                await self._transition_trade_state(trade_id, TradeState.REJECTED)
                return None

            # 2. Route by Action
            if action in {"BUY", "SELL"}:
                return await self._execute_entry(trade_id, signal)
            elif action in {"CLOSE_LONG", "CLOSE_SHORT"}:
                return await self._execute_exit(trade_id, signal)
            else:
                LOGGER.info(f"Signal {trade_id} rejected: unknown action {action}")
                await self._transition_trade_state(trade_id, TradeState.REJECTED)
                return None

        except Exception as exc:
            LOGGER.exception(f"❌ ExecutionEngine: Critical failure executing trade {trade_id}", exc_info=True)
            await self._transition_trade_state(trade_id, TradeState.FAILED)
            return None

    async def _execute_entry(self, trade_id: str, signal: Signal) -> str | None:
        """Internal helper for entry execution."""
        symbol = signal.symbol
        start_time = time.time()
        
        # Risk Check
        risk_manager = self._risk_manager or getattr(self._preflight_validator, "_risk_manager", None)
        if risk_manager is not None and hasattr(risk_manager, "allow_trade"):
            risk_allowed = risk_manager.allow_trade(signal)
            if risk_allowed is False or (isinstance(risk_allowed, tuple) and not risk_allowed[0]):
                reason = risk_allowed[1] if isinstance(risk_allowed, tuple) and len(risk_allowed) > 1 else "risk_blocked"
                await self._handle_rejection(trade_id, reason)
                return None

        # Prepare Order Request
        request = OrderRequest(
            symbol=symbol,
            side=cast(Literal["BUY", "SELL"], signal.action),
            quantity=int(signal.quantity),
            intent="ENTRY",
            price=signal.metadata.get("signal_price"),
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            metadata=dict(signal.metadata or {}),
        )
        request.metadata["execution_trade_id"] = trade_id
        request.metadata["execution_start_ts"] = start_time

        # Preflight Validation
        validation = await self._run_preflight(request)
        if validation is None:
            # Rejection already handled in _run_preflight via _persist_order/logs
            # but we need to update our internal trade state
            await self._handle_rejection(trade_id, "preflight_rejected")
            return None

        # Dispatch
        result = await asyncio.to_thread(self._execution_router.execute, request)
        if not result.order_id:
            await self._transition_trade_state(trade_id, TradeState.FAILED)
            return None

        await self._handle_execution_result(request, result)
        return result.order_id

    async def _handle_rejection(self, trade_id: str, reason: str) -> None:
        """Helper to handle rejected signals with structured logging."""
        async with self._lock:
            if trade_id in self.active_trades:
                trade = self.active_trades[trade_id]
                trade["rejection_reason"] = reason
                await self._transition_trade_state_inner_async(trade_id, TradeState.REJECTED)

    async def _transition_trade_state_inner_async(self, trade_id: str, new_state: TradeState) -> None:
        """Async-safe version of transition helper."""
        self._transition_trade_state_inner(trade_id, new_state)

    async def _execute_exit(self, trade_id: str, signal: Signal) -> str | None:
        """Internal helper for exit execution."""
        symbol = signal.symbol
        start_time = time.time()
        
        from nifty_scalper_bot.execution.order_manager import ExitIntent
        
        # We need to find the active position for this symbol
        pos_state = await self.get_position_state(symbol)
        if not pos_state or int(pos_state.get("quantity", 0)) == 0:
            await self._handle_rejection(trade_id, "no_active_position")
            return None

        exit_side = "SELL" if pos_state.get("side") == "LONG" else "BUY"
        entry_price = float(pos_state.get("entry_price") or 0.0)
        
        exit_intent = ExitIntent(
            symbol=symbol,
            side=exit_side,
            quantity=abs(int(pos_state["quantity"])),
            price=signal.metadata.get("signal_price"),
            reason=signal.reason,
        )

        await self._transition_trade_state(trade_id, TradeState.EXIT_PENDING)
        
        try:
            exit_order_id = await asyncio.to_thread(self._order_manager.place_reduce_only_exit, exit_intent)
            if exit_order_id:
                now = time.time()
                latency = (now - start_time) * 1000.0
                
                # Simple Realized PnL Calculation
                exit_price = float(signal.metadata.get("signal_price") or 0.0)
                pnl = 0.0
                if entry_price > 0:
                    multiplier = 1 if exit_side == "SELL" else -1
                    pnl = (exit_price - entry_price) * abs(int(pos_state["quantity"])) * multiplier

                async with self._lock:
                    trade = self.active_trades[trade_id]
                    trade["exit_order_id"] = exit_order_id
                    trade["realized_pnl"] = pnl
                    trade["execution_latency_ms"] = latency
                
                await self._transition_trade_state(trade_id, TradeState.EXITED)
                
                LOGGER.info({
                    "event": "TRADE_EXIT_COMPLETED",
                    "trade_id": trade_id,
                    "symbol": symbol,
                    "pnl": pnl,
                    "latency_ms": latency
                })
                return exit_order_id
            else:
                await self._transition_trade_state(trade_id, TradeState.FAILED)
                return None
        except Exception as exc:
            LOGGER.error(f"Exit failed for {trade_id}: {exc}")
            await self._transition_trade_state(trade_id, TradeState.FAILED)
            async with self._lock:
                self._last_trade_time = time.time() # Force cooldown
            return None

    async def _transition_trade_state(self, trade_id: str, new_state: TradeState) -> None:
        """Atomically update trade state and record timestamp."""
        async with self._lock:
            self._transition_trade_state_inner(trade_id, new_state)

    def _transition_trade_state_inner(self, trade_id: str, new_state: TradeState) -> None:
        """Internal synchronous helper for state transition. MUST BE CALLED UNDER LOCK."""
        if trade_id not in self.active_trades:
            return
        trade = self.active_trades[trade_id]
        now = time.time()
        old_state = trade.get("status")
        
        trade["status"] = new_state
        trade["timestamps"][new_state] = now
        
        LOGGER.info(
            {
                "event": "TRADE_STATE_TRANSITION",
                "trade_id": trade_id,
                "symbol": trade.get("symbol"),
                "action": trade.get("action"),
                "old_state": old_state,
                "new_state": new_state,
                "timestamp": now,
            }
        )
        self._save_trades()

    async def reconcile_trades(self) -> None:
        """Synchronize local active_trades with broker positions and orders."""
        LOGGER.debug("🚀 ExecutionEngine: Starting Trade Reconciliation...")
        
        try:
            # 1. Fetch Broker Ground Truth
            broker_positions = await asyncio.to_thread(self._execution_router._live_executor.broker.positions)
            broker_orders = await asyncio.to_thread(self._execution_router._live_executor.broker.orders)
            
            live_positions = {}
            if isinstance(broker_positions, dict):
                for p in broker_positions.get("net", []):
                    if abs(p.get("quantity", 0)) > 0:
                        live_positions[p["tradingsymbol"]] = p
            
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

    async def start(self) -> None:
        """Start lifecycle dependencies and queue worker."""
        LOGGER.debug("Entered ExecutionEngine.start", extra={"event": "execution_engine_start"})
        if self._worker_task is not None and not self._worker_task.done():
            return
        await self._lifecycle_manager.start()
        await self._post_fill_monitor.start()
        self._stop_event.clear()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            LOGGER.error("Failure in ExecutionEngine.start: %s", exc, extra={"event": "execution_engine_loop_missing"}, exc_info=exc)
            return
        self._worker_task = loop.create_task(self._worker_loop())
        self._reconcile_task = loop.create_task(self._reconciliation_loop())

    async def shutdown(self) -> None:
        """Shutdown worker and dependent subsystems."""
        LOGGER.debug("Entered ExecutionEngine.shutdown", extra={"event": "execution_engine_shutdown"})
        self._stop_event.set()
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
        LOGGER.debug("Entered ExecutionEngine.submit_order_request", extra={"event": "execution_engine_submit", "symbol": request.symbol, "intent": request.intent})
        self._stats["submitted"] += 1
        try:
            self._order_queue.submit_order_request(request)
            self._persist_order(request, status="pending")
        except Exception as exc:
            LOGGER.error("Failure in ExecutionEngine.submit_order_request: %s", exc, extra={"event": "execution_engine_submit_error"}, exc_info=exc)
            self._stats["failed"] += 1
            raise
        return f"req_{int(request.created_ts * 1000)}"

    def _persist_order(self, request: OrderRequest, status: str) -> None:
        """Persist order submission metadata for recovery."""
        LOGGER.debug("Entered ExecutionEngine._persist_order", extra={"event": "execution_engine_persist_order", "symbol": request.symbol, "status": status})
        request_id = f"req_{int(request.created_ts * 1000)}"
        try:
            self._order_log.parent.mkdir(parents=True, exist_ok=True)
            entry = {"timestamp": time.time(), "created_ts": request.created_ts, "request_id": request_id, "symbol": request.symbol, "side": request.side, "quantity": request.quantity, "intent": request.intent, "status": status}
            with self._order_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
        except Exception as exc:
            LOGGER.error("Failure in ExecutionEngine._persist_order: %s", exc, extra={"event": "execution_engine_persist_error", "symbol": request.symbol}, exc_info=exc)

    def _restore_pending_orders(self) -> None:
        """Reload pending order submissions from the persistence log."""
        LOGGER.debug("Entered ExecutionEngine._restore_pending_orders", extra={"event": "execution_engine_restore_enter"})
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
                        LOGGER.error("Failure in ExecutionEngine._restore_pending_orders parse: %s", exc, extra={"event": "execution_engine_restore_parse_error", "line": line_number}, exc_info=exc)
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
                    symbol, side, quantity, intent = str(entry.get("symbol") or "").strip(), str(entry.get("side") or "").strip(), int(entry.get("quantity") or 0), str(entry.get("intent") or "ENTRY").strip() or "ENTRY"
                    if not symbol or not side or quantity <= 0 or side not in {"BUY", "SELL"}:
                        continue
                    valid_intents: set[OrderIntent] = {"ENTRY", "EXIT_SL", "EXIT_TP1", "EXIT_TP2", "ADJUST_TRAIL"}
                    if intent not in valid_intents:
                        continue
                    request = OrderRequest(symbol=symbol, side=cast(Literal["BUY", "SELL"], side), quantity=quantity, intent=cast(OrderIntent, intent), source="restore")
                    self._order_queue.submit_order_request(request)
                    restored += 1
                except Exception as exc:
                    LOGGER.error("Failure in ExecutionEngine._restore_pending_orders enqueue: %s", exc, extra={"event": "execution_engine_restore_enqueue_error", "request_id": entry.get("request_id")}, exc_info=exc)
            if restored:
                LOGGER.info("Condition met: restored_pending_orders", extra={"event": "execution_engine_restore_complete", "restored": restored})
        except Exception as exc:
            LOGGER.error("Failure in ExecutionEngine._restore_pending_orders: %s", exc, extra={"event": "execution_engine_restore_error"}, exc_info=exc)

    async def get_position_state(self, symbol: str) -> dict[str, Any] | None:
        """Return position state stored for ``symbol``."""
        LOGGER.debug("Entered ExecutionEngine.get_position_state", extra={"event": "execution_engine_get_position", "symbol": symbol})
        return self._state_tracker.get_position_state(symbol)

    def emergency_stop(self) -> dict[str, Any]:
        """Pause queue processing and submit market exits for open positions."""
        LOGGER.debug("Entered ExecutionEngine.emergency_stop", extra={"event": "execution_engine_emergency_stop"})
        snapshot = {"timestamp": time.monotonic(), "positions_closed": [], "queue_paused": False}
        try:
            self._order_queue.pause()
            snapshot["queue_paused"] = True
        except Exception as exc:
            LOGGER.error("Failure in ExecutionEngine.emergency_stop pause: %s", exc, extra={"event": "execution_engine_emergency_pause_error"}, exc_info=exc)
        try:
            for position in self._state_tracker.get_open_positions():
                symbol = str(position.get("symbol") or "").strip()
                if not symbol: continue
                try:
                    self._lifecycle_manager.exit_at_market(symbol, "EMERGENCY_STOP")
                    positions_closed = snapshot.setdefault("positions_closed", [])
                    if isinstance(positions_closed, list): positions_closed.append(symbol)
                except Exception as exc:
                    LOGGER.error("Failure in ExecutionEngine.emergency_stop exit: %s", exc, extra={"event": "execution_engine_emergency_exit_error", "symbol": symbol}, exc_info=exc)
        except Exception as exc:
            LOGGER.error("Failure in ExecutionEngine.emergency_stop iterate: %s", exc, extra={"event": "execution_engine_emergency_iter_error"}, exc_info=exc)
        return snapshot

    def get_stats(self) -> dict[str, Any]:
        """Return execution statistics."""
        queue_depth = len(self._order_queue.get_queue_snapshot())
        router_stats = self._execution_router.get_stats()
        monitor_stats = self._post_fill_monitor.get_stats()
        return {**self._stats, "queue_depth": queue_depth, "router": router_stats, "reconciliation": monitor_stats}

    async def _worker_loop(self) -> None:
        """Process queued requests until shutdown signal."""
        LOGGER.debug("Entered ExecutionEngine._worker_loop", extra={"event": "execution_engine_worker_start"})
        try:
            while not self._stop_event.is_set():
                if self._should_halt_processing():
                    await asyncio.sleep(self._circuit_pause_interval)
                    continue
                request = await asyncio.to_thread(self._order_queue.get_next_request, 0.5)
                if request is None: continue
                validation = await self._run_preflight(request)
                if validation is None: continue
                await self._dispatch_request(request)
        except asyncio.CancelledError: raise
        except Exception as exc:
            LOGGER.error("Failure in ExecutionEngine._worker_loop: %s", exc, extra={"event": "execution_engine_worker_error"}, exc_info=exc)

    async def _run_preflight(self, request: OrderRequest) -> ValidationResult | None:
        """Run preflight validation returning the result when allowed."""
        LOGGER.debug("Entered ExecutionEngine._run_preflight", extra={"event": "execution_engine_preflight", "symbol": request.symbol})
        try:
            outcome = self._preflight_validator.validate(request.symbol, context={"intent": request.intent, "quantity": request.quantity})
        except Exception as exc:
            LOGGER.exception("[CRITICAL FAILURE]", extra={"event": "execution_engine_preflight_error"}, exc_info=True)
            VALIDATION_FAILURES.labels(symbol=request.symbol, gate="preflight", level="ERROR").inc()
            async with self._lock: self._stats["rejected"] += 1
            self._persist_order(request, status="rejected")
            raise
        if not outcome.allowed:
            for reason in outcome.reasons:
                gate_name = str(reason.get("gate") or "unknown")
                VALIDATION_FAILURES.labels(symbol=request.symbol, gate=gate_name, level=str(outcome.blocking_level or "UNKNOWN")).inc()
            async with self._lock: self._stats["rejected"] += 1
            self._persist_order(request, status="rejected")
            return None
        async with self._lock: self._stats["validated"] += 1
        return outcome

    async def _dispatch_request(self, request: OrderRequest) -> None:
        """Route ``request`` through the execution router."""
        stale_threshold_ms = 3000
        if self._data_hub is not None:
            try:
                freshness_check = getattr(self._data_hub, 'is_quote_fresh', None)
                if callable(freshness_check):
                    is_fresh = bool(freshness_check(request.symbol, stale_threshold_ms))
                    if not is_fresh:
                        async with self._lock:
                            self._stats['stale_rejects'] += 1
                            self._stats['rejected'] += 1
                        self._persist_order(request, status='rejected')
                        return
            except Exception as exc:
                LOGGER.exception('[CRITICAL FAILURE]', extra={'event': 'execution_engine_stale_check_error', 'symbol': request.symbol}, exc_info=True)
                raise
        LOGGER.debug("Entered ExecutionEngine._dispatch_request", extra={"event": "execution_engine_dispatch", "symbol": request.symbol})
        try:
            self._enrich_request_metadata(request)
            result = await asyncio.to_thread(self._execution_router.execute, request)
        except Exception as exc:
            LOGGER.exception("[CRITICAL FAILURE]", extra={"event": "execution_engine_dispatch_error"}, exc_info=True)
            async with self._lock: self._stats["failed"] += 1
            self._persist_order(request, status="failed")
            raise
        await self._handle_execution_result(request, result)

    def _enrich_request_metadata(self, request: OrderRequest) -> None:
        """Populate resolver metadata on ``request`` when available."""
        try:
            metadata = dict(request.metadata or {})
            resolver = None
            if self._data_hub is not None:
                resolver = getattr(self._data_hub, "_resolver", None) or getattr(self._data_hub, "resolver", None)
            if resolver is None:
                live_executor = getattr(self._execution_router, "_live_executor", None)
                if live_executor is not None:
                    order_manager = getattr(live_executor, "order_manager", None)
                    resolver = getattr(order_manager, "_resolver", None)
                    if resolver is None:
                        market_data = getattr(order_manager, "_market_data", None)
                        resolver = getattr(market_data, "resolver", None) or getattr(market_data, "_resolver", None)
            exchange, tradingsymbol, instrument_token = metadata.get("exchange"), metadata.get("tradingsymbol"), metadata.get("instrument_token")
            if resolver is not None:
                if not tradingsymbol and hasattr(resolver, "tradingsymbol_for_order"):
                    try: tradingsymbol = resolver.tradingsymbol_for_order(request.symbol)
                    except Exception: pass
                if not exchange and hasattr(resolver, "exchange_for_symbol"):
                    try: exchange = resolver.exchange_for_symbol(request.symbol)
                    except Exception: pass
                if instrument_token is None and hasattr(resolver, "resolve_symbol_to_token"):
                    try:
                        resolved_token = resolver.resolve_symbol_to_token(request.symbol)
                        if resolved_token is not None: instrument_token = int(resolved_token)
                    except Exception: pass
            updates: dict[str, Any] = {}
            if exchange: updates["exchange"] = exchange
            if tradingsymbol: updates["tradingsymbol"] = tradingsymbol
            if instrument_token is not None: updates["instrument_token"] = instrument_token
            if updates: metadata.update(updates); request.metadata = metadata
        except Exception as exc:
            LOGGER.error("Failure in ExecutionEngine._enrich_request_metadata: %s", exc, extra={"event": "execution_engine_enrich_metadata_error", "symbol": request.symbol}, exc_info=exc)

    async def _handle_execution_result(self, request: OrderRequest, result: ExecutionResult) -> None:
        """Record execution results and trigger lifecycle transitions."""
        trade_id = request.metadata.get("execution_trade_id")
        final_status = (result.status or "unknown").lower()
        if result.status in {"FILLED", "SUBMITTED"} and result.order_id:
            async with self._lock: self._stats["executed"] += 1
            await self._record_order(request, result)
            if trade_id:
                new_state = TradeState.FILLED if result.status == "FILLED" else TradeState.ORDER_PLACED
                await self._transition_trade_state(trade_id, new_state)
                async with self._lock:
                    if trade_id in self.active_trades:
                        self.active_trades[trade_id]["order_id"] = result.order_id
                        self._save_trades()
            if request.intent == "ENTRY": await self.trigger_lifecycle_on_entry(request, result)
            self._persist_order(request, status=final_status or "completed")
            return
        async with self._lock: self._stats["failed"] += 1
        if trade_id: await self._transition_trade_state(trade_id, TradeState.FAILED)
        self._persist_order(request, status=final_status or "failed")

    async def _execute_entry_exits_safely(self, request: OrderRequest, result: ExecutionResult) -> tuple[str | None, str | None]:
        """Register entry fills with the lifecycle manager for exit automation."""
        entry_id, symbol, trade_id = result.order_id or f"entry_{int(time.time() * 1000)}", request.symbol, request.metadata.get("execution_trade_id")
        filled_qty = result.fill_quantity if result.fill_quantity > 0 else request.quantity
        fill_price = self._safe_float(result.fill_price, self._safe_float(request.price))
        metadata = request.metadata or {}
        raw_atr = self._get_atr_for_symbol(symbol, metadata)
        bid, ask = self._safe_float(metadata.get("bid"), 0.0), self._safe_float(metadata.get("ask"), 0.0)
        atr = self._resolve_spread_aware_atr(raw_atr=raw_atr, execution_price=fill_price, bid=bid, ask=ask)
        regime, iv_value = self._get_current_regime(symbol, metadata), metadata.get("iv") or metadata.get("implied_volatility")
        try:
            self._lifecycle_manager.on_fill(symbol=symbol, entry_price=fill_price, quantity=filled_qty, atr=atr, regime=regime, iv=iv_value)
            if trade_id:
                await self._transition_trade_state(trade_id, TradeState.SL_PLACED)
                await self._transition_trade_state(trade_id, TradeState.TP_PLACED)
        except Exception as exc:
            LOGGER.error("Failure in ExecutionEngine._execute_entry_exits_safely: %s", exc, extra={"event": "execution_engine_entry_lifecycle_error", "symbol": symbol, "order_id": entry_id}, exc_info=exc)
            raise
        return None, None

    async def _record_order(self, request: OrderRequest, result: ExecutionResult) -> None:
        """Persist execution details to the state tracker."""
        fill_price = self._safe_float(result.fill_price, self._safe_float(request.price))
        payload = {"order_id": result.order_id, "symbol": request.symbol, "side": request.side, "quantity": request.quantity, "status": result.status.lower(), "fill_price": fill_price, "intent": request.intent, "timestamp": time.time(), "parent_id": request.parent_id}
        try: self._state_tracker.add_order(payload)
        except Exception as exc: LOGGER.error("Failure in ExecutionEngine._record_order: %s", exc, extra={"event": "execution_engine_record_order_error"}, exc_info=exc)
        quantity_signed = int(request.quantity) if request.side.upper() == "BUY" else -int(request.quantity)
        try:
            existing = self._state_tracker.get_position_state(request.symbol) or {}
            current_qty = int(existing.get("quantity", 0))
            new_quantity = current_qty + quantity_signed
            if request.intent.upper().startswith("EXIT") and new_quantity == 0:
                self._state_tracker.update_position(request.symbol, {"delete": True})
                return
            updates: dict[str, Any] = {"quantity": new_quantity}
            if request.intent.upper() == "ENTRY" and current_qty == 0:
                updates.update({"entry_price": fill_price, "entry_time": time.time(), "lifecycle_stage": "ENTRY"})
            elif request.intent.upper().startswith("EXIT"): updates["lifecycle_stage"] = "EXIT"
            self._state_tracker.update_position(request.symbol, updates)
        except Exception as exc: LOGGER.error("Failure in ExecutionEngine._record_order update: %s", exc, extra={"event": "execution_engine_record_position_error", "symbol": request.symbol}, exc_info=exc)

    async def trigger_lifecycle_on_entry(self, request: OrderRequest, result: ExecutionResult) -> None:
        """Initialise lifecycle manager and place protective exits safely."""
        try: await self._execute_entry_exits_safely(request, result)
        except Exception as exc:
            LOGGER.critical("Failure in ExecutionEngine.trigger_lifecycle_on_entry: %s", exc, extra={"event": "execution_engine_entry_protection_failed", "symbol": request.symbol, "order_id": result.order_id}, exc_info=exc)
            raise

    def _should_halt_processing(self) -> bool:
        """Return ``True`` when queue processing should pause."""
        try:
            risk_manager = self._risk_manager or getattr(self._preflight_validator, "_risk_manager", None)
            if risk_manager is None: return False
            breaker_active = bool(risk_manager.is_circuit_breaker_active()) if hasattr(risk_manager, "is_circuit_breaker_active") else False
            if breaker_active:
                now = time.monotonic()
                if now - self._last_circuit_log >= self._circuit_pause_interval:
                    LOGGER.info("Condition met: circuit breaker active")
                    self._last_circuit_log = now
                asyncio.run_coroutine_threadsafe(self._inc_circuit_pauses(), asyncio.get_event_loop())
                return True
            return False
        except Exception: return False

    async def _inc_circuit_pauses(self) -> None:
        async with self._lock: self._stats["circuit_breaker_pauses"] += 1

    def _get_atr_for_symbol(self, symbol: str, metadata: Mapping[str, Any] | None = None) -> float:
        """Return the ATR value for ``symbol`` from metadata or trackers."""
        default_atr, symbol_upper = 10.0, symbol.upper()
        if "NIFTY" in symbol_upper and "BANK" not in symbol_upper: default_atr = 50.0
        elif "BANKNIFTY" in symbol_upper or "FINNIFTY" in symbol_upper: default_atr = 150.0
        try:
            if metadata:
                for key in ("atr", "avg_true_range", "atr_value"):
                    if key in metadata and metadata[key] is not None: return self._safe_float(metadata[key], default_atr)
            position_state = self._state_tracker.get_position_state(symbol)
            if position_state and position_state.get("atr"): return self._safe_float(position_state["atr"], default_atr)
            if self._data_hub and hasattr(self._data_hub, "get_indicator"):
                atr_value = self._data_hub.get_indicator(symbol, "atr")
                if atr_value is not None: return self._safe_float(atr_value, default_atr)
        except Exception: pass
        return default_atr

    def _resolve_spread_aware_atr(self, *, raw_atr: float, execution_price: float, bid: float | None, ask: float | None) -> float:
        spread = float(ask - bid) if (bid is not None and ask is not None and ask >= bid) else 0.0
        return max(float(raw_atr), max(spread * 1.5, 1.0))

    def _get_current_regime(self, symbol: str, metadata: Mapping[str, Any] | None = None) -> str:
        """Return the most recent regime label with safe fallbacks."""
        try:
            if metadata and metadata.get("regime"): return str(metadata.get("regime"))
            regime_manager = self._regime_manager or getattr(self._preflight_validator, "_regime_manager", None)
            if regime_manager and hasattr(regime_manager, "get_current_regime"):
                regime_value = regime_manager.get_current_regime()
                if regime_value: return str(regime_value)
            state_snapshot = self._state_tracker.get_position_state(symbol)
            if state_snapshot and state_snapshot.get("regime"): return str(state_snapshot["regime"])
        except Exception: pass
        return "NEUTRAL"

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try: return float(value)
        except (TypeError, ValueError): return float(default)


OrderExecutionHub = ExecutionEngine

__all__ = ["ExecutionEngine", "OrderExecutionHub"]
