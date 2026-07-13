"""Thread-safe bracket manager with virtual (internal) SL/TP execution.

Runtime role:
- Replaces broker-side bracket orders with high-speed internal monitoring of
  stop-loss and take-profit for each open position.
- Supports ATR trailing, multi-target exits (TP1/TP2), partial scaling, and
  orphan/position resync.

Position in the pipeline:
    execution/order_manager.py -> THIS FILE (bracket_manager.py)
    (monitors fills/positions and issues exit orders back through the order path)

Owns / does NOT own:
- Owns: the virtual bracket state per position (SL/TP/trailing) and the decision
  to fire an exit.
- Does NOT own: entry decisions (runner) or raw order placement mechanics
  (order_manager). It decides WHEN to exit; placement still goes through the path.

Safe-edit notes:
- Live money and thread-sensitive (RLock-guarded). Preserve locking and the
  exit-trigger logic; a missed/duplicated exit is a real financial risk.
"""

from __future__ import annotations

from collections import deque
from contextlib import suppress
import threading
from threading import RLock
import time
import json
import logging
import os
import tempfile
from nifty_scalper_bot.config.env_utils import parse_float_env, parse_int_env
from datetime import datetime, timezone
import math
from pathlib import Path

_THREADING_MODULE = threading
_RLOCK_CLASS = RLock
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    cast,
    runtime_checkable,
)

from nifty_scalper_bot.utils.log_throttle import log_throttled as log_throttled_live
from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.symbols import normalize_symbol
from nifty_scalper_bot.execution.position_snapshot import (
    PositionSnapshot,
    PositionSnapshotError,
    decode_position_snapshot,
)

# --- NEW IMPORTS FOR WORLD-CLASS TRAILING ---
try:
    from nifty_scalper_bot.indicators.atr_provider import SafeATRProvider

    # Assuming TrailingSpec is defined in adaptive_trailing or we define a local fallback
    from nifty_scalper_bot.execution.adaptive_trailing import (
        AdaptiveTrailingController,
        TrailingSpec,
    )
except ImportError:
    SafeATRProvider = None
    AdaptiveTrailingController = None

    # Fallback definition if import fails
    @dataclass
    class TrailingSpec:  # type: ignore[no-redef]
        trail_by: float
        step: float
        activation: float


if TYPE_CHECKING:
    from journal.trade_journal import TradeJournal
    from nifty_scalper_bot.infra.metrics import MetricsCollector

# --------------------------------------------------------------------------
# METRICS INTEGRATION
# --------------------------------------------------------------------------
METRICS_AVAILABLE = False
METRICS = None

LOGGER = get_logger(__name__)


def _round_to_tick(price: float, tick_size: float = 0.05) -> float:
    """Round *price* to broker-compatible tick precision."""
    if tick_size <= 0:
        return round(price, 2)
    return round(round(float(price) / tick_size) * tick_size, 2)


def _normalize_bracket_side(side: str) -> str:
    """Normalize side to 'BUY'/'SELL' for consistent bracket comparisons.

    CRITICAL FIX (6 Feb 2026): attach_orphan_position stored 'LONG'/'SHORT'
    but _evaluate_exit_fast compared against 'BUY'/'SELL', causing SL to NEVER trigger.
    """
    s = side.strip().upper()
    if s in ("BUY", "LONG"):
        return "BUY"
    if s in ("SELL", "SHORT"):
        return "SELL"
    raise ValueError(f"Invalid bracket side: {side!r}. Must be BUY/LONG or SELL/SHORT.")


_FILLED_STATUSES = {"FILLED", "COMPLETE", "COMPLETED"}
_CANCELLED_STATUSES = {"CANCELLED", "REJECTED", "CANCELED"}
# Seconds after entry fill during which ticks older than the fill timestamp
# are rejected (pre-fill/replayed signal ticks must not fire a false exit).
STALE_TICK_ARM_WINDOW_SEC = 5.0


def tick_exchange_epoch(tick: Mapping[str, Any]) -> float | None:
    """Return the tick's exchange timestamp as epoch seconds, or None."""
    raw = tick.get("exchange_timestamp") or tick.get("timestamp")
    if raw is None:
        return None
    if hasattr(raw, "timestamp"):
        try:
            return float(raw.timestamp())
        except (TypeError, ValueError, OSError):
            return None
    if isinstance(raw, (int, float)):
        value = float(raw)
        return value / 1000.0 if value > 1e12 else value
    return None


_PROTECTIVE_EXIT_REASON_TOKENS = (
    "HARD_SL_BREACH",
    "WATCHDOG_HARD_SL",
    "FORCED_SL_EXIT",
    "EOD_FLATTEN",
    "EXIT_ESCALATED",
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------
# PROTOCOLS
# --------------------------------------------------------------------------
@runtime_checkable
class SupportsCancelOrder(Protocol):
    """Protocol representing broker cancel capability."""

    def cancel_order(self, order_id: str, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class SupportsModifyOrder(Protocol):
    """Protocol representing broker order modification capability."""

    def modify_order(self, order_id: str, **kwargs: Any) -> Any: ...


# --------------------------------------------------------------------------
# DATA STRUCTURES
# --------------------------------------------------------------------------


class BracketExitLifecycle(str, Enum):
    OPEN_PENDING_FILL = "OPEN_PENDING_FILL"
    OPEN_ACTIVE = "OPEN_ACTIVE"
    EXIT_TRIGGERED = "EXIT_TRIGGERED"
    EXIT_ORDER_PENDING = "EXIT_ORDER_PENDING"
    EXIT_ORDER_SUBMITTED = "EXIT_ORDER_SUBMITTED"
    EXIT_PARTIALLY_FILLED = "EXIT_PARTIALLY_FILLED"
    EXIT_FILLED = "EXIT_FILLED"
    EXIT_REJECTED_RETRYABLE = "EXIT_REJECTED_RETRYABLE"
    EXIT_REJECTED_FATAL = "EXIT_REJECTED_FATAL"
    EXIT_RECONCILED_FLAT = "EXIT_RECONCILED_FLAT"
    EXIT_FAILED_ESCALATED = "EXIT_FAILED_ESCALATED"
    CLOSED = "CLOSED"


class BracketTickDecision(str, Enum):
    HOLD = "HOLD"
    TRAIL_UPDATED = "TRAIL_UPDATED"
    EXIT_STOP = "EXIT_STOP"
    EXIT_TARGET = "EXIT_TARGET"
    EXIT_RISK = "EXIT_RISK"


@dataclass
class TargetLevel:
    """Represents a partial profit target level."""

    price: float
    quantity: int
    executed: bool = False
    name: str = "TP"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe target snapshot."""
        return {
            "price": float(self.price),
            "quantity": int(self.quantity),
            "executed": bool(self.executed),
            "name": str(self.name),
        }


@dataclass
class BracketState:
    """
    State container for a managed trade exit.
    Held in memory; survives restarts if persisted via TradeStore.
    """

    entry_order_id: str
    symbol: str
    side: str  # Entry Side (BUY/SELL)
    quantity: int  # Original Quantity
    entry_price: float

    # Execution Triggers
    sl_trigger_price: float
    tp_trigger_price: float  # Final/Ultimate TP
    initial_sl_trigger_price: float = 0.0

    # Multi-Target & Scaling State (NEW)
    remaining_quantity: int = 0
    tp_levels: List[TargetLevel] = field(default_factory=list)

    # Trailing & Logic State (NEW)
    is_virtual: bool = True
    active: bool = True  # If False, waits for confirmation or is finished
    trailing_enabled: bool = True
    trailing_config: Dict[str, Any] = field(
        default_factory=dict
    )  # e.g. {'mode': 'ATR', 'mult': 1.5}
    virtual_sl_id: str = ""  # ID for the Adaptive Controller

    # Market Data Tracking (NEW)
    highest_ltp: float = 0.0  # High water mark since entry (for BUY)
    lowest_ltp: float = float("inf")  # Low water mark since entry (for SELL)
    last_ltp: float = 0.0  # Latest price seen
    previous_ltp: float = 0.0  # Previous tick for stop-loss cross detection

    # Metadata
    tag: str | None = None
    entry_order_intent: str = "ENTRY"
    trade_lifecycle_id: str | None = None
    linked_exit_order_ids: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    exit_executed: bool = False
    # Tracks the most recently submitted (but not yet confirmed filled) exit order.
    # Set by _fire_exits_batch / _watchdog_exit_loop when the exit order is placed.
    # Cleared to None once the exit is confirmed FILLED, or used by
    # _handle_order_rejected / on_order_update(CANCELLED) to reactivate the bracket
    # when Zerodha rejects/cancels the exit order.
    pending_exit_order_id: str | None = None
    exit_in_progress: bool = False
    entry_confirmed: bool = False
    monitoring_only: bool = False
    entry_status: str = "PENDING_ENTRY"
    exit_state: str = BracketExitLifecycle.OPEN_PENDING_FILL.value
    exit_order_id: str | None = None
    entry_fill_price: float | None = None
    # Wall-clock time the entry fill was confirmed (bracket activation).
    # Used to reject pre-fill/replayed ticks against a freshly activated bracket.
    entry_fill_ts: float | None = None
    exit_reason: str | None = None
    exit_triggered_at: float | None = None
    exit_attempt_count: int = 0
    last_exit_attempt_at: float | None = None
    last_exit_error: str | None = None
    exit_pending: bool = False
    next_exit_attempt_at: float | None = None
    last_exit_summary_at: float = 0.0
    closed_at: float | None = None
    position_flat_confirmed: bool = False
    flat_nonterminal_since_monotonic: float | None = None
    flat_nonterminal_since_utc: str | None = None
    close_source: str | None = None
    exit_price: float | None = None
    escalated_at: float | None = None
    # True once the forced MARKET exit on escalation has been fired, so it happens
    # exactly once per bracket (not every reconcile cycle).
    _market_escalation_fired: bool = False
    _atr_warning_logged: bool = False
    ledger_realized_pnl: dict[str, Any] | None = None
    _ledger_pending_entry_price: float | None = None
    _ledger_pending_exit_order_id: str | None = None
    _ledger_pending_exit_quantity: int = 0
    _ledger_pending_exit_price: float | None = None
    _ledger_pending_exit_target: str | None = None
    _ledger_release_hook_fired: bool = False
    _filled_exit_sync_started_at: float = 0.0
    _filled_exit_sync_order_id: str | None = None
    _last_exit_reconcile_at: float = 0.0
    last_processed_tick_id: str | None = None
    last_trail_price: float | None = None
    trail_revision: int = 0

    @property
    def bracket_id(self) -> str:
        return self.entry_order_id

    @property
    def entry_qty(self) -> int:
        return self.quantity

    @property
    def remaining_qty(self) -> int:
        return self.remaining_quantity

    @remaining_qty.setter
    def remaining_qty(self, value: int) -> None:
        self.remaining_quantity = int(value)

    @property
    def current_sl(self) -> float:
        return self.sl_trigger_price

    @current_sl.setter
    def current_sl(self, value: float) -> None:
        self.sl_trigger_price = float(value)

    @property
    def current_target(self) -> float:
        return self.tp_trigger_price

    @current_target.setter
    def current_target(self, value: float) -> None:
        self.tp_trigger_price = float(value)

    @property
    def trailing_state(self) -> Dict[str, Any]:
        return self.trailing_config

    def __post_init__(self):
        self.side = _normalize_bracket_side(self.side)
        # Auto-initialize state fields if not set
        if self.remaining_quantity == 0:
            self.remaining_quantity = self.quantity

        # Initialize High/Low water marks with entry price
        if self.highest_ltp == 0.0 or self.highest_ltp < self.entry_price:
            self.highest_ltp = self.entry_price

        if self.lowest_ltp == float("inf") or self.lowest_ltp > self.entry_price:
            self.lowest_ltp = self.entry_price
        if self.initial_sl_trigger_price <= 0:
            self.initial_sl_trigger_price = self.sl_trigger_price

    # ✅ FIX: Add Serialization for Persistence
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for SQLite persistence."""
        return {
            "entry_order_id": self.entry_order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "sl_trigger_price": self.sl_trigger_price,
            "tp_trigger_price": self.tp_trigger_price,
            "initial_sl_trigger_price": self.initial_sl_trigger_price,
            "remaining_quantity": self.remaining_quantity,
            "tp_levels": [tp.to_dict() for tp in self.tp_levels],
            "is_virtual": self.is_virtual,
            "active": self.active,
            "trailing_enabled": self.trailing_enabled,
            "trailing_config": self.trailing_config,
            "highest_ltp": self.highest_ltp,
            "lowest_ltp": self.lowest_ltp,
            "last_ltp": self.last_ltp,
            "tag": self.tag,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "exit_executed": self.exit_executed,
            "pending_exit_order_id": self.pending_exit_order_id,
            "previous_ltp": self.previous_ltp,
            "exit_in_progress": self.exit_in_progress,
            "entry_confirmed": self.entry_confirmed,
            "monitoring_only": self.monitoring_only,
            "entry_status": self.entry_status,
            "exit_state": self.exit_state,
            "exit_order_id": self.exit_order_id or self.pending_exit_order_id,
            "entry_fill_price": self.entry_fill_price,
            "exit_reason": self.exit_reason,
            "exit_triggered_at": self.exit_triggered_at,
            "exit_attempt_count": self.exit_attempt_count,
            "last_exit_attempt_at": self.last_exit_attempt_at,
            "last_exit_error": self.last_exit_error,
            "exit_pending": self.exit_pending,
            "next_exit_attempt_at": self.next_exit_attempt_at,
            "closed_at": self.closed_at,
            "position_flat_confirmed": self.position_flat_confirmed,
            "flat_nonterminal_since_utc": self.flat_nonterminal_since_utc,
            "close_source": self.close_source,
            "exit_price": self.exit_price,
            "escalated_at": self.escalated_at,
            "atr_warning_logged": self._atr_warning_logged,
            "ledger_realized_pnl": self.ledger_realized_pnl,
            "ledger_pending_entry_price": self._ledger_pending_entry_price,
            "ledger_pending_exit_order_id": self._ledger_pending_exit_order_id,
            "ledger_pending_exit_quantity": self._ledger_pending_exit_quantity,
            "ledger_pending_exit_price": self._ledger_pending_exit_price,
            "ledger_pending_exit_target": self._ledger_pending_exit_target,
            "ledger_release_hook_fired": self._ledger_release_hook_fired,
            "market_escalation_fired": self._market_escalation_fired,
            "last_exit_summary_at": self.last_exit_summary_at,
            "filled_exit_sync_started_at": self._filled_exit_sync_started_at,
            "filled_exit_sync_order_id": self._filled_exit_sync_order_id,
            "last_exit_reconcile_at": self._last_exit_reconcile_at,
            "last_processed_tick_id": self.last_processed_tick_id,
            "last_trail_price": self.last_trail_price,
            "trail_revision": self.trail_revision,
        }


@dataclass
class ExitExecutionResult:
    """Result envelope for bracket exit execution attempts."""

    submitted: bool
    confirmed: bool
    order_id: str | None
    filled_qty: int
    reason: str
    status: str | None = None


@dataclass
class SubmitExitOrderResult:
    """Structured result returned by submit_exit_order."""

    accepted: bool
    order_id: str | None
    status: str
    error_type: str | None = None
    error_message: str | None = None
    retryable: bool = False
    broker_payload: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "order_id": self.order_id,
            "status": self.status,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "retryable": self.retryable,
            "broker_payload": dict(self.broker_payload),
        }


# Mock Journal for Adaptive Controller (In-Memory)
class MockJournal:
    def set(self, key, value):
        pass

    def get(self, key):
        return None


class BracketManager:
    """
    The 'Sniper' Engine.
    Monitors LTP/ATR internally and fires immediate MARKET exits when levels are hit.
    Supports TP1/TP2 scaling, ATR-based Trailing, and Broker Sync.
    """

    def __init__(
        self,
        order_manager: Any,
        indicator_engine: Any = None,
        market_data: Any = None,
        trade_journal: "TradeJournal | None" = None,
    ):
        """Initialize the canonical bracket runtime and recover durable state first."""
        self.order_manager = order_manager
        self._trade_journal = trade_journal
        self._brackets: Dict[str, BracketState] = {}
        self._order_to_entry: Dict[str, str] = {}
        self._symbol_map: Dict[str, List[str]] = {}
        self._indicator_engine = indicator_engine
        self._market_data = market_data
        self._atr_provider = None
        if SafeATRProvider and indicator_engine:
            self._atr_provider = SafeATRProvider(indicator_engine, max_cache_age=60.0)
        self._trailing_controllers: Dict[str, Any] = {}
        self._recent_ticks: Dict[str, deque] = {}
        self._max_tick_history = 20
        self._orphan_retry_count: Dict[str, int] = {}
        self._orphan_retry_last_attempt: Dict[str, float] = {}
        self._exit_executor: Callable[[str, int], Any] | None = None
        self._trailing_controller_factory: Callable[[BracketState], Any] | None = None
        self._on_exit_complete_hook: Callable[[str], None] | None = None
        self._on_position_open_priority_hook: Callable[[str], None] | None = None
        self._on_position_closed_priority_hook: Callable[[str], None] | None = None
        self._active_bracket_symbols_hook: Callable[[Iterable[str]], None] | None = None
        self._current_atr: Dict[str, float] = {}
        self._last_price_cache: Dict[str, float] = {}
        self._exit_cooldowns: Dict[str, float] = {}
        self._notifier: Callable[[str, Mapping[str, object] | None], None] | None = None
        self._trail_notify_at: Dict[str, float] = {}
        self._trail_notify_sl: Dict[str, float] = {}
        self._tick_error_logged: Dict[str, bool] = {}
        self._throttle_log_at: Dict[str, float] = {}
        self._lock = _RLOCK_CLASS()
        self._reconcile_lock = threading.Lock()
        self._running = True
        self._persistence_degraded_reason: str | None = None
        self._recovery_degraded_reason: str | None = None
        self._last_persist_success_at: float | None = None
        self._state_storage_path: str | None = None
        self._state_storage_durable = False

        self._auto_reduce_sl = True
        self._pending_entry_reconcile_after_sec = max(
            1.0,
            parse_float_env(
                os.getenv("BRACKET_PENDING_ENTRY_RECONCILE_AFTER_SEC"), 5.0
            ),
        )
        self._stale_cleanup_age = 86400
        self._trail_tier1_pct = parse_float_env(os.getenv("TRAIL_TIER1_PCT"), 1.0)
        self._trail_tier2_pct = parse_float_env(os.getenv("TRAIL_TIER2_PCT"), 2.0)
        self._trail_tier3_pct = parse_float_env(os.getenv("TRAIL_TIER3_PCT"), 4.0)
        self._trail_tier4_pct = parse_float_env(os.getenv("TRAIL_TIER4_PCT"), 6.0)
        self._exit_retry_enabled = os.getenv(
            "EXIT_RETRY_ENABLE", "true"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._exit_max_retry_attempts = max(
            1, parse_int_env(os.getenv("EXIT_MAX_RETRY_ATTEMPTS"), 4)
        )
        self._exit_retry_backoffs = self._parse_exit_backoffs(
            os.getenv("EXIT_RETRY_BACKOFF_SECONDS", "1,2,5")
        )
        self._exit_fatal_error_patterns = tuple(
            value.strip().lower()
            for value in os.getenv("EXIT_RETRY_FATAL_ERROR_PATTERNS", "").split(",")
            if value.strip()
        )
        self._exit_reconcile_interval_seconds = max(
            0.25,
            parse_float_env(os.getenv("EXIT_POSITION_RECONCILE_INTERVAL_SECONDS"), 1.0),
        )
        self._exit_flat_confirmation_required = os.getenv(
            "EXIT_FLAT_CONFIRMATION_REQUIRED", "true"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._exit_unresolved_escalation_seconds = max(
            1.0, parse_float_env(os.getenv("EXIT_UNRESOLVED_ESCALATION_SECONDS"), 15.0)
        )
        self._exit_continue_retry_after_escalation = os.getenv(
            "EXIT_CONTINUE_RETRY_AFTER_ESCALATION", "false"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._exit_force_market_on_escalation = os.getenv(
            "EXIT_FORCE_MARKET_ON_ESCALATION", "true"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._exit_protective_order_mode = (
            str(os.getenv("EXIT_PROTECTIVE_ORDER_MODE", "MARKET") or "MARKET")
            .strip()
            .upper()
        )
        self._exit_marketable_limit_slippage_ticks = max(
            0, parse_int_env(os.getenv("EXIT_MARKETABLE_LIMIT_SLIPPAGE_TICKS"), 5)
        )
        self._exit_marketable_limit_max_slippage_pct = max(
            0.0,
            parse_float_env(os.getenv("EXIT_MARKETABLE_LIMIT_MAX_SLIPPAGE_PCT"), 2.0),
        )
        self._exit_fallback_to_market_on_quote_missing = _env_bool(
            "EXIT_FALLBACK_TO_MARKET_ON_QUOTE_MISSING", True
        )

        if _env_bool("BRACKET_AUTO_RESTORE", True):
            try:
                self.load_state()
            except Exception as exc:  # noqa: BLE001
                self._mark_persistence_degraded("startup_restore_failed", exc)

        self._watchdog_thread = threading.Thread(
            target=self._watchdog_exit_loop,
            name="bracket-watchdog",
            daemon=True,
        )
        self._watchdog_thread.start()

    @staticmethod
    def _parse_exit_backoffs(raw: str | None) -> list[float]:
        values: list[float] = []
        for chunk in str(raw or "").split(","):
            try:
                value = float(chunk.strip())
            except ValueError:
                continue
            if value >= 0:
                values.append(value)
        return values or [1.0, 2.0, 5.0]

    def has_unresolved_exit(self) -> bool:
        """Return whether persistence, recovery, or an exit lifecycle blocks entries."""
        if self._persistence_degraded_reason or self._recovery_degraded_reason:
            return True
        unresolved = {
            BracketExitLifecycle.EXIT_TRIGGERED.value,
            BracketExitLifecycle.EXIT_ORDER_PENDING.value,
            BracketExitLifecycle.EXIT_ORDER_SUBMITTED.value,
            BracketExitLifecycle.EXIT_REJECTED_RETRYABLE.value,
            BracketExitLifecycle.EXIT_FAILED_ESCALATED.value,
        }
        with self._lock:
            return any(
                bracket.remaining_quantity > 0
                and (bracket.exit_pending or bracket.exit_state in unresolved)
                for bracket in self._brackets.values()
            )

    def get_first_unresolved_exit_bracket_id(self) -> str | None:
        """Return the first recovery blocker or unresolved bracket identifier."""
        if self._persistence_degraded_reason:
            return f"persistence:{self._persistence_degraded_reason}"
        if self._recovery_degraded_reason:
            return f"recovery:{self._recovery_degraded_reason}"
        with self._lock:
            for bracket in self._brackets.values():
                if bracket.remaining_quantity <= 0:
                    continue
                if bracket.exit_pending or bracket.exit_state.startswith("EXIT_"):
                    if bracket.exit_state not in {
                        BracketExitLifecycle.EXIT_FILLED.value,
                        BracketExitLifecycle.EXIT_RECONCILED_FLAT.value,
                        BracketExitLifecycle.CLOSED.value,
                    }:
                        return bracket.bracket_id
        return None

    def attach_exit_executor(self, executor: Callable[[str, int], Any] | None) -> None:
        """Attach an external market-exit executor. Args: executor; Returns: None; Raises: None."""
        self._exit_executor = executor

    def attach_on_exit_complete(self, hook: Callable[[str], None] | None) -> None:
        """Attach callback fired with symbol when bracket fully closes. Lets runner clear orchestrator direction lock."""
        self._on_exit_complete_hook = hook

    def attach_open_position_priority_hooks(
        self,
        on_position_open: Callable[[str], None] | None = None,
        on_position_closed: Callable[[str], None] | None = None,
    ) -> None:
        """Attach MDM open-position priority hooks for immediate tick prioritization."""
        self._on_position_open_priority_hook = on_position_open
        self._on_position_closed_priority_hook = on_position_closed

    def attach_active_bracket_symbols_hook(
        self, hook: Callable[[Iterable[str]], None] | None = None
    ) -> None:
        """Attach MDM active-bracket symbol sync hook."""
        self._active_bracket_symbols_hook = hook
        self._sync_active_bracket_symbols_to_mdm()

    def _sync_active_bracket_symbols_to_mdm(self) -> None:
        """Publish current bracket-owned symbols to MDM without adding a registry."""
        hook = self._active_bracket_symbols_hook
        if hook is None and self._market_data is not None:
            hook = getattr(self._market_data, "set_active_bracket_symbols", None)
        if not callable(hook):
            return
        with self._lock:
            symbols = [
                bracket.symbol
                for bracket in self._brackets.values()
                if getattr(bracket, "remaining_quantity", 0) > 0
            ]
        try:
            hook(symbols)
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("active bracket symbol sync failed: %s", exc)

    def _notify_open_position_priority(self, action: str, symbol: str) -> None:
        hook = (
            self._on_position_open_priority_hook
            if action == "open"
            else self._on_position_closed_priority_hook
        )
        if hook is None:
            return
        try:
            hook(symbol)
            if action == "open":
                LOGGER.info(
                    "OPEN_POSITION_PRIORITY_REGISTERED symbol=%s",
                    symbol,
                    extra={
                        "event": "OPEN_POSITION_PRIORITY_REGISTERED",
                        "symbol": symbol,
                    },
                )
            else:
                LOGGER.info(
                    "OPEN_POSITION_PRIORITY_REMOVED symbol=%s",
                    symbol,
                    extra={"event": "OPEN_POSITION_PRIORITY_REMOVED", "symbol": symbol},
                )
        except Exception as exc:  # noqa: BLE001 - hook must not break bracket lifecycle
            LOGGER.exception(
                "OPEN_POSITION_PRIORITY_HOOK_FAILED action=%s symbol=%s error=%s",
                action,
                symbol,
                exc,
                extra={
                    "event": "OPEN_POSITION_PRIORITY_HOOK_FAILED",
                    "action": action,
                    "symbol": symbol,
                    "error": str(exc),
                },
            )

    def attach_trailing_controller_factory(
        self, factory: Callable[[BracketState], Any] | None
    ) -> None:
        """Attach a trailing controller factory. Args: factory; Returns: None; Raises: None."""
        self._trailing_controller_factory = factory

    def _watchdog_exit_loop(self) -> None:
        """Run watchdog checks and force exits on hard SL breach. Args: none; Returns: none; Raises: none."""
        while self._running:
            try:
                pending: list[tuple[BracketState, dict[str, Any]]] = []
                reconcile_candidates: list[BracketState] = []
                with self._lock:
                    for bracket in self._brackets.values():
                        reconcile_candidates.append(bracket)
                        ltp = float(bracket.last_ltp or 0.0)
                        if bracket.exit_pending and bracket.remaining_quantity > 0:
                            pending.append(
                                (
                                    bracket,
                                    {
                                        "type": "RECONCILE",
                                        "price": ltp,
                                        "qty": bracket.remaining_quantity,
                                        "reason": bracket.exit_reason or "EXIT_PENDING",
                                    },
                                )
                            )
                            continue
                        if (
                            not bracket.active
                            or not bracket.entry_confirmed
                            or bracket.entry_status
                            not in {"ACTIVE", BracketExitLifecycle.OPEN_ACTIVE.value}
                            or bracket.exit_executed
                            or bracket.exit_in_progress
                            or bracket.remaining_quantity <= 0
                            or bracket.sl_trigger_price <= 0
                            or ltp <= 0
                        ):
                            self._log_throttled(
                                "debug",
                                f"watchdog_skip_{bracket.entry_order_id}",
                                60.0,
                                "BRACKET_WATCHDOG_SKIP symbol=%s reason=pending_or_inactive",
                                bracket.symbol,
                            )
                            continue
                        if bracket.side == "BUY" and self._sl_crossed(bracket, ltp):
                            pending.append(
                                (
                                    bracket,
                                    {
                                        "type": "SL",
                                        "price": ltp,
                                        "qty": bracket.remaining_quantity,
                                        "reason": "WATCHDOG_HARD_SL",
                                    },
                                )
                            )
                        elif bracket.side == "SELL" and self._sl_crossed(bracket, ltp):
                            pending.append(
                                (
                                    bracket,
                                    {
                                        "type": "SL",
                                        "price": ltp,
                                        "qty": bracket.remaining_quantity,
                                        "reason": "WATCHDOG_HARD_SL",
                                    },
                                )
                            )
                for bracket in reconcile_candidates:
                    self._reconcile_pending_entry(bracket)
                if pending:
                    self._log_throttled(
                        "critical",
                        "watchdog_exit_trigger",
                        3.0,
                        "WATCHDOG EXIT TRIGGER count=%s",
                        len(pending),
                    )
                    self._fire_exits_batch(pending)
                time.sleep(0.25)
            except Exception as e:
                LOGGER.error("Failure in _watchdog_exit_loop: %s", e)
                time.sleep(0.25)

    def _broker_order_filled(self, order_id: str) -> tuple[bool, float | None]:
        broker = getattr(self.order_manager, "_broker", None)
        if broker is None:
            return False, None
        for attr in ("get_order_status", "order_status"):
            fn = getattr(broker, attr, None)
            if callable(fn):
                try:
                    status = fn(order_id)
                except Exception:
                    continue
                if isinstance(status, Mapping):
                    status_text = str(status.get("status") or "").upper()
                    if status_text in _FILLED_STATUSES:
                        avg_price = (
                            status.get("average_price")
                            or status.get("avg_price")
                            or status.get("price")
                        )
                        try:
                            return True, float(avg_price) if avg_price else None
                        except (TypeError, ValueError):
                            return True, None
        get_orders = getattr(broker, "get_orders", None)
        if not callable(get_orders):
            get_orders = getattr(broker, "orders", None)
        if callable(get_orders):
            try:
                orders = get_orders() or []
            except Exception:
                orders = []
            for order in orders:
                if not isinstance(order, Mapping):
                    continue
                if str(order.get("order_id") or "") != str(order_id):
                    continue
                status_text = str(order.get("status") or "").upper()
                if status_text in _FILLED_STATUSES:
                    avg_price = (
                        order.get("average_price")
                        or order.get("avg_price")
                        or order.get("price")
                    )
                    try:
                        return True, float(avg_price) if avg_price else None
                    except (TypeError, ValueError):
                        return True, None
        return False, None

    def _reconcile_pending_entry(self, bracket: BracketState) -> None:
        if bracket.entry_confirmed or bracket.monitoring_only:
            return
        age = time.time() - float(bracket.created_at or 0.0)
        if age < self._pending_entry_reconcile_after_sec:
            return
        filled, fill_price = self._broker_order_filled(bracket.entry_order_id)
        if filled:
            self.confirm_entry_fill(
                bracket.entry_order_id, fill_price or bracket.entry_price
            )

    def _log_throttled(
        self,
        level: str,
        key: str,
        interval_sec: float,
        message: str,
        *args: object,
    ) -> None:
        """Emit a throttled log event. Args: level,key,interval_sec,message,args; Returns: none; Raises: none."""
        level_value = getattr(logging, str(level).upper(), logging.INFO)
        event = key.split(":", 1)[0].upper() if key else "BRACKET_MANAGER_THROTTLED_LOG"
        if log_throttled_live(
            LOGGER, level_value, event, key, interval_sec, message, *args
        ):
            self._throttle_log_at[key] = time.monotonic()

    def shutdown(self) -> None:
        """Stop watchdog processing. Args: none; Returns: none; Raises: none."""
        self._running = False

    # --------------------------------------------------------------------------
    # 1. CORE API (Backward Compatible & Enhanced)
    # --------------------------------------------------------------------------

    def set_notifier(
        self, notifier: Callable[[str, Mapping[str, object] | None], None] | None
    ) -> None:
        """Attach notifier callback for bracket lifecycle updates.

        Args:
            notifier: Callback accepting (event, payload) or ``None`` to clear.

        Returns:
            None.

        Raises:
            None.
        """
        LOGGER.debug(
            "Entered set_notifier",
            extra={"event": "bracket_manager_set_notifier_enter"},
        )
        try:
            self._notifier = notifier
            if notifier is None:
                LOGGER.info(
                    "Bracket notifier cleared",
                    extra={"event": "bracket_manager_notifier_cleared"},
                )
            else:
                LOGGER.info(
                    "Bracket notifier attached",
                    extra={"event": "bracket_manager_notifier_attached"},
                )
        except Exception as exc:
            LOGGER.error(
                "Failure in set_notifier: %s",
                exc,
                extra={"event": "bracket_manager_notifier_failed"},
                exc_info=exc,
            )

    def _notify_event(
        self, event: str, payload: Mapping[str, object] | None = None
    ) -> None:
        """Emit a bracket lifecycle notification when configured.

        Args:
            event: Event name describing the lifecycle action.
            payload: Optional metadata payload for the notifier.

        Returns:
            None.

        Raises:
            None.
        """
        LOGGER.debug(
            "Entered _notify_event",
            extra={"event": "bracket_manager_notify_enter", "notify_event": event},
        )
        if self._notifier is None:
            return
        try:
            self._notifier(event, payload)
        except Exception as exc:
            LOGGER.error(
                "Failure in _notify_event: %s",
                exc,
                extra={"event": "bracket_manager_notify_failed", "notify_event": event},
                exc_info=exc,
            )

    def _should_notify_trail(self, entry_id: str, new_sl: float, old_sl: float) -> bool:
        """Decide if a trailing update should be notified.

        Args:
            entry_id: Bracket entry identifier for tracking cooldowns.
            new_sl: Proposed updated stop-loss level.
            old_sl: Previous stop-loss level before the update.

        Returns:
            ``True`` when a notification should be emitted.

        Raises:
            None.
        """
        try:
            now = time.time()
            last_ts = self._trail_notify_at.get(entry_id, 0.0)
            last_sl = self._trail_notify_sl.get(entry_id, old_sl)
            price_delta = abs(new_sl - last_sl)
            pct_delta = (price_delta / last_sl * 100.0) if last_sl > 0 else 100.0
            min_seconds = parse_float_env(os.getenv("TRAIL_NOTIFY_COOLDOWN_SEC"), 60)
            min_pct = parse_float_env(os.getenv("TRAIL_NOTIFY_MIN_PCT"), 0.5)
            if (now - last_ts) >= min_seconds or pct_delta >= min_pct:
                self._trail_notify_at[entry_id] = now
                self._trail_notify_sl[entry_id] = new_sl
                return True
            return False
        except Exception as exc:
            LOGGER.error(
                "Failure in _should_notify_trail: %s",
                exc,
                extra={
                    "event": "bracket_manager_trail_notify_failed",
                    "entry_id": entry_id,
                },
                exc_info=exc,
            )
            return False

    def place_bracket_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        stop_loss: float,
        take_profit: float,
        tag: str = "auto_bracket",
        entry_order_id: str | None = None,
        # --- NEW OPTIONAL ARGUMENTS ---
        tp1_price: float | None = None,
        tp1_qty: int | None = None,
        trailing_atr_mult: float | None = None,
    ) -> str:
        """
        Legacy Bridge: Allows old code to call this method.
        Internally converts request to a Virtual Bracket.
        """
        if not entry_order_id:
            # Generate a synthetic ID if caller didn't provide one
            entry_order_id = f"virt_{int(time.time())}_{symbol}"

        self.register_virtual_bracket(
            order_id=entry_order_id,
            symbol=symbol,
            side=_normalize_bracket_side(side),
            qty=quantity,
            price=price,
            sl=stop_loss,
            tp=take_profit,
            tag=tag,
            tp1_price=tp1_price,
            tp1_qty=tp1_qty,
            trailing_atr_mult=trailing_atr_mult,
            activate_immediately=False,  # 🔒 Safety: Wait for fill or tick
        )

        LOGGER.info(f"🛡️ Bracket Registered: {entry_order_id} for {symbol}")
        return entry_order_id  # 🟢 FIX 2: Return the ID

    def register_virtual_bracket(
        self,
        order_id: str,
        symbol: str,
        side: str,
        qty: int,
        price: float,
        sl: float,
        tp: float,
        tag: str = "virtual",
        # --- NEW OPTIONAL ARGUMENTS ---
        tp1_price: float | None = None,
        tp1_qty: int | None = None,
        trailing_atr_mult: float | None = None,
        activate_immediately: bool = False,
        intent: str | None = None,
        resolved_lot_size: int | None = None,
    ) -> None:
        """Register a position for virtual bracket monitoring.

        Args:
            order_id: Entry order identifier for the bracket.
            symbol: Trading symbol for the bracket.
            side: Trade direction ("BUY" or "SELL").
            qty: Position size to monitor.
            price: Entry price used for bracket tracking.
            sl: Stop-loss trigger price.
            tp: Take-profit trigger price.
            tag: Optional strategy tag for auditing.
            tp1_price: Optional first profit target price.
            tp1_qty: Optional first profit target quantity.
            trailing_atr_mult: Optional ATR-based trailing multiplier.
            activate_immediately: Whether the bracket is active on registration.
            resolved_lot_size: Authoritative instrument lot size for TP allocation.

        Returns:
            None.

        Raises:
            None.
        """
        symbol = normalize_symbol(symbol)
        normalized_intent = str(intent or "").strip().upper()
        if normalized_intent in {"EXIT", "REDUCE"}:
            LOGGER.error(
                "BRACKET_ENTRY_REJECTED_FOR_EXIT_ORDER order_id=%s symbol=%s intent=%s",
                order_id,
                symbol,
                normalized_intent,
                extra={
                    "event": "BRACKET_ENTRY_REJECTED_FOR_EXIT_ORDER",
                    "order_id": order_id,
                    "symbol": symbol,
                    "intent": normalized_intent,
                },
            )
            return
        side = _normalize_bracket_side(side)
        if symbol in self.active_brackets:
            LOGGER.info(
                "Condition met: bracket_symbol_already_active",
                extra={"event": "bracket_symbol_already_active", "symbol": symbol},
            )
            return
        with self._lock:
            # 1. Deduplication: Update existing if found
            if order_id in self._brackets:
                LOGGER.warning(f"Bracket {order_id} exists. Updating triggers.")
                existing = self._brackets[order_id]
                # Only update non-zero values to avoid overwriting with bad data
                if sl > 0:
                    existing.sl_trigger_price = sl
                if tp > 0:
                    existing.tp_trigger_price = tp
                # Reset quantity if re-registering (e.g. scale-in)
                existing.quantity = abs(qty)
                existing.remaining_quantity = abs(qty)
                existing.exit_state = BracketExitLifecycle.OPEN_PENDING_FILL.value
                existing.exit_pending = False
                self.save_state()  # Persist updates
                self._sync_active_bracket_symbols_to_mdm()
                return

            # 2. Setup Trailing Config
            t_config = {}
            if trailing_atr_mult:
                t_config = {"mode": "ATR", "mult": trailing_atr_mult}
            elif self._auto_reduce_sl:
                # Default logic if enabled but no explicit ATR
                t_config = {"mode": "STANDARD"}

            # 3. Setup TP Levels (Partial Exits)
            targets = []
            lot_size = self._resolve_bracket_lot_size(resolved_lot_size)
            if (
                lot_size is not None
                and abs(qty) > lot_size
                and tp1_price
                and tp1_qty
                and tp1_qty >= lot_size
                and tp1_qty < abs(qty)
                and int(tp1_qty) % lot_size == 0
            ):
                targets.append(
                    TargetLevel(price=tp1_price, quantity=tp1_qty, name="TP1")
                )
                LOGGER.info(
                    f"🔹 Configured TP1 for {symbol}: {tp1_price} (Qty: {tp1_qty})"
                )

            # 4. ORPHAN SAFETY CHECK (CRITICAL)
            # If SL or TP are 0.0 (invalid/missing), we MUST NOT enable active exiting.
            # This prevents the bot from seeing LTP > 0 as "Target Hit".
            status_mode = "ACTIVE" if activate_immediately else "PENDING_ENTRY"
            if sl <= 0 and tp <= 0:
                LOGGER.warning(
                    f"⚠️ Bracket {symbol} has zero SL/TP (SL={sl}, TP={tp}). "
                    "Setting MONITORING_ONLY mode to prevent suicide exit."
                )
                # Force inactive so it monitors but doesn't fire
                status_mode = "MONITORING_ONLY"
                activate_immediately = False

            # 5. Create State Object
            state = BracketState(
                entry_order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=abs(qty),
                remaining_quantity=abs(qty),
                entry_price=price,
                sl_trigger_price=sl,
                tp_trigger_price=tp,  # This is effectively Final TP
                initial_sl_trigger_price=sl,
                tp_levels=targets,
                is_virtual=True,
                active=activate_immediately,
                entry_confirmed=activate_immediately,
                entry_status="ACTIVE" if activate_immediately else "PENDING_ENTRY",
                exit_state=(
                    BracketExitLifecycle.OPEN_ACTIVE.value
                    if activate_immediately
                    else BracketExitLifecycle.OPEN_PENDING_FILL.value
                ),
                entry_fill_price=price if activate_immediately else None,
                tag=tag,
                entry_order_intent=normalized_intent or "ENTRY",
                trade_lifecycle_id=order_id,
                trailing_config=t_config,
                virtual_sl_id=f"vsl_{order_id}",
            )
            state.monitoring_only = status_mode == "MONITORING_ONLY"

            self._brackets[order_id] = state

            if self._trailing_controller_factory:
                if state.entry_order_id in self._trailing_controllers:
                    return
                controller = self._trailing_controller_factory(state)
                self._trailing_controllers[state.entry_order_id] = controller

            # 6. Populate Indices
            self._order_to_entry[order_id] = order_id
            if symbol not in self._symbol_map:
                self._symbol_map[symbol] = []
            self._symbol_map[symbol].append(order_id)
            self._log_bracket_event(
                "BRACKET_REGISTERED",
                state,
                meta={"sl": sl, "tp": tp, "activate_immediately": activate_immediately},
            )
            LOGGER.info(
                "VIRTUAL_BRACKET_REGISTERED_PENDING entry_order_id=%s symbol=%s",
                order_id,
                symbol,
            )

            # 7. Initialize Adaptive Controller (The "Brain")
            if (
                not self._trailing_controller_factory
                and trailing_atr_mult
                and self._atr_provider
                and AdaptiveTrailingController
            ):
                try:
                    # Fallback trail distance must be premium-relative: the old
                    # absolute 20.0 points is ~13% on a ~150 option premium, so
                    # with stale/unavailable ATR the controller proposed stops
                    # far below the current SL and the monotonic guard rejected
                    # every update — trailing silently dead on the controller
                    # path while the legacy fallback math trails at ~2-3%.
                    _fallback_trail = max(
                        round(max(float(price or 0.0), 1.0) * 0.02, 2), 0.25
                    )
                    spec = TrailingSpec(
                        trail_by=_fallback_trail,  # ~2% of entry premium
                        step=0.25,  # Tighter early trailing step
                        activation=0.3,  # Earlier activation threshold
                    )

                    # We pass a lambda that returns the latest price from the bracket state
                    ctrl = AdaptiveTrailingController(
                        symbol=symbol,
                        side="LONG" if side == "BUY" else "SHORT",
                        entry=price,
                        sl_order_id=state.virtual_sl_id,
                        variety="virtual",
                        spec=spec,
                        get_ltp=lambda s: state.last_ltp,
                        modify_order=self._virtual_modify_sl,
                        atr_provider=self._atr_provider,
                        journal=MockJournal(),
                        atr_multiplier=trailing_atr_mult,
                    )
                    self._trailing_controllers[order_id] = ctrl
                    LOGGER.info(
                        f"🧠 Adaptive Controller Attached to {symbol} (Mult: {trailing_atr_mult}x)"
                    )

                except Exception as e:
                    LOGGER.error(f"Failed to attach Adaptive Controller: {e}")

            trail_msg = f"| Trail={t_config.get('mode', 'None')}"
            LOGGER.info(
                f"🛡️ Bracket Registered for {symbol} (Qty: {qty}): "
                f"Entry={price} | SL={sl} | TP={tp} {trail_msg} | Mode={status_mode}"
            )
            self._notify_event(
                "BRACKET_REGISTERED",
                {
                    "symbol": symbol,
                    "side": side,
                    "qty": abs(qty),
                    "entry": round(price, 2),
                    "sl": round(sl, 2) if sl else 0.0,
                    "tp": round(tp, 2) if tp else 0.0,
                    "mode": status_mode,
                },
            )

            # 8. Record metric & Persist
            if METRICS_AVAILABLE and METRICS:
                try:
                    METRICS.brackets_created.inc()
                except Exception as exc:
                    LOGGER.error(
                        "BRACKET_METRICS_INCREMENT_FAILED order_id=%s symbol=%s error=%s",
                        order_id,
                        symbol,
                        exc,
                        extra={
                            "event": "bracket_manager_metrics_increment_error",
                            "order_id": order_id,
                            "symbol": symbol,
                            "error_type": type(exc).__name__,
                        },
                    )

            # ✅ FIX: Persist immediately so we don't lose this if we crash now
            self.save_state()
            self._sync_active_bracket_symbols_to_mdm()

    def confirm_entry_fill(self, order_id: str, fill_price: float) -> None:
        """Activate a bracket once entry fill is confirmed.

        Args:
            order_id: Entry order identifier for the bracket.
            fill_price: Confirmed fill price from the broker.

        Returns:
            None.

        Raises:
            None.
        """
        with self._lock:
            bracket = self._brackets.get(order_id)
            if not bracket:
                LOGGER.warning(f"⚠️ No bracket found for order {order_id}")
                return
            entry_intent = str(
                getattr(bracket, "entry_order_intent", "ENTRY") or "ENTRY"
            ).upper()
            if entry_intent not in {"ENTRY", "SCALE_IN", "REVERSAL"}:
                LOGGER.error(
                    "BRACKET_ENTRY_FILL_REJECTED_FOR_EXIT_ORDER order_id=%s symbol=%s intent=%s",
                    order_id,
                    bracket.symbol,
                    entry_intent,
                    extra={
                        "event": "BRACKET_ENTRY_FILL_REJECTED_FOR_EXIT_ORDER",
                        "order_id": order_id,
                        "symbol": bracket.symbol,
                        "intent": entry_intent,
                    },
                )
                return

            # Update entry price with actual fill
            if fill_price and fill_price > 0:
                old_price = bracket.entry_price
                bracket.entry_price = fill_price

                # Adjust SL/TP if they were calculated from expected price
                if old_price > 0 and old_price != fill_price:
                    price_diff_pct = (fill_price - old_price) / old_price

                    # Adjust SL proportionally (tick-rounded: round(x, 2) put
                    # SL/TP on the 0.01 grid, e.g. 143.99 — invalid NSE tick)
                    if bracket.sl_trigger_price > 0:
                        bracket.sl_trigger_price = _round_to_tick(
                            bracket.sl_trigger_price * (1 + price_diff_pct)
                        )

                    # Adjust TP proportionally
                    if bracket.tp_trigger_price > 0:
                        bracket.tp_trigger_price = _round_to_tick(
                            bracket.tp_trigger_price * (1 + price_diff_pct)
                        )

                    LOGGER.info(
                        f"📊 Bracket adjusted for fill price: Entry={fill_price:.2f} "
                        f"SL={bracket.sl_trigger_price:.2f} TP={bracket.tp_trigger_price:.2f}"
                    )

            # ✅ Activate only after explicit fill confirmation
            # Post-fill reward:risk check. The real incident activated entry=223.90
            # SL=214.61 TP=227.76 -> risk 9.29, reward 3.86, RR 0.42 (reward < risk),
            # silently. Surface poor geometry loudly so it is auditable. We do NOT
            # reject here (the position is already filled; rejecting would leave it
            # unprotected) — but a sub-floor RR is logged CRITICAL for visibility.
            try:
                rr_risk = abs(fill_price - bracket.sl_trigger_price)
                rr_reward = abs(bracket.tp_trigger_price - fill_price)
                rr = (rr_reward / rr_risk) if rr_risk > 0 else 0.0
                rr_floor = parse_float_env(os.getenv("MIN_BRACKET_RR"), 1.5)
                if rr_risk <= 0 or rr_reward <= 0 or rr < rr_floor:
                    LOGGER.critical(
                        "BRACKET_RR_BELOW_FLOOR symbol=%s entry=%.2f sl=%.2f tp=%.2f risk=%.2f reward=%.2f rr=%.2f floor=%.2f",
                        bracket.symbol,
                        fill_price,
                        bracket.sl_trigger_price,
                        bracket.tp_trigger_price,
                        rr_risk,
                        rr_reward,
                        rr,
                        rr_floor,
                    )
                    self._notify_event(
                        "BRACKET_RR_BELOW_FLOOR",
                        {
                            "symbol": bracket.symbol,
                            "entry": round(fill_price, 2),
                            "sl": round(bracket.sl_trigger_price, 2),
                            "tp": round(bracket.tp_trigger_price, 2),
                            "rr": round(rr, 2),
                            "floor": rr_floor,
                        },
                    )
            except Exception:  # noqa: BLE001 - never let RR check block protection
                pass
            bracket.active = True
            bracket.entry_confirmed = True
            bracket.entry_status = "ACTIVE"
            bracket.exit_state = BracketExitLifecycle.OPEN_ACTIVE.value
            bracket.entry_fill_price = fill_price
            bracket.entry_fill_ts = time.time()
            bracket.exit_pending = False
            bracket.updated_at = time.time()

            # Initialize water marks
            bracket.highest_ltp = fill_price
            bracket.lowest_ltp = fill_price
            bracket.last_ltp = fill_price

            LOGGER.info(
                "BRACKET_ACTIVATED entry_order_id=%s fill_price=%.2f sl=%.2f tp=%.2f",
                bracket.entry_order_id,
                fill_price,
                bracket.sl_trigger_price,
                bracket.tp_trigger_price,
            )
            self._notify_event(
                "BRACKET_ACTIVATED",
                {
                    "symbol": bracket.symbol,
                    "side": bracket.side,
                    "entry": round(fill_price, 2),
                    "sl": round(bracket.sl_trigger_price, 2),
                    "tp": round(bracket.tp_trigger_price, 2),
                },
            )
            self._notify_open_position_priority("open", bracket.symbol)
            self._sync_active_bracket_symbols_to_mdm()

            # Persist state
            try:
                self.save_state()
            except Exception as exc:
                LOGGER.critical(
                    "BRACKET_ACTIVATION_PERSIST_FAILED order_id=%s symbol=%s error=%s",
                    order_id,
                    bracket.symbol,
                    exc,
                    extra={
                        "event": "BRACKET_ACTIVATION_PERSIST_FAILED",
                        "order_id": order_id,
                        "symbol": bracket.symbol,
                        "error_type": type(exc).__name__,
                    },
                )

    # --------------------------------------------------------------------------
    # 2. MARKET DATA INGESTION (NEW)
    # --------------------------------------------------------------------------

    def update_market_stats(
        self, symbol: str, atr: float = 0.0, volume: float = 0.0
    ) -> None:
        """Feed external calculations (ATR) into the manager."""
        if atr > 0:
            # Update Legacy Cache
            self._current_atr[symbol] = atr
            # Feed Safe Provider if available
            if self._atr_provider and hasattr(self._atr_provider, "feed_manual"):
                self._atr_provider.feed_manual(symbol, atr)

    def feed_atr_updates(self, symbol: str, atr: float) -> None:
        """Alias for update_market_stats."""
        self.update_market_stats(symbol, atr=atr)

    def _resolve_bracket_lot_size(self, resolved_lot_size: int | None) -> int | None:
        """Return authoritative lot size for TP splits, or None when unsafe."""
        try:
            explicit = int(resolved_lot_size or 0)
        except (TypeError, ValueError):
            explicit = 0
        if explicit > 0:
            return explicit
        # Compatibility only: older paper/shadow call sites did not provide the
        # resolved instrument lot.  In live mode, absence of the authoritative
        # lot size must not silently create a stale or fractional TP1 split.
        if self._is_live_execution():
            return None
        return max(1, parse_int_env(os.getenv("NIFTY_LOT_SIZE"), 65))

    def _trail_activation_r(self, bracket: BracketState) -> float:
        try:
            return float(
                bracket.trailing_config.get(
                    "breakeven_activation_r",
                    os.getenv("TRAIL_BREAKEVEN_ACTIVATION_R", 0.75),
                )
            )
        except (TypeError, ValueError):
            return 0.75

    def _is_trail_candidate_allowed(
        self, bracket: BracketState, candidate_sl: float, ltp: float
    ) -> bool:
        """Validate one trailing candidate for both controller and fallback paths."""
        if ltp <= 0 or bracket.entry_price <= 0:
            return False
        candidate = _round_to_tick(candidate_sl)
        current_sl = float(bracket.sl_trigger_price or 0.0)
        entry = float(bracket.entry_price)
        initial_sl = float(bracket.initial_sl_trigger_price or current_sl or 0.0)
        initial_risk = abs(entry - initial_sl)
        activation_r = self._trail_activation_r(bracket)

        if bracket.side == "BUY":
            if candidate <= current_sl or candidate >= ltp:
                return False
            if candidate >= entry:
                mfe = max(float(bracket.highest_ltp or entry), ltp) - entry
                return initial_risk > 0 and mfe >= (initial_risk * activation_r)
            return True

        if candidate >= current_sl or candidate <= ltp:
            return False
        if candidate <= entry:
            low_water = float(bracket.lowest_ltp or entry)
            mfe = entry - min(low_water, ltp)
            return initial_risk > 0 and mfe >= (initial_risk * activation_r)
        return True

    def _virtual_modify_sl(self, order_id: str, price: float) -> bool:
        """Callback for AdaptiveController to update Virtual SL."""
        # Find bracket by iterating (Safety lookup)
        target_bracket = None
        with self._lock:
            for b in self._brackets.values():
                if b.virtual_sl_id == order_id:
                    target_bracket = b
                    break

            if target_bracket:
                old_sl = target_bracket.sl_trigger_price
                rounded = _round_to_tick(price)
                ltp = float(target_bracket.last_ltp or 0.0)
                if not self._is_trail_candidate_allowed(target_bracket, rounded, ltp):
                    return False
                target_bracket.sl_trigger_price = rounded
                target_bracket.updated_at = time.time()
                target_bracket.last_trail_price = ltp or None
                target_bracket.trail_revision += 1
                LOGGER.info(
                    "BRACKET_TRAIL_UPDATED symbol=%s old_sl=%s new_sl=%s",
                    target_bracket.symbol,
                    round(old_sl, 2),
                    round(rounded, 2),
                    extra={
                        "event": "BRACKET_TRAIL_UPDATED",
                        "trade_lifecycle_id": target_bracket.trade_lifecycle_id,
                        "entry_order_id": target_bracket.entry_order_id,
                        "symbol": target_bracket.symbol,
                        "old_sl": old_sl,
                        "new_sl": rounded,
                        "ltp": ltp,
                        "trail_revision": target_bracket.trail_revision,
                    },
                )
                # ✅ FIX: Persist trailing update
                self.save_state()
                return True
        return False

    # --------------------------------------------------------------------------
    # 3. EXECUTION LOGIC (The "Sniper")
    # --------------------------------------------------------------------------

    def on_tick(
        self, symbol: str, ltp: float, exchange_ts: float | None = None
    ) -> None:
        """Process one tick through trailing and hard SL/TP evaluation.

        Args:
            symbol: Instrument symbol for the tick.
            ltp: Last traded price.
            exchange_ts: Optional exchange timestamp (epoch seconds). When
                provided, ticks older than a freshly activated bracket's fill
                time are not evaluated against it — a replayed pre-fill signal
                tick must not trigger an immediate false exit right after entry.
        """
        symbol = normalize_symbol(symbol)
        # ═══════════════════════════════════════════════════════════
        # FAST PATH: Early exit checks (no lock needed)
        # ═══════════════════════════════════════════════════════════
        if not self._brackets:
            return

        # Fast symbol lookup (dict access is atomic in Python)
        relevant_ids = self._symbol_map.get(symbol)
        if not relevant_ids:
            return

        if ltp <= 0:
            return

        # ═══════════════════════════════════════════════════════════
        # SNAPSHOT: Minimal lock scope - just get bracket references
        # ═══════════════════════════════════════════════════════════
        candidates = []
        now_ts = time.time()
        with self._lock:
            for eid in relevant_ids:
                b = self._brackets.get(eid)
                # Stale-tick guard: within the arming window after entry fill,
                # a tick whose exchange timestamp predates the fill is the
                # pre-fill/replayed signal tick — never evaluate it against
                # this bracket. Bounded window so clock skew cannot suppress
                # genuine SL/TP evaluation beyond the first seconds.
                if (
                    b is not None
                    and exchange_ts is not None
                    and b.entry_fill_ts is not None
                    and exchange_ts < b.entry_fill_ts
                    and (now_ts - b.entry_fill_ts) <= STALE_TICK_ARM_WINDOW_SEC
                ):
                    self._log_throttled(
                        "debug",
                        f"stale_prefill_tick_skip_{eid}",
                        5.0,
                        "BRACKET_STALE_TICK_SKIP symbol=%s tick_ts=%.3f fill_ts=%.3f",
                        b.symbol,
                        exchange_ts,
                        b.entry_fill_ts,
                    )
                    continue
                # 🟢 FIX: Remove 'b.active' check so we can catch inactive ones
                if (
                    b
                    and b.entry_confirmed
                    and b.remaining_quantity > 0
                    and not b.exit_executed
                    and not b.exit_in_progress
                    and (b.active or b.exit_pending)
                ):
                    candidates.append(b)
                elif b and not b.entry_confirmed:
                    self._log_throttled(
                        "debug",
                        f"pending_entry_skip_{b.entry_order_id}",
                        5.0,
                        "BRACKET_NOT_ACTIVE_SKIP symbol=%s reason=pending_entry",
                        b.symbol,
                    )

        if not candidates:
            return

        # ═══════════════════════════════════════════════════════════
        # TRACK TICK HISTORY (for momentum, outside lock)
        # ═══════════════════════════════════════════════════════════
        if not hasattr(self, "_recent_ticks"):
            self._recent_ticks = {}

        if symbol not in self._recent_ticks:
            from collections import deque

            self._recent_ticks[symbol] = deque(maxlen=20)
        self._recent_ticks[symbol].append(ltp)

        # ═══════════════════════════════════════════════════════════
        # EVALUATE: Check all brackets WITHOUT lock
        # ═══════════════════════════════════════════════════════════
        exits_to_fire = []

        for bracket in candidates:
            tick_id = f"{symbol}:{exchange_ts:.6f}" if exchange_ts is not None else None
            # Keep all bracket field mutations atomic against watchdog reads.
            with self._lock:
                if tick_id is not None and bracket.last_processed_tick_id == tick_id:
                    continue
                if tick_id is not None:
                    bracket.last_processed_tick_id = tick_id
                # Track previous tick for jump/cross stop-loss detection.
                bracket.previous_ltp = float(
                    bracket.last_ltp or bracket.entry_price or ltp
                )
                bracket.last_ltp = ltp
                old_committed_sl = float(bracket.sl_trigger_price or 0.0)

                # Update high/low water marks (atomic operations)
                if bracket.side == "BUY":
                    if ltp > bracket.highest_ltp:
                        bracket.highest_ltp = ltp
                else:
                    if ltp < bracket.lowest_ltp:
                        bracket.lowest_ltp = ltp

            if bracket.exit_pending:
                exits_to_fire.append(
                    (
                        bracket,
                        {
                            "type": "RECONCILE",
                            "qty": bracket.remaining_quantity,
                            "reason": bracket.exit_reason or "EXIT_PENDING",
                            "price": ltp,
                        },
                    )
                )
                continue

            exit_action = self._evaluate_exit_fast(
                bracket, ltp, committed_sl=old_committed_sl
            )
            if exit_action:
                exits_to_fire.append((bracket, exit_action))
                continue

            # Check trailing controller (if attached)
            # Wrapped in try/except so trailing failures cannot block SL/TP eval
            entry_id = bracket.entry_order_id
            trail_updated = False
            try:
                if entry_id in self._trailing_controllers:
                    ctrl = self._trailing_controllers[entry_id]

                    # Inject ATR if available
                    current_atr = self._current_atr.get(symbol)
                    if current_atr and current_atr > 0 and hasattr(ctrl, "update_atr"):
                        ctrl.update_atr(current_atr)

                    # Run controller
                    ctrl.on_tick({"ltp": ltp})
                else:
                    # Use built-in adaptive trailing
                    trail_updated = self._apply_trailing_math(bracket)
            except Exception as _trail_exc:
                LOGGER.debug(
                    "Trailing logic failed for %s: %s (SL/TP eval continues)",
                    bracket.symbol,
                    _trail_exc,
                    extra={"event": "trailing_error", "symbol": bracket.symbol},
                )
                trail_updated = False

            if trail_updated:
                continue

        # ═══════════════════════════════════════════════════════════
        # FIRE EXITS: Batch processing (takes lock once)
        # ═══════════════════════════════════════════════════════════
        if exits_to_fire:
            self._fire_exits_batch(exits_to_fire)

    def process_exit_checks(self, symbol: str, ltp: float) -> None:
        """Route tick to the single on-tick exit authority. Args: symbol, ltp; Returns: None; Raises: None."""
        LOGGER.debug("Entered process_exit_checks")
        try:
            self.on_tick(symbol, ltp)
        except Exception as e:
            LOGGER.error("Failure in process_exit_checks: %s", e)

    def _force_exit(self, bracket: BracketState) -> None:
        """Force bracket exit with confirmation and fallback. Args: bracket; Returns: None; Raises: None."""
        symbol = normalize_symbol(bracket.symbol)
        qty = bracket.remaining_quantity
        if not symbol or qty <= 0:
            return
        self._fire_exits_batch(
            [
                (
                    bracket,
                    {
                        "type": "SL",
                        "price": bracket.last_ltp,
                        "qty": qty,
                        "reason": "FORCED_SL_EXIT",
                    },
                )
            ]
        )

    def eod_flatten_all(
        self,
        reason: str = "EOD_FLATTEN_1524",
        *_: object,
        **__: object,
    ) -> None:
        """Force-exit all active brackets for EOD risk control. Args: reason. Returns: none. Raises: none."""
        exits_to_fire: list[tuple[BracketState, dict[str, object]]] = []
        with self._lock:
            for bracket in self._brackets.values():
                if (
                    bracket.remaining_quantity <= 0
                    or bracket.exit_executed
                    or bracket.exit_in_progress
                ):
                    continue
                exits_to_fire.append(
                    (
                        bracket,
                        {
                            "type": "SL",
                            "price": bracket.last_ltp or bracket.entry_price,
                            "qty": bracket.remaining_quantity,
                            "reason": reason,
                        },
                    )
                )
        if exits_to_fire:
            LOGGER.info(
                "EOD flatten triggered for %s active brackets", len(exits_to_fire)
            )
            self._fire_exits_batch(exits_to_fire)

    def _sl_crossed(self, bracket: BracketState, ltp: float) -> bool:
        """Return True when a tick crosses stop-loss. Args: bracket, ltp; Returns: bool; Raises: none."""
        prev_ltp = float(bracket.previous_ltp or bracket.last_ltp or ltp)
        if bracket.side == "BUY":
            return (
                prev_ltp > bracket.sl_trigger_price and ltp <= bracket.sl_trigger_price
            )
        return prev_ltp < bracket.sl_trigger_price and ltp >= bracket.sl_trigger_price

    def _evaluate_exit_fast(
        self,
        bracket: BracketState,
        ltp: float,
        *,
        committed_sl: float | None = None,
    ) -> dict | None:
        """
        Pure function to evaluate exit conditions.
        Returns exit action dict or None.

        ✅ NO LOCKS, NO SIDE EFFECTS - Just logic
        """
        sl_to_check = float(
            bracket.sl_trigger_price if committed_sl is None else committed_sl
        )
        # Check SL first against the committed stop that existed before any
        # current-tick trailing calculation.
        if sl_to_check > 0:
            prev_ltp = float(bracket.previous_ltp or bracket.entry_price or ltp)
            triggered = False
            if bracket.side == "BUY":
                crossed = prev_ltp > sl_to_check and ltp <= sl_to_check
                breached = ltp <= sl_to_check
                triggered = crossed or breached
            elif bracket.side == "SELL":
                crossed = prev_ltp < sl_to_check and ltp >= sl_to_check
                breached = ltp >= sl_to_check
                triggered = crossed or breached

            if triggered:
                return {
                    "decision": BracketTickDecision.EXIT_STOP.value,
                    "type": "SL",
                    "price": ltp,
                    "qty": bracket.remaining_quantity,
                    "old_sl": sl_to_check,
                    "new_sl": bracket.sl_trigger_price,
                    "reason": f"HARD_SL_BREACH prev={prev_ltp:.2f} curr={ltp:.2f} sl={sl_to_check:.2f}",
                }

        # Check partial targets (TP1, TP2, etc.)
        for target in bracket.tp_levels:
            if target.executed:
                continue

            triggered = False
            if bracket.side == "BUY" and ltp >= target.price:
                triggered = True
            elif bracket.side == "SELL" and ltp <= target.price:
                triggered = True

            if triggered:
                return {
                    "decision": BracketTickDecision.EXIT_TARGET.value,
                    "type": "PARTIAL_TP",
                    "target": target,
                    "price": ltp,
                    "qty": min(target.quantity, bracket.remaining_quantity),
                    "reason": f"{target.name} Hit ({ltp:.2f})",
                }

        # Hard TP breach must take precedence to guarantee immediate exit.
        if bracket.tp_trigger_price > 0:
            triggered = False
            if bracket.side == "BUY" and ltp >= bracket.tp_trigger_price:
                triggered = True
            elif bracket.side == "SELL" and ltp <= bracket.tp_trigger_price:
                triggered = True

            if triggered:
                return {
                    "decision": BracketTickDecision.EXIT_TARGET.value,
                    "type": "FINAL_TP",
                    "price": ltp,
                    "qty": bracket.remaining_quantity,
                    "reason": "HARD_TP_BREACH",
                }

        return None

    def _fire_exits_batch(self, exits: list) -> None:
        """Latch exit triggers and submit at most one controlled exit order per bracket."""
        now = time.time()
        approved: list[tuple[BracketState, dict[str, Any]]] = []

        with self._lock:
            for bracket, action in exits:
                if not self._exit_can_submit_or_reconcile_locked(bracket, now):
                    LOGGER.info(
                        "BRACKET_EXIT_SKIPPED_DUPLICATE bracket_id=%s symbol=%s exit_state=%s pending_exit_order_id=%s",
                        bracket.bracket_id,
                        bracket.symbol,
                        bracket.exit_state,
                        bracket.pending_exit_order_id,
                        extra={
                            "event": "BRACKET_EXIT_SKIPPED_DUPLICATE",
                            "trade_lifecycle_id": bracket.trade_lifecycle_id,
                            "entry_order_id": bracket.entry_order_id,
                            "symbol": bracket.symbol,
                            "exit_state": bracket.exit_state,
                            "pending_exit_order_id": bracket.pending_exit_order_id,
                        },
                    )
                    continue

                if not bracket.exit_pending:
                    reason = str(action.get("reason", "EXIT"))
                    bracket.exit_pending = True
                    bracket.exit_reason = reason
                    bracket.exit_triggered_at = now
                    bracket.exit_state = BracketExitLifecycle.EXIT_TRIGGERED.value
                    bracket.entry_status = BracketExitLifecycle.EXIT_TRIGGERED.value
                    bracket.updated_at = now
                    LOGGER.info(
                        "EXIT_TRIGGERED bracket_id=%s symbol=%s reason=%s qty=%s",
                        bracket.bracket_id,
                        bracket.symbol,
                        reason,
                        int(action.get("qty") or bracket.remaining_quantity),
                        extra={
                            "event": "EXIT_TRIGGERED",
                            "bypass_filters": True,
                            "bracket_id": bracket.bracket_id,
                            "symbol": bracket.symbol,
                            "reason": reason,
                            "quantity": int(
                                action.get("qty") or bracket.remaining_quantity
                            ),
                        },
                    )
                    self._log_bracket_event(
                        "EXIT_TRIGGERED",
                        bracket,
                        meta={
                            "reason": reason,
                            "qty": int(action.get("qty") or bracket.remaining_quantity),
                        },
                    )
                    event = (
                        "BRACKET_EXIT_TARGET"
                        if str(action.get("decision"))
                        == BracketTickDecision.EXIT_TARGET.value
                        else (
                            "BRACKET_EXIT_RISK"
                            if str(action.get("decision"))
                            == BracketTickDecision.EXIT_RISK.value
                            else "BRACKET_EXIT_STOP"
                        )
                    )
                    LOGGER.warning(
                        "%s trade_lifecycle_id=%s entry_order_id=%s symbol=%s quantity=%s entry_price=%s ltp=%s bid=%s ask=%s old_sl=%s new_sl=%s tp=%s exit_reason=%s exit_state=%s tick_timestamp=%s",
                        event,
                        bracket.trade_lifecycle_id,
                        bracket.entry_order_id,
                        bracket.symbol,
                        int(action.get("qty") or bracket.remaining_quantity),
                        bracket.entry_price,
                        action.get("price"),
                        action.get("bid"),
                        action.get("ask"),
                        action.get("old_sl", bracket.sl_trigger_price),
                        action.get("new_sl", bracket.sl_trigger_price),
                        bracket.tp_trigger_price,
                        reason,
                        (
                            "TRIGGERED"
                            if bracket.exit_state
                            == BracketExitLifecycle.EXIT_TRIGGERED.value
                            else bracket.exit_state
                        ),
                        action.get("tick_timestamp"),
                        extra={
                            "event": event,
                            "trade_lifecycle_id": bracket.trade_lifecycle_id,
                            "entry_order_id": bracket.entry_order_id,
                            "symbol": bracket.symbol,
                            "quantity": int(
                                action.get("qty") or bracket.remaining_quantity
                            ),
                            "entry_price": bracket.entry_price,
                            "ltp": action.get("price"),
                            "bid": action.get("bid"),
                            "ask": action.get("ask"),
                            "old_sl": action.get("old_sl", bracket.sl_trigger_price),
                            "new_sl": action.get("new_sl", bracket.sl_trigger_price),
                            "tp": bracket.tp_trigger_price,
                            "exit_reason": reason,
                            "exit_state": (
                                "TRIGGERED"
                                if bracket.exit_state
                                == BracketExitLifecycle.EXIT_TRIGGERED.value
                                else bracket.exit_state
                            ),
                            "tick_timestamp": action.get("tick_timestamp"),
                        },
                    )

                approved.append((bracket, action))

        for bracket, action in approved:
            self._process_exit_state(bracket, action, now=time.time())

    def _exit_can_submit_or_reconcile_locked(
        self, bracket: BracketState, now: float
    ) -> bool:
        if bracket.remaining_quantity <= 0:
            return False
        if bracket.exit_state in {
            BracketExitLifecycle.CLOSED.value,
            BracketExitLifecycle.EXIT_FILLED.value,
            BracketExitLifecycle.EXIT_RECONCILED_FLAT.value,
        }:
            return False
        if not bracket.exit_pending and (
            not bracket.active
            or not bracket.entry_confirmed
            or bracket.entry_status
            not in {"ACTIVE", BracketExitLifecycle.OPEN_ACTIVE.value}
            or bracket.exit_executed
        ):
            self._log_throttled(
                "debug",
                f"bracket_exit_skipped_{bracket.entry_order_id}",
                60.0,
                "BRACKET_EXIT_SKIPPED symbol=%s reason=inactive_or_unconfirmed",
                bracket.symbol,
            )
            return False
        if bracket.exit_in_progress:
            self._log_exit_pending_summary_locked(bracket, now)
            return False
        if not bracket.exit_pending and (
            bracket.exit_state != BracketExitLifecycle.OPEN_ACTIVE.value
            or bracket.pending_exit_order_id is not None
        ):
            return False
        return True

    def _process_exit_state(
        self, bracket: BracketState, action: Mapping[str, Any], *, now: float
    ) -> None:
        symbol = normalize_symbol(bracket.symbol)
        reason = str(action.get("reason") or bracket.exit_reason or "EXIT")
        qty = max(
            0,
            min(
                int(action.get("qty") or bracket.remaining_quantity),
                int(bracket.remaining_quantity),
            ),
        )
        if not symbol or qty <= 0:
            return

        if self._reconcile_exit_state(bracket, requested_by="pre_submit"):
            return

        with self._lock:
            if bracket.exit_in_progress:
                self._log_exit_pending_summary_locked(bracket, now)
                return
            if bracket.exit_order_id or bracket.pending_exit_order_id:
                bracket.exit_state = BracketExitLifecycle.EXIT_ORDER_SUBMITTED.value
                self._log_exit_pending_summary_locked(bracket, now)
                return
            if (
                bracket.exit_state == BracketExitLifecycle.EXIT_FAILED_ESCALATED.value
                and not self._exit_continue_retry_after_escalation
            ):
                self._log_exit_pending_summary_locked(bracket, now)
                return
            if bracket.next_exit_attempt_at and now < float(
                bracket.next_exit_attempt_at
            ):
                self._log_exit_pending_summary_locked(bracket, now)
                return
            if bracket.exit_attempt_count >= self._exit_max_retry_attempts:
                self._escalate_exit_locked(bracket, "max_attempts_exceeded")
                return

            bracket.exit_in_progress = True
            bracket.exit_state = BracketExitLifecycle.EXIT_ORDER_PENDING.value
            bracket.entry_status = BracketExitLifecycle.EXIT_ORDER_PENDING.value
            bracket.exit_attempt_count += 1
            bracket.last_exit_attempt_at = now
            attempt = bracket.exit_attempt_count
            self._exit_cooldowns[bracket.entry_order_id] = now

        LOGGER.warning(
            "EXIT_ORDER_SUBMIT_ATTEMPT bracket_id=%s symbol=%s attempt=%s side=%s qty=%s reason=%s",
            bracket.bracket_id,
            symbol,
            attempt,
            "SELL" if bracket.side == "BUY" else "BUY",
            qty,
            reason,
        )
        submit = self.submit_exit_order(
            symbol=symbol,
            qty=qty,
            reason=reason,
            bracket_id=bracket.bracket_id,
            preferred_order_type="LIMIT",
        )

        with self._lock:
            bracket.exit_in_progress = False
            bracket.updated_at = time.time()
            if submit.accepted and submit.order_id:
                bracket.exit_order_id = str(submit.order_id)
                bracket.pending_exit_order_id = str(submit.order_id)
                bracket.exit_state = BracketExitLifecycle.EXIT_ORDER_SUBMITTED.value
                bracket.entry_status = BracketExitLifecycle.EXIT_ORDER_SUBMITTED.value
                bracket.last_exit_error = None
                bracket.next_exit_attempt_at = None
                LOGGER.info(
                    "EXIT_ORDER_SUBMITTED bracket_id=%s order_id=%s symbol=%s attempt=%s",
                    bracket.bracket_id,
                    submit.order_id,
                    symbol,
                    attempt,
                )
            else:
                bracket.last_exit_error = submit.error_message or submit.status
                bracket.pending_exit_order_id = None
                bracket.exit_order_id = None
                raw_decision = (
                    getattr(self.order_manager, "_last_order_decision", {}) or {}
                )
                decision = (
                    dict(raw_decision) if isinstance(raw_decision, Mapping) else {}
                )
                LOGGER.error(
                    "BRACKET_EXIT_ORDER_FAILED symbol=%s bracket_id=%s remaining_qty=%s attempt=%s error_type=%s error_message=%s retryable=%s order_manager_reason=%s broker_attempted=%s kill_switch_active=%s broker_rejection=%s",
                    symbol,
                    bracket.bracket_id,
                    bracket.remaining_quantity,
                    attempt,
                    submit.error_type,
                    submit.error_message,
                    submit.retryable,
                    decision.get("block_reason"),
                    decision.get("broker_attempted"),
                    bool(getattr(self.order_manager, "_kill_switch_engaged_at", None)),
                    submit.broker_payload,
                    extra={
                        "event": "BRACKET_EXIT_ORDER_FAILED",
                        "bypass_filters": True,
                        "symbol": symbol,
                        "bracket_id": bracket.bracket_id,
                        "remaining_qty": bracket.remaining_quantity,
                        "attempt": attempt,
                        "error_type": submit.error_type,
                        "error_message": submit.error_message,
                        "retryable": submit.retryable,
                        "order_manager_reason": decision.get("block_reason"),
                        "broker_attempted": decision.get("broker_attempted"),
                        "kill_switch_active": bool(
                            getattr(self.order_manager, "_kill_switch_engaged_at", None)
                        ),
                        "broker_rejection": submit.broker_payload,
                    },
                )
                if (
                    submit.retryable
                    and self._exit_retry_enabled
                    and attempt < self._exit_max_retry_attempts
                ):
                    bracket.exit_state = (
                        BracketExitLifecycle.EXIT_REJECTED_RETRYABLE.value
                    )
                    bracket.entry_status = (
                        BracketExitLifecycle.EXIT_REJECTED_RETRYABLE.value
                    )
                    delay = self._retry_delay_for_attempt(attempt)
                    bracket.next_exit_attempt_at = time.time() + delay
                    LOGGER.warning(
                        "EXIT_ORDER_REJECTED bracket_id=%s attempt=%s retryable=True error_type=%s error_message=%s",
                        bracket.bracket_id,
                        attempt,
                        submit.error_type,
                        submit.error_message,
                    )
                    LOGGER.warning(
                        "EXIT_RETRY_SCHEDULED bracket_id=%s next_attempt_in_s=%s attempt=%s",
                        bracket.bracket_id,
                        delay,
                        attempt + 1,
                    )
                else:
                    bracket.exit_state = BracketExitLifecycle.EXIT_REJECTED_FATAL.value
                    bracket.entry_status = (
                        BracketExitLifecycle.EXIT_REJECTED_FATAL.value
                    )
                    LOGGER.critical(
                        "EXIT_ORDER_REJECTED bracket_id=%s attempt=%s retryable=False error_type=%s error_message=%s",
                        bracket.bracket_id,
                        attempt,
                        submit.error_type,
                        submit.error_message,
                    )
                    self._escalate_exit_locked(bracket, "fatal_or_retry_exhausted")
        self._reconcile_exit_state(bracket, requested_by="post_submit")

    def _retry_delay_for_attempt(self, attempt: int) -> float:
        idx = max(0, attempt - 1)
        if idx < len(self._exit_retry_backoffs):
            return float(self._exit_retry_backoffs[idx])
        return float(self._exit_retry_backoffs[-1])

    def _log_exit_pending_summary_locked(
        self, bracket: BracketState, now: float
    ) -> None:
        last = float(bracket.last_exit_summary_at or 0.0)
        if now - last < 5.0:
            return
        bracket.last_exit_summary_at = now
        age = now - float(bracket.exit_triggered_at or now)
        display_state = (
            "TRIGGERED"
            if bracket.exit_state == BracketExitLifecycle.EXIT_TRIGGERED.value
            else bracket.exit_state
        )
        LOGGER.warning(
            "EXIT_PENDING_SUMMARY bracket_id=%s symbol=%s age_s=%.1f attempts=%s remaining_qty=%s last_error=%s exit_order_id=%s state=%s",
            bracket.bracket_id,
            bracket.symbol,
            age,
            bracket.exit_attempt_count,
            bracket.remaining_quantity,
            bracket.last_exit_error,
            bracket.exit_order_id or bracket.pending_exit_order_id,
            display_state,
        )

    def _escalate_exit_locked(self, bracket: BracketState, reason: str) -> None:
        if bracket.exit_state == BracketExitLifecycle.EXIT_FAILED_ESCALATED.value:
            return
        bracket.exit_pending = True
        bracket.exit_state = BracketExitLifecycle.EXIT_FAILED_ESCALATED.value
        bracket.entry_status = BracketExitLifecycle.EXIT_FAILED_ESCALATED.value
        bracket.escalated_at = time.time()
        LOGGER.critical(
            "EXIT_ESCALATED bracket_id=%s symbol=%s remaining_qty=%s attempts=%s last_error=%s reason=%s",
            bracket.bracket_id,
            bracket.symbol,
            bracket.remaining_quantity,
            bracket.exit_attempt_count,
            bracket.last_exit_error,
            reason,
        )
        self._notify_event(
            "EXIT_ESCALATED",
            {
                "symbol": bracket.symbol,
                "bracket_id": bracket.bracket_id,
                "remaining_qty": bracket.remaining_quantity,
                "attempts": bracket.exit_attempt_count,
                "last_error": bracket.last_exit_error,
                "message": "⚠️ Exit unresolved. Forcing MARKET exit.",
            },
        )
        # Escalation must actually FLATTEN the position, not just freeze. A stuck
        # LIMIT exit (OPEN PENDING, never filling) previously left the position
        # exposed for minutes while this method only logged. Cancel the dead
        # pending order and fire ONE forced MARKET exit. Done outside the lock to
        # avoid re-entrancy (place_order / cancel acquire their own locks).
        if not self._exit_force_market_on_escalation:
            return
        if getattr(bracket, "_market_escalation_fired", False):
            return
        bracket._market_escalation_fired = True
        stuck_order_id = bracket.exit_order_id or bracket.pending_exit_order_id
        symbol = normalize_symbol(bracket.symbol)
        qty = int(bracket.remaining_quantity or 0)
        side = "SELL" if bracket.side == "BUY" else "BUY"

        def _force_market_flatten() -> None:
            # Cancel the unfilled pending limit so it can't fill alongside the market order.
            if stuck_order_id:
                try:
                    self.order_manager.cancel_order(str(stuck_order_id))
                    LOGGER.warning(
                        "EXIT_ESCALATION_CANCELLED_STUCK_ORDER bracket_id=%s order_id=%s",
                        bracket.bracket_id,
                        stuck_order_id,
                    )
                except (
                    Exception
                ) as exc:  # noqa: BLE001 - cancel best-effort; still try market
                    LOGGER.warning(
                        "EXIT_ESCALATION_CANCEL_FAILED bracket_id=%s order_id=%s error=%s",
                        bracket.bracket_id,
                        stuck_order_id,
                        exc,
                    )
            with self._lock:
                bracket.exit_order_id = None
                bracket.pending_exit_order_id = None
            if not symbol or qty <= 0:
                return
            try:
                order_id = self.order_manager.place_order(
                    symbol=symbol,
                    side=side,
                    quantity=qty,
                    order_type="MARKET",
                    tag=f"EXIT_MKT_{bracket.bracket_id[:8]}",
                    check_risk=False,
                    product="MIS",
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.critical(
                    "EXIT_ESCALATION_MARKET_EXIT_FAILED bracket_id=%s symbol=%s error=%s",
                    bracket.bracket_id,
                    symbol,
                    exc,
                )
                return
            if order_id:
                with self._lock:
                    bracket.exit_order_id = str(order_id)
                    bracket.pending_exit_order_id = str(order_id)
                LOGGER.critical(
                    "EXIT_ESCALATION_MARKET_EXIT_SENT bracket_id=%s symbol=%s order_id=%s qty=%s",
                    bracket.bracket_id,
                    symbol,
                    order_id,
                    qty,
                )
            else:
                LOGGER.critical(
                    "EXIT_ESCALATION_MARKET_EXIT_NO_ORDER_ID bracket_id=%s symbol=%s",
                    bracket.bracket_id,
                    symbol,
                )

        try:
            _force_market_flatten()
        except Exception as exc:  # noqa: BLE001 - never let escalation raise
            LOGGER.error(
                "EXIT_ESCALATION_DISPATCH_FAILED bracket_id=%s error=%s",
                bracket.bracket_id,
                exc,
            )

    def on_tick_event(self, tick: dict[str, Any]) -> None:
        """Args: tick; Returns: none; Raises: none."""
        try:
            symbol = str(tick.get("symbol") or "")
            ltp = tick.get("last_price") or tick.get("ltp")
            if not symbol or not isinstance(ltp, (int, float)):
                return
            exchange_ts = tick_exchange_epoch(tick)
            self.on_tick(symbol, float(ltp), exchange_ts)
        except Exception as e:
            LOGGER.error("Failure in BracketManager.on_tick_event: %s", e)

    @property
    def active_brackets(self) -> Dict[str, BracketState]:
        """Return symbol-indexed active bracket map for recovery hooks."""
        with self._lock:
            return {b.symbol: b for b in self._brackets.values() if b.active}

    def _process_trailing_logic(self, bracket: BracketState, ltp: float) -> None:
        """Updates High/Low marks and adjusts SL if Trailing is enabled."""

        # A. Update Water Marks
        if bracket.side == "BUY":
            if ltp > bracket.highest_ltp:
                bracket.highest_ltp = ltp
        else:  # SELL
            if ltp < bracket.lowest_ltp:
                bracket.lowest_ltp = ltp

        # B. Delegate to World-Class Controller (If attached)
        if bracket.entry_order_id in self._trailing_controllers:
            ctrl = self._trailing_controllers[bracket.entry_order_id]
            # ✅ FIX: Inject dynamic ATR from feed
            current_atr = self._current_atr.get(bracket.symbol)
            if current_atr and hasattr(ctrl, "update_atr"):
                ctrl.update_atr(current_atr)
            # The controller calculates using ATR and calls _virtual_modify_sl if needed
            # We updated bracket.last_ltp in on_tick, so controller reads fresh data
            ctrl.on_tick(None)
            return

        # C. Fallback Legacy Logic
        self._apply_trailing_math(bracket)

    def _apply_trailing_math(self, bracket: BracketState) -> bool:
        """Apply adaptive trailing rules for the bracket.

        SYNC CONTRACT: this is the FALLBACK trailing path (runs only when the
        AdaptiveTrailingController could not attach). It must uphold exactly
        the invariants of the canonical authority
        HardenedBracketManager._virtual_modify_sl — under-lock compare-and-set,
        monotonic per side (BUY only raises, SELL only lowers), tick rounding,
        and BRACKET_TRAIL_UPDATE notify. Any change to either implementation
        must be mirrored in the other. Verified end-to-end by
        test_bracket_lifecycle_trailing_and_exits_on_live_class.

        Args:
            bracket: Active bracket state with the latest LTP snapshot.

        Returns:
            True when one committed trailing state mutation occurred.

        Raises:
            None.
        """
        if not bracket.trailing_enabled:
            return False
        if (
            not bracket.active
            or not bracket.entry_confirmed
            or bracket.entry_status != "ACTIVE"
        ):
            return False

        ltp = bracket.last_ltp
        if not ltp or ltp <= 0:
            return False

        entry = bracket.entry_price
        if entry <= 0:
            return False

        current_sl = bracket.sl_trigger_price

        # Get ATR for trailing calculations.
        atr = self._get_current_atr(bracket.symbol)
        if atr <= 0:
            if not bracket._atr_warning_logged:
                LOGGER.warning(
                    "ATR unavailable — trailing fallback for %s",
                    bracket.symbol,
                    extra={"event": "trailing_atr_missing", "symbol": bracket.symbol},
                )
                bracket._atr_warning_logged = True
            # CRITICAL FIX: Use 2% of entry_price as ATR fallback.
            # NEVER use stop_loss_price (e.g. ₹120) as ATR — that causes
            # Tier 3/4 to compute: high_water - (120 * 1.5) = negative SL,
            # which disables the stop-loss entirely → unlimited loss exposure.
            atr = bracket.entry_price * 0.02 if bracket.entry_price > 0 else 1.0

        # Calculate profit metrics
        if bracket.side == "BUY":
            profit_pct = ((ltp - entry) / entry) * 100
            high_water = bracket.highest_ltp
            profit_points = ltp - entry
        else:
            profit_pct = ((entry - ltp) / entry) * 100
            high_water = bracket.lowest_ltp
            profit_points = entry - ltp

        # Calculate new SL based on tiered system
        new_sl = self._calculate_tiered_trailing_sl(
            bracket=bracket,
            ltp=ltp,
            profit_pct=profit_pct,
            high_water=high_water,
            atr=atr,
        )

        if new_sl is None:
            return False

        # Apply the new SL (only if it improves protection).
        # FIX: re-read current_sl INSIDE the lock — the watchdog thread can modify
        # sl_trigger_price between our stale read above and this write. Using the
        # stale value could write a LOWER SL than what the watchdog already set,
        # making protection WORSE on a trailing update.
        with self._lock:
            if bracket.side == "BUY":
                current_sl = bracket.sl_trigger_price  # authoritative read under lock
                rounded_sl = _round_to_tick(new_sl)
                if self._is_trail_candidate_allowed(bracket, rounded_sl, ltp):
                    old_sl = bracket.sl_trigger_price
                    bracket.sl_trigger_price = rounded_sl
                    bracket.updated_at = time.time()
                    bracket.last_trail_price = ltp
                    bracket.trail_revision += 1

                    LOGGER.debug(
                        f"📈 TRAIL UPDATE {bracket.symbol}: "
                        f"SL {old_sl:.2f} → {new_sl:.2f} | "
                        f"Profit: {profit_pct:.1f}% | LTP: {ltp:.2f}"
                    )
                    LOGGER.debug(
                        "TRAILING_SL_UPDATED symbol=%s old_sl=%s new_sl=%s reason=protective_move",
                        bracket.symbol,
                        round(old_sl, 2),
                        round(bracket.sl_trigger_price, 2),
                    )
                    if self._should_notify_trail(
                        bracket.entry_order_id, bracket.sl_trigger_price, old_sl
                    ):
                        self._notify_event(
                            "BRACKET_TRAIL_UPDATED",
                            {
                                "symbol": bracket.symbol,
                                "side": bracket.side,
                                "sl": round(bracket.sl_trigger_price, 2),
                                "ltp": round(ltp, 2),
                                "profit_pct": round(profit_pct, 2),
                            },
                        )
                    return True
            else:  # SELL
                current_sl = bracket.sl_trigger_price  # authoritative read under lock
                rounded_sl = _round_to_tick(new_sl)
                if self._is_trail_candidate_allowed(bracket, rounded_sl, ltp):
                    old_sl = bracket.sl_trigger_price
                    bracket.sl_trigger_price = rounded_sl
                    bracket.updated_at = time.time()
                    bracket.last_trail_price = ltp
                    bracket.trail_revision += 1

                    LOGGER.debug(
                        f"📉 TRAIL UPDATE {bracket.symbol}: "
                        f"SL {old_sl:.2f} → {new_sl:.2f} | "
                        f"Profit: {profit_pct:.1f}% | LTP: {ltp:.2f}"
                    )
                    LOGGER.debug(
                        "TRAILING_SL_UPDATED symbol=%s old_sl=%s new_sl=%s reason=protective_move",
                        bracket.symbol,
                        round(old_sl, 2),
                        round(bracket.sl_trigger_price, 2),
                    )
                    if self._should_notify_trail(
                        bracket.entry_order_id, bracket.sl_trigger_price, old_sl
                    ):
                        self._notify_event(
                            "BRACKET_TRAIL_UPDATED",
                            {
                                "symbol": bracket.symbol,
                                "side": bracket.side,
                                "sl": round(bracket.sl_trigger_price, 2),
                                "ltp": round(ltp, 2),
                                "profit_pct": round(profit_pct, 2),
                            },
                        )
                    return True
        return False

    def _calculate_tiered_trailing_sl(
        self,
        bracket: BracketState,
        ltp: float,
        profit_pct: float,
        high_water: float,
        atr: float,
    ) -> float | None:
        """
        Calculate trailing SL using tiered protection system.

        TIERS:
        - Tier 0 (< 1%): No trailing, use original SL
        - Tier 1 (1-2%): Lock breakeven
        - Tier 2 (2-4%): Protect 40% of profit
        - Tier 3 (4-6%): Protect 50% of profit + ATR trail
        - Tier 4 (> 6%): Protect 60% of profit + tight ATR trail
        """
        entry = bracket.entry_price
        current_sl = bracket.sl_trigger_price
        initial_sl = float(bracket.initial_sl_trigger_price or current_sl or 0.0)
        initial_risk = abs(entry - initial_sl)
        if bracket.side == "BUY":
            mfe = float(bracket.highest_ltp or ltp or entry) - entry
        else:
            mfe = entry - float(bracket.lowest_ltp or ltp or entry)
        activation_r = float(
            bracket.trailing_config.get(
                "breakeven_activation_r",
                os.getenv("TRAIL_BREAKEVEN_ACTIVATION_R", 0.75),
            )
        )

        # Use instance-cached thresholds (set at __init__) — not os.getenv on every tick.
        tier1_threshold = self._trail_tier1_pct
        tier2_threshold = self._trail_tier2_pct
        tier3_threshold = self._trail_tier3_pct
        tier4_threshold = self._trail_tier4_pct

        # ═══════════════════════════════════════════════════════════
        # TIER 0: NO PROFIT (< 1%) - Use original SL
        # ═══════════════════════════════════════════════════════════
        if profit_pct < tier1_threshold:
            return None  # No change

        # ═══════════════════════════════════════════════════════════
        # TIER 1: SMALL PROFIT (1-2%) - BREAKEVEN LOCK
        # ═══════════════════════════════════════════════════════════
        if tier1_threshold <= profit_pct < tier2_threshold:
            if initial_risk <= 0 or mfe < (initial_risk * activation_r):
                return None
            # Lock at breakeven (entry price)
            if bracket.side == "BUY":
                if entry > current_sl:
                    LOGGER.debug(f"🔒 Tier 1: Breakeven lock at {entry:.2f}")
                    return entry
            else:
                if entry < current_sl:
                    LOGGER.debug(f"🔒 Tier 1: Breakeven lock at {entry:.2f}")
                    return entry
            return None

        # ═══════════════════════════════════════════════════════════
        # TIER 2: MODERATE PROFIT (2-4%) - PROTECT 40%
        # ═══════════════════════════════════════════════════════════
        if tier2_threshold <= profit_pct < tier3_threshold:
            protection_pct = 0.40

            if bracket.side == "BUY":
                profit_amount = high_water - entry
                protected_sl = entry + (profit_amount * protection_pct)
                LOGGER.debug(f"🛡️ Tier 2: 40% protection at {protected_sl:.2f}")
                return protected_sl
            else:
                profit_amount = entry - high_water
                protected_sl = entry - (profit_amount * protection_pct)
                LOGGER.debug(f"🛡️ Tier 2: 40% protection at {protected_sl:.2f}")
                return protected_sl

        # ═══════════════════════════════════════════════════════════
        # TIER 3: GOOD PROFIT (4-6%) - PROTECT 50% + ATR TRAIL
        # ═══════════════════════════════════════════════════════════
        if tier3_threshold <= profit_pct < tier4_threshold:
            protection_pct = 0.50

            # Calculate momentum-adjusted ATR multiplier
            momentum = self._calculate_momentum(bracket.symbol)
            atr_mult = self._get_momentum_adjusted_atr_mult(momentum, base_mult=1.5)

            if bracket.side == "BUY":
                # Minimum: 50% profit protection
                profit_amount = high_water - entry
                min_sl = entry + (profit_amount * protection_pct)

                # ATR trail from high water
                if atr > 0:
                    atr_sl = high_water - (atr * atr_mult)
                    # Use whichever is HIGHER (more protective)
                    protected_sl = max(min_sl, atr_sl)
                else:
                    protected_sl = min_sl

                LOGGER.debug(f"📊 Tier 3: 50% + ATR trail at {protected_sl:.2f}")
                return protected_sl
            else:
                profit_amount = entry - high_water
                min_sl = entry - (profit_amount * protection_pct)

                if atr > 0:
                    atr_sl = high_water + (atr * atr_mult)
                    protected_sl = min(min_sl, atr_sl)
                else:
                    protected_sl = min_sl

                LOGGER.debug(f"📊 Tier 3: 50% + ATR trail at {protected_sl:.2f}")
                return protected_sl

        # ═══════════════════════════════════════════════════════════
        # TIER 4: EXCELLENT PROFIT (> 6%) - PROTECT 60% + TIGHT TRAIL
        # ═══════════════════════════════════════════════════════════
        if profit_pct >= tier4_threshold:
            protection_pct = 0.60

            # Tighter ATR multiplier for big winners
            momentum = self._calculate_momentum(bracket.symbol)
            atr_mult = self._get_momentum_adjusted_atr_mult(momentum, base_mult=1.0)

            if bracket.side == "BUY":
                profit_amount = high_water - entry
                min_sl = entry + (profit_amount * protection_pct)

                if atr > 0:
                    atr_sl = high_water - (atr * atr_mult)
                    protected_sl = max(min_sl, atr_sl)
                else:
                    protected_sl = min_sl

                LOGGER.info(f"🏆 Tier 4: 60% + tight trail at {protected_sl:.2f}")
                return protected_sl
            else:
                profit_amount = entry - high_water
                min_sl = entry - (profit_amount * protection_pct)

                if atr > 0:
                    atr_sl = high_water + (atr * atr_mult)
                    protected_sl = min(min_sl, atr_sl)
                else:
                    protected_sl = min_sl

                LOGGER.info(f"🏆 Tier 4: 60% + tight trail at {protected_sl:.2f}")
                return protected_sl

        return None

    def _get_current_atr(self, symbol: str) -> float:
        """Return current ATR, falling back to the validated local cache."""
        if self._atr_provider:
            try:
                atr_value = None
                if hasattr(self._atr_provider, "get_current_atr"):
                    atr_value = self._atr_provider.get_current_atr(symbol)
                elif hasattr(self._atr_provider, "get_atr"):
                    snapshot = self._atr_provider.get_atr(symbol)
                    atr_value = getattr(snapshot, "value", snapshot)
                if atr_value is not None and float(atr_value) > 0:
                    return float(atr_value)
            except Exception as exc:  # noqa: BLE001 - hot path must retain SL/TP checks
                self._log_throttled(
                    "warning",
                    f"atr_provider_failure_{symbol}",
                    30.0,
                    "ATR_PROVIDER_FAILED_USING_CACHE symbol=%s error_type=%s error=%s",
                    symbol,
                    type(exc).__name__,
                    exc,
                )

        atr_raw = self._current_atr.get(symbol, 0.0)
        if isinstance(atr_raw, Mapping):
            atr_raw = atr_raw.get("value", 0.0)
        try:
            atr_value = float(atr_raw or 0.0)
        except (TypeError, ValueError):
            return 0.0
        return atr_value if math.isfinite(atr_value) and atr_value > 0 else 0.0

    def _calculate_momentum(self, symbol: str) -> float:
        """
        Calculate price momentum from recent ticks.
        Returns: Momentum as percentage change (positive = up, negative = down)
        """
        # Use last_ltp changes if available
        # This is a simplified momentum calculation
        # For production, you might want to track a deque of recent prices

        # Check if we have historical tick data
        if hasattr(self, "_recent_ticks") and symbol in self._recent_ticks:
            ticks = self._recent_ticks[symbol]
            if len(ticks) >= 5:
                first = ticks[-5]
                last = ticks[-1]
                if first > 0:
                    return ((last - first) / first) * 100

        # Fallback: use bracket's high/low water marks
        bracket = None
        with self._lock:
            for b in self._brackets.values():
                if b.symbol == symbol and b.active:
                    bracket = b
                    break

        if bracket:
            ltp = bracket.last_ltp
            entry = bracket.entry_price
            if entry > 0 and ltp > 0:
                # Simple momentum: current move from entry
                return ((ltp - entry) / entry) * 100

        return 0.0

    def _get_momentum_adjusted_atr_mult(
        self, momentum: float, base_mult: float
    ) -> float:
        """
        Adjust ATR multiplier based on momentum.

        Strong momentum = wider trail (avoid whipsaws in fast breakouts)
        Weak momentum = tighter trail (lock profits in chop)
        """
        abs_momentum = abs(momentum)

        if abs_momentum > 1.0:
            # Very strong momentum: give it room to breathe
            return base_mult * 1.3
        elif abs_momentum > 0.5:
            # Strong momentum: slightly wider
            return base_mult * 1.1
        elif abs_momentum > 0.2:
            # Moderate momentum: standard
            return base_mult
        else:
            # Weak/choppy: tighten the trail to lock in whatever profit exists
            return base_mult * 0.8

    def _check_stop_loss(self, bracket: BracketState, ltp: float) -> bool:
        """Returns True if SL hit and exit fired (Safe against 0.0)."""
        triggered = False
        reason = ""

        # 🛑 SAFETY: Ignore if SL is 0.0 (Disabled/Unset)
        if bracket.sl_trigger_price <= 0:
            return False

        if bracket.side == "BUY":
            if ltp <= bracket.sl_trigger_price:
                triggered = True
                reason = f"SL Hit ({ltp} <= {bracket.sl_trigger_price:.2f})"
        else:  # SELL
            if ltp >= bracket.sl_trigger_price:
                triggered = True
                reason = f"SL Hit ({ltp} >= {bracket.sl_trigger_price:.2f})"

        if triggered:
            LOGGER.warning(f"🛑 STOP LOSS TRIGGERED for {bracket.symbol} | {reason}")
            # Exit full remaining quantity
            result = self._execute_exit(
                bracket, bracket.remaining_quantity, reason, is_partial=False
            )
            return result.confirmed
        return False

    def _check_partial_targets(self, bracket: BracketState, ltp: float) -> None:
        """Checks TP1/Intermediate levels."""
        for target in bracket.tp_levels:
            if target.executed:
                continue

            triggered = False
            if bracket.side == "BUY":
                if ltp >= target.price:
                    triggered = True
            else:  # SELL
                if ltp <= target.price:
                    triggered = True

            if triggered:
                reason = f"{target.name} Hit ({ltp})"
                qty_to_close = min(target.quantity, bracket.remaining_quantity)

                # Execute Partial
                success = self._execute_exit(
                    bracket, qty_to_close, reason, is_partial=True
                ).confirmed

                if success:
                    target.executed = True
                    # AUTO-ADJUST: Move SL to Breakeven after TP1
                    if target.name == "TP1":
                        self._move_sl_to_breakeven(bracket)

    def _check_final_target(self, bracket: BracketState, ltp: float) -> None:
        """Checks Final TP (Safe against 0.0)."""
        # 🛑 SAFETY: Ignore if TP is 0.0 (Disabled/Unset)
        if bracket.tp_trigger_price <= 0:
            return

        triggered = False
        if bracket.side == "BUY":
            if ltp >= bracket.tp_trigger_price:
                triggered = True
        else:  # SELL
            if ltp <= bracket.tp_trigger_price:
                triggered = True

        if triggered:
            reason = f"FINAL TP Hit ({ltp})"
            self._execute_exit(
                bracket, bracket.remaining_quantity, reason, is_partial=False
            )

    def _move_sl_to_breakeven(self, bracket: BracketState) -> None:
        """Moves SL to Entry Price (Cost)."""
        with self._lock:
            if bracket.side == "BUY":
                if bracket.entry_price > bracket.sl_trigger_price:
                    bracket.sl_trigger_price = bracket.entry_price
                    LOGGER.info(
                        f"🔒 {bracket.symbol}: SL Moved to Breakeven ({bracket.entry_price})"
                    )
            else:
                if bracket.entry_price < bracket.sl_trigger_price:
                    bracket.sl_trigger_price = bracket.entry_price
                    LOGGER.info(
                        f"🔒 {bracket.symbol}: SL Moved to Breakeven ({bracket.entry_price})"
                    )

    def _extract_exit_quote(
        self, symbol: str
    ) -> tuple[float | None, float | None, float | None]:
        source = self._market_data
        quote: Any = None
        for name in ("get_quote", "get_latest_tick", "get_tick", "get_ltp_snapshot"):
            getter = getattr(source, name, None) if source is not None else None
            if not callable(getter):
                continue
            try:
                quote = getter(symbol)
                if quote:
                    break
            except TypeError:
                try:
                    quote = getter(symbol, allow_pull=False)
                    if quote:
                        break
                except Exception:
                    continue
            except Exception:
                continue
        if quote is None:
            return None, None, None
        if not isinstance(quote, Mapping):
            quote = getattr(quote, "__dict__", {}) or {}

        def _num(*keys: str) -> float | None:
            for key in keys:
                try:
                    value = float(quote.get(key) or 0.0)
                except (TypeError, ValueError, AttributeError):
                    continue
                if value > 0:
                    return value
            return None

        return (
            _num("bid", "best_bid"),
            _num("ask", "best_ask"),
            _num("ltp", "last_price", "last_traded_price"),
        )

    def _is_protective_exit_reason(self, reason: str) -> bool:
        upper = str(reason or "").upper()
        return any(token in upper for token in _PROTECTIVE_EXIT_REASON_TOKENS)

    def _price_exit_order(
        self,
        *,
        bracket: BracketState | None,
        symbol: str,
        side: str,
        reason: str,
        preferred_order_type: str,
        qty: int,
    ) -> tuple[str, float | None, dict[str, Any]]:
        mode = str(preferred_order_type or "LIMIT").upper()
        protective = self._is_protective_exit_reason(reason)
        bid: float | None = None
        ask: float | None = None
        ltp: float | None = None
        fallback = False
        if protective:
            configured = self._exit_protective_order_mode
            mode = (
                "MARKET" if configured not in {"AGGRESSIVE_LIMIT", "LIMIT"} else "LIMIT"
            )
            if configured == "AGGRESSIVE_LIMIT":
                bid, ask, ltp = self._extract_exit_quote(symbol)
                tick_size = 0.05
                reference = bid if side == "SELL" else ask
                if reference is None and ltp is not None:
                    reference = ltp
                if reference is None:
                    if self._exit_fallback_to_market_on_quote_missing:
                        mode = "MARKET"
                        fallback = True
                    else:
                        return (
                            "LIMIT",
                            None,
                            {
                                "quote_missing": True,
                                "mode": "AGGRESSIVE_LIMIT",
                                "bid": bid,
                                "ask": ask,
                                "ltp": ltp,
                                "fallback": False,
                            },
                        )
                else:
                    tick_buffer = self._exit_marketable_limit_slippage_ticks * tick_size
                    pct_buffer = reference * (
                        self._exit_marketable_limit_max_slippage_pct / 100.0
                    )
                    buffer = min(
                        max(tick_buffer, tick_size),
                        pct_buffer if pct_buffer > 0 else tick_buffer,
                    )
                    raw = reference - buffer if side == "SELL" else reference + buffer
                    return (
                        "LIMIT",
                        _round_to_tick(max(raw, tick_size), tick_size=tick_size),
                        {
                            "mode": "AGGRESSIVE_LIMIT",
                            "bid": bid,
                            "ask": ask,
                            "ltp": ltp,
                            "fallback": False,
                        },
                    )
        elif mode == "LIMIT" and bracket is not None:
            # Profit-target exits (TP/TP1/TP2/trailing-profit) must NOT take the 2%
            # downward concession used for generic limits — that converts a winning
            # TP into a loss (real incident: TP latched, SELL LIMIT priced at
            # ltp*0.98 = 224.35, cancelled, replacement filled at 222.30 => -₹104).
            # Price profit exits at the executable bid (SELL) / ask (BUY) with only a
            # small tick buffer; fall back to LTP minus a tick, never minus 2%.
            is_profit_exit = any(
                tok in str(reason or "").upper()
                for tok in ("TP", "TAKE_PROFIT", "TRAILING_PROFIT", "TARGET")
            )
            if is_profit_exit:
                bid, ask, ltp = self._extract_exit_quote(symbol)
                tick_size = 0.05
                reference = bid if side == "SELL" else ask
                if reference is None or reference <= 0:
                    reference = (
                        ltp if (ltp and ltp > 0) else float(bracket.last_ltp or 0.0)
                    )
                if reference and reference > 0:
                    tick_buffer = self._exit_marketable_limit_slippage_ticks * tick_size
                    raw = (
                        reference - tick_buffer
                        if side == "SELL"
                        else reference + tick_buffer
                    )
                    return (
                        "LIMIT",
                        _round_to_tick(max(raw, tick_size), tick_size=tick_size),
                        {
                            "mode": "PROFIT_LIMIT",
                            "bid": bid,
                            "ask": ask,
                            "ltp": ltp,
                            "fallback": False,
                        },
                    )
                # no usable quote -> fall through to MARKET (do not give 2% away)
                mode = "MARKET"
                fallback = True
            else:
                current_ltp = float(bracket.last_ltp or bracket.entry_price or 0.0)
                if current_ltp > 0:
                    max_slippage_pct = parse_float_env(
                        os.getenv("EXIT_MAX_SLIPPAGE_PCT"), 2.0
                    )
                    raw = (
                        current_ltp * (1 - max_slippage_pct / 100)
                        if side == "SELL"
                        else current_ltp * (1 + max_slippage_pct / 100)
                    )
                    return (
                        "LIMIT",
                        _round_to_tick(max(raw, 0.05), tick_size=0.05),
                        {
                            "mode": "LIMIT",
                            "bid": bid,
                            "ask": ask,
                            "ltp": current_ltp,
                            "fallback": False,
                        },
                    )
                mode = "MARKET"
                fallback = True
        return (
            mode,
            None,
            {
                "mode": "MARKET" if mode == "MARKET" else mode,
                "bid": bid,
                "ask": ask,
                "ltp": ltp,
                "fallback": fallback,
            },
        )

    def submit_exit_order(
        self,
        symbol: str,
        qty: int,
        reason: str,
        bracket_id: str,
        preferred_order_type: str = "LIMIT",
    ) -> SubmitExitOrderResult:
        """Submit one broker exit order and return a sanitized structured result."""
        normalized_symbol = normalize_symbol(symbol)
        bracket = self.get_bracket(bracket_id)
        side = "SELL" if (bracket and bracket.side == "BUY") else "BUY"
        order_type, price, pricing_meta = self._price_exit_order(
            bracket=bracket,
            symbol=normalized_symbol,
            side=side,
            reason=reason,
            preferred_order_type=preferred_order_type,
            qty=qty,
        )
        if pricing_meta.get("quote_missing"):
            LOGGER.warning(
                "EXIT_ORDER_PRICING_DECISION bracket_id=%s reason=%s mode=aggressive_limit side=%s qty=%s bid=%s ask=%s ltp=%s price=%s fallback=%s",
                bracket_id,
                reason,
                side,
                qty,
                pricing_meta.get("bid"),
                pricing_meta.get("ask"),
                pricing_meta.get("ltp"),
                price,
                pricing_meta.get("fallback"),
                extra={
                    "event": "EXIT_ORDER_PRICING_DECISION",
                    "bracket_id": bracket_id,
                    "reason": reason,
                    "mode": "aggressive_limit",
                    "side": side,
                    "qty": qty,
                    "bid": pricing_meta.get("bid"),
                    "ask": pricing_meta.get("ask"),
                    "ltp": pricing_meta.get("ltp"),
                    "price": price,
                    "fallback": pricing_meta.get("fallback"),
                },
            )
            return SubmitExitOrderResult(
                False,
                None,
                "quote_missing",
                "quote_missing",
                "protective aggressive limit quote missing",
                True,
                {},
            )
        LOGGER.info(
            "EXIT_ORDER_PRICING_DECISION bracket_id=%s reason=%s mode=%s side=%s qty=%s bid=%s ask=%s ltp=%s price=%s fallback=%s",
            bracket_id,
            reason,
            str(pricing_meta.get("mode") or order_type).lower(),
            side,
            qty,
            pricing_meta.get("bid"),
            pricing_meta.get("ask"),
            pricing_meta.get("ltp"),
            price,
            pricing_meta.get("fallback"),
            extra={
                "event": "EXIT_ORDER_PRICING_DECISION",
                "bracket_id": bracket_id,
                "reason": reason,
                "mode": str(pricing_meta.get("mode") or order_type).lower(),
                "side": side,
                "qty": qty,
                "bid": pricing_meta.get("bid"),
                "ask": pricing_meta.get("ask"),
                "ltp": pricing_meta.get("ltp"),
                "price": price,
                "fallback": pricing_meta.get("fallback"),
            },
        )
        try:
            kwargs: dict[str, Any] = {
                "symbol": normalized_symbol,
                "side": side,
                "quantity": int(qty),
                "order_type": order_type,
                "tag": f"exit_{reason[:3]}_{bracket_id[:8]}",
                "check_risk": False,
                "product": "MIS",
            }
            if price is not None:
                kwargs["price"] = price
            order_id = self.order_manager.place_order(**kwargs)
            if order_id:
                return SubmitExitOrderResult(
                    accepted=True,
                    order_id=str(order_id),
                    status="submitted",
                    retryable=False,
                    broker_payload={
                        "order_id": str(order_id),
                        "order_type": order_type,
                        "side": side,
                    },
                )
            decision = dict(
                getattr(self.order_manager, "_last_order_decision", {}) or {}
            )
            details = dict(decision.get("details") or {})
            broker_payload = dict(details.get("broker_payload") or details)
            error_type = str(
                decision.get("failure_class")
                or decision.get("block_reason")
                or "missing_order_id"
            )
            error_message = str(
                decision.get("error_message")
                or details.get("error_message")
                or details.get("broker_rejection")
                or broker_payload.get("message")
                or broker_payload.get("error")
                or decision.get("block_reason")
                or "place_order returned no order_id"
            )
            return SubmitExitOrderResult(
                accepted=False,
                order_id=None,
                status="rejected",
                error_type=error_type,
                error_message=error_message,
                retryable=bool(
                    decision.get(
                        "retryable",
                        error_type not in {"broker_config_error", "fatal_order_error"},
                    )
                ),
                broker_payload={
                    "order_manager_decision": decision,
                    "broker_payload": broker_payload,
                    "kill_switch_active": bool(
                        getattr(self.order_manager, "_kill_switch_engaged_at", None)
                    ),
                },
            )
        except (
            Exception
        ) as exc:  # noqa: BLE001 - process boundary; result is structured and safe
            message = str(exc)
            retryable = not self._is_fatal_exit_error(message)
            return SubmitExitOrderResult(
                accepted=False,
                order_id=None,
                status="error",
                error_type=type(exc).__name__,
                error_message=message,
                retryable=retryable,
                broker_payload={},
            )

    def _is_fatal_exit_error(self, message: str) -> bool:
        lower = str(message or "").lower()
        if any(pattern in lower for pattern in self._exit_fatal_error_patterns):
            return True
        fatal_defaults = (
            "invalid symbol",
            "invalid instrument",
            "invalid quantity",
            "permission",
            "auth",
            "token",
        )
        return any(pattern in lower for pattern in fatal_defaults)

    def _reconcile_exit_state(
        self, bracket: BracketState, *, requested_by: str
    ) -> bool:
        """Reconcile broker order/position state. Returns True when bracket is closed."""
        now = time.time()
        with self._lock:
            if bracket.exit_state == BracketExitLifecycle.CLOSED.value:
                return True
            if requested_by not in {"pre_submit", "post_submit"}:
                last = float(getattr(bracket, "_last_exit_reconcile_at", 0.0) or 0.0)
                if now - last < self._exit_reconcile_interval_seconds:
                    return False
            setattr(bracket, "_last_exit_reconcile_at", now)
            order_id = bracket.exit_order_id or bracket.pending_exit_order_id
            exit_pending = bool(bracket.exit_pending or order_id)
        LOGGER.info(
            "EXIT_RECONCILE_REQUESTED bracket_id=%s symbol=%s requested_by=%s exit_order_id=%s",
            bracket.bracket_id,
            bracket.symbol,
            requested_by,
            order_id,
        )

        filled = False
        fill_price: float | None = None
        order_status = ""
        try:
            if order_id:
                status = self._get_broker_order_status(str(order_id))
                order_status = str((status or {}).get("status", "")).upper()
                fill_price = self._extract_status_price(status)
                if order_status in _FILLED_STATUSES:
                    filled = True
                elif not order_status:
                    waiter = getattr(self.order_manager, "wait_for_fill", None)
                    if callable(waiter):
                        try:
                            filled = bool(waiter(str(order_id), timeout_sec=0.0))
                        except TypeError:
                            filled = bool(waiter(str(order_id)))
                elif order_status in _CANCELLED_STATUSES:
                    with self._lock:
                        bracket.last_exit_error = f"exit_order_{order_status.lower()}"
                        bracket.exit_order_id = None
                        bracket.pending_exit_order_id = None
                        if (
                            self._exit_retry_enabled
                            and bracket.exit_attempt_count
                            < self._exit_max_retry_attempts
                        ):
                            bracket.exit_state = (
                                BracketExitLifecycle.EXIT_REJECTED_RETRYABLE.value
                            )
                            bracket.next_exit_attempt_at = (
                                time.time()
                                + self._retry_delay_for_attempt(
                                    max(bracket.exit_attempt_count, 1)
                                )
                            )
                        else:
                            self._escalate_exit_locked(
                                bracket, "broker_order_rejected_or_cancelled"
                            )
            flat = self._position_flat_for_symbol(bracket.symbol)
        except Exception as exc:  # noqa: BLE001 - broker reconciliation boundary
            with self._lock:
                bracket.last_exit_error = f"reconcile_failed:{type(exc).__name__}:{exc}"
                age_basis = float(
                    bracket.last_exit_attempt_at or bracket.exit_triggered_at or now
                )
                age = now - age_basis
                if age >= self._exit_unresolved_escalation_seconds:
                    self._escalate_exit_locked(bracket, "reconcile_failed_timeout")
            LOGGER.error(
                "EXIT_RECONCILE_RESULT bracket_id=%s symbol=%s flat=False error_type=%s error_message=%s",
                bracket.bracket_id,
                bracket.symbol,
                type(exc).__name__,
                exc,
            )
            return False

        LOGGER.info(
            "EXIT_RECONCILE_RESULT bracket_id=%s symbol=%s flat=%s order_status=%s filled=%s",
            bracket.bracket_id,
            bracket.symbol,
            flat,
            order_status,
            filled,
        )
        if filled:
            self._close_bracket(
                bracket, close_source="broker_fill", exit_price=fill_price
            )
            LOGGER.info(
                "EXIT_FILLED_CONFIRMED bracket_id=%s order_id=%s",
                bracket.bracket_id,
                order_id,
            )
            return True
        if flat and not order_id:
            if requested_by != "direct_pre_submit":
                prospective_order_id = str(
                    getattr(self.order_manager, "order_id", "") or ""
                )
                if prospective_order_id:
                    with suppress(Exception):
                        prospective_status = str(
                            (
                                self._get_broker_order_status(prospective_order_id)
                                or {}
                            ).get("status", "")
                        ).upper()
                        if prospective_status in _FILLED_STATUSES:
                            return False
                self._close_bracket(
                    bracket,
                    close_source="reconciled_flat",
                    exit_price=fill_price,
                )
                LOGGER.info(
                    "EXIT_RECONCILED_FLAT bracket_id=%s symbol=%s",
                    bracket.bracket_id,
                    bracket.symbol,
                )
                return True
            LOGGER.info(
                "EXIT_RECONCILED_FLAT_IGNORED_WITHOUT_ORDER bracket_id=%s symbol=%s requested_by=%s",
                bracket.bracket_id,
                bracket.symbol,
                requested_by,
            )
            return False
        if flat:
            self._close_bracket(
                bracket, close_source="reconciled_flat", exit_price=fill_price
            )
            LOGGER.info(
                "EXIT_RECONCILED_FLAT bracket_id=%s symbol=%s",
                bracket.bracket_id,
                bracket.symbol,
            )
            return True

        with self._lock:
            age_basis = float(
                bracket.last_exit_attempt_at or bracket.exit_triggered_at or now
            )
            age = now - age_basis
            if age >= self._exit_unresolved_escalation_seconds:
                self._escalate_exit_locked(bracket, "unresolved_timeout")
            else:
                self._log_exit_pending_summary_locked(bracket, now)
        return False

    def _get_broker_order_status(self, order_id: str) -> Mapping[str, Any]:
        broker = getattr(self.order_manager, "_broker", None)
        getter = (
            getattr(broker, "get_order_status", None) if broker is not None else None
        )
        if callable(getter):
            result = getter(order_id)
            return result if isinstance(result, Mapping) else {}
        getter = getattr(broker, "get_orders", None) if broker is not None else None
        if callable(getter):
            for order in getter() or []:
                if str(order.get("order_id") or order.get("id") or "") == str(order_id):
                    return order if isinstance(order, Mapping) else {}
        return {}

    @staticmethod
    def _extract_status_price(status: Mapping[str, Any] | None) -> float | None:
        if not status:
            return None
        for key in ("average_price", "avg_price", "price", "fill_price"):
            try:
                value = float(status.get(key) or 0.0)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
        return None

    def _position_flat_for_symbol(self, symbol: str) -> bool:
        """Return flatness only from the canonical authoritative snapshot."""
        try:
            return self._authoritative_position_quantity(symbol) == 0
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "EXIT_POSITION_SNAPSHOT_INVALID symbol=%s error=%s",
                normalize_symbol(symbol),
                exc,
                extra={
                    "event": "EXIT_POSITION_SNAPSHOT_INVALID",
                    "symbol": normalize_symbol(symbol),
                    "error_type": type(exc).__name__,
                },
            )
            return False

    def _close_bracket(
        self,
        bracket: BracketState,
        *,
        close_source: str,
        exit_price: float | None = None,
    ) -> None:
        with self._lock:
            bracket.remaining_quantity = 0
            bracket.exit_executed = True
            bracket.exit_pending = False
            bracket.exit_in_progress = False
            bracket.active = False
            bracket.position_flat_confirmed = True
            bracket.flat_nonterminal_since_monotonic = None
            bracket.flat_nonterminal_since_utc = None
            bracket.exit_state = BracketExitLifecycle.CLOSED.value
            bracket.entry_status = "CLOSED"
            bracket.pending_exit_order_id = None
            bracket.close_source = close_source
            bracket.closed_at = time.time()
            if exit_price is not None:
                bracket.exit_price = exit_price
            bracket.updated_at = bracket.closed_at
            self._exit_cooldowns.pop(bracket.entry_order_id, None)
        # Compute realized P&L so the log/event carries it (lets the dashboard and
        # Streamlit terminal show daily P&L without re-deriving from the broker).
        # Long option (BUY): (exit-entry)*qty; short: (entry-exit)*qty.
        entry_px = (
            bracket.entry_fill_price
            if bracket.entry_fill_price is not None
            else bracket.entry_price
        )
        exit_px = bracket.exit_price
        filled_qty = int(bracket.quantity or 0)
        realized_pnl: float | None = None
        try:
            if entry_px is not None and exit_px is not None and filled_qty > 0:
                if bracket.side == "BUY":
                    realized_pnl = round(
                        (float(exit_px) - float(entry_px)) * filled_qty, 2
                    )
                else:
                    realized_pnl = round(
                        (float(entry_px) - float(exit_px)) * filled_qty, 2
                    )
        except Exception:  # noqa: BLE001 - never let P&L math break a close
            realized_pnl = None
        LOGGER.info(
            "BRACKET_CLOSED bracket_id=%s symbol=%s close_source=%s side=%s qty=%s entry=%s exit=%s pnl=%s",
            bracket.bracket_id,
            bracket.symbol,
            close_source,
            bracket.side,
            filled_qty,
            entry_px,
            exit_px,
            realized_pnl,
        )
        self._log_bracket_event(
            "BRACKET_CLOSED",
            bracket,
            meta={
                "close_source": close_source,
                "exit_order_id": bracket.exit_order_id,
                "side": bracket.side,
                "qty": filled_qty,
                "entry": entry_px,
                "exit": exit_px,
                "pnl": realized_pnl,
            },
        )
        self._notify_open_position_priority("close", bracket.symbol)
        hook = self._on_exit_complete_hook
        if hook is not None:
            try:
                hook(bracket.symbol)
            except Exception:
                LOGGER.exception(
                    "BRACKET_EXIT_COMPLETE_HOOK_FAILED symbol=%s", bracket.symbol
                )
        try:
            self.save_state()
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "BRACKET_CLOSE_PERSIST_FAILED bracket_id=%s error=%s",
                bracket.bracket_id,
                exc,
            )

    def _execute_exit(
        self,
        bracket: BracketState,
        qty: int,
        reason: str,
        is_partial: bool,
    ) -> ExitExecutionResult:
        """Compatibility wrapper around the exit state-machine submit/reconcile path."""
        if qty <= 0:
            return ExitExecutionResult(
                False, False, None, 0, reason, status="INVALID_QTY"
            )
        if self._reconcile_exit_state(bracket, requested_by="direct_pre_submit"):
            return ExitExecutionResult(
                True,
                True,
                bracket.exit_order_id,
                int(qty),
                reason,
                status="RECONCILED_FLAT",
            )
        with self._lock:
            if not bracket.exit_pending:
                bracket.exit_pending = True
                bracket.exit_reason = reason
                bracket.exit_triggered_at = time.time()
                bracket.exit_state = BracketExitLifecycle.EXIT_TRIGGERED.value
            if bracket.exit_order_id or bracket.pending_exit_order_id:
                return ExitExecutionResult(
                    True,
                    False,
                    bracket.exit_order_id or bracket.pending_exit_order_id,
                    0,
                    reason,
                    status="PENDING",
                )
        result = self.submit_exit_order(
            symbol=bracket.symbol,
            qty=int(qty),
            reason=reason,
            bracket_id=bracket.bracket_id,
            preferred_order_type="LIMIT",
        )
        with self._lock:
            if result.accepted and result.order_id:
                bracket.exit_order_id = result.order_id
                bracket.pending_exit_order_id = result.order_id
                bracket.exit_attempt_count += 1
                bracket.last_exit_attempt_at = time.time()
                bracket.exit_state = BracketExitLifecycle.EXIT_ORDER_SUBMITTED.value
                submitted = True
            else:
                bracket.last_exit_error = result.error_message or result.status
                bracket.exit_state = (
                    BracketExitLifecycle.EXIT_REJECTED_RETRYABLE.value
                    if result.retryable
                    else BracketExitLifecycle.EXIT_REJECTED_FATAL.value
                )
                submitted = False
        closed = self._reconcile_exit_state(bracket, requested_by="direct_post_submit")
        return ExitExecutionResult(
            submitted=submitted,
            confirmed=closed,
            order_id=result.order_id,
            filled_qty=int(qty) if closed else 0,
            reason=reason,
            status="FILLED" if closed else result.status.upper(),
        )

    def _verify_position_closed(self, symbol: str) -> bool:
        """Return true only when a complete authoritative snapshot proves flatness."""
        normalized = normalize_symbol(symbol)
        try:
            return self._authoritative_position_quantity(normalized) == 0
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "POSITION_FLAT_VERIFY_FAILED symbol=%s error=%s",
                normalized,
                exc,
                extra={
                    "event": "POSITION_FLAT_VERIFY_FAILED",
                    "symbol": normalized,
                    "error_type": type(exc).__name__,
                },
            )
            return False

    def _market_fallback_exit(
        self,
        bracket: BracketState,
        qty: int,
        exit_side: str,
        reason: str,
    ) -> ExitExecutionResult:
        """Force market exit after retries. Args: bracket,qty,exit_side,reason; Returns: bool; Raises: None."""
        try:
            LOGGER.warning(
                "EMERGENCY_EXIT_SUBMITTED symbol=%s side=%s qty=%s",
                bracket.symbol,
                exit_side,
                qty,
            )
            if hasattr(self.order_manager, "cancel_orders_for_symbol"):
                try:
                    self.order_manager.cancel_orders_for_symbol(bracket.symbol)
                except Exception as e:
                    LOGGER.error("Failure in _market_fallback_exit: %s", e)

            order_id = self.order_manager.place_order(
                symbol=bracket.symbol,
                side=exit_side,
                quantity=qty,
                order_type="MARKET",
                tag=f"mkt_exit_{reason[:3]}",
                check_risk=False,
                product="MIS",
            )
            if not order_id:
                LOGGER.critical(
                    "MARKET_FALLBACK_REJECTED symbol=%s qty=%s", bracket.symbol, qty
                )
                return ExitExecutionResult(
                    True, False, None, 0, reason, status="REJECTED"
                )

            if hasattr(self.order_manager, "wait_for_fill"):
                filled = bool(
                    self.order_manager.wait_for_fill(order_id, timeout_sec=3.0)
                )
                if not filled:
                    LOGGER.critical(
                        "MARKET_FALLBACK_UNFILLED symbol=%s order_id=%s",
                        bracket.symbol,
                        order_id,
                    )
                    return ExitExecutionResult(
                        True, False, str(order_id), 0, reason, status="UNFILLED"
                    )

            position_closed = self._verify_position_closed(bracket.symbol)
            if not position_closed:
                LOGGER.critical(
                    "MARKET_FALLBACK_POSITION_OPEN symbol=%s order_id=%s",
                    bracket.symbol,
                    order_id,
                )
                return ExitExecutionResult(
                    True, False, str(order_id), 0, reason, status="POSITION_OPEN"
                )

            LOGGER.info(
                "MARKET_FALLBACK_EXECUTED symbol=%s order_id=%s",
                bracket.symbol,
                order_id,
            )
            return ExitExecutionResult(
                True, True, str(order_id), int(qty), reason, status="FILLED"
            )
        except Exception as e:
            LOGGER.critical("Failure in _market_fallback_exit: %s", e)
            return ExitExecutionResult(False, False, None, 0, reason, status="ERROR")

    # --------------------------------------------------------------------------
    # 4. SYNC & MANUAL INTERVENTION (World Class)
    # --------------------------------------------------------------------------

    def sync_manual_exit(self, symbol: str, quantity_left: int) -> None:
        """
        Called by OrderManager/PositionManager when position size changes externally.
        ORPHAN HANDLING: If qty goes to 0, kill all brackets for symbol.
        """
        if quantity_left <= 0:
            self.manual_override_close(symbol, reason="External/Manual Exit Detected")
        else:
            # Logic for partial manual exit can be added here
            # For now, we assume if some qty remains, we keep brackets active
            pass

    def manual_override_close(
        self, symbol: str, reason: str = "Manual Override"
    ) -> None:
        """Force close/remove all brackets for a symbol."""
        with self._lock:
            relevant_ids = self._symbol_map.get(symbol, [])
            if not relevant_ids:
                return

            count = 0
            for eid in list(relevant_ids):
                if eid in self._brackets:
                    # We strictly unregister, assuming the position is already gone/closing
                    self.unregister_bracket(eid)
                    count += 1

            if count > 0:
                LOGGER.info(
                    f"🧹 Cleaned up {count} brackets for {symbol} due to: {reason}"
                )

    def reconcile_with_broker(self, callback: Callable[[], Any]) -> Any:
        """Args: callback. Returns: callback result. Raises: Exception from callback."""
        with self._reconcile_lock:
            return callback()

    def sync_order_status(
        self, broker_order_id: str, status: str, filled_qty: int
    ) -> None:
        """
        Detects if an Exit order initiated externally has filled.
        Used to keep internal state consistent.
        """
        if status not in _FILLED_STATUSES:
            return

        # This is a hook for future expansion where we map every broker order back to a bracket.
        # Currently handled via sync_manual_exit based on net position.
        pass

    # --------------------------------------------------------------------------
    # 5. DYNAMIC UPDATES (Trailing & Utils)
    # --------------------------------------------------------------------------

    def update_trailing_sl(self, symbol: str, new_sl: float) -> None:
        """Update SL monotonically for all active brackets on a symbol. Args: symbol,new_sl; Returns: None; Raises: None."""
        rounded_sl = _round_to_tick(new_sl)
        with self._lock:
            relevant_ids = self._symbol_map.get(symbol, [])
            if not relevant_ids:
                return

            for eid in relevant_ids:
                bracket = self._brackets.get(eid)
                if not bracket or not bracket.active or bracket.exit_in_progress:
                    continue

                old_sl = bracket.sl_trigger_price
                if bracket.side == "BUY":
                    bracket.sl_trigger_price = max(old_sl, rounded_sl)
                else:
                    bracket.sl_trigger_price = min(old_sl, rounded_sl)

                if bracket.sl_trigger_price != old_sl:
                    bracket.updated_at = time.time()
                    LOGGER.debug(
                        "TRAILING_SL_UPDATED symbol=%s old=%s new=%s",
                        symbol,
                        round(old_sl, 2),
                        round(bracket.sl_trigger_price, 2),
                    )

    # --------------------------------------------------------------------------
    # 6. HOUSEKEEPING & UTILS
    # --------------------------------------------------------------------------

    def is_symbol_managed(self, symbol: str) -> bool:
        """Return whether a non-terminal protective lifecycle owns ``symbol``.

        Pending-entry brackets are deliberately included. They are not active yet,
        but they prove that the position belongs to the bot and prevent orphan
        recovery from installing a second exit authority during fill latency.
        """

        symbol_key = normalize_symbol(symbol)
        terminal_states = {
            BracketExitLifecycle.CLOSED.value,
            BracketExitLifecycle.EXIT_FILLED.value,
            BracketExitLifecycle.EXIT_RECONCILED_FLAT.value,
        }
        with self._lock:
            for entry_id in self._symbol_map.get(symbol_key, []):
                bracket = self._brackets.get(entry_id)
                if bracket is None or bracket.remaining_quantity <= 0:
                    continue
                if bracket.monitoring_only or bracket.position_flat_confirmed:
                    continue
                if str(bracket.exit_state or "").upper() in terminal_states:
                    continue
                return True
        return False

    def get_bracket(self, entry_id: str) -> Optional[BracketState]:
        with self._lock:
            return self._brackets.get(entry_id)

    def reconcile_symbol_flat(self, symbol: str) -> int:
        """Drop all brackets for a symbol the broker now reports as flat.

        Called when broker reconciliation finds a position closed externally
        (manual square-off / Zerodha auto-square-off). Without this, a bracket
        lingers as 'managed' after the underlying position is gone, so the bot
        keeps trying to manage / re-adopt a phantom position (observed: the same
        23950CE orphan re-adopted thousands of times). Returns count removed.
        """
        symbol = normalize_symbol(symbol)
        with self._lock:
            entry_ids = list(self._symbol_map.get(symbol, []))
        removed = 0
        for eid in entry_ids:
            try:
                self.unregister_bracket(eid)
                removed += 1
            except Exception:  # noqa: BLE001
                continue
        if removed:
            LOGGER.info(
                "BRACKET_RECONCILED_FLAT symbol=%s removed=%s reason=broker_position_closed",
                symbol,
                removed,
            )
        return removed

    def unregister_bracket(self, entry_id: str) -> None:
        """Remove a bracket from memory and indices."""
        with self._lock:
            if entry_id in self._brackets:
                bracket = self._brackets[entry_id]
                symbol = bracket.symbol

                # Cleanup Main Dict
                del self._brackets[entry_id]

                # Cleanup Symbol Map
                if symbol in self._symbol_map:
                    if entry_id in self._symbol_map[symbol]:
                        self._symbol_map[symbol].remove(entry_id)
                    if not self._symbol_map[symbol]:
                        del self._symbol_map[symbol]

                # Cleanup Controller
                if entry_id in self._trailing_controllers:
                    del self._trailing_controllers[entry_id]

                # ✅ NEW: Cleanup Exit Cooldown
                if entry_id in self._exit_cooldowns:
                    del self._exit_cooldowns[entry_id]

                self._notify_open_position_priority("close", symbol)

                # ✅ FIX: Persist removal immediately
                self.save_state()
                self._sync_active_bracket_symbols_to_mdm()

            # Cleanup reverse index (outside main check if orphaned)
            if entry_id in self._order_to_entry:
                del self._order_to_entry[entry_id]

    def cleanup_stale_brackets(self, max_age_seconds: int = 86400) -> int:
        """Remove old inactive brackets."""
        now = time.time()
        with self._lock:
            to_remove = [
                eid
                for eid, b in self._brackets.items()
                if (now - b.created_at) > max_age_seconds
            ]
            for eid in to_remove:
                self.unregister_bracket(eid)

            if to_remove:
                LOGGER.info(f"🧹 Cleaned up {len(to_remove)} stale brackets.")
            return len(to_remove)

    def get_stats(self) -> dict[str, Any]:
        """Diagnostic stats."""
        with self._lock:
            return {
                "active_brackets": len(self._brackets),
                "symbols_managed": len(self._symbol_map),
                "atr_tracked_symbols": len(self._current_atr),
                "adaptive_controllers": len(self._trailing_controllers),
            }

    def _log_bracket_event(
        self,
        event_type: str,
        bracket: BracketState,
        *,
        meta: Mapping[str, object] | None = None,
    ) -> None:
        """Queue non-blocking bracket journal event. Args: event_type,bracket,meta; Returns: None; Raises: None."""
        journal = self._trade_journal
        if journal is None:
            return
        try:
            journal.log_event(
                {
                    "event_type": event_type,
                    "timestamp": time.time(),
                    "symbol": bracket.symbol,
                    "side": bracket.side,
                    "qty": int(bracket.remaining_quantity),
                    "price": float(bracket.last_ltp or bracket.entry_price),
                    "order_id": bracket.entry_order_id,
                    "meta": dict(meta or {}),
                }
            )
        except Exception as e:
            LOGGER.error("Failure in _log_bracket_event: %s", e)

    # ----------------------------------------------------------------
    # 💾 PERSISTENCE LAYER (Add to BracketManager)
    # ----------------------------------------------------------------
    def _is_live_execution(self) -> bool:
        checker = getattr(self.order_manager, "is_live_mode", None)
        if callable(checker):
            try:
                return bool(checker())
            except Exception:  # noqa: BLE001
                pass
        mode = str(os.getenv("EXECUTION_MODE", "SHADOW") or "SHADOW").strip().upper()
        enabled = str(
            os.getenv("ENABLE_LIVE") or os.getenv("ENABLE_LIVE_TRADING") or "false"
        ).strip().lower() in {"1", "true", "yes", "on"}
        return mode == "LIVE" and enabled

    def _mark_persistence_degraded(
        self,
        reason: str,
        error: Exception | None = None,
    ) -> None:
        self._persistence_degraded_reason = str(reason)
        LOGGER.critical(
            "BRACKET_PERSISTENCE_DEGRADED reason=%s path=%s error=%s",
            reason,
            self._state_storage_path,
            error,
            extra={
                "event": "BRACKET_PERSISTENCE_DEGRADED",
                "reason": str(reason),
                "path": self._state_storage_path,
                "error_type": type(error).__name__ if error is not None else None,
            },
        )
        self._notify_event(
            "BRACKET_PERSISTENCE_DEGRADED",
            {
                "reason": str(reason),
                "path": self._state_storage_path,
                "message": "Protective exits continue; new entries are frozen until durable state is healthy.",
            },
        )

    def _clear_persistence_degraded(self) -> None:
        self._persistence_degraded_reason = None
        self._last_persist_success_at = time.time()

    def _authoritative_position_snapshot(self) -> PositionSnapshot:
        broker = getattr(self.order_manager, "_broker", None)
        getter = getattr(broker, "get_positions", None) if broker is not None else None
        if not callable(getter):
            raise PositionSnapshotError("broker get_positions is unavailable")
        return decode_position_snapshot(getter())

    def _authoritative_position_quantity(self, symbol: str) -> int:
        snapshot = self._authoritative_position_snapshot()
        return abs(int(snapshot.quantity_for(normalize_symbol(symbol))))

    def _decode_restored_bracket(
        self,
        entry_id: str,
        payload: Mapping[str, Any],
    ) -> BracketState:
        def finite_float(name: str, value: Any, minimum: float = 0.0) -> float:
            try:
                result = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{entry_id}: invalid {name}") from exc
            if not math.isfinite(result) or result < minimum:
                raise ValueError(f"{entry_id}: invalid {name}")
            return result

        stored_id = str(payload.get("entry_order_id") or entry_id).strip()
        if stored_id != str(entry_id):
            raise ValueError(f"{entry_id}: entry id mismatch")
        symbol = normalize_symbol(str(payload.get("symbol") or ""))
        if not symbol:
            raise ValueError(f"{entry_id}: symbol missing")
        quantity = int(payload.get("quantity") or 0)
        remaining = int(payload.get("remaining_quantity", quantity) or 0)
        if quantity <= 0 or remaining < 0 or remaining > quantity:
            raise ValueError(f"{entry_id}: invalid quantity state")
        trailing_config = payload.get("trailing_config") or {}
        if not isinstance(trailing_config, Mapping):
            raise ValueError(f"{entry_id}: invalid trailing config")
        raw_targets = payload.get("tp_levels") or []
        if isinstance(raw_targets, (str, bytes, Mapping)):
            raise ValueError(f"{entry_id}: invalid target list")
        targets: list[TargetLevel] = []
        for index, raw_target in enumerate(raw_targets):
            if not isinstance(raw_target, Mapping):
                raise ValueError(f"{entry_id}: invalid target {index}")
            target_qty = int(raw_target.get("quantity") or 0)
            target_price = finite_float("target price", raw_target.get("price"))
            if target_qty <= 0 or target_qty > quantity or target_price <= 0:
                raise ValueError(f"{entry_id}: invalid target {index}")
            targets.append(
                TargetLevel(
                    price=target_price,
                    quantity=target_qty,
                    executed=bool(raw_target.get("executed", False)),
                    name=str(raw_target.get("name") or "TP"),
                )
            )
        entry_price = finite_float("entry price", payload.get("entry_price"))
        state = BracketState(
            entry_order_id=stored_id,
            symbol=symbol,
            side=str(payload.get("side") or ""),
            quantity=quantity,
            entry_price=entry_price,
            sl_trigger_price=finite_float(
                "stop loss", payload.get("sl_trigger_price", 0.0)
            ),
            tp_trigger_price=finite_float(
                "take profit", payload.get("tp_trigger_price", 0.0)
            ),
            initial_sl_trigger_price=finite_float(
                "initial stop loss",
                payload.get(
                    "initial_sl_trigger_price",
                    payload.get("sl_trigger_price", 0.0),
                ),
            ),
            remaining_quantity=remaining,
            tp_levels=targets,
            is_virtual=bool(payload.get("is_virtual", True)),
            active=bool(payload.get("active", True)),
            trailing_enabled=bool(payload.get("trailing_enabled", False)),
            trailing_config=dict(trailing_config),
            virtual_sl_id=str(payload.get("virtual_sl_id") or f"vsl_{stored_id}"),
            highest_ltp=finite_float(
                "highest ltp", payload.get("highest_ltp", entry_price)
            ),
            lowest_ltp=finite_float(
                "lowest ltp", payload.get("lowest_ltp", entry_price)
            ),
            last_ltp=finite_float("last ltp", payload.get("last_ltp", entry_price)),
            previous_ltp=finite_float(
                "previous ltp",
                payload.get("previous_ltp", payload.get("last_ltp", entry_price)),
            ),
            tag=payload.get("tag"),
            created_at=finite_float(
                "created at", payload.get("created_at", time.time())
            ),
            updated_at=finite_float(
                "updated at", payload.get("updated_at", time.time())
            ),
            exit_executed=bool(payload.get("exit_executed", False)),
            pending_exit_order_id=payload.get("pending_exit_order_id"),
            exit_in_progress=bool(payload.get("exit_in_progress", False)),
            entry_confirmed=bool(
                payload.get("entry_confirmed", payload.get("active", True))
            ),
            monitoring_only=bool(payload.get("monitoring_only", False)),
            entry_status=str(
                payload.get("entry_status")
                or ("ACTIVE" if payload.get("active", True) else "PENDING_ENTRY")
            ),
            exit_state=str(
                payload.get("exit_state")
                or (
                    BracketExitLifecycle.OPEN_ACTIVE.value
                    if payload.get("active", True)
                    else BracketExitLifecycle.OPEN_PENDING_FILL.value
                )
            ),
            exit_order_id=payload.get("exit_order_id")
            or payload.get("pending_exit_order_id"),
            entry_fill_price=payload.get("entry_fill_price"),
            exit_reason=payload.get("exit_reason"),
            exit_triggered_at=payload.get("exit_triggered_at"),
            exit_attempt_count=int(payload.get("exit_attempt_count", 0) or 0),
            last_exit_attempt_at=payload.get("last_exit_attempt_at"),
            last_exit_error=payload.get("last_exit_error"),
            exit_pending=bool(payload.get("exit_pending", False)),
            next_exit_attempt_at=payload.get("next_exit_attempt_at"),
            last_exit_summary_at=float(payload.get("last_exit_summary_at", 0.0) or 0.0),
            closed_at=payload.get("closed_at"),
            position_flat_confirmed=bool(payload.get("position_flat_confirmed", False)),
            flat_nonterminal_since_utc=payload.get("flat_nonterminal_since_utc")
            or payload.get("_flat_nonterminal_since_utc"),
            close_source=payload.get("close_source"),
            exit_price=payload.get("exit_price"),
            escalated_at=payload.get("escalated_at"),
            _market_escalation_fired=bool(
                payload.get("market_escalation_fired", False)
            ),
            _atr_warning_logged=bool(payload.get("atr_warning_logged", False)),
            ledger_realized_pnl=payload.get("ledger_realized_pnl"),
            _ledger_pending_entry_price=payload.get("ledger_pending_entry_price"),
            _ledger_pending_exit_order_id=payload.get("ledger_pending_exit_order_id"),
            _ledger_pending_exit_quantity=int(
                payload.get("ledger_pending_exit_quantity", 0) or 0
            ),
            _ledger_pending_exit_price=payload.get("ledger_pending_exit_price"),
            _ledger_pending_exit_target=payload.get("ledger_pending_exit_target"),
            _ledger_release_hook_fired=bool(
                payload.get("ledger_release_hook_fired", False)
            ),
            _filled_exit_sync_started_at=float(
                payload.get("filled_exit_sync_started_at", 0.0) or 0.0
            ),
            _filled_exit_sync_order_id=payload.get("filled_exit_sync_order_id"),
            _last_exit_reconcile_at=float(
                payload.get("last_exit_reconcile_at", 0.0) or 0.0
            ),
            last_processed_tick_id=payload.get("last_processed_tick_id"),
            last_trail_price=payload.get("last_trail_price"),
            trail_revision=int(payload.get("trail_revision", 0) or 0),
        )
        return state

    def _get_storage_path(self) -> Path:
        """Return durable state storage; ephemeral fallback is forbidden in LIVE."""
        configured = Path(os.getenv("DATA_DIR", "data")) / "virtual_brackets.json"
        try:
            configured.parent.mkdir(parents=True, exist_ok=True)
            probe = configured.parent / f".bracket_write_test_{os.getpid()}"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            resolved_parent = configured.parent.resolve()
            normalized_parent = str(resolved_parent).replace("\\", "/").lower()
            temp_root = Path(tempfile.gettempdir()).resolve()
            durable = True
            if normalized_parent.startswith("/tmp"):
                durable = False
            elif any(
                part.lower().startswith(("pytest-", ".pytest-"))
                for part in resolved_parent.parts
            ):
                durable = False
            elif resolved_parent == temp_root or temp_root in resolved_parent.parents:
                durable = False
            if self._is_live_execution() and not durable:
                raise OSError("LIVE bracket state cannot use ephemeral /tmp storage")
            self._state_storage_path = str(configured)
            self._state_storage_durable = durable
            return configured
        except (OSError, PermissionError) as exc:
            if self._is_live_execution():
                self._state_storage_path = str(configured)
                self._state_storage_durable = False
                raise OSError(
                    f"durable bracket storage unavailable: {configured}"
                ) from exc
            fallback = Path("/tmp") / "virtual_brackets.json"
            fallback.parent.mkdir(parents=True, exist_ok=True)
            self._state_storage_path = str(fallback)
            self._state_storage_durable = False
            LOGGER.warning(
                "BRACKET_STORAGE_EPHEMERAL_FALLBACK path=%s error=%s",
                fallback,
                exc,
                extra={
                    "event": "BRACKET_STORAGE_EPHEMERAL_FALLBACK",
                    "path": str(fallback),
                },
            )
            return fallback

    def save_state(self) -> None:
        """Atomically persist one versioned coherent bracket snapshot."""
        temp_path: Path | None = None
        try:
            path = self._get_storage_path()
            with self._lock:
                payload = {
                    "schema_version": 2,
                    "saved_at": time.time(),
                    "brackets": {
                        entry_id: bracket.to_dict()
                        for entry_id, bracket in self._brackets.items()
                    },
                    "exit_rescue_attempts": dict(
                        getattr(self, "_exit_rescue_attempts", {})
                    ),
                    "exit_order_open_since": dict(
                        getattr(self, "_exit_order_open_since", {})
                    ),
                }
                snapshots = list(self._brackets.values())
            temp_path = path.with_suffix(
                f"{path.suffix}.tmp.{threading.get_ident()}.{time.time_ns()}"
            )
            with open(temp_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
            try:
                directory_fd = os.open(str(path.parent), os.O_RDONLY)
            except OSError:
                directory_fd = None
            if directory_fd is not None:
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            self._clear_persistence_degraded()
        except Exception as exc:
            if temp_path is not None:
                with suppress(OSError):
                    temp_path.unlink()
            self._mark_persistence_degraded("snapshot_write_failed", exc)
            raise

        for bracket in snapshots:
            self._log_bracket_event(
                "BRACKET_SNAPSHOT",
                bracket,
                meta={
                    "active": bracket.active,
                    "sl_trigger_price": bracket.sl_trigger_price,
                    "tp_trigger_price": bracket.tp_trigger_price,
                    "remaining_quantity": bracket.remaining_quantity,
                    "exit_executed": bracket.exit_executed,
                    "pending_exit_order_id": bracket.pending_exit_order_id,
                    "storage_durable": self._state_storage_durable,
                },
            )

    def load_state(self) -> bool:
        """Restore a complete bracket snapshot atomically before watchdog startup."""
        path = self._get_storage_path()
        if not path.exists():
            LOGGER.info("No bracket state file found - starting fresh")
            return False
        try:
            decoded = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(decoded, Mapping):
                raise ValueError("bracket state payload must be an object")
            if "brackets" in decoded:
                version = int(decoded.get("schema_version", 0) or 0)
                if version not in {1, 2}:
                    raise ValueError(f"unsupported bracket state schema {version}")
                records = decoded.get("brackets")
                rescue_attempts = decoded.get("exit_rescue_attempts") or {}
                open_since = decoded.get("exit_order_open_since") or {}
            else:
                records = decoded
                rescue_attempts = {}
                open_since = {}
            if not isinstance(records, Mapping):
                raise ValueError("bracket records must be an object")
            if not isinstance(rescue_attempts, Mapping) or not isinstance(
                open_since, Mapping
            ):
                raise ValueError("bracket recovery maps are invalid")

            temp_brackets: Dict[str, BracketState] = {}
            temp_order_map: Dict[str, str] = {}
            temp_symbol_map: Dict[str, List[str]] = {}
            temp_controllers: Dict[str, Any] = {}
            controller_errors: list[str] = []
            for raw_entry_id, record in records.items():
                entry_id = str(raw_entry_id).strip()
                if not entry_id or not isinstance(record, Mapping):
                    raise ValueError(f"invalid bracket record {raw_entry_id!r}")
                bracket = self._decode_restored_bracket(entry_id, record)
                temp_brackets[entry_id] = bracket
                temp_order_map[entry_id] = entry_id
                temp_symbol_map.setdefault(bracket.symbol, []).append(entry_id)
                if bracket.trailing_enabled:
                    try:
                        if self._trailing_controller_factory is not None:
                            temp_controllers[entry_id] = (
                                self._trailing_controller_factory(bracket)
                            )
                        elif (
                            bracket.trailing_config.get("mode") == "ATR"
                            and self._atr_provider
                            and AdaptiveTrailingController
                        ):
                            mult = float(
                                bracket.trailing_config.get("mult", 1.5) or 1.5
                            )
                            spec = TrailingSpec(
                                trail_by=20.0, step=0.25, activation=0.3
                            )
                            temp_controllers[entry_id] = AdaptiveTrailingController(
                                symbol=bracket.symbol,
                                side="LONG" if bracket.side == "BUY" else "SHORT",
                                entry=bracket.entry_price,
                                sl_order_id=bracket.virtual_sl_id,
                                variety="virtual",
                                spec=spec,
                                get_ltp=lambda _symbol, _b=bracket: _b.last_ltp,
                                modify_order=self._virtual_modify_sl,
                                atr_provider=self._atr_provider,
                                journal=MockJournal(),
                                atr_multiplier=mult,
                            )
                    except Exception as exc:  # noqa: BLE001
                        controller_errors.append(
                            f"{entry_id}:{type(exc).__name__}:{exc}"
                        )

            parsed_rescue = {
                str(key): int(value) for key, value in rescue_attempts.items()
            }
            parsed_open_since = {
                str(key): float(value) for key, value in open_since.items()
            }
            with self._lock:
                self._brackets = temp_brackets
                self._order_to_entry = temp_order_map
                self._symbol_map = temp_symbol_map
                self._trailing_controllers = temp_controllers
                if hasattr(self, "_exit_rescue_attempts"):
                    self._exit_rescue_attempts = parsed_rescue
                if hasattr(self, "_exit_order_open_since"):
                    self._exit_order_open_since = parsed_open_since
            self._clear_persistence_degraded()
            self._recovery_degraded_reason = None
            if controller_errors:
                self._recovery_degraded_reason = "trailing_controller_restore_failed"
                LOGGER.critical(
                    "BRACKET_TRAILING_RESTORE_DEGRADED errors=%s",
                    controller_errors,
                    extra={
                        "event": "BRACKET_TRAILING_RESTORE_DEGRADED",
                        "errors": controller_errors,
                    },
                )
            LOGGER.info(
                "BRACKET_STATE_RESTORED count=%s path=%s",
                len(temp_brackets),
                path,
                extra={
                    "event": "BRACKET_STATE_RESTORED",
                    "count": len(temp_brackets),
                    "path": str(path),
                },
            )
            self._resubscribe_restored_brackets()
            self._sync_active_bracket_symbols_to_mdm()
            return True
        except Exception as exc:
            self._mark_persistence_degraded("snapshot_restore_failed", exc)
            raise

    def _resubscribe_restored_brackets(self) -> None:
        """Resubscribe to market data for all restored brackets."""
        if not self._market_data:
            LOGGER.warning("Cannot resubscribe: MarketDataManager not available")
            return

        unique_symbols = set()
        with self._lock:
            for bracket in self._brackets.values():
                if bracket.active and bracket.remaining_quantity > 0:
                    unique_symbols.add(bracket.symbol)

        for symbol in unique_symbols:
            try:
                # Create callback closure for this symbol
                # NOTE: market_data.subscribe passes a tick dict, NOT (sym, ltp)
                def tick_callback(tick_data: dict, _sym=symbol) -> None:
                    try:
                        ltp = (
                            tick_data.get("ltp")
                            or tick_data.get("last_price")
                            or tick_data.get("price")
                        )
                        if ltp and float(ltp) > 0:
                            self.on_tick(_sym, float(ltp))
                    except Exception as e:
                        LOGGER.exception("Unhandled exception", exc_info=True)
                        raise

                # Register with market data manager
                if hasattr(self._market_data, "subscribe"):
                    self._market_data.subscribe(symbol, tick_callback)
                elif hasattr(self._market_data, "register_callback"):
                    self._market_data.register_callback(symbol, tick_callback)

                LOGGER.info(f"🔔 Resubscribed {symbol} to market data feed")

            except Exception as e:
                LOGGER.error(f"Failed to resubscribe {symbol}: {e}")

        if unique_symbols:
            LOGGER.info(f"✅ Resubscribed {len(unique_symbols)} symbols to market data")

    def attach_orphan_position(
        self, symbol: str, side: str, qty: int, entry_price: float
    ) -> str:
        """
        Wraps an existing naked position in a protective bracket.
        Called by Runner when 'ORPHAN GUARD' triggers.
        """
        symbol = normalize_symbol(symbol)
        if self._reconcile_lock.locked():
            return ""
        # ── FIX: use symbol-stable ID so repeated adoption attempts hit the
        # dedup path in register_virtual_bracket (which updates triggers on an
        # existing bracket) instead of creating a fresh bracket every second.
        # Timestamp-based oids bypassed dedup and filled _symbol_map with dead
        # brackets, causing is_symbol_managed() to always return False.
        oid = f"orphan_{symbol}"
        now = time.time()
        last_attempt = self._orphan_retry_last_attempt.get(symbol)
        if last_attempt is not None and (now - last_attempt) < 10:
            return oid
        if self._orphan_retry_count.get(symbol, 0) >= 3:
            LOGGER.error("Orphan adoption disabled after max retries: %s", symbol)
            return oid

        # 1. Dynamic ATR Calculation (if provider available)
        atr = max(entry_price * 0.005, 1.0)
        try:
            if self._atr_provider:
                calc_atr = None
                if hasattr(self._atr_provider, "get_current_atr"):
                    calc_atr = self._atr_provider.get_current_atr(symbol)
                elif hasattr(self._atr_provider, "get_atr"):
                    atr_snapshot = self._atr_provider.get_atr(symbol)
                    calc_atr = getattr(atr_snapshot, "value", atr_snapshot)
                if calc_atr and float(calc_atr) > 0:
                    atr = float(calc_atr)
        except Exception as exc:
            self._orphan_retry_count[symbol] = (
                self._orphan_retry_count.get(symbol, 0) + 1
            )
            self._orphan_retry_last_attempt[symbol] = now
            LOGGER.exception("[CRITICAL FAILURE]", exc_info=True)
            raise

        # 2. Define Rescue Levels (1.5x Risk / 3.0x Reward)
        # Handle 'BUY'/'LONG' vs 'SELL'/'SHORT'
        is_long = _normalize_bracket_side(side) == "BUY"

        if is_long:
            sl = entry_price - (atr * 0.7)
            tp = entry_price + (atr * 3.0)
        else:
            sl = entry_price + (atr * 0.7)
            tp = entry_price - (atr * 3.0)

        # 3. Register as IMMEDIATELY ACTIVE
        self.register_virtual_bracket(
            order_id=oid,
            symbol=symbol,
            side="BUY" if is_long else "SELL",
            qty=abs(qty),
            price=entry_price,
            sl=sl,
            tp=tp,
            tag="orphan_recovery",
            trailing_atr_mult=1.5,
            activate_immediately=True,  # 🟢 Critical: It's already live
        )

        self._orphan_retry_count.pop(symbol, None)
        self._orphan_retry_last_attempt.pop(symbol, None)
        LOGGER.warning(
            f"🧯 ORPHAN ATTACHED: {symbol} | Entry={entry_price:.2f} | "
            f"SL={sl:.2f} | TP={tp:.2f} | ID={oid}"
        )
        return oid

    def create_bracket(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        quantity: int,
        strategy: str = "auto",
        order_id: str | None = None,
        confirmed_position: bool = False,
    ) -> str:
        """
        Create a virtual bracket with specified SL/TP levels.

        This is an alias method that provides the API expected by runner.py.
        Unlike attach_orphan_position(), this uses the SPECIFIED SL/TP values
        rather than auto-calculating them from ATR.

        Args:
            symbol: Trading symbol (e.g., "NIFTY2620324650CE")
            side: "LONG", "SHORT", "BUY", or "SELL"
            entry_price: Entry price for the position
            stop_loss: Stop loss price (must be > 0)
            take_profit: Take profit price (must be > 0)
            quantity: Position size (will use absolute value)
            strategy: Strategy name for tagging (default: "auto")
            order_id: Optional specific order ID (auto-generated if None)

        Returns:
            The order_id (entry_id) used for bracket registration.

        Example:
            >>> bm.create_bracket(
            ...     symbol="NIFTY2620324650CE",
            ...     side="LONG",
            ...     entry_price=165.0,
            ...     stop_loss=155.0,
            ...     take_profit=180.0,
            ...     quantity=65,
            ...     strategy="VWAP_Pro"
            ... )
        """
        import time

        # Normalize side to LONG/SHORT
        normalized_side = _normalize_bracket_side(side)
        if normalized_side == "BUY":
            normalized_side = "LONG"
        elif normalized_side == "SELL":
            normalized_side = "SHORT"

        # Validate side
        if normalized_side not in ("LONG", "SHORT"):
            LOGGER.warning(
                f"⚠️ create_bracket: Invalid side '{side}', defaulting to LONG"
            )
            normalized_side = "LONG"

        # Auto-generate order_id if not provided
        if not order_id:
            safe_symbol = symbol.replace(":", "_")[:20]
            order_id = f"bracket_{int(time.time() * 1000)}_{safe_symbol}"

        # Validate SL/TP (use defaults if invalid)
        if stop_loss <= 0:
            LOGGER.warning(
                f"⚠️ create_bracket: Invalid SL={stop_loss}, using 5% default"
            )
            stop_loss = (
                entry_price * 0.95 if normalized_side == "BUY" else entry_price * 1.05
            )

        if take_profit <= 0:
            LOGGER.warning(
                f"⚠️ create_bracket: Invalid TP={take_profit}, using 10% default"
            )
            take_profit = (
                entry_price * 1.10 if normalized_side == "BUY" else entry_price * 0.90
            )

        # Register the bracket
        self.register_virtual_bracket(
            order_id=order_id,
            symbol=symbol,
            side=normalized_side,
            qty=abs(quantity),
            price=entry_price,
            sl=stop_loss,
            tp=take_profit,
            tag=strategy,
            trailing_atr_mult=1.5,  # Enable ATR-based trailing
            activate_immediately=confirmed_position,
        )

        LOGGER.info(
            f"✅ create_bracket: {symbol} | {normalized_side} | "
            f"Entry={entry_price:.2f} | SL={stop_loss:.2f} | TP={take_profit:.2f} | "
            f"ID={order_id}"
        )

        return order_id

    # ═══════════════════════════════════════════════════════════════════════════
    # ✅ FIX: Add get_bracket_by_symbol() for Symbol-Based Lookup
    # ═══════════════════════════════════════════════════════════════════════════

    def get_bracket_by_symbol(self, symbol: str) -> Optional[BracketState]:
        """
        Get the first active bracket for a given symbol.

        This is different from get_bracket(entry_id) which looks up by order_id.
        Use this when you have a symbol but not the order_id.

        Args:
            symbol: Trading symbol (e.g., "NIFTY2620324650CE")

        Returns:
            BracketState if found and active, None otherwise.

        Example:
            >>> bracket = bm.get_bracket_by_symbol("NIFTY2620324650CE")
            >>> if bracket:
            ...     print(f"SL: {bracket.sl_trigger_price}")
        """
        with self._lock:
            entry_ids = self._symbol_map.get(symbol, [])

            # First pass: Look for active brackets
            for entry_id in entry_ids:
                bracket = self._brackets.get(entry_id)
                if bracket and bracket.active and bracket.remaining_quantity > 0:
                    return bracket

            # Second pass: Look for any bracket (even inactive)
            for entry_id in entry_ids:
                bracket = self._brackets.get(entry_id)
                if bracket and bracket.remaining_quantity > 0:
                    return bracket

            return None

    def reactivate_bracket_after_rejected_exit(
        self, symbol: str, rejected_order_id: str, reason: str
    ) -> bool:
        """Reactivate a bracket whose exit order was REJECTED or CANCELLED by the broker.

        Called by OrderManager._handle_order_rejected / on_order_update(CANCELLED)
        when Zerodha rejects or cancels an exit order.  Without this, the bracket
        stays permanently dead and the open position has zero SL protection.

        Args:
            symbol: Normalised trading symbol.
            rejected_order_id: The order_id that was rejected/cancelled.
            reason: Human-readable rejection reason for logging.

        Returns:
            True if a bracket was found and reactivated; False otherwise.
        """
        sym = normalize_symbol(symbol)
        with self._lock:
            entry_ids = self._symbol_map.get(sym, [])
            for entry_id in entry_ids:
                bracket = self._brackets.get(entry_id)
                if bracket is None:
                    continue
                if bracket.remaining_quantity <= 0:
                    continue
                # Match by pending_exit_order_id when possible; fall back to any
                # bracket marked exit_executed that still has remaining quantity.
                is_match = bracket.pending_exit_order_id == rejected_order_id or (
                    bracket.exit_executed and bracket.pending_exit_order_id is None
                )
                if not is_match:
                    continue

                # Reactivate for retry on next tick / watchdog cycle.
                bracket.exit_executed = False
                bracket.active = True
                bracket.pending_exit_order_id = None
                # Clear exit cooldown so the very next tick can retry.
                self._exit_cooldowns.pop(entry_id, None)
                LOGGER.critical(
                    "🔁 EXIT ORDER %s REJECTED/CANCELLED for %s (qty=%s, reason=%s) "
                    "— bracket REACTIVATED for retry.",
                    rejected_order_id,
                    sym,
                    bracket.remaining_quantity,
                    reason,
                )
                return True
        return False

    def get_all_brackets_for_symbol(self, symbol: str) -> List[BracketState]:
        """
        Get all brackets (active and inactive) for a given symbol.

        Args:
            symbol: Trading symbol

        Returns:
            List of BracketState objects (may be empty).
        """
        with self._lock:
            entry_ids = self._symbol_map.get(symbol, [])
            brackets = []
            for entry_id in entry_ids:
                bracket = self._brackets.get(entry_id)
                if bracket:
                    brackets.append(bracket)
            return brackets

    def has_active_bracket(self, symbol: str) -> bool:
        """
        Quick check if a symbol has any active bracket.

        This is an alias for is_symbol_managed() for clarity.

        Args:
            symbol: Trading symbol

        Returns:
            True if symbol has at least one active bracket with remaining quantity.
        """
        return self.is_symbol_managed(symbol)
