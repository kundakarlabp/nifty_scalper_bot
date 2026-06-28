from __future__ import annotations

from scripts._execution_patch_utils import (
    assert_parses,
    method_text,
    replace_method,
    replace_once,
)

PATH = "src/nifty_scalper_bot/execution/bracket_core.py"

replace_once(
    PATH,
    "from nifty_scalper_bot.utils.symbols import normalize_symbol\n",
    "from nifty_scalper_bot.utils.symbols import normalize_symbol\n"
    "from nifty_scalper_bot.execution.position_snapshot import (\n"
    "    PositionSnapshot,\n"
    "    PositionSnapshotError,\n"
    "    decode_position_snapshot,\n"
    ")\n",
)
replace_once(
    PATH,
    "    _ledger_release_hook_fired: bool = False\n",
    "    _ledger_release_hook_fired: bool = False\n"
    "    _filled_exit_sync_started_at: float = 0.0\n"
    "    _filled_exit_sync_order_id: str | None = None\n"
    "    _last_exit_reconcile_at: float = 0.0\n",
)
replace_once(
    PATH,
    '            "ledger_release_hook_fired": self._ledger_release_hook_fired,\n',
    '            "ledger_release_hook_fired": self._ledger_release_hook_fired,\n'
    '            "market_escalation_fired": self._market_escalation_fired,\n'
    '            "last_exit_summary_at": self.last_exit_summary_at,\n'
    '            "filled_exit_sync_started_at": self._filled_exit_sync_started_at,\n'
    '            "filled_exit_sync_order_id": self._filled_exit_sync_order_id,\n'
    '            "last_exit_reconcile_at": self._last_exit_reconcile_at,\n',
)

replace_method(
    PATH,
    "BracketManager",
    "__init__",
    '''
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
        parse_float_env(os.getenv("BRACKET_PENDING_ENTRY_RECONCILE_AFTER_SEC"), 5.0),
    )
    self._stale_cleanup_age = 86400
    self._trail_tier1_pct = parse_float_env(os.getenv("TRAIL_TIER1_PCT"), 1.0)
    self._trail_tier2_pct = parse_float_env(os.getenv("TRAIL_TIER2_PCT"), 2.0)
    self._trail_tier3_pct = parse_float_env(os.getenv("TRAIL_TIER3_PCT"), 4.0)
    self._trail_tier4_pct = parse_float_env(os.getenv("TRAIL_TIER4_PCT"), 6.0)
    self._exit_retry_enabled = os.getenv("EXIT_RETRY_ENABLE", "true").strip().lower() in {"1", "true", "yes", "on"}
    self._exit_max_retry_attempts = max(1, parse_int_env(os.getenv("EXIT_MAX_RETRY_ATTEMPTS"), 4))
    self._exit_retry_backoffs = self._parse_exit_backoffs(os.getenv("EXIT_RETRY_BACKOFF_SECONDS", "1,2,5"))
    self._exit_fatal_error_patterns = tuple(
        value.strip().lower()
        for value in os.getenv("EXIT_RETRY_FATAL_ERROR_PATTERNS", "").split(",")
        if value.strip()
    )
    self._exit_reconcile_interval_seconds = max(0.25, parse_float_env(os.getenv("EXIT_POSITION_RECONCILE_INTERVAL_SECONDS"), 1.0))
    self._exit_flat_confirmation_required = os.getenv("EXIT_FLAT_CONFIRMATION_REQUIRED", "true").strip().lower() in {"1", "true", "yes", "on"}
    self._exit_unresolved_escalation_seconds = max(1.0, parse_float_env(os.getenv("EXIT_UNRESOLVED_ESCALATION_SECONDS"), 15.0))
    self._exit_continue_retry_after_escalation = os.getenv("EXIT_CONTINUE_RETRY_AFTER_ESCALATION", "false").strip().lower() in {"1", "true", "yes", "on"}
    self._exit_force_market_on_escalation = os.getenv("EXIT_FORCE_MARKET_ON_ESCALATION", "true").strip().lower() in {"1", "true", "yes", "on"}
    self._exit_protective_order_mode = str(os.getenv("EXIT_PROTECTIVE_ORDER_MODE", "MARKET") or "MARKET").strip().upper()
    self._exit_marketable_limit_slippage_ticks = max(0, parse_int_env(os.getenv("EXIT_MARKETABLE_LIMIT_SLIPPAGE_TICKS"), 5))
    self._exit_marketable_limit_max_slippage_pct = max(0.0, parse_float_env(os.getenv("EXIT_MARKETABLE_LIMIT_MAX_SLIPPAGE_PCT"), 2.0))
    self._exit_fallback_to_market_on_quote_missing = _env_bool("EXIT_FALLBACK_TO_MARKET_ON_QUOTE_MISSING", True)

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
''',
)

replace_method(
    PATH,
    "BracketManager",
    "has_unresolved_exit",
    '''
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
''',
)
replace_method(
    PATH,
    "BracketManager",
    "get_first_unresolved_exit_bracket_id",
    '''
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
''',
)

replace_once(
    PATH,
    "    def _get_storage_path(self) -> Path:\n",
    method_text('''
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
''') + "    def _get_storage_path(self) -> Path:\n",
)

replace_method(
    PATH,
    "BracketManager",
    "_verify_position_closed",
    '''
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
''',
)
replace_method(
    PATH,
    "BracketManager",
    "_position_flat_for_symbol",
    '''
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
''',
)

assert_parses(PATH)
print("patched bracket core part A")
