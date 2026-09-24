"""File purpose:
    Bind the canonical bracket authority to the canonical order-entry gate.

Key responsibilities:
    - Register the active bracket manager as the unresolved-exit provider.
    - Preserve a compatibility fallback for noncanonical external test doubles.
    - Resolve live-mode consistently before durable bracket-state enforcement.
    - Surface position/reconciliation lifecycle blockers to the native entry gate.
    - Keep active virtual brackets supplied with fresh canonical market data.

Operational constraints:
    - Production wiring must use the native provider contract, not method replacement.
    - Protective exits must remain executable while new entries are blocked.
    - LIVE executions must never persist bracket state to ephemeral storage.
    - Market-data recovery must reuse MDM freshness policy and fallback ownership.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import suppress
from enum import Enum
from typing import Any, Mapping, Sequence

import nifty_scalper_bot.execution.market_aware_profit_extension as _profit_extension
from nifty_scalper_bot.execution import bracket_core as _core
from nifty_scalper_bot.execution.position_snapshot import BrokerExposureState
from nifty_scalper_bot.execution.readiness import is_pnl_diagnostic_reason
from nifty_scalper_bot.execution.runtime_bracket_manager import RuntimeBracketManager
from nifty_scalper_bot.utils.symbols import normalize_symbol

_TRUTHY = {"1", "true", "yes", "y", "on", "live"}


def _env_truthy(name: str) -> bool:
    """Return True when an environment variable carries a truthy operator value."""
    return str(os.getenv(name, "") or "").strip().lower() in _TRUTHY


def _running_under_test_harness() -> bool:
    """Return True only for explicit pytest/test harness execution.

    This is deliberately environment-based rather than importing pytest or checking
    installed packages. Production hosts may have pytest installed for validation,
    but they should not be treated as tests unless a test runner marks the process.
    """
    return bool(os.getenv("PYTEST_CURRENT_TEST")) or _env_truthy("NSB_TEST_MODE")


def _order_manager_position_manager(order_manager: Any) -> Any | None:
    for name in ("_position_manager", "position_manager", "positions"):
        value = getattr(order_manager, name, None)
        if value is not None:
            return value
    return None


def _call_blocker(source: Any, method_name: str) -> Any | None:
    method = getattr(source, method_name, None)
    if not callable(method):
        return None
    try:
        return method()
    except TypeError:
        return None


def _call_sequence(source: Any, names: Sequence[str]) -> list[Any]:
    for name in names:
        method = getattr(source, name, None)
        if callable(method):
            try:
                value = method()
            except Exception:
                continue
        else:
            value = getattr(source, name, None)
        if value is None:
            continue
        try:
            return list(value)
        except TypeError:
            continue
    return []


def _block(reason: str, *, source: str, **details: Any) -> dict[str, Any]:
    return {
        "block_reason": str(reason),
        "block_source": source,
        "broker_attempted": False,
        "retryable": False,
        **details,
    }


def _synthetic_position_blocker(position_manager: Any) -> dict[str, Any] | None:
    failures = int(getattr(position_manager, "_consecutive_reconcile_failures", 0) or 0)
    last_error = getattr(position_manager, "_last_reconcile_error", None)
    if failures > 0 or last_error:
        return _block(
            "position_reconciliation_unhealthy",
            source="position_manager_reconciliation_state",
            consecutive_reconcile_failures=failures,
            last_reconcile_error=last_error,
        )

    positions = _call_sequence(
        position_manager,
        ("get_open_positions", "get_all_positions", "open_positions"),
    )
    unmanaged = [
        getattr(position, "symbol", None)
        for position in positions
        if getattr(position, "order_id", None) in (None, "")
    ]
    unmanaged = [str(symbol) for symbol in unmanaged if symbol]
    if unmanaged:
        return _block(
            "broker_synced_unmanaged_position",
            source="position_manager_positions",
            unmanaged_position_count=len(unmanaged),
            unmanaged_symbols=unmanaged[:5],
        )
    return None


class BoundBracketManager(RuntimeBracketManager):
    """Bracket authority that configures the OrderManager native entry gate."""

    def _capture_same_tick_cached_quote(
        self,
        symbol: str,
        ltp: float,
        exchange_ts: float | None,
    ) -> None:
        """Recover executable depth from the cached SSOT without mixing ticks."""
        source = getattr(self, "_market_data", None)
        getter = (
            getattr(source, "get_latest_tick", None) if source is not None else None
        )
        if not callable(getter):
            return
        try:
            cached = getter(symbol)
        except Exception:
            return
        if not isinstance(cached, Mapping):
            return
        try:
            cached_ltp = float(
                cached.get("ltp")
                or cached.get("last_price")
                or cached.get("price")
                or 0.0
            )
            current_ltp = float(ltp)
        except (TypeError, ValueError):
            return
        if (
            cached_ltp <= 0.0
            or current_ltp <= 0.0
            or abs(cached_ltp - current_ltp) > 1e-9
        ):
            return
        if exchange_ts is not None:
            cached_ts = _core.tick_exchange_epoch(cached)
            try:
                current_ts = float(exchange_ts)
            except (TypeError, ValueError):
                return
            if cached_ts is None or abs(float(cached_ts) - current_ts) > 0.001:
                return
        self._capture_exit_quote(normalize_symbol(symbol), cached)

    def on_tick(
        self,
        symbol: str,
        ltp: float,
        exchange_ts: float | None = None,
        *,
        defer_submission: bool = False,
    ) -> None:
        """Preserve executable bid/ask before native bracket tick evaluation."""
        self._capture_same_tick_cached_quote(symbol, ltp, exchange_ts)
        super().on_tick(
            symbol,
            ltp,
            exchange_ts,
            defer_submission=defer_submission,
        )

    def _evaluate_exit_fast(
        self,
        bracket: Any,
        ltp: float,
        *,
        committed_sl: float | None = None,
    ) -> Any:
        """Apply market-aware extension only after canonical FINAL_TP evaluation."""
        action = super()._evaluate_exit_fast(bracket, ltp, committed_sl=committed_sl)
        if (
            not isinstance(action, Mapping)
            or str(action.get("type") or "") != "FINAL_TP"
        ):
            return action
        try:
            return _profit_extension.extend_final_target_if_supported(
                self,
                bracket,
                float(ltp),
                action,
            )
        except Exception as exc:  # noqa: BLE001 - fail closed to canonical FINAL_TP
            _profit_extension.LOGGER.error(
                "PROFIT_EXTENSION_EVALUATION_FAILED symbol=%s error=%s",
                getattr(bracket, "symbol", ""),
                exc,
                extra={
                    "event": "PROFIT_EXTENSION_EVALUATION_FAILED",
                    "symbol": getattr(bracket, "symbol", ""),
                    "error_type": type(exc).__name__,
                },
                exc_info=exc,
            )
            return action

    def _apply_trailing_math(self, bracket: Any) -> bool:
        """Preserve canonical trailing, then optionally tighten profitable trades."""
        canonical_changed = bool(super()._apply_trailing_math(bracket))
        ltp = _profit_extension._positive(getattr(bracket, "last_ltp", None))
        if ltp is None:
            return canonical_changed
        try:
            market_changed = _profit_extension.tighten_market_aware_floor(
                self,
                bracket,
                ltp,
            )
        except (
            Exception
        ) as exc:  # noqa: BLE001 - canonical trailing remains authoritative
            _profit_extension.LOGGER.debug(
                "PROFIT_TIGHTEN_EVALUATION_FAILED symbol=%s error=%s",
                getattr(bracket, "symbol", ""),
                exc,
            )
            market_changed = False
        return canonical_changed or market_changed

    def confirm_entry_fill(
        self,
        order_id: str,
        fill_price: float,
        filled_qty: int | None = None,
    ) -> Any:
        """Capture market baselines only after canonical fill activation succeeds."""
        result = super().confirm_entry_fill(order_id, fill_price, filled_qty)
        bracket = None
        getter = getattr(self, "get_bracket", None)
        if callable(getter):
            with suppress(Exception):
                bracket = getter(order_id)
        if bracket is None or not bool(getattr(bracket, "entry_confirmed", False)):
            return result
        try:
            changed = _profit_extension.capture_entry_market_baseline(self, bracket)
            if changed:
                saver = getattr(self, "save_state", None)
                if callable(saver):
                    saver()
        except (
            Exception
        ) as exc:  # noqa: BLE001 - baseline is optional, protection is not
            _profit_extension.LOGGER.debug(
                "PROFIT_EXTENSION_BASELINE_CAPTURE_FAILED symbol=%s error=%s",
                getattr(bracket, "symbol", ""),
                exc,
            )
        return result

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # State is initialized before the inherited exit watchdog starts. The
        # dedicated market-data watchdog is started only after canonical bracket
        # construction is complete and only when the market-data owner exposes
        # the canonical freshness/recovery contract.
        self._bracket_stale_refresh_at: dict[str, float] = {}
        try:
            refresh_interval = float(
                os.getenv("BRACKET_STALE_REFRESH_MIN_INTERVAL_SEC", "1.0") or 1.0
            )
        except (TypeError, ValueError):
            refresh_interval = 1.0
        self._bracket_stale_refresh_min_interval_seconds = max(0.25, refresh_interval)
        self._bracket_market_data_thread: threading.Thread | None = None
        super().__init__(*args, **kwargs)
        if self._supports_bracket_market_data_watchdog():
            self._bracket_market_data_thread = threading.Thread(
                target=self._bracket_market_data_watchdog_loop,
                name="bracket-market-data-watchdog",
                daemon=True,
            )
            self._bracket_market_data_thread.start()

    def shutdown(self) -> None:
        """Stop canonical bracket workers, including stale-data recovery."""
        super().shutdown()
        worker = self._bracket_market_data_thread
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=1.0)

    def _market_data_freshness_owner(self) -> Any | None:
        """Return canonical MDM behind the DataHub facade when available."""
        source = getattr(self, "_market_data", None)
        if source is None:
            return None
        return getattr(source, "_mdm", None) or source

    def _supports_bracket_market_data_watchdog(self) -> bool:
        """Return whether the bound market-data owner can perform stale recovery."""
        mdm = self._market_data_freshness_owner()
        if mdm is None:
            return False
        return all(
            callable(getattr(mdm, name, None))
            for name in (
                "time_since_last_tick",
                "_ltp_stale_threshold_for_symbol",
                "request_fallback_refresh",
            )
        )

    def _active_bracket_market_data_symbols(self) -> set[str]:
        """Snapshot symbols whose virtual protection still depends on live price."""
        with self._lock:
            symbols = {
                normalize_symbol(bracket.symbol)
                for bracket in self._brackets.values()
                if bool(getattr(bracket, "active", False))
                and bool(getattr(bracket, "entry_confirmed", False))
                and not bool(getattr(bracket, "exit_pending", False))
                and not bool(getattr(bracket, "exit_executed", False))
                and int(getattr(bracket, "remaining_quantity", 0) or 0) > 0
            }
        return {symbol for symbol in symbols if symbol}

    def _refresh_stale_active_brackets_once(self) -> None:
        """Request non-blocking MDM recovery for stale protected instruments."""
        mdm = self._market_data_freshness_owner()
        if mdm is None:
            return
        age_getter = getattr(mdm, "time_since_last_tick", None)
        threshold_getter = getattr(mdm, "_ltp_stale_threshold_for_symbol", None)
        refresher = getattr(mdm, "request_fallback_refresh", None)
        if not (
            callable(age_getter) and callable(threshold_getter) and callable(refresher)
        ):
            return

        active_symbols = self._active_bracket_market_data_symbols()
        for tracked in tuple(self._bracket_stale_refresh_at):
            if tracked not in active_symbols:
                self._bracket_stale_refresh_at.pop(tracked, None)

        now = time.monotonic()
        for symbol in active_symbols:
            try:
                age = age_getter(symbol)
                threshold = float(threshold_getter(symbol) or 0.0)
            except Exception as exc:  # noqa: BLE001 - watchdog must remain alive
                with suppress(Exception):
                    self._log_throttled(
                        "warning",
                        f"bracket_ltp_freshness_error_{symbol}",
                        30.0,
                        "BRACKET_LTP_FRESHNESS_CHECK_FAILED symbol=%s error=%s",
                        symbol,
                        exc,
                    )
                continue

            if threshold <= 0.0:
                continue

            ltp_stale = age is None or float(age) > threshold
            quote = self._exit_quotes.get(symbol)
            depth_age = (
                None if quote is None else max(time.time() - float(quote[2]), 0.0)
            )
            depth_stale = (
                depth_age is None or depth_age > float(self._exit_quote_max_age)
            )
            if not ltp_stale and not depth_stale:
                self._bracket_stale_refresh_at.pop(symbol, None)
                continue

            previous = self._bracket_stale_refresh_at.get(symbol, 0.0)
            if (
                previous > 0.0
                and now - previous < self._bracket_stale_refresh_min_interval_seconds
            ):
                continue
            self._bracket_stale_refresh_at[symbol] = now

            ltp_dispatched = False
            depth_dispatched = False
            refresh_error: Exception | None = None
            try:
                if ltp_stale:
                    ltp_dispatched = bool(
                        refresher(symbol, reason="bracket_ltp_stale")
                    )
                depth_refresher = getattr(mdm, "request_depth_refresh", None)
                if depth_stale and callable(depth_refresher):
                    depth_dispatched = bool(
                        depth_refresher(symbol, reason="bracket_depth_stale")
                    )
            except Exception as exc:  # noqa: BLE001 - retry on the next interval
                refresh_error = exc

            if ltp_stale:
                age_label = "missing" if age is None else f"{float(age):.3f}"
                with suppress(Exception):
                    self._log_throttled(
                        "warning",
                        f"bracket_ltp_stale_{symbol}",
                        5.0,
                        (
                            "BRACKET_LTP_STALE symbol=%s age_s=%s threshold_s=%.3f "
                            "fallback_dispatched=%s error=%s"
                        ),
                        symbol,
                        age_label,
                        threshold,
                        ltp_dispatched,
                        str(refresh_error) if refresh_error is not None else "none",
                    )
            if depth_stale:
                depth_label = (
                    "missing" if depth_age is None else f"{float(depth_age):.3f}"
                )
                with suppress(Exception):
                    self._log_throttled(
                        "warning",
                        f"bracket_depth_stale_{symbol}",
                        5.0,
                        (
                            "BRACKET_DEPTH_STALE symbol=%s age_s=%s threshold_s=%.3f "
                            "recovery_dispatched=%s error=%s"
                        ),
                        symbol,
                        depth_label,
                        float(self._exit_quote_max_age),
                        depth_dispatched,
                        str(refresh_error) if refresh_error is not None else "none",
                    )

    def _bracket_market_data_watchdog_loop(self) -> None:
        """Backstop virtual exits when per-symbol market-data delivery goes silent."""
        while bool(getattr(self, "_running", False)):
            try:
                self._refresh_stale_active_brackets_once()
            except Exception as exc:  # noqa: BLE001 - liveness is safety-critical
                with suppress(Exception):
                    self._log_throttled(
                        "error",
                        "bracket_market_data_watchdog_error",
                        30.0,
                        "BRACKET_MARKET_DATA_WATCHDOG_ERROR error=%s",
                        exc,
                    )
            time.sleep(0.25)

    def _is_live_execution(self) -> bool:
        """Return True only when real broker-live order execution is enabled."""
        if _running_under_test_harness():
            return False

        checker = getattr(getattr(self, "order_manager", None), "is_live_mode", None)
        if callable(checker):
            with suppress(Exception):
                return bool(checker())

        mode = str(os.getenv("EXECUTION_MODE", "SHADOW") or "SHADOW").strip().upper()
        live_enabled = _env_truthy("ENABLE_LIVE") or _env_truthy("ENABLE_LIVE_TRADING")
        shadow_or_paper = (
            _env_truthy("SHADOW_MODE")
            or _env_truthy("PAPER_MODE")
            or _env_truthy("PAPER__ENABLED")
        )
        return mode == "LIVE" and live_enabled and not shadow_or_paper

    def _strict_ledger_release_required(self) -> bool:
        """Use the same live predicate for ledger release and state durability."""
        return self._is_live_execution()

    def _install_unresolved_exit_entry_guard(self) -> None:
        order_manager = getattr(self, "order_manager", None)
        setter = getattr(order_manager, "set_unresolved_exit_provider", None)
        if callable(setter):
            setter(self)
            with suppress(Exception):
                from nifty_scalper_bot.execution import bracket_core

                bracket_core.LOGGER.info(
                    "UNRESOLVED_EXIT_NATIVE_GATE_BOUND",
                    extra={"event": "UNRESOLVED_EXIT_NATIVE_GATE_BOUND"},
                )
            return
        super()._install_unresolved_exit_entry_guard()

    def current_entry_blocker(self) -> Mapping[str, Any] | None:
        """Return the first live-safety blocker for new entries.

        This keeps the bracket manager as the single provider registered with
        RuntimeOrderManager while allowing the native entry gate to consume
        position-manager safety state: unprotected fills, P&L mismatch,
        unresolved terminal exits, broker/local reconciliation uncertainty, and
        broker-synced positions that do not yet have local order ownership.
        """

        checker = getattr(self, "has_unresolved_exit", None)
        try:
            if callable(checker) and bool(checker()):
                bracket_id = None
                getter = getattr(self, "get_first_unresolved_exit_bracket_id", None)
                if callable(getter):
                    with suppress(Exception):
                        bracket_id = getter()
                return _block(
                    "unresolved_exit_position",
                    source="bracket_manager",
                    bracket_id=bracket_id,
                )
        except Exception as exc:  # noqa: BLE001 - fail closed
            return _block(
                "entry_blocker_provider_error",
                source="bracket_manager",
                provider_error=f"{type(exc).__name__}: {exc}",
            )

        position_manager = _order_manager_position_manager(
            getattr(self, "order_manager", None)
        )
        if position_manager is None:
            return None

        post_pnl_methods = (
            "current_position_reconciliation_blocker",
            "current_orphan_position_blocker",
            "current_exit_lifecycle_blocker",
        )
        required_after_pnl = (
            *post_pnl_methods,
            "unresolved_terminal_summary",
            "get_open_positions",
        )

        for method_name in (
            "current_entry_protection_blocker",
            "current_pnl_reconciliation_blocker",
            *post_pnl_methods,
        ):
            reason = _call_blocker(position_manager, method_name)
            if not reason:
                continue
            if (
                method_name == "current_pnl_reconciliation_blocker"
                and is_pnl_diagnostic_reason(reason)
                and all(
                    callable(getattr(position_manager, name, None))
                    for name in required_after_pnl
                )
            ):
                continue
            return _block(str(reason), source=method_name)

        summary_getter = getattr(position_manager, "unresolved_terminal_summary", None)
        if callable(summary_getter):
            with suppress(Exception):
                summary = summary_getter()
                if isinstance(summary, Mapping) and int(summary.get("count") or 0) > 0:
                    return _block(
                        "unresolved_terminal_order",
                        source="unresolved_terminal_summary",
                        unresolved_terminal_count=int(summary.get("count") or 0),
                        oldest_unresolved_terminal_age_s=summary.get("oldest_age_s"),
                    )

        return _synthetic_position_blocker(position_manager)


class SymbolLifecycleClassification(str, Enum):
    """Read-only symbol lifecycle classification for reconciliation callers."""

    PROTECTED_OPEN = "protected_open"
    PENDING_ENTRY = "pending_entry"
    EXIT_CONVERGING = "exit_converging"
    TRUE_ORPHAN = "true_orphan"
    GHOST_FLAT = "ghost_flat"
    UNRESOLVED = "unresolved"


def classify_symbol_lifecycle(
    symbol: str,
    *,
    bracket_manager: Any,
    local_position_present: bool,
    broker_exposure_state: BrokerExposureState,
) -> SymbolLifecycleClassification:
    """Classify current symbol ownership without mutating brackets or orders."""

    normalized = normalize_symbol(symbol) or str(symbol or "").strip().upper()
    snapshot_getter = getattr(bracket_manager, "get_symbol_lifecycle_snapshot", None)
    if callable(snapshot_getter):
        try:
            snapshot = snapshot_getter(normalized)
        except Exception:
            return SymbolLifecycleClassification.UNRESOLVED
        if bool(snapshot.get("exit_converging")):
            return SymbolLifecycleClassification.EXIT_CONVERGING
        if bool(snapshot.get("pending_entry")):
            return SymbolLifecycleClassification.PENDING_ENTRY
        managed = bool(snapshot.get("managed"))
    else:
        try:
            managed = bool(bracket_manager.is_symbol_managed(normalized))
        except Exception:
            return SymbolLifecycleClassification.UNRESOLVED
        checker = getattr(bracket_manager, "is_exit_converging", None)
        if callable(checker):
            try:
                if bool(checker(normalized)):
                    return SymbolLifecycleClassification.EXIT_CONVERGING
            except Exception:
                return SymbolLifecycleClassification.UNRESOLVED

    if managed and local_position_present:
        return SymbolLifecycleClassification.PROTECTED_OPEN
    if managed and not local_position_present:
        if broker_exposure_state in (
            BrokerExposureState.FLAT,
            BrokerExposureState.ABSENT,
        ):
            return SymbolLifecycleClassification.GHOST_FLAT
        return SymbolLifecycleClassification.UNRESOLVED
    if local_position_present and broker_exposure_state == BrokerExposureState.NONZERO:
        return SymbolLifecycleClassification.TRUE_ORPHAN
    return SymbolLifecycleClassification.UNRESOLVED


__all__ = [
    "BoundBracketManager",
    "SymbolLifecycleClassification",
    "classify_symbol_lifecycle",
]
