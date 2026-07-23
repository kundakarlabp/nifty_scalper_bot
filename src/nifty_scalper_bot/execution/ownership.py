"""File purpose:
    Bind the canonical bracket authority to the canonical order-entry gate.

Key responsibilities:
    - Register the active bracket manager as the unresolved-exit provider.
    - Preserve a compatibility fallback for noncanonical external test doubles.
    - Resolve live-mode consistently before durable bracket-state enforcement.
    - Surface position/reconciliation lifecycle blockers to the native entry gate.

Operational constraints:
    - Production wiring must use the native provider contract, not method replacement.
    - Protective exits must remain executable while new entries are blocked.
    - LIVE executions must never persist bracket state to ephemeral storage.
"""

from __future__ import annotations

from contextlib import suppress
import os
from typing import Any, Mapping, Sequence

from nifty_scalper_bot.execution.runtime_bracket_manager import RuntimeBracketManager

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

        position_manager = _order_manager_position_manager(getattr(self, "order_manager", None))
        if position_manager is None:
            return None

        for method_name in (
            "current_entry_protection_blocker",
            "current_pnl_reconciliation_blocker",
            "current_position_reconciliation_blocker",
            "current_orphan_position_blocker",
            "current_exit_lifecycle_blocker",
        ):
            reason = _call_blocker(position_manager, method_name)
            if reason:
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


__all__ = ["BoundBracketManager"]
from enum import Enum
from nifty_scalper_bot.execution.position_snapshot import BrokerExposureState


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
    """Classify current symbol ownership without mutating brackets or orders.

    This centralizes the narrow orphan/ghost questions used by app and runner so
    future execution corrections do not add another broker-position parser.
    """

    pending_entry = False
    exit_converging = False
    managed = False
    try:
        managed = bool(bracket_manager.is_symbol_managed(symbol))
    except Exception:
        return SymbolLifecycleClassification.UNRESOLVED
    checker = getattr(bracket_manager, "is_exit_converging", None)
    if callable(checker):
        try:
            exit_converging = bool(checker(symbol))
        except Exception:
            return SymbolLifecycleClassification.UNRESOLVED
    if exit_converging:
        return SymbolLifecycleClassification.EXIT_CONVERGING
    for bracket_id in list((getattr(bracket_manager, "_symbol_map", {}) or {}).get(symbol) or []):
        bracket = (getattr(bracket_manager, "_brackets", {}) or {}).get(bracket_id)
        if bracket is not None and not getattr(bracket, "entry_confirmed", False):
            entry_order_id = str(getattr(bracket, "entry_order_id", "") or "")
            tag = str(getattr(bracket, "tag", "") or "")
            if not entry_order_id.startswith("orphan_") and tag != "orphan_recovery":
                pending_entry = True
                break
    if pending_entry:
        return SymbolLifecycleClassification.PENDING_ENTRY
    if managed and local_position_present:
        return SymbolLifecycleClassification.PROTECTED_OPEN
    if managed and not local_position_present:
        if broker_exposure_state in (BrokerExposureState.FLAT, BrokerExposureState.ABSENT):
            return SymbolLifecycleClassification.GHOST_FLAT
        return SymbolLifecycleClassification.UNRESOLVED
    if local_position_present and broker_exposure_state == BrokerExposureState.NONZERO:
        return SymbolLifecycleClassification.TRUE_ORPHAN
    return SymbolLifecycleClassification.UNRESOLVED


__all__ = ["BoundBracketManager", "SymbolLifecycleClassification", "classify_symbol_lifecycle"]
