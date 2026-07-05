"""File purpose:
    Bind the canonical bracket authority to the canonical order-entry gate.

Key responsibilities:
    - Register the active bracket manager as the unresolved-exit provider.
    - Surface bracket, protection, P&L and reconciliation blockers to the gate.
    - Preserve a compatibility fallback for noncanonical external test doubles.

Operational constraints:
    - Production wiring must use the native provider contract, not method replacement.
    - Protective exits must remain executable while new entries are blocked.
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any, Mapping, Sequence

from nifty_scalper_bot.execution.runtime_bracket_manager import RuntimeBracketManager


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