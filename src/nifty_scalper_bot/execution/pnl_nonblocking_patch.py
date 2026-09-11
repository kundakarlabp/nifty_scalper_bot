"""Keep P&L reconciliation observable without allowing it to block entries.

P&L/baseline state remains available through PositionManager diagnostics and broker
reconciliation.  The live readiness and canonical bracket entry gates deliberately
ignore only the three P&L-reconciliation reasons below.  Position ownership,
protection, broker reconciliation, emergency, risk, and daily-loss controls are
unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from typing import Any

_PNL_DIAGNOSTIC_REASONS = {
    "pnl_baseline_uninitialized",
    "pnl_session_date_unverified",
    "pnl_reconciliation_mismatch",
}
_PATCH_APPLIED = False
_ORIGINAL_NORMALIZE_READINESS: Any = None
_ORIGINAL_BOUND_ENTRY_BLOCKER: Any = None


def _reason_token(value: object) -> str:
    text = str(value or "").strip()
    if ":" in text:
        text = text.split(":", 1)[-1]
    return text


def _is_pnl_diagnostic(value: object) -> bool:
    return _reason_token(value) in _PNL_DIAGNOSTIC_REASONS


def _normalize_readiness_without_pnl_blocking(
    blockers: list[str] | tuple[str, ...] | set[str],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Drop P&L-only diagnostics before the canonical readiness decision."""
    filtered = [item for item in (blockers or []) if not _is_pnl_diagnostic(item)]
    return _ORIGINAL_NORMALIZE_READINESS(filtered, *args, **kwargs)


def _bound_entry_blocker_without_pnl(self: Any) -> Mapping[str, Any] | None:
    """Skip a P&L-only blocker and continue checking later safety blockers."""
    blocker = _ORIGINAL_BOUND_ENTRY_BLOCKER(self)
    if not isinstance(blocker, Mapping):
        return blocker
    if not _is_pnl_diagnostic(blocker.get("block_reason")):
        return blocker

    # The original blocker order checks entry protection before P&L.  Re-enter
    # only the checks that follow P&L so a diagnostic P&L state cannot mask a
    # real position/lifecycle safety blocker.
    from nifty_scalper_bot.execution.ownership import (
        _block,
        _call_blocker,
        _order_manager_position_manager,
        _synthetic_position_blocker,
    )

    position_manager = _order_manager_position_manager(
        getattr(self, "order_manager", None)
    )
    if position_manager is None:
        return None

    for method_name in (
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


def apply_patches() -> None:
    """Install the non-blocking P&L policy once for the canonical live path."""
    global _PATCH_APPLIED, _ORIGINAL_NORMALIZE_READINESS, _ORIGINAL_BOUND_ENTRY_BLOCKER
    if _PATCH_APPLIED:
        return

    from nifty_scalper_bot.execution import readiness
    from nifty_scalper_bot.execution.ownership import BoundBracketManager

    if getattr(BoundBracketManager, "_pnl_nonblocking_patch", False):
        _PATCH_APPLIED = True
        return

    _ORIGINAL_NORMALIZE_READINESS = readiness.normalize_readiness_blockers
    _ORIGINAL_BOUND_ENTRY_BLOCKER = BoundBracketManager.current_entry_blocker

    readiness.normalize_readiness_blockers = _normalize_readiness_without_pnl_blocking
    BoundBracketManager.current_entry_blocker = _bound_entry_blocker_without_pnl
    BoundBracketManager._pnl_nonblocking_patch = True
    _PATCH_APPLIED = True


__all__ = ["apply_patches"]
