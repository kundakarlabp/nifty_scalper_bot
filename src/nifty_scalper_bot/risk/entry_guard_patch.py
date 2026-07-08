"""Final risk guard patch for order-entry SSOT enforcement.

The production order path calls RiskManager.check_order() immediately before
broker submission.  Daily trade count and open-position failsafes must therefore
live here, not only in read-only/status helpers.
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any

_PATCH_APPLIED = False
_ORIGINAL_CHECK_ORDER: Any = None


def _call_count(owner: Any, *names: str) -> int:
    for name in names:
        value = getattr(owner, name, None)
        if callable(value):
            with suppress(Exception):
                return int(value() or 0)
        elif value is not None:
            with suppress(Exception):
                return int(value or 0)
    return 0


def _open_position_count(position_manager: Any) -> int:
    getter = getattr(position_manager, "get_open_positions", None)
    if callable(getter):
        with suppress(Exception):
            return len(getter() or [])
    value = getattr(position_manager, "open_positions", None)
    if callable(value):
        with suppress(Exception):
            return len(value() or [])
    if value is not None and not isinstance(value, str):
        with suppress(Exception):
            return len(value)
    return 0


def _daily_limit_block_reason(manager: Any) -> tuple[str, str] | None:
    settings = getattr(manager, "settings", None)
    position_manager = getattr(manager, "position_manager", None)
    if settings is None or position_manager is None:
        return None

    max_trades = int(getattr(settings, "max_trades_per_day", 0) or 0)
    if max_trades > 0:
        trades_today = _call_count(
            position_manager,
            "trades_today",
            "daily_trade_count",
            "trade_count_today",
        )
        if trades_today >= max_trades:
            return (
                f"max_trades_per_day breached: {trades_today}/{max_trades}",
                f"MAX_TRADES:{trades_today}/{max_trades}",
            )

    max_open = int(getattr(settings, "max_open_positions", 0) or 0)
    if max_open > 0:
        open_positions = _open_position_count(position_manager)
        if open_positions > max_open:
            return (
                f"max_open_positions breached: {open_positions}/{max_open}",
                f"MAX_OPEN:{open_positions}/{max_open}",
            )
    return None


def _patched_check_order(self: Any, signal: Any, live_enabled: bool) -> tuple[bool, str]:
    blocker = _daily_limit_block_reason(self)
    if blocker is not None:
        reason, code = blocker
        self._last_rejection = code
        trip = getattr(self, "_trip_breaker", None)
        if callable(trip):
            with suppress(Exception):
                trip(reason)
        logger = getattr(self, "_logger", None)
        log = getattr(logger, "critical", None)
        if callable(log):
            log(
                "RISK_FINAL_GATE_BLOCK reason=%s symbol=%s",
                reason,
                getattr(signal, "symbol", None),
                extra={
                    "event": "RISK_FINAL_GATE_BLOCK",
                    "reason": reason,
                    "code": code,
                    "symbol": getattr(signal, "symbol", None),
                    "final_order_gate": True,
                },
            )
        return False, reason
    return _ORIGINAL_CHECK_ORDER(self, signal, live_enabled)


def apply_patches() -> None:
    global _PATCH_APPLIED, _ORIGINAL_CHECK_ORDER
    if _PATCH_APPLIED:
        return
    from nifty_scalper_bot.risk.risk_manager import RiskManager

    if getattr(RiskManager, "_entry_guard_patch", False):
        _PATCH_APPLIED = True
        return
    _ORIGINAL_CHECK_ORDER = RiskManager.check_order
    RiskManager.check_order = _patched_check_order
    RiskManager._entry_guard_patch = True
    _PATCH_APPLIED = True


__all__ = ["apply_patches", "_daily_limit_block_reason"]
