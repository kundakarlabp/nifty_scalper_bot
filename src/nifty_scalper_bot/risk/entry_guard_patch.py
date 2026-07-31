"""Final risk guard patch for order-entry SSOT enforcement.

The production order path calls RiskManager.check_order() immediately before
broker submission. Daily trade count, structural re-entry and economic edge
failsafes must therefore live here, not only in candidate/status helpers.
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any

from nifty_scalper_bot.risk.net_rr_gate import NetRRResult, evaluate_final_net_rr

_PATCH_APPLIED = False
_ORIGINAL_CHECK_ORDER: Any = None
_REDUCING_INTENTS = {"EXIT", "REDUCE", "FLATTEN", "SQUARE_OFF", "SQUAREOFF"}


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


def _position_symbol(position: Any) -> str:
    return str(getattr(position, "symbol", "") or getattr(position, "tradingsymbol", "") or "").strip()


def _signal_symbol(signal: Any) -> str:
    return str(getattr(signal, "symbol", "") or getattr(signal, "tradingsymbol", "") or "").strip()


def _position_quantity(position: Any) -> int:
    with suppress(Exception):
        return abs(int(float(getattr(position, "quantity", 0) or 0)))
    return 0


def _iter_open_positions(position_manager: Any) -> list[Any]:
    positions = getattr(position_manager, "_positions", None)
    if isinstance(positions, dict):
        return list(positions.values())
    getter = getattr(position_manager, "get_open_positions", None)
    if callable(getter):
        with suppress(Exception):
            return list(getter() or [])
    value = getattr(position_manager, "open_positions", None)
    if callable(value):
        with suppress(Exception):
            return list(value() or [])
    if value is not None and not isinstance(value, str):
        with suppress(Exception):
            return list(value)
    return []


def _is_reducing_order(position_manager: Any, signal: Any) -> bool:
    intent = str(getattr(signal, "intent", "") or "").strip().upper()
    reduce_only = bool(getattr(signal, "reduce_only", False))
    side = str(getattr(signal, "side", "") or getattr(signal, "transaction_type", "") or "").strip().upper()
    symbol = _signal_symbol(signal)
    if intent in _REDUCING_INTENTS or reduce_only:
        return True
    if not symbol or side not in {"BUY", "SELL"}:
        return False
    for position in _iter_open_positions(position_manager):
        if _position_symbol(position) != symbol or _position_quantity(position) <= 0:
            continue
        existing_side = str(getattr(position, "side", "") or "").strip().upper()
        if existing_side == "LONG" and side == "SELL":
            return True
        if existing_side == "SHORT" and side == "BUY":
            return True
    return False


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


def _stop_reentry_block_reason(position_manager: Any, signal: Any) -> str | None:
    resolver = getattr(position_manager, "stop_reentry_block_reason", None)
    if not callable(resolver):
        return None
    with suppress(Exception):
        reason = resolver(signal)
        return str(reason) if reason else None
    return None


def _net_rr_block_reason(signal: Any) -> tuple[str, NetRRResult] | None:
    with suppress(Exception):
        result = evaluate_final_net_rr(signal)
        if result is not None and not result.allowed:
            return (
                f"net reward-risk insufficient: {result.net_rr:.2f}/{result.minimum:.2f}",
                result,
            )
    return None


def _patched_check_order(self: Any, signal: Any, live_enabled: bool) -> tuple[bool, str]:
    # Protective exits/reductions must never be blocked by entry-only limits.
    position_manager = getattr(self, "position_manager", None)
    if position_manager is not None and _is_reducing_order(position_manager, signal):
        return _ORIGINAL_CHECK_ORDER(self, signal, live_enabled)

    if live_enabled and position_manager is not None:
        reentry_reason = _stop_reentry_block_reason(position_manager, signal)
        if reentry_reason is not None:
            self._last_rejection = "STOP_REENTRY_COOLDOWN"
            logger = getattr(self, "_logger", None)
            log = getattr(logger, "warning", None)
            if callable(log):
                log(
                    "RISK_FINAL_GATE_BLOCK reason=%s symbol=%s",
                    reentry_reason,
                    getattr(signal, "symbol", None),
                    extra={
                        "event": "RISK_FINAL_GATE_BLOCK",
                        "reason": reentry_reason,
                        "code": "STOP_REENTRY_COOLDOWN",
                        "symbol": getattr(signal, "symbol", None),
                        "final_order_gate": True,
                    },
                )
            return False, reentry_reason

    if live_enabled:
        net_rr_block = _net_rr_block_reason(signal)
        if net_rr_block is not None:
            reason, result = net_rr_block
            self._last_rejection = "NET_RR_INSUFFICIENT"
            logger = getattr(self, "_logger", None)
            log = getattr(logger, "warning", None)
            if callable(log):
                log(
                    "RISK_FINAL_GATE_BLOCK reason=%s symbol=%s",
                    reason,
                    getattr(signal, "symbol", None),
                    extra={
                        "event": "RISK_FINAL_GATE_BLOCK",
                        "reason": reason,
                        "code": "NET_RR_INSUFFICIENT",
                        "symbol": getattr(signal, "symbol", None),
                        "final_order_gate": True,
                        "net_rr": round(result.net_rr, 4),
                        "min_net_rr": result.minimum,
                        "gross_reward": round(result.gross_reward, 2),
                        "gross_risk": round(result.gross_risk, 2),
                        "net_reward": round(result.net_reward, 2),
                        "net_risk": round(result.net_risk, 2),
                        "target_cost": round(result.target_cost, 2),
                        "stop_cost": round(result.stop_cost, 2),
                        "half_spread": round(result.half_spread, 4),
                    },
                )
            return False, reason

    if live_enabled and not bool(getattr(self, "_breaker_tripped", False)):
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


__all__ = [
    "apply_patches",
    "_daily_limit_block_reason",
    "_is_reducing_order",
    "_net_rr_block_reason",
    "_stop_reentry_block_reason",
]
