"""Final risk guard patch for order-entry SSOT enforcement.

The production order path calls RiskManager.check_order() immediately before
broker submission. Daily trade count, structural re-entry and economic edge
failsafes must therefore live here, not only in candidate/status helpers.
"""

from __future__ import annotations

import os
from contextlib import suppress
from typing import Any

from nifty_scalper_bot.risk.net_rr_gate import NetRRResult, evaluate_final_net_rr

_PATCH_APPLIED = False
_ORIGINAL_CHECK_ORDER: Any = None
_ORIGINAL_SUGGEST_POSITION_SIZE: Any = None
_REDUCING_INTENTS = {"EXIT", "REDUCE", "FLATTEN", "SQUARE_OFF", "SQUAREOFF"}
_TRUE_VALUES = {"1", "true", "yes", "y", "on"}


def _env_true(name: str) -> bool:
    return str(os.getenv(name, "") or "").strip().lower() in _TRUE_VALUES


def _real_broker_live(live_enabled: bool) -> bool:
    """Apply economic rejection only to real broker-live entry submission."""
    if not live_enabled:
        return False
    return not (
        _env_true("BROKER_SIMULATION")
        or _env_true("PAPER_MODE")
        or _env_true("PAPER__ENABLED")
        or _env_true("SHADOW_MODE")
    )


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
    return str(
        getattr(position, "symbol", "")
        or getattr(position, "tradingsymbol", "")
        or ""
    ).strip()


def _signal_symbol(signal: Any) -> str:
    return str(
        getattr(signal, "symbol", "")
        or getattr(signal, "tradingsymbol", "")
        or ""
    ).strip()


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
    side = str(
        getattr(signal, "side", "")
        or getattr(signal, "transaction_type", "")
        or getattr(signal, "action", "")
        or ""
    ).strip().upper()
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


def _confidence_value(value: Any) -> float:
    if value is None:
        return 1.0
    with suppress(TypeError, ValueError):
        return max(0.0, min(1.0, float(value)))
    return 0.0


def _sizing_risk_distance(
    *, side: str, price: float, stop_loss: Any, atr: Any
) -> float:
    effective_stop = stop_loss
    if effective_stop is None:
        default_sl_pct = float(os.getenv("DEFAULT_SL_PCT", "2.0"))
        effective_stop = (
            price * (1 - default_sl_pct / 100.0)
            if str(side).strip().upper() == "BUY"
            else price * (1 + default_sl_pct / 100.0)
        )
    with suppress(TypeError, ValueError):
        distance = abs(float(price) - float(effective_stop))
        if atr is not None:
            with suppress(TypeError, ValueError):
                distance = max(distance, abs(float(atr)))
        return max(distance, float(price) * 0.005)
    return 0.0


def _daily_risk_budget_state(manager: Any) -> tuple[float | None, float, float]:
    """Return remaining day-loss budget, current day loss and configured cap."""
    switches = getattr(manager, "_switches", None)
    if switches is None:
        return None, 0.0, 0.0
    with suppress(TypeError, ValueError):
        max_day_loss = max(float(getattr(switches, "max_day_loss", 0.0) or 0.0), 0.0)
        if max_day_loss <= 0.0:
            return None, 0.0, 0.0
        day_loss_reader = getattr(switches, "day_loss", None)
        if not callable(day_loss_reader):
            return 0.0, 0.0, max_day_loss
        try:
            current_day_loss = max(float(day_loss_reader() or 0.0), 0.0)
        except Exception:
            return 0.0, 0.0, max_day_loss
        return max(max_day_loss - current_day_loss, 0.0), current_day_loss, max_day_loss
    return None, 0.0, 0.0


def _signal_stop_risk(signal: Any) -> float | None:
    """Return deterministic stop-risk for a normalized entry signal."""
    with suppress(TypeError, ValueError):
        quantity = abs(int(float(getattr(signal, "quantity", 0) or 0)))
        price = float(getattr(signal, "price", 0.0) or 0.0)
        stop_loss = getattr(signal, "stop_loss", None)
        if quantity <= 0 or price <= 0.0 or stop_loss is None:
            return None
        return abs(price - float(stop_loss)) * quantity
    return None


def _patched_suggest_position_size(
    self: Any,
    *,
    side: str,
    price: float,
    stop_loss: float | None,
    atr: float | None,
    requested_quantity: int,
    confidence: float | None = None,
    symbol: str | None = None,
) -> int:
    """Preserve existing sizing, then enforce percentage and remaining-day caps."""
    confidence_value = _confidence_value(confidence)
    if confidence_value <= 0.0:
        logger = getattr(self, "_logger", None)
        log = getattr(logger, "info", None)
        if callable(log):
            log(
                "RISK_SIZING_BLOCKED_ZERO_CONFIDENCE symbol=%s confidence=%s",
                symbol,
                confidence,
                extra={
                    "event": "RISK_SIZING_BLOCKED_ZERO_CONFIDENCE",
                    "symbol": symbol,
                    "confidence": confidence,
                },
            )
        return 0

    quantity = int(
        _ORIGINAL_SUGGEST_POSITION_SIZE(
            self,
            side=side,
            price=price,
            stop_loss=stop_loss,
            atr=atr,
            requested_quantity=requested_quantity,
            confidence=confidence,
            symbol=symbol,
        )
        or 0
    )
    if quantity <= 0:
        return 0

    with suppress(TypeError, ValueError, AttributeError):
        balance = float(getattr(self, "account_balance", 0.0) or 0.0)
        if balance <= 0.0:
            balance = float(getattr(self, "_cached_balance", 0.0) or 0.0)
        risk_pct = float(
            getattr(getattr(self, "settings", None), "per_trade_risk_pct", 0.0)
            or 0.0
        )
        allowed_risk = balance * (risk_pct / 100.0)
        remaining_day_budget, current_day_loss, max_day_loss = _daily_risk_budget_state(
            self
        )
        effective_allowed_risk = allowed_risk
        if remaining_day_budget is not None:
            effective_allowed_risk = min(allowed_risk, remaining_day_budget)
        distance = _sizing_risk_distance(
            side=side,
            price=float(price),
            stop_loss=stop_loss,
            atr=atr,
        )
        if balance > 0.0 and allowed_risk > 0.0 and distance > 0.0:
            try:
                lot_size = int(self._resolve_lot_size(symbol))
            except Exception:
                lot_size = int(os.getenv("DEFAULT_LOT_SIZE", "25"))
            if lot_size <= 0:
                return 0
            max_lots = int(effective_allowed_risk // (distance * lot_size))
            safe_quantity = max(0, max_lots * lot_size)
            if quantity > safe_quantity:
                logger = getattr(self, "_logger", None)
                log = getattr(logger, "warning", None)
                daily_budget_tighter = (
                    remaining_day_budget is not None
                    and remaining_day_budget < allowed_risk
                )
                event = (
                    "RISK_SIZING_CLAMPED_TO_REMAINING_DAY_BUDGET"
                    if daily_budget_tighter
                    else "RISK_SIZING_CLAMPED_TO_PERCENT_CAP"
                )
                if callable(log):
                    log(
                        "%s symbol=%s requested_sized=%s safe_qty=%s allowed_risk=%.2f effective_risk=%.2f risk_distance=%.4f",
                        event,
                        symbol,
                        quantity,
                        safe_quantity,
                        allowed_risk,
                        effective_allowed_risk,
                        distance,
                        extra={
                            "event": event,
                            "symbol": symbol,
                            "sized_quantity": quantity,
                            "safe_quantity": safe_quantity,
                            "allowed_risk": allowed_risk,
                            "effective_allowed_risk": effective_allowed_risk,
                            "risk_distance": distance,
                            "per_trade_risk_pct": risk_pct,
                            "remaining_day_budget": remaining_day_budget,
                            "current_day_loss": current_day_loss,
                            "max_day_loss": max_day_loss,
                        },
                    )
                return safe_quantity
    return quantity


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

    if _real_broker_live(live_enabled):
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

    allowed, reason = _ORIGINAL_CHECK_ORDER(self, signal, live_enabled)
    if not allowed:
        return allowed, reason

    if live_enabled:
        remaining_day_budget, current_day_loss, max_day_loss = _daily_risk_budget_state(
            self
        )
        prospective_stop_risk = _signal_stop_risk(signal)
        if (
            remaining_day_budget is not None
            and prospective_stop_risk is not None
            and prospective_stop_risk > remaining_day_budget
        ):
            reason = (
                "remaining daily loss budget insufficient: "
                f"{prospective_stop_risk:.2f}/{remaining_day_budget:.2f}"
            )
            self._last_rejection = "DAILY_RISK_BUDGET"
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
                        "code": "DAILY_RISK_BUDGET",
                        "symbol": getattr(signal, "symbol", None),
                        "final_order_gate": True,
                        "prospective_stop_risk": prospective_stop_risk,
                        "remaining_day_budget": remaining_day_budget,
                        "current_day_loss": current_day_loss,
                        "max_day_loss": max_day_loss,
                    },
                )
            return False, reason
    return allowed, reason


def apply_patches() -> None:
    global _PATCH_APPLIED, _ORIGINAL_CHECK_ORDER, _ORIGINAL_SUGGEST_POSITION_SIZE
    if _PATCH_APPLIED:
        return
    from nifty_scalper_bot.risk.risk_manager import RiskManager

    if getattr(RiskManager, "_entry_guard_patch", False):
        _PATCH_APPLIED = True
        return
    _ORIGINAL_CHECK_ORDER = RiskManager.check_order
    _ORIGINAL_SUGGEST_POSITION_SIZE = RiskManager.suggest_position_size
    RiskManager.check_order = _patched_check_order
    RiskManager.suggest_position_size = _patched_suggest_position_size
    RiskManager._entry_guard_patch = True
    _PATCH_APPLIED = True


__all__ = [
    "apply_patches",
    "_confidence_value",
    "_daily_limit_block_reason",
    "_daily_risk_budget_state",
    "_is_reducing_order",
    "_net_rr_block_reason",
    "_patched_suggest_position_size",
    "_real_broker_live",
    "_signal_stop_risk",
    "_sizing_risk_distance",
    "_stop_reentry_block_reason",
]
