"""Pure policy helpers for native final risk admission."""

from __future__ import annotations

import os
from contextlib import suppress
from typing import Any

from nifty_scalper_bot.risk.net_rr_gate import NetRRResult, evaluate_final_net_rr

REDUCING_INTENTS = {"EXIT", "REDUCE", "FLATTEN", "SQUARE_OFF", "SQUAREOFF"}
_TRUE_VALUES = {"1", "true", "yes", "y", "on"}


def _env_true(name: str) -> bool:
    return str(os.getenv(name, "") or "").strip().lower() in _TRUE_VALUES


def _real_broker_live(live_enabled: bool) -> bool:
    """Return whether this is a real broker-live submission."""
    return bool(live_enabled) and not any(
        _env_true(name)
        for name in ("BROKER_SIMULATION", "PAPER_MODE", "PAPER__ENABLED", "SHADOW_MODE")
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


def _open_positions(position_manager: Any) -> list[Any]:
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


def _open_position_count(position_manager: Any) -> int:
    return len(_open_positions(position_manager))


def _is_reducing_order(position_manager: Any, signal: Any) -> bool:
    intent = str(getattr(signal, "intent", "") or "").strip().upper()
    if intent in REDUCING_INTENTS or bool(getattr(signal, "reduce_only", False)):
        return True
    side = str(
        getattr(signal, "side", "")
        or getattr(signal, "transaction_type", "")
        or getattr(signal, "action", "")
        or ""
    ).strip().upper()
    symbol = str(
        getattr(signal, "symbol", "")
        or getattr(signal, "tradingsymbol", "")
        or ""
    ).strip()
    if not symbol or side not in {"BUY", "SELL"}:
        return False
    for position in _open_positions(position_manager):
        position_symbol = str(
            getattr(position, "symbol", "")
            or getattr(position, "tradingsymbol", "")
            or ""
        ).strip()
        with suppress(TypeError, ValueError):
            quantity = abs(int(float(getattr(position, "quantity", 0) or 0)))
            existing_side = str(getattr(position, "side", "") or "").upper()
            if position_symbol == symbol and quantity > 0:
                if (existing_side, side) in {("LONG", "SELL"), ("SHORT", "BUY")}:
                    return True
    return False


def _daily_limit_block_reason(manager: Any) -> tuple[str, str] | None:
    settings = getattr(manager, "settings", None)
    position_manager = getattr(manager, "position_manager", None)
    if settings is None or position_manager is None:
        return None
    max_trades = int(getattr(settings, "max_trades_per_day", 0) or 0)
    if max_trades > 0:
        count = _call_count(
            position_manager, "trades_today", "daily_trade_count", "trade_count_today"
        )
        if count >= max_trades:
            return (
                f"max_trades_per_day breached: {count}/{max_trades}",
                f"MAX_TRADES:{count}/{max_trades}",
            )
    max_open = int(getattr(settings, "max_open_positions", 0) or 0)
    if max_open > 0:
        count = _open_position_count(position_manager)
        if count >= max_open:
            return (
                f"max_open_positions breached: {count}/{max_open}",
                f"MAX_OPEN:{count}/{max_open}",
            )
    return None


def _daily_limit_should_trip_breaker(manager: Any, code: str) -> bool:
    normalized = str(code or "")
    if normalized.startswith("MAX_TRADES:"):
        return False
    if not normalized.startswith("MAX_OPEN:"):
        return True
    max_open = int(
        getattr(getattr(manager, "settings", None), "max_open_positions", 0) or 0
    )
    position_manager = getattr(manager, "position_manager", None)
    return bool(
        max_open > 0
        and position_manager is not None
        and _open_position_count(position_manager) > max_open
    )


def _stop_reentry_block_reason(position_manager: Any, signal: Any) -> str | None:
    resolver = getattr(position_manager, "stop_reentry_block_reason", None)
    if callable(resolver):
        with suppress(Exception):
            reason = resolver(signal)
            return str(reason) if reason else None
    return None


def _net_rr_block_reason(signal: Any) -> tuple[str, NetRRResult] | None:
    with suppress(Exception):
        result = evaluate_final_net_rr(signal)
        if result is not None and not result.allowed:
            return (
                "net reward-risk insufficient: "
                f"{result.net_rr:.2f}/{result.minimum:.2f}",
                result,
            )
    return None


def _daily_risk_budget_state(manager: Any) -> tuple[float | None, float, float]:
    """Return remaining day-loss budget, current loss, and configured cap."""
    switches = getattr(manager, "_switches", None)
    if switches is None:
        return None, 0.0, 0.0
    with suppress(TypeError, ValueError):
        cap = max(float(getattr(switches, "max_day_loss", 0.0) or 0.0), 0.0)
        if cap <= 0.0:
            return None, 0.0, 0.0
        reader = getattr(switches, "day_loss", None)
        if not callable(reader):
            return 0.0, 0.0, cap
        try:
            loss = max(float(reader() or 0.0), 0.0)
        except Exception:
            return 0.0, 0.0, cap
        return max(cap - loss, 0.0), loss, cap
    return None, 0.0, 0.0


def _signal_stop_risk(signal: Any) -> float | None:
    with suppress(TypeError, ValueError):
        quantity = abs(int(float(getattr(signal, "quantity", 0) or 0)))
        price = float(getattr(signal, "price", 0.0) or 0.0)
        stop_loss = getattr(signal, "stop_loss", None)
        if quantity > 0 and price > 0.0 and stop_loss is not None:
            return abs(price - float(stop_loss)) * quantity
    return None


__all__ = [
    "_daily_limit_block_reason",
    "_daily_limit_should_trip_breaker",
    "_daily_risk_budget_state",
    "_is_reducing_order",
    "_net_rr_block_reason",
    "_real_broker_live",
    "_signal_stop_risk",
    "_stop_reentry_block_reason",
]
