"""Pure helpers for broker P&L evidence and session diagnostics.

This module has no runtime installation side effects. Broker adapters own broker
I/O; PositionManager owns trading/accounting state.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from typing import Any

from nifty_scalper_bot.utils.market_hours import (
    get_runtime_market_mode,
    post_market_broker_refresh_seconds,
    post_market_quiet_mode_enabled,
)
from nifty_scalper_bot.utils.symbols import is_strategy_instrument

_DEFAULT_REFRESH_SECONDS = 15.0
_DEFAULT_MAX_AGE_SECONDS = 120.0
_MATCH_TOLERANCE_RUPEES = 1.0
_EPS = 1e-6


def _finite_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _refresh_seconds() -> float:
    raw = os.getenv("BROKER_PNL_REFRESH_SECONDS", str(_DEFAULT_REFRESH_SECONDS))
    try:
        return max(2.0, min(float(raw), 300.0))
    except (TypeError, ValueError):
        return _DEFAULT_REFRESH_SECONDS


def _effective_refresh_seconds() -> float:
    base = _refresh_seconds()
    if not post_market_quiet_mode_enabled():
        return base
    try:
        mode = get_runtime_market_mode()
    except Exception:
        return base
    if mode not in {"POST_MARKET", "HOLIDAY"}:
        return base
    return max(base, post_market_broker_refresh_seconds())


def _max_age_seconds() -> float:
    raw = os.getenv("BROKER_PNL_MAX_AGE_SECONDS", str(_DEFAULT_MAX_AGE_SECONDS))
    try:
        return max(5.0, min(float(raw), 900.0))
    except (TypeError, ValueError):
        return _DEFAULT_MAX_AGE_SECONDS


def _extract_account_m2m(payload: object) -> tuple[float | None, float | None]:
    """Extract Zerodha account M2M from a raw margin payload."""

    if not isinstance(payload, Mapping):
        return None, None
    utilised = payload.get("utilised")
    if not isinstance(utilised, Mapping):
        utilised = payload.get("utilized")
    if not isinstance(utilised, Mapping):
        utilised = payload

    realized = None
    unrealized = None
    for key in ("m2m_realised", "m2m_realized"):
        if key in utilised:
            realized = _finite_float(utilised.get(key))
            break
    for key in ("m2m_unrealised", "m2m_unrealized"):
        if key in utilised:
            unrealized = _finite_float(utilised.get(key))
            break
    return realized, unrealized


def _strategy_symbol(record: Mapping[str, Any]) -> str:
    raw_symbol = (
        record.get("tradingsymbol")
        or record.get("symbol")
        or record.get("instrument")
        or ""
    )
    text = str(raw_symbol).strip().upper()
    exchange = str(record.get("exchange") or "").strip().upper()
    if text and ":" not in text and exchange:
        return f"{exchange}:{text}"
    return text


def _strategy_day_marked_pnl(
    rows: object,
) -> tuple[float | None, float | None, int]:
    """Calculate marked and closed gross P&L from Zerodha day-position rows."""

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return None, None, 0
    marked_total = 0.0
    closed_total = 0.0
    seen = 0
    for item in rows:
        if not isinstance(item, Mapping):
            continue
        symbol = _strategy_symbol(item)
        if not symbol or not is_strategy_instrument(symbol):
            continue
        if str(item.get("product") or "").strip().upper() != "MIS":
            continue
        buy_value = _finite_float(item.get("buy_value"))
        sell_value = _finite_float(item.get("sell_value"))
        quantity = _finite_float(
            item.get("quantity", item.get("net_quantity", item.get("net_qty")))
        )
        last_price = _finite_float(item.get("last_price", item.get("ltp")))
        multiplier = _finite_float(item.get("multiplier"))
        if buy_value is None or sell_value is None or quantity is None:
            continue
        if multiplier is None:
            multiplier = 1.0
        if last_price is None:
            if abs(quantity) > 1e-9:
                continue
            last_price = 0.0
        marked_total += sell_value - buy_value + quantity * last_price * multiplier
        if abs(quantity) <= 1e-9:
            closed_total += sell_value - buy_value
        seen += 1
    if not seen:
        return None, None, 0
    return marked_total, closed_total, seen


def _strategy_tradebook_realized_pnl(
    rows: object,
) -> tuple[float | None, int]:
    """Calculate realized gross P&L from current-day Zerodha trade fills."""

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return None, 0

    inventory: dict[str, list[list[float]]] = {}
    realized = 0.0
    seen = 0
    for item in rows:
        if not isinstance(item, Mapping):
            continue
        symbol = _strategy_symbol(item)
        if not symbol or not is_strategy_instrument(symbol):
            continue
        if str(item.get("product") or "").strip().upper() != "MIS":
            continue
        raw_side = item.get("transaction_type") or item.get("side") or ""
        side = str(raw_side).strip().upper()
        quantity = _finite_float(item.get("quantity", item.get("filled_quantity")))
        price = _finite_float(item.get("average_price", item.get("price")))
        if side not in {"BUY", "SELL"} or quantity is None or price is None:
            continue
        if quantity <= 0 or price <= 0:
            continue

        signed = float(quantity) if side == "BUY" else -float(quantity)
        remaining = abs(signed)
        lots = inventory.setdefault(symbol, [])
        while remaining > 1e-9 and lots and lots[0][0] * signed < 0:
            lot_qty, lot_price = lots[0]
            matched = min(abs(lot_qty), remaining)
            if signed < 0:
                realized += (float(price) - lot_price) * matched
            else:
                realized += (lot_price - float(price)) * matched
            remaining -= matched
            lot_remaining = abs(lot_qty) - matched
            if lot_remaining <= 1e-9:
                lots.pop(0)
            else:
                lots[0][0] = lot_remaining if lot_qty > 0 else -lot_remaining

        if remaining > 1e-9:
            lots.append([remaining if signed > 0 else -remaining, float(price)])
        seen += 1

    if not seen:
        return None, 0
    return realized, seen


def _materialize_positions(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        return payload
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        return payload
    try:
        return list(payload)
    except TypeError:
        return payload


def _strip_row_legacy_realized(row: object) -> object:
    if not isinstance(row, Mapping):
        return row
    clean = dict(row)
    clean.pop("realised", None)
    clean.pop("realized", None)
    return clean


def _strip_legacy_position_pnl(payload: Any) -> Any:
    """Remove legacy realised fields while preserving exposure payload shape."""

    materialized = _materialize_positions(payload)
    if isinstance(materialized, Mapping):
        clean = dict(materialized)
        if (
            "net" in clean
            and isinstance(clean.get("net"), Sequence)
            and not isinstance(clean.get("net"), (str, bytes))
        ):
            clean["net"] = [
                _strip_row_legacy_realized(row) for row in clean.get("net", [])
            ]
        elif (
            "positions" in clean
            and isinstance(clean.get("positions"), Sequence)
            and not isinstance(clean.get("positions"), (str, bytes))
        ):
            clean["positions"] = [
                _strip_row_legacy_realized(row)
                for row in clean.get("positions", [])
            ]
        else:
            clean = dict(_strip_row_legacy_realized(clean))
        return clean
    if isinstance(materialized, Sequence) and not isinstance(
        materialized, (str, bytes)
    ):
        return [_strip_row_legacy_realized(row) for row in materialized]
    return materialized


def _broker_proves_no_current_day_trading(snapshot: Mapping[str, Any]) -> bool:
    """Return True only when broker evidence supports a clean zero-trade day."""

    if snapshot.get("positions_error"):
        return False
    realized = _finite_float(snapshot.get("account_realized"))
    unrealized = _finite_float(snapshot.get("account_unrealized"))
    if realized is None or abs(realized) > _EPS:
        return False
    if unrealized is not None and abs(unrealized) > _EPS:
        return False
    try:
        day_rows = int(snapshot.get("strategy_day_rows", 0) or 0)
    except (TypeError, ValueError):
        return False
    return day_rows == 0


__all__ = [
    "_MATCH_TOLERANCE_RUPEES",
    "_broker_proves_no_current_day_trading",
    "_effective_refresh_seconds",
    "_extract_account_m2m",
    "_finite_float",
    "_max_age_seconds",
    "_strategy_day_marked_pnl",
    "_strategy_tradebook_realized_pnl",
    "_strip_legacy_position_pnl",
]
