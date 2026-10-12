"""Pure Zerodha account and tradebook P&L evidence extraction.

Broker-data layer only: no PositionManager or import-time method modification.
The strategy fill ledger remains the authority for realized strategy P&L.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, cast

from nifty_scalper_bot.utils.symbols import is_strategy_instrument


def _finite_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


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


def _strategy_day_marked_pnl(rows: object) -> tuple[float | None, float | None, int]:
    """Calculate marked and closed gross P&L from Zerodha ``data.day`` rows.

    Formula follows Zerodha's position economics:
    sell_value - buy_value + net_quantity * last_price * multiplier.
    This is diagnostic account/broker evidence only and never overwrites the
    bot-owned strategy fill ledger.
    """

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


def _strategy_tradebook_realized_pnl(rows: object) -> tuple[float | None, int]:
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


def _zerodha_get_pnl_snapshot(self: Any) -> dict[str, Any]:
    """Return dedicated broker P&L evidence without changing exposure semantics."""

    realized: float | None = None
    unrealized: float | None = None
    margins_error: str | None = None
    margins_fetcher = getattr(self, "get_account_margins", None)
    if callable(margins_fetcher):
        try:
            margins = margins_fetcher(segment="equity")
            realized, unrealized = _extract_account_m2m(margins)
        except Exception as exc:
            margins_error = f"{type(exc).__name__}: {exc}"
    else:
        margins_error = "broker account margins endpoint unavailable"

    day_marked: float | None = None
    day_closed: float | None = None
    day_rows = 0
    positions_error: str | None = None
    tradebook_realized: float | None = None
    tradebook_fill_count = 0
    tradebook_error: str | None = None

    acquire = getattr(self, "_acquire_bucket", None)
    bucket = getattr(self, "_GENERAL_BUCKET", None)
    request = getattr(self, "_make_request", None)
    ensure_json = getattr(self, "_ensure_json", None)

    try:
        if callable(acquire) and bucket is not None:
            acquire(bucket)
        if callable(request) and callable(ensure_json):
            response = ensure_json(
                request(
                    "GET",
                    "/portfolio/positions",
                    operation_label="pnl.positions",
                )
            )
            data = response.get("data") if isinstance(response, Mapping) else None
            day = data.get("day") if isinstance(data, Mapping) else None
            day_marked, day_closed, day_rows = _strategy_day_marked_pnl(day)
    except Exception as exc:  # diagnostic enrichment must never impair P&L authority
        positions_error = f"{type(exc).__name__}: {exc}"

    try:
        if callable(acquire) and bucket is not None:
            acquire(bucket)
        if callable(request) and callable(ensure_json):
            response = ensure_json(
                request("GET", "/trades", operation_label="pnl.trades")
            )
            trades = response.get("data") if isinstance(response, Mapping) else None
            tradebook_realized, tradebook_fill_count = _strategy_tradebook_realized_pnl(
                trades
            )
    except Exception as exc:  # diagnostic enrichment must never impair P&L authority
        tradebook_error = f"{type(exc).__name__}: {exc}"

    margin_realized = None if realized is None else float(realized)
    margin_unrealized = None if unrealized is None else float(unrealized)
    return {
        "account_realized": margin_realized,
        "account_unrealized": margin_unrealized,
        "account_total": (
            None
            if margin_realized is None
            else margin_realized + float(margin_unrealized or 0.0)
        ),
        "margin_m2m_realized": margin_realized,
        "margin_m2m_unrealized": margin_unrealized,
        "margins_error": margins_error,
        "strategy_tradebook_realized_gross": tradebook_realized,
        "strategy_tradebook_fill_count": int(tradebook_fill_count),
        "tradebook_source": "zerodha_trades",
        "tradebook_error": tradebook_error,
        "strategy_day_marked_gross": day_marked,
        "strategy_day_closed_gross": day_closed,
        "strategy_day_rows": int(day_rows),
        "source": "zerodha_margins_m2m",
        "positions_source": "zerodha_positions_day",
        "positions_error": positions_error,
        "observed_at": datetime.now(timezone.utc).isoformat(),
    }
