"""Purpose: Verify exchange-resident stop coverage against broker order truth.

Responsibilities:
- verify_exchange_stop_order validates the exact broker stop-order identity.
- Match symbol, side, product, type, live status, quantity and stop geometry.
- Return fixed diagnostic reason tokens without leaking broker payloads.

Boundaries: Pure validation only; never queries brokers, submits, or modifies orders.
Safety: A submitted/modified order ID alone is not protection proof.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def _positive_quantity(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or number <= 0 or not number.is_integer():
        return None
    return int(number)


def _positive_price(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _order_symbol(value: Any) -> str:
    """Canonicalize an NFO-prefixed tradingsymbol without fuzzy matching."""
    return str(value or "").strip().upper().removeprefix("NFO:")


def verify_exchange_stop_order(
    evidence: Mapping[str, Any],
    *,
    order_id: str,
    symbol: str,
    entry_side: str,
    product: str,
    quantity: int,
    trigger_price: float | None,
    limit_price: float | None,
) -> tuple[bool, str | None]:
    """Return broker-confirmed protection or a fixed fail-closed reason code."""
    state = str(evidence.get("status") or "").strip().upper()
    if state not in {"OPEN", "TRIGGER PENDING"}:
        known = {"MODIFY PENDING", "CANCELLED", "REJECTED", "COMPLETE", "OPEN PENDING"}
        safe_state = state if state in known else "UNKNOWN"
        return False, "broker_stop_status_unverified:" + safe_state
    broker_id = str(evidence.get("order_id") or evidence.get("id") or "").strip()
    if not broker_id or broker_id != str(order_id).strip():
        return False, "broker_stop_order_id_mismatch"
    expected_symbol = _order_symbol(symbol)
    broker_symbol = _order_symbol(
        evidence.get("tradingsymbol") or evidence.get("symbol")
    )
    if not expected_symbol or broker_symbol != expected_symbol:
        return False, "broker_stop_symbol_mismatch"
    broker_exchange = str(evidence.get("exchange") or "").strip().upper()
    if broker_exchange != "NFO":
        return False, "broker_stop_exchange_unverified"
    expected_side = "SELL" if str(entry_side).upper() == "BUY" else "BUY"
    broker_side = str(
        evidence.get("transaction_type") or evidence.get("side") or ""
    ).strip().upper()
    if broker_side != expected_side:
        return False, "broker_stop_side_mismatch"
    if (
        str(evidence.get("product") or "").strip().upper()
        != str(product).strip().upper()
    ):
        return False, "broker_stop_product_mismatch"
    order_type = str(evidence.get("order_type") or "").strip().upper()
    if order_type not in {"SL", "SL-M"}:
        return False, "broker_stop_order_type_unverified"
    requested = _positive_quantity(quantity)
    broker_qty = _positive_quantity(evidence.get("quantity"))
    if requested is None or broker_qty is None or broker_qty != requested:
        return False, "broker_stop_quantity_mismatch"
    # A partly filled stop does not prove full outstanding coverage.
    filled = evidence.get("filled_quantity")
    if filled is not None and filled != "":
        try:
            if float(filled) != 0.0:
                return False, "broker_stop_partially_filled"
        except (TypeError, ValueError, OverflowError):
            return False, "broker_stop_fill_unverified"
    if "pending_quantity" in evidence:
        pending = _positive_quantity(evidence.get("pending_quantity"))
        if pending != requested:
            return False, "broker_stop_remaining_quantity_mismatch"
    expected_trigger = _positive_price(trigger_price)
    broker_trigger = _positive_price(evidence.get("trigger_price"))
    if (
        expected_trigger is None
        or broker_trigger is None
        or abs(broker_trigger - expected_trigger) > 0.011
    ):
        return False, "broker_stop_trigger_mismatch"
    if order_type == "SL":
        expected_limit = _positive_price(limit_price)
        broker_limit = _positive_price(evidence.get("price"))
        if (
            expected_limit is None
            or broker_limit is None
            or abs(broker_limit - expected_limit) > 0.011
        ):
            return False, "broker_stop_limit_mismatch"
        if (expected_side == "SELL" and broker_limit >= broker_trigger) or (
            expected_side == "BUY" and broker_limit <= broker_trigger
        ):
            return False, "broker_stop_price_geometry_invalid"
    return True, None
