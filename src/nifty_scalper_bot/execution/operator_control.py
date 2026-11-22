"""Purpose: Broker-safe operator cancellation and emergency flattening.

Responsibilities:
- Accept RuntimeOrderManager operator requests without a parallel trading owner.
- Validate LIVE net exposure, contract lots, and terminal cancellation readback.
Boundaries: All order submissions use RuntimeOrderManager.place_order.
Safety: Unknown broker position, lot or working stop holds replacement exits.
"""

from __future__ import annotations

import time
from contextlib import suppress
from typing import Any, Iterable, Mapping

from nifty_scalper_bot.core.trading_switch import trading_switch
from nifty_scalper_bot.execution.position_snapshot import decode_position_snapshot
from nifty_scalper_bot.utils.symbols import normalize_symbol

_OPEN_ORDER_STATUSES = {
    "OPEN",
    "OPEN PENDING",
    "PENDING",
    "TRIGGER PENDING",
    "VALIDATION PENDING",
    "PUT ORDER REQ RECEIVED",
    "MODIFY PENDING",
    "AMO REQ RECEIVED",
}


def _logger(self: Any) -> Any:
    return getattr(self, "_logger", None) or __import__("logging").getLogger(__name__)


def _broker(self: Any) -> Any:
    return getattr(self, "_broker", None) or getattr(self, "broker", None)


def _position_manager(self: Any) -> Any:
    return (
        getattr(self, "_position_manager", None)
        or getattr(self, "position_manager", None)
        or getattr(self, "positions", None)
    )


def _as_iterable(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        for key in ("data", "positions", "orders", "net", "day"):
            nested = value.get(key)
            if isinstance(nested, Iterable) and not isinstance(
                nested, (str, bytes, Mapping)
            ):
                return list(nested)
        return [value]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


def _call_first(owner: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        func = getattr(owner, name, None)
        if callable(func):
            return func()
    return None


def _position_rows(self: Any) -> list[Any]:
    broker = _broker(self)
    pm = _position_manager(self)
    for owner, names in (
        (broker, ("get_positions", "positions", "get_open_positions")),
        (pm, ("get_all_positions", "get_open_positions", "open_positions")),
    ):
        if owner is None:
            continue
        rows = _as_iterable(_call_first(owner, names))
        if rows:
            return rows
    return []


def _order_rows(self: Any) -> list[Any]:
    broker = _broker(self)
    for owner, names in (
        (self, ("get_open_orders", "open_orders", "pending_orders")),
        (broker, ("get_orders", "orders", "get_open_orders")),
    ):
        if owner is None:
            continue
        rows = _as_iterable(_call_first(owner, names))
        if rows:
            return rows
    return []


def _field(row: Any, *names: str) -> Any:
    if isinstance(row, Mapping):
        for name in names:
            if name in row:
                return row.get(name)
        return None
    for name in names:
        if hasattr(row, name):
            return getattr(row, name)
    return None


def _symbol(row: Any) -> str:
    raw = _field(row, "symbol", "tradingsymbol", "trading_symbol", "instrument")
    return normalize_symbol(str(raw or ""))


def _quantity(row: Any) -> int:
    for name in ("quantity", "net_quantity", "net_qty", "net", "qty"):
        raw = _field(row, name)
        if raw is None:
            continue
        with suppress(Exception):
            return int(float(raw or 0))
    return 0


def _order_id(row: Any) -> str:
    return str(_field(row, "order_id", "id", "exchange_order_id") or "").strip()


def _order_status(row: Any) -> str:
    return str(_field(row, "status", "order_status") or "").strip().upper()


def _cancel_order(self: Any, order_id: str) -> bool:
    if not order_id:
        return False
    for owner in (self, _broker(self)):
        cancel = getattr(owner, "cancel_order", None) if owner is not None else None
        if not callable(cancel):
            continue
        for args, kwargs in (
            ((order_id,), {}),
            ((), {"order_id": order_id}),
            (("regular", order_id), {}),
            ((), {"variety": "regular", "order_id": order_id}),
        ):
            try:
                # An explicit false/empty cancellation result is not success.
                # Do not retry a potentially accepted call via another owner.
                return bool(cancel(*args, **kwargs))
            except TypeError:
                continue
            except Exception:
                return False
    return False


def cancel_pending_orders(self: Any) -> dict[str, Any]:
    cancelled: list[str] = []
    failed: list[str] = []
    for row in _order_rows(self):
        status = _order_status(row)
        if status and status not in _OPEN_ORDER_STATUSES:
            continue
        order_id = _order_id(row)
        if not order_id:
            continue
        if _cancel_order(self, order_id):
            cancelled.append(order_id)
        else:
            failed.append(order_id)
    _logger(self).warning(
        "OPERATOR_CANCEL_PENDING_ORDERS cancelled=%s failed=%s",
        len(cancelled),
        len(failed),
        extra={
            "event": "OPERATOR_CANCEL_PENDING_ORDERS",
            "cancelled": cancelled,
            "failed": failed,
        },
    )
    return {"cancelled": cancelled, "failed": failed}


def _place_flatten_order(
    self: Any, symbol: str, qty: int, *, product: str = "MIS"
) -> str | None:
    side = "SELL" if qty > 0 else "BUY"
    quantity = abs(int(qty))
    if quantity <= 0 or not symbol:
        return None
    order_id = self.place_order(
        symbol=symbol,
        side=side,
        quantity=quantity,
        order_type="MARKET",
        tag="EXIT_FLATTEN_TELEGRAM",
        check_risk=False,
        product=product,
        intent="REDUCE",
        strategy_name="operator_flatten",
    )
    return str(order_id) if order_id else None


def flatten_all(
    self: Any,
    reason: str = "telegram_flatten",
    *,
    cancel_first: bool = True,
) -> dict[str, Any]:
    """Cancel pending orders and market-flatten all non-zero broker/local rows."""

    with suppress(Exception):
        trading_switch().pause()
    setattr(self, "_kill_switch_engaged_at", time.time())
    setattr(self, "_kill_switch_reason", str(reason))
    submitted: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    cancel_result: dict[str, Any] = {"cancelled": [], "failed": []}
    checker = getattr(self, "is_live_mode", None)
    try:
        live = bool(checker()) if callable(checker) else False
        broker = _broker(self)
        initial: dict[str, tuple[int, str, int]] = {}
        if live:
            # Preflight FIRST: do not cancel a protected stop if broker truth,
            # held product or authoritative instrument lot cannot be proven.
            order_fetch = getattr(broker, "get_orders", None)
            position_fetch = getattr(broker, "get_positions", None)
            if not callable(order_fetch) or not callable(position_fetch):
                raise ValueError("broker_readback_unavailable")
            pre_orders = order_fetch()
            if not isinstance(pre_orders, list):
                raise ValueError("broker_orders_unavailable")
            for row in decode_position_snapshot(position_fetch()).raw_rows():
                symbol = _symbol(row)
                qty = _quantity(row)
                if qty == 0 or not symbol.startswith("NFO:NIFTY"):
                    continue
                if not symbol.endswith(("CE", "PE")):
                    continue
                product = str(_field(row, "product") or "").upper()
                lot = int(self._lot_size_for_symbol(symbol))
                if product not in {"MIS", "NRML"} or lot <= 0 or abs(qty) % lot:
                    raise ValueError("broker_position_lot_or_product_invalid")
                initial[symbol] = (qty, product, lot)
        if cancel_first:
            cancel_result = cancel_pending_orders(self)
        if live:
            remote_orders = order_fetch() if callable(order_fetch) else None
            if not isinstance(remote_orders, list):
                raise ValueError("broker_orders_unavailable_after_cancel")
            snapshot = position_fetch() if callable(position_fetch) else None
            positions = decode_position_snapshot(snapshot).raw_rows()
        else:
            remote_orders = []
            positions = _position_rows(self)
    except Exception as exc:
        failed.append({"reason": f"broker_proof_unavailable:{type(exc).__name__}"})
        return {"cancel": cancel_result, "submitted": submitted, "failed": failed}

    seen: set[str] = set()
    for row in positions:
        symbol = _symbol(row)
        qty = _quantity(row)
        if not symbol or qty == 0 or symbol in seen:
            continue
        seen.add(symbol)
        product = "MIS"
        if live:
            prior = initial.get(symbol)
            product = str(_field(row, "product") or "").upper()
            if prior is None:
                failed.append(
                    {"symbol": symbol, "reason": "new_or_unsupported_exposure"}
                )
                continue
            old_qty, old_product, lot = prior
            if (
                product != old_product
                or (old_qty > 0) != (qty > 0)
                or abs(qty) > abs(old_qty)
                or abs(qty) % lot
            ):
                failed.append({"symbol": symbol, "reason": "broker_residual_mismatch"})
                continue
            same_symbol_orders = [
                order for order in remote_orders if _symbol(order) == symbol
            ]
            terminal = {"COMPLETE", "CANCELLED", "CANCELED", "REJECTED"}
            if any(
                _order_status(order) not in terminal for order in same_symbol_orders
            ):
                failed.append({"symbol": symbol, "reason": "cancel_unconfirmed"})
                continue
            cancelled_ids = set(cancel_result.get("cancelled", []))
            if any(
                _order_id(order) in cancelled_ids and _order_status(order) == "COMPLETE"
                for order in same_symbol_orders
            ):
                failed.append({"symbol": symbol, "reason": "stop_filled_sync_pending"})
                continue
        try:
            order_id = _place_flatten_order(self, symbol, qty, product=product)
        except Exception as exc:  # noqa: BLE001 - operator control boundary
            failed.append(
                {
                    "symbol": symbol,
                    "qty": qty,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        if order_id:
            submitted.append(
                {
                    "symbol": symbol,
                    "qty": abs(qty),
                    "side": "SELL" if qty > 0 else "BUY",
                    "order_id": order_id,
                }
            )
        else:
            failed.append({"symbol": symbol, "qty": qty, "error": "missing_order_id"})

    result = {"cancel": cancel_result, "submitted": submitted, "failed": failed}
    _logger(self).critical(
        "OPERATOR_FLATTEN_ALL submitted=%s failed=%s cancelled=%s",
        len(submitted),
        len(failed),
        len(cancel_result.get("cancelled", [])),
        extra={"event": "OPERATOR_FLATTEN_ALL", **result},
    )
    return result


def emergency_stop(self: Any, reason: str = "telegram_emergency") -> dict[str, Any]:
    """Pause entries, latch the kill flag, cancel orders and flatten exposure."""

    with suppress(Exception):
        trading_switch().pause()
    setattr(self, "_kill_switch_engaged_at", time.time())
    setattr(self, "_kill_switch_reason", str(reason))
    flatten_result = flatten_all(self, reason=reason, cancel_first=True)
    cancel_result = flatten_result["cancel"]
    _logger(self).critical(
        "OPERATOR_EMERGENCY_STOP reason=%s cancelled=%s failed=%s "
        "flattened=%s flatten_failed=%s",
        reason,
        len(cancel_result.get("cancelled", [])),
        len(cancel_result.get("failed", [])),
        len(flatten_result.get("submitted", [])),
        len(flatten_result.get("failed", [])),
        extra={
            "event": "OPERATOR_EMERGENCY_STOP",
            "reason": reason,
            "cancel": cancel_result,
            "flatten": flatten_result,
        },
    )
    return {
        "kill_switch": True,
        "reason": str(reason),
        "cancel": cancel_result,
        "flatten": flatten_result,
    }


__all__ = [
    "cancel_pending_orders",
    "emergency_stop",
    "flatten_all",
    "_place_flatten_order",
]
