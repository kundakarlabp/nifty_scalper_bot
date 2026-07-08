"""Operator control hooks for Telegram emergency and flatten commands.

The Telegram command layer calls methods on the service/order-manager if they
exist. Production previously registered the commands but did not expose concrete
RuntimeOrderManager methods, so /emergency and /flatten could end as not_wired.
This patch adds bounded, broker-aware controls without introducing a second entry
path.
"""

from __future__ import annotations

from contextlib import suppress
import time
from typing import Any, Iterable, Mapping

from nifty_scalper_bot.core.trading_switch import trading_switch
from nifty_scalper_bot.utils.symbols import normalize_symbol

_PATCH_APPLIED = False

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
            if isinstance(nested, Iterable) and not isinstance(nested, (str, bytes, Mapping)):
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
                cancel(*args, **kwargs)
                return True
            except TypeError:
                continue
            except Exception:
                break
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
        extra={"event": "OPERATOR_CANCEL_PENDING_ORDERS", "cancelled": cancelled, "failed": failed},
    )
    return {"cancelled": cancelled, "failed": failed}


def _place_flatten_order(self: Any, symbol: str, qty: int) -> str | None:
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
        product="MIS",
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
    cancel_result = cancel_pending_orders(self) if cancel_first else {"cancelled": [], "failed": []}

    submitted: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in _position_rows(self):
        symbol = _symbol(row)
        qty = _quantity(row)
        if not symbol or qty == 0 or symbol in seen:
            continue
        seen.add(symbol)
        try:
            order_id = _place_flatten_order(self, symbol, qty)
        except Exception as exc:  # noqa: BLE001 - operator control boundary
            failed.append({"symbol": symbol, "qty": qty, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if order_id:
            submitted.append({"symbol": symbol, "qty": abs(qty), "side": "SELL" if qty > 0 else "BUY", "order_id": order_id})
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
    """Fail closed immediately: pause entries, latch kill flag, cancel orders, flatten exposure."""

    with suppress(Exception):
        trading_switch().pause()
    setattr(self, "_kill_switch_engaged_at", time.time())
    setattr(self, "_kill_switch_reason", str(reason))
    cancel_result = cancel_pending_orders(self)
    flatten_result = flatten_all(self, reason=reason, cancel_first=False)
    _logger(self).critical(
        "OPERATOR_EMERGENCY_STOP reason=%s cancelled=%s failed=%s flattened=%s flatten_failed=%s",
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
    return {"kill_switch": True, "reason": str(reason), "cancel": cancel_result, "flatten": flatten_result}


def apply_patches() -> None:
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return
    from nifty_scalper_bot.execution.runtime_order_manager import RuntimeOrderManager

    RuntimeOrderManager.emergency_stop = emergency_stop
    RuntimeOrderManager.engage_kill_switch = emergency_stop
    RuntimeOrderManager.kill_switch = emergency_stop
    RuntimeOrderManager.cancel_pending_orders = cancel_pending_orders
    RuntimeOrderManager.cancel_all_open_orders = cancel_pending_orders
    RuntimeOrderManager.cancel_non_protective_orders = cancel_pending_orders
    RuntimeOrderManager.flatten_all = flatten_all
    RuntimeOrderManager.flatten_positions = flatten_all
    RuntimeOrderManager.close_all_positions = flatten_all
    RuntimeOrderManager._operator_control_patch = True
    _PATCH_APPLIED = True


__all__ = [
    "apply_patches",
    "emergency_stop",
    "flatten_all",
    "cancel_pending_orders",
    "_place_flatten_order",
]
