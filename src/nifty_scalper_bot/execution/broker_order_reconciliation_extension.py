"""Deterministic broker-order reconciliation extension.

This patch closes the restart/reconcile gap where broker orders exist but the
bot's in-memory order map does not know them. Unknown broker orders are first
recorded in a durable ledger, then classified exactly once as resolved,
active-unknown, broker-state-unverified, or quarantined exposure.
"""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import Any, Mapping

from nifty_scalper_bot.execution import position_identity_extension as _position_identity
from nifty_scalper_bot.execution import position_manager as _position_manager
from nifty_scalper_bot.execution.broker_order_ledger import (
    ACTIVE_STATUSES,
    BrokerOrderLedger,
    BrokerOrderLedgerEntry,
    normalise_broker_order_id,
    normalise_broker_status,
    normalise_broker_symbol,
)
from nifty_scalper_bot.utils.symbols import normalize_symbol

_PATCH_APPLIED = False
_ORIGINALS: dict[str, Any] = {}
_UNKNOWN_INTENTS = {"", "UNKNOWN", "BROKER_IMPORTED_ORDER", "MANUAL_ORDER_QUARANTINED"}
_TERMINAL_NO_EXPOSURE = {"CANCELLED", "REJECTED", "EXPIRED"}
_UNRESOLVED_STATES = {"broker_state_unverified", "quarantined_broker_position"}


def _canonical(symbol: object) -> str:
    return normalize_symbol(str(symbol or ""))


def _state_path_for(manager: Any) -> Path:
    raw_path = getattr(manager, "_state_path", None)
    base_path = Path(raw_path) if raw_path is not None else Path("positions.json")
    return base_path.parent / "broker_order_ledger.json"


def _ledger_for(manager: Any) -> BrokerOrderLedger:
    ledger = getattr(manager, "_broker_order_ledger", None)
    if isinstance(ledger, BrokerOrderLedger):
        return ledger
    ledger = BrokerOrderLedger(_state_path_for(manager))
    manager._broker_order_ledger = ledger
    return ledger


def _field_int(payload: Mapping[str, Any], *fields: str) -> int:
    for field_name in fields:
        value = payload.get(field_name)
        if value in (None, ""):
            continue
        with suppress(Exception):
            return abs(int(float(value)))
    return 0


def _field_float(payload: Mapping[str, Any], *fields: str) -> float:
    for field_name in fields:
        value = payload.get(field_name)
        if value in (None, ""):
            continue
        with suppress(Exception):
            number = float(value)
            if number > 0:
                return number
    return 0.0


def _payload_side(payload: Mapping[str, Any], fallback: str = "BUY") -> str:
    raw = str(
        payload.get("transaction_type")
        or payload.get("side")
        or payload.get("order_side")
        or fallback
    ).strip().upper()
    if raw in {"BUY", "B"}:
        return "BUY"
    if raw in {"SELL", "S"}:
        return "SELL"
    return fallback


def _payload_order_type(payload: Mapping[str, Any]) -> str:
    raw = str(payload.get("order_type") or payload.get("type") or "MARKET").strip().upper()
    if raw in {"MARKET", "LIMIT", "SL", "SL-M"}:
        return raw
    return "MARKET"


def _materialise_unknown_order(
    manager: Any,
    entry: BrokerOrderLedgerEntry,
    payload: Mapping[str, Any],
) -> None:
    """Create a minimal pending-order row so existing lifecycle guards can run."""

    orders = getattr(manager, "_orders", {})
    if isinstance(orders, dict) and entry.order_id in orders:
        return
    add_pending = getattr(manager, "add_pending_order", None)
    if not callable(add_pending):
        return
    qty = entry.quantity or entry.filled_quantity or _field_int(
        payload,
        "quantity",
        "qty",
        "filled_quantity",
        "filled_qty",
        "filled",
    )
    price = entry.average_price or _field_float(
        payload,
        "average_price",
        "avg_price",
        "fill_price",
        "price",
    )
    symbol = entry.symbol or normalise_broker_symbol(payload)
    side = entry.side or _payload_side(payload)
    add_pending(
        entry.order_id,
        symbol,
        side,
        qty,
        price,
        _payload_order_type(payload),
        intent="UNKNOWN",
        signal_id=str(payload.get("tag") or payload.get("guid") or "") or None,
    )


def _classify_unknown_order(
    manager: Any,
    entry: BrokerOrderLedgerEntry,
) -> tuple[str, str]:
    """Classify an unknown broker order without applying accounting side effects."""

    status = normalise_broker_status(entry.status)
    if status in _TERMINAL_NO_EXPOSURE:
        return "resolved_terminal_no_fill", status.lower()
    if status == "FILLED" and max(entry.filled_quantity, entry.quantity) <= 0:
        return "resolved_terminal_no_fill", "filled_zero_quantity"
    if status in ACTIVE_STATUSES:
        return "active_unknown_broker_order", "broker_order_still_active"
    if status == "FILLED":
        state, broker_qty, broker_error = _position_identity._broker_position_quantity(
            manager,
            entry.symbol,
        )
        if state == "flat":
            return "resolved_broker_flat", "broker_position_flat"
        if state == "unverified":
            return "broker_state_unverified", broker_error or "broker_position_unverified"
        return "quarantined_broker_position", f"broker_position_open:{broker_qty}"
    return "broker_state_unverified", f"unsupported_broker_status:{status}"


def _canonicalize_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    cloned = dict(payload)
    symbol = normalise_broker_symbol(cloned)
    if symbol:
        cloned["symbol"] = symbol
        cloned["tradingsymbol"] = symbol
    return cloned


def _log_once(manager: Any, event: str, message: str, *args: Any, **extra: Any) -> None:
    logger = getattr(manager, "_logger", None)
    log = getattr(logger, "warning", None)
    if event.endswith("RESOLVED"):
        log = getattr(logger, "info", None)
    if callable(log):
        log(message, *args, extra={"event": event, **extra})


def _should_skip_repeated_unknown_terminal(
    manager: Any,
    order_id: str,
    entry: BrokerOrderLedgerEntry,
) -> bool:
    if entry.reconciliation_state not in _UNRESOLVED_STATES:
        return False
    terminal_orders = getattr(manager, "_terminal_orders", {})
    unresolved_orders = getattr(manager, "_unresolved_terminal_orders", {})
    return (
        isinstance(terminal_orders, dict)
        and order_id in terminal_orders
        and isinstance(unresolved_orders, dict)
        and order_id in unresolved_orders
    )


def _current_reconciliation_blocker(manager: Any, symbol: str | None = None) -> str | None:
    ledger = _ledger_for(manager)
    blockers = ledger.blocking_entries(symbol)
    if blockers:
        if any(entry.reconciliation_state == "broker_state_unverified" for entry in blockers):
            return "broker_state_unverified"
        if any(entry.reconciliation_state == "quarantined_broker_position" for entry in blockers):
            return "broker_position_quarantined"
        return "active_unknown_broker_order"
    original = _ORIGINALS.get("PositionManager.current_reconciliation_blocker")
    if callable(original):
        return original(manager, symbol)
    return None


def apply_patches() -> None:
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return
    cls = getattr(_position_manager, "PositionManager", None)
    if cls is None or getattr(cls, "_broker_order_reconciliation_patch", False):
        _PATCH_APPLIED = True
        return

    _ORIGINALS["PositionManager.__init__"] = cls.__init__
    _ORIGINALS["PositionManager.add_pending_order"] = cls.add_pending_order
    _ORIGINALS["PositionManager.apply_broker_order_update"] = cls.apply_broker_order_update
    if hasattr(cls, "current_reconciliation_blocker"):
        _ORIGINALS["PositionManager.current_reconciliation_blocker"] = cls.current_reconciliation_blocker

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        _ORIGINALS["PositionManager.__init__"](self, *args, **kwargs)
        self._broker_order_ledger = BrokerOrderLedger(_state_path_for(self))

    def add_pending_order(
        self: Any,
        order_id: str,
        symbol: str,
        side: str,
        qty: int,
        price: float,
        order_type: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        result = _ORIGINALS["PositionManager.add_pending_order"](
            self,
            order_id,
            _canonical(symbol),
            side,
            qty,
            price,
            order_type,
            *args,
            **kwargs,
        )
        intent = kwargs.get("intent")
        if intent is None and args:
            intent = args[0]
        try:
            _ledger_for(self).upsert_local_order(
                order_id,
                symbol=_canonical(symbol),
                side=side,
                quantity=qty,
                price=price,
                status="PENDING",
                intent=intent or "UNKNOWN",
                tag=kwargs.get("signal_id") or "",
            )
        except Exception as exc:  # noqa: BLE001 - ledger failure must not hide local registration
            logger = getattr(self, "_logger", None)
            log = getattr(logger, "error", None)
            if callable(log):
                log(
                    "BROKER_ORDER_LEDGER_LOCAL_REGISTER_FAILED order_id=%s error=%s",
                    order_id,
                    exc,
                    extra={
                        "event": "BROKER_ORDER_LEDGER_LOCAL_REGISTER_FAILED",
                        "order_id": order_id,
                        "symbol": _canonical(symbol),
                        "error_type": type(exc).__name__,
                    },
                )
        return result

    def apply_broker_order_update(
        self: Any,
        order_id: str,
        broker_payload: Mapping[str, Any],
    ) -> Any:
        payload = _canonicalize_payload(broker_payload)
        order_key = normalise_broker_order_id(payload, order_id)
        if not order_key:
            return _ORIGINALS["PositionManager.apply_broker_order_update"](
                self,
                order_id,
                payload,
            )
        ledger = _ledger_for(self)
        try:
            entry = ledger.upsert_from_broker(payload, fallback_order_id=order_key)
        except Exception:
            return _ORIGINALS["PositionManager.apply_broker_order_update"](
                self,
                order_key,
                payload,
            )

        orders = getattr(self, "_orders", {})
        order = orders.get(order_key) if isinstance(orders, dict) else None
        order_intent = str(getattr(order, "intent", "UNKNOWN") or "UNKNOWN").strip().upper()
        if order is not None and order_intent not in _UNKNOWN_INTENTS:
            return _ORIGINALS["PositionManager.apply_broker_order_update"](
                self,
                order_key,
                payload,
            )

        state, reason = _classify_unknown_order(self, entry)
        previous_state = entry.reconciliation_state
        entry = ledger.mark(order_key, state, reason) or entry

        if state in {"resolved_terminal_no_fill", "resolved_broker_flat"}:
            if previous_state != state:
                _log_once(
                    self,
                    "BROKER_ORDER_LEDGER_RESOLVED",
                    "BROKER_ORDER_LEDGER_RESOLVED order_id=%s symbol=%s state=%s reason=%s",
                    order_key,
                    entry.symbol,
                    state,
                    reason,
                    order_id=order_key,
                    symbol=entry.symbol,
                    reconciliation_state=state,
                    reason=reason,
                )
            return None

        if _should_skip_repeated_unknown_terminal(self, order_key, entry):
            return None

        if previous_state != state:
            _log_once(
                self,
                "BROKER_ORDER_LEDGER_IMPORTED_UNKNOWN_ORDER",
                "BROKER_ORDER_LEDGER_IMPORTED_UNKNOWN_ORDER order_id=%s symbol=%s state=%s reason=%s",
                order_key,
                entry.symbol,
                state,
                reason,
                order_id=order_key,
                symbol=entry.symbol,
                reconciliation_state=state,
                reason=reason,
            )
        _materialise_unknown_order(self, entry, payload)
        return _ORIGINALS["PositionManager.apply_broker_order_update"](
            self,
            order_key,
            payload,
        )

    def current_reconciliation_blocker(self: Any, symbol: str | None = None) -> str | None:
        return _current_reconciliation_blocker(self, symbol)

    def get_broker_order_ledger_snapshot(self: Any) -> dict[str, Any]:
        return _ledger_for(self).snapshot()

    cls.__init__ = __init__
    cls.add_pending_order = add_pending_order
    cls.apply_broker_order_update = apply_broker_order_update
    cls.current_reconciliation_blocker = current_reconciliation_blocker
    cls.get_broker_order_ledger_snapshot = get_broker_order_ledger_snapshot
    cls._broker_order_reconciliation_patch = True
    _PATCH_APPLIED = True


apply_patches()

__all__ = ["apply_patches"]
