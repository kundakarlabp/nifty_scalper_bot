"""Durable broker-order ledger and deterministic reconciliation patch.

Unknown broker orders are persisted, classified once, and exposed to entry gates
without repeatedly replaying the same unmanaged broker state through fill
accounting.
"""

from __future__ import annotations

from contextlib import suppress
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from nifty_scalper_bot.execution import (
    position_identity_extension as _position_identity,
)
from nifty_scalper_bot.execution import position_manager as _position_manager
from nifty_scalper_bot.utils.symbols import normalize_symbol

_PATCH_APPLIED = False
_ORIGINALS: dict[str, Any] = {}
_TERMINAL_CLASSIFICATIONS = {"resolved_external_flat", "resolved_external_terminal"}
_ACTIVE_CLASSIFICATIONS = {
    "active_external_order",
    "broker_position_quarantined",
    "broker_state_unverified",
}
_STATUS_MAP = {
    "COMPLETE": "FILLED",
    "COMPLETED": "FILLED",
    "FILLED": "FILLED",
    "OPEN": "OPEN",
    "TRIGGER PENDING": "OPEN",
    "PENDING": "PENDING",
    "SUBMITTED": "PENDING",
    "PARTIALLY FILLED": "PARTIALLY_FILLED",
    "PARTIALLY_FILLED": "PARTIALLY_FILLED",
    "CANCELLED": "CANCELLED",
    "CANCELED": "CANCELLED",
    "REJECTED": "REJECTED",
    "EXPIRED": "EXPIRED",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(symbol: object) -> str:
    return normalize_symbol(str(symbol or ""))


def _get(row: Any, *names: str) -> Any:
    if isinstance(row, Mapping):
        for name in names:
            if name in row:
                return row.get(name)
        return None
    for name in names:
        if hasattr(row, name):
            return getattr(row, name)
    return None


def _to_int(value: Any, default: int = 0) -> int:
    with suppress(Exception):
        return int(float(value or 0))
    return default


def _to_float(value: Any, default: float = 0.0) -> float:
    with suppress(Exception):
        return float(value or 0.0)
    return default


def _status(row: Any) -> str:
    raw = str(_get(row, "status", "order_status", "state") or "UNKNOWN").strip().upper()
    normalized = getattr(
        _position_manager, "normalize_broker_order_status", lambda value: value
    )(raw)
    token = str(normalized or raw or "UNKNOWN").strip().upper()
    return _STATUS_MAP.get(token, token)


def _order_id(row: Any) -> str:
    return str(
        _get(row, "order_id", "broker_order_id", "exchange_order_id", "id") or ""
    ).strip()


def _symbol(row: Any) -> str:
    return _canonical(
        _get(row, "symbol", "tradingsymbol", "trading_symbol", "instrument")
    )


def _side(row: Any) -> str:
    raw = str(_get(row, "side", "transaction_type", "order_side") or "").strip().upper()
    if raw in {"BUY", "B"}:
        return "BUY"
    if raw in {"SELL", "S"}:
        return "SELL"
    return raw


def _quantity(row: Any) -> int:
    return abs(
        _to_int(
            _get(row, "quantity", "qty", "order_quantity", "filled_quantity", "filled")
        )
    )


def _filled_quantity(row: Any) -> int:
    return abs(
        _to_int(_get(row, "filled_quantity", "filled", "filled_qty", "filledQuantity"))
    )


def _average_price(row: Any) -> float:
    return _to_float(_get(row, "average_price", "avg_price", "fill_price", "price"))


def _product(row: Any) -> str:
    return str(_get(row, "product", "product_type") or "MIS").strip().upper()


def _timestamp_key(row: Any) -> tuple[str, str]:
    ts = str(
        _get(
            row,
            "exchange_timestamp",
            "order_timestamp",
            "timestamp",
            "created_at",
            "updated_at",
        )
        or ""
    )
    return ts, _order_id(row)


def _normalise_order_rows(payload: Any) -> list[Any]:
    if payload is None:
        return []
    if isinstance(payload, Mapping):
        for key in ("orders", "data", "net", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return list(value)
        return [payload] if _order_id(payload) else []
    try:
        return list(payload)
    except TypeError:
        return []


def _row_to_payload(row: Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        payload = dict(row)
    else:
        payload = {
            name: getattr(row, name)
            for name in dir(row)
            if not name.startswith("_") and not callable(getattr(row, name, None))
        }
    symbol = _symbol(payload)
    if symbol:
        payload["symbol"] = symbol
        payload["tradingsymbol"] = symbol
    payload["order_id"] = _order_id(payload)
    payload["status"] = _status(payload)
    payload["side"] = _side(payload)
    payload["quantity"] = _quantity(payload)
    payload["filled_quantity"] = _filled_quantity(payload) or _quantity(payload)
    payload["average_price"] = _average_price(payload)
    payload["product"] = _product(payload)
    return payload


def _read_state(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_state(path: Path, updates: Mapping[str, Any]) -> None:
    payload = _read_state(path)
    payload.update(dict(updates))
    _position_manager._atomic_write_json(path, payload)


def _state_path(self: Any) -> Path:
    return Path(getattr(self, "_state_path", "positions.json"))


def _ledger_row(
    *,
    existing: Mapping[str, Any] | None,
    broker_payload: Mapping[str, Any],
    classification: str,
    broker_position_state: str | None,
    broker_position_qty: int | None,
    reason: str | None,
    managed: bool,
) -> dict[str, Any]:
    previous = dict(existing or {})
    order_id = str(broker_payload.get("order_id") or "").strip()
    symbol = _canonical(
        broker_payload.get("symbol") or broker_payload.get("tradingsymbol")
    )
    now = _now_iso()
    return {
        **previous,
        "order_id": order_id,
        "broker_order_id": order_id,
        "symbol": symbol,
        "tradingsymbol": symbol,
        "side": str(broker_payload.get("side") or "").strip().upper(),
        "quantity": _to_int(broker_payload.get("quantity")),
        "filled_quantity": _to_int(broker_payload.get("filled_quantity")),
        "average_price": _to_float(broker_payload.get("average_price")),
        "product": str(broker_payload.get("product") or "MIS").strip().upper(),
        "broker_status": str(broker_payload.get("status") or "UNKNOWN").strip().upper(),
        "classification": classification,
        "broker_position_state": broker_position_state,
        "broker_position_qty": broker_position_qty,
        "managed_by_bot": bool(managed),
        "reason": reason,
        "first_seen_at": str(previous.get("first_seen_at") or now),
        "last_seen_at": now,
        "updated_at": now,
    }


def _classification_changed(
    previous: Mapping[str, Any] | None, current: Mapping[str, Any]
) -> bool:
    if not previous:
        return True
    keys = (
        "broker_status",
        "classification",
        "broker_position_state",
        "broker_position_qty",
        "reason",
    )
    return any(previous.get(key) != current.get(key) for key in keys)


def _active_external_exposure(row: Mapping[str, Any], reason: str) -> dict[str, Any]:
    if reason == "active_external_order":
        status = "BROKER_EXTERNAL_ORDER_ACTIVE"
    elif reason == "broker_state_unverified":
        status = "BROKER_STATE_UNVERIFIED"
    else:
        status = "BROKER_POSITION_QUARANTINED"
    return {
        "symbol": row["symbol"],
        "tradingsymbol": row["symbol"],
        "quantity": abs(
            _to_int(row.get("filled_quantity")) or _to_int(row.get("quantity"))
        ),
        "side": str(row.get("side") or "").strip().upper(),
        "product": str(row.get("product") or "MIS").strip().upper(),
        "average_price": _to_float(row.get("average_price")),
        "status": status,
        "reason": reason,
        "intent": "BROKER_IMPORTED_ORDER",
        "order_id": str(row.get("order_id") or ""),
        "managed_position": False,
        "entry_accounting_allowed": False,
        "realized_pnl_accounting_allowed": False,
        "requires_history_recovery": reason == "broker_position_quarantined",
        "created_at": _now_iso(),
        "source": "broker_order_ledger",
    }


def _resolve_broker_order_fetcher(manager: Any) -> Any | None:
    broker = (
        getattr(manager, "_broker_client", None)
        or getattr(manager, "broker_client", None)
        or getattr(manager, "broker", None)
    )
    if broker is None:
        return None
    for name in ("get_orders", "list_orders", "orders", "fetch_orders"):
        fetcher = getattr(broker, name, None)
        if callable(fetcher):
            return fetcher
    return None


def _record_ledger(self: Any, row: Mapping[str, Any]) -> None:
    ledger = getattr(self, "_broker_order_ledger", None)
    if not isinstance(ledger, dict):
        ledger = {}
    ledger[str(row["order_id"])] = dict(row)
    self._broker_order_ledger = ledger


def _clear_exposure(self: Any, symbol: str) -> None:
    exposures = getattr(self, "_quarantined_broker_exposures", None)
    if isinstance(exposures, dict):
        exposures.pop(_canonical(symbol), None)
        self._quarantined_broker_exposures = exposures


def _set_exposure(self: Any, row: Mapping[str, Any], reason: str) -> None:
    exposures = getattr(self, "_quarantined_broker_exposures", None)
    if not isinstance(exposures, dict):
        exposures = {}
    exposure = _active_external_exposure(row, reason)
    exposures[str(exposure["symbol"])] = exposure
    self._quarantined_broker_exposures = exposures


def _classify_unknown(
    self: Any, payload: Mapping[str, Any]
) -> tuple[str, str | None, int | None, str | None]:
    status = str(payload.get("status") or "UNKNOWN").upper()
    symbol = str(payload.get("symbol") or "")
    filled_qty = _to_int(payload.get("filled_quantity"))
    if status in {"CANCELLED", "REJECTED", "EXPIRED"}:
        return "resolved_external_terminal", "flat", 0, None
    if status in {"PENDING", "OPEN", "PARTIALLY_FILLED"}:
        return "active_external_order", None, None, "active_external_order"
    if status == "FILLED" or filled_qty > 0:
        state, qty, error = _position_identity._broker_position_quantity(self, symbol)
        if state == "flat":
            return "resolved_external_flat", state, qty, None
        if state == "open":
            return (
                "broker_position_quarantined",
                state,
                qty,
                "broker_position_unowned_or_cost_basis_unresolved",
            )
        return "broker_state_unverified", state, qty, "broker_state_unverified"
    return "active_external_order", None, None, "active_external_order"


def _ledger_blocker(row: Mapping[str, Any]) -> str | None:
    classification = str(row.get("classification") or "")
    if classification == "active_external_order":
        return "active_external_order"
    if classification == "broker_state_unverified":
        return "broker_state_unverified"
    if classification == "broker_position_quarantined":
        return "broker_exposure_quarantined"
    return None


def _persist_extra_state(self: Any) -> None:
    ledger = getattr(self, "_broker_order_ledger", {}) or {}
    exposures = getattr(self, "_quarantined_broker_exposures", {}) or {}
    _write_state(
        _state_path(self),
        {
            "broker_order_ledger": dict(ledger) if isinstance(ledger, Mapping) else {},
            "quarantined_broker_exposures": (
                {
                    key: dict(value)
                    for key, value in exposures.items()
                    if isinstance(value, Mapping)
                }
                if isinstance(exposures, Mapping)
                else {}
            ),
        },
    )


def apply_patches() -> None:
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return
    cls = getattr(_position_manager, "PositionManager", None)
    if cls is None or getattr(cls, "_broker_order_ledger_patch", False):
        _PATCH_APPLIED = True
        return

    for name in (
        "__init__",
        "save_state",
        "load_state",
        "add_pending_order",
        "apply_broker_order_update",
        "current_entry_protection_blocker",
        "reconcile_now",
    ):
        if hasattr(cls, name):
            _ORIGINALS[f"PositionManager.{name}"] = getattr(cls, name)

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        _ORIGINALS["PositionManager.__init__"](self, *args, **kwargs)
        extra = _read_state(_state_path(self))
        ledger = extra.get("broker_order_ledger", {})
        self._broker_order_ledger = dict(ledger) if isinstance(ledger, Mapping) else {}
        exposures = extra.get("quarantined_broker_exposures")
        if isinstance(exposures, Mapping):
            self._quarantined_broker_exposures = {
                _canonical(key): dict(value)
                for key, value in exposures.items()
                if isinstance(value, Mapping)
            }

    def load_state(self: Any) -> None:
        _ORIGINALS["PositionManager.load_state"](self)
        extra = _read_state(_state_path(self))
        ledger = extra.get("broker_order_ledger", {})
        self._broker_order_ledger = dict(ledger) if isinstance(ledger, Mapping) else {}
        exposures = extra.get("quarantined_broker_exposures")
        if isinstance(exposures, Mapping):
            self._quarantined_broker_exposures = {
                _canonical(key): dict(value)
                for key, value in exposures.items()
                if isinstance(value, Mapping)
            }

    def save_state(self: Any) -> None:
        _ORIGINALS["PositionManager.save_state"](self)
        try:
            _persist_extra_state(self)
        except Exception as exc:  # noqa: BLE001
            log = getattr(getattr(self, "_logger", None), "error", None)
            if callable(log):
                log(
                    "BROKER_ORDER_LEDGER_SAVE_FAILED error=%s",
                    exc,
                    extra={
                        "event": "BROKER_ORDER_LEDGER_SAVE_FAILED",
                        "error": str(exc),
                    },
                )

    def add_pending_order(
        self: Any, order_id: str, symbol: str, *args: Any, **kwargs: Any
    ) -> Any:
        result = _ORIGINALS["PositionManager.add_pending_order"](
            self,
            order_id,
            _canonical(symbol),
            *args,
            **kwargs,
        )
        orders = getattr(self, "_orders", {})
        order = orders.get(str(order_id).strip()) if isinstance(orders, dict) else None
        if order is not None:
            broker_payload = {
                "order_id": str(order_id).strip(),
                "symbol": _canonical(getattr(order, "symbol", symbol)),
                "side": getattr(order, "side", None),
                "quantity": getattr(order, "quantity", 0),
                "filled_quantity": getattr(order, "filled_quantity", 0),
                "average_price": getattr(order, "fill_price", None)
                or getattr(order, "price", 0.0),
                "product": "MIS",
                "status": getattr(order, "status", "PENDING"),
            }
            row = _ledger_row(
                existing=(getattr(self, "_broker_order_ledger", {}) or {}).get(
                    str(order_id).strip()
                ),
                broker_payload=broker_payload,
                classification="managed_order",
                broker_position_state=None,
                broker_position_qty=None,
                reason=None,
                managed=True,
            )
            _record_ledger(self, row)
            with suppress(Exception):
                save_state(self)
        return result

    def apply_broker_order_update(
        self: Any, order_id: str, broker_payload: Mapping[str, Any]
    ) -> Any:
        payload = _row_to_payload(
            dict(broker_payload or {}, order_id=str(order_id).strip())
        )
        oid = str(payload.get("order_id") or order_id or "").strip()
        if not oid:
            return None
        payload["order_id"] = oid
        orders = getattr(self, "_orders", {})
        managed = isinstance(orders, dict) and oid in orders
        client_order_id = str(
            payload.get("client_order_id")
            or payload.get("clientOrderId")
            or payload.get("tag")
            or ""
        ).strip()
        if (
            not managed
            and client_order_id
            and isinstance(orders, dict)
            and client_order_id in orders
        ):
            oid = client_order_id
            payload["order_id"] = oid
            managed = True
        ledger = getattr(self, "_broker_order_ledger", {}) or {}
        previous = ledger.get(oid) if isinstance(ledger, Mapping) else None

        if managed:
            row = _ledger_row(
                existing=previous,
                broker_payload=payload,
                classification="managed_order",
                broker_position_state=None,
                broker_position_qty=None,
                reason=None,
                managed=True,
            )
            _record_ledger(self, row)
            result = _ORIGINALS["PositionManager.apply_broker_order_update"](
                self, oid, payload
            )
            with suppress(Exception):
                save_state(self)
            return result

        classification, broker_state, broker_qty, reason = _classify_unknown(
            self, payload
        )
        row = _ledger_row(
            existing=previous,
            broker_payload=payload,
            classification=classification,
            broker_position_state=broker_state,
            broker_position_qty=broker_qty,
            reason=reason,
            managed=False,
        )
        changed = _classification_changed(previous, row)
        _record_ledger(self, row)

        if classification in _TERMINAL_CLASSIFICATIONS:
            _clear_exposure(self, str(row.get("symbol") or ""))
            if changed:
                log = getattr(getattr(self, "_logger", None), "info", None)
                if callable(log):
                    log(
                        "BROKER_UNKNOWN_ORDER_RESOLVED order_id=%s symbol=%s classification=%s",
                        oid,
                        row.get("symbol"),
                        classification,
                        extra={
                            "event": "BROKER_UNKNOWN_ORDER_RESOLVED",
                            "order_id": oid,
                            "symbol": row.get("symbol"),
                            "classification": classification,
                            "broker_status": row.get("broker_status"),
                        },
                    )
            save_state(self)
            return None

        blocker = _ledger_blocker(row) or str(reason or classification)
        _set_exposure(self, row, blocker)
        if changed:
            log = getattr(getattr(self, "_logger", None), "warning", None)
            if callable(log):
                log(
                    "BROKER_EXTERNAL_ORDER_QUARANTINED order_id=%s symbol=%s classification=%s reason=%s",
                    oid,
                    row.get("symbol"),
                    classification,
                    blocker,
                    extra={
                        "event": "BROKER_EXTERNAL_ORDER_QUARANTINED",
                        "order_id": oid,
                        "symbol": row.get("symbol"),
                        "classification": classification,
                        "reason": blocker,
                        "broker_status": row.get("broker_status"),
                        "broker_position_state": broker_state,
                        "broker_position_qty": broker_qty,
                    },
                )
        save_state(self)
        return None

    def reconcile_broker_orders(
        self: Any, broker_orders: Any | None = None
    ) -> dict[str, int]:
        if broker_orders is None:
            fetcher = _resolve_broker_order_fetcher(self)
            if fetcher is None:
                return {"seen": 0, "managed": 0, "external": 0, "resolved": 0}
            broker_orders = fetcher()
        rows = sorted(_normalise_order_rows(broker_orders), key=_timestamp_key)
        counts = {"seen": 0, "managed": 0, "external": 0, "resolved": 0}
        for raw in rows:
            payload = _row_to_payload(raw)
            oid = str(payload.get("order_id") or "").strip()
            if not oid:
                continue
            counts["seen"] += 1
            apply_broker_order_update(self, oid, payload)
            ledger_row = (getattr(self, "_broker_order_ledger", {}) or {}).get(oid, {})
            classification = str(ledger_row.get("classification") or "")
            if classification == "managed_order":
                counts["managed"] += 1
            elif classification in _TERMINAL_CLASSIFICATIONS:
                counts["resolved"] += 1
            elif classification:
                counts["external"] += 1
        return counts

    def reconcile_now(self: Any) -> bool:
        result = bool(_ORIGINALS["PositionManager.reconcile_now"](self))
        try:
            counts = reconcile_broker_orders(self)
            if counts.get("seen", 0):
                log = getattr(getattr(self, "_logger", None), "info", None)
                if callable(log):
                    log(
                        "BROKER_ORDER_RECONCILE_OK seen=%s managed=%s external=%s resolved=%s",
                        counts["seen"],
                        counts["managed"],
                        counts["external"],
                        counts["resolved"],
                        extra={"event": "BROKER_ORDER_RECONCILE_OK", **counts},
                    )
        except Exception as exc:  # noqa: BLE001
            log = getattr(getattr(self, "_logger", None), "warning", None)
            if callable(log):
                log(
                    "BROKER_ORDER_RECONCILE_FAILED error=%s",
                    exc,
                    extra={"event": "BROKER_ORDER_RECONCILE_FAILED", "error": str(exc)},
                    exc_info=exc,
                )
        return result

    def current_entry_protection_blocker(
        self: Any, symbol: str | None = None
    ) -> str | None:
        ledger = getattr(self, "_broker_order_ledger", {}) or {}
        wanted = _canonical(symbol) if symbol else None
        if isinstance(ledger, Mapping):
            for row in ledger.values():
                if not isinstance(row, Mapping):
                    continue
                if str(row.get("classification") or "") not in _ACTIVE_CLASSIFICATIONS:
                    continue
                row_symbol = _canonical(row.get("symbol") or row.get("tradingsymbol"))
                if wanted is not None and row_symbol != wanted:
                    continue
                blocker = _ledger_blocker(row)
                if blocker:
                    return blocker
        original = _ORIGINALS.get("PositionManager.current_entry_protection_blocker")
        if callable(original):
            return original(self, _canonical(symbol) if symbol else None)
        return None

    def get_broker_order_ledger(
        self: Any, symbol: str | None = None
    ) -> dict[str, dict[str, Any]]:
        ledger = getattr(self, "_broker_order_ledger", {}) or {}
        wanted = _canonical(symbol) if symbol else None
        out: dict[str, dict[str, Any]] = {}
        if isinstance(ledger, Mapping):
            for order_id, row in ledger.items():
                if not isinstance(row, Mapping):
                    continue
                row_symbol = _canonical(row.get("symbol") or row.get("tradingsymbol"))
                if wanted is not None and row_symbol != wanted:
                    continue
                out[str(order_id)] = dict(row)
        return out

    cls.__init__ = __init__
    cls.load_state = load_state
    cls.save_state = save_state
    cls.add_pending_order = add_pending_order
    cls.apply_broker_order_update = apply_broker_order_update
    cls.reconcile_broker_orders = reconcile_broker_orders
    if "PositionManager.current_entry_protection_blocker" in _ORIGINALS:
        cls.current_entry_protection_blocker = current_entry_protection_blocker
    if "PositionManager.reconcile_now" in _ORIGINALS:
        cls.reconcile_now = reconcile_now
    cls.get_broker_order_ledger = get_broker_order_ledger
    cls._broker_order_ledger_patch = True
    _PATCH_APPLIED = True


apply_patches()

__all__ = ["apply_patches"]
