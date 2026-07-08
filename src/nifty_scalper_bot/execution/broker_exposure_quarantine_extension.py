"""Broker exposure quarantine extension.

Unresolved broker positions should be visible as quarantined exposures instead
of being represented only by an entry blocker. They remain excluded from normal
position/P&L accounting until cost basis is recovered.
"""

from __future__ import annotations

from contextlib import suppress
import time
from typing import Any

from nifty_scalper_bot.execution import position_identity_extension as _position_identity
from nifty_scalper_bot.execution import position_manager as _position_manager
from nifty_scalper_bot.utils.symbols import normalize_symbol

_PATCH_APPLIED = False
_ORIGINALS: dict[str, Any] = {}
_MANUAL_INTENTS = {"", "UNKNOWN", "BROKER_IMPORTED_ORDER", "MANUAL_ORDER_QUARANTINED"}


def _canonical(symbol: object) -> str:
    return normalize_symbol(str(symbol or ""))


def _net_quantity(row: dict[str, Any]) -> int:
    for key in ("quantity", "net_qty", "net_quantity", "netQuantity", "net"):
        if key not in row:
            continue
        try:
            return int(float(row.get(key) or 0))
        except Exception:
            continue
    return 0


def _row_symbol(row: Any) -> str:
    if not isinstance(row, dict):
        return ""
    return _canonical(row.get("symbol") or row.get("tradingsymbol"))


def _order_symbol(order: Any) -> str:
    return _canonical(getattr(order, "symbol", "") or getattr(order, "tradingsymbol", ""))


def _order_side(order: Any) -> str:
    return str(getattr(order, "side", "") or getattr(order, "transaction_type", "")).strip().upper()


def _order_quantity(order: Any) -> int:
    for name in ("filled_quantity", "quantity", "qty"):
        raw = getattr(order, name, None)
        if raw is None:
            continue
        with suppress(Exception):
            return abs(int(float(raw or 0)))
    return 0


def _position_for_symbol(manager: Any, symbol: str) -> Any | None:
    positions = getattr(manager, "_positions", None)
    if isinstance(positions, dict):
        return positions.get(symbol) or positions.get(_canonical(symbol))
    getter = getattr(manager, "get_open_positions", None)
    if callable(getter):
        with suppress(Exception):
            for position in getter() or []:
                if _canonical(getattr(position, "symbol", "")) == symbol:
                    return position
    return None


def _is_manual_reduction_order(manager: Any, order: Any) -> bool:
    symbol = _order_symbol(order)
    qty = _order_quantity(order)
    side = _order_side(order)
    if not symbol or qty <= 0 or side not in {"BUY", "SELL"}:
        return False
    existing = _position_for_symbol(manager, symbol)
    if existing is None:
        return False
    existing_side = str(getattr(existing, "side", "") or "").strip().upper()
    with suppress(Exception):
        existing_qty = abs(int(float(getattr(existing, "quantity", 0) or 0)))
    if existing_qty <= 0 or qty > existing_qty:
        return False
    if existing_side == "LONG" and side == "SELL":
        return True
    if existing_side == "SHORT" and side == "BUY":
        return True
    return False


def _quarantined_exposure(row: dict[str, Any], reason: str) -> dict[str, Any]:
    symbol = _row_symbol(row)
    qty = _net_quantity(row)
    side = "LONG" if qty > 0 else "SHORT" if qty < 0 else "FLAT"
    out = dict(row)
    out.update(
        {
            "symbol": symbol,
            "tradingsymbol": symbol,
            "quantity": abs(qty),
            "signed_quantity": qty,
            "side": side,
            "status": "BROKER_POSITION_QUARANTINED",
            "reason": reason,
            "cost_basis_unresolved": True,
            "managed_position": False,
            "entry_accounting_allowed": False,
            "realized_pnl_accounting_allowed": False,
            "requires_history_recovery": True,
        }
    )
    return out


def _manual_order_exposure(order: Any, intent: str) -> dict[str, Any] | None:
    symbol = _order_symbol(order)
    if not symbol:
        return None
    qty = _order_quantity(order)
    try:
        price = float(
            getattr(order, "average_price", 0.0)
            or getattr(order, "fill_price", 0.0)
            or getattr(order, "price", 0.0)
            or 0.0
        )
    except Exception:
        price = 0.0
    return {
        "symbol": symbol,
        "tradingsymbol": symbol,
        "quantity": abs(qty),
        "signed_quantity": qty,
        "side": _order_side(order),
        "product": str(getattr(order, "product", "MIS") or "MIS").upper(),
        "average_price": price,
        "status": "MANUAL_ORDER_QUARANTINED",
        "reason": "manual_order_quarantined",
        "intent": intent,
        "order_id": str(getattr(order, "order_id", "") or ""),
        "managed_position": False,
        "entry_accounting_allowed": False,
        "realized_pnl_accounting_allowed": False,
        "requires_history_recovery": True,
        "created_at": time.time(),
        "source": "broker_order_update",
    }


def _build_exposures(prepared: Any, unresolved: set[str]) -> dict[str, dict[str, Any]]:
    if not isinstance(prepared, list) or not unresolved:
        return {}
    exposures: dict[str, dict[str, Any]] = {}
    for row in prepared:
        if not isinstance(row, dict):
            continue
        symbol = _row_symbol(row)
        if symbol in unresolved:
            exposures[symbol] = _quarantined_exposure(row, "cost_basis_unresolved")
    return exposures


def apply_patches() -> None:
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return
    cls = getattr(_position_manager, "PositionManager", None)
    if cls is None or getattr(cls, "_broker_exposure_quarantine_patch", False):
        _PATCH_APPLIED = True
        return
    _ORIGINALS["PositionManager.__init__"] = cls.__init__
    _ORIGINALS["PositionManager.synchronize_with_broker"] = cls.synchronize_with_broker
    if hasattr(cls, "current_entry_protection_blocker"):
        _ORIGINALS["PositionManager.current_entry_protection_blocker"] = cls.current_entry_protection_blocker
    if hasattr(cls, "_handle_filled_order"):
        _ORIGINALS["PositionManager._handle_filled_order"] = cls._handle_filled_order

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        _ORIGINALS["PositionManager.__init__"](self, *args, **kwargs)
        self._quarantined_broker_exposures = {}

    def synchronize_with_broker(self: Any, broker_positions: Any) -> Any:
        prepared, unresolved = _position_identity._prepare_broker_positions(self, broker_positions)
        self._quarantined_broker_exposures = _build_exposures(prepared, set(unresolved))
        return _ORIGINALS["PositionManager.synchronize_with_broker"](self, broker_positions)

    def current_entry_protection_blocker(self: Any, symbol: str | None = None) -> str | None:
        exposures = dict(getattr(self, "_quarantined_broker_exposures", {}) or {})
        if exposures and (symbol is None or _canonical(symbol) in exposures):
            return "broker_exposure_quarantined"
        original = _ORIGINALS.get("PositionManager.current_entry_protection_blocker")
        if callable(original):
            return original(self, symbol)
        return None

    def get_quarantined_broker_exposures(self: Any, symbol: str | None = None) -> dict[str, dict[str, Any]] | list[dict[str, Any]]:
        exposures = dict(getattr(self, "_quarantined_broker_exposures", {}) or {})
        if symbol is None:
            return {key: dict(value) for key, value in exposures.items()}
        exposure = exposures.get(_canonical(symbol))
        return [dict(exposure)] if exposure is not None else []

    def clear_quarantined_broker_exposure(self: Any, symbol: str) -> bool:
        exposures = getattr(self, "_quarantined_broker_exposures", {})
        if not isinstance(exposures, dict):
            return False
        return exposures.pop(_canonical(symbol), None) is not None

    def _handle_filled_order(self: Any, order: Any) -> Any:
        intent = str(getattr(order, "intent", "UNKNOWN") or "UNKNOWN").strip().upper()
        original = _ORIGINALS["PositionManager._handle_filled_order"]
        if intent in _MANUAL_INTENTS and _is_manual_reduction_order(self, order):
            original_intent = getattr(order, "intent", None)
            with suppress(Exception):
                setattr(order, "intent", "REDUCE")
            result = original(self, order)
            with suppress(Exception):
                setattr(order, "intent", original_intent)
            symbol = _order_symbol(order)
            exposures = dict(getattr(self, "_quarantined_broker_exposures", {}) or {})
            exposures.pop(symbol, None)
            self._quarantined_broker_exposures = exposures
            logger = getattr(self, "_logger", None)
            log = getattr(logger, "warning", None)
            if callable(log):
                log(
                    "MANUAL_EXIT_RECOGNISED order_id=%s symbol=%s side=%s qty=%s",
                    getattr(order, "order_id", None),
                    symbol,
                    _order_side(order),
                    _order_quantity(order),
                    extra={
                        "event": "MANUAL_EXIT_RECOGNISED",
                        "order_id": getattr(order, "order_id", None),
                        "symbol": symbol,
                        "side": _order_side(order),
                        "quantity": _order_quantity(order),
                        "intent": "REDUCE",
                    },
                )
            return result

        result = original(self, order)
        if intent in _MANUAL_INTENTS:
            exposure = _manual_order_exposure(order, intent)
            if exposure is not None:
                exposures = dict(getattr(self, "_quarantined_broker_exposures", {}) or {})
                exposures[exposure["symbol"]] = exposure
                self._quarantined_broker_exposures = exposures
        return result

    cls.__init__ = __init__
    cls.synchronize_with_broker = synchronize_with_broker
    if "PositionManager.current_entry_protection_blocker" in _ORIGINALS:
        cls.current_entry_protection_blocker = current_entry_protection_blocker
    if "PositionManager._handle_filled_order" in _ORIGINALS:
        cls._handle_filled_order = _handle_filled_order
    cls.get_quarantined_broker_exposures = get_quarantined_broker_exposures
    cls.clear_quarantined_broker_exposure = clear_quarantined_broker_exposure
    cls._broker_exposure_quarantine_patch = True
    _PATCH_APPLIED = True


apply_patches()

__all__ = [
    "apply_patches",
    "_build_exposures",
    "_manual_order_exposure",
    "_quarantined_exposure",
    "_is_manual_reduction_order",
]
