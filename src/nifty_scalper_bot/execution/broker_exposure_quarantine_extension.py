"""Compatibility wrapper for manual broker-fill quarantine handling.

PositionManager natively owns quarantine registry, persistence, broker-position
synchronization, accessors, and entry-blocker semantics. This overlay remains
only for the not-yet-migrated manual/unknown fill orchestration.
"""

from __future__ import annotations

from contextlib import suppress
import time
from typing import Any

from nifty_scalper_bot.execution import position_manager as _position_manager
from nifty_scalper_bot.utils.symbols import normalize_symbol

_PATCH_APPLIED = False
_ORIGINALS: dict[str, Any] = {}
_MANUAL_INTENTS = {"", "UNKNOWN", "BROKER_IMPORTED_ORDER", "MANUAL_ORDER_QUARANTINED"}


def _canonical(symbol: object) -> str:
    return normalize_symbol(str(symbol or ""))


def _order_symbol(order: Any) -> str:
    return _canonical(getattr(order, "symbol", "") or getattr(order, "tradingsymbol", ""))


def _order_side(order: Any) -> str:
    return str(
        getattr(order, "side", "") or getattr(order, "transaction_type", "")
    ).strip().upper()


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
    existing_qty = 0
    with suppress(Exception):
        existing_qty = abs(int(float(getattr(existing, "quantity", 0) or 0)))
    if existing_qty <= 0 or qty > existing_qty:
        return False
    if existing_side == "LONG" and side == "SELL":
        return True
    if existing_side == "SHORT" and side == "BUY":
        return True
    return False


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
        "status": "BROKER_POSITION_QUARANTINED",
        "reason": "broker_position_unowned_or_cost_basis_unresolved",
        "intent": intent,
        "order_id": str(getattr(order, "order_id", "") or ""),
        "managed_position": False,
        "entry_accounting_allowed": False,
        "realized_pnl_accounting_allowed": False,
        "requires_history_recovery": True,
        "created_at": time.time(),
        "source": "broker_order_update",
    }


def apply_patches() -> None:
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return
    cls = getattr(_position_manager, "PositionManager", None)
    if cls is None or getattr(cls, "_broker_exposure_quarantine_patch", False):
        _PATCH_APPLIED = True
        return
    if hasattr(cls, "_handle_filled_order"):
        _ORIGINALS["PositionManager._handle_filled_order"] = cls._handle_filled_order

    def _handle_filled_order(self: Any, order: Any) -> Any:
        intent = str(getattr(order, "intent", "UNKNOWN") or "UNKNOWN").strip().upper()
        original = _ORIGINALS["PositionManager._handle_filled_order"]
        if intent in _MANUAL_INTENTS and _is_manual_reduction_order(self, order):
            with suppress(Exception):
                setattr(order, "intent", "REDUCE")
            result = original(self, order)
            symbol = _order_symbol(order)
            with self._lock:
                self._quarantined_broker_exposures.pop(symbol, None)
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
            symbol = _order_symbol(order)
            result_reason = str(getattr(result, "reason", "") or "")
            with self._lock:
                if result_reason == "broker_flat_confirmed_unknown_order":
                    self._quarantined_broker_exposures.pop(symbol, None)
                    return result
                exposure = _manual_order_exposure(order, intent)
                if exposure is not None:
                    if result_reason == "broker_state_unverified":
                        exposure["status"] = "BROKER_STATE_UNVERIFIED"
                        exposure["reason"] = "broker_state_unverified"
                        exposure["requires_history_recovery"] = False
                    elif result_reason:
                        exposure["reason"] = result_reason
                    self._quarantined_broker_exposures[exposure["symbol"]] = exposure
        return result

    if "PositionManager._handle_filled_order" in _ORIGINALS:
        cls._handle_filled_order = _handle_filled_order
    cls._broker_exposure_quarantine_patch = True
    _PATCH_APPLIED = True


apply_patches()

__all__ = [
    "apply_patches",
    "_manual_order_exposure",
    "_is_manual_reduction_order",
]
