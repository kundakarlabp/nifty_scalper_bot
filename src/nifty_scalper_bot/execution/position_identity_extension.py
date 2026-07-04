"""Canonical PositionManager ingress patch for broker and pending-order paths.

Follow-up scope: broker reconciliation ownership and orphan protection are handled
by the loaded runtime guards in this module.
"""

from __future__ import annotations

from contextlib import suppress
import threading
from typing import Any

from nifty_scalper_bot.execution import live_safety_identity as _live_safety_identity
from nifty_scalper_bot.execution import position_manager as _position_manager
from nifty_scalper_bot.utils.symbols import normalize_symbol

_PATCH_APPLIED = False
_ORIGINALS: dict[str, Any] = {}
_SYMBOL_FIELDS = ("symbol", "tradingsymbol", "trading_symbol")
_AVG_PRICE_FIELDS = ("average_price", "avg_price", "buy_price", "price")
_QTY_FIELDS = ("quantity", "net_qty", "net_quantity", "netQuantity", "net")


def _canonical_key(symbol: object) -> str:
    return normalize_symbol(str(symbol or ""))


def _canonicalize_payload_symbol(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    cloned = dict(payload)
    for key in _SYMBOL_FIELDS:
        value = cloned.get(key)
        if isinstance(value, str) and value.strip():
            cloned[key] = _canonical_key(value)
    return cloned


def _canonicalize_broker_positions(broker_positions: Any) -> Any:
    if broker_positions is None:
        return None
    if isinstance(broker_positions, dict):
        return [_canonicalize_payload_symbol(broker_positions)]
    try:
        return [_canonicalize_payload_symbol(position) for position in broker_positions]
    except TypeError:
        return broker_positions


def _positive_float(payload: dict[str, Any], keys: tuple[str, ...]) -> float:
    for key in keys:
        with suppress(Exception):
            value = float(payload.get(key) or 0.0)
            if value > 0.0:
                return value
    return 0.0


def _net_quantity(payload: dict[str, Any]) -> int:
    for key in _QTY_FIELDS:
        if key not in payload:
            continue
        with suppress(Exception):
            return int(float(payload.get(key) or 0))
    return 0


def _prepare_broker_positions(manager: Any, broker_positions: Any) -> tuple[Any, set[str]]:
    canonicalized = _canonicalize_broker_positions(broker_positions)
    if not isinstance(canonicalized, list):
        return canonicalized, set()
    positions = getattr(manager, "_positions", {})
    unresolved: set[str] = set()
    prepared: list[Any] = []
    for row in canonicalized:
        if not isinstance(row, dict):
            prepared.append(row)
            continue
        cloned = dict(row)
        symbol = _canonical_key(cloned.get("tradingsymbol") or cloned.get("symbol"))
        if symbol:
            cloned["tradingsymbol"] = symbol
            cloned["symbol"] = symbol
        avg_price = _positive_float(cloned, _AVG_PRICE_FIELDS)
        if _net_quantity(cloned) != 0 and avg_price <= 0.0:
            existing = positions.get(symbol) if isinstance(positions, dict) else None
            existing_entry = float(getattr(existing, "entry_price", 0.0) or 0.0) if existing else 0.0
            if existing_entry > 0.0:
                cloned["average_price"] = existing_entry
            else:
                unresolved.add(symbol)
        prepared.append(cloned)
    return prepared, unresolved


def _canonicalize_position_store(manager: Any) -> None:
    positions = getattr(manager, "_positions", None)
    if not isinstance(positions, dict):
        return
    canonical: dict[str, Any] = {}
    for raw_key, position in list(positions.items()):
        key = _canonical_key(getattr(position, "symbol", raw_key))
        if not key:
            key = str(raw_key).strip().upper()
        with suppress(Exception):
            position.symbol = key
        existing = canonical.get(key)
        if existing is None:
            canonical[key] = position
            continue
        with suppress(Exception):
            if abs(int(getattr(position, "quantity", 0) or 0)) > abs(
                int(getattr(existing, "quantity", 0) or 0)
            ):
                canonical[key] = position
    positions.clear()
    positions.update(canonical)


def _restore_persistent_state_methods(cls: Any) -> None:
    """Undo broad restore/save canonicalization while retaining live ingress guards."""

    live_originals = getattr(_live_safety_identity, "_ORIGINALS", {})
    for name in ("__init__", "save_state"):
        original = live_originals.get(f"PositionManager.{name}")
        if callable(original):
            setattr(cls, name, original)


def apply_patches() -> None:
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return
    cls = getattr(_position_manager, "PositionManager", None)
    if cls is None or getattr(cls, "_canonical_position_ingress_patch", False):
        _PATCH_APPLIED = True
        return

    _restore_persistent_state_methods(cls)

    for name in (
        "__init__",
        "_symbol_lifecycle_lock_for",
        "reconcile_now",
        "add_pending_order",
        "get_pending_orders",
        "synchronize_with_broker",
        "apply_broker_order_update",
        "current_entry_protection_blocker",
    ):
        if hasattr(cls, name):
            _ORIGINALS[f"PositionManager.{name}"] = getattr(cls, name)

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        _ORIGINALS["PositionManager.__init__"](self, *args, **kwargs)
        self._single_reconcile_lock = threading.Lock()
        self._single_reconcile_generation = 0
        self._single_reconcile_coalesced = 0
        self._cost_basis_unresolved_symbols = set()

    def _symbol_lifecycle_lock_for(self: Any, symbol: str) -> Any:
        return _ORIGINALS["PositionManager._symbol_lifecycle_lock_for"](
            self,
            _canonical_key(symbol),
        )

    def reconcile_now(self: Any) -> bool:
        lock = getattr(self, "_single_reconcile_lock", None)
        if lock is None:
            self._single_reconcile_lock = threading.Lock()
            lock = self._single_reconcile_lock
        if not lock.acquire(False):
            self._single_reconcile_coalesced = int(getattr(self, "_single_reconcile_coalesced", 0)) + 1
            return bool(getattr(self, "_last_reconcile_success_at", None))
        try:
            self._single_reconcile_generation = int(getattr(self, "_single_reconcile_generation", 0)) + 1
            return bool(_ORIGINALS["PositionManager.reconcile_now"](self))
        finally:
            lock.release()

    def add_pending_order(
        self: Any,
        order_id: str,
        symbol: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return _ORIGINALS["PositionManager.add_pending_order"](
            self,
            order_id,
            _canonical_key(symbol),
            *args,
            **kwargs,
        )

    def get_pending_orders(
        self: Any,
        symbol: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return _ORIGINALS["PositionManager.get_pending_orders"](
            self,
            _canonical_key(symbol) if symbol else None,
            *args,
            **kwargs,
        )

    def synchronize_with_broker(self: Any, broker_positions: Any) -> Any:
        prepared, unresolved = _prepare_broker_positions(self, broker_positions)
        self._cost_basis_unresolved_symbols = set(unresolved)
        if unresolved:
            raise ValueError("cost_basis_unresolved")
        result = _ORIGINALS["PositionManager.synchronize_with_broker"](
            self,
            prepared,
        )
        _canonicalize_position_store(self)
        return result

    def apply_broker_order_update(
        self: Any,
        order_id: str,
        broker_payload: Any,
    ) -> Any:
        return _ORIGINALS["PositionManager.apply_broker_order_update"](
            self,
            order_id,
            _canonicalize_payload_symbol(broker_payload),
        )

    def current_entry_protection_blocker(self: Any, symbol: str | None = None) -> str | None:
        unresolved = set(getattr(self, "_cost_basis_unresolved_symbols", set()) or set())
        if unresolved and (symbol is None or _canonical_key(symbol) in unresolved):
            return "cost_basis_unresolved"
        original = _ORIGINALS.get("PositionManager.current_entry_protection_blocker")
        if callable(original):
            return original(self, _canonical_key(symbol) if symbol else None)
        return None

    if "PositionManager.__init__" in _ORIGINALS:
        cls.__init__ = __init__
    if "PositionManager._symbol_lifecycle_lock_for" in _ORIGINALS:
        cls._symbol_lifecycle_lock_for = _symbol_lifecycle_lock_for
    if "PositionManager.reconcile_now" in _ORIGINALS:
        cls.reconcile_now = reconcile_now
    if "PositionManager.add_pending_order" in _ORIGINALS:
        cls.add_pending_order = add_pending_order
    if "PositionManager.get_pending_orders" in _ORIGINALS:
        cls.get_pending_orders = get_pending_orders
    if "PositionManager.synchronize_with_broker" in _ORIGINALS:
        cls.synchronize_with_broker = synchronize_with_broker
    if "PositionManager.apply_broker_order_update" in _ORIGINALS:
        cls.apply_broker_order_update = apply_broker_order_update
    if "PositionManager.current_entry_protection_blocker" in _ORIGINALS:
        cls.current_entry_protection_blocker = current_entry_protection_blocker
    cls._canonical_position_ingress_patch = True
    _PATCH_APPLIED = True


apply_patches()

__all__ = ["apply_patches"]
