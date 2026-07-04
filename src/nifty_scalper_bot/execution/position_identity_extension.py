"""Canonical PositionManager ingress patch for broker and pending-order paths."""

from __future__ import annotations

from contextlib import suppress
from typing import Any

from nifty_scalper_bot.execution import position_manager as _position_manager
from nifty_scalper_bot.utils.symbols import normalize_symbol

_PATCH_APPLIED = False
_ORIGINALS: dict[str, Any] = {}


def _canonical_key(symbol: object) -> str:
    return normalize_symbol(str(symbol or ""))


def _canonicalize_payload_symbol(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    cloned = dict(payload)
    for key in (
        "symbol",
        "tradingsymbol",
        "trading_symbol",
        "instrument",
        "instrument_symbol",
    ):
        if cloned.get(key):
            cloned[key] = _canonical_key(cloned[key])
    return cloned


def apply_patches() -> None:
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return
    cls = getattr(_position_manager, "PositionManager", None)
    if cls is None or getattr(cls, "_canonical_position_ingress_patch", False):
        _PATCH_APPLIED = True
        return

    for name in (
        "_symbol_lifecycle_lock_for",
        "add_pending_order",
        "get_pending_orders",
        "synchronize_with_broker",
        "apply_broker_order_update",
    ):
        if hasattr(cls, name):
            _ORIGINALS[f"PositionManager.{name}"] = getattr(cls, name)

    def _symbol_lifecycle_lock_for(self: Any, symbol: str) -> Any:
        return _ORIGINALS["PositionManager._symbol_lifecycle_lock_for"](
            self,
            _canonical_key(symbol),
        )

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

    def get_pending_orders(self: Any, symbol: str | None = None) -> Any:
        return _ORIGINALS["PositionManager.get_pending_orders"](
            self,
            _canonical_key(symbol) if symbol else None,
        )

    def synchronize_with_broker(self: Any, broker_positions: Any) -> Any:
        canonical_positions = None
        if broker_positions is not None:
            with suppress(TypeError):
                canonical_positions = [
                    _canonicalize_payload_symbol(position)
                    for position in broker_positions
                ]
        if canonical_positions is None:
            canonical_positions = broker_positions
        result = _ORIGINALS["PositionManager.synchronize_with_broker"](
            self,
            canonical_positions,
        )
        positions = getattr(self, "_positions", None)
        if isinstance(positions, dict):
            canonical: dict[str, Any] = {}
            for raw_key, position in list(positions.items()):
                key = _canonical_key(getattr(position, "symbol", raw_key))
                if key:
                    with suppress(Exception):
                        position.symbol = key
                    canonical[key] = position
            positions.clear()
            positions.update(canonical)
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

    if "PositionManager._symbol_lifecycle_lock_for" in _ORIGINALS:
        cls._symbol_lifecycle_lock_for = _symbol_lifecycle_lock_for
    if "PositionManager.add_pending_order" in _ORIGINALS:
        cls.add_pending_order = add_pending_order
    if "PositionManager.get_pending_orders" in _ORIGINALS:
        cls.get_pending_orders = get_pending_orders
    if "PositionManager.synchronize_with_broker" in _ORIGINALS:
        cls.synchronize_with_broker = synchronize_with_broker
    if "PositionManager.apply_broker_order_update" in _ORIGINALS:
        cls.apply_broker_order_update = apply_broker_order_update
    cls._canonical_position_ingress_patch = True
    _PATCH_APPLIED = True


apply_patches()

__all__ = ["apply_patches"]
