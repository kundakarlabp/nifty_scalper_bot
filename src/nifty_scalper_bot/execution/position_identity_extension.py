"""Compatibility overlay for remaining PositionManager ingress guards.

Broker-position reconciliation identity, cost-basis preparation, lifecycle
preservation, quarantine blocking, and single-flight ownership now live natively
in PositionManager.
"""

from __future__ import annotations

from contextlib import suppress
import inspect
from typing import Any

from nifty_scalper_bot.execution import live_safety_identity as _live_safety_identity
from nifty_scalper_bot.execution import position_manager as _position_manager
from nifty_scalper_bot.execution.position_reconciliation_identity import (
    _canonical_key,
    _canonicalize_payload_symbol,
)
from nifty_scalper_bot.execution.position_snapshot import (
    PositionSnapshotError,
    decode_position_snapshot,
)

_PATCH_APPLIED = False
_ORIGINALS: dict[str, Any] = {}
_QUARANTINE_INTENTS = {
    "",
    "UNKNOWN",
    "BROKER_IMPORTED_ORDER",
    "MANUAL_ORDER_QUARANTINED",
}


def _install_position_ownership_property() -> None:
    """Expose durable bot ownership to existing orphan/capital diagnostics."""

    position_cls = getattr(_position_manager, "Position", None)
    if position_cls is None or hasattr(position_cls, "strategy_name"):
        return

    def strategy_name(position: Any) -> str:
        return "BotManaged" if str(getattr(position, "order_id", "") or "").strip() else ""

    position_cls.strategy_name = property(strategy_name)


def _restore_persistent_state_methods(cls: Any) -> None:
    """Undo broad restore/save canonicalization while retaining live ingress guards."""

    live_originals = getattr(_live_safety_identity, "_ORIGINALS", {})
    for name in ("__init__", "save_state"):
        original = live_originals.get(f"PositionManager.{name}")
        if callable(original):
            setattr(cls, name, original)


def _resolve_broker_position_fetcher(manager: Any) -> Any | None:
    resolver = getattr(manager, "_resolve_broker_position_fetcher", None)
    if callable(resolver):
        with suppress(Exception):
            fetcher = resolver()
            if callable(fetcher):
                return fetcher
    broker = (
        getattr(manager, "_broker_client", None)
        or getattr(manager, "broker_client", None)
        or getattr(manager, "broker", None)
    )
    if broker is None:
        return None
    for name in ("get_positions", "list_positions", "positions", "fetch_positions"):
        fetcher = getattr(broker, name, None)
        if callable(fetcher):
            return fetcher
    return None


def _broker_position_quantity(manager: Any, symbol: str) -> tuple[str, int, str | None]:
    """Return broker truth for *symbol*: flat, open, or unverified."""

    fetcher = _resolve_broker_position_fetcher(manager)
    if fetcher is None:
        return "unverified", 0, "broker_position_fetcher_missing"
    try:
        payload = fetcher()
        if inspect.isawaitable(payload):
            with suppress(Exception):
                close = getattr(payload, "close", None)
                if callable(close):
                    close()
            return "unverified", 0, "async_broker_position_fetcher_unsupported"
        snapshot = decode_position_snapshot(payload)
        qty = int(snapshot.quantity_for(symbol))
    except PositionSnapshotError as exc:
        return "unverified", 0, f"position_snapshot_invalid:{exc}"
    except Exception as exc:  # noqa: BLE001 - broker boundary must fail closed
        return "unverified", 0, f"broker_position_fetch_failed:{type(exc).__name__}:{exc}"
    return ("flat", 0, None) if qty == 0 else ("open", qty, None)


def _clear_symbol_quarantine(manager: Any, symbol: str) -> None:
    lock = getattr(manager, "_lock", None)
    if lock is None:
        exposures = getattr(manager, "_quarantined_broker_exposures", None)
        if isinstance(exposures, dict):
            exposures.pop(symbol, None)
        unresolved = getattr(manager, "_cost_basis_unresolved_symbols", None)
        if isinstance(unresolved, set):
            unresolved.discard(symbol)
        return
    with lock:
        exposures = getattr(manager, "_quarantined_broker_exposures", None)
        if isinstance(exposures, dict):
            exposures.pop(symbol, None)
        unresolved = getattr(manager, "_cost_basis_unresolved_symbols", None)
        if isinstance(unresolved, set):
            unresolved.discard(symbol)


def apply_patches() -> None:
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return
    _install_position_ownership_property()
    cls = getattr(_position_manager, "PositionManager", None)
    if cls is None or getattr(cls, "_canonical_position_ingress_patch", False):
        _PATCH_APPLIED = True
        return

    _restore_persistent_state_methods(cls)

    for name in (
        "_symbol_lifecycle_lock_for",
        "add_pending_order",
        "get_pending_orders",
        "apply_broker_order_update",
        "current_entry_protection_blocker",
        "_handle_filled_order",
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

    def current_entry_protection_blocker(
        self: Any, symbol: str | None = None
    ) -> str | None:
        original = _ORIGINALS.get("PositionManager.current_entry_protection_blocker")
        if callable(original):
            return original(self, _canonical_key(symbol) if symbol else None)
        return None

    def _handle_filled_order(self: Any, order: Any) -> Any:
        intent = str(getattr(order, "intent", "UNKNOWN") or "UNKNOWN").strip().upper()
        if intent in _QUARANTINE_INTENTS:
            symbol = _canonical_key(getattr(order, "symbol", None))
            state, broker_qty, broker_error = _broker_position_quantity(self, symbol)
            logger = getattr(self, "_logger", None)
            log_warning = getattr(logger, "warning", None)
            log_info = getattr(logger, "info", None)

            if state == "flat":
                _clear_symbol_quarantine(self, symbol)
                if callable(log_info):
                    log_info(
                        "BROKER_FLAT_CONFIRMED_FOR_UNKNOWN_ORDER order_id=%s symbol=%s side=%s",
                        getattr(order, "order_id", None),
                        symbol,
                        getattr(order, "side", None),
                        extra={
                            "event": "BROKER_FLAT_CONFIRMED_FOR_UNKNOWN_ORDER",
                            "order_id": getattr(order, "order_id", None),
                            "symbol": symbol,
                            "side": getattr(order, "side", None),
                            "intent": intent,
                        },
                    )
                return _position_manager.FillApplicationResult(
                    accounting_finalized=True,
                    lifecycle_resolved=True,
                    reason="broker_flat_confirmed_unknown_order",
                )

            if state == "unverified":
                if callable(log_warning):
                    log_warning(
                        "BROKER_STATE_UNVERIFIED_FOR_UNKNOWN_ORDER order_id=%s symbol=%s side=%s reason=%s",
                        getattr(order, "order_id", None),
                        symbol,
                        getattr(order, "side", None),
                        broker_error,
                        extra={
                            "event": "BROKER_STATE_UNVERIFIED_FOR_UNKNOWN_ORDER",
                            "order_id": getattr(order, "order_id", None),
                            "symbol": symbol,
                            "side": getattr(order, "side", None),
                            "intent": intent,
                            "reason": broker_error,
                        },
                    )
                return _position_manager.FillApplicationResult(
                    reason="broker_state_unverified"
                )

            if callable(log_warning):
                log_warning(
                    "BROKER_POSITION_QUARANTINED_FOR_UNKNOWN_ORDER order_id=%s symbol=%s side=%s broker_qty=%s",
                    getattr(order, "order_id", None),
                    symbol,
                    getattr(order, "side", None),
                    broker_qty,
                    extra={
                        "event": "BROKER_POSITION_QUARANTINED_FOR_UNKNOWN_ORDER",
                        "order_id": getattr(order, "order_id", None),
                        "symbol": symbol,
                        "side": getattr(order, "side", None),
                        "intent": intent,
                        "broker_qty": broker_qty,
                    },
                )
            return _position_manager.FillApplicationResult(
                reason="broker_position_unowned_or_cost_basis_unresolved"
            )

        result = _ORIGINALS["PositionManager._handle_filled_order"](self, order)
        if (
            intent == "ENTRY"
            and int(getattr(order, "pre_order_quantity", 0) or 0) == 0
            and getattr(result, "reason", "")
            == "entry_fill_already_reflected_by_broker_sync"
        ):
            symbol = _canonical_key(getattr(order, "symbol", None))
            positions = getattr(self, "_positions", {})
            position = positions.get(symbol) if isinstance(positions, dict) else None
            basis = float(
                getattr(order, "last_cumulative_average_price", 0.0)
                or getattr(order, "fill_price", 0.0)
                or 0.0
            )
            if (
                position is not None
                and str(getattr(position, "order_id", "") or "").strip()
                == str(getattr(order, "order_id", "") or "").strip()
                and basis > 0.0
            ):
                broker_day_basis = float(getattr(position, "entry_price", 0.0) or 0.0)
                position.entry_price = basis
                logger = getattr(self, "_logger", None)
                log_info = getattr(logger, "info", None)
                if callable(log_info):
                    log_info(
                        "ENTRY_LIFECYCLE_BASIS_RESTORED order_id=%s symbol=%s broker_day_basis=%.2f fill_basis=%.2f",
                        getattr(order, "order_id", None),
                        symbol,
                        broker_day_basis,
                        basis,
                        extra={
                            "event": "ENTRY_LIFECYCLE_BASIS_RESTORED",
                            "order_id": getattr(order, "order_id", None),
                            "symbol": symbol,
                            "broker_day_basis": broker_day_basis,
                            "fill_basis": basis,
                        },
                    )
        return result

    if "PositionManager._symbol_lifecycle_lock_for" in _ORIGINALS:
        cls._symbol_lifecycle_lock_for = _symbol_lifecycle_lock_for
    if "PositionManager.add_pending_order" in _ORIGINALS:
        cls.add_pending_order = add_pending_order
    if "PositionManager.get_pending_orders" in _ORIGINALS:
        cls.get_pending_orders = get_pending_orders
    if "PositionManager.apply_broker_order_update" in _ORIGINALS:
        cls.apply_broker_order_update = apply_broker_order_update
    if "PositionManager.current_entry_protection_blocker" in _ORIGINALS:
        cls.current_entry_protection_blocker = current_entry_protection_blocker
    if "PositionManager._handle_filled_order" in _ORIGINALS:
        cls._handle_filled_order = _handle_filled_order
    cls._canonical_position_ingress_patch = True
    _PATCH_APPLIED = True


apply_patches()

__all__ = ["apply_patches"]
