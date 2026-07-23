"""Runtime safety patches for immutable exit identity and canonical positions.

This module is deliberately narrow: it patches only the live execution identity
surface that caused the production incident where a protective SELL could lose
EXIT identity and a bare option symbol could diverge from its NFO-qualified
canonical key.
"""

from __future__ import annotations

from contextlib import suppress
import time
from typing import Any, Mapping

from nifty_scalper_bot.execution import bracket_core as _bracket_core
from nifty_scalper_bot.execution import position_manager as _position_manager
from nifty_scalper_bot.utils.symbols import normalize_symbol

_PATCH_APPLIED = False
_ORIGINALS: dict[str, Any] = {}


def _safe_trade_lifecycle_id(bracket: Any) -> str | None:
    value = getattr(bracket, "trade_lifecycle_id", None)
    if value:
        return str(value)
    entry_order_id = getattr(bracket, "entry_order_id", None)
    return str(entry_order_id) if entry_order_id else None


def _exit_identity_kwargs(bracket: Any | None, bracket_id: str | None) -> dict[str, Any]:
    return {
        "intent": "EXIT",
        "linked_entry_order_id": (
            str(getattr(bracket, "entry_order_id", "") or "") or None
        ),
        "trade_lifecycle_id": _safe_trade_lifecycle_id(bracket),
        "bracket_id": str(
            getattr(bracket, "bracket_id", "")
            or bracket_id
            or ""
        )
        or None,
    }


def _canonical_key(symbol: object) -> str:
    return normalize_symbol(str(symbol or ""))


def _canonicalize_position_store(manager: Any) -> None:
    positions = getattr(manager, "_positions", None)
    if not isinstance(positions, dict):
        return
    canonical: dict[str, Any] = {}
    collisions: list[str] = []
    for raw_key, position in list(positions.items()):
        symbol = getattr(position, "symbol", raw_key)
        key = _canonical_key(symbol or raw_key)
        if not key:
            key = str(raw_key).strip().upper()
        with suppress(Exception):
            position.symbol = key
        existing = canonical.get(key)
        if existing is not None and existing is not position:
            collisions.append(key)
            # Do not add quantities. Keep the larger absolute broker/local exposure
            # as the safer single authoritative in-memory representation.
            try:
                existing_qty = abs(int(getattr(existing, "quantity", 0) or 0))
                incoming_qty = abs(int(getattr(position, "quantity", 0) or 0))
            except Exception:
                existing_qty = incoming_qty = 0
            if incoming_qty > existing_qty:
                canonical[key] = position
            continue
        canonical[key] = position
    if canonical != positions:
        positions.clear()
        positions.update(canonical)
    if collisions:
        logger = getattr(manager, "_logger", None)
        log = getattr(logger, "critical", None)
        if callable(log):
            log(
                "POSITION_SYMBOL_CANONICALIZATION_COLLISION symbols=%s",
                sorted(set(collisions)),
                extra={
                    "event": "POSITION_SYMBOL_CANONICALIZATION_COLLISION",
                    "symbols": sorted(set(collisions)),
                },
            )


def _patch_bracket_manager() -> None:
    cls = getattr(_bracket_core, "BracketManager", None)
    if cls is None:
        return
    if getattr(cls, "_immutable_exit_identity_patch", False):
        return

    _ORIGINALS["BracketManager.submit_exit_order"] = cls.submit_exit_order
    _ORIGINALS["BracketManager._escalate_exit_locked"] = cls._escalate_exit_locked

    def submit_exit_order(
        self: Any,
        symbol: str,
        qty: int,
        reason: str,
        bracket_id: str,
        preferred_order_type: str = "LIMIT",
        correlation_tag: str | None = None,
    ) -> Any:
        """Submit an EXIT order with immutable lifecycle metadata."""
        normalized_symbol = normalize_symbol(symbol)
        bracket = self.get_bracket(bracket_id)
        side = "SELL" if (bracket and bracket.side == "BUY") else "BUY"
        order_type, price, pricing_meta = self._price_exit_order(
            bracket=bracket,
            symbol=normalized_symbol,
            side=side,
            reason=reason,
            preferred_order_type=preferred_order_type,
            qty=qty,
        )
        if pricing_meta.get("quote_missing"):
            _bracket_core.LOGGER.warning(
                "EXIT_ORDER_PRICING_DECISION bracket_id=%s reason=%s mode=aggressive_limit side=%s qty=%s bid=%s ask=%s ltp=%s price=%s fallback=%s",
                bracket_id,
                reason,
                side,
                qty,
                pricing_meta.get("bid"),
                pricing_meta.get("ask"),
                pricing_meta.get("ltp"),
                price,
                pricing_meta.get("fallback"),
                extra={
                    "event": "EXIT_ORDER_PRICING_DECISION",
                    "bracket_id": bracket_id,
                    "reason": reason,
                    "mode": "aggressive_limit",
                    "side": side,
                    "qty": qty,
                    "bid": pricing_meta.get("bid"),
                    "ask": pricing_meta.get("ask"),
                    "ltp": pricing_meta.get("ltp"),
                    "price": price,
                    "fallback": pricing_meta.get("fallback"),
                },
            )
            return _bracket_core.SubmitExitOrderResult(
                False,
                None,
                "quote_missing",
                "quote_missing",
                "protective aggressive limit quote missing",
                True,
                {},
            )
        _bracket_core.LOGGER.info(
            "EXIT_ORDER_PRICING_DECISION bracket_id=%s reason=%s mode=%s side=%s qty=%s bid=%s ask=%s ltp=%s price=%s fallback=%s",
            bracket_id,
            reason,
            str(pricing_meta.get("mode") or order_type).lower(),
            side,
            qty,
            pricing_meta.get("bid"),
            pricing_meta.get("ask"),
            pricing_meta.get("ltp"),
            price,
            pricing_meta.get("fallback"),
            extra={
                "event": "EXIT_ORDER_PRICING_DECISION",
                "bracket_id": bracket_id,
                "reason": reason,
                "mode": str(pricing_meta.get("mode") or order_type).lower(),
                "side": side,
                "qty": qty,
                "bid": pricing_meta.get("bid"),
                "ask": pricing_meta.get("ask"),
                "ltp": pricing_meta.get("ltp"),
                "price": price,
                "fallback": pricing_meta.get("fallback"),
            },
        )
        try:
            kwargs: dict[str, Any] = {
                "symbol": normalized_symbol,
                "side": side,
                "quantity": int(qty),
                "order_type": order_type,
                "tag": correlation_tag or f"exit_{reason[:3]}_{bracket_id[:8]}",
                "check_risk": False,
                "product": "MIS",
                **_exit_identity_kwargs(bracket, bracket_id),
            }
            if price is not None:
                kwargs["price"] = price
            order_id = self.order_manager.place_order(**kwargs)
            if order_id:
                return _bracket_core.SubmitExitOrderResult(
                    accepted=True,
                    order_id=str(order_id),
                    status="submitted",
                    retryable=False,
                    broker_payload={
                        "order_id": str(order_id),
                        "order_type": order_type,
                        "side": side,
                        "intent": "EXIT",
                        "linked_entry_order_id": kwargs.get("linked_entry_order_id"),
                        "trade_lifecycle_id": kwargs.get("trade_lifecycle_id"),
                        "bracket_id": kwargs.get("bracket_id"),
                    },
                )
            decision = dict(getattr(self.order_manager, "_last_order_decision", {}) or {})
            details = dict(decision.get("details") or {})
            broker_payload = dict(details.get("broker_payload") or details)
            error_type = str(
                decision.get("failure_class")
                or decision.get("block_reason")
                or "missing_order_id"
            )
            error_message = str(
                decision.get("error_message")
                or details.get("error_message")
                or details.get("broker_rejection")
                or broker_payload.get("message")
                or broker_payload.get("error")
                or decision.get("block_reason")
                or "place_order returned no order_id"
            )
            return _bracket_core.SubmitExitOrderResult(
                accepted=False,
                order_id=None,
                status="rejected",
                error_type=error_type,
                error_message=error_message,
                retryable=bool(
                    decision.get(
                        "retryable",
                        error_type not in {"broker_config_error", "fatal_order_error"},
                    )
                ),
                broker_payload={
                    "order_manager_decision": decision,
                    "broker_payload": broker_payload,
                    "kill_switch_active": bool(
                        getattr(self.order_manager, "_kill_switch_engaged_at", None)
                    ),
                },
            )
        except Exception as exc:  # noqa: BLE001 - process boundary; result is structured and safe
            message = str(exc)
            retryable = not self._is_fatal_exit_error(message)
            return _bracket_core.SubmitExitOrderResult(
                accepted=False,
                order_id=None,
                status="error",
                error_type=type(exc).__name__,
                error_message=message,
                retryable=retryable,
                broker_payload={},
            )

    def _escalate_exit_locked(self: Any, bracket: Any, reason: str) -> None:
        if bracket.exit_state == _bracket_core.BracketExitLifecycle.EXIT_FAILED_ESCALATED.value:
            return
        bracket.exit_pending = True
        bracket.exit_state = _bracket_core.BracketExitLifecycle.EXIT_FAILED_ESCALATED.value
        bracket.entry_status = _bracket_core.BracketExitLifecycle.EXIT_FAILED_ESCALATED.value
        bracket.escalated_at = time.time()
        _bracket_core.LOGGER.critical(
            "EXIT_ESCALATED bracket_id=%s symbol=%s remaining_qty=%s attempts=%s last_error=%s reason=%s",
            bracket.bracket_id,
            bracket.symbol,
            bracket.remaining_quantity,
            bracket.exit_attempt_count,
            bracket.last_exit_error,
            reason,
        )
        self._notify_event(
            "EXIT_ESCALATED",
            {
                "symbol": bracket.symbol,
                "bracket_id": bracket.bracket_id,
                "remaining_qty": bracket.remaining_quantity,
                "attempts": bracket.exit_attempt_count,
                "last_error": bracket.last_exit_error,
                "message": "⚠️ Exit unresolved. Forcing MARKET exit.",
            },
        )
        if not self._exit_force_market_on_escalation:
            return
        if getattr(bracket, "_market_escalation_fired", False):
            return
        bracket._market_escalation_fired = True
        stuck_order_id = bracket.exit_order_id or bracket.pending_exit_order_id
        symbol = normalize_symbol(bracket.symbol)
        qty = int(bracket.remaining_quantity or 0)
        side = "SELL" if bracket.side == "BUY" else "BUY"

        def _force_market_flatten() -> None:
            if stuck_order_id:
                try:
                    self.order_manager.cancel_order(str(stuck_order_id))
                    _bracket_core.LOGGER.warning(
                        "EXIT_ESCALATION_CANCELLED_STUCK_ORDER bracket_id=%s order_id=%s",
                        bracket.bracket_id,
                        stuck_order_id,
                    )
                except Exception as exc:  # noqa: BLE001 - cancel best-effort; still try market
                    _bracket_core.LOGGER.warning(
                        "EXIT_ESCALATION_CANCEL_FAILED bracket_id=%s order_id=%s error=%s",
                        bracket.bracket_id,
                        stuck_order_id,
                        exc,
                    )
            with self._lock:
                bracket.exit_order_id = None
                bracket.pending_exit_order_id = None
            if not symbol or qty <= 0:
                return
            try:
                order_id = self.order_manager.place_order(
                    symbol=symbol,
                    side=side,
                    quantity=qty,
                    order_type="MARKET",
                    tag=f"EXIT_MKT_{bracket.bracket_id[:8]}",
                    check_risk=False,
                    product="MIS",
                    **_exit_identity_kwargs(bracket, bracket.bracket_id),
                )
            except Exception as exc:  # noqa: BLE001
                _bracket_core.LOGGER.critical(
                    "EXIT_ESCALATION_MARKET_EXIT_FAILED bracket_id=%s symbol=%s error=%s",
                    bracket.bracket_id,
                    symbol,
                    exc,
                )
                return
            if order_id:
                with self._lock:
                    bracket.exit_order_id = str(order_id)
                    bracket.pending_exit_order_id = str(order_id)
                _bracket_core.LOGGER.critical(
                    "EXIT_ESCALATION_MARKET_EXIT_SENT bracket_id=%s symbol=%s order_id=%s qty=%s",
                    bracket.bracket_id,
                    symbol,
                    order_id,
                    qty,
                )
            else:
                _bracket_core.LOGGER.critical(
                    "EXIT_ESCALATION_MARKET_EXIT_NO_ORDER_ID bracket_id=%s symbol=%s",
                    bracket.bracket_id,
                    symbol,
                )

        try:
            _force_market_flatten()
        except Exception as exc:  # noqa: BLE001 - never let escalation raise
            _bracket_core.LOGGER.error(
                "EXIT_ESCALATION_DISPATCH_FAILED bracket_id=%s error=%s",
                bracket.bracket_id,
                exc,
            )

    cls.submit_exit_order = submit_exit_order
    cls._escalate_exit_locked = _escalate_exit_locked
    cls._immutable_exit_identity_patch = True


def _patch_position_manager() -> None:
    cls = getattr(_position_manager, "PositionManager", None)
    if cls is None:
        return
    if getattr(cls, "_canonical_position_key_patch", False):
        return

    _ORIGINALS["PositionManager.__init__"] = cls.__init__
    _ORIGINALS["PositionManager.save_state"] = cls.save_state
    _ORIGINALS["PositionManager.open_position"] = cls.open_position
    _ORIGINALS["PositionManager.close_position"] = cls.close_position
    _ORIGINALS["PositionManager.update_position_price"] = cls.update_position_price
    _ORIGINALS["PositionManager.get_position"] = cls.get_position
    _ORIGINALS["PositionManager.has_position"] = cls.has_position
    _ORIGINALS["PositionManager.is_flat"] = cls.is_flat
    _ORIGINALS["PositionManager.clear_active_contract_by_symbol"] = cls.clear_active_contract_by_symbol
    if hasattr(cls, "current_entry_protection_blocker"):
        _ORIGINALS[
            "PositionManager.current_entry_protection_blocker"
        ] = cls.current_entry_protection_blocker

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        _ORIGINALS["PositionManager.__init__"](self, *args, **kwargs)
        with getattr(self, "_lock", suppress()):
            _canonicalize_position_store(self)

    def save_state(self: Any, *args: Any, **kwargs: Any) -> Any:
        lock = getattr(self, "_lock", None)
        if lock is None:
            _canonicalize_position_store(self)
        else:
            with lock:
                _canonicalize_position_store(self)
        return _ORIGINALS["PositionManager.save_state"](self, *args, **kwargs)

    def open_position(self: Any, symbol: str, *args: Any, **kwargs: Any) -> Any:
        return _ORIGINALS["PositionManager.open_position"](
            self,
            _canonical_key(symbol),
            *args,
            **kwargs,
        )

    def close_position(self: Any, symbol: str, *args: Any, **kwargs: Any) -> Any:
        return _ORIGINALS["PositionManager.close_position"](
            self,
            _canonical_key(symbol),
            *args,
            **kwargs,
        )

    def update_position_price(self: Any, symbol: str, *args: Any, **kwargs: Any) -> Any:
        return _ORIGINALS["PositionManager.update_position_price"](
            self,
            _canonical_key(symbol),
            *args,
            **kwargs,
        )

    def get_position(self: Any, symbol: str) -> Any:
        return _ORIGINALS["PositionManager.get_position"](self, _canonical_key(symbol))

    def has_position(self: Any, symbol: str) -> bool:
        return bool(_ORIGINALS["PositionManager.has_position"](self, _canonical_key(symbol)))

    def is_flat(self: Any, symbol: str) -> bool:
        return bool(_ORIGINALS["PositionManager.is_flat"](self, _canonical_key(symbol)))

    def clear_active_contract_by_symbol(self: Any, symbol: str) -> Any:
        return _ORIGINALS["PositionManager.clear_active_contract_by_symbol"](
            self,
            _canonical_key(symbol),
        )

    def current_entry_protection_blocker(
        self: Any, symbol: str | None = None
    ) -> str | None:
        original = _ORIGINALS.get("PositionManager.current_entry_protection_blocker")
        if original is None:
            return None
        return original(self, _canonical_key(symbol) if symbol else None)

    cls.__init__ = __init__
    cls.save_state = save_state
    cls.open_position = open_position
    cls.close_position = close_position
    cls.update_position_price = update_position_price
    cls.get_position = get_position
    cls.has_position = has_position
    cls.has_open_position = has_position
    cls.is_flat = is_flat
    cls.clear_active_contract_by_symbol = clear_active_contract_by_symbol
    if "PositionManager.current_entry_protection_blocker" in _ORIGINALS:
        cls.current_entry_protection_blocker = current_entry_protection_blocker
    cls._position_key = staticmethod(_canonical_key)
    cls._canonical_position_key_patch = True


def apply_patches() -> None:
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return
    _patch_bracket_manager()
    _patch_position_manager()
    _PATCH_APPLIED = True


apply_patches()

__all__ = ["apply_patches"]
