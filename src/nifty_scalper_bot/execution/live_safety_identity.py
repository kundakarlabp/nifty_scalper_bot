"""Compatibility helper for legacy protective-exit identity tests.

Position identity is owned natively by PositionManager. The bracket helper is
retained only for explicit compatibility tests while BracketManager exit
identity is migrated separately; importing this module does not mutate runtime
PositionManager behavior.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

from nifty_scalper_bot.execution import bracket_core as _bracket_core
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


def _exit_product(bracket: Any | None) -> str:
    """Resolve protective-exit product from the entry bracket."""
    product = str(getattr(bracket, "product", "") or "").strip().upper()
    return product if product in {"MIS", "NRML"} else "MIS"


def _canonical_key(symbol: object) -> str:
    return normalize_symbol(str(symbol or ""))


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
            product = _exit_product(bracket)
            kwargs: dict[str, Any] = {
                "symbol": normalized_symbol,
                "side": side,
                "quantity": int(qty),
                "order_type": order_type,
                "tag": correlation_tag or f"exit_{reason[:3]}_{bracket_id[:8]}",
                "check_risk": False,
                "product": product,
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
                        "product": product,
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
                product = _exit_product(bracket)
                order_id = self.order_manager.place_order(
                    symbol=symbol,
                    side=side,
                    quantity=qty,
                    order_type="MARKET",
                    tag=f"EXIT_MKT_{bracket.bracket_id[:8]}",
                    check_risk=False,
                    product=product,
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


def apply_patches() -> None:
    """Compatibility no-op; PositionManager identity is native."""

    global _PATCH_APPLIED
    _PATCH_APPLIED = True


apply_patches()

__all__ = ["apply_patches"]
