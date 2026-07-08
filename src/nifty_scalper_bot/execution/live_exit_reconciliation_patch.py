"""Live exit reconciliation hardening.

Runtime purpose:
- Prevent LIVE brackets from closing on a broker position-flat snapshot while the
  linked exit order is still OPEN/PENDING.
- Keep ledger-blocked or exit-pending symbols managed so orphan adoption cannot
  create a second synthetic bracket for the same live exposure.

This module is intentionally patch-style because the project already loads
runtime execution hardening from package import hooks.
"""

from __future__ import annotations

from contextlib import suppress
import time
from typing import Any, Mapping

from nifty_scalper_bot.execution import bracket_core as _legacy
from nifty_scalper_bot.utils.symbols import normalize_symbol

_PATCH_APPLIED = False
_ORIGINALS: dict[str, Any] = {}

_OPEN_OR_PENDING_STATUSES = {
    "",
    "OPEN",
    "OPEN PENDING",
    "PENDING",
    "TRIGGER PENDING",
    "VALIDATION PENDING",
    "PUT ORDER REQ RECEIVED",
    "MODIFY PENDING",
    "AMO REQ RECEIVED",
}

_TERMINAL_FILLED = set(getattr(_legacy, "_FILLED_STATUSES", {"COMPLETE", "FILLED"}))
_TERMINAL_CANCELLED = set(
    getattr(_legacy, "_CANCELLED_STATUSES", {"CANCELLED", "REJECTED", "EXPIRED"})
)


def _strict_live(self: Any) -> bool:
    checker = getattr(self, "_strict_ledger_release_required", None)
    if callable(checker):
        with suppress(Exception):
            return bool(checker())
    return False


def _original_reconcile(self: Any, bracket: Any, *, requested_by: str) -> bool:
    original = _ORIGINALS.get("BoundBracketManager._reconcile_exit_state")
    if callable(original):
        return bool(original(self, bracket, requested_by=requested_by))
    return False


def _block_release(self: Any, bracket: Any, *, reason: str, payload: Mapping[str, Any]) -> None:
    blocker = getattr(self, "_block_ledger_release", None)
    if callable(blocker):
        blocker(bracket, reason=reason, payload=dict(payload))
        return
    with self._lock:
        bracket.last_exit_error = str(reason)
        bracket.updated_at = time.time()


def _defer_close(
    self: Any,
    bracket: Any,
    *,
    reason: str,
    order_id: str | None,
    order_status: str,
    fill_price: float | None,
    requested_by: str,
    mark_position_flat: bool,
) -> None:
    """Latch unresolved exit state without releasing the bracket lifecycle."""

    payload = {
        "order_id": order_id or "",
        "order_status": order_status,
        "fill_price": fill_price,
        "requested_by": requested_by,
        "symbol": str(getattr(bracket, "symbol", "")),
        "bracket_id": str(getattr(bracket, "bracket_id", "")),
        "remaining_quantity": int(getattr(bracket, "remaining_quantity", 0) or 0),
    }
    _block_release(self, bracket, reason=reason, payload=payload)
    with self._lock:
        bracket.exit_pending = True
        bracket.exit_in_progress = False
        bracket.exit_state = _legacy.BracketExitLifecycle.EXIT_FAILED_ESCALATED.value
        bracket.entry_status = bracket.exit_state
        bracket.last_exit_error = reason
        if mark_position_flat:
            bracket.position_flat_confirmed = True
            bracket.active = False
        bracket.updated_at = time.time()
    _legacy.LOGGER.critical(
        "EXIT_CLOSE_DEFERRED_PENDING_ACCOUNTING bracket_id=%s symbol=%s reason=%s order_id=%s order_status=%s fill_price=%s requested_by=%s",
        getattr(bracket, "bracket_id", None),
        getattr(bracket, "symbol", None),
        reason,
        order_id,
        order_status,
        fill_price,
        requested_by,
        extra={
            "event": "EXIT_CLOSE_DEFERRED_PENDING_ACCOUNTING",
            "bracket_id": getattr(bracket, "bracket_id", None),
            "symbol": getattr(bracket, "symbol", None),
            "reason": reason,
            "order_id": order_id,
            "order_status": order_status,
            "fill_price": fill_price,
        },
    )


def _patched_reconcile_exit_state(self: Any, bracket: Any, *, requested_by: str) -> bool:
    """Reconcile exit without closing LIVE brackets on provisional flat proof.

    Non-live/test execution delegates to the original runtime method. The
    production-only change is that strict LIVE mode requires terminal broker fill
    identity before durable bracket close.
    """

    if not _strict_live(self):
        return _original_reconcile(self, bracket, requested_by=requested_by)

    now = time.time()
    with self._lock:
        if bracket.exit_state == _legacy.BracketExitLifecycle.CLOSED.value:
            return True
        if requested_by not in {"pre_submit", "post_submit"}:
            last = float(getattr(bracket, "_last_exit_reconcile_at", 0.0) or 0.0)
            if now - last < float(getattr(self, "_exit_reconcile_interval_seconds", 0.0) or 0.0):
                return False
        setattr(bracket, "_last_exit_reconcile_at", now)
        order_id = bracket.exit_order_id or bracket.pending_exit_order_id

    _legacy.LOGGER.info(
        "EXIT_RECONCILE_REQUESTED bracket_id=%s symbol=%s requested_by=%s exit_order_id=%s",
        bracket.bracket_id,
        bracket.symbol,
        requested_by,
        order_id,
    )

    filled = False
    fill_price: float | None = None
    order_status = ""
    try:
        if order_id:
            raw_status = self._get_broker_order_status(str(order_id))
            status_payload = raw_status if isinstance(raw_status, Mapping) else {}
            order_status = str((status_payload or {}).get("status", "")).strip().upper()
            fill_price = self._extract_status_price(status_payload)
            if order_status in _TERMINAL_FILLED:
                filled = True
            elif not order_status:
                waiter = getattr(self.order_manager, "wait_for_fill", None)
                if callable(waiter):
                    try:
                        filled = bool(waiter(str(order_id), timeout_sec=0.0))
                    except TypeError:
                        filled = bool(waiter(str(order_id)))
            elif order_status in _TERMINAL_CANCELLED:
                with self._lock:
                    bracket.last_exit_error = f"exit_order_{order_status.lower()}"
                    bracket.exit_order_id = None
                    bracket.pending_exit_order_id = None
                    if (
                        bool(getattr(self, "_exit_retry_enabled", False))
                        and int(getattr(bracket, "exit_attempt_count", 0) or 0)
                        < int(getattr(self, "_exit_max_retry_attempts", 0) or 0)
                    ):
                        bracket.exit_state = _legacy.BracketExitLifecycle.EXIT_REJECTED_RETRYABLE.value
                        bracket.next_exit_attempt_at = time.time() + self._retry_delay_for_attempt(
                            max(int(getattr(bracket, "exit_attempt_count", 0) or 0), 1)
                        )
                    else:
                        self._escalate_exit_locked(bracket, "broker_order_rejected_or_cancelled")
        flat = bool(self._position_flat_for_symbol(bracket.symbol))
    except Exception as exc:  # noqa: BLE001 - broker reconciliation boundary
        with self._lock:
            bracket.last_exit_error = f"reconcile_failed:{type(exc).__name__}:{exc}"
            age_basis = float(bracket.last_exit_attempt_at or bracket.exit_triggered_at or now)
            age = now - age_basis
            if age >= float(getattr(self, "_exit_unresolved_escalation_seconds", 0.0) or 0.0):
                self._escalate_exit_locked(bracket, "reconcile_failed_timeout")
        _legacy.LOGGER.error(
            "EXIT_RECONCILE_RESULT bracket_id=%s symbol=%s flat=False error_type=%s error_message=%s",
            bracket.bracket_id,
            bracket.symbol,
            type(exc).__name__,
            exc,
        )
        return False

    _legacy.LOGGER.info(
        "EXIT_RECONCILE_RESULT bracket_id=%s symbol=%s flat=%s order_status=%s filled=%s",
        bracket.bracket_id,
        bracket.symbol,
        flat,
        order_status,
        filled,
    )

    if filled:
        if fill_price is None:
            _defer_close(
                self,
                bracket,
                reason="exit_fill_price_pending",
                order_id=str(order_id or "") or None,
                order_status=order_status,
                fill_price=fill_price,
                requested_by=requested_by,
                mark_position_flat=flat,
            )
            return False
        self._close_bracket(bracket, close_source="broker_fill", exit_price=fill_price)
        _legacy.LOGGER.info("EXIT_FILLED_CONFIRMED bracket_id=%s order_id=%s", bracket.bracket_id, order_id)
        return True

    if flat and order_id:
        if order_status in _OPEN_OR_PENDING_STATUSES or order_status not in _TERMINAL_FILLED:
            with self._lock:
                bracket.exit_pending = True
                bracket.exit_state = _legacy.BracketExitLifecycle.EXIT_ORDER_SUBMITTED.value
                bracket.entry_status = bracket.exit_state
                bracket.position_flat_confirmed = False
                bracket.updated_at = time.time()
            _legacy.LOGGER.warning(
                "EXIT_FLAT_BUT_ORDER_NOT_TERMINAL bracket_id=%s symbol=%s order_id=%s order_status=%s fill_price=%s close_deferred=True",
                bracket.bracket_id,
                bracket.symbol,
                order_id,
                order_status or "UNKNOWN",
                fill_price,
                extra={
                    "event": "EXIT_FLAT_BUT_ORDER_NOT_TERMINAL",
                    "bracket_id": bracket.bracket_id,
                    "symbol": bracket.symbol,
                    "order_id": str(order_id),
                    "order_status": order_status or "UNKNOWN",
                    "fill_price": fill_price,
                },
            )
            return False

    if flat and not order_id:
        _defer_close(
            self,
            bracket,
            reason="exit_flat_without_order_identity",
            order_id=None,
            order_status=order_status,
            fill_price=fill_price,
            requested_by=requested_by,
            mark_position_flat=True,
        )
        return False

    with self._lock:
        age_basis = float(bracket.last_exit_attempt_at or bracket.exit_triggered_at or now)
        age = now - age_basis
        if age >= float(getattr(self, "_exit_unresolved_escalation_seconds", 0.0) or 0.0):
            self._escalate_exit_locked(bracket, "unresolved_timeout")
        else:
            self._log_exit_pending_summary_locked(bracket, now)
    return False


def _ledger_or_exit_managed(self: Any, symbol: str) -> bool:
    original = _ORIGINALS.get("BoundBracketManager.is_symbol_managed")
    if not _strict_live(self):
        return bool(original(self, symbol)) if callable(original) else False
    if callable(original):
        with suppress(Exception):
            if bool(original(self, symbol)):
                return True

    symbol_key = normalize_symbol(symbol)
    ledger_blocked = set(getattr(self, "_ledger_blocked", {}) or {})
    with self._lock:
        for entry_id in list(getattr(self, "_symbol_map", {}).get(symbol_key, [])):
            bracket = getattr(self, "_brackets", {}).get(entry_id)
            if bracket is None:
                continue
            bracket_id = str(getattr(bracket, "bracket_id", "") or "")
            if bracket_id in ledger_blocked:
                return True
            if bool(getattr(bracket, "exit_pending", False)) or bool(getattr(bracket, "exit_in_progress", False)):
                return True
            if getattr(bracket, "exit_order_id", None) or getattr(bracket, "pending_exit_order_id", None):
                return True
            state = str(getattr(bracket, "exit_state", "") or "").upper()
            if state in {
                _legacy.BracketExitLifecycle.EXIT_TRIGGERED.value,
                _legacy.BracketExitLifecycle.EXIT_ORDER_PENDING.value,
                _legacy.BracketExitLifecycle.EXIT_ORDER_SUBMITTED.value,
                _legacy.BracketExitLifecycle.EXIT_FAILED_ESCALATED.value,
                _legacy.BracketExitLifecycle.EXIT_REJECTED_RETRYABLE.value,
            }:
                return True
    return False


def apply_patches() -> None:
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return
    from nifty_scalper_bot.execution.ownership import BoundBracketManager

    if not getattr(BoundBracketManager, "_live_exit_reconciliation_patch", False):
        _ORIGINALS["BoundBracketManager._reconcile_exit_state"] = BoundBracketManager._reconcile_exit_state
        _ORIGINALS["BoundBracketManager.is_symbol_managed"] = BoundBracketManager.is_symbol_managed
        BoundBracketManager._reconcile_exit_state = _patched_reconcile_exit_state
        BoundBracketManager.is_symbol_managed = _ledger_or_exit_managed
        BoundBracketManager._live_exit_reconciliation_patch = True
    _PATCH_APPLIED = True


__all__ = ["apply_patches", "_patched_reconcile_exit_state"]
