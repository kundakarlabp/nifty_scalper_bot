"""Deterministic live-money hardening for the virtual bracket manager.

The legacy manager remains the implementation owner. This subclass tightens its
state transitions, rescues stale protective orders without duplicate exposure,
and installs a final unresolved-exit guard on the attached OrderManager.
"""

from __future__ import annotations

from contextlib import suppress
import functools
import math
import os
import time
from typing import Any, Mapping

from nifty_scalper_bot.execution import bracket_manager as _legacy


_LegacyBracketManager = _legacy.BracketManager


class HardenedBracketManager(_LegacyBracketManager):
    """Virtual bracket manager with latched exits and controlled rescue orders."""

    _OPEN_ORDER_STATUSES = {
        "OPEN",
        "OPEN PENDING",
        "PENDING",
        "TRIGGER PENDING",
        "VALIDATION PENDING",
        "PUT ORDER REQ RECEIVED",
        "MODIFY PENDING",
        "AMO REQ RECEIVED",
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Initialize hardening state before the parent starts its watchdog thread.
        self._exit_order_open_since: dict[str, float] = {}
        self._exit_rescue_attempts: dict[str, int] = {}
        self._exit_escalation_notifications: set[tuple[str, str]] = set()
        self._exit_open_order_timeout_seconds = max(
            0.5,
            _legacy.parse_float_env(
                os.getenv("EXIT_OPEN_ORDER_RESCUE_AFTER_SECONDS"), 3.0
            ),
        )
        self._exit_cancel_confirm_timeout_seconds = max(
            0.25,
            _legacy.parse_float_env(
                os.getenv("EXIT_CANCEL_CONFIRM_TIMEOUT_SECONDS"), 1.5
            ),
        )
        self._exit_cancel_poll_interval_seconds = max(
            0.05,
            _legacy.parse_float_env(
                os.getenv("EXIT_CANCEL_POLL_INTERVAL_SECONDS"), 0.10
            ),
        )
        self._exit_rescue_max_attempts = max(
            1,
            _legacy.parse_int_env(os.getenv("EXIT_RESCUE_MAX_ATTEMPTS"), 2),
        )
        super().__init__(*args, **kwargs)
        self._install_unresolved_exit_entry_guard()

    # ------------------------------------------------------------------
    # Order-entry freeze: exits continue, new entries fail closed.
    # ------------------------------------------------------------------
    def _install_unresolved_exit_entry_guard(self) -> None:
        order_manager = getattr(self, "order_manager", None)
        if order_manager is None or getattr(
            order_manager, "_unresolved_exit_guard_installed", False
        ):
            return

        def blocked_details() -> dict[str, Any] | None:
            if not self.has_unresolved_exit():
                return None
            bracket_id = self.get_first_unresolved_exit_bracket_id()
            details = {
                "block_reason": "unresolved_exit_position",
                "bracket_id": bracket_id,
                "broker_attempted": False,
                "retryable": False,
            }
            setattr(order_manager, "_last_order_decision", dict(details))
            setter = getattr(order_manager, "set_last_skip_reason", None)
            if callable(setter):
                with suppress(Exception):
                    setter("unresolved_exit_position")
            _legacy.LOGGER.critical(
                "ENTRY_BLOCKED_UNRESOLVED_EXIT bracket_id=%s",
                bracket_id,
                extra={
                    "event": "ENTRY_BLOCKED_UNRESOLVED_EXIT",
                    "bracket_id": bracket_id,
                    "broker_attempted": False,
                },
            )
            return details

        def is_protective_call(method_name: str, kwargs: Mapping[str, Any]) -> bool:
            if method_name != "place_order":
                return False
            if bool(kwargs.get("reduce_only")) or bool(kwargs.get("is_exit")):
                return True
            tag = str(kwargs.get("tag") or "").strip().upper()
            protective_prefixes = (
                "EXIT_",
                "SL_",
                "TP_",
                "EOD_",
                "PANIC",
                "FLATTEN",
                "SQUAREOFF",
            )
            return tag.startswith(protective_prefixes)

        def rejection_for(method_name: str, details: dict[str, Any]) -> Any:
            if method_name == "submit_trade_plan_result":
                from nifty_scalper_bot.execution.order_manager import (
                    TradePlanSubmitResult,
                )

                return TradePlanSubmitResult(
                    accepted=False,
                    order_id=None,
                    reason="unresolved_exit_position",
                    details=details,
                    broker_attempted=False,
                )
            if method_name == "place_managed_order_result":
                from nifty_scalper_bot.execution.order_manager import ManagedOrderResult

                return ManagedOrderResult(
                    accepted=False,
                    order_id=None,
                    reason="unresolved_exit_position",
                    details=details,
                    broker_attempted=False,
                )
            return None

        for method_name in (
            "submit_trade_plan_result",
            "submit_trade_plan",
            "place_managed_order_result",
            "place_managed_order",
            "place_order",
        ):
            original = getattr(order_manager, method_name, None)
            if not callable(original):
                continue

            @functools.wraps(original)
            def guarded(
                *args: Any,
                __original: Any = original,
                __method_name: str = method_name,
                **method_kwargs: Any,
            ) -> Any:
                if is_protective_call(__method_name, method_kwargs):
                    return __original(*args, **method_kwargs)
                details = blocked_details()
                if details is not None:
                    return rejection_for(__method_name, details)
                return __original(*args, **method_kwargs)

            setattr(order_manager, method_name, guarded)

        setattr(order_manager, "_unresolved_exit_guard_installed", True)
        _legacy.LOGGER.info(
            "UNRESOLVED_EXIT_ENTRY_GUARD_INSTALLED",
            extra={"event": "UNRESOLVED_EXIT_ENTRY_GUARD_INSTALLED"},
        )

    # ------------------------------------------------------------------
    # Virtual trailing-stop integrity.
    # ------------------------------------------------------------------
    def register_virtual_bracket(self, *args: Any, **kwargs: Any) -> None:
        _LegacyBracketManager.register_virtual_bracket(self, *args, **kwargs)
        order_id = str(kwargs.get("order_id") or (args[0] if args else ""))
        bracket = self.get_bracket(order_id) if order_id else None
        if bracket is None:
            return
        with self._lock:
            if not bracket.exit_pending and bracket.exit_state in {
                _legacy.BracketExitLifecycle.OPEN_PENDING_FILL.value,
                _legacy.BracketExitLifecycle.OPEN_ACTIVE.value,
            }:
                bracket.exit_order_id = None
                bracket.pending_exit_order_id = None
                bracket.escalated_at = None
                bracket.last_exit_error = None
                bracket.next_exit_attempt_at = None
            controller = self._trailing_controllers.get(bracket.entry_order_id)
            if controller is not None:
                controller.current_sl = float(bracket.sl_trigger_price)
                controller.entry_price = float(bracket.entry_price)
                controller.highest_price = float(bracket.entry_price)
                controller.lowest_price = float(bracket.entry_price)

    def confirm_entry_fill(self, order_id: str, fill_price: float) -> None:
        _LegacyBracketManager.confirm_entry_fill(self, order_id, fill_price)
        bracket = self.get_bracket(order_id)
        if bracket is None:
            return
        with self._lock:
            controller = self._trailing_controllers.get(order_id)
            if controller is not None:
                controller.entry_price = float(bracket.entry_price)
                controller.current_sl = float(bracket.sl_trigger_price)
                controller.highest_price = float(bracket.entry_price)
                controller.lowest_price = float(bracket.entry_price)

    def _virtual_modify_sl(self, order_id: str, price: float) -> bool:
        """Apply only finite, market-side, monotonic virtual stop updates."""
        try:
            proposed = float(price)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(proposed) or proposed <= 0:
            return False
        proposed = _legacy._round_to_tick(proposed)

        target = None
        with self._lock:
            for bracket in self._brackets.values():
                if bracket.virtual_sl_id == order_id:
                    target = bracket
                    break
            if target is None or target.exit_pending or target.exit_executed:
                return False
            current = float(target.sl_trigger_price)
            ltp = float(target.last_ltp or 0.0)
            if target.side == "BUY":
                if proposed <= current or (ltp > 0 and proposed >= ltp):
                    return False
            else:
                if proposed >= current or (ltp > 0 and proposed <= ltp):
                    return False
            target.sl_trigger_price = proposed
            target.updated_at = time.time()

        with suppress(Exception):
            self.save_state()
        _legacy.LOGGER.info(
            "VIRTUAL_TRAILING_SL_RATCHET symbol=%s old_sl=%.2f new_sl=%.2f ltp=%.2f",
            target.symbol,
            current,
            proposed,
            ltp,
            extra={
                "event": "VIRTUAL_TRAILING_SL_RATCHET",
                "symbol": target.symbol,
                "old_sl": current,
                "new_sl": proposed,
                "ltp": ltp,
            },
        )
        return True

    # ------------------------------------------------------------------
    # Protective exit state machine and stale-order rescue.
    # ------------------------------------------------------------------
    def _process_exit_state(
        self,
        bracket: Any,
        action: Mapping[str, Any],
        *,
        now: float,
    ) -> None:
        symbol = _legacy.normalize_symbol(bracket.symbol)
        qty = max(
            0,
            min(
                int(action.get("qty") or bracket.remaining_quantity),
                int(bracket.remaining_quantity),
            ),
        )
        if not symbol or qty <= 0:
            return

        with self._lock:
            if (
                bracket.exit_state
                == _legacy.BracketExitLifecycle.EXIT_FAILED_ESCALATED.value
                and not self._exit_continue_retry_after_escalation
            ):
                self._log_exit_pending_summary_locked(bracket, now)
                return
            order_id = bracket.exit_order_id or bracket.pending_exit_order_id

        if order_id:
            status_payload = self._get_broker_order_status(str(order_id))
            status = str((status_payload or {}).get("status") or "").strip().upper()
            if status in _legacy._FILLED_STATUSES:
                self._reconcile_exit_state(bracket, requested_by="hardening_filled")
                return
            if status in _legacy._CANCELLED_STATUSES:
                with self._lock:
                    bracket.exit_order_id = None
                    bracket.pending_exit_order_id = None
                self._submit_rescue_exit(bracket, qty=qty, prior_order_id=str(order_id))
                return

            base = float(
                bracket.last_exit_attempt_at
                or bracket.exit_triggered_at
                or self._exit_order_open_since.get(str(order_id), now)
            )
            self._exit_order_open_since.setdefault(str(order_id), base)
            open_age = max(0.0, now - self._exit_order_open_since[str(order_id)])
            if status in self._OPEN_ORDER_STATUSES or not status:
                if open_age >= self._exit_open_order_timeout_seconds:
                    self._rescue_stale_exit_order(
                        bracket,
                        order_id=str(order_id),
                        qty=qty,
                        status=status or "UNKNOWN",
                    )
                    return

            if self._reconcile_exit_state(bracket, requested_by="existing_exit_order"):
                return
            with self._lock:
                # Reconciliation may have escalated. Never overwrite the terminal
                # unresolved state merely because an old broker order id exists.
                if (
                    bracket.exit_state
                    == _legacy.BracketExitLifecycle.EXIT_FAILED_ESCALATED.value
                ):
                    self._log_exit_pending_summary_locked(bracket, now)
                    return
                bracket.exit_state = (
                    _legacy.BracketExitLifecycle.EXIT_ORDER_SUBMITTED.value
                )
                bracket.entry_status = bracket.exit_state
                self._log_exit_pending_summary_locked(bracket, now)
            return

        if self._reconcile_exit_state(bracket, requested_by="pre_submit_hardened"):
            return
        with self._lock:
            if (
                bracket.exit_state
                == _legacy.BracketExitLifecycle.EXIT_FAILED_ESCALATED.value
                and not self._exit_continue_retry_after_escalation
            ):
                self._log_exit_pending_summary_locked(bracket, now)
                return
        _LegacyBracketManager._process_exit_state(self, bracket, action, now=now)

    def _rescue_stale_exit_order(
        self,
        bracket: Any,
        *,
        order_id: str,
        qty: int,
        status: str,
    ) -> None:
        with self._lock:
            if bracket.exit_in_progress:
                return
            attempts = self._exit_rescue_attempts.get(bracket.bracket_id, 0)
            if attempts >= self._exit_rescue_max_attempts:
                self._escalate_exit_locked(bracket, "rescue_attempts_exhausted")
                return
            bracket.exit_in_progress = True
            self._exit_rescue_attempts[bracket.bracket_id] = attempts + 1

        _legacy.LOGGER.critical(
            "EXIT_STALE_ORDER_RESCUE bracket_id=%s order_id=%s status=%s qty=%s rescue_attempt=%s",
            bracket.bracket_id,
            order_id,
            status,
            qty,
            attempts + 1,
            extra={
                "event": "EXIT_STALE_ORDER_RESCUE",
                "bracket_id": bracket.bracket_id,
                "order_id": order_id,
                "status": status,
                "qty": qty,
                "rescue_attempt": attempts + 1,
            },
        )

        cancel_requested = self._cancel_exit_order(order_id)
        terminal = self._wait_for_cancel_or_fill(order_id) if cancel_requested else "unconfirmed"
        if terminal == "filled" or self._safe_position_flat(bracket.symbol):
            with self._lock:
                bracket.exit_in_progress = False
            self._close_bracket(bracket, close_source="rescue_reconciled_flat")
            return
        if terminal != "cancelled":
            with self._lock:
                bracket.exit_in_progress = False
                bracket.last_exit_error = "stale_exit_cancel_unconfirmed"
                self._escalate_exit_locked(bracket, "stale_exit_cancel_unconfirmed")
            return

        with self._lock:
            bracket.exit_in_progress = False
            bracket.exit_order_id = None
            bracket.pending_exit_order_id = None
            bracket.last_exit_error = None
        self._submit_rescue_exit(bracket, qty=qty, prior_order_id=order_id)

    def _submit_rescue_exit(
        self,
        bracket: Any,
        *,
        qty: int,
        prior_order_id: str,
    ) -> None:
        if self._safe_position_flat(bracket.symbol):
            self._close_bracket(bracket, close_source="rescue_pre_submit_flat")
            return

        with self._lock:
            bracket.exit_in_progress = True
            bracket.exit_attempt_count += 1
            bracket.last_exit_attempt_at = time.time()
            attempt = bracket.exit_attempt_count
            bracket.exit_state = _legacy.BracketExitLifecycle.EXIT_ORDER_PENDING.value
            bracket.entry_status = bracket.exit_state

        result = self.submit_exit_order(
            symbol=_legacy.normalize_symbol(bracket.symbol),
            qty=qty,
            reason="EXIT_ESCALATED_RESCUE",
            bracket_id=bracket.bracket_id,
            preferred_order_type="MARKET",
        )

        with self._lock:
            bracket.exit_in_progress = False
            if result.accepted and result.order_id:
                new_order_id = str(result.order_id)
                bracket.exit_order_id = new_order_id
                bracket.pending_exit_order_id = new_order_id
                bracket.exit_state = (
                    _legacy.BracketExitLifecycle.EXIT_ORDER_SUBMITTED.value
                )
                bracket.entry_status = bracket.exit_state
                bracket.last_exit_error = None
                bracket.next_exit_attempt_at = None
                bracket.escalated_at = None
                self._exit_order_open_since[new_order_id] = time.time()
                _legacy.LOGGER.critical(
                    "EXIT_RESCUE_ORDER_SUBMITTED bracket_id=%s prior_order_id=%s new_order_id=%s attempt=%s qty=%s",
                    bracket.bracket_id,
                    prior_order_id,
                    new_order_id,
                    attempt,
                    qty,
                    extra={
                        "event": "EXIT_RESCUE_ORDER_SUBMITTED",
                        "bracket_id": bracket.bracket_id,
                        "prior_order_id": prior_order_id,
                        "new_order_id": new_order_id,
                        "attempt": attempt,
                        "qty": qty,
                    },
                )
                return

            bracket.exit_order_id = None
            bracket.pending_exit_order_id = None
            bracket.last_exit_error = result.error_message or result.status
            if (
                result.retryable
                and self._exit_retry_enabled
                and self._exit_rescue_attempts.get(bracket.bracket_id, 0)
                < self._exit_rescue_max_attempts
            ):
                bracket.exit_state = (
                    _legacy.BracketExitLifecycle.EXIT_REJECTED_RETRYABLE.value
                )
                bracket.entry_status = bracket.exit_state
                bracket.next_exit_attempt_at = time.time() + self._retry_delay_for_attempt(
                    max(attempt, 1)
                )
            else:
                self._escalate_exit_locked(bracket, "rescue_submit_failed")

    def _cancel_exit_order(self, order_id: str) -> bool:
        targets = (
            getattr(self, "order_manager", None),
            getattr(getattr(self, "order_manager", None), "_broker", None),
        )
        last_error: Exception | None = None
        for target in targets:
            cancel = getattr(target, "cancel_order", None)
            if not callable(cancel):
                continue
            calls = (
                lambda: cancel(order_id),
                lambda: cancel(order_id=order_id),
                lambda: cancel("regular", order_id),
                lambda: cancel(variety="regular", order_id=order_id),
            )
            for invoke in calls:
                try:
                    invoke()
                    return True
                except TypeError as exc:
                    last_error = exc
                    continue
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    break
        _legacy.LOGGER.error(
            "EXIT_CANCEL_REQUEST_FAILED order_id=%s error=%s",
            order_id,
            last_error,
            extra={
                "event": "EXIT_CANCEL_REQUEST_FAILED",
                "order_id": order_id,
                "error_type": type(last_error).__name__ if last_error else "missing_cancel_api",
            },
        )
        return False

    def _wait_for_cancel_or_fill(self, order_id: str) -> str:
        deadline = time.monotonic() + self._exit_cancel_confirm_timeout_seconds
        while time.monotonic() <= deadline:
            status_payload = self._get_broker_order_status(order_id)
            status = str((status_payload or {}).get("status") or "").strip().upper()
            if status in _legacy._FILLED_STATUSES:
                return "filled"
            if status in _legacy._CANCELLED_STATUSES:
                return "cancelled"
            time.sleep(self._exit_cancel_poll_interval_seconds)
        return "unconfirmed"

    def _safe_position_flat(self, symbol: str) -> bool:
        try:
            return bool(self._position_flat_for_symbol(symbol))
        except Exception as exc:  # noqa: BLE001
            _legacy.LOGGER.error(
                "EXIT_POSITION_RECONCILE_FAILED symbol=%s error=%s",
                symbol,
                exc,
                extra={
                    "event": "EXIT_POSITION_RECONCILE_FAILED",
                    "symbol": symbol,
                    "error_type": type(exc).__name__,
                },
            )
            return False

    def _escalate_exit_locked(self, bracket: Any, reason: str) -> None:
        order_key = str(bracket.exit_order_id or bracket.pending_exit_order_id or "none")
        key = (str(bracket.bracket_id), order_key)
        if bracket.escalated_at is not None or key in self._exit_escalation_notifications:
            bracket.exit_pending = True
            bracket.exit_state = (
                _legacy.BracketExitLifecycle.EXIT_FAILED_ESCALATED.value
            )
            bracket.entry_status = bracket.exit_state
            self._exit_escalation_notifications.add(key)
            return
        _LegacyBracketManager._escalate_exit_locked(self, bracket, reason)
        self._exit_escalation_notifications.add(key)


__all__ = ["HardenedBracketManager"]
