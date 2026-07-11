"""Canonical runtime bracket manager for staged BO lifecycle consolidation.

Stage 1 intentionally preserves the established hardened manager behaviour and
adds only fill-integrity invariants:

* confirmed entry slippage re-anchors every unexecuted target;
* a broker FILLED exit cannot close local state while broker exposure remains;
* a confirmed TP1 fill resumes protection for the residual position;
* any other FILLED/non-flat mismatch fails closed and blocks new entries.

The compatibility inheritance is temporary. Later stages will fold these
invariants into the single explicit bracket implementation and remove runtime
class replacement.
"""

from __future__ import annotations

from contextlib import suppress
import os
import time
from datetime import datetime, timezone
from typing import Any, Mapping

from nifty_scalper_bot.execution import bracket_manager as _legacy
from nifty_scalper_bot.execution.hardened_bracket_manager import (
    HardenedBracketManager,
)
from nifty_scalper_bot.execution.broker_position_evidence import (
    BrokerPositionEvidence,
    BrokerPositionState,
    normalize_authoritative_quantity,
)
from nifty_scalper_bot.execution.order_state import (
    DomainOrderState,
    map_broker_order_status,
)


class CanonicalBracketManager(HardenedBracketManager):
    """Single runtime-exported bracket manager with strict fill reconciliation."""

    _FILLED_NONFLAT_PREFIX = "filled_exit_nonflat"
    _FILLED_SYNC_PREFIX = "filled_exit_position_sync_pending"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Broker order status can lead the positions endpoint briefly.  Keep this
        # bounded: we defer a decision, but never declare the position flat.
        self._filled_position_sync_grace_seconds = max(
            0.0,
            _legacy.parse_float_env(
                os.getenv("EXIT_FILLED_POSITION_SYNC_GRACE_SECONDS"),
                1.5,
            ),
        )
        self._exit_fill_confirmation_grace_seconds = max(
            0.001,
            _legacy.parse_float_env(
                os.getenv(
                    "EXECUTION_EXIT_FILL_CONFIRMATION_GRACE_SECONDS",
                    os.getenv("EXIT_FILL_CONFIRMATION_GRACE_SECONDS"),
                ),
                10.0,
            ),
        )
        super().__init__(*args, **kwargs)

    def confirm_entry_fill(self, order_id: str, fill_price: float) -> None:
        """Activate a confirmed entry and re-anchor every outstanding target.

        The legacy implementation already re-anchors SL and the final TP. This
        wrapper captures planned partial targets before that call and applies the
        same proportional re-anchor exactly once.
        """

        before = self.get_bracket(order_id)
        planned_entry = 0.0
        planned_targets: list[float] = []
        if before is not None:
            with self._lock:
                planned_entry = float(before.entry_price or 0.0)
                planned_targets = [
                    float(target.price or 0.0) for target in before.tp_levels
                ]

        super().confirm_entry_fill(order_id, fill_price)

        bracket = self.get_bracket(order_id)
        if bracket is None:
            return
        try:
            confirmed_fill = float(fill_price or 0.0)
        except (TypeError, ValueError):
            return
        if planned_entry <= 0 or confirmed_fill <= 0 or confirmed_fill == planned_entry:
            return

        ratio = confirmed_fill / planned_entry
        changed: list[dict[str, object]] = []
        with self._lock:
            for index, target in enumerate(bracket.tp_levels):
                if target.executed or index >= len(planned_targets):
                    continue
                old_price = planned_targets[index]
                if old_price <= 0:
                    continue
                new_price = _legacy._round_to_tick(old_price * ratio)
                if new_price <= 0 or new_price == float(target.price):
                    continue
                target.price = new_price
                changed.append(
                    {
                        "name": str(target.name),
                        "old_price": old_price,
                        "new_price": new_price,
                    }
                )
            if changed:
                bracket.updated_at = time.time()

        if not changed:
            return
        with suppress(Exception):
            self.save_state()
        _legacy.LOGGER.info(
            "BRACKET_TARGETS_REANCHORED bracket_id=%s symbol=%s planned_entry=%.2f fill_price=%.2f targets=%s",
            bracket.bracket_id,
            bracket.symbol,
            planned_entry,
            confirmed_fill,
            changed,
            extra={
                "event": "BRACKET_TARGETS_REANCHORED",
                "bracket_id": bracket.bracket_id,
                "symbol": bracket.symbol,
                "planned_entry": planned_entry,
                "fill_price": confirmed_fill,
                "targets": changed,
            },
        )
        with suppress(Exception):
            self._notify_event(
                "BRACKET_TARGETS_REANCHORED",
                {
                    "symbol": bracket.symbol,
                    "planned_entry": round(planned_entry, 2),
                    "fill_price": round(confirmed_fill, 2),
                    "targets": changed,
                },
            )

    def _broker_position_quantity(self, symbol: str) -> int | None:
        """Return absolute broker quantity, or ``None`` when exposure is unknown."""
        try:
            return self._authoritative_position_quantity(symbol)
        except Exception as exc:  # noqa: BLE001
            _legacy.LOGGER.error(
                "BROKER_POSITION_QUANTITY_UNKNOWN symbol=%s error=%s",
                _legacy.normalize_symbol(symbol),
                exc,
                extra={
                    "event": "BROKER_POSITION_QUANTITY_UNKNOWN",
                    "symbol": _legacy.normalize_symbol(symbol),
                    "error_type": type(exc).__name__,
                },
            )
            return None

    @staticmethod
    def _matching_partial_target(bracket: Any) -> Any | None:
        reason = str(getattr(bracket, "exit_reason", "") or "").strip().upper()
        for target in getattr(bracket, "tp_levels", []) or []:
            if bool(getattr(target, "executed", False)):
                continue
            name = str(getattr(target, "name", "") or "").strip().upper()
            if name and (reason.startswith(name) or f"{name} HIT" in reason):
                return target
        return None

    @staticmethod
    def _clear_fill_sync_state(bracket: Any) -> None:
        bracket._filled_exit_sync_started_at = 0.0
        bracket._filled_exit_sync_order_id = None

    def _fill_sync_grace_expired(self, bracket: Any, order_id: str) -> bool:
        now = time.time()
        if (
            bracket._filled_exit_sync_order_id != order_id
            or bracket._filled_exit_sync_started_at <= 0
        ):
            bracket._filled_exit_sync_order_id = order_id
            bracket._filled_exit_sync_started_at = now
        return (
            now - bracket._filled_exit_sync_started_at
        ) >= self._filled_position_sync_grace_seconds

    def _defer_filled_position_sync(
        self,
        bracket: Any,
        *,
        residual_quantity: int | None,
        order_id: str,
        requested_by: str,
    ) -> None:
        with self._lock:
            bracket.exit_in_progress = False
            bracket.exit_pending = True
            bracket.exit_executed = False
            bracket.active = True
            bracket.position_flat_confirmed = False
            bracket.exit_state = (
                _legacy.BracketExitLifecycle.EXIT_PARTIALLY_FILLED.value
            )
            bracket.entry_status = bracket.exit_state
            bracket.last_exit_error = (
                f"{self._FILLED_SYNC_PREFIX}:"
                f"residual={residual_quantity if residual_quantity is not None else 'unknown'}"
            )
            bracket.updated_at = time.time()

        _legacy.LOGGER.warning(
            "EXIT_FILLED_POSITION_SYNC_PENDING bracket_id=%s symbol=%s order_id=%s residual_qty=%s requested_by=%s grace_seconds=%.2f",
            bracket.bracket_id,
            bracket.symbol,
            order_id,
            residual_quantity,
            requested_by,
            self._filled_position_sync_grace_seconds,
            extra={
                "event": "EXIT_FILLED_POSITION_SYNC_PENDING",
                "bracket_id": bracket.bracket_id,
                "symbol": bracket.symbol,
                "order_id": order_id,
                "residual_qty": residual_quantity,
                "requested_by": requested_by,
                "grace_seconds": self._filled_position_sync_grace_seconds,
            },
        )

    def _resume_after_partial_target(
        self,
        bracket: Any,
        *,
        target: Any,
        residual_quantity: int,
        order_id: str,
        status_payload: Mapping[str, Any],
        requested_by: str,
    ) -> bool:
        previous_remaining = int(bracket.remaining_quantity or 0)
        expected_residual = max(previous_remaining - int(target.quantity or 0), 0)
        if (
            residual_quantity >= previous_remaining
            or residual_quantity > expected_residual
        ):
            return False

        fill_price = self._extract_status_price(status_payload)
        filled_quantity = previous_remaining - residual_quantity
        with self._lock:
            target.executed = True
            bracket.remaining_quantity = residual_quantity
            bracket.exit_order_id = None
            bracket.pending_exit_order_id = None
            bracket.exit_pending = False
            bracket.exit_in_progress = False
            bracket.exit_executed = False
            bracket.active = True
            bracket.entry_confirmed = True
            bracket.position_flat_confirmed = False
            bracket.exit_state = _legacy.BracketExitLifecycle.OPEN_ACTIVE.value
            bracket.entry_status = "ACTIVE"
            bracket.last_exit_error = None
            bracket.exit_reason = None
            bracket.exit_triggered_at = None
            bracket.exit_attempt_count = 0
            bracket.last_exit_attempt_at = None
            bracket.next_exit_attempt_at = None
            bracket.escalated_at = None
            bracket._market_escalation_fired = False
            bracket.updated_at = time.time()
            self._exit_order_open_since.pop(order_id, None)
            self._exit_rescue_attempts.pop(bracket.bracket_id, None)
            self._clear_fill_sync_state(bracket)

        self._move_sl_to_breakeven(bracket)
        with suppress(Exception):
            self.save_state()
        _legacy.LOGGER.info(
            "PARTIAL_EXIT_CONFIRMED_RESIDUAL_PROTECTED bracket_id=%s symbol=%s target=%s filled_qty=%s remaining_qty=%s fill_price=%s requested_by=%s",
            bracket.bracket_id,
            bracket.symbol,
            target.name,
            filled_quantity,
            residual_quantity,
            fill_price,
            requested_by,
            extra={
                "event": "PARTIAL_EXIT_CONFIRMED_RESIDUAL_PROTECTED",
                "bracket_id": bracket.bracket_id,
                "symbol": bracket.symbol,
                "target": str(target.name),
                "filled_qty": filled_quantity,
                "remaining_qty": residual_quantity,
                "fill_price": fill_price,
                "requested_by": requested_by,
            },
        )
        with suppress(Exception):
            self._notify_event(
                "PARTIAL_EXIT_CONFIRMED",
                {
                    "symbol": bracket.symbol,
                    "target": str(target.name),
                    "filled_qty": filled_quantity,
                    "remaining_qty": residual_quantity,
                    "fill_price": fill_price,
                    "sl": round(float(bracket.sl_trigger_price), 2),
                },
            )
        return True

    def _latch_filled_nonflat_mismatch(
        self,
        bracket: Any,
        *,
        residual_quantity: int | None,
        order_id: str | None,
        requested_by: str,
    ) -> None:
        now = time.time()
        error = (
            f"{self._FILLED_NONFLAT_PREFIX}:"
            f"residual={residual_quantity if residual_quantity is not None else 'unknown'}"
        )
        with self._lock:
            if residual_quantity is not None and residual_quantity > 0:
                bracket.remaining_quantity = residual_quantity
            # Retain the completed order id for audit and later reconciliation.
            bracket.exit_in_progress = False
            bracket.exit_pending = True
            bracket.exit_executed = False
            bracket.active = True
            bracket.position_flat_confirmed = False
            bracket.exit_state = (
                _legacy.BracketExitLifecycle.EXIT_FAILED_ESCALATED.value
            )
            bracket.entry_status = bracket.exit_state
            bracket.last_exit_error = error
            bracket.escalated_at = bracket.escalated_at or now
            # Prevent inherited escalation from sending a guessed extra market exit.
            bracket._market_escalation_fired = True
            bracket.updated_at = now
            if order_id:
                self._exit_order_open_since.pop(order_id, None)

        with suppress(Exception):
            self.save_state()
        _legacy.LOGGER.critical(
            "EXIT_FILLED_POSITION_MISMATCH bracket_id=%s symbol=%s order_id=%s residual_qty=%s requested_by=%s",
            bracket.bracket_id,
            bracket.symbol,
            order_id,
            residual_quantity,
            requested_by,
            extra={
                "event": "EXIT_FILLED_POSITION_MISMATCH",
                "bracket_id": bracket.bracket_id,
                "symbol": bracket.symbol,
                "order_id": order_id,
                "residual_qty": residual_quantity,
                "requested_by": requested_by,
            },
        )
        with suppress(Exception):
            self._notify_event(
                "EXIT_FILLED_POSITION_MISMATCH",
                {
                    "symbol": bracket.symbol,
                    "order_id": order_id,
                    "remaining_qty": residual_quantity,
                    "message": "Exit order filled but broker position remains. New entries are blocked pending reconciliation.",
                },
            )

    def _reconcile_exit_state(self, bracket: Any, *, requested_by: str) -> bool:
        """Reconcile filled exits without allowing optimistic local closure."""

        with self._lock:
            order_id = bracket.exit_order_id or bracket.pending_exit_order_id
            mismatch_latched = str(bracket.last_exit_error or "").startswith(
                self._FILLED_NONFLAT_PREFIX
            )

        if mismatch_latched and not order_id:
            residual = self._broker_position_quantity(bracket.symbol)
            if residual == 0:
                self._clear_fill_sync_state(bracket)
                self._close_bracket(
                    bracket,
                    close_source="filled_nonflat_reconciled_flat",
                )
                return True
            return False

        if order_id:
            try:
                status_payload = self._get_broker_order_status(str(order_id))
                status = str((status_payload or {}).get("status") or "").strip().upper()
            except Exception:
                return super()._reconcile_exit_state(
                    bracket,
                    requested_by=requested_by,
                )

            mapped_status = map_broker_order_status(status)
            if mapped_status.state is not DomainOrderState.FILLED:
                flat_evidence = self._broker_position_evidence(bracket.symbol)
                if flat_evidence.state is not BrokerPositionState.FLAT_CONFIRMED:
                    bracket.flat_nonterminal_since_monotonic = None
                    bracket.flat_nonterminal_since_utc = None
                strict_live = bool(getattr(self, "_is_live_execution", lambda: True)())
                if (
                    strict_live
                    and flat_evidence.state not in {
                        BrokerPositionState.FLAT_CONFIRMED,
                        BrokerPositionState.NON_FLAT_CONFIRMED,
                    }
                ):
                    with self._lock:
                        bracket.exit_pending = True
                        bracket.exit_in_progress = False
                        bracket.position_flat_confirmed = False
                        bracket.last_exit_error = (
                            f"broker_position_{flat_evidence.state.value}"
                        )
                        age_basis = float(
                            bracket.last_exit_attempt_at
                            or bracket.exit_triggered_at
                            or time.time()
                        )
                        age = time.time() - age_basis
                        if age >= float(
                            getattr(
                                self,
                                "_exit_unresolved_escalation_seconds",
                                0.0,
                            )
                            or 0.0
                        ):
                            bracket.exit_state = (
                                _legacy.BracketExitLifecycle.EXIT_FAILED_ESCALATED.value
                            )
                            bracket.entry_status = bracket.exit_state
                            bracket.escalated_at = bracket.escalated_at or time.time()
                            bracket._market_escalation_fired = True
                        bracket.updated_at = time.time()
                    _legacy.LOGGER.warning(
                        "EXIT_RECONCILE_DEFERRED_BROKER_POSITION_UNKNOWN bracket_id=%s symbol=%s order_id=%s state=%s",
                        bracket.bracket_id,
                        bracket.symbol,
                        order_id,
                        flat_evidence.state.value,
                        extra={
                            "event": "EXIT_RECONCILE_DEFERRED_BROKER_POSITION_UNKNOWN",
                            "bracket_id": bracket.bracket_id,
                            "symbol": bracket.symbol,
                            "order_id": str(order_id),
                            "broker_position_state": flat_evidence.state.value,
                            "error": flat_evidence.error,
                        },
                    )
                    return False
                if flat_evidence.state is BrokerPositionState.FLAT_CONFIRMED:
                    now_monotonic = time.monotonic()
                    since = getattr(bracket, "flat_nonterminal_since_monotonic", None)
                    if since is None:
                        bracket.flat_nonterminal_since_monotonic = now_monotonic
                        bracket.flat_nonterminal_since_utc = datetime.now(
                            timezone.utc
                        ).isoformat()
                        age = 0.0
                    else:
                        age = now_monotonic - float(since)
                    with self._lock:
                        bracket.exit_pending = True
                        bracket.exit_in_progress = False
                        bracket.position_flat_confirmed = False
                        bracket.exit_state = (
                            _legacy.BracketExitLifecycle.EXIT_ORDER_SUBMITTED.value
                        )
                        bracket.entry_status = "EXIT_PENDING"
                        bracket.updated_at = time.time()
                    log = (
                        _legacy.LOGGER.info
                        if age < self._exit_fill_confirmation_grace_seconds
                        else _legacy.LOGGER.warning
                    )
                    log(
                        "EXIT_FLAT_BUT_ORDER_NOT_TERMINAL bracket_id=%s symbol=%s order_id=%s order_status=%s age_seconds=%.3f grace_seconds=%.3f close_deferred=True",
                        bracket.bracket_id,
                        bracket.symbol,
                        order_id,
                        mapped_status.raw_status or "UNKNOWN",
                        age,
                        self._exit_fill_confirmation_grace_seconds,
                        extra={
                            "event": "EXIT_FLAT_BUT_ORDER_NOT_TERMINAL",
                            "bracket_id": bracket.bracket_id,
                            "symbol": bracket.symbol,
                            "order_id": str(order_id),
                            "order_status": mapped_status.raw_status or "UNKNOWN",
                            "broker_position_state": flat_evidence.state.value,
                            "age_seconds": age,
                            "grace_seconds": self._exit_fill_confirmation_grace_seconds,
                        },
                    )
                    return False

            if mapped_status.state is DomainOrderState.FILLED:
                residual = self._broker_position_quantity(bracket.symbol)
                if residual == 0:
                    self._clear_fill_sync_state(bracket)
                    return super()._reconcile_exit_state(
                        bracket,
                        requested_by=requested_by,
                    )

                target = self._matching_partial_target(bracket)
                if (
                    residual is not None
                    and target is not None
                    and self._resume_after_partial_target(
                        bracket,
                        target=target,
                        residual_quantity=residual,
                        order_id=str(order_id),
                        status_payload=status_payload,
                        requested_by=requested_by,
                    )
                ):
                    return False

                if not self._fill_sync_grace_expired(bracket, str(order_id)):
                    self._defer_filled_position_sync(
                        bracket,
                        residual_quantity=residual,
                        order_id=str(order_id),
                        requested_by=requested_by,
                    )
                    return False

                self._latch_filled_nonflat_mismatch(
                    bracket,
                    residual_quantity=residual,
                    order_id=str(order_id),
                    requested_by=requested_by,
                )
                return False

        return super()._reconcile_exit_state(
            bracket,
            requested_by=requested_by,
        )

    def _process_exit_state(
        self,
        bracket: Any,
        action: Mapping[str, Any],
        *,
        now: float,
    ) -> None:
        """Keep a filled/non-flat mismatch frozen until broker truth is resolved."""

        if str(bracket.last_exit_error or "").startswith(self._FILLED_NONFLAT_PREFIX):
            self._reconcile_exit_state(
                bracket,
                requested_by="filled_nonflat_followup",
            )
            if bracket.exit_state != _legacy.BracketExitLifecycle.CLOSED.value:
                with self._lock:
                    self._log_exit_pending_summary_locked(bracket, now)
            return
        super()._process_exit_state(bracket, action, now=now)

    def _broker_position_evidence(self, symbol: str) -> BrokerPositionEvidence:
        """Return typed broker-position truth; unknown/API errors fail closed."""
        now = datetime.now(timezone.utc)
        try:
            qty = self._authoritative_position_quantity(symbol)
        except Exception as exc:  # noqa: BLE001 - broker boundary
            _legacy.LOGGER.error(
                "BROKER_POSITION_EVIDENCE_API_ERROR symbol=%s error_type=%s",
                _legacy.normalize_symbol(symbol),
                type(exc).__name__,
                extra={
                    "event": "BROKER_POSITION_EVIDENCE_API_ERROR",
                    "symbol": _legacy.normalize_symbol(symbol),
                    "error_type": type(exc).__name__,
                },
            )
            return BrokerPositionEvidence(
                BrokerPositionState.API_ERROR,
                _legacy.normalize_symbol(symbol),
                None,
                now,
                0.0,
                "authoritative_position_quantity",
                type(exc).__name__,
            )
        parsed_qty = normalize_authoritative_quantity(qty)
        if parsed_qty is None:
            _legacy.LOGGER.warning(
                "BROKER_POSITION_EVIDENCE_UNKNOWN symbol=%s raw_type=%s",
                _legacy.normalize_symbol(symbol),
                type(qty).__name__,
                extra={
                    "event": "BROKER_POSITION_EVIDENCE_UNKNOWN",
                    "symbol": _legacy.normalize_symbol(symbol),
                    "raw_type": type(qty).__name__,
                },
            )
            return BrokerPositionEvidence(
                BrokerPositionState.UNKNOWN,
                _legacy.normalize_symbol(symbol),
                None,
                now,
                0.0,
                "authoritative_position_quantity",
                "invalid_quantity",
            )
        state = (
            BrokerPositionState.FLAT_CONFIRMED
            if parsed_qty == 0
            else BrokerPositionState.NON_FLAT_CONFIRMED
        )
        return BrokerPositionEvidence(
            state,
            _legacy.normalize_symbol(symbol),
            parsed_qty,
            now,
            0.0,
            "authoritative_position_quantity",
        )

    def _safe_position_flat(self, symbol: str) -> bool:
        return (
            self._broker_position_evidence(symbol).state
            is BrokerPositionState.FLAT_CONFIRMED
        )

    def _rescue_stale_exit_order(
        self,
        bracket: Any,
        *,
        order_id: str,
        qty: int,
        status: str,
    ) -> None:
        """Cancel/replace stale exits without treating a non-flat fill as closure."""

        flat_evidence = self._broker_position_evidence(bracket.symbol)
        strict_live = bool(getattr(self, "_is_live_execution", lambda: True)())
        if strict_live and flat_evidence.state not in {
            BrokerPositionState.FLAT_CONFIRMED,
            BrokerPositionState.NON_FLAT_CONFIRMED,
        }:
            with self._lock:
                bracket.exit_in_progress = False
                bracket.exit_pending = True
                bracket.position_flat_confirmed = False
                bracket.last_exit_error = f"broker_position_{flat_evidence.state.value}"
                bracket.updated_at = time.time()
            _legacy.LOGGER.warning(
                "EXIT_RESCUE_DEFERRED_BROKER_POSITION_UNKNOWN bracket_id=%s symbol=%s order_id=%s state=%s",
                bracket.bracket_id,
                bracket.symbol,
                order_id,
                flat_evidence.state.value,
                extra={
                    "event": "EXIT_RESCUE_DEFERRED_BROKER_POSITION_UNKNOWN",
                    "bracket_id": bracket.bracket_id,
                    "symbol": bracket.symbol,
                    "order_id": str(order_id),
                    "broker_position_state": flat_evidence.state.value,
                    "error": flat_evidence.error,
                },
            )
            return
        if flat_evidence.state is BrokerPositionState.NON_FLAT_CONFIRMED:
            bracket.flat_nonterminal_since_monotonic = None
            bracket.flat_nonterminal_since_utc = None
        if flat_evidence.state is BrokerPositionState.FLAT_CONFIRMED:
            now_monotonic = time.monotonic()
            _since = getattr(bracket, "flat_nonterminal_since_monotonic", None)
            if _since is None:
                bracket.flat_nonterminal_since_monotonic = now_monotonic
                bracket.flat_nonterminal_since_utc = datetime.now(
                    timezone.utc
                ).isoformat()
                _flat_age = 0.0
            else:
                _flat_age = now_monotonic - float(_since)

            if _flat_age < self._exit_fill_confirmation_grace_seconds:
                # Broker already FLAT: the exit filled and only the order
                # status is propagating. Cancel-racing a filled order is pure
                # churn ('Skipping cancel: Already FILLED', 2026-07-10);
                # the reconcile loop confirms and closes within the grace.
                _legacy.LOGGER.info(
                    "EXIT_RESCUE_SKIPPED_FLAT_LATENCY bracket_id=%s order_id=%s status=%s",
                    bracket.bracket_id,
                    order_id,
                    status,
                    extra={
                        "event": "EXIT_RESCUE_SKIPPED_FLAT_LATENCY",
                        "bracket_id": bracket.bracket_id,
                        "order_id": order_id,
                    },
                )
                return
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
        terminal = (
            self._wait_for_cancel_or_fill(order_id)
            if cancel_requested
            else "unconfirmed"
        )
        if terminal == "filled":
            with self._lock:
                bracket.exit_in_progress = False
            self._reconcile_exit_state(
                bracket,
                requested_by="stale_rescue_filled",
            )
            return
        flat_evidence = self._broker_position_evidence(bracket.symbol)
        if flat_evidence.state is BrokerPositionState.FLAT_CONFIRMED:
            with self._lock:
                bracket.exit_in_progress = False
            self._close_bracket(
                bracket,
                close_source="rescue_reconciled_flat",
            )
            return
        if terminal != "cancelled":
            with self._lock:
                bracket.exit_in_progress = False
                bracket.last_exit_error = "stale_exit_cancel_unconfirmed"
                self._escalate_exit_locked(
                    bracket,
                    "stale_exit_cancel_unconfirmed",
                )
            return

        with self._lock:
            bracket.exit_in_progress = False
            bracket.exit_order_id = None
            bracket.pending_exit_order_id = None
            bracket.last_exit_error = None
        self._submit_rescue_exit(
            bracket,
            qty=qty,
            prior_order_id=order_id,
        )


__all__ = ["CanonicalBracketManager"]
