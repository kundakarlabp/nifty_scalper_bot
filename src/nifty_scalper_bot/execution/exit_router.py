"""Deterministic exit routing without unsafe product or AMO fallbacks."""

from __future__ import annotations

import datetime as _dt
import time
from dataclasses import dataclass
from typing import Any, Callable

from nifty_scalper_bot.execution.broker_rejects import (
    BrokerReject,
    RecoveryAction,
    recovery_decision,
)


@dataclass(slots=True)
class ExitAttempt:
    """Describe a single routed exit submission attempt."""

    attempt: int
    product: str
    validity: str
    variety: str
    is_amo: bool
    note: str
    reject_reason: BrokerReject | None = None


@dataclass(slots=True)
class ExitResult:
    """Outcome of exit routing.

    ``ok`` means the broker accepted or reconciliation found an exit order.  It
    does not imply that the position is flat.  ``confirmed`` is true only when a
    caller-provided confirmation callback verifies the completed exit.
    """

    ok: bool
    order_id: str | None
    reason: BrokerReject
    attempts: list[ExitAttempt]
    confirmed: bool = False
    reconciliation_required: bool = False


ReconcileExit = Callable[[str], tuple[bool, str | None]]
ConfirmExit = Callable[[str], bool]


def _parse_time_token(value: str) -> _dt.time:
    """Return an ``HH:MM[:SS]`` string parsed into a time value."""

    parts = value.split(":")
    hour = int(parts[0]) if parts else 0
    minute = int(parts[1]) if len(parts) > 1 else 0
    second = int(parts[2]) if len(parts) > 2 else 0
    return _dt.time(hour=hour, minute=minute, second=second)


def _refresh_quote(
    refresh_quote: Callable[[str], None] | None,
    *,
    symbol: str,
    logger: Any,
    attempt: int,
) -> bool:
    if refresh_quote is None:
        return True
    try:
        refresh_quote(symbol)
        return True
    except Exception as exc:  # noqa: BLE001 - quote refresh is an external boundary
        logger.error(
            "EXIT_QUOTE_REFRESH_FAILED symbol=%s attempt=%s error=%s",
            symbol,
            attempt,
            exc,
            extra={
                "event": "EXIT_QUOTE_REFRESH_FAILED",
                "symbol": symbol,
                "attempt": attempt,
                "error_type": type(exc).__name__,
            },
        )
        return False


def plan_and_send_exit(
    *,
    symbol: str,
    quantity: int,
    product: str,
    now_time: _dt.time,
    submit: Callable[[str, str, str], tuple[bool, str | None, str | None]],
    logger: Any,
    refresh_quote: Callable[[str], None] | None = None,
    reconcile: ReconcileExit | None = None,
    confirm: ConfirmExit | None = None,
    mis_cutoff: str = "15:25",
    eod_flatten_start: str = "15:24:30",
    retry_max: int = 3,
) -> ExitResult:
    """Submit an exit with bounded, reason-specific recovery.

    The position's product is preserved.  This function never converts MIS to
    NRML and never creates a post-close AMO opposite order, because either can
    create a new exposure instead of closing the existing position.

    Ambiguous broker responses are reconciled before any resubmission.  Broker
    acknowledgement alone is reported as submitted but unconfirmed unless the
    optional ``confirm`` callback verifies the completed exit.
    """

    logger.debug(
        "Entered plan_and_send_exit",
        extra={
            "event": "plan_and_send_exit_enter",
            "symbol": symbol,
            "quantity": quantity,
            "product": product,
        },
    )
    attempts: list[ExitAttempt] = []
    safe_quantity = abs(int(quantity or 0))
    planned_product = str(product or "").strip().upper()
    if safe_quantity <= 0 or not planned_product:
        logger.error(
            "EXIT_ROUTE_INVALID_INPUT symbol=%s quantity=%s product=%s",
            symbol,
            quantity,
            product,
            extra={"event": "EXIT_ROUTE_INVALID_INPUT", "symbol": symbol},
        )
        return ExitResult(False, None, BrokerReject.QUANTITY_INVALID, attempts)

    # No protective exit can be safely manufactured after the exchange closes.
    # An AMO opposite-side order may open a new position on the next session.
    if now_time >= _dt.time(hour=15, minute=30):
        logger.critical(
            "EXIT_ROUTE_BLOCKED_POST_CLOSE symbol=%s product=%s quantity=%s",
            symbol,
            planned_product,
            safe_quantity,
            extra={
                "event": "EXIT_ROUTE_BLOCKED_POST_CLOSE",
                "symbol": symbol,
                "product": planned_product,
                "quantity": safe_quantity,
            },
        )
        return ExitResult(False, None, BrokerReject.MARKET_CLOSED, attempts)

    mis_cutoff_time = _parse_time_token(mis_cutoff)
    eod_time = _parse_time_token(eod_flatten_start)
    reason = BrokerReject.UNKNOWN
    retry_budget = max(1, int(retry_max or 1))

    for attempt in range(1, retry_budget + 1):
        note = "eod_flatten" if now_time >= eod_time else "market"
        if planned_product == "MIS" and now_time >= mis_cutoff_time:
            note = "mis_cutoff_window"

        attempt_meta = ExitAttempt(
            attempt=attempt,
            product=planned_product,
            validity="IOC",
            variety="regular",
            is_amo=False,
            note=note,
        )
        attempts.append(attempt_meta)

        if not _refresh_quote(
            refresh_quote,
            symbol=symbol,
            logger=logger,
            attempt=attempt,
        ):
            reason = BrokerReject.TRANSIENT_BROKER
            break

        logger.info(
            "EXIT_SUBMIT_ATTEMPT symbol=%s attempt=%s product=%s validity=IOC",
            symbol,
            attempt,
            planned_product,
            extra={
                "event": "EXIT_SUBMIT_ATTEMPT",
                "symbol": symbol,
                "attempt": attempt,
                "product": planned_product,
                "quantity": safe_quantity,
            },
        )
        ok, order_id, error_text = submit(planned_product, "IOC", "regular")
        if ok and order_id:
            confirmed = False
            if confirm is not None:
                try:
                    confirmed = bool(confirm(order_id))
                except Exception as exc:  # noqa: BLE001 - broker confirmation boundary
                    logger.error(
                        "EXIT_CONFIRMATION_FAILED symbol=%s order_id=%s error=%s",
                        symbol,
                        order_id,
                        exc,
                        extra={
                            "event": "EXIT_CONFIRMATION_FAILED",
                            "symbol": symbol,
                            "order_id": order_id,
                            "error_type": type(exc).__name__,
                        },
                    )
            logger.info(
                "EXIT_SUBMITTED symbol=%s order_id=%s confirmed=%s",
                symbol,
                order_id,
                confirmed,
                extra={
                    "event": "EXIT_SUBMITTED",
                    "symbol": symbol,
                    "attempt": attempt,
                    "order_id": order_id,
                    "confirmed": confirmed,
                },
            )
            return ExitResult(
                True,
                order_id,
                BrokerReject.UNKNOWN,
                attempts,
                confirmed=confirmed,
                reconciliation_required=not confirmed,
            )

        decision = recovery_decision(error_text)
        reason = decision.reason
        attempt_meta.reject_reason = reason
        logger.warning(
            "EXIT_SUBMIT_REJECTED symbol=%s attempt=%s reason=%s action=%s error=%s",
            symbol,
            attempt,
            reason.value,
            decision.action.value,
            error_text,
            extra={
                "event": "EXIT_SUBMIT_REJECTED",
                "symbol": symbol,
                "attempt": attempt,
                "reason": reason.value,
                "recovery_action": decision.action.value,
                "error": error_text,
            },
        )

        if decision.reconcile_before_retry:
            if reconcile is None:
                logger.error(
                    "EXIT_RECONCILIATION_REQUIRED symbol=%s reason=%s",
                    symbol,
                    reason.value,
                    extra={
                        "event": "EXIT_RECONCILIATION_REQUIRED",
                        "symbol": symbol,
                        "reason": reason.value,
                    },
                )
                return ExitResult(
                    False,
                    None,
                    reason,
                    attempts,
                    reconciliation_required=True,
                )
            try:
                recovered, recovered_order_id = reconcile(symbol)
            except Exception as exc:  # noqa: BLE001 - broker reconciliation boundary
                logger.error(
                    "EXIT_RECONCILIATION_FAILED symbol=%s error=%s",
                    symbol,
                    exc,
                    extra={
                        "event": "EXIT_RECONCILIATION_FAILED",
                        "symbol": symbol,
                        "error_type": type(exc).__name__,
                    },
                )
                return ExitResult(
                    False,
                    None,
                    reason,
                    attempts,
                    reconciliation_required=True,
                )
            if recovered:
                confirmed = False
                if recovered_order_id and confirm is not None:
                    try:
                        confirmed = bool(confirm(recovered_order_id))
                    except Exception:  # noqa: BLE001 - handled as pending reconciliation
                        confirmed = False
                return ExitResult(
                    True,
                    recovered_order_id,
                    reason,
                    attempts,
                    confirmed=confirmed,
                    reconciliation_required=not confirmed,
                )
            # Reconciliation proved absence, so a bounded retry is permitted.
            if attempt < retry_budget:
                continue
            break

        if decision.action is RecoveryAction.RETRY_WITH_BACKOFF:
            if attempt < retry_budget:
                time.sleep(min(0.25 * (2 ** (attempt - 1)), 1.0))
                continue
            break

        if decision.action is RecoveryAction.REFRESH_MARKET_AND_REVALIDATE:
            # The submit callback is responsible for rebuilding any price fields
            # from the newly refreshed quote.  Do not retry when no refresh path
            # exists because that would resend the same invalid request.
            if refresh_quote is not None and attempt < retry_budget:
                continue
            break

        # Margin resize, freeze splitting and product conversion require a
        # higher-level position/risk context.  This router never guesses them.
        break

    logger.error(
        "EXIT_ROUTE_FAILED symbol=%s attempts=%s reason=%s",
        symbol,
        len(attempts),
        reason.value,
        extra={
            "event": "EXIT_ROUTE_FAILED",
            "symbol": symbol,
            "attempts": len(attempts),
            "reason": reason.value,
        },
    )
    return ExitResult(False, None, reason, attempts)


__all__ = ["ExitAttempt", "ExitResult", "plan_and_send_exit"]
