"""Deterministic exit routing with MIS cutoff and AMO fallbacks."""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Any, Callable

from nifty_scalper_bot.execution.broker_rejects import BrokerReject, parse_broker_error


@dataclass(slots=True)
class ExitAttempt:
    """Describe a single routed exit attempt."""

    attempt: int
    product: str
    validity: str
    variety: str
    is_amo: bool
    note: str


@dataclass(slots=True)
class ExitResult:
    """Outcome of exit routing."""

    ok: bool
    order_id: str | None
    reason: BrokerReject
    attempts: list[ExitAttempt]


def _parse_time_token(value: str) -> _dt.time:
    """Return HH:MM[:SS] string parsed into time."""

    parts = value.split(":")
    hour = int(parts[0]) if parts else 0
    minute = int(parts[1]) if len(parts) > 1 else 0
    second = int(parts[2]) if len(parts) > 2 else 0
    return _dt.time(hour=hour, minute=minute, second=second)


def plan_and_send_exit(
    *,
    symbol: str,
    quantity: int,
    product: str,
    now_time: _dt.time,
    submit: Callable[[str, str, str], tuple[bool, str | None, str | None]],
    logger: Any,
    refresh_quote: Callable[[str], None] | None = None,
    mis_cutoff: str = "15:25",
    eod_flatten_start: str = "15:24:30",
    retry_max: int = 3,
) -> ExitResult:
    """Route exit order with market/AMO fallbacks.

    Args:
        symbol: Tradable symbol for the exit.
        quantity: Absolute exit quantity.
        product: Preferred product (for example ``MIS``).
        now_time: Current IST time for routing decisions.
        submit: Callback that executes an exit attempt.
        logger: Project logger for diagnostics.
        refresh_quote: Optional quote refresh callable.
        mis_cutoff: MIS cutoff time in ``HH:MM[:SS]`` format.
        eod_flatten_start: Time when forced exits begin.
        retry_max: Maximum retry attempts.

    Returns:
        ExitResult: Structured outcome with attempts.
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
    if refresh_quote is not None:
        try:
            refresh_quote(symbol)
        except Exception as exc:  # noqa: BLE001 - best effort
            logger.error(
                "Failure in plan_and_send_exit: %s",
                exc,
                extra={
                    "event": "plan_and_send_exit_quote_refresh_failed",
                    "symbol": symbol,
                },
            )
    mis_cutoff_time = _parse_time_token(mis_cutoff)
    eod_time = _parse_time_token(eod_flatten_start)
    reason = BrokerReject.UNKNOWN

    for attempt in range(1, max(retry_max, 1) + 1):
        planned_product = product or "MIS"
        validity = "IOC"
        variety = "regular"
        note = "market"
        amo = False
        route_time = now_time
        if route_time >= mis_cutoff_time:
            planned_product = "NRML"
        if route_time >= _dt.time(hour=15, minute=30):
            amo = True
            variety = "amo"
            validity = "DAY"
            planned_product = "NRML"
            note = "post_close"
        elif route_time >= eod_time:
            planned_product = "NRML"
            note = "eod_flatten"
        attempt_meta = ExitAttempt(
            attempt=attempt,
            product=planned_product,
            validity=validity,
            variety=variety,
            is_amo=amo,
            note=note,
        )
        attempts.append(attempt_meta)
        logger.info(
            "Condition met: exit_plan attempt=%s product=%s validity=%s variety=%s",
            attempt,
            planned_product,
            validity,
            variety,
            extra={
                "event": "plan_and_send_exit_attempt",
                "symbol": symbol,
                "attempt": attempt,
                "product": planned_product,
                "validity": validity,
                "variety": variety,
                "amo": amo,
            },
        )
        ok, order_id, error_text = submit(planned_product, validity, variety)
        if ok:
            logger.info(
                "Condition met: exit_ok order_id=%s",
                order_id,
                extra={
                    "event": "plan_and_send_exit_success",
                    "symbol": symbol,
                    "attempt": attempt,
                    "order_id": order_id,
                },
            )
            return ExitResult(True, order_id, BrokerReject.UNKNOWN, attempts)
        reason = parse_broker_error(error_text)
        logger.warning(
            "exit_retry reason=%s attempt=%s",
            reason.value,
            attempt,
            extra={
                "event": "plan_and_send_exit_retry",
                "symbol": symbol,
                "attempt": attempt,
                "reason": reason.value,
                "error": error_text,
            },
        )
        if reason is BrokerReject.MARKET_CLOSED:
            now_time = _dt.time(hour=15, minute=30)
        elif reason is BrokerReject.MIS_CUTOFF:
            now_time = max(now_time, mis_cutoff_time)
        elif reason is BrokerReject.THROTTLED:
            continue
    logger.error(
        "Failure in plan_and_send_exit: attempts_exhausted",
        extra={
            "event": "plan_and_send_exit_failed",
            "symbol": symbol,
            "attempts": len(attempts),
            "reason": reason.value,
        },
    )
    return ExitResult(False, None, reason, attempts)


__all__ = ["ExitAttempt", "ExitResult", "plan_and_send_exit"]
