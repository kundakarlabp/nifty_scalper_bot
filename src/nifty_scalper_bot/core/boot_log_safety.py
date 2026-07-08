"""Rate controls for unchanged boot and live-readiness diagnostics.

This module is observational only. It does not alter contracts, orders,
strategy scores, or readiness decisions.
"""

from __future__ import annotations

import logging
import time
from typing import Any

CONTRACT_EVENTS = {
    "CONTRACT_SSOT_INSTRUMENTS_READY",
    "CONTRACT_SSOT_FUTURE_SELECTED",
    "CONTRACT_SSOT_OPTION_EXPIRY_SELECTED",
    "CONTRACT_SSOT_UNIVERSE_CAPPED",
    "CONTRACT_SSOT_ATM_PAIR_SELECTED",
    "CONTRACT_SSOT_BASKET_SELECTED",
}
RUNNER_EVENTS = {
    "strategy_stall_check_skipped_market_closed",
    "runner_post_grace_blocked",
    "RUNNER_EVAL_DECISION",
}
BOOTSTRAP_EVENTS = {
    "LIVE_UNIVERSE_BOOTSTRAP_STATUS",
    "SELECTED_OPTION_SUBSCRIPTION_STATE",
}
READINESS_EVENTS = {
    "READINESS_BLOCKER_SUMMARY",
    "LIVE_READINESS_COMPUTED",
    "LIVE_VALIDATION_CHECKLIST",
}
ORDERFLOW_EVENTS = {
    "ORDERFLOW_TRIGGER_DECISION",
    "ORDERFLOW_DIRECTION_BIAS_CONFLICT",
}


THROTTLED_EVENTS = CONTRACT_EVENTS | RUNNER_EVENTS | BOOTSTRAP_EVENTS | READINESS_EVENTS | ORDERFLOW_EVENTS


def _event(record: logging.LogRecord) -> str:
    value = getattr(record, "event", "")
    if value:
        return str(value)
    return str(record.getMessage() or "").split(" ", 1)[0]


def _fingerprint(record: logging.LogRecord) -> tuple[Any, ...]:
    return (
        _event(record),
        getattr(record, "symbol", None),
        getattr(record, "selected_ce", None),
        getattr(record, "selected_pe", None),
        getattr(record, "ce_symbol", None),
        getattr(record, "pe_symbol", None),
        getattr(record, "futures_symbol", None),
        getattr(record, "atm_strike", None),
        getattr(record, "option_count", None),
        getattr(record, "token_count", None),
        getattr(record, "count", None),
        getattr(record, "reason", None),
        getattr(record, "ready", None),
        getattr(record, "fresh_tick", None),
        getattr(record, "subscribed", None),
        getattr(record, "primary_blocker", None),
        tuple(getattr(record, "blockers", ()) or ()),
        tuple(getattr(record, "secondary_blockers", ()) or ()),
        getattr(record, "data_hard_ready", None),
        getattr(record, "evaluation_ready", None),
        getattr(record, "execution_ready", None),
        getattr(record, "live_orders_armed", None),
        getattr(record, "trigger_block_reason", None),
        getattr(record, "trigger_conditions_met", None),
        getattr(record, "side", None),
        getattr(record, "contract_side", None),
    )


class BootLogRateControl(logging.Filter):
    """Allow state changes and periodic unchanged-state heartbeats."""

    def __init__(self, interval_seconds: float = 300.0) -> None:
        super().__init__()
        self.interval_seconds = max(30.0, float(interval_seconds))
        self._last: dict[str, tuple[tuple[Any, ...], float]] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        event = _event(record)
        if event not in THROTTLED_EVENTS:
            return True
        fp = _fingerprint(record)
        now = time.monotonic()
        last = self._last.get(event)
        if last is None or last[0] != fp or now - last[1] >= self.interval_seconds:
            self._last[event] = (fp, now)
            return True
        return False


def _installed(logger: logging.Logger) -> bool:
    return any(isinstance(item, BootLogRateControl) for item in logger.filters)


def apply_filters() -> None:
    """Install idempotent boot/live diagnostic rate controls."""

    for name in (
        "nifty_scalper_bot.core.instrument_manager",
        "nifty_scalper_bot.core.app",
        "nifty_scalper_bot.execution.readiness",
        "nifty_scalper_bot.strategies.runner",
        "nifty_scalper_bot.strategies.elite_strategies.order_flow",
    ):
        logger = logging.getLogger(name)
        if not _installed(logger):
            logger.addFilter(BootLogRateControl())


__all__ = ["BootLogRateControl", "apply_filters"]
