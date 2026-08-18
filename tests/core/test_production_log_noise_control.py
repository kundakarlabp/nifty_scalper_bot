from __future__ import annotations

import logging
from pathlib import Path

from nifty_scalper_bot.core.boot_log_safety import BootLogRateControl
from nifty_scalper_bot.utils.log_throttle import event_is_never_throttled


def _record(event: str, *, level: int = logging.INFO, **extra: object) -> logging.LogRecord:
    record = logging.LogRecord(
        name="nifty_scalper_bot.core.app",
        level=level,
        pathname=__file__,
        lineno=1,
        msg=event,
        args=(),
        exc_info=None,
    )
    record.event = event
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_reconcile_started_is_debug_only() -> None:
    control = BootLogRateControl(interval_seconds=300.0)
    record = _record("POSITION_RECONCILE_STARTED")
    assert control.filter(record) is True
    assert record.levelno == logging.DEBUG


def test_reconcile_success_is_state_throttled_and_failure_forces_recovery_visibility() -> None:
    control = BootLogRateControl(interval_seconds=300.0)
    first = _record("POSITION_RECONCILE_SUCCESS")
    duplicate = _record("POSITION_RECONCILE_SUCCESS")
    failed = _record("POSITION_RECONCILE_FAILED", level=logging.ERROR)
    recovered = _record("POSITION_RECONCILE_SUCCESS")

    assert control.filter(first) is True
    assert control.filter(duplicate) is False
    assert control.filter(failed) is True
    assert failed.levelno == logging.ERROR
    assert control.filter(recovered) is True


def test_same_bar_periodic_skip_is_debug_but_real_pregate_block_stays_info() -> None:
    control = BootLogRateControl(interval_seconds=300.0)
    routine = _record(
        "RUNNER_EVAL_PREGATE_SKIPPED",
        reason="same_bar_periodic_eval_throttled",
    )
    actionable = _record(
        "RUNNER_EVAL_PREGATE_SKIPPED",
        reason="option_bid_ask_missing_or_invalid",
    )

    assert control.filter(routine) is True
    assert routine.levelno == logging.DEBUG
    assert control.filter(actionable) is True
    assert actionable.levelno == logging.INFO


def test_routine_strategy_rejection_is_debug_but_warning_is_preserved() -> None:
    control = BootLogRateControl(interval_seconds=300.0)
    routine = _record("STRATEGY_NO_VOTE", reason="smc_quality_gate_failed")
    readiness_warning = _record(
        "STRATEGY_NO_VOTE",
        level=logging.WARNING,
        reason="smc_history_count_missing",
    )

    assert control.filter(routine) is True
    assert routine.levelno == logging.DEBUG
    assert control.filter(readiness_warning) is True
    assert readiness_warning.levelno == logging.WARNING


def test_context_only_and_no_combined_diagnostics_are_debug_only() -> None:
    control = BootLogRateControl(interval_seconds=300.0)
    for event in (
        "strategy_manager_no_combined_signal",
        "PERMANENT_CONTEXT_ONLY_PROMOTION_BLOCKED",
        "ORDERFLOW_TRIGGER_SCORE",
        "POSITION_RECONCILE_COALESCED",
    ):
        record = _record(event)
        assert control.filter(record) is True
        assert record.levelno == logging.DEBUG


def test_trade_and_safety_events_remain_unthrottled() -> None:
    for event in (
        "ORDER_SUBMITTED",
        "ORDER_FILLED",
        "POSITION_OPENED",
        "POSITION_CLOSED",
        "POSITION_RECONCILE_FAILED",
        "RISK_LIMIT_BREACHED",
        "AUTHENTICATION_FAILED",
        "WEBSOCKET_DISCONNECTED",
    ):
        assert event_is_never_throttled(event) is True


def test_context_direction_logging_is_state_based() -> None:
    source = Path("src/nifty_scalper_bot/core/strategy_manager.py").read_text(encoding="utf-8")
    assert "CONTEXT_DIRECTION_STATE" in source
    assert "context_direction_state:{role}:{symbol}" in source
    assert "CONTEXT_DIRECTION_UNAVAILABLE role=%s symbol=%s" not in source


def test_throttle_summary_default_is_five_minutes() -> None:
    source = Path("src/nifty_scalper_bot/utils/log_throttle.py").read_text(encoding="utf-8")
    assert 'LOG_THROTTLE_SUMMARY_SECONDS", "300"' in source
