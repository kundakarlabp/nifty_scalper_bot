from __future__ import annotations

import logging

from nifty_scalper_bot.utils.log_throttle import (
    LogThrottle,
    log_on_change,
    log_throttled,
    maybe_emit_strategy_rejection_summary,
    record_strategy_evaluation,
)
from nifty_scalper_bot.utils.logging import setup_logging


def test_same_state_logs_once_then_reminder_only(caplog):
    throttle = LogThrottle()
    logger = logging.getLogger("tests.change.once")
    with caplog.at_level(logging.INFO):
        assert log_on_change(logger, key="hydration", state="READY", message="HYDRATION READY", reminder_seconds=600, throttle=throttle)
        assert not log_on_change(logger, key="hydration", state="READY", message="HYDRATION READY", reminder_seconds=600, throttle=throttle)
        assert log_on_change(logger, key="hydration", state="READY", message="HYDRATION READY", reminder_seconds=0, throttle=throttle)
    records = [r for r in caplog.records if r.message == "HYDRATION READY"]
    assert len(records) == 2
    assert getattr(records[-1], "suppressed_count") == 1


def test_changed_state_logs_immediately(caplog):
    throttle = LogThrottle()
    logger = logging.getLogger("tests.change.changed")
    with caplog.at_level(logging.INFO):
        assert log_on_change(logger, key="ws", state="DISCONNECTED", message="WS DISCONNECTED", throttle=throttle)
        assert log_on_change(logger, key="ws", state="CONNECTED", message="WS CONNECTED", throttle=throttle)
    assert [r.message for r in caplog.records] == ["WS DISCONNECTED", "WS CONNECTED"]


def test_critical_events_are_never_throttled(caplog):
    throttle = LogThrottle()
    logger = logging.getLogger("tests.critical.never")
    with caplog.at_level(logging.INFO):
        assert log_throttled(logger, logging.INFO, "ORDER_SUBMITTED", "order:1", 300, "ORDER_SUBMITTED", throttle=throttle)
        assert log_throttled(logger, logging.INFO, "ORDER_SUBMITTED", "order:1", 300, "ORDER_SUBMITTED", throttle=throttle)
    assert sum(1 for r in caplog.records if r.message == "ORDER_SUBMITTED") == 2


def test_strategy_rejection_aggregation_preserves_distribution_and_scores(caplog):
    throttle = LogThrottle()
    logger = logging.getLogger("tests.strategy.summary")
    record_strategy_evaluation(strategy="s1", symbol="NIFTY1CE", accepted=False, reason="low_score", score=40, throttle=throttle)
    record_strategy_evaluation(strategy="s1", symbol="NIFTY1CE", accepted=False, reason="low_score", score=42, throttle=throttle)
    record_strategy_evaluation(strategy="s2", symbol="NIFTY1PE", accepted=False, reason="spread_too_wide", score=35, throttle=throttle)
    record_strategy_evaluation(strategy="s2", symbol="NIFTY1PE", accepted=True, throttle=throttle)
    with caplog.at_level(logging.INFO):
        assert maybe_emit_strategy_rejection_summary(logger, interval_seconds=0, throttle=throttle)
    rec = next(r for r in caplog.records if getattr(r, "event", "") == "STRATEGY_REJECTION_SUMMARY")
    assert rec.evaluation_count == 4
    assert rec.accepted_count == 1
    assert rec.rejected_count == 3
    assert rec.top_reasons["low_score"] == 2
    assert rec.first_score == 40
    assert rec.latest_score == 35
    assert rec.min_score == 35
    assert rec.max_score == 42


def test_accepted_signal_log_is_not_throttled_when_interval_zero(caplog):
    throttle = LogThrottle()
    logger = logging.getLogger("tests.accepted.signal")
    with caplog.at_level(logging.INFO):
        assert log_throttled(logger, logging.INFO, "SIGNAL_ACCEPTED", "accepted:s1:sym", 0, "ACCEPTED_SIGNAL", throttle=throttle)
        assert log_throttled(logger, logging.INFO, "SIGNAL_ACCEPTED", "accepted:s1:sym", 0, "ACCEPTED_SIGNAL", throttle=throttle)
    assert sum(1 for r in caplog.records if r.message == "ACCEPTED_SIGNAL") == 2


def test_hydration_ready_once_degraded_and_recovered_immediate(caplog):
    throttle = LogThrottle()
    logger = logging.getLogger("tests.hydration.states")
    with caplog.at_level(logging.INFO):
        assert log_on_change(logger, key="hydration:ce", state="READY", message="HYDRATION READY", throttle=throttle)
        assert not log_on_change(logger, key="hydration:ce", state="READY", message="HYDRATION READY", throttle=throttle)
        assert log_on_change(logger, key="hydration:ce", state="DEGRADED", message="HYDRATION DEGRADED", throttle=throttle)
        assert log_on_change(logger, key="hydration:ce", state="RECOVERED", message="HYDRATION RECOVERED", throttle=throttle)
    assert [r.message for r in caplog.records] == ["HYDRATION READY", "HYDRATION DEGRADED", "HYDRATION RECOVERED"]


def test_separate_strategy_execution_file_handlers_no_duplicate_records(tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    monkeypatch.setenv("LOG_DEDUP_ENABLED", "false")
    monkeypatch.setenv("LOG_RATE_LIMIT_SEC", "0")
    setup_logging("INFO")
    logging.getLogger("nifty_scalper_bot.strategies.runner").info("strategy-only", extra={"event": "test_strategy"})
    logging.getLogger("nifty_scalper_bot.execution.order_manager").info("execution-only", extra={"event": "test_execution"})
    logging.getLogger("nifty_scalper_bot.execution.order_manager").error("execution-critical", extra={"event": "ORDER_REJECTED"})
    for handler in logging.getLogger().handlers:
        handler.flush()
    strategy_log = (tmp_path / "strategy.log").read_text()
    execution_log = (tmp_path / "execution.log").read_text()
    bot_log = (tmp_path / "bot.log").read_text()
    assert strategy_log.count("strategy-only") == 1
    assert execution_log.count("execution-only") == 1
    assert "strategy-only" not in bot_log
    assert "execution-only" not in bot_log
    assert bot_log.count("execution-critical") == 1


def test_malformed_logging_does_not_raise(caplog):
    throttle = LogThrottle()
    logger = logging.getLogger("tests.malformed.safe")
    with caplog.at_level(logging.INFO):
        assert log_throttled(logger, logging.INFO, "EV", "badfmt", 0, "bad %s %s", "one", throttle=throttle)


def test_runner_repeated_signal_rejection_is_throttled(caplog):
    from nifty_scalper_bot.strategies.runner import StrategyRunner
    from nifty_scalper_bot.utils.log_throttle import DEFAULT_LOG_THROTTLE

    with DEFAULT_LOG_THROTTLE._lock:
        DEFAULT_LOG_THROTTLE._states.clear()
        DEFAULT_LOG_THROTTLE._suppressed.clear()
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._logger = logging.getLogger("nifty_scalper_bot.strategies.runner.test")
    runner._record_trade_decision_snapshot = lambda **_kwargs: None
    with caplog.at_level(logging.INFO):
        first = runner._reject_signal_execution(symbol="NFO:NIFTY1CE", trace_id="t1", reason="strategy_score_below_threshold", details={"strategy": "s1"})
        second = runner._reject_signal_execution(symbol="NFO:NIFTY1CE", trace_id="t2", reason="strategy_score_below_threshold", details={"strategy": "s1"})
    assert first.accepted is False
    assert second.accepted is False
    records = [r for r in caplog.records if getattr(r, "event", "") == "SIGNAL_EXECUTION_RESULT"]
    assert len(records) == 1


def test_order_manager_kill_switch_status_logs_on_change_only(caplog):
    from nifty_scalper_bot.execution.order_manager import OrderManager
    from nifty_scalper_bot.utils.log_throttle import DEFAULT_LOG_THROTTLE

    with DEFAULT_LOG_THROTTLE._lock:
        DEFAULT_LOG_THROTTLE._change_states.clear()
    manager = OrderManager.__new__(OrderManager)
    manager._logger = logging.getLogger("nifty_scalper_bot.execution.order_manager.test")
    manager._kill_switch_engaged_at = None
    manager._kill_switch_allow_auto_reset = False
    manager._kill_switch_auto_reset_seconds = 900
    manager._kill_switch_reason = None
    manager._consecutive_failures = 0
    manager._kill_switch_failure_history = []
    manager._kill_switch_last_reset = None
    with caplog.at_level(logging.INFO):
        manager._log_kill_switch_status()
        manager._log_kill_switch_status()
    records = [r for r in caplog.records if getattr(r, "event", "") == "ORDER_KILL_SWITCH_STATUS"]
    assert len(records) == 1


def test_non_allowlisted_order_prefix_event_is_throttleable(caplog):
    throttle = LogThrottle()
    logger = logging.getLogger("tests.order.prefix.throttleable")
    with caplog.at_level(logging.INFO):
        assert log_throttled(logger, logging.INFO, "ORDER_KILL_SWITCH_BLOCK", "ks:block", 300, "blocked", throttle=throttle)
        assert not log_throttled(logger, logging.INFO, "ORDER_KILL_SWITCH_BLOCK", "ks:block", 300, "blocked", throttle=throttle)
    assert sum(1 for r in caplog.records if r.message == "blocked") == 1


def test_logging_wrapper_invalid_extra_does_not_raise() -> None:
    from nifty_scalper_bot.utils.logging import log_state_change, log_throttled as compat_log_throttled

    logger = logging.getLogger("tests.invalid.extra.safe")
    compat_log_throttled(logger, "bad-extra", "bad extra survives", interval_sec=0, extra={"message": "reserved"})
    assert log_state_change(logger, "bad-change-extra", "A", msg="bad change survives", extra={"message": "reserved"}) is False


def test_canonical_helpers_malformed_logger_does_not_raise() -> None:
    class BrokenLogger:
        def log(self, *_args, **_kwargs):
            raise RuntimeError("logger failed")

        def info(self, *_args, **_kwargs):
            raise RuntimeError("logger failed")

    broken = BrokenLogger()
    assert not log_throttled(broken, logging.INFO, "EV", "broken", 0, "msg")  # type: ignore[arg-type]
    assert not log_on_change(broken, key="broken-change", state="A", message="msg")  # type: ignore[arg-type]
