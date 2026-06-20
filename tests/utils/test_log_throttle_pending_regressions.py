from __future__ import annotations

import logging

from nifty_scalper_bot.utils.log_throttle import LogThrottle


def test_empty_suppression_summary_does_not_start_cooldown(caplog) -> None:
    throttle = LogThrottle()
    logger = logging.getLogger("tests.log_throttle.empty_summary")

    throttle.maybe_emit_summary(logger, interval_seconds=300)
    assert throttle._summary_last_emit_mono == 0.0

    throttle.record_suppressed("candidate:rejected")
    with caplog.at_level(logging.INFO):
        throttle.maybe_emit_summary(logger, interval_seconds=300)

    records = [
        record
        for record in caplog.records
        if getattr(record, "event", "") == "LOG_THROTTLE_SUMMARY"
    ]
    assert len(records) == 1
    assert records[0].total_suppressed == 1
    assert throttle._summary_last_emit_mono > 0.0


def test_empty_strategy_summary_does_not_start_cooldown(caplog) -> None:
    throttle = LogThrottle()
    logger = logging.getLogger("tests.log_throttle.empty_strategy_summary")

    assert not throttle.maybe_emit_strategy_rejection_summary(
        logger, interval_seconds=300
    )
    assert throttle._strategy_summary_last_emit_mono == 0.0

    throttle.record_strategy_evaluation(
        strategy="OrderFlow",
        symbol="NFO:NIFTY1CE",
        accepted=False,
        reason="final_trade_score_below_threshold",
        score=4.5,
    )
    with caplog.at_level(logging.INFO):
        assert throttle.maybe_emit_strategy_rejection_summary(
            logger, interval_seconds=300
        )

    records = [
        record
        for record in caplog.records
        if getattr(record, "event", "") == "STRATEGY_REJECTION_SUMMARY"
    ]
    assert len(records) == 1
    assert records[0].rejected_count == 1
    assert throttle._strategy_summary_last_emit_mono > 0.0
