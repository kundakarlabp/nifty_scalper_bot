from __future__ import annotations

import logging

from nifty_scalper_bot.utils.logging import DedupLogger


def test_dedup_logger_suppresses_ordinary_duplicate_info(caplog) -> None:
    logger = logging.getLogger("tests.dedup_logger.ordinary")
    dedup = DedupLogger(logger, cooldown_seconds=60.0)

    with caplog.at_level(logging.INFO, logger=logger.name):
        dedup.info("duplicate audit message")
        dedup.info("duplicate audit message")

    assert caplog.text.count("duplicate audit message") == 1


def test_dedup_logger_bypass_filters_keeps_critical_info(caplog) -> None:
    logger = logging.getLogger("tests.dedup_logger.bypass")
    dedup = DedupLogger(logger, cooldown_seconds=60.0)

    with caplog.at_level(logging.INFO, logger=logger.name):
        dedup.info(
            "critical lifecycle event",
            extra={"event": "CRITICAL_LIFECYCLE", "bypass_filters": True},
        )
        dedup.info(
            "critical lifecycle event",
            extra={"event": "CRITICAL_LIFECYCLE", "bypass_filters": True},
        )

    assert caplog.text.count("critical lifecycle event") == 2
