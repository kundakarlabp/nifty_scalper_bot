from __future__ import annotations

import logging

from nifty_scalper_bot.core.status_log_hygiene import (
    CanonicalHistorySoftFailureFilter,
    apply_logging_filters,
    infer_reconciliation_complete_from_logs,
)


def test_expected_context_history_miss_is_downgraded() -> None:
    record = logging.LogRecord(
        name="nifty_scalper_bot.core.app",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="CANONICAL_HISTORY_RESULT role=option_context failure_reason=broker_fetch_not_allowed",
        args=(),
        exc_info=None,
    )

    assert CanonicalHistorySoftFailureFilter().filter(record) is True
    assert record.levelno == logging.INFO
    assert getattr(record, "gating") is False


def test_status_log_hygiene_filter_is_installable_on_runtime_loggers() -> None:
    apply_logging_filters()

    for logger_name in (
        "",
        "nifty_scalper_bot.core.app",
        "nifty_scalper_bot.core.history_readiness",
    ):
        logger = logging.getLogger(logger_name)
        assert any(isinstance(item, CanonicalHistorySoftFailureFilter) for item in logger.filters)


def test_reconciliation_complete_can_be_inferred_from_bounded_logs() -> None:
    text = "RECONCILE_START: checking broker\nRECONCILE_ORDERS: found 0\nRECONCILE_COMPLETE"

    assert infer_reconciliation_complete_from_logs(text) is True


def test_reconciliation_failure_after_start_is_not_complete() -> None:
    text = "RECONCILE_START: checking broker\nPOSITION_RECONCILE_FAILED\nRECONCILE_COMPLETE"

    assert infer_reconciliation_complete_from_logs(text) is False
