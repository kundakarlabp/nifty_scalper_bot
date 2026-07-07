from __future__ import annotations

import logging

from nifty_scalper_bot.utils.alert_log_handler_hygiene import apply_patch, should_drop_alert_record
from nifty_scalper_bot.utils.alerts import AlertLogHandler


def _record(message: str, *, name: str = "telegram.ext.Updater", func: str = "default_error_callback") -> logging.LogRecord:
    return logging.LogRecord(
        name=name,
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
        func=func,
    )


def test_should_drop_generic_ptb_updater_record() -> None:
    record = _record("Exception happened while polling for updates.")

    assert should_drop_alert_record(record) is True


def test_should_not_drop_classified_or_unrelated_records() -> None:
    assert should_drop_alert_record(_record("TELEGRAM_POLLING_CONFLICT active", func="custom")) is False
    assert should_drop_alert_record(_record("Exception happened while polling for updates.", name="other")) is False


def test_alert_log_handler_drops_generic_ptb_record() -> None:
    emitted: list[dict[str, str]] = []
    apply_patch()
    handler = AlertLogHandler(emitted.append, repeat_window_seconds=0)

    handler.emit(_record("Exception happened while polling for updates."))

    assert emitted == []


def test_alert_log_handler_keeps_other_errors() -> None:
    emitted: list[dict[str, str]] = []
    apply_patch()
    handler = AlertLogHandler(emitted.append, repeat_window_seconds=0)

    handler.emit(_record("classified problem", func="custom_error_callback"))

    assert emitted
    assert emitted[0]["message"] == "classified problem"
