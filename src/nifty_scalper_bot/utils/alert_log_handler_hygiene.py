"""Alert-log hygiene for non-actionable third-party Telegram polling records."""

from __future__ import annotations

import logging
from typing import Any

_GENERIC_UPDATER_MESSAGE = "Exception happened while polling for updates"
_PATCHED = False


def should_drop_alert_record(record: logging.LogRecord) -> bool:
    """Return True for records that should not enter operator alert aggregation."""

    if record.name != "telegram.ext.Updater":
        return False
    if getattr(record, "funcName", "") != "default_error_callback":
        return False
    try:
        message = record.getMessage()
    except Exception:  # pragma: no cover - defensive logging boundary
        message = str(record.msg)
    return _GENERIC_UPDATER_MESSAGE in str(message or "")


def apply_patch() -> bool:
    """Patch AlertLogHandler.emit so Telegram alert aggregation drops generic PTB noise.

    The logger-level filter in notifications.__init__ is not sufficient when the
    root alert handler receives records directly through propagation. Patching the
    alert bridge itself makes suppression independent of logger configuration.
    """

    global _PATCHED
    if _PATCHED:
        return False
    try:
        from nifty_scalper_bot.utils.alerts import AlertLogHandler
    except Exception:
        return False
    if getattr(AlertLogHandler, "_telegram_generic_drop_patch", False):
        _PATCHED = True
        return False

    original_emit = AlertLogHandler.emit

    def emit(self: Any, record: logging.LogRecord) -> None:
        if should_drop_alert_record(record):
            return
        return original_emit(self, record)

    AlertLogHandler._telegram_generic_drop_original_emit = original_emit
    AlertLogHandler.emit = emit
    AlertLogHandler._telegram_generic_drop_patch = True
    _PATCHED = True
    return True


__all__ = ["apply_patch", "should_drop_alert_record"]
