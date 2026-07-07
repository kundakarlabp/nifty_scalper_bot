"""Notification helpers exposed to the rest of the application."""

from __future__ import annotations

import logging


class _TelegramUpdaterDefaultErrorFilter(logging.Filter):
    _GENERIC = "Exception happened while polling for updates"

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == "telegram.ext.Updater" and record.funcName == "default_error_callback":
            try:
                return self._GENERIC not in record.getMessage()
            except Exception:
                return False
        return True


def _install_telegram_log_filters() -> None:
    logger = logging.getLogger("telegram.ext.Updater")
    if any(isinstance(filter_, _TelegramUpdaterDefaultErrorFilter) for filter_ in logger.filters):
        return
    logger.addFilter(_TelegramUpdaterDefaultErrorFilter())


_install_telegram_log_filters()

__all__ = [
    "TelegramEnhancedNotifier",
    "TelegramWebhookController",
    "register_webhook",
    "start_fallback_polling",
]


def __getattr__(name: str):
    """Lazily load optional Telegram notification components. Args: name. Returns: object. Raises: AttributeError."""
    if name == "TelegramEnhancedNotifier":
        from .telegram_enhanced import TelegramEnhancedNotifier

        return TelegramEnhancedNotifier

    if name in {
        "TelegramWebhookController",
        "register_webhook",
        "start_fallback_polling",
    }:
        from .telegram_webhook_enhanced import (
            TelegramWebhookController,
            register_webhook,
            start_fallback_polling,
        )

        return {
            "TelegramWebhookController": TelegramWebhookController,
            "register_webhook": register_webhook,
            "start_fallback_polling": start_fallback_polling,
        }[name]

    raise AttributeError(name)
