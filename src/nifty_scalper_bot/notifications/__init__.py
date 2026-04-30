"""Notification helpers exposed to the rest of the application."""

from __future__ import annotations

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
