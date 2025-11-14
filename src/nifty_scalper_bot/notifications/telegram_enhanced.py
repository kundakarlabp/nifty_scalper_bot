"""Compatibility shim re-exporting the webhook-first notifier."""

from __future__ import annotations

from nifty_scalper_bot.notifications.telegram_webhook_enhanced import (
    TelegramEnhancedNotifier,
)

__all__ = ["TelegramEnhancedNotifier"]
