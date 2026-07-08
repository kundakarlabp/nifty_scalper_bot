"""Notification helpers exposed to the rest of the application."""

from __future__ import annotations

import importlib
import logging
from typing import Any, Mapping


class _TelegramUpdaterDefaultErrorFilter(logging.Filter):
    _GENERIC = "while polling for updates"

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


def _install_alert_log_hygiene() -> None:
    try:
        from nifty_scalper_bot.utils.alert_log_handler_hygiene import apply_patch

        apply_patch()
    except Exception:
        return


def _install_operator_command_aliases() -> None:
    try:
        _operator = importlib.import_module("nifty_scalper_bot.notifications.operator_telegram")
        existing = {spec.name for spec in _operator.OPERATOR_COMMANDS}
        if "flat" not in existing:
            flat_spec = _operator.CommandSpec(
                "flat",
                "alias for confirmed flatten of bot-owned open positions",
                _operator.cmd_flatten,
                "Control",
                "confirmed-destructive",
            )
            names = [spec.name for spec in _operator.OPERATOR_COMMANDS]
            try:
                insert_at = names.index("flatten") + 1
            except ValueError:
                insert_at = len(_operator.OPERATOR_COMMANDS)
            _operator.OPERATOR_COMMANDS.insert(insert_at, flat_spec)
            _operator.OPERATOR_COMMAND_NAMES = tuple(spec.name for spec in _operator.OPERATOR_COMMANDS)
    except Exception:
        return


def _install_operator_execution_hygiene() -> None:
    try:
        _operator = importlib.import_module("nifty_scalper_bot.notifications.operator_telegram")
        if getattr(_operator, "_execution_hygiene_patch", False):
            return
        original = _operator._execution_items

        def _patched_execution_items(snap: Mapping[str, Any]) -> dict[str, Any]:
            items = dict(original(snap))
            raw_execution_reason = snap.get("execution_block_reason")
            items["execution_block_reason"] = "None" if raw_execution_reason is None else raw_execution_reason
            last_order_rejection = snap.get("last_order_rejection")
            has_current_blocker = bool(
                snap.get("live_block_reason")
                or raw_execution_reason
                or snap.get("current_execution_blocker")
                or snap.get("current_pnl_reconciliation_blocker")
                or snap.get("current_risk_breaker_reason")
            )
            if last_order_rejection and not has_current_blocker:
                items["stale_last_skip_reason"] = last_order_rejection
            return items

        _operator._execution_items = _patched_execution_items
        _operator._execution_hygiene_patch = True
    except Exception:
        return


_install_telegram_log_filters()
_install_alert_log_hygiene()
_install_operator_command_aliases()
_install_operator_execution_hygiene()

__all__ = [
    "TelegramEnhancedNotifier",
    "TelegramWebhookController",
    "register_webhook",
    "start_fallback_polling",
]


def __getattr__(name: str):
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
