"""Patch Telegram polling to classify updater transport errors.

This module patches TelegramBot.start without changing trading code. The legacy
start path added an Application error handler, but PTB polling transport errors
come from Updater and use PTB's default error callback unless start_polling gets
an explicit callback.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from telegram.error import NetworkError, TimedOut

from nifty_scalper_bot.notifications.telegram_runtime_registry import claim_polling_owner, release_polling_owner
from nifty_scalper_bot.utils.logging import get_logger

LOG = get_logger(__name__)
_PATCHED = False


def _classify_polling_error(bot: Any, exc: object) -> None:
    text = str(exc or "")
    lowered = text.lower()
    if "conflict" in lowered or "terminated by other getupdates" in lowered:
        bot._polling_conflict_count = getattr(bot, "_polling_conflict_count", 0) + 1
        LOG.error(
            "TELEGRAM_POLLING_CONFLICT another poller is active; inbound commands may not arrive. count=%s err=%s",
            bot._polling_conflict_count,
            text,
            extra={"event": "TELEGRAM_POLLING_CONFLICT", "count": bot._polling_conflict_count},
        )
        return
    if "timed out" in lowered or "timeout" in lowered or isinstance(exc, NetworkError):
        LOG.warning(
            "telegram polling transient error: %s",
            text,
            extra={"event": "telegram_polling_transient", "error_type": type(exc).__name__},
        )
        return
    LOG.error(
        "TELEGRAM_UPDATER_ERROR type=%s err=%s",
        type(exc).__name__ if exc else "None",
        text,
        extra={"event": "TELEGRAM_UPDATER_ERROR", "error_type": type(exc).__name__ if exc else "None"},
    )


def apply_patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return False
    try:
        from nifty_scalper_bot.notifications import telegram_controller as module
    except Exception:
        return False

    cls = module.TelegramBot
    if getattr(cls, "_polling_callback_patch_installed", False):
        _PATCHED = True
        return False

    async def start(self: Any) -> None:
        if self._started:
            module.log.debug("Telegram bot start skipped: already started")
            return
        self._started = True
        self._running = True
        try:
            if not claim_polling_owner(token=self.deps.token, owner=type(self).__name__):
                self._started = False
                self._running = False
                return
            module.log.info("Telegram bot startup initiated", extra={"event": "telegram_start_enter"})
            self._register_handlers()
            try:
                self.application.add_error_handler(self._on_ptb_error)
            except Exception as exc:  # noqa: BLE001
                module.log.debug("telegram add_error_handler failed: %s", exc)
            command_count = len(module.registered_command_names(self.application))
            module.log.info(
                "TELEGRAM_RUNTIME_CONFIG bot_token_present=%s chat_id_configured=%s command_count=%s polling_mode=%s",
                bool(self.deps.token),
                self.deps.chat_id is not None,
                command_count,
                self.mode == "polling",
            )
            await self.application.initialize()
            await self.application.start()
            if self.application.updater is not None:
                module.log.info("TELEGRAM_POLLING_START_REQUESTED")
                try:
                    await self.application.bot.delete_webhook(drop_pending_updates=True)
                except Exception as exc:  # noqa: BLE001
                    module.log.warning(
                        "telegram_delete_webhook_failed err=%s",
                        exc,
                        extra={"event": "telegram_delete_webhook_failed"},
                    )
                await self.application.updater.start_polling(
                    drop_pending_updates=True,
                    error_callback=lambda exc: _classify_polling_error(self, exc),
                )
                module.log.info("TELEGRAM_POLLING_STARTED")
            await self._safe_send("🤖 Telegram bot started")
            module.log.info("Telegram bot started")
            self._ensure_alert_worker()
            if not self._bg_task_started:
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(), name="telegram-heartbeat-loop")
                self._bg_task_started = True
                module.log.info("telegram_heartbeat_task_started")
        except Exception as exc:  # noqa: BLE001
            release_polling_owner(token=self.deps.token, owner=type(self).__name__)
            self._started = False
            self._running = False
            text = str(exc).lower()
            transient = (
                isinstance(exc, (NetworkError, TimedOut))
                or "timed out" in text
                or "timeout" in text
                or "connection" in text
            )
            if transient:
                attempt = getattr(self, "_start_retry_attempt", 0) + 1
                self._start_retry_attempt = attempt
                max_retries = int(os.getenv("TELEGRAM_START_MAX_RETRIES", "5") or "5")
                if attempt <= max_retries:
                    delay = min(60.0, 5.0 * attempt)
                    module.log.warning(
                        "Telegram bot start transient failure (%s); retrying in %ss (attempt %s/%s)",
                        type(exc).__name__,
                        delay,
                        attempt,
                        max_retries,
                        extra={"event": "telegram_start_retry", "attempt": attempt},
                    )

                    async def _retry_start() -> None:
                        await asyncio.sleep(delay)
                        await self.start()

                    try:
                        asyncio.create_task(_retry_start(), name="telegram-start-retry")
                    except Exception as task_exc:  # pragma: no cover
                        module.log.debug("telegram start retry schedule failed: %s", task_exc)
                    return
            self._start_retry_attempt = 0
            module.log.error(
                "Telegram bot start failed type=%s err=%s",
                type(exc).__name__,
                exc,
                exc_info=True,
                extra={"event": "telegram_start_failed"},
            )

    cls._polling_callback_patch_original_start = cls.start
    cls.start = start
    cls._polling_callback_patch_installed = True
    _PATCHED = True
    return True


__all__ = ["apply_patch", "_classify_polling_error"]
