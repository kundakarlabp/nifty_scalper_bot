"""Telegram bot setup and dispatcher."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ...config.settings import TelegramSettings

log = logging.getLogger(__name__)


class TelegramBot:
    """Wraps python-telegram-bot v21 for async operation."""

    def __init__(self, settings: "TelegramSettings") -> None:
        self._settings = settings
        self._app = None
        self._chat_id = settings.chat_id

    async def setup(self) -> bool:
        if not self._settings.is_configured:
            log.info("Telegram not configured — notifications disabled")
            return False
        try:
            from telegram.ext import Application  # type: ignore[import]
            token = self._settings.bot_token.get_secret_value()  # type: ignore[union-attr]
            self._app = Application.builder().token(token).build()
            self._register_commands()
            log.info("Telegram bot configured for chat_id=%s", self._chat_id)
            return True
        except Exception as e:
            log.error("Telegram setup failed: %s", e)
            return False

    def _register_commands(self) -> None:
        if not self._app:
            return
        # Commands registered separately in command modules
        pass

    async def start(self) -> None:
        if self._app and self._settings.use_polling:
            await self._app.initialize()
            await self._app.start()
            await self._app.updater.start_polling()  # type: ignore[union-attr]

    async def stop(self) -> None:
        if self._app:
            try:
                await self._app.updater.stop()  # type: ignore[union-attr]
                await self._app.stop()
                await self._app.shutdown()
            except Exception as e:
                log.warning("Telegram stop: %s", e)

    async def send_message(self, text: str, parse_mode: str = "Markdown") -> None:
        if not self._app or not self._chat_id:
            return
        try:
            await self._app.bot.send_message(
                chat_id=self._chat_id,
                text=text,
                parse_mode=parse_mode,
            )
        except Exception as e:
            log.error("Telegram send_message failed: %s", e)
