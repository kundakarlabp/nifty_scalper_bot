from __future__ import annotations

import pytest
from telegram.ext import CommandHandler

from nifty_scalper_bot.notifications.telegram_controller import (
    TelegramBot,
    TelegramDeps,
)


class FakeApp:
    def __init__(self) -> None:
        self.handlers: dict[int, list[object]] = {0: []}

    def add_handler(self, handler: object, group: int = 0) -> None:
        self.handlers.setdefault(group, []).append(handler)


@pytest.fixture()
def telegram_bot() -> TelegramBot:
    deps = TelegramDeps(token="dummy", chat_id=12345, app_version="test")
    return TelegramBot(deps)


def test_safe_command_registration(telegram_bot: TelegramBot) -> None:
    app = FakeApp()
    preexisting = CommandHandler("status", lambda *_: None)
    app.add_handler(preexisting)

    telegram_bot._wire_handlers(app)  # type: ignore[arg-type]

    status_handlers = [
        handler
        for handler in app.handlers.get(0, [])
        if isinstance(handler, CommandHandler)
        and "status" in getattr(handler, "commands", set())
    ]
    assert len(status_handlers) == 1
