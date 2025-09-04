from __future__ import annotations

from types import SimpleNamespace

from src.config import settings
from src.notifications.telegram_controller import TelegramController


def _prep_settings(monkeypatch) -> None:
    monkeypatch.setattr(
        settings,
        "telegram",
        SimpleNamespace(bot_token="t", chat_id=1, enabled=True, extra_admin_ids=[]),
    )


def test_help_lists_all_commands(monkeypatch) -> None:
    _prep_settings(monkeypatch)
    tc = TelegramController(status_provider=lambda: {})
    sent: list[str] = []
    tc._send = lambda text, parse_mode=None: sent.append(text)
    cmds = tc._list_commands()
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/help"}})
    msg = sent[0]
    for cmd in cmds:
        assert cmd in msg
