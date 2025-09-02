from types import SimpleNamespace

from src.notifications.telegram_controller import TelegramController
from src.config import settings


def test_filecheck_uses_provider(monkeypatch):
    monkeypatch.setattr(
        settings,
        "telegram",
        SimpleNamespace(bot_token="t", chat_id=1, enabled=True, extra_admin_ids=[]),
    )

    def fake_provider(path: str) -> str:
        return f"checked {path}"

    tc = TelegramController(status_provider=lambda: {}, filecheck_provider=fake_provider)
    sent: list[str] = []
    tc._send = lambda text, parse_mode=None: sent.append(text)
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/filecheck foo.py"}})
    assert sent and sent[0] == "checked foo.py"
