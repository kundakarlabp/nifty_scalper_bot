from types import SimpleNamespace

from src.notifications.telegram_controller import TelegramController
from src.config import settings


def test_microcap_sets_spread(monkeypatch) -> None:
    monkeypatch.setattr(
        settings,
        "telegram",
        SimpleNamespace(bot_token="t", chat_id=1, enabled=True, extra_admin_ids=[]),
        raising=False,
    )
    tc = TelegramController(status_provider=lambda: {})
    sent: list[str] = []
    tc._send = lambda text, parse_mode=None: sent.append(text)
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/microcap 1.0"}})
    assert abs(settings.executor.max_spread_pct - 0.01) < 1e-9
    assert "1.00%" in sent[0]
