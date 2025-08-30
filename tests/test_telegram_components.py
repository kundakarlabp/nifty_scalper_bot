from types import SimpleNamespace

from src.notifications.telegram_controller import TelegramController
from src.config import settings


def test_components_command(monkeypatch) -> None:
    monkeypatch.setattr(
        settings,
        "telegram",
        SimpleNamespace(bot_token="t", chat_id=1, enabled=True, extra_admin_ids=[]),
        raising=False,
    )
    status = {
        "components": {"strategy": "scalping", "data_provider": "auto", "order_connector": "kite"},
        "data_provider_health": {"status": "OK"},
        "order_connector_health": {"status": "OK"},
    }
    tc = TelegramController(status_provider=lambda: status)
    sent: list[str] = []
    tc._send = lambda text, parse_mode=None: sent.append(text)
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/components"}})
    assert "Components" in sent[0]
    assert "strategy" in sent[0].lower()
