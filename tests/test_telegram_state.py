from types import SimpleNamespace

from src.notifications.telegram_controller import TelegramController
from src.config import settings


def test_state_command(monkeypatch) -> None:
    monkeypatch.setattr(
        settings,
        "telegram",
        SimpleNamespace(bot_token="t", chat_id=1, enabled=True, extra_admin_ids=[]),
        raising=False,
    )
    status = {
        "equity": 1000,
        "trades_today": 2,
        "cooloff_until": "-",
        "day_realized_loss": -50,
        "eval_count": 7,
    }
    tc = TelegramController(status_provider=lambda: status)
    sent: list[str] = []
    tc._send = lambda text, parse_mode=None: sent.append(text)
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/state"}})
    assert "Equity: 1000" in sent[0]
    assert "Eval Count: 7" in sent[0]
