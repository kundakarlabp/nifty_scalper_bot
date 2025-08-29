from types import SimpleNamespace

from src.notifications.telegram_controller import TelegramController
from src.config import settings

def test_backtest_command(monkeypatch):
    monkeypatch.setattr(
        settings,
        "telegram",
        SimpleNamespace(bot_token="t", chat_id=1, enabled=True, extra_admin_ids=[]),
    )
    called = {}
    def run_backtest(path=None):
        called['path'] = path
        return "ok"
    tc = TelegramController(status_provider=lambda: {}, backtest_provider=run_backtest)
    sent = []
    tc._send = lambda text, parse_mode=None: sent.append(text)
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/backtest foo.csv"}})
    assert sent[0] == "ok"
    assert called['path'] == "foo.csv"
