from types import SimpleNamespace

from src.notifications.telegram_controller import TelegramController
from src.config import settings
from src.diagnostics.registry import CheckResult


def test_selftest_runs_checks(monkeypatch):
    monkeypatch.setattr(
        settings,
        "telegram",
        SimpleNamespace(bot_token="t", chat_id=1, enabled=True, extra_admin_ids=[]),
    )

    fake_results = [CheckResult(name="config", ok=True, msg="ok", details={}, took_ms=5)]
    monkeypatch.setattr(
        "src.notifications.telegram_controller.run_all", lambda: fake_results
    )

    tc = TelegramController(status_provider=lambda: {})
    sent: list[str] = []
    tc._send = lambda text, parse_mode=None: sent.append(text)
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/selftest"}})
    assert sent and "config" in sent[0]
