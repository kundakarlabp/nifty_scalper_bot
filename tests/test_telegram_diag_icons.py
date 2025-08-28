from types import SimpleNamespace
from src.notifications.telegram_controller import TelegramController
from src.config import settings


def make_controller(monkeypatch, status_messages):
    monkeypatch.setattr(
        settings,
        "telegram",
        SimpleNamespace(bot_token="t", chat_id=1, enabled=True, extra_admin_ids=[]),
    )
    tc = TelegramController(
        status_provider=lambda: {},
        compact_diag_provider=lambda: {"ok": True, "status_messages": status_messages},
    )
    sent = []
    tc._send = lambda text, parse_mode=None: sent.append(text)
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/diag"}})
    return sent[0]


def test_risk_gates_skipped_green(monkeypatch):
    msg = make_controller(monkeypatch, {"risk_gates": "skipped"})
    assert "🟢 Risk gates" in msg


def test_broker_session_dry_mode_green(monkeypatch):
    msg = make_controller(monkeypatch, {"broker_session": "dry mode"})
    assert "🟢 Broker session" in msg
