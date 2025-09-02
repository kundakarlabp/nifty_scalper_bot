from types import SimpleNamespace

from src.notifications.telegram_controller import TelegramController
from src.config import settings


def test_probe_reports_expected_fields(monkeypatch):
    monkeypatch.setattr(
        settings,
        "telegram",
        SimpleNamespace(bot_token="t", chat_id=1, enabled=True, extra_admin_ids=[]),
    )
    probe_info = {
        "bars": 10,
        "bar_age_s": 30,
        "tick_age_s": 5,
        "regime": "TREND",
        "atr_pct": 0.5,
        "score": 7,
    }
    tc = TelegramController(status_provider=lambda: {}, probe_provider=lambda: probe_info)
    sent = []
    tc._send = lambda text, parse_mode=None: sent.append(text)
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/probe"}})
    msg = sent[0]
    assert "bar_s=30" in msg
    assert "tick_s=5" in msg
    assert "regime=TREND" in msg
    assert "ATR%=0.5" in msg
    assert "score=7" in msg
