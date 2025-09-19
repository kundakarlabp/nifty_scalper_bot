from types import SimpleNamespace

from src.config import settings
from src.diagnostics import checks
from src.notifications.telegram_controller import TelegramController


def _prep(monkeypatch) -> None:
    monkeypatch.setattr(
        settings,
        "telegram",
        SimpleNamespace(bot_token="t", chat_id=1, enabled=True, extra_admin_ids=[]),
    )
    monkeypatch.setattr(settings, "diag_trace_events", True, raising=False)
    checks.clear_traces()


def test_diag_shows_trace_tail(monkeypatch):
    _prep(monkeypatch)
    checks.record_trace_event({"trace_id": "abc", "event": "tick", "score": 1})
    checks.record_trace_event({"trace_id": "def", "event": "tick", "score": 2})

    tc = TelegramController(status_provider=lambda: {})
    sent: list[str] = []
    tc._send = lambda text, parse_mode=None: sent.append(text)
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/diag 2"}})
    assert sent, "Expected /diag to produce output"
    msg = sent[0]
    assert "abc" in msg and "def" in msg
    assert msg.startswith("```json")
    checks.clear_traces()


def test_trace_lookup_by_id(monkeypatch):
    _prep(monkeypatch)
    checks.record_trace_event({"trace_id": "trace-1", "event": "tick", "score": 3})
    checks.record_trace_event({"trace_id": "trace-2", "event": "tick", "score": 4})
    checks.record_trace_event({"trace_id": "trace-1", "event": "done", "score": 5})

    tc = TelegramController(status_provider=lambda: {})
    sent: list[str] = []
    tc._send = lambda text, parse_mode=None: sent.append(text)
    tc._handle_update(
        {"message": {"chat": {"id": 1}, "text": "/trace trace-1"}}
    )
    assert sent, "Expected /trace to respond"
    msg = sent[0]
    assert "trace-1" in msg and "trace-2" not in msg
    assert msg.startswith("```json")
    checks.clear_traces()
