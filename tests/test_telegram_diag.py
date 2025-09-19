import json
from typing import Optional
from types import SimpleNamespace
from src.notifications.telegram_controller import TelegramController
from src.notifications import telegram_controller
from src.config import settings


def test_diag_shows_recent_trace_events(monkeypatch):
    monkeypatch.setattr(
        settings,
        "telegram",
        SimpleNamespace(bot_token="t", chat_id=1, enabled=True, extra_admin_ids=[]),
    )
    monkeypatch.setattr(settings, "diag_trace_events", 2, raising=False)
    records = [
        {"ts": "t1", "event": "a", "trace_id": "x", "msg": "one"},
        {"ts": "t2", "event": "b", "trace_id": "y", "msg": "two"},
        {"ts": "t3", "event": "c", "trace_id": "z", "msg": "three"},
    ]
    ring = SimpleNamespace(
        enabled=lambda: True,
        tail=lambda limit=None: list(records),
    )
    monkeypatch.setattr(telegram_controller.checks, "TRACE_RING", ring)
    tc = TelegramController(status_provider=lambda: {})
    sent: list[tuple[str, Optional[str]]] = []
    tc._send = lambda text, parse_mode=None: sent.append((text, parse_mode))
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/diag"}})
    assert sent, "Expected /diag to produce output"
    text, mode = sent[0]
    assert mode == "Markdown"
    assert text.startswith("```text\n") and text.endswith("\n```")
    body = text[len("```text\n") : -len("\n```")]
    lines = body.splitlines()
    assert len(lines) == 2
    assert "t2" in lines[0] and "t3" in lines[1]
    assert "trace=y" in lines[0]


def test_diagstatus_uses_compact_summary(monkeypatch):
    monkeypatch.setattr(
        settings,
        "telegram",
        SimpleNamespace(bot_token="t", chat_id=1, enabled=True, extra_admin_ids=[]),
    )
    summary = {
        "ok": True,
        "status_messages": {
            "risk_gates": "skipped",
            "data_feed": "ok",
        },
    }
    tc = TelegramController(
        status_provider=lambda: {},
        compact_diag_provider=lambda: summary,
    )
    sent: list[tuple[str, Optional[str]]] = []
    tc._send = lambda text, parse_mode=None: sent.append((text, parse_mode))
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/diagstatus"}})
    assert sent, "Expected /diagstatus to produce output"
    text, mode = sent[0]
    assert mode == "Markdown"
    assert text.startswith("```text\n")
    assert "overall: ok" in text
    assert "risk_gates: skipped" in text


def test_diagtrace_respects_limit_and_format(monkeypatch):
    monkeypatch.setattr(
        settings,
        "telegram",
        SimpleNamespace(bot_token="t", chat_id=1, enabled=True, extra_admin_ids=[]),
    )
    monkeypatch.setattr(settings, "diag_trace_events", 2, raising=False)
    records = [
        {"ts": "t1", "event": "a", "trace_id": "x", "msg": "one"},
        {"ts": "t2", "event": "b", "trace_id": "x", "msg": "two"},
        {"ts": "t3", "event": "c", "trace_id": "y", "msg": "three"},
    ]
    ring = SimpleNamespace(
        enabled=lambda: True,
        tail=lambda limit=None: list(records),
    )
    monkeypatch.setattr(telegram_controller.checks, "TRACE_RING", ring)
    tc = TelegramController(status_provider=lambda: {})
    sent: list[tuple[str, Optional[str]]] = []
    tc._send = lambda text, parse_mode=None: sent.append((text, parse_mode))
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/diagtrace"}})
    assert sent, "Expected /diagtrace to produce output"
    text, mode = sent[0]
    assert mode == "Markdown"
    assert text.startswith("```json\n") and text.endswith("\n```")
    payload = json.loads(text[len("```json\n") : -len("\n```")])
    assert isinstance(payload, list)
    assert len(payload) == 2
    assert payload[-1]["event"] == "c"


def test_trace_returns_events_for_trace_id(monkeypatch):
    monkeypatch.setattr(
        settings,
        "telegram",
        SimpleNamespace(bot_token="t", chat_id=1, enabled=True, extra_admin_ids=[]),
    )
    records = [
        {"ts": "t1", "event": "a", "trace_id": "x", "msg": "one"},
        {"ts": "t2", "event": "b", "trace_id": "x", "msg": "two"},
        {"ts": "t3", "event": "c", "trace_id": "y", "msg": "three"},
    ]
    ring = SimpleNamespace(
        enabled=lambda: True,
        tail=lambda limit=None: list(records),
    )
    monkeypatch.setattr(telegram_controller.checks, "TRACE_RING", ring)
    tc = TelegramController(status_provider=lambda: {})
    sent: list[tuple[str, Optional[str]]] = []
    tc._send = lambda text, parse_mode=None: sent.append((text, parse_mode))
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/trace x"}})
    assert sent, "Expected /trace to produce output"
    text, mode = sent[0]
    assert mode == "Markdown"
    payload = json.loads(text[len("```json\n") : -len("\n```")])
    assert isinstance(payload, list)
    assert len(payload) == 2
    assert all(item["trace_id"] == "x" for item in payload)
