from types import SimpleNamespace

import json
from typing import Optional
from types import SimpleNamespace

from src.notifications.telegram_controller import TelegramController
from src.notifications import telegram_controller
from src.config import settings


def test_diag_shows_status_and_signal(monkeypatch):
    monkeypatch.setattr(
        settings,
        "telegram",
        SimpleNamespace(bot_token="t", chat_id=1, enabled=True, extra_admin_ids=[]),
    )
    status = {
        "market_open": True,
        "within_window": True,
        "daily_dd_hit": False,
        "cooloff_until": "-",
        "trades_today": 1,
        "consecutive_losses": 0,
    }
    plan = {
        "action": "BUY",
        "option_type": "CE",
        "strike": "100",
        "qty_lots": 1,
        "expiry": "2024-08-06",
        "regime": "TREND",
        "score": 5,
        "rr": 2.0,
        "atr_pct": 0.5,
        "micro": {"spread_pct": 0.1, "depth_ok": True},
        "entry": 100.0,
        "sl": 95.0,
        "tp1": 102.0,
        "tp2": 104.0,
        "reason_block": None,
        "reasons": ["test"],
        "ts": "2024-01-01T00:00:00",
    }
    tc = TelegramController(
        status_provider=lambda: status,
        last_signal_provider=lambda: plan,
    )
    sent = []
    tc._send = lambda text, parse_mode=None: sent.append(text)
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/diag"}})
    msg = sent[0]
    assert "📊 Status" in msg and "📈 Signal" in msg
    assert "Market Open: ✅" in msg
    assert "Action: BUY" in msg
    assert "Expiry: 2024-08-06" in msg


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
    monkeypatch.setattr(telegram_controller.ringlog, "enabled", lambda: True)
    monkeypatch.setattr(
        telegram_controller.ringlog, "tail", lambda limit=None: list(records)
    )
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
