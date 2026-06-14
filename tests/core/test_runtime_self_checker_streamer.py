"""Runtime self-check streamer connectivity tests."""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

from nifty_scalper_bot.core.app import RuntimeSelfChecker


class _TradingWindowDatetime(dt.datetime):
    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        return cls(2026, 1, 5, 10, 0, 0, tzinfo=tz)


class _DummyStreamer:
    def __init__(self, *, connected: bool, backlog: int, last_tick_ts: float) -> None:
        self._connected = connected
        self._backlog = backlog
        self.last_tick_ts = last_tick_ts

    def is_connected(self) -> bool:
        return self._connected

    def backlog_size(self) -> int:
        return self._backlog


def test_streamer_backlog_high_keeps_connected(monkeypatch) -> None:
    monkeypatch.setattr(dt, "datetime", _TradingWindowDatetime)
    now_ts = _TradingWindowDatetime.now(dt.timezone.utc).timestamp()
    ctx = SimpleNamespace(
        streamer=_DummyStreamer(connected=True, backlog=1500, last_tick_ts=now_ts),
    )
    checker = RuntimeSelfChecker(ctx)

    ok, detail, payload = checker._check_streamer()

    assert ok is True
    assert detail == "backlog_high"
    assert payload["connected"] is True


def test_streamer_no_recent_ticks_marks_disconnected(monkeypatch) -> None:
    monkeypatch.setattr(dt, "datetime", _TradingWindowDatetime)
    now_ts = _TradingWindowDatetime.now(dt.timezone.utc).timestamp()
    ctx = SimpleNamespace(
        streamer=_DummyStreamer(connected=True, backlog=10, last_tick_ts=now_ts - 30.0),
    )
    checker = RuntimeSelfChecker(ctx)

    ok, detail, payload = checker._check_streamer()

    assert ok is False
    assert detail == "no_recent_ticks"
    assert payload["connected"] is False


def test_canonical_history_ensurer_injection_failure_is_unhealthy() -> None:
    ctx = SimpleNamespace(
        canonical_history_ensurer_injection_failed=True,
        live_block_reason="canonical_history_ensurer_injection_failed",
        execution_block_reason="canonical_history_ensurer_injection_failed",
    )
    checker = RuntimeSelfChecker(ctx)

    ok, detail, payload = checker._check_canonical_history_ensurer()

    assert ok is False
    assert detail == "canonical_history_ensurer_injection_failed"
    assert payload["live_block_reason"] == "canonical_history_ensurer_injection_failed"
