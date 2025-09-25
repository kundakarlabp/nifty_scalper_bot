from __future__ import annotations

from typing import Any

import pytest

from src.data import source
from src.data.source import LiveKiteSource, QuoteState


class FakeClock:
    def __init__(self, *, wall: float = 1_000_000.0, mono: float = 5_000.0) -> None:
        self._wall = wall
        self._mono = mono

    def advance(self, seconds: float) -> None:
        self._wall += seconds
        self._mono += seconds

    def time(self) -> float:
        return self._wall

    def monotonic(self) -> float:
        return self._mono


def _setup_source(monkeypatch: pytest.MonkeyPatch, clock: FakeClock) -> LiveKiteSource:
    monkeypatch.setattr(source.time, "time", clock.time)
    monkeypatch.setattr(source.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(source.time, "sleep", lambda _s: None)
    monkeypatch.setattr(source.settings, "WATCHDOG_STALE_MS", 1_000, raising=False)
    monkeypatch.setattr(source.settings, "STALE_X_N", 3, raising=False)
    monkeypatch.setattr(source.settings, "STALE_WINDOW_MS", 60_000, raising=False)
    monkeypatch.setattr(source.settings, "RESUB_DEBOUNCE_MS", 20_000, raising=False)
    monkeypatch.setattr(source.settings, "RECONNECT_DEBOUNCE_MS", 30_000, raising=False)
    monkeypatch.setattr(source.settings, "MICRO__STALE_MS", 500, raising=False)

    src = LiveKiteSource(kite=None)
    token = 12345
    state = QuoteState(
        token=token,
        ts=clock.time() - 5.0,
        monotonic_ts=clock.monotonic() - 5.0,
        bid=100.0,
        ask=101.0,
        bid_qty=10,
        ask_qty=12,
        spread_pct=0.5,
        has_depth=True,
    )
    with src._tick_lock:  # noqa: SLF001
        src._quotes[token] = state
        src._last_tick_mono[token] = clock.monotonic() - 5.0
        src._last_any_tick_mono = clock.monotonic() - 5.0
    with src._ws_state_lock:  # noqa: SLF001
        src._subs.add(token)
        src._subscribed_tokens[token] = "FULL"
    monkeypatch.setattr(src, "_maybe_seed_via_rest", lambda *a, **k: None)
    return src


def test_stale_escalation_triggers_single_reconnect_and_resets(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock()
    src = _setup_source(monkeypatch, clock)
    token = next(iter(src._subs))  # noqa: SLF001

    reconnect_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(src, "reconnect_ws", lambda **kwargs: reconnect_calls.append(kwargs))

    events: list[tuple[str, dict[str, Any]]] = []

    def _log(event: str, **payload: Any) -> None:
        events.append((event, payload))

    monkeypatch.setattr(source.structured_log, "event", _log)

    monkeypatch.setattr(src, "resubscribe_current", lambda: None)

    triggered_flags: list[bool] = []
    for idx in range(3):
        fired = src._record_ws_stale_event(  # noqa: SLF001 - exercising internals for watchdog logic
            token, age_ms=5_000, threshold_ms=500
        )
        triggered_flags.append(fired)
        if idx < 2:
            clock.advance(25.0)

    assert triggered_flags.count(True) == 1
    assert triggered_flags[-1] is True
    assert len(reconnect_calls) == 1
    escalate = [payload for name, payload in events if name == "ws_reconnect_escalate"]
    assert escalate and token in escalate[0]["tokens"]
    assert all("ws_mid" not in name for name, _payload in events)

    # Simulate a fresh websocket tick to reset counters.
    clock.advance(1.0)
    tick = {
        "instrument_token": token,
        "last_price": 102.0,
        "depth": {
            "buy": [{"price": 101.5, "quantity": 10}],
            "sell": [{"price": 102.5, "quantity": 10}],
        },
    }
    src._on_tick(tick)  # noqa: SLF001

    diag = src.ws_diag_snapshot()
    assert diag["stale_counts"] == {}
    assert diag["last_tick_age_ms"] is not None and diag["last_tick_age_ms"] <= 5


def test_reconnect_debounce_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock()
    src = _setup_source(monkeypatch, clock)
    token = next(iter(src._subs))  # noqa: SLF001

    monkeypatch.setattr(src, "resubscribe_current", lambda: None)

    reconnect_calls: list[int] = []
    monkeypatch.setattr(
        src,
        "reconnect_ws",
        lambda **kwargs: reconnect_calls.append(int(clock.monotonic() * 1000)),
    )

    for _ in range(3):
        src.resubscribe_if_stale(token)
        clock.advance(10.0)
    assert len(reconnect_calls) == 1

    # Second attempt within debounce window should not trigger another reconnect.
    clock.advance(2.0)
    src.resubscribe_if_stale(token)
    assert len(reconnect_calls) == 1

    # Once debounce elapsed, reconnect can happen again.
    clock.advance(20.0)
    src.resubscribe_if_stale(token)
    assert len(reconnect_calls) == 2


def test_force_hard_reconnect_resets_watchdog_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    src = _setup_source(monkeypatch, clock)
    token = next(iter(src._subs))  # noqa: SLF001

    # Seed stale counters.
    fired = src._record_ws_stale_event(  # noqa: SLF001 - exercising watchdog internals
        token,
        age_ms=5_000,
        threshold_ms=500,
    )
    assert fired is False  # first escalation requires multiple events
    assert src._watchdog_stale_count > 0  # noqa: SLF001
    assert src._token_stale_events  # noqa: SLF001

    reconnect_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        src,
        "reconnect_with_backoff",
        lambda **kwargs: reconnect_calls.append(kwargs),
    )

    src.force_hard_reconnect(reason="unit", context={"via": "test"})

    assert reconnect_calls and reconnect_calls[-1]["reason"] == "unit"
    assert reconnect_calls[-1]["context"] == {"via": "test"}
    assert src._watchdog_stale_count == 0  # noqa: SLF001
    assert src._watchdog_first_stale_mono is None  # noqa: SLF001
    assert src._watchdog_last_resub_mono is None  # noqa: SLF001
    assert src._token_stale_events == {}  # noqa: SLF001
