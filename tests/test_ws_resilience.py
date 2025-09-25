from __future__ import annotations

import time
from typing import Any

import pytest

from src.data import source
from src.data.source import LiveKiteSource, QuoteState


@pytest.fixture(autouse=True)
def _reset_log(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(source.settings, "MICRO__STALE_MS", 50, raising=False)
    monkeypatch.setattr(source.settings, "STALE_X_N", 3, raising=False)
    monkeypatch.setattr(source.settings, "STALE_WINDOW_MS", 60_000, raising=False)
    monkeypatch.setattr(source.settings, "RECONNECT_DEBOUNCE_MS", 10_000, raising=False)
    monkeypatch.setattr(source.settings, "WATCHDOG_STALE_MS", 3_500, raising=False)


def _install_time(monkeypatch: pytest.MonkeyPatch) -> dict[str, float]:
    clock = {"mono": 1_000.0, "wall": 10_000.0}

    monkeypatch.setattr(source.time, "monotonic", lambda: clock["mono"])
    monkeypatch.setattr(source.time, "time", lambda: clock["wall"])
    monkeypatch.setattr(source, "monotonic_ms", lambda: int(clock["mono"] * 1000))
    return clock


def _make_state(token: int, mono: float) -> QuoteState:
    return QuoteState(
        token=token,
        ts=time.time(),
        monotonic_ts=mono,
        bid=150.0,
        ask=151.0,
        bid_qty=10,
        ask_qty=12,
        spread_pct=0.5,
        has_depth=True,
    )


def test_ws_diag_snapshot_uses_monotonic_and_debounce(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _install_time(monkeypatch)
    src = LiveKiteSource(kite=None)
    token = 101

    with src._ws_state_lock:  # noqa: SLF001 - deterministic test wiring
        src._subs.add(token)
        src._subscribed_tokens[token] = "FULL"
        src._ws_stale_counts[token] = 2
    with src._tick_lock:  # noqa: SLF001 - deterministic test wiring
        recent_ms = int((clock["mono"] - 0.05) * 1000)
        src._token_last_tick_ms[token] = recent_ms
        src._last_any_tick_ms = recent_ms
        src._last_tick_ts_ms = recent_ms
    src._last_force_reconnect_ts = clock["wall"] - 5.0

    diag = src.ws_diag_snapshot()
    assert diag["ws_connected"] is True
    assert diag["subs_count"] == 1
    assert diag["per_token_age_ms"][token] <= 60
    assert diag["stale_counts"][token] == 2
    assert diag["reconnect_debounce_left_ms"] == pytest.approx(5000, abs=2)

    # Fresh tick should reset ages to near zero using monotonic timestamps
    clock["mono"] = 2_000.0
    clock["wall"] = 20_000.0
    tick = {
        "instrument_token": token,
        "last_price": 152.0,
        "depth": {
            "buy": [{"price": 151.5, "quantity": 20}],
            "sell": [{"price": 152.5, "quantity": 18}],
        },
    }
    src._on_tick(tick)
    diag_fresh = src.ws_diag_snapshot()
    assert diag_fresh["last_tick_age_ms"] == 0
    assert diag_fresh["per_token_age_ms"][token] == 0


def test_stale_resubscribe_escalates_once_and_resets(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _install_time(monkeypatch)
    src = LiveKiteSource(kite=None)
    token = 202

    with src._tick_lock:  # noqa: SLF001 - deterministic test wiring
        src._quotes[token] = _make_state(token, clock["mono"] - 20.0)
    with src._ws_state_lock:  # noqa: SLF001 - deterministic test wiring
        src._subs.add(token)
        src._subscribed_tokens[token] = "FULL"

    events: list[tuple[str, dict[str, Any]]] = []

    def _record(event: str, **payload: Any) -> None:
        events.append((event, payload))

    monkeypatch.setattr(source.structured_log, "event", _record)

    reconnect_calls: list[dict[str, Any]] = []

    def _reconnect(*, reason: str | None = None, context: dict[str, Any] | None = None) -> None:
        reconnect_calls.append({"reason": reason, "context": context})

    monkeypatch.setattr(src, "reconnect_ws", _reconnect)
    src.ensure_token_subscribed = lambda *_a, **_k: True  # type: ignore[assignment]
    src.prime_option_quote = lambda *_a, **_k: (None, None, None)  # type: ignore[assignment]

    for _ in range(3):
        src._last_quote_ready_attempt[token] = 0.0  # noqa: SLF001 - controlled retry
        src._last_ws_resub_ms[token] = 0  # noqa: SLF001 - controlled retry
        src.resubscribe_if_stale(token)
        clock["mono"] += 0.2
        clock["wall"] += 0.2

    assert reconnect_calls and reconnect_calls[-1]["reason"] == "stale_x3"
    assert len(reconnect_calls) == 1

    escalate_payloads = [payload for name, payload in events if name == "ws_reconnect_escalate"]
    assert escalate_payloads and escalate_payloads[-1]["tokens"] == [token]
    assert not any(payload.get("source") == "ws_mid" for _name, payload in events)

    # Debounce prevents additional reconnects within the configured window
    for _ in range(3):
        src._last_quote_ready_attempt[token] = 0.0  # noqa: SLF001 - controlled retry
        src._last_ws_resub_ms[token] = 0  # noqa: SLF001 - controlled retry
        src.resubscribe_if_stale(token)
        clock["mono"] += 0.2
        clock["wall"] += 0.2
    assert len(reconnect_calls) == 1

    # Fresh tick resets stale counters
    clock["mono"] += 0.1
    clock["wall"] += 0.1
    tick = {
        "instrument_token": token,
        "last_price": 153.0,
        "depth": {
            "buy": [{"price": 152.0, "quantity": 25}],
            "sell": [{"price": 154.0, "quantity": 30}],
        },
    }
    src._on_tick(tick)
    with src._ws_state_lock:  # noqa: SLF001 - deterministic test wiring
        assert token not in src._ws_stale_counts

    # After tick, stale counters restart from zero and escalate once debounce expires
    clock["mono"] += 20.0
    clock["wall"] += 20.0
    with src._tick_lock:  # noqa: SLF001 - deterministic test wiring
        src._quotes[token] = _make_state(token, clock["mono"] - 20.0)
    for _ in range(3):
        src._last_quote_ready_attempt[token] = 0.0  # noqa: SLF001 - controlled retry
        src._last_ws_resub_ms[token] = 0  # noqa: SLF001 - controlled retry
        src.resubscribe_if_stale(token)
        clock["mono"] += 0.2
        clock["wall"] += 0.2
    assert len(reconnect_calls) == 2
