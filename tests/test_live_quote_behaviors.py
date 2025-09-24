"""Unit tests for LiveKiteSource quote readiness helpers."""

from __future__ import annotations

import time
from typing import Any

import pytest

from src.data import source
from src.data.source import LiveKiteSource, QuoteReadyStatus, QuoteState


def _build_state(
    *,
    token: int,
    ts: float | None = None,
    monotonic_ts: float | None = None,
    bid: float | None = None,
    ask: float | None = None,
    bid_qty: int | None = None,
    ask_qty: int | None = None,
) -> QuoteState:
    """Helper to construct ``QuoteState`` instances for tests."""

    now = time.time()
    now_mono = time.monotonic()
    return QuoteState(
        token=token,
        ts=now if ts is None else ts,
        monotonic_ts=now_mono if monotonic_ts is None else monotonic_ts,
        bid=bid,
        ask=ask,
        bid_qty=bid_qty,
        ask_qty=ask_qty,
        spread_pct=None,
        has_depth=bool(bid and ask and bid > 0 and ask > 0),
    )


def test_get_last_tick_age_ms_uses_monotonic(monkeypatch: pytest.MonkeyPatch) -> None:
    src = LiveKiteSource(kite=None)
    with src._tick_lock:  # noqa: SLF001 - accessed for deterministic setup
        src._last_tick_ts_ms = 500

    monkeypatch.setattr(source, "monotonic_ms", lambda: 2_000)

    assert src.get_last_tick_age_ms() == 1_500


def test_subscribe_tracks_highest_mode() -> None:
    class DummyBroker:
        MODE_FULL = "full"

        def __init__(self) -> None:
            self.sub_calls: list[tuple[int, ...]] = []
            self.mode_calls: list[tuple[Any, tuple[int, ...]]] = []

        def subscribe(self, payload: list[int]) -> None:
            self.sub_calls.append(tuple(payload))

        def set_mode(self, mode: str, payload: list[int]) -> None:
            self.mode_calls.append((mode, tuple(payload)))

    broker = DummyBroker()
    src = LiveKiteSource(kite=broker)
    token = 111

    assert src.subscribe([token], mode="quote") is True
    assert broker.sub_calls == [(token,)]
    assert broker.mode_calls[-1][0] == "quote"
    assert src._subscribed_tokens[token] == "QUOTE"  # noqa: SLF001
    assert token in src._subscribed  # noqa: SLF001

    assert src.subscribe([token], mode="full") is True
    assert broker.sub_calls[-1] == (token,)
    assert src._subscribed_tokens[token] == "FULL"  # noqa: SLF001

    call_count = len(broker.sub_calls)
    assert src.subscribe([token], mode="ltp") is True
    assert len(broker.sub_calls) == call_count  # no downgrade re-subscription
    assert src._subscribed_tokens[token] == "FULL"  # noqa: SLF001


def test_get_last_bbo_seed_from_snapshot() -> None:
    src = LiveKiteSource(kite=None)
    token = 555
    ts_ms = 1_700_000_000_000

    snapshot = {
        "bid": 100.0,
        "ask": 101.0,
        "bid_qty": 5,
        "ask_qty": 6,
        "ts_ms": ts_ms,
    }
    with src._tick_lock:  # noqa: SLF001 - deterministic fixture
        src._option_quote_cache[token] = snapshot

    bid, ask, bid_qty, ask_qty, ts_val = src.get_last_bbo(token)

    assert (bid, ask, bid_qty, ask_qty, ts_val) == (100.0, 101.0, 5, 6, ts_ms)
    with src._tick_lock:  # noqa: SLF001
        cached = src._quotes[token]
    assert cached.bid == 100.0 and cached.ask == 101.0


def test_resubscribe_if_stale_seeds_from_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(source.settings, "MICRO__STALE_MS", 1_000, raising=False)
    src = LiveKiteSource(kite=None)
    token = 777

    with src._tick_lock:  # noqa: SLF001
        src._quotes[token] = _build_state(token=token, bid=None, ask=None)
        src._option_quote_cache[token] = {
            "bid": 110.0,
            "ask": 112.0,
            "bid_qty": 10,
            "ask_qty": 12,
            "ts_ms": int(time.time() * 1000) - 10_000,
        }

    monkeypatch.setattr(source.structured_log, "event", lambda *a, **k: None)

    def _fail(*_a: Any, **_k: Any) -> None:
        raise AssertionError("resubscribe should not be forced when snapshot seeds data")

    src.ensure_token_subscribed = _fail  # type: ignore[attr-defined]

    src.resubscribe_if_stale(token)

    with src._tick_lock:  # noqa: SLF001
        state = src._quotes[token]
    assert state.bid == 110.0 and state.ask == 112.0


def test_resubscribe_if_stale_triggers_force(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(source.settings, "MICRO__STALE_MS", 10, raising=False)
    src = LiveKiteSource(kite=None)
    token = 888
    past_mono = time.monotonic() - 5.0

    with src._tick_lock:  # noqa: SLF001
        src._quotes[token] = _build_state(
            token=token,
            monotonic_ts=past_mono,
            bid=120.0,
            ask=121.0,
            bid_qty=20,
            ask_qty=25,
        )
    src._last_quote_ready_attempt[token] = time.monotonic() - 1.0

    calls: list[tuple[int, str | None, bool]] = []

    def _ensure(token_id: int, *, mode: str | None = None, force: bool = False) -> bool:
        calls.append((token_id, mode, force))
        return True

    src.ensure_token_subscribed = _ensure  # type: ignore[attr-defined]

    events: list[dict[str, Any]] = []
    monkeypatch.setattr(
        source.structured_log,
        "event",
        lambda *a, **k: events.append(k),
    )

    src.resubscribe_if_stale(token)

    assert calls == [(token, "FULL", True)]
    assert events and events[-1]["reason"] == "stale_quote"


def test_resubscribe_current_reconnects_after_repeated_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src = LiveKiteSource(kite=None)
    token = 4321
    src._subscribed = {token}  # noqa: SLF001 - test setup

    def _fail_full(_tokens: list[int]) -> None:
        raise RuntimeError("subscribe failed")

    def _fail_force(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("force subscribe failed")

    monkeypatch.setattr(src, "_subscribe_tokens_full", _fail_full)
    monkeypatch.setattr(source, "_force_subscribe_tokens", _fail_force)

    reconnect_calls: list[dict[str, Any] | None] = []

    def _reconnect(*, reason: str | None = None, context: dict[str, Any] | None = None) -> None:
        reconnect_calls.append({"reason": reason, "context": context})

    monkeypatch.setattr(src, "reconnect_with_backoff", _reconnect)

    for _ in range(3):
        src.resubscribe_current()

    assert reconnect_calls == [
        {"reason": "resubscribe_failed", "context": {"failures": 3, "tokens": [token]}}
    ]
    assert src._resubscribe_failures == 0  # noqa: SLF001 - ensure reset after reconnect


def test_resubscribe_current_resets_failure_count_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src = LiveKiteSource(kite=None)
    token = 8765
    src._subscribed = {token}  # noqa: SLF001 - test setup

    attempts = {"count": 0}

    def _maybe_fail(tokens: list[int]) -> None:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("transient failure")

    def _force_fail(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("force failure")

    reconnect_calls: list[Any] = []

    monkeypatch.setattr(src, "_subscribe_tokens_full", _maybe_fail)
    monkeypatch.setattr(source, "_force_subscribe_tokens", _force_fail)
    monkeypatch.setattr(src, "reconnect_with_backoff", lambda *a, **k: reconnect_calls.append((a, k)))

    for attempt in range(3):
        src.resubscribe_current()
        if attempt < 2:
            assert src._resubscribe_failures == attempt + 1  # noqa: SLF001 - inspect counter

    assert reconnect_calls == []
    assert src._resubscribe_failures == 0  # noqa: SLF001 - counter reset after success


def test_ensure_quote_ready_returns_live_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(source.settings, "QUOTES__RETRY_ATTEMPTS", 1, raising=False)
    monkeypatch.setattr(source.settings, "MICRO__STALE_MS", 2_000, raising=False)
    monkeypatch.setattr(source.settings, "QUOTES__PRIME_TIMEOUT_MS", 0, raising=False)
    monkeypatch.setattr(source.settings, "QUOTES__RETRY_JITTER_MS", 0, raising=False)
    monkeypatch.setattr(source.settings, "QUOTES__SNAPSHOT_DELAY_MS", 100, raising=False)

    base_mono = time.monotonic()
    monkeypatch.setattr(source, "monotonic_ms", lambda: int((base_mono + 0.2) * 1000))
    monkeypatch.setattr(source.time, "sleep", lambda _x: None)
    monkeypatch.setattr(source.structured_log, "event", lambda *a, **k: None)

    src = LiveKiteSource(kite=None)
    token = 999

    with src._tick_lock:  # noqa: SLF001
        src._quotes[token] = QuoteState(
            token=token,
            ts=time.time(),
            monotonic_ts=base_mono,
            bid=130.0,
            ask=131.0,
            bid_qty=15,
            ask_qty=18,
            spread_pct=0.5,
            has_depth=True,
        )

    ensure_calls: list[tuple[int, str | None, bool]] = []

    def _ensure(token_id: int, *, mode: str | None = None, force: bool = False) -> bool:
        ensure_calls.append((token_id, mode, force))
        return True

    src.ensure_token_subscribed = _ensure  # type: ignore[attr-defined]

    diag_calls: list[QuoteReadyStatus] = []
    monkeypatch.setattr(
        source,
        "emit_quote_diag",
        lambda **kwargs: diag_calls.append(
            QuoteReadyStatus(
                ok=kwargs["reason"] == "ready",
                reason=kwargs["reason"],
                retries=kwargs["retries"],
                bid=kwargs["bid"],
                ask=kwargs["ask"],
                bid_qty=kwargs["bid_qty"],
                ask_qty=kwargs["ask_qty"],
                last_tick_age_ms=kwargs["last_tick_age_ms"],
                source=kwargs["source"],
            )
        ),
    )

    status = src.ensure_quote_ready(token, mode="full", symbol="TEST")

    assert status.ok is True
    assert status.reason == "ready"
    assert status.source == "live"
    assert status.last_tick_age_ms == 200
    assert ensure_calls == [(token, "FULL", False)]
    assert diag_calls and diag_calls[-1].reason == "ready"
    assert src._last_quote_ready_attempt[token] >= base_mono  # noqa: SLF001

