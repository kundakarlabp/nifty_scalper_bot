"""Tests for the polling-based market data streamer."""

from __future__ import annotations

import time

from nifty_scalper_bot.streaming.polling_streamer import PollingStreamer


class _DummyBroker:
    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []

    def get_ltp_bulk(self, tokens: list[int]) -> dict[int, float]:
        self.calls.append(tuple(sorted(tokens)))
        now = time.time()
        return {int(token): 100.0 + idx + now % 1 for idx, token in enumerate(tokens)}


class _DepthBroker:
    def __init__(self) -> None:
        self.ltp_calls = 0
        self.quote_calls = 0

    def get_ltp_bulk(self, tokens: list[int]) -> dict[int, float]:
        self.ltp_calls += 1
        return {int(token): 90.0 for token in tokens}

    def get_quote_bulk(self, tokens: list[int]) -> dict[int, dict[str, object]]:
        self.quote_calls += 1
        return {
            int(token): {
                "last_price": 120.0,
                "depth": {"buy": [{"price": 119.0}], "sell": [{"price": 121.0}]},
            }
            for token in tokens
        }


def test_polling_streamer_tick_flow() -> None:
    ticks: list[dict[str, float]] = []
    broker = _DummyBroker()
    streamer = PollingStreamer(
        broker_client=broker,
        on_tick=lambda tick: ticks.append(tick),
        instrument_resolver=None,
        poll_interval_ms=200,
        batch_size=3,
    )
    streamer.subscribe([1, 2, 3, 4])
    streamer.start()
    try:
        time.sleep(0.8)
    finally:
        streamer.stop()

    assert broker.calls, "expected broker to be invoked"
    assert any(len(call) <= 3 for call in broker.calls)
    tokens = {tick.get("instrument_token") for tick in ticks}
    assert 1 in tokens
    assert 4 in tokens


def test_polling_streamer_requires_depth_prefers_quotes() -> None:
    broker = _DepthBroker()
    ticks: list[dict[str, object]] = []
    streamer = PollingStreamer(
        broker_client=broker,
        on_tick=lambda tick: ticks.append(tick),
        instrument_resolver=None,
        poll_interval_ms=150,
        batch_size=2,
        require_depth=True,
    )
    streamer.subscribe([10, 11])
    streamer.start()
    try:
        time.sleep(0.4)
    finally:
        streamer.stop()

    assert broker.quote_calls > 0
    assert broker.ltp_calls == 0
    assert ticks, "expected ticks from quote bulk"
    assert ticks[-1].get("depth")


def test_polling_streamer_rate_limit_warning_reset() -> None:
    broker = _DummyBroker()
    streamer = PollingStreamer(
        broker_client=broker,
        on_tick=lambda _: None,
        instrument_resolver=None,
        poll_interval_ms=500,
        batch_size=2,
    )

    streamer.subscribe(range(10))
    assert streamer._rate_limit_warned is True  # type: ignore[attr-defined]

    streamer.unsubscribe(range(8))
    assert streamer._rate_limit_warned is False  # type: ignore[attr-defined]


def test_polling_streamer_subscribe_tokens_alias() -> None:
    broker = _DummyBroker()
    streamer = PollingStreamer(
        broker_client=broker,
        on_tick=lambda _: None,
        instrument_resolver=None,
    )

    streamer.subscribe_tokens([7, 7, 9])

    assert streamer.tracked_tokens() == [7, 9]
