from __future__ import annotations

import time

from nifty_scalper_bot.streaming.stream_supervisor import StreamSupervisor


class _DummyStreamer:
    def __init__(self) -> None:
        self.running = False
        self.start_calls = 0
        self.subscribe_calls = 0
        self.unsubscribe_calls = 0

    _interval_s = 0.7
    _batch_size = 200

    def start(self) -> None:
        self.running = True
        self.start_calls += 1

    def stop(self) -> None:
        self.running = False

    def is_running(self) -> bool:
        return self.running

    def subscribe_tokens(self, tokens: list[int]) -> None:
        self.subscribe_calls += 1

    def unsubscribe_tokens(self, tokens: list[int]) -> None:
        self.unsubscribe_calls += 1


class _DummyResolver:
    def __init__(self, mapping: dict[str, int]) -> None:
        self.mapping = {key.upper(): value for key, value in mapping.items()}

    def resolve_many(self, symbols: list[str]) -> list[int | None]:
        return [self.mapping.get(symbol.upper()) for symbol in symbols]

    def resolve(self, symbol: str) -> int | None:
        return self.mapping.get(symbol.upper())


class _DummyMarketDataManager:
    def __init__(self) -> None:
        self.desired: set[int] = set()
        self.subscribe_calls = 0
        self.unsubscribe_calls = 0

    def request_token_subscriptions(self, tokens: list[int]) -> int:
        self.subscribe_calls += 1
        before = len(self.desired)
        self.desired.update(int(token) for token in tokens)
        return len(self.desired) - before

    def request_token_unsubscriptions(self, tokens: list[int]) -> int:
        self.unsubscribe_calls += 1
        before = len(self.desired)
        self.desired.difference_update(int(token) for token in tokens)
        return before - len(self.desired)

    def desired_token_count(self) -> int:
        return len(self.desired)

    def desired_tokens_snapshot(self) -> list[int]:
        return sorted(self.desired)


def test_supervisor_bootstrap_and_status() -> None:
    streamer = _DummyStreamer()
    resolver = _DummyResolver({"NIFTY": 101})
    supervisor = StreamSupervisor(
        streamer=streamer,
        resolver=resolver,
        market_data_manager=_DummyMarketDataManager(),
        default_symbols=["NIFTY"],
        autostart=True,
        monitor_interval_s=0.05,
    )

    supervisor.bootstrap()
    time.sleep(0.05)

    assert streamer.start_calls == 1
    assert supervisor.tracked_tokens() == [101]
    assert "tokens=1" in supervisor.status_line()

    supervisor.on_tick({"instrument_token": 101})
    health = supervisor.get_health()
    assert health.running is True
    assert health.tokens == 1

    supervisor.stop()


def test_supervisor_auto_restart_on_stop() -> None:
    streamer = _DummyStreamer()
    resolver = _DummyResolver({"BANKNIFTY": 202})
    supervisor = StreamSupervisor(
        streamer=streamer,
        resolver=resolver,
        market_data_manager=_DummyMarketDataManager(),
        autostart=True,
        monitor_interval_s=0.05,
    )
    supervisor.subscribe_symbols(["BANKNIFTY"])
    supervisor.ensure_started()
    time.sleep(0.05)

    first_start = streamer.start_calls
    streamer.running = False  # simulate unexpected stop
    time.sleep(0.35)

    assert streamer.start_calls >= first_start + 1

    supervisor.stop()


def test_supervisor_resolve_symbols_mapping() -> None:
    streamer = _DummyStreamer()
    resolver = _DummyResolver({"NIFTY": 123})
    supervisor = StreamSupervisor(
        streamer=streamer,
        resolver=resolver,
        market_data_manager=_DummyMarketDataManager(),
    )

    tokens, unresolved, mapping = supervisor.resolve_symbols(["NIFTY", "UNKNOWN"])

    assert tokens == [123]
    assert unresolved == ["UNKNOWN"]
    assert mapping == {"NIFTY": 123}


def test_supervisor_subscription_methods_route_via_mdm() -> None:
    streamer = _DummyStreamer()
    resolver = _DummyResolver({})
    mdm = _DummyMarketDataManager()
    supervisor = StreamSupervisor(
        streamer=streamer,
        resolver=resolver,
        market_data_manager=mdm,
        autostart=True,
    )

    total = supervisor.subscribe_tokens([101, 202, 101])
    assert total == 2
    assert mdm.subscribe_calls == 1
    assert streamer.subscribe_calls == 0
    assert streamer.start_calls == 1

    remaining = supervisor.unsubscribe_tokens([202, 999])
    assert remaining == 1
    assert mdm.unsubscribe_calls == 1
    assert streamer.unsubscribe_calls == 0
