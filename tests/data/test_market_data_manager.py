from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import time
from typing import Any, Iterable
from unittest.mock import patch

import pytest

from nifty_scalper_bot.data.market_data_manager import MarketDataManager
from nifty_scalper_bot.streaming.websocket_manager import ConnectionState
from nifty_scalper_bot.utils.market_hours import MarketState


class DummyBroker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self._quotes: dict[str, dict[str, Any]] = {}
        self._tokens: dict[str, int] = {}

    def set_quote(self, symbol: str, quote: dict[str, Any]) -> None:
        self._quotes[symbol] = quote

    def set_token(self, symbol: str, token: int) -> None:
        self._tokens[symbol] = token

    def get_quote(self, symbol: str) -> dict[str, Any]:
        self.calls.append(("get_quote", (symbol,)))
        return dict(self._quotes.get(symbol, {"symbol": symbol, "ltp": 0.0}))

    def get_instrument_token(self, symbol: str) -> int:
        if symbol not in self._tokens:
            raise RuntimeError(f"missing token for {symbol}")
        return self._tokens[symbol]


class _RunnerProbe:
    def __init__(self) -> None:
        self.calls = 0

    def ingest_historical_bar(self, _bar: dict[str, Any]) -> None:
        self.calls += 1


class DummyWebSocket:
    def __init__(self) -> None:
        self.on_tick = None
        self.subscribed: list[tuple[str, Iterable[Any]]] = []
        self.unsubscribed: list[Iterable[Any]] = []
        self.start_calls: int = 0
        self.stop_calls: int = 0

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1

    def subscribe_tokens(self, tokens: Iterable[Any], mode: str = "ltp") -> None:
        self.subscribed.append((mode, list(tokens)))

    def set_tokens(self, tokens: Iterable[Any]) -> bool:
        self.subscribed.append(("full", sorted(int(token) for token in tokens)))
        return True

    def unsubscribe_tokens(self, tokens: Iterable[Any]) -> None:
        self.unsubscribed.append(list(tokens))

    def connection_state(self) -> ConnectionState:
        return ConnectionState.CONNECTED

    def is_connected(self) -> bool:
        return True


@pytest.fixture()
def broker() -> DummyBroker:
    broker = DummyBroker()
    broker.set_token("NIFTY23", 123)
    broker.set_token("NSE:NIFTY23", 123)
    broker.set_quote(
        "NIFTY23", {"ltp": 100.0, "bid": 99.5, "ask": 100.5, "ts": 1_000.0}
    )
    return broker


@pytest.fixture()
def ws() -> DummyWebSocket:
    return DummyWebSocket()


def test_market_data_fanout_and_warm_callback(
    monkeypatch: pytest.MonkeyPatch, broker: DummyBroker, ws: DummyWebSocket
) -> None:
    manager = MarketDataManager(broker, ws, cache_len=10, duplicate_window_ms=200)
    assert ws.on_tick is not None

    events: list[dict[str, Any]] = []

    def listener(tick: dict[str, Any]) -> None:
        events.append(tick)

    manager.subscribe("NIFTY23", listener)
    assert ws.subscribed[-1] == ("ltp", [123])

    # Simulate tick arrival
    now = iter([1000.0, 1000.05, 1000.5])
    monkeypatch.setattr(
        "nifty_scalper_bot.data.market_data_manager.time.monotonic",
        lambda: next(now),
    )
    assert ws.on_tick is not None
    ws.on_tick(
        {
            "instrument_token": 123,
            "last_price": 100.0,
            "depth": {"buy": [{"price": 99.5}], "sell": [{"price": 100.5}]},
        }
    )
    assert events[-1]["ltp"] == 100.0

    # Duplicate within debounce window should not fire
    ws.on_tick({"instrument_token": 123, "last_price": 100.0})
    assert len(events) == 1

    # Next tick outside window should reach listener
    ws.on_tick({"instrument_token": 123, "last_price": 101.0})
    assert events[-1]["ltp"] == 101.0

    # Late subscriber receives warm tick
    warmed: list[dict[str, Any]] = []
    manager.subscribe("NIFTY23", warmed.append)
    assert warmed[-1]["ltp"] == 101.0


def test_handle_tick_accepts_token_alias(
    broker: DummyBroker, ws: DummyWebSocket
) -> None:
    manager = MarketDataManager(broker, ws)
    events: list[dict[str, Any]] = []
    manager.subscribe("NIFTY23", events.append)
    assert ws.on_tick is not None

    ws.on_tick({"token": "123", "last_price": 105.0})

    assert events and events[-1]["ltp"] == 105.0
    latest = manager.get_latest_tick("NIFTY23")
    assert latest is not None
    assert latest["ltp"] == 105.0


def test_stale_tick_dropped(
    monkeypatch: pytest.MonkeyPatch, broker: DummyBroker, ws: DummyWebSocket
) -> None:
    monkeypatch.setenv("TICK_STALE_MS", "5")
    manager = MarketDataManager(broker, ws)
    events: list[dict[str, Any]] = []
    manager.subscribe("NIFTY23", events.append)
    assert ws.on_tick is not None

    stale_timestamp = time.time() - 1.0
    ws.on_tick(
        {"instrument_token": 123, "last_price": 100.0, "timestamp": stale_timestamp}
    )

    assert events == []
    assert manager.get_latest_tick("NIFTY23") is None


def test_pull_quote_updates_cache(broker: DummyBroker, ws: DummyWebSocket) -> None:
    manager = MarketDataManager(broker, ws)
    quote = manager.pull_quote("NIFTY23")
    assert quote["ltp"] == 100.0
    cached = manager.get_latest_tick("NIFTY23")
    assert cached is not None
    assert cached["ltp"] == 100.0


def test_nifty_alias_subscription_collapses_to_canonical(
    broker: DummyBroker,
    ws: DummyWebSocket,
) -> None:
    manager = MarketDataManager(broker, ws)

    manager.subscribe("NSE:NIFTY 50", lambda _: None)

    assert ws.subscribed[-1] == ("full", [256265])
    assert manager._active_subscribed_symbols == {"NSE:NIFTY"}


def test_pull_quote_canonicalizes_nifty_spot_alias(
    broker: DummyBroker,
    ws: DummyWebSocket,
) -> None:
    broker.set_quote(
        "NSE:NIFTY",
        {"symbol": "NSE:NIFTY", "ltp": 25100.0, "bid": 25099.5, "ask": 25100.5},
    )
    manager = MarketDataManager(broker, ws)

    quote = manager.pull_quote("NSE:NIFTY 50")

    assert broker.calls[-1] == ("get_quote", ("NSE:NIFTY",))
    assert quote["symbol"] == "NSE:NIFTY"
    cached = manager.get_latest_tick("NSE:NIFTY")
    assert cached is not None
    assert cached["ltp"] == pytest.approx(25100.0)


def test_pull_quote_uses_cached_tick_on_recoverable_direct_miss(
    broker: DummyBroker,
    ws: DummyWebSocket,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class MissingQuoteBroker(DummyBroker):
        def get_quote(self, symbol: str) -> dict[str, Any]:
            self.calls.append(("get_quote", (symbol,)))
            raise RuntimeError("Quote data missing for NSE:NIFTY")

    miss_broker = MissingQuoteBroker()
    miss_broker.set_token("NSE:NIFTY", 256265)
    manager = MarketDataManager(miss_broker, ws)
    manager._store_tick(  # noqa: SLF001 - explicit cache priming for fallback test
        "NSE:NIFTY",
        {"symbol": "NSE:NIFTY", "ltp": 25000.0, "source": "ws"},
    )
    caplog.set_level("WARNING")

    quote = manager.pull_quote("NSE:NIFTY")

    assert quote["ltp"] == pytest.approx(25000.0)
    assert quote["source"] == "ws"
    assert any("direct get_quote" in rec.message for rec in caplog.records)


def test_wait_for_symbol_hits_after_tick(
    monkeypatch: pytest.MonkeyPatch, broker: DummyBroker, ws: DummyWebSocket
) -> None:
    manager = MarketDataManager(broker, ws)

    # Ensure token resolution occurs so simulated ticks can be routed.
    manager.subscribe("NIFTY23", lambda _: None)
    assert ws.on_tick is not None
    ws.on_tick(
        {
            "instrument_token": 123,
            "last_price": 99.9,
            "depth": {"buy": [{"price": 99.8}], "sell": [{"price": 100.2}]},
        }
    )

    assert manager.wait_for_symbol("NIFTY23", timeout=0.1)


def test_wait_for_symbol_times_out(broker: DummyBroker, ws: DummyWebSocket) -> None:
    manager = MarketDataManager(broker, ws)
    assert not manager.wait_for_symbol("NIFTY23", timeout=0.0)


def test_startup_hydration_ingests_into_mdm_not_runner(
    broker: DummyBroker, ws: DummyWebSocket
) -> None:
    manager = MarketDataManager(broker, ws)
    probe = _RunnerProbe()
    manager._runner = probe  # noqa: SLF001
    accepted = manager.ingest_historical_ohlc(
        "NSE:NIFTY",
        [
            {
                "open": 1,
                "high": 2,
                "low": 0.5,
                "close": 1.5,
                "volume": 10,
                "timestamp": datetime.now(timezone.utc),
            }
        ],
    )
    assert accepted == 1
    assert probe.calls == 0
    assert manager.get_ohlc_bars("NSE:NIFTY")


@pytest.mark.asyncio
async def test_readiness_requires_spot_and_futures_even_when_option_ready(
    monkeypatch: pytest.MonkeyPatch, broker: DummyBroker, ws: DummyWebSocket
) -> None:
    monkeypatch.setattr(
        "nifty_scalper_bot.data.market_data_manager.get_market_state",
        lambda: MarketState.OPEN,
    )
    manager = MarketDataManager(broker, ws)
    manager._min_required_bars = 1  # noqa: SLF001
    manager._active_subscribed_symbols = {  # noqa: SLF001
        "NSE:NIFTY",
        "NFO:NIFTY26MAYFUT",
        "NFO:NIFTY26MAY25000CE",
        "NFO:NIFTY26MAY24950CE",
        "NFO:NIFTY26MAY25000PE",
        "NFO:NIFTY26MAY24950PE",
    }
    manager.set_readiness_requirements(
        spot_symbol="NSE:NIFTY",
        futures_symbol="NFO:NIFTY26MAYFUT",
        atm_ce_symbol="NFO:NIFTY26MAY25000CE",
        atm_pe_symbol="NFO:NIFTY26MAY25000PE",
        option_symbols=[
            "NFO:NIFTY26MAY25000CE",
            "NFO:NIFTY26MAY24950CE",
            "NFO:NIFTY26MAY25000PE",
            "NFO:NIFTY26MAY24950PE",
        ],
    )
    manager.ingest_historical_bar(
        {
            "symbol": "NFO:NIFTY26MAY25000CE",
            "open": 1,
            "high": 2,
            "low": 1,
            "close": 1.5,
            "volume": 1,
            "timestamp": datetime.now(timezone.utc),
        }
    )
    await manager.wait_until_ready(timeout=0.15)
    assert manager.ready is False


@pytest.mark.asyncio
async def test_readiness_passes_with_required_quorum(
    monkeypatch: pytest.MonkeyPatch, broker: DummyBroker, ws: DummyWebSocket
) -> None:
    monkeypatch.setattr(
        "nifty_scalper_bot.data.market_data_manager.get_market_state",
        lambda: MarketState.OPEN,
    )
    manager = MarketDataManager(broker, ws)
    manager._min_required_bars = 1  # noqa: SLF001
    symbols = [
        "NSE:NIFTY",
        "NFO:NIFTY26MAYFUT",
        "NFO:NIFTY26MAY25000CE",
        "NFO:NIFTY26MAY24950CE",
        "NFO:NIFTY26MAY25000PE",
        "NFO:NIFTY26MAY24950PE",
    ]
    manager._active_subscribed_symbols = set(symbols)  # noqa: SLF001
    manager.set_readiness_requirements(
        spot_symbol="NSE:NIFTY",
        futures_symbol="NFO:NIFTY26MAYFUT",
        atm_ce_symbol="NFO:NIFTY26MAY25000CE",
        atm_pe_symbol="NFO:NIFTY26MAY25000PE",
        option_symbols=symbols[2:],
    )
    for symbol in symbols:
        manager.ingest_historical_bar(
            {
                "symbol": symbol,
                "open": 1,
                "high": 2,
                "low": 1,
                "close": 1.5,
                "volume": 1,
                "timestamp": datetime.now(timezone.utc),
            }
        )
    await manager.wait_until_ready(timeout=0.5)
    assert manager.ready is True


@pytest.mark.asyncio
async def test_readiness_allows_outer_option_basket_to_lag(
    monkeypatch: pytest.MonkeyPatch, broker: DummyBroker, ws: DummyWebSocket
) -> None:
    monkeypatch.setattr(
        "nifty_scalper_bot.data.market_data_manager.get_market_state",
        lambda: MarketState.OPEN,
    )
    manager = MarketDataManager(broker, ws)
    manager._min_required_bars = 1  # noqa: SLF001
    symbols = [
        "NSE:NIFTY",
        "NFO:NIFTY26MAYFUT",
        "NFO:NIFTY26MAY25000CE",
        "NFO:NIFTY26MAY25000PE",
        "NFO:NIFTY26MAY24950CE",
        "NFO:NIFTY26MAY24950PE",
    ]
    manager._active_subscribed_symbols = set(symbols)  # noqa: SLF001
    manager.set_readiness_requirements(
        spot_symbol="NSE:NIFTY",
        futures_symbol="NFO:NIFTY26MAYFUT",
        atm_ce_symbol="NFO:NIFTY26MAY25000CE",
        atm_pe_symbol="NFO:NIFTY26MAY25000PE",
        option_symbols=symbols[2:],
    )
    for symbol in symbols[:4]:
        manager.ingest_historical_bar(
            {
                "symbol": symbol,
                "open": 1,
                "high": 2,
                "low": 1,
                "close": 1.5,
                "volume": 1,
                "timestamp": datetime.now(timezone.utc),
            }
        )
    await manager.wait_until_ready(timeout=0.5)
    assert manager.ready is True


def test_store_tick_refreshes_quote_age_for_live_ticks(
    broker: DummyBroker, ws: DummyWebSocket
) -> None:
    manager = MarketDataManager(broker, ws)
    symbol = "NSE:NIFTY"
    assert manager.quote_age_ms(symbol) >= 1_000_000_000

    manager._store_tick(  # noqa: SLF001 - verify core cache write path
        symbol,
        {"symbol": symbol, "ltp": 25100.0, "timestamp": time.time(), "source": "ws"},
    )

    assert manager.quote_age_ms(symbol) < 1_000_000_000


def test_quote_age_recovers_after_fresh_tick(
    broker: DummyBroker, ws: DummyWebSocket
) -> None:
    manager = MarketDataManager(broker, ws)
    symbol = "NSE:NIFTY"
    manager._last_quote_ts_ms[symbol] = 0  # noqa: SLF001 - stale sentinel setup
    assert manager.quote_age_ms(symbol) > 1_000

    manager._store_tick(  # noqa: SLF001 - verify live tick freshness overwrite
        symbol,
        {"symbol": symbol, "ltp": 25200.0, "timestamp": time.time(), "source": "ws"},
    )

    assert manager.quote_age_ms(symbol) < 1_000


def test_disable_rest_polling_stops_internal_thread(
    broker: DummyBroker, ws: DummyWebSocket
) -> None:
    class _ThreadProbe:
        def __init__(self) -> None:
            self.join_calls = 0

        def join(self, timeout: float | None = None) -> None:
            self.join_calls += 1

    manager = MarketDataManager(broker, ws)
    probe = _ThreadProbe()
    manager._rest_poll_thread = probe  # type: ignore[assignment]  # noqa: SLF001

    manager.disable_rest_polling(reason="test")

    assert manager._rest_poll_enabled is False  # noqa: SLF001
    assert manager._rest_poll_thread is None  # noqa: SLF001
    assert probe.join_calls == 1


def test_nifty_pull_quote_logs_structured_warning_on_total_failure(
    ws: DummyWebSocket, caplog: pytest.LogCaptureFixture
) -> None:
    class _NoQuoteBroker(DummyBroker):
        def get_quote(self, symbol: str) -> dict[str, Any]:
            raise RuntimeError(f"missing quote {symbol}")

    broker = _NoQuoteBroker()
    broker.set_token("NSE:NIFTY", 256265)
    manager = MarketDataManager(broker, ws)
    caplog.set_level("WARNING")
    quote = manager.pull_quote("NSE:NIFTY")
    assert quote.get("symbol") == "NSE:NIFTY"
    assert any("nifty_quote_pull_failed" in rec.message for rec in caplog.records)


def test_rest_poll_fallback_dispatches_ticks(
    monkeypatch: pytest.MonkeyPatch, broker: DummyBroker, ws: DummyWebSocket
) -> None:
    monkeypatch.setenv("MDM_POLL_FALLBACK", "1")
    monkeypatch.setenv("MDM_POLL_INTERVAL_SECONDS", "0.05")
    manager = MarketDataManager(broker, ws)
    events: list[dict[str, Any]] = []
    manager.subscribe("NIFTY23", events.append)
    manager.start()
    try:
        deadline = time.time() + 1.0
        while time.time() < deadline and not events:
            time.sleep(0.05)
        assert events, "expected REST fallback to deliver ticks"
    finally:
        manager.stop()


def test_rest_poll_uses_observed_time_for_freshness(
    monkeypatch: pytest.MonkeyPatch, broker: DummyBroker, ws: DummyWebSocket
) -> None:
    observed_at = 10_000.0
    broker.set_quote(
        "NIFTY23",
        {
            "ltp": 100.0,
            "bid": 99.5,
            "ask": 100.5,
            "timestamp": 1_000.0,
        },
    )
    monkeypatch.setattr(
        "nifty_scalper_bot.data.market_data_manager.time.time",
        lambda: observed_at,
    )
    manager = MarketDataManager(broker, ws)

    manager._poll_symbol("NIFTY23")

    latest = manager.get_latest_tick("NIFTY23")
    assert latest is not None
    assert latest["timestamp"] == pytest.approx(observed_at)
    assert latest["broker_timestamp"] == pytest.approx(1_000.0)
    assert latest["source"] == "rest"
    assert manager._has_recent_rest_ticks()


def test_ohlc_builder_generates_minute_bars(
    monkeypatch: pytest.MonkeyPatch, broker: DummyBroker, ws: DummyWebSocket
) -> None:
    monkeypatch.setenv("TICK_STALE_MS", "0")
    manager = MarketDataManager(broker, ws)
    broker.set_token("NIFTY", 301)
    broker.set_token("NIFTYFUT", 401)
    broker.set_quote("NIFTY", {"ltp": 0.0})
    broker.set_quote("NIFTYFUT", {"ltp": 0.0})
    captured: list[dict[str, Any]] = []
    manager.subscribe("NIFTY", captured.append)
    manager.subscribe("NIFTYFUT", lambda _: None)
    assert ws.on_tick is not None
    base = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    ws.on_tick(
        {
            "instrument_token": 301,
            "last_price": 100.0,
            "volume_traded": 0,
            "timestamp": base.timestamp(),
        }
    )
    ws.on_tick(
        {
            "instrument_token": 301,
            "last_price": 101.5,
            "volume_traded": 40,
            "timestamp": (base + timedelta(seconds=45)).timestamp(),
        }
    )
    ws.on_tick(
        {
            "instrument_token": 301,
            "last_price": 102.0,
            "volume_traded": 90,
            "timestamp": (base + timedelta(minutes=1, seconds=5)).timestamp(),
        }
    )
    ws.on_tick(
        {
            "instrument_token": 401,
            "last_price": 199.5,
            "volume_traded": 10,
            "timestamp": (base + timedelta(minutes=1)).timestamp(),
        }
    )

    nifty_bars = manager.get_ohlc_bars("NIFTY")
    assert len(nifty_bars) == 2
    first_bar = nifty_bars[0]
    second_bar = nifty_bars[-1]
    assert first_bar["open"] == pytest.approx(100.0)
    assert first_bar["high"] == pytest.approx(101.5)
    assert first_bar["low"] == pytest.approx(100.0)
    assert first_bar["close"] == pytest.approx(101.5)
    assert first_bar["volume"] == pytest.approx(40.0)
    assert second_bar["close"] == pytest.approx(102.0)
    assert second_bar["volume"] == pytest.approx(50.0)
    latest_only = manager.get_ohlc_bars("NIFTY", limit=1)
    assert len(latest_only) == 1
    snapshot = manager.market_data
    assert "NIFTY_bars" in snapshot
    assert "NIFTYFUT_bars" in snapshot


def test_out_of_order_live_tick_is_dropped(
    broker: DummyBroker, ws: DummyWebSocket
) -> None:
    manager = MarketDataManager(broker, ws)
    events: list[dict[str, Any]] = []
    manager.subscribe('NIFTY23', events.append)
    assert ws.on_tick is not None
    ws.on_tick(
        {
            'instrument_token': 123,
            'last_price': 100.0,
            'timestamp': 1_700_000_000.0,
        }
    )
    ws.on_tick(
        {
            'instrument_token': 123,
            'last_price': 99.5,
            'timestamp': 1_699_999_990.0,
        }
    )
    assert len(events) == 1
    assert events[0]['ltp'] == 100.0


def test_resolver_candidates() -> None:
    class _Resolver:
        def lookup(self, symbol: str) -> dict[str, Any]:  # noqa: D401 - test stub
            assert symbol == "NIFTY25O2025650CE"
            return {
                "tradingsymbol": "NIFTY25O2025650CE",
                "exchange": "NFO",
                "instrument_token": 321654,
            }

    manager = MarketDataManager(DummyBroker(), None, resolver=_Resolver())
    keys = manager._candidate_quote_keys("NIFTY25O2025650CE")
    assert keys == ["NIFTY25O2025650CE", "NFO:NIFTY25O2025650CE", 321654]


def test_refresh_candidates_success() -> None:
    symbol = "NIFTY25O2025650CE"

    class _Resolver:
        def lookup(self, _: str) -> dict[str, Any]:  # noqa: D401 - test stub
            return {
                "tradingsymbol": symbol,
                "exchange": "NFO",
                "instrument_token": 999999,
            }

    class _Broker:
        def __init__(self) -> None:
            self.symbol_calls: list[str] = []
            self.token_calls: list[int] = []

        def get_quote(self, key: str) -> dict[str, Any]:  # noqa: D401 - test stub
            self.symbol_calls.append(key)
            return {}

        def get_quote_by_token(self, token: int) -> dict[str, Any]:
            self.token_calls.append(token)
            return {"bid": 100.0, "ask": 101.0}

    broker = _Broker()
    manager = MarketDataManager(broker, None, resolver=_Resolver())
    assert manager.quote_age_ms(symbol) >= 1_000_000_000
    quote = manager.refresh_quote_now(symbol)
    assert quote is not None
    assert broker.token_calls == [999999]
    assert manager.quote_age_ms(symbol) < 1_000_000_000
    assert manager.has_quote(symbol)


def test_heartbeat_callback_invoked(broker: DummyBroker, ws: DummyWebSocket) -> None:
    manager = MarketDataManager(broker, ws)
    captured: list[float] = []

    manager.register_heartbeat_callback(captured.append)
    manager.bump_heartbeat(1.5)
    assert captured == [1.5]

    manager.register_heartbeat_callback(captured.append)
    manager.bump_heartbeat(2.5)
    assert captured[-1] == 2.5
    assert len(captured) == 2


@pytest.mark.asyncio
async def test_wait_for_live_tick_rejects_stale_tick(
    broker: DummyBroker, ws: DummyWebSocket
) -> None:
    manager = MarketDataManager(broker, ws)
    manager.subscribe("NIFTY23", lambda _: None)
    assert ws.on_tick is not None
    ws.on_tick(
        {
            "instrument_token": 123,
            "last_price": 100.0,
            "timestamp": time.time() - 10,
        }
    )

    with pytest.raises(RuntimeError, match="Live tick unavailable"):
        await manager.wait_for_live_tick(123, timeout=0.2)


@pytest.mark.asyncio
async def test_ensure_fresh_tick_schedules_background_rest_refresh(
    monkeypatch: pytest.MonkeyPatch, broker: DummyBroker
) -> None:
    monkeypatch.setenv("TICK_STALE_MS", "5")
    manager = MarketDataManager(broker, None)
    scheduled: list[str] = []

    async def _fake_refresh(symbol: str) -> None:
        scheduled.append(symbol)

    monkeypatch.setattr(manager, "_rest_refresh", _fake_refresh)
    await manager.ensure_fresh_tick("NIFTY23")
    await asyncio.sleep(0)

    assert scheduled == ["NSE:NIFTY23"]


def test_zombie_restart_respects_cooldown(
    monkeypatch: pytest.MonkeyPatch, broker: DummyBroker, ws: DummyWebSocket
) -> None:
    """Limit websocket zombie reconnect attempts using cooldown pacing."""

    manager = MarketDataManager(broker, ws)
    manager._zombie_restart_cooldown_sec = 5.0
    reconnect_calls: list[float] = []

    def _reconnect() -> None:
        reconnect_calls.append(1.0)

    ws.force_reconnect = _reconnect
    monotonic_values = iter([10.0, 12.0, 16.1])
    monkeypatch.setattr(
        "nifty_scalper_bot.data.market_data_manager.time.monotonic",
        lambda: next(monotonic_values),
    )

    manager._trigger_zombie_ws_restart()
    manager._trigger_zombie_ws_restart()
    manager._trigger_zombie_ws_restart()

    assert len(reconnect_calls) == 2


def test_zombie_restart_circuit_opens_after_failed_reconnects(
    monkeypatch: pytest.MonkeyPatch, broker: DummyBroker, ws: DummyWebSocket
) -> None:
    """Open zombie reconnect circuit only after repeated reconnect failures."""

    manager = MarketDataManager(broker, ws)
    manager._zombie_restart_limit = 2
    manager._zombie_restart_window = 120.0
    manager._zombie_restart_cooldown_sec = 0.0

    def _fail_reconnect() -> None:
        raise RuntimeError("ws down")

    ws.force_reconnect = _fail_reconnect
    monkeypatch.setattr(
        "nifty_scalper_bot.data.market_data_manager.time.monotonic",
        lambda: 42.0,
    )

    manager._trigger_zombie_ws_restart()
    manager._trigger_zombie_ws_restart()
    manager._trigger_zombie_ws_restart()

    assert manager._zombie_breaker_open_until == 162.0


def test_zombie_detection_uses_active_symbols_with_ticks_only(
    monkeypatch: pytest.MonkeyPatch, broker: DummyBroker, ws: DummyWebSocket
) -> None:
    monkeypatch.setattr(
        "nifty_scalper_bot.utils.market_hours.is_market_open",
        lambda: True,
    )
    manager = MarketDataManager(broker, ws)
    manager._zombie_tick_threshold_sec = 1.0
    manager._ws_connected = True
    manager._active_subscribed_symbols = {"NSE:NIFTY23", "NSE:NIFTY24"}
    manager._symbols_with_tick = {"NSE:NIFTY23"}
    manager._last_tick_time["NSE:NIFTY23"] = time.time()

    calls: list[int] = []
    manager._trigger_zombie_ws_restart = lambda: calls.append(1)  # type: ignore[method-assign]
    manager._check_zombie_ticks()
    assert calls == []


@pytest.mark.asyncio
async def test_wait_until_ready_enters_degraded_during_market_hours(
    broker: DummyBroker, ws: DummyWebSocket
) -> None:
    manager = MarketDataManager(broker, ws, cache_len=5)
    manager._min_required_bars = 2
    manager._active_subscribed_symbols = {"NSE:NIFTY23"}
    manager.hydration_complete = True

    with patch(
        "nifty_scalper_bot.data.market_data_manager.get_market_state",
        return_value=MarketState.OPEN,
    ):
        await manager.wait_until_ready(timeout=0.1)

    assert manager.ready is False
    assert manager.degraded is True


@pytest.mark.asyncio
async def test_wait_until_ready_bypasses_off_market(
    broker: DummyBroker, ws: DummyWebSocket
) -> None:
    manager = MarketDataManager(broker, ws, cache_len=5)
    manager._min_required_bars = 2
    manager._active_subscribed_symbols = {"NSE:NIFTY23"}
    manager._history["NSE:NIFTY23"].append({"ltp": 100.0, "timestamp": time.time()})

    with patch(
        "nifty_scalper_bot.data.market_data_manager.get_market_state",
        return_value=MarketState.CLOSED,
    ):
        await manager.wait_until_ready(timeout=0.1)

    assert manager.ready is True
    assert manager.degraded is False

    manager._history["NSE:NIFTY23"].append({"ltp": 101.0, "timestamp": time.time()})
    await manager.wait_until_ready(timeout=0.2)
    assert manager.ready is True
    assert manager.degraded is False


@pytest.mark.asyncio
async def test_wait_until_ready_stays_not_degraded_before_first_hydration(
    broker: DummyBroker, ws: DummyWebSocket
) -> None:
    manager = MarketDataManager(broker, ws, cache_len=5)
    manager._min_required_bars = 2
    manager._active_subscribed_symbols = {"NSE:NIFTY23"}

    with patch(
        "nifty_scalper_bot.data.market_data_manager.get_market_state",
        return_value=MarketState.OPEN,
    ):
        await manager.wait_until_ready(timeout=0.1)

    assert manager.ready is False
    assert manager.degraded is False


@pytest.mark.asyncio
async def test_warmup_history_primes_cache_without_emitting_callbacks(
    broker: DummyBroker, ws: DummyWebSocket
) -> None:
    manager = MarketDataManager(broker, ws)
    manager.register_symbol("NSE:NIFTY23", 123)

    events: list[dict[str, Any]] = []
    manager.subscribe("NSE:NIFTY23", events.append)

    class RestStub:
        async def get_historical_data(self, **kwargs: Any) -> list[dict[str, Any]]:
            return [
                {
                    "date": datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
                    "close": 101.0,
                    "volume": 50,
                },
                {
                    "date": datetime(2024, 1, 1, 10, 1, tzinfo=timezone.utc),
                    "close": 102.0,
                    "volume": 80,
                },
            ]

    manager._rest_client = RestStub()

    await manager.warmup_history(["NSE:NIFTY23"], lookback_minutes=30)

    assert events == []
    latest = manager.get_latest_tick("NSE:NIFTY23")
    assert latest is None
    bars = manager.get_ohlc_bars("NSE:NIFTY23")
    assert bars


def test_out_of_order_tick_is_discarded(
    monkeypatch: pytest.MonkeyPatch, broker: DummyBroker, ws: DummyWebSocket
) -> None:
    monkeypatch.setenv("TICK_STALE_MS", "0")
    manager = MarketDataManager(broker, ws)
    events: list[dict[str, Any]] = []
    manager.subscribe("NIFTY23", events.append)
    assert ws.on_tick is not None

    ws.on_tick({"instrument_token": 123, "last_price": 100.0, "timestamp": 1000.0})
    ws.on_tick({"instrument_token": 123, "last_price": 101.0, "timestamp": 999.0})

    assert len(events) == 1
    latest = manager.get_latest_tick("NIFTY23")
    assert latest is not None
    assert latest["ltp"] == 100.0


def test_all_ticks_are_processed_without_burst_throttle(
    monkeypatch: pytest.MonkeyPatch, broker: DummyBroker, ws: DummyWebSocket
) -> None:
    monkeypatch.setenv("MDM_TICK_BURST_LIMIT", "2")
    monkeypatch.setenv("MDM_TICK_BURST_WINDOW_SEC", "10")
    manager = MarketDataManager(broker, ws)
    events: list[dict[str, Any]] = []
    manager.subscribe("NIFTY23", events.append)
    assert ws.on_tick is not None

    ws.on_tick({"instrument_token": 123, "last_price": 100.0, "timestamp": 1000.0})
    ws.on_tick({"instrument_token": 123, "last_price": 101.0, "timestamp": 1001.0})
    ws.on_tick({"instrument_token": 123, "last_price": 102.0, "timestamp": 1002.0})

    assert len(events) == 3
    latest = manager.get_latest_tick("NIFTY23")
    assert latest is not None
    assert latest["ltp"] == pytest.approx(102.0)


def test_start_with_defer_ws_does_not_connect_websocket(
    broker: DummyBroker, ws: DummyWebSocket
) -> None:
    """``start(defer_ws=True)`` must not call ``ws.start``.

    This guards the startup ordering fix: the WebSocket must stay offline
    until the full token universe is resolved so the initial ``_on_connect``
    handshake never subscribes with only the spot token.
    """

    manager = MarketDataManager(broker, ws)
    manager.start(defer_ws=True)
    assert ws.start_calls == 0

    manager.start_websocket()
    assert ws.start_calls == 1

    # Idempotent — calling again must not reconnect
    manager.start_websocket()
    assert ws.start_calls == 1


def test_start_without_defer_ws_starts_websocket_immediately(
    broker: DummyBroker, ws: DummyWebSocket
) -> None:
    manager = MarketDataManager(broker, ws)
    manager.start()
    assert ws.start_calls == 1

    # Second start() call with defer_ws=False is a no-op because ws already up
    manager.start()
    assert ws.start_calls == 1
