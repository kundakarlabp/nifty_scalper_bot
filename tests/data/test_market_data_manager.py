from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from datetime import date
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


class DisconnectedWebSocket(DummyWebSocket):
    def is_connected(self) -> bool:
        return False


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


def test_pull_quote_403_marks_quote_api_unavailable(
    ws: DummyWebSocket,
) -> None:
    class ForbiddenBroker(DummyBroker):
        def get_quote(self, symbol: str) -> dict[str, Any]:
            raise RuntimeError(f"HTTP 403 access denied for {symbol}")

    manager = MarketDataManager(ForbiddenBroker(), ws)
    response = manager.pull_quote("NSE:NIFTY")
    assert response["symbol"] == "NSE:NIFTY"
    status = manager.quote_api_status_snapshot()
    assert status["available"] is False
    assert status["error"] == "access_denied"
    assert isinstance(status["last_checked_at"], float)


def test_pull_quote_direct_missing_is_rate_limited(ws: DummyWebSocket) -> None:
    class MissingBroker(DummyBroker):
        def get_quote(self, symbol: str) -> dict[str, Any]:
            self.calls.append(("get_quote", (symbol,)))
            raise RuntimeError(f"Quote data missing for {symbol}")

    manager = MarketDataManager(MissingBroker(), ws)
    first = manager.pull_quote("NFO:NIFTY26MAYFUT")
    second = manager.pull_quote("NFO:NIFTY26MAYFUT")
    assert first["symbol"] == "NFO:NIFTY26MAYFUT"
    assert second.get("quote_unavailable_reason") == "quote_symbol_missing_cooldown"
    assert manager._quote_direct_miss_count["NFO:NIFTY26MAYFUT"] >= 1  # noqa: SLF001


def test_active_future_resolver_replaces_expired_month_future(broker: DummyBroker, ws: DummyWebSocket) -> None:
    resolver = type(
        "Resolver",
        (),
        {
            "list_instruments": lambda self: [
                {"segment": "NFO-FUT", "instrument_type": "FUT", "tradingsymbol": "NIFTY26MAYFUT", "expiry": "2026-05-20"},
                {"segment": "NFO-FUT", "instrument_type": "FUT", "tradingsymbol": "NIFTY26JUNFUT", "expiry": "2026-06-25"},
            ]
        },
    )()
    manager = MarketDataManager(broker, ws, resolver=resolver)
    symbol = manager.resolve_active_nifty_future_symbol(now=datetime(2026, 5, 27, tzinfo=timezone.utc))
    assert symbol == "NFO:NIFTY26JUNFUT"


def test_active_future_resolver_uses_broker_instruments_when_resolver_unavailable(ws: DummyWebSocket) -> None:
    class BrokerWithInstruments(DummyBroker):
        def instruments(self, exchange: str):
            assert exchange == "NFO"
            return [
                {"segment": "NFO-FUT", "instrument_type": "FUT", "name": "NIFTY", "tradingsymbol": "NIFTY26MAYFUT", "expiry": "2026-05-20"},
                {"segment": "NFO-FUT", "instrument_type": "FUT", "name": "NIFTY", "tradingsymbol": "NIFTY26JUNFUT", "expiry": "2026-06-25"},
            ]

    manager = MarketDataManager(BrokerWithInstruments(), ws, resolver=None)

    assert manager.get_active_nifty_future_symbol_cached(now=datetime(2026, 5, 27, tzinfo=timezone.utc)) == "NFO:NIFTY26JUNFUT"


def test_active_future_resolver_rejects_banknifty_and_finnifty(broker: DummyBroker, ws: DummyWebSocket) -> None:
    resolver = type(
        "Resolver",
        (),
        {"list_instruments": lambda self: [
            {"segment": "NFO-FUT", "instrument_type": "FUT", "tradingsymbol": "BANKNIFTY26JUNFUT", "expiry": "2026-06-25"},
            {"segment": "NFO-FUT", "instrument_type": "FUT", "tradingsymbol": "FINNIFTY26JUNFUT", "expiry": "2026-06-25"},
            {"segment": "NFO-FUT", "instrument_type": "FUT", "tradingsymbol": "NIFTY26JUN24000CE", "expiry": "2026-06-25"},
            {"segment": "NFO-FUT", "instrument_type": "FUT", "name": "NIFTY", "tradingsymbol": "NIFTY26JUNFUT", "expiry": "2026-06-25"},
        ]}
    )()
    manager = MarketDataManager(broker, ws, resolver=resolver)
    assert manager.resolve_active_nifty_future_symbol(now=datetime(2026, 5, 27, tzinfo=timezone.utc)) == "NFO:NIFTY26JUNFUT"


def test_active_future_resolver_calls_instruments_with_nfo_when_required(broker: DummyBroker, ws: DummyWebSocket) -> None:
    class Resolver:
        def instruments(self, exchange=None):
            if exchange == "NFO":
                return [{"segment": "NFO-FUT", "instrument_type": "FUTIDX", "name": "NIFTY", "tradingsymbol": "NIFTY26JUNFUT", "expiry": "2026-06-25"}]
            return []
    manager = MarketDataManager(broker, ws, resolver=Resolver())
    assert manager.resolve_active_nifty_future_symbol(now=datetime(2026, 5, 27, tzinfo=timezone.utc)) == "NFO:NIFTY26JUNFUT"


def test_active_future_resolver_accepts_mapping_payloads(broker: DummyBroker, ws: DummyWebSocket) -> None:
    class Resolver:
        def instruments(self, exchange=None):
            return {"1": {"exchange": "NFO", "segment": "NFO-FUT", "instrument_type": "FUTIDX", "name": "NIFTY", "tradingsymbol": "NIFTY26JUNFUT", "expiry": "2026-06-25"}}
    manager = MarketDataManager(broker, ws, resolver=Resolver())
    assert manager.resolve_active_nifty_future_symbol(now=datetime(2026, 5, 27, tzinfo=timezone.utc)) == "NFO:NIFTY26JUNFUT"


def test_active_future_resolver_handles_date_expiry(broker: DummyBroker, ws: DummyWebSocket) -> None:
    resolver = type("Resolver", (), {"list_instruments": lambda self: [{"segment": "NFO-FUT", "instrument_type": "FUT", "name": "NIFTY", "tradingsymbol": "NIFTY26JUNFUT", "expiry": date(2026, 6, 25)}]})()
    manager = MarketDataManager(broker, ws, resolver=resolver)
    assert manager.resolve_active_nifty_future_symbol(now=datetime(2026, 5, 27, tzinfo=timezone.utc)) == "NFO:NIFTY26JUNFUT"


@pytest.mark.asyncio
async def test_wait_until_ready_not_ready_log_is_throttled(
    broker: DummyBroker,
    ws: DummyWebSocket,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = MarketDataManager(broker, ws)
    caplog.set_level("INFO")
    await manager.wait_until_ready(timeout=0.35)
    messages = [
        rec.message for rec in caplog.records if "DATA_PIPELINE_NOT_READY" in rec.message
    ]
    assert len(messages) <= 1


@pytest.mark.asyncio
async def test_spot_ready_not_logged_for_huge_age(
    broker: DummyBroker,
    ws: DummyWebSocket,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(
        "nifty_scalper_bot.data.market_data_manager.get_market_state",
        lambda: MarketState.OPEN,
    )
    manager = MarketDataManager(broker, ws)
    manager._active_subscribed_symbols = {"NSE:NIFTY"}  # noqa: SLF001
    manager._min_required_bars = 1  # noqa: SLF001
    manager._history["NSE:NIFTY"].append({"ltp": 25000.0})  # noqa: SLF001
    manager.set_readiness_requirements(
        spot_symbol="NSE:NIFTY",
        futures_symbol="",
        atm_ce_symbol="",
        atm_pe_symbol="",
        option_symbols=[],
    )
    monkeypatch.setattr(manager, "symbol_data_age_ms", lambda _symbol: 1_000_000_000)
    caplog.set_level("INFO")
    await manager.wait_until_ready(timeout=0.15)
    assert not any("SPOT_READY" in rec.message for rec in caplog.records)


def test_nifty_alias_subscription_collapses_to_canonical(
    broker: DummyBroker,
    ws: DummyWebSocket,
) -> None:
    manager = MarketDataManager(broker, ws)

    manager.subscribe("NSE:NIFTY 50", lambda _: None)

    assert ws.subscribed[-1] == ("full", [256265])
    assert manager._active_subscribed_symbols == {"NSE:NIFTY"}


@pytest.mark.parametrize("alias", ["NIFTY", "NIFTY 50", "NSE:NIFTY", "NSE:NIFTY 50"])
def test_nifty_aliases_share_same_token(
    alias: str, broker: DummyBroker, ws: DummyWebSocket
) -> None:
    manager = MarketDataManager(broker, ws)
    token = manager._resolve_token(alias)  # noqa: SLF001
    assert token == 256265


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
        manager._store_tick(  # noqa: SLF001 - readiness is raw/live tick based
            symbol,
            {
                "symbol": symbol,
                "ltp": 1.5,
                "timestamp": datetime.now(timezone.utc),
                "received_at": datetime.now(timezone.utc),
                "source": "ws",
            },
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
        manager._store_tick(  # noqa: SLF001 - readiness is raw/live tick based
            symbol,
            {
                "symbol": symbol,
                "ltp": 1.5,
                "timestamp": datetime.now(timezone.utc),
                "received_at": datetime.now(timezone.utc),
                "source": "ws",
            },
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


def test_ws_tick_proof_storage(
    broker: DummyBroker,
    ws: DummyWebSocket,
) -> None:
    manager = MarketDataManager(broker, ws)
    manager.register_symbol("NSE:NIFTY", 256265)
    manager.process_ticks(
        [
            {
                "instrument_token": 256265,
                "last_price": 25250.0,
                "exchange_timestamp": datetime.now(timezone.utc),
            }
        ]
    )
    manager._drain_tick_queue_sync()  # noqa: SLF001
    price = manager.get_cached_ltp("NSE:NIFTY", require_ws=True)
    assert price == pytest.approx(25250.0)


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


@pytest.mark.asyncio
async def test_ensure_fresh_tick_uses_symbol_level_refresh_even_if_global_fresh(
    monkeypatch: pytest.MonkeyPatch, broker: DummyBroker
) -> None:
    manager = MarketDataManager(broker, None)
    manager.last_tick_time = time.time()  # authoritative stream is globally fresh
    manager._store_tick(  # noqa: SLF001
        "NSE:NIFTY",
        {
            "symbol": "NSE:NIFTY",
            "ltp": 25000.0,
            "timestamp": time.time() - 5000.0,
            "received_at": time.time() - 5000.0,
        },
    )
    scheduled: list[str] = []

    async def _fake_refresh(symbol: str) -> None:
        scheduled.append(symbol)

    monkeypatch.setattr(manager, "_rest_refresh", _fake_refresh)
    await manager.ensure_fresh_tick("NSE:NIFTY")
    await asyncio.sleep(0)
    assert scheduled == ["NSE:NIFTY"]


@pytest.mark.asyncio
async def test_rest_refresh_updates_tick_cache_with_rest_source(
    broker: DummyBroker, ws: DummyWebSocket
) -> None:
    broker.set_quote("NSE:NIFTY", {"ltp": 25123.4, "last_price": 25123.4})
    manager = MarketDataManager(broker, ws)
    await manager._rest_refresh("NSE:NIFTY")  # noqa: SLF001
    tick = manager.get_latest_tick("NSE:NIFTY")
    assert tick is not None
    assert tick["ltp"] == pytest.approx(25123.4)
    assert tick["source"] in {"rest", "rest_ltp"}
    assert isinstance(tick.get("received_at"), float)


@pytest.mark.asyncio
async def test_rest_refresh_invalid_payload_does_not_fake_tick(
    broker: DummyBroker, ws: DummyWebSocket
) -> None:
    broker.set_quote("NSE:NIFTY", {"ltp": None, "last_price": None})
    manager = MarketDataManager(broker, ws)
    await manager._rest_refresh("NSE:NIFTY")  # noqa: SLF001
    assert manager.get_latest_tick("NSE:NIFTY") is None


def test_subscription_queued_until_ws_connected(broker: DummyBroker) -> None:
    ws = DisconnectedWebSocket()
    manager = MarketDataManager(broker, ws)
    manager.subscribe("NSE:NIFTY", lambda _: None)
    assert 256265 in manager._pending_subscriptions  # noqa: SLF001
    assert 256265 not in manager._confirmed_subscriptions  # noqa: SLF001
    manager.set_ws_connected(True)
    assert 256265 not in manager._pending_subscriptions  # noqa: SLF001


def test_first_tick_marks_subscription_confirmed(
    broker: DummyBroker, ws: DummyWebSocket
) -> None:
    manager = MarketDataManager(broker, ws)
    manager.subscribe("NSE:NIFTY", lambda _: None)
    assert ws.on_tick is not None
    ws.on_tick({"instrument_token": 256265, "last_price": 25100.0})
    assert 256265 in manager._confirmed_subscriptions  # noqa: SLF001


def test_tick_wallclock_prefers_received_at(
    broker: DummyBroker, ws: DummyWebSocket
) -> None:
    manager = MarketDataManager(broker, ws)
    now = time.time()
    tick = {
        "received_at": now,
        "timestamp": now - 100.0,
    }
    assert manager._tick_wallclock(tick) == pytest.approx(now)  # noqa: SLF001


def test_tick_wallclock_parses_datetime_and_string(
    broker: DummyBroker, ws: DummyWebSocket
) -> None:
    manager = MarketDataManager(broker, ws)
    dt = datetime.now(timezone.utc) - timedelta(seconds=1)
    parsed_dt = manager._tick_wallclock({"timestamp": dt})  # noqa: SLF001
    parsed_str = manager._tick_wallclock({"timestamp": dt.isoformat()})  # noqa: SLF001
    assert parsed_dt is not None
    assert parsed_str is not None


def test_market_snapshot_handles_missing_bid_ask(
    broker: DummyBroker, ws: DummyWebSocket
) -> None:
    manager = MarketDataManager(broker, ws)
    manager._store_tick(  # noqa: SLF001
        "NSE:NIFTY",
        {
            "symbol": "NSE:NIFTY",
            "ltp": 25010.0,
            "timestamp": time.time(),
            "received_at": time.time(),
            "source": "ws",
        },
    )
    snapshot = manager.get_symbol_snapshot("NIFTY 50")
    assert snapshot.canonical_symbol == "NSE:NIFTY"
    assert snapshot.bid is None
    assert snapshot.ask is None
    assert snapshot.tick_age_s is not None


def test_market_snapshot_uses_candle_close_when_ltp_missing(
    broker: DummyBroker, ws: DummyWebSocket
) -> None:
    manager = MarketDataManager(broker, ws)
    manager._store_tick(  # noqa: SLF001
        "NSE:NIFTY",
        {
            "symbol": "NSE:NIFTY",
            "ltp": None,
            "timestamp": time.time(),
            "received_at": time.time(),
            "source": "ws",
        },
    )
    bar_key = manager._bar_symbol_key("NSE:NIFTY")  # noqa: SLF001
    manager._ohlc[bar_key].append({"close": 25234.5, "provisional": False})  # noqa: SLF001
    snapshot = manager.get_symbol_snapshot("NSE:NIFTY")
    assert snapshot.ltp == 25234.5
    assert snapshot.source == "ws_candle_fallback"


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
    calls: list[dict[str, Any]] = []

    async def fake_ensure_history(symbol: str, **kwargs: Any):
        calls.append({"symbol": symbol, **kwargs})

    manager.ensure_history = fake_ensure_history  # type: ignore[method-assign]

    await manager.warmup_history(["NSE:NIFTY23"], lookback_minutes=30)

    assert calls == [
        {
            "symbol": "NSE:NIFTY23",
            "interval": "minute",
            "required_bars": 30,
            "target_bars": 30,
            "reason": "warmup_history",
        }
    ]


@pytest.mark.asyncio
async def test_warmup_history_delegates_without_resolver_or_broker_fallback(
    ws: DummyWebSocket,
) -> None:
    class ResolverMiss:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def resolve(self, symbol: str):
            self.calls.append(symbol)
            return None

    class BrokerFallback:
        def __init__(self) -> None:
            self.token_calls: list[str] = []
            self.history_tokens: list[int] = []

        def get_instrument_token(self, symbol: str) -> int:
            self.token_calls.append(symbol)
            return 222

    resolver = ResolverMiss()
    broker = BrokerFallback()
    manager = MarketDataManager(broker, ws, resolver=resolver)
    calls: list[dict[str, Any]] = []

    async def fake_ensure_history(symbol: str, **kwargs: Any):
        calls.append({"symbol": symbol, **kwargs})

    manager.ensure_history = fake_ensure_history  # type: ignore[method-assign]

    await manager.warmup_history(["NFO:NIFTY26JUNFUT"], lookback_minutes=30)

    assert calls and calls[0]["symbol"] == "NFO:NIFTY26JUNFUT"
    assert resolver.calls == []
    assert broker.token_calls == []
    assert broker.history_tokens == []


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


def test_mdm_datahub_register_no_recursion(broker: DummyBroker, ws: DummyWebSocket) -> None:
    class Hub:
        def __init__(self) -> None:
            self.calls = 0

        def register_local_symbol(self, symbol: str, token: int) -> str:
            self.calls += 1
            return symbol

    manager = MarketDataManager(broker, ws)
    hub = Hub()
    manager._datahub = hub  # noqa: SLF001
    manager.register_symbol("NSE:NIFTY", 256265)
    assert hub.calls == 1

    # Second start() call with defer_ws=False is a no-op because ws already up
    manager.start()
    assert ws.start_calls == 1


def test_readiness_state_marks_futures_soft_requirement() -> None:
    manager = MarketDataManager.__new__(MarketDataManager)
    manager._tick_stale_threshold_ms = 60_000  # noqa: SLF001
    manager._is_symbol_fresh = lambda *_a, **_k: True  # type: ignore[method-assign]  # noqa: SLF001
    state = manager._readiness_state(  # noqa: SLF001
        {"NFO:NIFTYCE": 20, "NFO:NIFTYPE": 20, "NFO:FUT": 0},
        20,
        {"spot": "NSE:NIFTY", "futures": "NFO:FUT", "options": ["NFO:NIFTYCE", "NFO:NIFTYPE"]},
    )
    assert state["hard_ready"] is True
    assert "futures" in state["missing_soft"]
    assert "futures" not in state["missing_hard"]


def test_readiness_state_accepts_nearby_options_when_exact_atm_missing() -> None:
    manager = MarketDataManager.__new__(MarketDataManager)
    manager._tick_stale_threshold_ms = 60_000  # noqa: SLF001
    manager._is_symbol_fresh = lambda *_a, **_k: True  # type: ignore[method-assign]  # noqa: SLF001
    state = manager._readiness_state(  # noqa: SLF001
        {"NFO:NIFTY24CE": 20, "NFO:NIFTY24PE": 20, "NFO:ATMCE": 0, "NFO:ATMPE": 0},
        20,
        {
            "spot": "NSE:NIFTY",
            "atm_ce": "NFO:ATMCE",
            "atm_pe": "NFO:ATMPE",
            "options": ["NFO:NIFTY24CE", "NFO:NIFTY24PE"],
        },
    )
    assert state["hard_ready"] is True
    assert state["strict_atm_ce_ready"] is False
    assert state["strict_atm_pe_ready"] is False
    assert state["selected_ce"] == "NFO:NIFTY24CE"
    assert state["selected_pe"] == "NFO:NIFTY24PE"


def test_coerce_timestamp_prefers_exchange_timestamp() -> None:
    tick = {"exchange_timestamp": "2026-05-06T10:00:00+00:00", "timestamp": 0}
    coerced = MarketDataManager._coerce_timestamp(tick)  # noqa: SLF001
    assert coerced > 1_700_000_000


def test_selected_option_month_fallback_rotates_may_future_to_jun_future(ws: DummyWebSocket) -> None:
    manager = MarketDataManager(DummyBroker(), ws, resolver=None)
    out = manager.maybe_rotate_nifty_futures_context_result(
        "NFO:NIFTY26MAYFUT",
        reason="test",
        selected_option_symbols=["NFO:NIFTY26JUN23950CE", "NFO:NIFTY26JUN23950PE"],
    )
    assert out.symbol == "NFO:NIFTY26JUNFUT"
    assert out.rotated is True


def test_maybe_rotate_unresolved_returns_symbol_none(ws: DummyWebSocket) -> None:
    manager = MarketDataManager(DummyBroker(), ws, resolver=None)
    out = manager.maybe_rotate_nifty_futures_context_result("NFO:NIFTY26MAYFUT", reason="no_data")
    assert out.unresolved is True
    assert out.symbol is None

def test_get_cached_ltp_suspended_future_returns_float_from_cache(broker: DummyBroker, ws: DummyWebSocket) -> None:
    manager = MarketDataManager(broker, ws)
    manager._suspended_context_symbols.add("NFO:NIFTY26MAYFUT")  # noqa: SLF001
    manager._latest_ticks["NFO:NIFTY26MAYFUT"] = {"ltp": 123.5}  # noqa: SLF001
    assert manager.get_cached_ltp("NFO:NIFTY26MAYFUT") == 123.5


def test_get_cached_ltp_suspended_future_returns_none_without_cache(broker: DummyBroker, ws: DummyWebSocket) -> None:
    manager = MarketDataManager(broker, ws)
    manager._suspended_context_symbols.add("NFO:NIFTY26MAYFUT")  # noqa: SLF001
    assert manager.get_cached_ltp("NFO:NIFTY26MAYFUT") is None


def test_get_cached_ltp_never_returns_dict(broker: DummyBroker, ws: DummyWebSocket) -> None:
    manager = MarketDataManager(broker, ws)
    manager._suspended_context_symbols.add("NFO:NIFTY26MAYFUT")  # noqa: SLF001
    value = manager.get_cached_ltp("NFO:NIFTY26MAYFUT")
    assert value is None or isinstance(value, float)


def test_pull_quote_suspended_future_returns_unavailable_dict_without_broker_call(ws: DummyWebSocket) -> None:
    class FailBroker(DummyBroker):
        def get_quote(self, symbol: str) -> dict[str, Any]:
            raise AssertionError("broker must not be called")

    manager = MarketDataManager(FailBroker(), ws)
    manager._suspended_context_symbols.add("NFO:NIFTY26MAYFUT")  # noqa: SLF001
    out = manager.pull_quote("NFO:NIFTY26MAYFUT")
    assert out == {}

def test_market_data_manager_initializes_suspended_context_symbols(broker: DummyBroker, ws: DummyWebSocket) -> None:
    manager = MarketDataManager(broker, ws)
    assert isinstance(manager._suspended_context_symbols, set)  # noqa: SLF001
    assert isinstance(manager._suspended_context_symbol_until, dict)  # noqa: SLF001


def test_suspend_context_symbol_blocks_pull_quote_until_ttl(broker: DummyBroker, ws: DummyWebSocket) -> None:
    manager = MarketDataManager(broker, ws)
    manager.suspend_context_symbol('NFO:NIFTY26MAYFUT', reason='stale', ttl_s=300)
    out = manager.pull_quote('NFO:NIFTY26MAYFUT')
    assert out == {}


def test_suspended_context_symbol_expires_after_ttl(broker: DummyBroker, ws: DummyWebSocket) -> None:
    manager = MarketDataManager(broker, ws)
    manager.suspend_context_symbol('NFO:NIFTY26MAYFUT', reason='stale', ttl_s=0.001)
    import time as _t
    _t.sleep(0.01)
    assert manager.is_context_symbol_suspended('NFO:NIFTY26MAYFUT') is False


def test_rotate_active_future_context_purges_expired_future_state(ws: DummyWebSocket) -> None:
    manager = MarketDataManager(DummyBroker(), ws)
    manager._readiness_requirements = {"spot": "NSE:NIFTY", "futures": "NFO:NIFTY26MAYFUT", "options": []}  # noqa: SLF001
    manager._tracked_symbols.update({"NFO:NIFTY26MAYFUT"})  # noqa: SLF001
    manager._subscribers["NFO:NIFTY26MAYFUT"].add(lambda *_a: None)  # noqa: SLF001
    manager._latest_ticks["NFO:NIFTY26MAYFUT"] = {"ltp": 1.0}  # noqa: SLF001
    manager._tick_cache["NFO:NIFTY26MAYFUT"] = {"ltp": 1.0}  # noqa: SLF001
    manager._history["NFO:NIFTY26MAYFUT"].append({"ltp": 1.0})  # noqa: SLF001
    manager._ohlc[manager._bar_symbol_key("NFO:NIFTY26MAYFUT")].append({"close": 1.0})  # noqa: SLF001
    manager.rotate_active_nifty_future_context("NFO:NIFTY26MAYFUT", "NFO:NIFTY26JUNFUT", reason="test")
    assert "NFO:NIFTY26MAYFUT" not in manager._tracked_symbols  # noqa: SLF001
    assert "NFO:NIFTY26MAYFUT" not in manager._subscribers  # noqa: SLF001
    assert "NFO:NIFTY26MAYFUT" not in manager._latest_ticks  # noqa: SLF001
    assert "NFO:NIFTY26MAYFUT" not in manager._tick_cache  # noqa: SLF001
    assert "NFO:NIFTY26MAYFUT" not in manager._history  # noqa: SLF001
    assert manager._readiness_requirements["futures"] == "NFO:NIFTY26JUNFUT"  # noqa: SLF001
    assert "NFO:NIFTY26JUNFUT" in manager._tracked_symbols  # noqa: SLF001


def test_pull_quote_expired_future_is_suppressed_without_broker_call(ws: DummyWebSocket) -> None:
    class Resolver:
        def list_instruments(self, *_a: Any, **_k: Any) -> list[dict[str, Any]]:
            return [{"exchange": "NFO", "instrument_type": "FUT", "name": "NIFTY", "tradingsymbol": "NIFTY26JUNFUT", "expiry": date(2026, 6, 25)}]

    class FailBroker(DummyBroker):
        def get_quote(self, symbol: str) -> dict[str, Any]:
            raise AssertionError(f"broker.get_quote must not be called for {symbol}")

    manager = MarketDataManager(FailBroker(), ws, resolver=Resolver())
    out = manager.pull_quote("NFO:NIFTY26MAYFUT")
    assert out == {}


def test_repeated_expired_future_pull_quote_rotates_only_once(ws: DummyWebSocket, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = MarketDataManager(DummyBroker(), ws)
    may = "NFO:NIFTY26MAYFUT"
    jun = "NFO:NIFTY26JUNFUT"
    manager._token_by_symbol.update({may: 111, jun: 222})  # noqa: SLF001
    manager._symbol_to_token.update({may: 111, jun: 222})  # noqa: SLF001
    manager._symbol_by_token.update({111: may, 222: jun})  # noqa: SLF001
    manager._token_to_symbol.update({111: may, 222: jun})  # noqa: SLF001
    manager._desired_tokens.update({111, 222})  # noqa: SLF001
    monkeypatch.setattr(manager, "_is_nifty_future_symbol_expired", lambda sym: str(sym).upper() == may)
    monkeypatch.setattr(manager, "get_active_nifty_future_symbol_cached", lambda *args, **kwargs: jun)
    calls: list[tuple[str | None, str, str]] = []

    def _rotate(old_symbol: str | None, new_symbol: str, *, reason: str, trace_id: str | None = None) -> None:
        calls.append((old_symbol, new_symbol, reason))
        manager._suspended_context_symbols.add(may)  # noqa: SLF001

    monkeypatch.setattr(manager, "rotate_active_nifty_future_context", _rotate)
    first = manager.pull_quote(may)
    second = manager.pull_quote(may)
    assert first == {}
    assert second == {}
    assert len(calls) == 1
    assert 111 not in manager.desired_tokens_snapshot()
    assert 222 in manager.desired_tokens_snapshot()


def test_symbols_for_poll_excludes_expired_futures(monkeypatch: pytest.MonkeyPatch, ws: DummyWebSocket) -> None:
    manager = MarketDataManager(DummyBroker(), ws)
    manager._tracked_symbols.update({"NFO:NIFTY26MAYFUT", "NFO:NIFTY26JUNFUT"})  # noqa: SLF001
    monkeypatch.setattr(manager, "_is_ws_healthy", lambda: False)
    monkeypatch.setattr(manager, "_poll_candidates", lambda _now: ["NFO:NIFTY26MAYFUT", "NFO:NIFTY26JUNFUT"])
    monkeypatch.setattr(
        manager,
        "resolve_active_nifty_future_symbol",
        lambda now=None: "NFO:NIFTY26JUNFUT",
    )
    symbols = manager._symbols_for_poll()  # noqa: SLF001
    assert "NFO:NIFTY26MAYFUT" not in symbols
    assert "NFO:NIFTY26JUNFUT" in symbols


def test_get_ltp_expired_future_returns_none_and_never_dict(ws: DummyWebSocket) -> None:
    class Resolver:
        def list_instruments(self, *_a: Any, **_k: Any) -> list[dict[str, Any]]:
            return [{"exchange": "NFO", "instrument_type": "FUT", "name": "NIFTY", "tradingsymbol": "NIFTY26JUNFUT", "expiry": date(2026, 6, 25)}]

    manager = MarketDataManager(DummyBroker(), ws, resolver=Resolver())
    out = manager.get_ltp("NFO:NIFTY26MAYFUT")
    assert out is None
    assert not isinstance(out, dict)


def test_rotate_future_updates_readiness_when_initially_empty(ws: DummyWebSocket) -> None:
    manager = MarketDataManager(DummyBroker(), ws)
    manager._readiness_requirements = {}  # noqa: SLF001
    manager.rotate_active_nifty_future_context("NFO:NIFTY26MAYFUT", "NFO:NIFTY26JUNFUT", reason="test")
    assert manager._readiness_requirements.get("futures") == "NFO:NIFTY26JUNFUT"  # noqa: SLF001


def test_set_active_contract_basket_replaces_desired_tokens(ws: DummyWebSocket) -> None:
    manager = MarketDataManager(None, ws)
    first = {
        "selected_ce": "NFO:NIFTY26JUN25000CE",
        "selected_pe": "NFO:NIFTY26JUN25000PE",
        "option_symbols": ["NFO:NIFTY26JUN25000CE", "NFO:NIFTY26JUN25000PE"],
        "all_symbols": ["NFO:NIFTY26JUN25000CE", "NFO:NIFTY26JUN25000PE"],
        "all_tokens": [11, 12],
        "token_by_symbol": {"NFO:NIFTY26JUN25000CE": 11, "NFO:NIFTY26JUN25000PE": 12},
    }
    second = {
        "selected_ce": "NFO:NIFTY26JUN25100CE",
        "selected_pe": "NFO:NIFTY26JUN25100PE",
        "option_symbols": ["NFO:NIFTY26JUN25100CE", "NFO:NIFTY26JUN25100PE"],
        "all_symbols": ["NFO:NIFTY26JUN25100CE", "NFO:NIFTY26JUN25100PE"],
        "all_tokens": [21, 22],
        "token_by_symbol": {"NFO:NIFTY26JUN25100CE": 21, "NFO:NIFTY26JUN25100PE": 22},
    }

    manager.set_active_contract_basket(first)
    assert manager._desired_tokens == {11, 12}  # noqa: SLF001

    manager.set_active_contract_basket(second)

    assert manager._desired_tokens == {21, 22}  # noqa: SLF001
    assert 11 not in manager._symbol_by_token  # noqa: SLF001
    assert 12 not in manager._token_to_symbol  # noqa: SLF001


def test_selected_option_ltp_only_and_insufficient_bars_not_live_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIN_INDICATOR_BARS", "20")
    monkeypatch.setenv("HISTORY_REQUIRED_BARS", "30")
    monkeypatch.setenv("HYDRATION_SELECTED_OPTION_MIN_BARS", "30")
    manager = MarketDataManager(None, None)
    symbol = "NFO:NIFTY26JUN25000CE"
    basket = {
        "selected_ce": symbol,
        "selected_pe": "NFO:NIFTY26JUN25000PE",
        "option_symbols": [symbol],
        "all_symbols": [symbol],
        "all_tokens": [11],
        "token_by_symbol": {symbol: 11},
    }
    quote = {"symbol": symbol, "ltp": 100.0, "last_price": 100.0, "timestamp": time.time(), "received_at": time.time(), "source": "poll"}
    manager.get_latest_tick = lambda _sym: quote  # type: ignore[method-assign]
    manager.get_quote = lambda _sym: quote  # type: ignore[method-assign]
    manager.get_ohlc_bars = lambda _sym, *a, **k: [{"close": 100.0}] * 5  # type: ignore[method-assign]

    report = manager.hydrate_active_contract_basket(basket)
    entry = report["symbols"][symbol]

    assert report["hard_ready"] is False
    assert entry["quote_ready"] is True
    assert entry["tradable_quote_ready"] is False
    assert entry["ohlc_ready"] is False
    assert entry["bars_count"] == 5


def test_process_poll_quote_option_volume_uses_cumulative_delta() -> None:
    manager = MarketDataManager(None, None)
    emitted: list[dict[str, Any]] = []
    manager._emit_poll_candle = emitted.append  # type: ignore[method-assign]
    symbol = "NFO:NIFTY26JUN25000CE"

    manager._process_poll_quote(symbol, {"timestamp": 60.0, "last_price": 100.0, "volume_traded_today": 100})
    manager._process_poll_quote(symbol, {"timestamp": 70.0, "last_price": 101.0, "volume_traded_today": 130})
    manager._process_poll_quote(symbol, {"timestamp": 120.0, "last_price": 102.0, "volume_traded_today": 180})

    assert emitted
    assert emitted[0]["volume"] == 80
    assert emitted[0]["volume_source"] == "cumulative_delta"
