from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import nifty_scalper_bot.data.data_hub as data_hub_module
from nifty_scalper_bot.data.data_hub import DataHub
from nifty_scalper_bot.data.market_data_manager import MarketDataManager


class DummyBroker:
    pass


def _unexpired_option_symbol(strike: int = 25000, side: str = "CE") -> str:
    """Build an NFO option symbol with an expiry that is always in the future.

    A previously hardcoded "NIFTY26AUG25000CE" fixture expired mid-run once
    the calendar passed 28-Aug-2026, silently disabling the IV-derivation
    path this test exercises (``_parse_option_symbol`` -> ``ttm_years <= 0``
    -> early return). Deriving the month/year from "now" keeps the contract
    perpetually unexpired.
    """
    expiry = datetime.now(timezone.utc) + timedelta(days=45)
    return f"NFO:NIFTY{expiry:%y%b}{strike}{side}".upper()


def test_sync_callback_none_no_error() -> None:
    mdm = MarketDataManager(DummyBroker(), websocket=None)
    seen: list[dict[str, Any]] = []

    def callback(tick: dict[str, Any]) -> None:
        seen.append(tick)
        return None

    mdm.subscribe("NSE:NIFTY", callback)
    mdm._emit_tick(
        "NSE:NIFTY", {"symbol": "NSE:NIFTY", "ltp": 10.0}, source="ws"
    )  # noqa: SLF001
    assert len(seen) == 1


def test_sync_callback_custom_awaitable_no_type_error() -> None:
    mdm = MarketDataManager(DummyBroker(), websocket=None)

    class CustomAwaitable:
        def __await__(self):
            if False:
                yield None
            return None

    def callback(_tick: dict[str, Any]) -> CustomAwaitable:
        return CustomAwaitable()

    mdm.subscribe("NSE:NIFTY", callback)
    mdm._emit_tick(
        "NSE:NIFTY", {"symbol": "NSE:NIFTY", "ltp": 10.0}, source="ws"
    )  # noqa: SLF001


def test_async_callback_scheduled_on_main_loop() -> None:
    mdm = MarketDataManager(DummyBroker(), websocket=None)
    ran = {"ok": False}

    async def callback(_tick: dict[str, Any]) -> None:
        ran["ok"] = True

    async def runner() -> None:
        mdm.set_event_loop(asyncio.get_running_loop())
        mdm.subscribe("NSE:NIFTY", callback)
        mdm._emit_tick(
            "NSE:NIFTY", {"symbol": "NSE:NIFTY", "ltp": 11.0}, source="ws"
        )  # noqa: SLF001
        await asyncio.sleep(0.01)

    asyncio.run(runner())
    assert ran["ok"] is True


def test_datahub_ingest_tick_sync_returns_none() -> None:
    mdm = MarketDataManager(DummyBroker(), websocket=None)
    hub = DataHub(mdm)
    result = hub.ingest_tick_sync({"symbol": "NSE:NIFTY", "ltp": 1.0})
    assert result is None


def test_tick_ingestion_defers_missing_option_analytics(monkeypatch) -> None:
    hub = DataHub(MarketDataManager(DummyBroker(), websocket=None))
    symbol = _unexpired_option_symbol()
    calls: list[float] = []
    monkeypatch.setattr(hub, "get_latest_price", lambda _symbol: 25_000.0)
    monkeypatch.setattr(
        data_hub_module,
        "implied_volatility",
        lambda **_kwargs: calls.append(1.0) or 0.2,
    )
    monkeypatch.setattr(
        data_hub_module,
        "black_scholes_greeks",
        lambda **_kwargs: {"delta": 0.5},
    )

    hub.ingest_tick_sync(
        {
            "symbol": symbol,
            "instrument_token": 123,
            "ltp": 100.0,
            "source": "ws",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )

    assert calls == []
    monkeypatch.setattr(hub, "get_quote", lambda _symbol: hub._quotes[symbol])
    assert hub.get_iv(symbol) == 0.2
    assert calls == [1.0]
    assert hub.get_greeks(symbol) == {"delta": 0.5}


def test_tick_ingestion_still_caches_broker_option_metrics(monkeypatch) -> None:
    hub = DataHub(MarketDataManager(DummyBroker(), websocket=None))
    symbol = "NFO:NIFTY26AUG25000CE"
    monkeypatch.setattr(
        data_hub_module,
        "implied_volatility",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not derive")),
    )

    hub.ingest_tick_sync(
        {
            "symbol": symbol,
            "instrument_token": 123,
            "ltp": 100.0,
            "iv": 0.25,
            "greeks": {"delta": 0.6},
            "oi": 12_345,
            "source": "ws",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )

    assert hub.get_iv(symbol) == 0.25
    assert hub.get_greeks(symbol) == {"delta": 0.6}
    assert hub.get_oi(symbol) == 12_345


def test_datahub_repairs_stale_mdm_delegate_subscription() -> None:
    """Stale DataHub bookkeeping must not sever the live MDM -> DataHub bridge."""
    mdm = MarketDataManager(DummyBroker(), websocket=None)
    hub = DataHub(mdm)
    symbol = "NFO:NIFTY26AUG25000CE"
    token = 123
    mdm.register_symbol(symbol, token)

    # Reproduce production drift: DataHub remembers a delegate subscription,
    # while the actual MDM callback set no longer contains DataHub.ingest_tick_sync.
    hub._mdm_subscribed_symbols.add(symbol)
    assert hub.ingest_tick_sync not in mdm._subscribers[mdm._canonical_symbol(symbol)]

    seen: list[dict[str, Any]] = []

    def runner_callback(tick: dict[str, Any]) -> None:
        seen.append(tick)

    hub.subscribe_ticks(symbol, runner_callback, token=token, force_live=True)

    assert hub.ingest_tick_sync in mdm._subscribers[mdm._canonical_symbol(symbol)]
    mdm._emit_tick(
        symbol,
        {
            "symbol": symbol,
            "instrument_token": token,
            "ltp": 101.0,
            "bid": 100.9,
            "ask": 101.1,
        },
        source="ws",
    )
    assert seen
    assert seen[-1]["ltp"] == 101.0


def test_ws_tick_still_reaches_subscribers_when_cache_rejects_older_event() -> None:
    """Live WS ticks must reach DataHub even if MDM cache kept a newer timestamp.

    Startup replay can leave _latest_ticks with a wall-clock/synthetic event
    time ahead of subsequent exchange timestamps. Readiness still looks fresh
    via _last_valid_live_tick_mono, but the old `_store_tick` reject returned
    before subscriber fanout — the replay-vs-live divergence.
    """
    mdm = MarketDataManager(DummyBroker(), websocket=None)
    symbol = "NFO:NIFTY26AUG25000CE"
    token = 123
    mdm.register_symbol(symbol, token)
    mdm._selected_ce_symbol = symbol
    seen: list[dict[str, Any]] = []

    def callback(tick: dict[str, Any]) -> None:
        seen.append(tick)

    mdm.subscribe(symbol, callback)
    future = datetime.now(timezone.utc).timestamp() + 60.0
    mdm._latest_ticks[mdm._canonical_symbol(symbol)] = {
        "symbol": symbol,
        "instrument_token": token,
        "ltp": 100.0,
        "timestamp": future,
        "exchange_timestamp": future,
        "source": "ws",
        "depth": {"buy": [{"price": 99.9}], "sell": [{"price": 100.1}]},
        "bid": 99.9,
        "ask": 100.1,
    }
    exchange_ts = datetime.now(timezone.utc).timestamp()
    mdm._emit_tick(
        symbol,
        {
            "symbol": symbol,
            "instrument_token": token,
            "ltp": 101.5,
            "bid": 101.4,
            "ask": 101.6,
            "timestamp": exchange_ts,
            "exchange_timestamp": exchange_ts,
            "source": "ws",
            "depth": {"buy": [{"price": 101.4}], "sell": [{"price": 101.6}]},
        },
        source="ws",
    )
    assert (
        seen
    ), "current-generation WS tick must fan out even if cache write is rejected"
    assert seen[-1]["ltp"] == 101.5
    assert mdm._mdm_selected_tick_count >= 1
