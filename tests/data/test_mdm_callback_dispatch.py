from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import nifty_scalper_bot.data.data_hub as data_hub_module
from nifty_scalper_bot.data.data_hub import DataHub
from nifty_scalper_bot.data.market_data_manager import MarketDataManager


class DummyBroker:
    pass


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
    symbol = "NFO:NIFTY26AUG25000CE"
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
