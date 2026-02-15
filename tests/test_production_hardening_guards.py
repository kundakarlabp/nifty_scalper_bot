from __future__ import annotations

from typing import Any

from nifty_scalper_bot.data.data_hub import DataHub
from nifty_scalper_bot.streaming.websocket_manager import WebSocketManager
from nifty_scalper_bot.utils.symbols import canonical, is_canonical_symbol


class _MDM:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def subscribe(self, symbol: str, callback: Any) -> None:
        self.calls.append(symbol)


class _Resolver:
    pass


def test_symbol_canonical_validation() -> None:
    assert canonical("nifty 50") == "NSE:NIFTY"
    assert is_canonical_symbol("NSE:NIFTY 50")
    assert not is_canonical_symbol("NIFTY50")


def test_data_hub_subscribe_deduplicates() -> None:
    mdm = _MDM()
    hub = DataHub(market_data_manager=mdm, instrument_resolver=_Resolver())

    hub.subscribe_ticks("NSE:NIFTY 50", lambda _: None)
    hub.subscribe_ticks("NSE:NIFTY 50", lambda _: None)
    hub.subscribe_ticks("256265", lambda _: None)

    assert mdm.calls == ["NSE:NIFTY"]


def test_data_hub_historical_source_always_fresh() -> None:
    hub = DataHub(market_data_manager=None, instrument_resolver=_Resolver())
    hub.store_quote(
        "NSE:NIFTY 50",
        {"symbol": "NSE:NIFTY 50", "timestamp": 1, "last_price": 1.0},
        source="historical",
    )

    ok, meta = hub.is_fresh("NSE:NIFTY 50")

    assert ok is True
    assert meta.get("source") == "historical"


def test_websocket_manager_owns_single_ticker_instance() -> None:
    mgr = WebSocketManager('k1', 'token', on_tick=lambda _: None)
    assert mgr.ticker is mgr.ticker
