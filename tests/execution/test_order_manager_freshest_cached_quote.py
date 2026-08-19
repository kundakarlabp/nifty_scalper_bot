from __future__ import annotations

import time

from nifty_scalper_bot.execution.order_manager import OrderManager


class _TickProvider:
    def __init__(self, quote: dict[str, object]) -> None:
        self.quote = dict(quote)

    def get_latest_tick(self, _symbol: str) -> dict[str, object]:
        return dict(self.quote)


def _manager(*, data_hub: _TickProvider, mdm: _TickProvider) -> OrderManager:
    manager = object.__new__(OrderManager)
    manager._data_hub = data_hub
    manager.data_hub = None
    manager._market_data = None
    manager._market_data_manager = mdm
    manager.market_data_manager = None
    return manager


def _quote(symbol: str, *, marker: str, age_s: float) -> dict[str, object]:
    return {
        "symbol": symbol,
        "ltp": 100.0,
        "bid": 99.90,
        "ask": 100.10,
        "received_at": time.time() - age_s,
        "marker": marker,
    }


def test_order_preflight_uses_freshest_cached_provider_quote() -> None:
    symbol = "NFO:NIFTY26AUG24050PE"
    manager = _manager(
        data_hub=_TickProvider(_quote(symbol, marker="processed", age_s=2.4)),
        mdm=_TickProvider(_quote(symbol, marker="ws_fast", age_s=0.03)),
    )

    quote = manager._get_latest_quote_safe(symbol)

    assert quote is not None
    assert quote["marker"] == "ws_fast"


def test_order_preflight_keeps_first_quote_when_it_is_already_freshest() -> None:
    symbol = "NFO:NIFTY26AUG24050PE"
    manager = _manager(
        data_hub=_TickProvider(_quote(symbol, marker="processed", age_s=0.02)),
        mdm=_TickProvider(_quote(symbol, marker="ws_fast", age_s=0.20)),
    )

    quote = manager._get_latest_quote_safe(symbol)

    assert quote is not None
    assert quote["marker"] == "processed"
