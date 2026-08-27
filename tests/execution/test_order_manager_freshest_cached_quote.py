from __future__ import annotations

import time

from nifty_scalper_bot.execution.order_manager import OrderManager, TradePlan


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


def _quote(
    symbol: str,
    *,
    marker: str,
    age_s: float,
    executable: bool = True,
) -> dict[str, object]:
    quote: dict[str, object] = {
        "symbol": symbol,
        "ltp": 100.0,
        "received_at": time.time() - age_s,
        "marker": marker,
    }
    if executable:
        quote.update(
            {
                "bid": 99.90,
                "ask": 100.10,
                "bid_quantity": 65,
                "ask_quantity": 65,
            }
        )
    return quote


def _depth_quote(
    symbol: str,
    *,
    marker: str = "ws_full_depth",
    bid_qty: int = 130,
    ask_qty: int = 130,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "ltp": 100.0,
        "received_at": time.time() - 0.02,
        "marker": marker,
        "depth_available": True,
        "depth": {
            "buy": [{"price": 99.90, "quantity": bid_qty}],
            "sell": [{"price": 100.10, "quantity": ask_qty}],
        },
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


def test_order_preflight_preserves_older_executable_quote_over_fresher_ltp() -> None:
    symbol = "NFO:NIFTY26AUG24050PE"
    manager = _manager(
        data_hub=_TickProvider(_quote(symbol, marker="executable", age_s=0.20)),
        mdm=_TickProvider(
            _quote(symbol, marker="ltp_only", age_s=0.01, executable=False)
        ),
    )

    quote = manager._get_latest_quote_safe(symbol)

    assert quote is not None
    assert quote["marker"] == "executable"


def test_order_preflight_upgrades_ltp_quote_to_executable_quote() -> None:
    symbol = "NFO:NIFTY26AUG24050PE"
    manager = _manager(
        data_hub=_TickProvider(
            _quote(symbol, marker="ltp_only", age_s=0.01, executable=False)
        ),
        mdm=_TickProvider(_quote(symbol, marker="executable", age_s=0.20)),
    )

    quote = manager._get_latest_quote_safe(symbol)

    assert quote is not None
    assert quote["marker"] == "executable"


def test_order_preflight_keeps_freshest_quote_when_only_ltp_is_available() -> None:
    symbol = "NFO:NIFTY26AUG24050PE"
    manager = _manager(
        data_hub=_TickProvider(
            _quote(symbol, marker="processed", age_s=0.20, executable=False)
        ),
        mdm=_TickProvider(
            _quote(symbol, marker="ws_fast", age_s=0.01, executable=False)
        ),
    )

    quote = manager._get_latest_quote_safe(symbol)

    assert quote is not None
    assert quote["marker"] == "ws_fast"


def test_order_preflight_extracts_zerodha_depth_as_executable_quote() -> None:
    symbol = "NFO:NIFTY26AUG24050PE"
    manager = _manager(
        data_hub=_TickProvider(_depth_quote(symbol)),
        mdm=_TickProvider(_depth_quote(symbol)),
    )

    diagnostics = manager._extract_quote_diagnostics(_depth_quote(symbol))

    assert diagnostics["bid"] == 99.90
    assert diagnostics["ask"] == 100.10
    assert diagnostics["bid_qty"] == 130
    assert diagnostics["ask_qty"] == 130
    assert diagnostics["depth_qty"] == 260
    assert diagnostics["spread_pct"] > 0


def test_live_entry_accepts_complete_zerodha_depth_quote() -> None:
    symbol = "NFO:NIFTY26AUG24050PE"
    manager = _manager(
        data_hub=_TickProvider(_depth_quote(symbol)),
        mdm=_TickProvider(_depth_quote(symbol)),
    )
    plan = TradePlan(
        symbol=symbol,
        side="BUY",
        quantity=65,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        max_spread_pct=5.0,
    )

    diagnostics = manager._extract_quote_diagnostics(_depth_quote(symbol))
    rejection = manager._validate_live_entry_quote(plan, diagnostics)

    assert rejection is None


def test_live_entry_keeps_depth_quantity_guard_for_zerodha_depth_quote() -> None:
    symbol = "NFO:NIFTY26AUG24050PE"
    quote = _depth_quote(symbol, ask_qty=25)
    manager = _manager(data_hub=_TickProvider(quote), mdm=_TickProvider(quote))
    plan = TradePlan(
        symbol=symbol,
        side="BUY",
        quantity=65,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        max_spread_pct=5.0,
    )

    diagnostics = manager._extract_quote_diagnostics(quote)
    rejection = manager._validate_live_entry_quote(plan, diagnostics)

    assert rejection is not None
    assert rejection.reason == "entry_executable_depth_insufficient"
