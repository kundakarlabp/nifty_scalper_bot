from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from nifty_scalper_bot.config.settings import LiquiditySettings, SelectorSettings
from nifty_scalper_bot.options import strike_selector as strike_module
from nifty_scalper_bot.options.strike_selector import StrikeSelector


class Hub:
    def __init__(self, basket: dict[str, Any] | None):
        self.basket = basket
        self.option_chain_called = False
    def get_active_contract_basket(self):
        return self.basket
    def get_quote(self, symbol: str, allow_pull: bool = False):
        return {"ltp": 123.45, "bid": 123.0, "ask": 124.0, "source": "test"}
    def get_option_chain(self, *args, **kwargs):
        self.option_chain_called = True
        raise AssertionError("option chain fallback must not be called in LIVE")


def _selector(hub: Hub) -> StrikeSelector:
    return StrikeSelector(
        data_hub=hub,
        selector_settings=SelectorSettings(mode="ATM", expiry="weekly"),
        liquidity_settings=LiquiditySettings(min_oi=0, max_spread_pct=99, min_trades=0, min_top3_depth=0),
    )


def _basket() -> dict[str, Any]:
    return {
        "selected_ce": "NFO:NIFTY26JUN25000CE",
        "selected_ce_token": 111,
        "selected_pe": "NFO:NIFTY26JUN25000PE",
        "selected_pe_token": 112,
        "option_expiry": date.today(),
        "atm_strike": 25000,
        "token_by_symbol": {"NFO:NIFTY26JUN25000CE": 111, "NFO:NIFTY26JUN25000PE": 112},
    }


@pytest.mark.parametrize(("option_type", "expected", "token"), [("CE", "NFO:NIFTY26JUN25000CE", 111), ("PE", "NFO:NIFTY26JUN25000PE", 112)])
def test_live_strike_selector_returns_active_basket_selection(monkeypatch, option_type, expected, token):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setattr(strike_module, "OptionResolver", lambda *a, **k: (_ for _ in ()).throw(AssertionError("OptionResolver called")))
    hub = Hub(_basket())
    selected = _selector(hub).select_contract(
        underlying="NIFTY", side="BUY", option_type=option_type, underlying_price=25001.0
    )
    assert selected is not None
    assert selected.symbol == expected
    assert selected.metadata["instrument_token"] == token
    assert selected.metadata["source"] == "active_contract_basket"
    assert hub.option_chain_called is False


def test_live_strike_selector_missing_basket_returns_none(monkeypatch, caplog):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    selected = _selector(Hub(None)).select_contract(
        underlying="NIFTY", side="BUY", option_type="CE", underlying_price=25001.0
    )
    assert selected is None
    assert "STRIKE_SELECTOR_ACTIVE_BASKET_MISSING" in caplog.text


def test_non_live_option_chain_fallback_requires_env(monkeypatch):
    monkeypatch.delenv("EXECUTION_MODE", raising=False)
    monkeypatch.delenv("MODE", raising=False)
    monkeypatch.setenv("OPTION_CHAIN_SELECTOR_FALLBACK_ENABLED", "true")

    class ChainHub(Hub):
        def __init__(self):
            super().__init__(None)
        def get_option_chain(self, *args, **kwargs):
            self.option_chain_called = True
            return [{
                "tradingsymbol": "NIFTY26JUN25000CE",
                "strike": 25000,
                "expiry": date.today().isoformat(),
                "ltp": 100,
                "bid": 99,
                "ask": 101,
                "open_interest": 1000,
                "trades": 10,
                "depth": {"buy": [{"quantity": 100}], "sell": [{"quantity": 100}]},
            }]

    hub = ChainHub()
    selected = _selector(hub).select_contract(
        underlying="NIFTY", side="BUY", option_type="CE", underlying_price=25001.0
    )
    assert hub.option_chain_called is True
    assert selected is not None
