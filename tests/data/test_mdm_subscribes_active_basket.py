from __future__ import annotations

from datetime import date

from nifty_scalper_bot.core.instrument_manager import ActiveContractBasket
from nifty_scalper_bot.data.market_data_manager import MarketDataManager


class WS:
    def __init__(self):
        self.tokens = []
    def set_tokens(self, tokens):
        self.tokens = list(tokens)
        return True


def basket():
    return ActiveContractBasket(
        spot_symbol="NSE:NIFTY", spot_token=256265,
        futures_symbol=None, futures_token=None, futures_expiry=None,
        selected_ce="NFO:NIFTY25000CE", selected_ce_token=11,
        selected_pe="NFO:NIFTY25000PE", selected_pe_token=12,
        option_expiry=date.today(), atm_strike=25000,
        option_symbols=("NFO:NIFTY25000CE", "NFO:NIFTY25000PE"),
        option_tokens=(11, 12),
        all_symbols=("NSE:NIFTY", "NFO:NIFTY25000CE", "NFO:NIFTY25000PE"),
        all_tokens=(256265, 11, 12),
        token_by_symbol={"NSE:NIFTY": 256265, "NFO:NIFTY25000CE": 11, "NFO:NIFTY25000PE": 12},
        symbol_by_token={256265: "NSE:NIFTY", 11: "NFO:NIFTY25000CE", 12: "NFO:NIFTY25000PE"},
        metadata={},
    )


def test_mdm_set_active_contract_basket_subscribes_exact_tokens_and_reports_missing_oi_soft():
    ws = WS()
    mdm = MarketDataManager(websocket=ws)
    b = basket()
    mdm.set_active_contract_basket(b)
    assert set(ws.tokens) == set(b.all_tokens)
    assert mdm.resolve_symbol_token("NFO:NIFTY25000CE") == 11
    report = mdm.hydrate_active_contract_basket(b)
    assert set(report["symbols"]) == set(b.all_symbols)
    assert report["symbols"]["NFO:NIFTY25000CE"]["oi_ready"] is False
    assert report["hard_ready"] is False


def test_mdm_hydration_never_classifies_fut_as_tradable_option(caplog):
    ws = WS()
    mdm = MarketDataManager(websocket=ws)
    b = basket()
    payload = {
        "spot_symbol": b.spot_symbol,
        "spot_token": b.spot_token,
        "futures_symbol": None,
        "futures_token": None,
        "selected_ce": b.selected_ce,
        "selected_pe": b.selected_pe,
        "option_symbols": list(b.option_symbols),
        "option_tokens": list(b.option_tokens),
        "all_symbols": ["NSE:NIFTY", "NFO:NIFTY26JUNFUT", *b.option_symbols],
        "all_tokens": [256265, 222, *b.option_tokens],
        "token_by_symbol": {"NSE:NIFTY": 256265, "NFO:NIFTY26JUNFUT": 222, b.selected_ce: 11, b.selected_pe: 12},
        "symbol_by_token": {256265: "NSE:NIFTY", 222: "NFO:NIFTY26JUNFUT", 11: b.selected_ce, 12: b.selected_pe},
        "atm_strike": b.atm_strike,
    }
    mdm.set_active_contract_basket(payload)
    report = mdm.hydrate_active_contract_basket(payload)
    assert report["symbols"]["NFO:NIFTY26JUNFUT"]["role"] == "futures_context"
    assert report["symbols"]["NFO:NIFTY26JUNFUT"]["ohlc_ready"] is True
    assert "NFO:NIFTY26JUNFUT:quote_missing" not in report["missing"]
    assert "MDM_BASKET_ROLE_INCONSISTENCY" in caplog.text


def test_mdm_hydrate_counts_list_and_dataframe_bars(monkeypatch):
    import pandas as pd

    mdm = MarketDataManager(websocket=WS())
    b = basket()
    mdm._min_required_bars = 2
    mdm.set_active_contract_basket(b)
    quotes = {sym: {"ltp": 100.0, "bid": 99.0, "ask": 101.0, "source": "test"} for sym in b.all_symbols}
    monkeypatch.setattr(mdm, "get_latest_tick", lambda sym: quotes.get(sym))
    monkeypatch.setattr(mdm, "get_quote", lambda sym: quotes.get(sym))

    def bars(sym):
        if sym == b.selected_ce:
            return [{"close": 1}, {"close": 2}]
        if sym == b.selected_pe:
            return pd.DataFrame([{"close": 1}, {"close": 2}])
        return []

    monkeypatch.setattr(mdm, "get_ohlc_bars", bars)
    report = mdm.hydrate_active_contract_basket(b)
    assert report["hard_ready"] is True
    assert report["symbols"][b.selected_ce]["bars_count"] == 2
    assert report["symbols"][b.selected_pe]["bars_count"] == 2


def test_mdm_selected_missing_ohlc_triggers_reseed_attempt(monkeypatch):
    mdm = MarketDataManager(websocket=WS())
    b = basket()
    mdm._min_required_bars = 3
    mdm.set_active_contract_basket(b)
    quotes = {sym: {"ltp": 100.0, "bid": 99.0, "ask": 101.0, "source": "test"} for sym in b.all_symbols}
    monkeypatch.setattr(mdm, "get_latest_tick", lambda sym: quotes.get(sym))
    monkeypatch.setattr(mdm, "get_quote", lambda sym: quotes.get(sym))
    attempts = []

    def hydrate_symbol_history(symbol, **kwargs):
        attempts.append((symbol, kwargs))
        return []

    monkeypatch.setattr(mdm, "hydrate_symbol_history", hydrate_symbol_history)
    monkeypatch.setattr(mdm, "get_ohlc_bars", lambda sym: [])
    report = mdm.hydrate_active_contract_basket(b)
    assert b.selected_ce in [item[0] for item in attempts]
    assert b.selected_pe in [item[0] for item in attempts]
    assert report["hard_ready"] is False
    assert f"{b.selected_ce}:ohlc_insufficient" in report["missing"]
