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
