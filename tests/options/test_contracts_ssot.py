from __future__ import annotations

from datetime import date

from nifty_scalper_bot.options.contracts import OptionsContractStore


class _IM:
    def __init__(self) -> None:
        self.calls = []

    def get_active_nifty_contracts(self, spot_price: float, **kwargs):
        self.calls.append((spot_price, kwargs))
        return {
            "spot_symbol": "NSE:NIFTY",
            "futures_symbol": "NFO:NIFTY26JUNFUT",
            "option_symbols": ("NFO:NIFTY26JUN25000CE", "NFO:NIFTY26JUN25000PE"),
            "selected_ce": "NFO:NIFTY26JUN25000CE",
            "selected_pe": "NFO:NIFTY26JUN25000PE",
            "option_expiry": date(2026, 6, 25),
            "atm_strike": 25000,
            "futures_token": 900,
            "spot_token": 256265,
        }

    def is_loaded(self):
        return True


def test_options_contract_store_delegates_to_instrument_manager_basket():
    im = _IM()
    store = OptionsContractStore(im)

    symbols = store.get_trading_universe(25001.0, include_spot=True, include_futures=False)

    assert symbols == ["NSE:NIFTY", "NFO:NIFTY26JUN25000CE", "NFO:NIFTY26JUN25000PE"]
    assert im.calls
    assert im.calls[0][1]["include_future"] is False
