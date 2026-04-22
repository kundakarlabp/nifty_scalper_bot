from __future__ import annotations

from datetime import date

from nifty_scalper_bot.core.contract_selector import get_atm_contracts


class _Kite:
    def __init__(self) -> None:
        self.instruments_calls = 0

    def instruments(self, exchange: str):
        self.instruments_calls += 1
        if exchange != "NFO":
            return []
        return []


def _rows() -> list[dict]:
    exp = date.today()
    return [
        {
            "name": "NIFTY",
            "instrument_type": "CE",
            "expiry": exp,
            "strike": 24500.0,
            "instrument_token": 111,
            "tradingsymbol": "NIFTYXX24500CE",
            "exchange": "NFO",
        },
        {
            "name": "NIFTY",
            "instrument_type": "PE",
            "expiry": exp,
            "strike": 24500.0,
            "instrument_token": 222,
            "tradingsymbol": "NIFTYXX24500PE",
            "exchange": "NFO",
        },
    ]


def test_contract_selector_prefers_preloaded_cache() -> None:
    kite = _Kite()
    contracts = get_atm_contracts(
        kite,
        underlying_price=24510.0,
        strike_band=100,
        strike_step=50,
        preloaded_instruments=_rows(),
    )
    assert len(contracts) == 2
    assert kite.instruments_calls == 0
