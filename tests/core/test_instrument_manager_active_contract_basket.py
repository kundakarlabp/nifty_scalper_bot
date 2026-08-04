from __future__ import annotations

from datetime import date, timedelta

import pytest

from nifty_scalper_bot.core.instrument_manager import ActiveContractBasket, InstrumentManager


class KiteDump:
    def __init__(self, rows):
        self.rows = rows

    def instruments(self, exchange):
        assert exchange == "NFO"
        return list(self.rows)


def _row(token, symbol, typ, expiry, strike=0):
    return {
        "instrument_token": token,
        "tradingsymbol": symbol,
        "exchange": "NFO",
        "name": "NIFTY",
        "instrument_type": typ,
        "expiry": expiry,
        "strike": strike,
        "lot_size": 65,
    }


def _dump(include_future=True, include_atm=True):
    today = date.today()
    weekly = today + timedelta(days=2)
    monthly = today + timedelta(days=25)
    rows = [_row(1, "NIFTYOLD", "FUT", today - timedelta(days=1))]
    if include_future:
        rows.append(_row(2, "NIFTYCURFUT", "FUT", monthly))
    token = 10
    for expiry, prefix in ((weekly, "W"), (monthly, "M")):
        for strike in (24900, 24950, 25000, 25050, 25100):
            if not include_atm and expiry == weekly and strike == 25000:
                continue
            for side in ("CE", "PE"):
                rows.append(_row(token, f"NIFTY{prefix}{strike}{side}", side, expiry, strike))
                token += 1
    return rows, weekly, monthly


def test_active_contract_basket_uses_independent_option_expiry_and_maps_tokens():
    rows, weekly, monthly = _dump()
    mgr = InstrumentManager(KiteDump(rows))
    basket = mgr.get_active_nifty_contracts(25010, strikes_around_atm=1)

    assert isinstance(basket, ActiveContractBasket)
    assert basket.futures_symbol == "NFO:NIFTYCURFUT"
    assert basket.futures_expiry == monthly
    assert basket.option_expiry == weekly
    assert basket.atm_strike == 25000
    assert basket.selected_ce == "NFO:NIFTYW25000CE"
    assert basket.selected_pe == "NFO:NIFTYW25000PE"
    assert basket.all_symbols[0] == "NSE:NIFTY"
    assert basket.all_symbols[1] == "NFO:NIFTYCURFUT"
    assert set(basket.option_symbols) >= {basket.selected_ce, basket.selected_pe}
    assert dict(basket.token_by_symbol)[basket.selected_ce] == basket.selected_ce_token
    assert dict(basket.symbol_by_token)[basket.selected_pe_token] == basket.selected_pe


def test_active_contract_basket_future_missing_is_soft_but_atm_missing_is_hard():
    rows, _weekly, _monthly = _dump(include_future=False)
    mgr = InstrumentManager(KiteDump(rows))
    basket = mgr.get_active_nifty_contracts(25000)
    assert basket.futures_symbol is None
    assert basket.selected_ce.endswith("CE")

    rows, _weekly, _monthly = _dump(include_atm=False)
    mgr = InstrumentManager(KiteDump(rows))
    with pytest.raises(RuntimeError, match="ATM_PAIR_MISSING"):
        mgr.get_active_nifty_contracts(25000)


def test_select_tokens_for_universe_uses_nearest_option_expiry_not_future_expiry():
    rows, weekly, monthly = _dump()
    mgr = InstrumentManager(KiteDump(rows))
    mgr.load()
    tokens = set(mgr.select_tokens_for_universe(spot_price=25000, strikes_around_atm=0))
    weekly_tokens = {r["instrument_token"] for r in rows if r.get("expiry") == weekly and r.get("instrument_type") in {"CE", "PE"}}
    monthly_option_tokens = {r["instrument_token"] for r in rows if r.get("expiry") == monthly and r.get("instrument_type") in {"CE", "PE"}}
    assert tokens & weekly_tokens
    assert not (tokens & monthly_option_tokens)


def test_instrument_manager_uses_ist_exchange_trading_date(monkeypatch):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from nifty_scalper_bot.core import instrument_manager as im

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            assert tz == ZoneInfo("Asia/Kolkata")
            return cls(2026, 6, 2, 1, 0, tzinfo=tz)

    monkeypatch.setattr(im, "datetime", FixedDateTime)
    assert im._exchange_trading_date().isoformat() == "2026-06-02"


def test_active_contract_basket_holds_atm_inside_hysteresis_band(monkeypatch):
    rows, _weekly, _monthly = _dump()
    mgr = InstrumentManager(KiteDump(rows))
    monkeypatch.setenv("ATM_STRIKE_HYSTERESIS_POINTS", "5")

    assert mgr.get_active_nifty_contracts(25010).atm_strike == 25000
    assert mgr.get_active_nifty_contracts(25024).atm_strike == 25000
    assert mgr.get_active_nifty_contracts(25026).atm_strike == 25000
    assert mgr.get_active_nifty_contracts(25023).atm_strike == 25000


def test_active_contract_basket_shifts_once_after_confirmed_crossing(monkeypatch):
    rows, _weekly, _monthly = _dump()
    mgr = InstrumentManager(KiteDump(rows))
    monkeypatch.setenv("ATM_STRIKE_HYSTERESIS_POINTS", "5")

    assert mgr.get_active_nifty_contracts(25010).atm_strike == 25000
    assert mgr.get_active_nifty_contracts(25030).atm_strike == 25050
    assert mgr.get_active_nifty_contracts(25024).atm_strike == 25050
    assert mgr.get_active_nifty_contracts(25020).atm_strike == 25000


def test_hysteresis_does_not_block_option_or_future_rollover(monkeypatch):
    from nifty_scalper_bot.core import instrument_manager as im

    rows, weekly, monthly = _dump()
    rows.append(_row(3, "NIFTYWEEKFUT", "FUT", weekly))
    mgr = InstrumentManager(KiteDump(rows))
    monkeypatch.setenv("ATM_STRIKE_HYSTERESIS_POINTS", "5")
    trading_day = {"value": weekly - timedelta(days=1)}
    monkeypatch.setattr(im, "_exchange_trading_date", lambda: trading_day["value"])

    initial = mgr.get_active_nifty_contracts(25010)
    assert initial.atm_strike == 25000
    assert initial.option_expiry == weekly
    assert initial.futures_symbol == "NFO:NIFTYWEEKFUT"

    trading_day["value"] = weekly + timedelta(days=1)
    rolled = mgr.get_active_nifty_contracts(25026)
    assert rolled.atm_strike == 25000
    assert rolled.option_expiry == monthly
    assert rolled.futures_symbol == "NFO:NIFTYCURFUT"


def test_hysteresis_falls_back_when_held_pair_is_unavailable(monkeypatch):
    from nifty_scalper_bot.core import instrument_manager as im

    rows, weekly, monthly = _dump()
    rows = [
        row
        for row in rows
        if not (
            row.get("expiry") == monthly
            and row.get("instrument_type") in {"CE", "PE"}
            and row.get("strike") == 25000
        )
    ]
    mgr = InstrumentManager(KiteDump(rows))
    monkeypatch.setenv("ATM_STRIKE_HYSTERESIS_POINTS", "5")
    trading_day = {"value": weekly - timedelta(days=1)}
    monkeypatch.setattr(im, "_exchange_trading_date", lambda: trading_day["value"])

    assert mgr.get_active_nifty_contracts(25010).atm_strike == 25000
    trading_day["value"] = weekly + timedelta(days=1)

    rolled = mgr.get_active_nifty_contracts(25026)
    assert rolled.option_expiry == monthly
    assert rolled.atm_strike == 25050
    assert rolled.selected_ce == "NFO:NIFTYM25050CE"
    assert rolled.selected_pe == "NFO:NIFTYM25050PE"
