from __future__ import annotations

import math

from nifty_scalper_bot.data.instruments import (
    InstrumentResolver,
    atm_strike_for_spot,
    strike_for_delta,
)
from nifty_scalper_bot.utils.options_math import black_scholes_greeks


class DummyBroker:
    def list_instruments(self):  # pragma: no cover - not exercised
        return []


def test_lot_size_fallback_env(monkeypatch) -> None:
    resolver = InstrumentResolver(DummyBroker())
    resolver._instruments_cache = {}
    monkeypatch.setenv("INSTRUMENTS__NIFTY_LOT_SIZE", "90")
    assert resolver.lot_size_for_symbol("NIFTY24SEP20000CE") == 90
    monkeypatch.delenv("INSTRUMENTS__NIFTY_LOT_SIZE", raising=False)


def test_atm_and_delta_helpers() -> None:
    assert atm_strike_for_spot(19987.0, 50) == 20000

    spot = 20000.0
    iv = 0.2
    ttm = 20 / 365.0
    strike = strike_for_delta(spot, iv, ttm, 0.5, call=True)
    greeks = black_scholes_greeks(spot, float(strike), ttm, 0.06, iv, is_call=True)
    assert math.isclose(greeks["delta"], 0.5, rel_tol=0.1)

    strike_put = strike_for_delta(spot, iv, ttm, 0.5, call=False)
    greeks_put = black_scholes_greeks(
        spot, float(strike_put), ttm, 0.06, iv, is_call=False
    )
    assert math.isclose(abs(greeks_put["delta"]), 0.5, rel_tol=0.1)
