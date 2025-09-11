from __future__ import annotations

import datetime as dt

from src.data import source as src


def test_ensure_atm_tokens_widens_strike(monkeypatch) -> None:
    underlying = "NIFTY"
    expiry = dt.date(2025, 1, 7)
    items = [
        {
            "name": underlying,
            "expiry": expiry,
            "strike": 17050,
            "instrument_type": "CE",
            "instrument_token": 111,
        },
        {
            "name": underlying,
            "expiry": expiry,
            "strike": 17050,
            "instrument_type": "PE",
            "instrument_token": 222,
        },
    ]
    monkeypatch.setattr(src, "_refresh_instruments_nfo", lambda _b: items)
    monkeypatch.setattr(src, "_pick_expiry", lambda _i, _u, _t: expiry)
    monkeypatch.setattr(src, "_subscribe_tokens", lambda _o, _t: True)
    monkeypatch.setattr(src, "_have_quote", lambda _o, _t: True)

    class _Settings:
        class instruments:
            spot_symbol = underlying
            trade_symbol = underlying

    monkeypatch.setattr(src, "settings", _Settings())

    class Dummy(src.DataSource):
        def get_last_price(self, _s):  # type: ignore[override]
            return 17010.0

    ds = Dummy()
    ds.ensure_atm_tokens(underlying=underlying)
    assert ds.atm_tokens == (111, 222)
    assert ds.current_atm_strike == 17050
    assert ds.current_atm_expiry == expiry


def test_drift_triggers_once(monkeypatch) -> None:
    underlying = "NIFTY"
    expiry = dt.date(2025, 1, 7)
    items = [
        {"name": underlying, "expiry": expiry, "strike": 17050, "instrument_type": "CE", "instrument_token": 111},
        {"name": underlying, "expiry": expiry, "strike": 17050, "instrument_type": "PE", "instrument_token": 222},
    ]
    calls: list[list[int]] = []
    monkeypatch.setattr(src, "_refresh_instruments_nfo", lambda _b: items)
    monkeypatch.setattr(src, "_pick_expiry", lambda _i, _u, _t: expiry)
    monkeypatch.setattr(src, "_have_quote", lambda _o, _t: True)

    def _sub(_o, t):
        calls.append(t)
        return True

    monkeypatch.setattr(src, "_subscribe_tokens", _sub)

    class _Settings:
        class instruments:
            spot_symbol = underlying
            trade_symbol = underlying

    monkeypatch.setattr(src, "settings", _Settings())

    class Dummy(src.DataSource):
        def __init__(self) -> None:
            self.atm_tokens = (111, 222)
            self.current_atm_strike = 16950
            self._atm_next_resolve_ts = 0.0

        def get_last_price(self, _s):  # type: ignore[override]
            return 17050.0

    ds = Dummy()
    monkeypatch.setenv("ATM_ROLL_DRIFT_PTS", "50")
    monkeypatch.setenv("ATM_RESOLVE_COOLDOWN_S", "60")
    ds.ensure_atm_tokens(underlying=underlying)
    ds.ensure_atm_tokens(underlying=underlying)
    assert calls == [[111, 222]]
