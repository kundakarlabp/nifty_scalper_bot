"""Range VWAP option model reuses DataHub broker metadata and options_math."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

from nifty_scalper_bot.data.data_hub import DataHub

IST = timezone(timedelta(hours=5, minutes=30))
SYMBOL = "NFO:NIFTY26NOV25250CE"


def _hub(*, expiry: date = date(2026, 11, 26), base: str = "NIFTY") -> DataHub:
    row = {
        "expiry": expiry,
        "strike": 25250.0,
        "instrument_type": "CE",
        "name": base,
        "lot_size": 65,
    }
    hub = object.__new__(DataHub)
    hub._resolver = SimpleNamespace(lookup=lambda symbol: row if symbol == SYMBOL else None)
    hub._now = lambda: datetime(2026, 11, 18, 10, 1, tzinfo=IST).timestamp()
    return hub


def test_fresh_model_reuses_broker_contract_and_underlying_spot() -> None:
    hub = _hub()
    risk = hub.get_option_model_snapshot(
        SYMBOL, spot_price=25230.0, bid=99.8, ask=100.0
    )
    assert risk["days_to_expiry"] == 8
    assert risk["lot_size"] == 65
    assert risk["range_option_contract_expiry"] == "2026-11-26"
    assert risk["range_option_model_source"] == "live_spot_bid_ask_black_scholes"
    assert risk["delta_underlying"] == "NIFTY_SPOT"
    assert 0 < risk["delta"] < 1
    assert risk["theta_period"] == "calendar_day"
    assert risk["vega_convention"] == "per_iv_percentage_point"
    assert risk["theta"] < 0
    assert risk["vega_per_iv_point"] > 0
    # The risk snapshot must not use a Greeks cache with unknown age/source.
    hub._greeks_cache = {SYMBOL: {"delta": -0.99}}
    assert hub.get_option_model_snapshot(
        SYMBOL, spot_price=25230.0, bid=99.8, ask=100.0
    )["delta"] == risk["delta"]


def test_missing_invalid_expiring_or_wrong_contract_fails_closed() -> None:
    assert _hub().get_option_model_snapshot(
        SYMBOL, spot_price=None, bid=99.8, ask=100.0
    ) == {}
    assert _hub().get_option_model_snapshot(
        SYMBOL, spot_price=25230, bid=100.0, ask=99.8
    ) == {}
    assert _hub(expiry=date(2026, 11, 18)).get_option_model_snapshot(
        SYMBOL, spot_price=25230, bid=99.8, ask=100
    ) == {}
    assert _hub(base="BANKNIFTY").get_option_model_snapshot(
        SYMBOL, spot_price=25230, bid=99.8, ask=100
    ) == {}
    assert _hub().get_option_model_snapshot(
        "NFO:UNKNOWN", spot_price=25230, bid=99.8, ask=100
    ) == {}
