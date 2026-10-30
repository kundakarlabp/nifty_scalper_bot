"""Canonical broker instrument and FULL websocket quote provenance tests."""

from datetime import date, datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from nifty_scalper_bot.backtesting.volatility_research import (
    assess_long_atm_straddle,
    quote_from_canonical_cache,
)

_AT = datetime(2026, 10, 30, 10, 15, tzinfo=ZoneInfo("Asia/Kolkata"))


def sources(option_type="CE", token=7654):
    symbol = f"NFO:NIFTY26NOV24000{option_type}"
    contract = {
        "exchange": "NFO",
        "name": "NIFTY",
        "tradingsymbol": symbol.partition(":")[2],
        "instrument_type": option_type,
        "instrument_token": token,
        "strike": 24000.0,
        "lot_size": 65,
        "expiry": date(2026, 11, 3),
    }
    tick = {
        "instrument_token": token,
        "source": "ws",
        "bid_ask_source": "ws_full",
        "depth_available": True,
        "depth": {
            "buy": [{"price": 62.0, "quantity": 130}],
            "sell": [{"price": 63.5, "quantity": 130}],
        },
        "bid": 62.0,
        "ask": 63.5,
        "exchange_timestamp": _AT - timedelta(milliseconds=200),
    }
    return symbol, contract, tick


def read_sources(symbol, contract, tick):
    instruments = SimpleNamespace(lookup=lambda requested: contract)
    market = SimpleNamespace(get_quote=lambda requested: tick)
    return quote_from_canonical_cache(
        symbol=symbol,
        instrument_manager=instruments,
        market_data_manager=market,
    )


def test_canonical_full_depth_options_feed_existing_two_leg_risk_assessor(
    monkeypatch,
):
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    ce = read_sources(*sources("CE", 7654))
    pe = read_sources(*sources("PE", 7655))
    assert ce.source == pe.source == "FULL"
    assert ce.instrument_token != pe.instrument_token
    assert ce.lot_size == pe.lot_size == 65
    outcome = assess_long_atm_straddle(
        call=ce,
        put=pe,
        at=_AT,
        spot=24000.0,
        implied_annual_vol=0.20,
        forecast_annual_vol=0.25,
        forecast_asof=_AT,
    )
    assert outcome.debit_per_unit == 127.0
    assert outcome.live_eligible is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source", "rest"),
        ("source", "LTP"),
        ("bid_ask_source", "quote"),
        ("depth_available", False),
        ("instrument_token", 999),
        ("ask", 65.0),
        ("exchange_timestamp", datetime(2026, 10, 30, 10, 15)),
    ],
)
def test_unverified_quote_provenance_rejected(monkeypatch, field, value):
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    symbol, contract, tick = sources()
    tick[field] = value
    with pytest.raises(ValueError, match="research_contract_or_depth_unverified"):
        read_sources(symbol, contract, tick)


def test_quote_missing_registry_expiry_fails_without_guessing_date(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    symbol, contract, tick = sources()
    del contract["expiry"]
    with pytest.raises(ValueError, match="research_contract_or_depth_unverified"):
        read_sources(symbol, contract, tick)


def test_quote_mixed_tokens_and_missing_full_depth_fail_closed(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    symbol, contract, tick = sources()
    tick["depth"]["buy"] = []
    with pytest.raises(ValueError, match="research_contract_or_depth_unverified"):
        read_sources(symbol, contract, tick)
    symbol, contract, tick = sources()
    contract["instrument_token"] = 999
    with pytest.raises(ValueError, match="research_contract_or_depth_unverified"):
        read_sources(symbol, contract, tick)


def test_live_cannot_access_research_source_adapter(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    symbol, contract, tick = sources()
    with pytest.raises(ValueError, match="volatility_research_requires_shadow_process"):
        read_sources(symbol, contract, tick)
