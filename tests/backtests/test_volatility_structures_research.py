"""Deterministic, broker-free tests for research-only volatility structures."""

from dataclasses import replace
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from nifty_scalper_bot.backtesting.volatility_research import (
    VolatilityOptionQuote,
    assess_credit_iron_condor,
    assess_long_atm_straddle,
)

_AT = datetime(2026, 10, 30, 10, 15, tzinfo=ZoneInfo("Asia/Kolkata"))
_EXPIRY = date(2026, 11, 5)


@pytest.fixture(autouse=True)
def shadow_only(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")


def quote(option_type, strike, *, bid=15.0, ask=17.0, token=1):
    return VolatilityOptionQuote(
        symbol=f"NFO:NIFTY26NOV{strike}{option_type}",
        instrument_token=token,
        option_type=option_type,
        strike=strike,
        expiry=_EXPIRY,
        lot_size=75,
        bid=bid,
        ask=ask,
        bid_quantity=150,
        ask_quantity=150,
        observed_at=_AT - timedelta(milliseconds=250),
        source="ARCHIVED_FULL",
    )


def condor():
    return {
        "long_put": quote("PE", 23800, bid=9, ask=10, token=1),
        "short_put": quote("PE", 23900, bid=30, ask=32, token=2),
        "short_call": quote("CE", 24100, bid=35, ask=37, token=3),
        "long_call": quote("CE", 24200, bid=11, ask=12, token=4),
        "at": _AT,
    }


def test_four_leg_condor_has_capped_gross_payoff_not_certified_net_loss():
    result = assess_credit_iron_condor(**condor())
    assert result.credit_per_unit == pytest.approx(43.0)
    assert result.gross_max_profit_rupees == pytest.approx(3225.0)
    assert result.gross_max_loss_rupees == pytest.approx(4275.0)
    assert result.entry_fees_rupees > 0
    assert result.max_call_width == result.max_put_width == 100
    assert result.live_eligible is False
    assert result.net_loss_bound_certified is False
    assert result.evidence_label == "RESEARCH_CANDIDATE"


def test_condor_asymmetric_wings_use_worst_width():
    legs = condor()
    legs["long_call"] = replace(
        legs["long_call"],
        strike=24250,
        symbol="NFO:NIFTY26NOV24250CE",
    )
    result = assess_credit_iron_condor(**legs)
    assert result.max_call_width == 150
    assert result.max_put_width == 100
    assert result.gross_max_loss_rupees == pytest.approx(10725.0)


@pytest.mark.parametrize(
    ("field", "change", "reason"),
    [
        ("long_put", {"ask": 60}, "research_condor_credit_invalid"),
        ("long_put", {"strike": 23900}, "research_condor_wings_invalid"),
        ("short_put", {"bid_quantity": 0}, "research_contract_or_depth_unverified"),
        ("short_put", {"source": "LTP"}, "research_contract_or_depth_unverified"),
        (
            "short_put",
            {"expiry": date(2026, 11, 12)},
            "research_contract_or_depth_unverified",
        ),
        ("short_put", {"bid": 40, "ask": 40}, "research_crossed_or_nonfinite_book"),
        (
            "short_put",
            {"observed_at": _AT - timedelta(seconds=4)},
            "research_quote_stale_or_future",
        ),
        (
            "short_put",
            {"observed_at": _AT + timedelta(seconds=1)},
            "research_quote_stale_or_future",
        ),
        (
            "short_put",
            {"observed_at": _AT - timedelta(seconds=1)},
            "research_multileg_quote_skew",
        ),
    ],
)
def test_condor_unverified_inputs_fail_closed(field, change, reason):
    legs = condor()
    legs[field] = replace(legs[field], **change)
    with pytest.raises(ValueError, match=reason):
        assess_credit_iron_condor(**legs)


def test_long_atm_straddle_prices_at_asks_and_reports_vol_gap_only():
    call = quote("CE", 24000, bid=65, ask=70, token=5)
    put = quote("PE", 24000, bid=60, ask=65, token=6)
    result = assess_long_atm_straddle(
        call=call,
        put=put,
        at=_AT,
        implied_annual_vol=0.17,
        forecast_annual_vol=0.22,
        forecast_asof=_AT - timedelta(seconds=15),
    )
    assert result.debit_per_unit == pytest.approx(135.0)
    assert result.lower_expiry_breakeven == pytest.approx(23865.0)
    assert result.upper_expiry_breakeven == pytest.approx(24135.0)
    assert result.premium_plus_entry_fees_rupees > 135 * 75
    assert result.forecast_minus_implied_vol == pytest.approx(0.05)
    assert result.live_eligible is False


@pytest.mark.parametrize(
    ("forecast", "market", "age", "reason"),
    [
        (0.0, 0.17, 0, "research_volatility_input_invalid"),
        (float("nan"), 0.17, 0, "research_volatility_input_invalid"),
        (0.22, float("inf"), 0, "research_volatility_input_invalid"),
        (0.22, 0.17, 75, "research_forecast_stale_or_future"),
        (0.22, 0.17, -1, "research_forecast_stale_or_future"),
    ],
)
def test_long_straddle_unusable_vol_forecasts_fail_closed(
    forecast, market, age, reason
):
    with pytest.raises(ValueError, match=reason):
        assess_long_atm_straddle(
            call=quote("CE", 24000, token=5),
            put=quote("PE", 24000, token=6),
            at=_AT,
            implied_annual_vol=market,
            forecast_annual_vol=forecast,
            forecast_asof=_AT - timedelta(seconds=age),
        )


def test_unequal_straddle_strikes_cannot_masquerade_as_atm():
    with pytest.raises(ValueError, match="research_straddle_strikes_mismatched"):
        assess_long_atm_straddle(
            call=quote("CE", 24000, token=5),
            put=quote("PE", 24050, token=6),
            at=_AT,
            implied_annual_vol=0.17,
            forecast_annual_vol=0.22,
            forecast_asof=_AT,
        )


@pytest.mark.parametrize("mode", ["LIVE", "PAPER", "LIVE_SIMULATION", ""])
def test_volatility_research_cannot_be_used_outside_explicit_shadow(
    monkeypatch, mode
):
    monkeypatch.setenv("EXECUTION_MODE", mode)
    with pytest.raises(ValueError, match="volatility_research_requires_shadow_process"):
        assess_credit_iron_condor(**condor())


def test_quote_from_unverified_expiry_or_after_hours_cannot_be_researched():
    legs = condor()
    legs["long_put"] = replace(legs["long_put"], expiry=_AT.date())
    with pytest.raises(ValueError, match="research_expiry_or_lot_invalid"):
        assess_credit_iron_condor(**legs)
    legs = condor()
    legs["at"] = _AT.replace(hour=16)
    with pytest.raises(ValueError, match="research_outside_market_session"):
        assess_credit_iron_condor(**legs)
