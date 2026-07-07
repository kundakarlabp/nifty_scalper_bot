from __future__ import annotations

from nifty_scalper_bot.execution.quote_readiness import (
    evaluate_execution_quote,
    resolve_real_tick_count,
    resolve_tick_age_ms,
)


def test_missing_age_never_becomes_fresh_in_live_mode():
    result = evaluate_execution_quote(
        "NFO:NIFTY26JUN24000CE",
        {"bid": 99.9, "ask": 100.1, "depth_available": True},
        live_mode=True,
        max_tick_age_ms=2500,
        max_spread_pct=0.75,
        require_depth=True,
    )
    assert result.allowed is False
    assert result.reason == "tick_age_missing"


def test_tick_age_schema_is_canonical():
    assert resolve_tick_age_ms({"tick_age_ms": 125}) == 125
    assert resolve_tick_age_ms({"tick_age_s": 0.25}) == 250
    assert resolve_tick_age_ms({"quote_age_s": 0.25}) == 250
    assert resolve_tick_age_ms({"quote_age_ms": 125}) == 125
    assert resolve_tick_age_ms({"last_tick_age_s": 0.25}) == 250
    assert resolve_tick_age_ms({"market_data_age_ms": 125}) == 125


def test_quote_age_seconds_allows_live_orderflow_quote():
    result = evaluate_execution_quote(
        "NFO:NIFTY26JUN24000CE",
        {
            "bid": 99.9,
            "ask": 100.1,
            "quote_age_s": 0.10,
            "depth_available": True,
            "tradable_quote": True,
        },
        live_mode=True,
        max_tick_age_ms=2500,
        max_spread_pct=0.75,
        require_depth=True,
    )
    assert result.allowed is True
    assert result.reason == "ready"
    assert result.tick_age_ms == 100


def test_fresh_ms_quote_can_prove_one_recent_update():
    ticks, derived = resolve_real_tick_count(
        {"tick_age_ms": 100},
        tick_age_ms=100,
        max_age_ms=2500,
        has_bid_ask=True,
    )
    assert ticks == 1
    assert derived is True


def test_quote_age_seconds_does_not_invent_tick_count():
    ticks, derived = resolve_real_tick_count(
        {"quote_age_s": 0.1},
        tick_age_ms=100,
        max_age_ms=2500,
        has_bid_ask=True,
    )
    assert ticks == 0
    assert derived is False
