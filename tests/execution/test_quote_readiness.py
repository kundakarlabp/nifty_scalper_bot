from __future__ import annotations

from nifty_scalper_bot.execution.quote_readiness import (
    evaluate_execution_quote,
    resolve_real_tick_count,
    resolve_tick_age_ms,
)
from nifty_scalper_bot.execution.readiness import (
    evaluate_quote_readiness,
    resolve_quote_bid_ask_spread,
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


def test_quote_age_seconds_allows_runtime_quote_readiness():
    result = evaluate_quote_readiness(
        "NFO:NIFTY26JUN24000CE",
        {
            "ltp": 100.0,
            "bid": 99.9,
            "ask": 100.1,
            "quote_age_s": 0.10,
            "depth_available": True,
            "tradable_quote": True,
        },
        require_fresh=True,
        max_age_s=2.5,
        max_spread_pct=0.75,
    )

    assert result.tradable_quote_ready is True
    assert result.reason == "ready"


def test_synthetic_timestamp_quality_blocks_live_quote_readiness():
    payload = {
        "bid": 99.9,
        "ask": 100.1,
        "tick_age_ms": 100,
        "depth_available": True,
        "tradable_quote": True,
        "timestamp_quality": "synthetic",
    }

    result = evaluate_execution_quote(
        "NFO:NIFTY26JUN24000CE",
        payload,
        live_mode=True,
        max_tick_age_ms=2500,
        max_spread_pct=0.75,
        require_depth=True,
    )
    bid, ask, spread, source = resolve_quote_bid_ask_spread(payload)

    assert result.allowed is False
    assert result.reason == "timestamp_quality_unusable"
    assert (bid, ask, spread, source) == (
        None,
        None,
        None,
        "timestamp_quality_unusable",
    )


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


def test_canonical_quote_age_converts_milliseconds_to_seconds():
    from nifty_scalper_bot.execution.readiness import resolve_quote_age_seconds

    assert resolve_quote_age_seconds({"quote_age_ms": 2500}) == 2.5


def test_canonical_quote_age_keeps_seconds_unchanged():
    from nifty_scalper_bot.execution.readiness import resolve_quote_age_seconds

    assert resolve_quote_age_seconds({"quote_age_s": 2.5}) == 2.5


def test_max_quote_age_seconds_blank_env_uses_legacy_ms(monkeypatch):
    from nifty_scalper_bot.execution.readiness import resolve_max_quote_age_seconds

    monkeypatch.setenv("TEST_MAX_AGE_SECONDS", "   ")
    monkeypatch.setenv("TEST_MAX_AGE_MS", "2500")

    assert (
        resolve_max_quote_age_seconds(
            "TEST_MAX_AGE_SECONDS", "TEST_MAX_AGE_MS", default_seconds=60.0
        )
        == 2.5
    )


def test_max_quote_age_seconds_empty_env_uses_legacy_ms(monkeypatch):
    from nifty_scalper_bot.execution.readiness import resolve_max_quote_age_seconds

    monkeypatch.setenv("TEST_MAX_AGE_SECONDS", "")
    monkeypatch.setenv("TEST_MAX_AGE_MS", "2500")

    assert (
        resolve_max_quote_age_seconds(
            "TEST_MAX_AGE_SECONDS", "TEST_MAX_AGE_MS", default_seconds=60.0
        )
        == 2.5
    )


def test_max_quote_age_seconds_valid_env_takes_precedence(monkeypatch):
    from nifty_scalper_bot.execution.readiness import resolve_max_quote_age_seconds

    monkeypatch.setenv("TEST_MAX_AGE_SECONDS", "5")
    monkeypatch.setenv("TEST_MAX_AGE_MS", "2500")

    assert (
        resolve_max_quote_age_seconds(
            "TEST_MAX_AGE_SECONDS", "TEST_MAX_AGE_MS", default_seconds=60.0
        )
        == 5.0
    )


def test_max_quote_age_seconds_invalid_non_empty_uses_default(monkeypatch):
    from nifty_scalper_bot.execution.readiness import resolve_max_quote_age_seconds

    monkeypatch.setenv("TEST_MAX_AGE_SECONDS", "bad # comment")
    monkeypatch.setenv("TEST_MAX_AGE_MS", "2500")

    assert (
        resolve_max_quote_age_seconds(
            "TEST_MAX_AGE_SECONDS", "TEST_MAX_AGE_MS", default_seconds=60.0
        )
        == 60.0
    )
