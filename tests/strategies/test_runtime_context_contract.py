from __future__ import annotations

from nifty_scalper_bot.strategies.indicators import IndicatorEngine
from nifty_scalper_bot.strategies.runtime_context_contract import normalise_live_direction_context


def test_runtime_context_preserves_live_direction_contract_keys() -> None:
    engine = IndicatorEngine()
    symbol = "NFO:NIFTY26JULFUT"

    engine.set_runtime_context(
        symbol,
        {
            "direction_bias": "ce",
            "context_age_seconds": "0.42",
            "spot_fresh": True,
            "fut_fresh": True,
            "ignored_payload_key": "must_not_leak",
        },
    )

    indicators = engine.get_indicators(symbol, names={"direction_bias", "context_age_seconds"})
    assert indicators["direction_bias"] == "CE"
    assert indicators["context_age_seconds"] == 0.42
    assert indicators["spot_fresh"] is True
    assert indicators["fut_fresh"] is True
    assert "ignored_payload_key" not in indicators


def test_runtime_context_preserves_live_quote_age_contract_keys() -> None:
    engine = IndicatorEngine()
    symbol = "NFO:NIFTY2670724400CE"

    engine.set_runtime_context(
        symbol,
        {
            "quote_age_s": 0.12,
            "tick_age_ms": 120,
            "quote_update_version": 42,
            "real_ticks_last_60s": 3,
            "quote_depth_valid": True,
            "ignored_payload_key": "must_not_leak",
        },
    )

    indicators = engine.get_indicators(symbol, names={"quote_age_s", "tick_age_ms", "quote_update_version"})
    assert indicators["quote_age_s"] == 0.12
    assert indicators["tick_age_ms"] == 120
    assert indicators["quote_update_version"] == 42
    assert indicators["real_ticks_last_60s"] == 3
    assert indicators["quote_depth_valid"] is True
    assert "ignored_payload_key" not in indicators


def test_runtime_context_derives_direction_bias_from_alias() -> None:
    preserved = normalise_live_direction_context(
        {
            "underlying_direction": "bearish",
            "futures_fresh": True,
            "context_age_seconds": 1.5,
        }
    )

    assert preserved["direction_bias"] == "PE"
    assert preserved["fut_fresh"] is True
    assert preserved["futures_fresh"] is True
    assert preserved["context_age_seconds"] == 1.5
