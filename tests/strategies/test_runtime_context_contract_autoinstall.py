from __future__ import annotations

import importlib


def test_strategy_package_installs_quote_context_contract() -> None:
    importlib.import_module("nifty_scalper_bot.strategies")
    from nifty_scalper_bot.strategies.indicators import IndicatorEngine

    engine = IndicatorEngine()
    symbol = "NFO:NIFTY2670724400CE"
    engine.set_runtime_context(
        symbol,
        {
            "tick_age_ms": 120,
            "quote_age_s": 0.12,
            "quote_update_version": 42,
            "real_ticks_last_60s": 3,
            "quote_depth_valid": True,
            "ignored_payload_key": "must_not_leak",
        },
    )

    indicators = engine.get_indicators(
        symbol,
        names={"tick_age_ms", "quote_age_s", "quote_update_version", "real_ticks_last_60s", "quote_depth_valid"},
    )

    assert indicators["tick_age_ms"] == 120
    assert indicators["quote_age_s"] == 0.12
    assert indicators["quote_update_version"] == 42
    assert indicators["real_ticks_last_60s"] == 3
    assert indicators["quote_depth_valid"] is True
    assert "ignored_payload_key" not in indicators
