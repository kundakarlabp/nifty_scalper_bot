"""Regression tests for non-veto SMC premium extension diagnostics."""

from nifty_scalper_bot.strategies.smc_entry_quality import assess_smc_entry_extension


def test_reports_late_overextended_smc_without_a_veto() -> None:
    result = assess_smc_entry_extension(
        premium=53.35,
        premium_vwap=34.12,
        option_atr=3.2265,
        sweep_age_minutes=3.45,
        vwap_source="exchange_vwap",
    )
    assert result["combined_extension_flag"] is True
    assert result["premium_distance_atr"] > 5
    assert result["policy"] == "shadow_observation_only"


def test_missing_vwap_does_not_invent_an_extension() -> None:
    result = assess_smc_entry_extension(
        premium=50,
        premium_vwap=None,
        option_atr=4,
        sweep_age_minutes=5,
        vwap_source="unavailable",
    )
    assert result["data_complete"] is False
    assert result["combined_extension_flag"] is None
    assert result["extended_fraction"] is None


def test_not_every_aged_smc_sweep_is_extended() -> None:
    result = assess_smc_entry_extension(
        premium=52,
        premium_vwap=50,
        option_atr=3,
        sweep_age_minutes=4,
        vwap_source="session_vwap",
    )
    assert result["aged_sweep"] is True
    assert result["combined_extension_flag"] is False
