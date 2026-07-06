from __future__ import annotations

from nifty_scalper_bot.strategies.signal_quality import score_signal_quality, trigger_threshold


def test_ratio_threshold_is_normalized_to_ten_point_scale(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("TRIGGER_SMC_LIVE_MIN", "0.68")

    assert trigger_threshold("smc_lite") == 6.8


def test_percent_number_threshold_is_normalized_to_ten_point_scale(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("TRIGGER_SMC_LIVE_MIN", "68")

    assert trigger_threshold("smc_lite") == 6.8


def test_percent_string_threshold_is_normalized(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("TRIGGER_SMC_LIVE_MIN", "68%")

    assert trigger_threshold("smc_lite") == 6.8


def test_invalid_threshold_falls_back_to_live_default(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("TRIGGER_SMC_LIVE_MIN", "not-a-number")

    assert trigger_threshold("smc_lite") == 7.0


def test_global_confidence_floor_cannot_be_undercut(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("TRIGGER_SMC_LIVE_MIN", "0.60")
    monkeypatch.setenv("GLOBAL_MIN_SIGNAL_CONFIDENCE", "0.72")

    assert trigger_threshold("smc_lite") == 7.2


def test_low_score_rejected_when_ratio_threshold_normalized(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("TRIGGER_SMC_LIVE_MIN", "0.68")

    result = score_signal_quality(
        direction_score=6.0,
        strategy_score=6.0,
        option_score=6.0,
        data_score=6.0,
        rr_score=6.0,
        strategy_name="smc_lite",
    )

    assert result.final_score == 6.0
    assert result.components["threshold"] == 6.8
    assert result.allowed is False
    assert "score_below_threshold" in result.reasons
