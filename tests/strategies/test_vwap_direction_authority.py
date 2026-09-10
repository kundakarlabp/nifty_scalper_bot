"""VWAP alignment must use explicit underlying direction before generic bias."""

import pytest

from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    VWAPProStrategyConfig,
)
from nifty_scalper_bot.strategies.elite_strategies.vwap_pro import VWAPProStrategy


def _evaluate(monkeypatch, side, underlying, generic, **updates):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("VWAP_PRO_REQUIRE_UNDERLYING_ALIGNMENT_LIVE", "true")
    strategy = VWAPProStrategy(VWAPProStrategyConfig(), None)
    indicators = {
        "vwap": 100.0,
        "atr": 5.0,
        "open": 100.0,
        "high": 104.0,
        "low": 98.0,
        "close": 103.0,
        "volume": 1000.0,
        "avg_volume": 900.0,
        "spread_pct": 0.3,
        "direction_bias": generic,
        "underlying_direction_bias": underlying,
        "underlying_direction_confidence": 0.95,
        "context_age_seconds": 1.0,
        "latest_bar_ts": "2026-09-10T10:00:00+05:30",
    }
    indicators.update(updates)
    signal = strategy._evaluate_signal(f"NFO:NIFTY2691523400{side}", indicators, 103.0)
    return strategy, signal


@pytest.mark.parametrize("side,opposite", [("CE", "PE"), ("PE", "CE")])
def test_generic_alignment_cannot_override_strong_underlying_conflict(
    monkeypatch, side, opposite
):
    strategy, signal = _evaluate(monkeypatch, side, opposite, side)

    assert signal is None
    assert strategy.last_no_vote_reason == "underlying_direction_conflict"


@pytest.mark.parametrize("side,opposite", [("CE", "PE"), ("PE", "CE")])
def test_underlying_alignment_survives_conflicting_generic_bias(
    monkeypatch, side, opposite
):
    _, signal = _evaluate(monkeypatch, side, side, opposite)

    assert signal is not None
    assert signal.metadata["trend_alignment"] is True
    assert signal.metadata["context_direction_used"] == side
    assert "trend_alignment" in signal.metadata["score_reasons"]
    assert "direction_conflict" not in signal.metadata["score_reasons"]


@pytest.mark.parametrize("underlying", [None, "UNKNOWN"])
@pytest.mark.parametrize("side", ["CE", "PE"])
def test_generic_fallback_remains_available_without_underlying_direction(
    monkeypatch, side, underlying
):
    _, signal = _evaluate(monkeypatch, side, underlying, side)

    assert signal is not None
    assert signal.metadata["context_direction_used"] == side
    assert signal.metadata["trend_alignment"] is True


@pytest.mark.parametrize(
    "updates",
    [{"context_age_seconds": 121.0}, {"underlying_direction_confidence": 0.5}],
)
def test_weak_or_stale_conflict_keeps_existing_soft_penalty(monkeypatch, updates):
    strategy, signal = _evaluate(monkeypatch, "CE", "PE", "CE", **updates)

    assert signal is None
    assert strategy.last_no_vote_reason == "weak_score"
