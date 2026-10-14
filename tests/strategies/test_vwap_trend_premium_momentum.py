"""Opt-in VWAP research mode must confirm trend and option premium independently."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    VWAPProStrategyConfig,
)
from nifty_scalper_bot.strategies.elite_strategies.vwap_pro import VWAPProStrategy


def _evaluate(monkeypatch, *, side="CE", enabled=True, **updates):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("VWAP_TREND_MOMENTUM_ENABLED", "true" if enabled else "false")
    monkeypatch.setenv("VWAP_TREND_MOMENTUM_OPTION_RVOL_MIN", "1.5")
    monkeypatch.setenv("VWAP_TREND_REQUIRE_RETEST", "false")
    indicators = {
        "vwap": 100.0,
        "session_vwap": 100.0,
        "atr": 5.0,
        "open": 100.0,
        "high": 104.0,
        "low": 98.0,
        "close": 103.0,
        "volume": 1800.0,
        "avg_volume": 900.0,
        "spread_pct": 0.3,
        "direction_bias": side,
        "underlying_direction_bias": side,
        "underlying_direction_confidence": 0.95,
        "context_age_seconds": 1.0,
        "latest_bar_ts": "2026-09-10T10:00:00+05:30",
        "regime": "TREND",
        "futures_vwap_slope": 0.01 if side == "CE" else -0.01,
        "futures_volume_ratio": 1.2,
        "is_complete": True,
        "is_provisional": False,
    }
    indicators.update(updates)

    class _CompletedBarEngine:
        def get_history(self, _symbol, count=None, *, field="close"):
            assert field == "bars"
            assert count in (None, 1, 21)
            current_ts = datetime.fromisoformat(indicators["latest_bar_ts"])
            prior = [
                {
                    "timestamp": current_ts
                    - timedelta(minutes=max(i, indicators.get("prior_gap_minutes", 1))),
                    "open": 100.0,
                    "high": 103.0,
                    "low": 98.0,
                    "close": 101.0,
                    "volume": v,
                    "is_complete": True,
                    "is_provisional": False,
                }
                for i, v in enumerate(
                    reversed(indicators.get("prior_volumes", [900.0] * 10)),
                    start=1,
                )
            ]
            bars = list(reversed(prior)) + [
                {
                    "timestamp": indicators.get(
                        "history_timestamp", indicators["latest_bar_ts"]
                    ),
                    "open": indicators["open"],
                    "high": indicators.get("history_high", indicators["high"]),
                    "low": indicators.get("history_low", indicators["low"]),
                    "close": indicators["close"],
                    "volume": indicators.get("history_volume", indicators["volume"]),
                    "is_complete": indicators.get("is_complete", True),
                    "is_provisional": indicators.get("is_provisional", False),
                }
            ]
            return bars[-count:] if count is not None else bars

    strategy = VWAPProStrategy(VWAPProStrategyConfig(), _CompletedBarEngine())
    signal = strategy._evaluate_signal(f"NFO:NIFTY2691523400{side}", indicators, 103.0)
    return strategy, signal


@pytest.mark.parametrize("side", ["CE", "PE"])
def test_trending_underlying_and_confirmed_premium_generate_candidate(
    monkeypatch, side
):
    _, signal = _evaluate(monkeypatch, side=side)

    assert signal is not None
    assert signal.metadata["contract_side"] == side
    assert signal.metadata["trend_momentum_enabled"] is True
    assert signal.metadata["trend_momentum_regime"] == "TREND"
    assert signal.metadata["trend_momentum_premium_confirmed"] is True
    assert signal.metadata["option_volume_ratio"] == 2.0
    assert signal.metadata["trend_momentum_volume_baseline_bars"] == 10
    assert signal.metadata["vwap_reference_source"] == "session_vwap"
    assert signal.metadata["trend_momentum_completed_option_rvol"] == 2.0
    assert signal.metadata["trend_momentum_close_location"] > 0.8
    assert signal.metadata["trend_momentum_underlying_quality_confirmed"] is True
    assert signal.metadata["setup_pass"] is True


@pytest.mark.parametrize("regime", ["RANGE", "VOLATILE", "EVENT", "UNKNOWN", None])
def test_unfavourable_or_missing_regime_is_rejected(monkeypatch, regime):
    strategy, signal = _evaluate(monkeypatch, regime=regime)

    assert signal is None
    assert strategy.last_no_vote_reason == "trend_momentum_regime_not_trend"


def test_generic_direction_does_not_replace_underlying_confirmation(monkeypatch):
    strategy, signal = _evaluate(
        monkeypatch, underlying_direction_bias=None, direction_bias="CE"
    )

    assert signal is None
    assert strategy.last_no_vote_reason == "trend_momentum_underlying_not_confirmed"


def test_futures_volume_alone_cannot_confirm_option_premium(monkeypatch):
    strategy, signal = _evaluate(
        monkeypatch, volume=450.0, avg_volume=900.0, futures_volume_ratio=2.0
    )

    assert signal is None
    assert strategy.last_no_vote_reason == "trend_momentum_option_volume_not_confirmed"


def test_weak_option_candle_does_not_confirm_momentum(monkeypatch):
    strategy, signal = _evaluate(monkeypatch, open=102.8, close=103.0)

    assert signal is None
    assert strategy.last_no_vote_reason == "trend_momentum_premium_not_confirmed"


@pytest.mark.parametrize(
    "updates",
    [
        {"is_complete": False},
        {"is_provisional": True},
    ],
)
def test_explicitly_unfinished_option_candle_is_rejected(monkeypatch, updates):
    strategy, signal = _evaluate(monkeypatch, **updates)

    assert signal is None
    assert strategy.last_no_vote_reason == "trend_momentum_bar_unconfirmed"


def test_research_mode_disabled_preserves_legacy_vwap_candidacy(monkeypatch):
    _, signal = _evaluate(
        monkeypatch,
        enabled=False,
        regime="RANGE",
        volume=450.0,
        futures_volume_ratio=2.0,
        open=102.8,
        close=103.0,
    )

    assert signal is not None
    assert signal.metadata["trend_momentum_enabled"] is False


def test_stale_or_mismatched_premium_bar_cannot_confirm_entry(monkeypatch):
    strategy, signal = _evaluate(
        monkeypatch, history_timestamp="2026-09-10T09:59:00+05:30"
    )

    assert signal is None
    assert strategy.last_no_vote_reason == "trend_momentum_bar_unconfirmed"


@pytest.mark.parametrize(
    "updates",
    [
        {"underlying_direction_confidence": 0.4},
        {"context_age_seconds": 10.0},
    ],
)
def test_low_quality_underlying_context_rejected(monkeypatch, updates):
    strategy, signal = _evaluate(monkeypatch, **updates)

    assert signal is None
    assert (
        strategy.last_no_vote_reason == "trend_momentum_underlying_quality_unconfirmed"
    )


def test_upper_wick_reversal_not_mistaken_for_confirmed_entry(monkeypatch):
    strategy, signal = _evaluate(monkeypatch, high=118.0, history_high=118.0)

    assert signal is None
    assert strategy.last_no_vote_reason == "trend_momentum_weak_close"


def test_completed_premium_volume_overrules_mismatched_snapshot(monkeypatch):
    strategy, signal = _evaluate(monkeypatch, history_volume=400.0)

    assert signal is None
    assert strategy.last_no_vote_reason == "trend_momentum_option_volume_not_confirmed"


def test_invalid_completed_bar_geometry_never_authorizes_entry(monkeypatch):
    strategy, signal = _evaluate(monkeypatch, history_high=101.0)

    assert signal is None
    assert strategy.last_no_vote_reason == "trend_momentum_bar_unconfirmed"


def test_opt_in_refuses_rolling_vwap_without_session_reference(monkeypatch):
    strategy, signal = _evaluate(monkeypatch, session_vwap=None)

    assert signal is None
    assert strategy.last_no_vote_reason == "trend_momentum_session_vwap_unavailable"


def test_legacy_mode_keeps_rolling_fallback(monkeypatch):
    _, signal = _evaluate(monkeypatch, enabled=False, session_vwap=None)

    assert signal is not None
    assert signal.metadata["vwap_reference_source"] == "rolling_vwap"


def test_volume_compared_with_prior_completed_bars_not_current_average(monkeypatch):
    _, signal = _evaluate(monkeypatch, avg_volume=3000.0)

    assert signal is not None
    assert signal.metadata["trend_momentum_completed_option_rvol"] == 2.0


def test_insufficient_independent_option_volume_history_is_rejected(monkeypatch):
    strategy, signal = _evaluate(monkeypatch, prior_volumes=[900.0] * 3)

    assert signal is None
    assert strategy.last_no_vote_reason == "trend_momentum_volume_history_insufficient"


def test_retest_mode_requires_prior_reclaim_and_current_hold(monkeypatch):
    monkeypatch.setenv("VWAP_TREND_REQUIRE_RETEST", "true")
    strategy, signal = _evaluate(monkeypatch, low=100.0)

    assert signal is not None
    assert signal.metadata["trend_momentum_retest_confirmed"] is True


def test_retest_mode_rejects_unheld_vwap(monkeypatch):
    monkeypatch.setenv("VWAP_TREND_REQUIRE_RETEST", "true")
    strategy, signal = _evaluate(monkeypatch, low=98.0)

    assert signal is None
    assert strategy.last_no_vote_reason == "trend_momentum_retest_unconfirmed"


def test_retest_mode_rejects_stale_previous_candle(monkeypatch):
    monkeypatch.setenv("VWAP_TREND_REQUIRE_RETEST", "true")
    strategy, signal = _evaluate(monkeypatch, low=100.0, prior_gap_minutes=2)

    assert signal is None
    assert strategy.last_no_vote_reason == "trend_momentum_retest_unconfirmed"
