"""Opt-in VWAP research mode must confirm trend and option premium independently."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    VWAPProStrategyConfig,
)
from nifty_scalper_bot.strategies.elite_strategies.vwap_pro import VWAPProStrategy


def _evaluate(
    monkeypatch,
    *,
    side="CE",
    enabled=True,
    retest=False,
    execution_mode="LIVE",
    **updates,
):
    monkeypatch.setenv("EXECUTION_MODE", execution_mode)
    monkeypatch.setenv("VWAP_TREND_MOMENTUM_ENABLED", "true" if enabled else "false")
    monkeypatch.setenv("VWAP_TREND_MOMENTUM_OPTION_RVOL_MIN", "1.5")
    monkeypatch.setenv("VWAP_TREND_REQUIRE_RETEST", "true" if retest else "false")
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
                    "open": indicators.get("prior_open", 100.0),
                    "high": indicators.get("prior_high", 103.0),
                    "low": indicators.get("prior_low", 98.0),
                    "close": indicators.get("prior_close", 101.0),
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

        def get_session_vwap_at(self, _symbol, timestamp):
            current = datetime.fromisoformat(indicators["latest_bar_ts"])
            assert timestamp == current - timedelta(minutes=1)
            return indicators.get("prior_session_vwap", 100.0)

    strategy = VWAPProStrategy(VWAPProStrategyConfig(), _CompletedBarEngine())
    signal = strategy._evaluate_signal(
        f"NFO:NIFTY2691523400{side}",
        indicators,
        indicators.get("current_price", 103.0),
    )
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


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1.0])
def test_invalid_session_reference_never_authorizes_candidate(monkeypatch, invalid):
    strategy, signal = _evaluate(monkeypatch, session_vwap=invalid)

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
    strategy, signal = _evaluate(monkeypatch, low=100.0, retest=True)

    assert signal is not None
    assert signal.metadata["trend_momentum_retest_confirmed"] is True


@pytest.mark.parametrize("prior_volume", [0.0, -1.0, float("nan"), float("inf")])
def test_retest_rejects_untraded_prior_reclaim_without_losing_momentum(
    monkeypatch, prior_volume
):
    """Untraded or invalid prior-volume bars cannot establish a VWAP reclaim."""
    prior_volumes = [900.0] * 9 + [prior_volume]
    strategy, signal = _evaluate(
        monkeypatch,
        retest=True,
        low=100.0,
        prior_volumes=prior_volumes,
    )
    assert signal is None
    assert strategy.last_no_vote_reason == "trend_momentum_prior_bar_no_trades"

    # Preserve the independent immediate-momentum research arm.
    _, immediate_signal = _evaluate(
        monkeypatch,
        retest=False,
        low=100.0,
        prior_volumes=prior_volumes,
    )
    assert immediate_signal is not None


def test_retest_mode_rejects_unheld_vwap(monkeypatch):
    strategy, signal = _evaluate(monkeypatch, low=98.0, retest=True)

    assert signal is None
    assert strategy.last_no_vote_reason == "trend_momentum_retest_unconfirmed"


def test_retest_mode_rejects_stale_previous_candle(monkeypatch):
    strategy, signal = _evaluate(
        monkeypatch, low=100.0, prior_gap_minutes=2, retest=True
    )

    assert signal is None
    assert strategy.last_no_vote_reason == "trend_momentum_retest_unconfirmed"


def test_nonfinite_futures_slope_cannot_confirm_vwap_entry(monkeypatch) -> None:
    strategy, signal = _evaluate(
        monkeypatch,
        enabled=False,
        regime="RANGE",
        volume=450.0,
        futures_volume_ratio=2.0,
        futures_vwap_slope=float("inf"),
        open=102.8,
        close=103.0,
    )

    assert signal is None
    assert strategy.last_no_vote_reason == "futures_slope_not_aligned"


def test_nonfinite_spread_cannot_masquerade_as_tradable_quote(monkeypatch) -> None:
    strategy, signal = _evaluate(
        monkeypatch,
        enabled=False,
        regime="RANGE",
        volume=450.0,
        futures_volume_ratio=2.0,
        spread_pct=float("nan"),
        open=102.8,
        close=103.0,
    )

    assert signal is None
    assert strategy.last_no_vote_reason == "invalid_option_market_data"


def test_nonfinite_futures_volume_cannot_confirm_thin_option(monkeypatch) -> None:
    strategy, signal = _evaluate(
        monkeypatch,
        enabled=False,
        regime="RANGE",
        volume=450.0,
        futures_volume_ratio=float("inf"),
        open=102.8,
        close=103.0,
    )

    assert signal is None
    assert strategy.last_no_vote_reason == "volume_confirmation_missing"


def test_finalized_signal_bar_gate_is_opt_in_and_preserves_default(monkeypatch):
    monkeypatch.delenv("VWAP_REQUIRE_FINALIZED_SIGNAL_BAR", raising=False)
    baseline_strategy, baseline = _evaluate(
        monkeypatch, enabled=False, is_complete=False
    )
    assert baseline is not None

    monkeypatch.setenv("VWAP_REQUIRE_FINALIZED_SIGNAL_BAR", "true")
    gated_strategy, gated = _evaluate(monkeypatch, enabled=False, is_complete=False)
    assert gated is None
    assert gated_strategy.last_no_vote_reason == "vwap_signal_bar_unconfirmed"


@pytest.mark.parametrize("side", ["CE", "PE"])
def test_finalized_signal_bar_gate_accepts_completed_premium_event(monkeypatch, side):
    monkeypatch.setenv("VWAP_REQUIRE_FINALIZED_SIGNAL_BAR", "true")
    strategy, signal = _evaluate(monkeypatch, side=side, enabled=False)
    assert signal is not None
    assert signal.metadata["vwap_finalized_bar_required"] is True
    assert signal.metadata["vwap_finalized_bar_confirmed"] is True


def test_finalized_signal_bar_gate_rejects_mismatched_ohlc(monkeypatch):
    monkeypatch.setenv("VWAP_REQUIRE_FINALIZED_SIGNAL_BAR", "true")
    strategy, signal = _evaluate(monkeypatch, enabled=False, history_high=99.0)
    assert signal is None
    assert strategy.last_no_vote_reason == "vwap_signal_bar_unconfirmed"


def test_retest_compares_prior_candle_to_prior_asof_vwap(monkeypatch):
    """A prior candle below its own VWAP cannot be rescued by later VWAP drift."""
    strategy, signal = _evaluate(
        monkeypatch, retest=True, low=100.0, prior_session_vwap=102.0
    )
    assert signal is None
    assert strategy.last_no_vote_reason == "trend_momentum_retest_unconfirmed"


def test_retest_fails_closed_without_timestamp_aligned_prior_vwap(monkeypatch):
    strategy, signal = _evaluate(
        monkeypatch, retest=True, low=100.0, prior_session_vwap=None
    )
    assert signal is None
    assert strategy.last_no_vote_reason == "trend_momentum_prior_vwap_unavailable"


def test_retest_preserves_genuine_prior_reclaim_when_prior_vwap_available(
    monkeypatch,
):
    _, signal = _evaluate(monkeypatch, retest=True, low=100.0, prior_session_vwap=100.0)
    assert signal is not None
    assert signal.metadata["trend_momentum_retest_confirmed"] is True
    assert signal.metadata["trend_momentum_prior_session_vwap"] == 100.0


@pytest.mark.parametrize("side", ["CE", "PE"])
def test_research_neutral_futures_slope_requires_explicit_shadow_opt_in(
    monkeypatch, side
):
    near_neutral_same_side = 0.0001 if side == "CE" else -0.0001
    _, baseline = _evaluate(
        monkeypatch,
        side=side,
        execution_mode="SHADOW",
        futures_vwap_slope=near_neutral_same_side,
    )
    assert baseline is None

    monkeypatch.setenv("VWAP_RESEARCH_NEUTRAL_FUTURES_SLOPE", "true")
    strategy, admitted = _evaluate(
        monkeypatch,
        side=side,
        execution_mode="SHADOW",
        futures_vwap_slope=near_neutral_same_side,
    )
    assert admitted is not None
    assert admitted.metadata["research_neutral_slope_admitted"] is True
    assert admitted.metadata["futures_slope_alignment"] is False
    assert admitted.metadata["trend_momentum_completed_option_rvol"] >= 1.5
    assert admitted.metadata["setup_pass"] is True

    # The original SHADOW flag must never override an explicit LIVE kill switch.
    monkeypatch.setenv("VWAP_LIVE_NEUTRAL_SLOPE_ENABLED", "false")
    live_strategy, live = _evaluate(
        monkeypatch,
        side=side,
        execution_mode="LIVE",
        futures_vwap_slope=near_neutral_same_side,
    )
    assert live is None
    assert live_strategy.last_no_vote_reason == "futures_slope_not_aligned"


@pytest.mark.parametrize(
    "side, updates, expected_reason",
    [
        ("CE", {"futures_vwap_slope": -0.0001}, "futures_slope_not_aligned"),
        ("PE", {"futures_vwap_slope": 0.0001}, "futures_slope_not_aligned"),
        ("CE", {"futures_vwap_slope": None}, "futures_slope_not_aligned"),
        ("CE", {"futures_vwap_slope": float("inf")}, "futures_slope_not_aligned"),
        (
            "CE",
            {"futures_vwap_slope": 0.0001, "futures_volume_ratio": 0.4},
            "futures_slope_not_aligned",
        ),
        (
            "CE",
            {"futures_vwap_slope": 0.0001, "underlying_direction_confidence": 0.4},
            "trend_momentum_underlying_quality_unconfirmed",
        ),
        (
            "CE",
            {"futures_vwap_slope": 0.0001, "context_age_seconds": 10.0},
            "trend_momentum_underlying_quality_unconfirmed",
        ),
        (
            "CE",
            {"futures_vwap_slope": 0.0001, "regime": "RANGE"},
            "trend_momentum_regime_not_trend",
        ),
        (
            "CE",
            {"futures_vwap_slope": 0.0001, "volume": 0.0},
            "option_candle_no_trades",
        ),
        (
            "CE",
            {
                "futures_vwap_slope": 0.0001,
                "history_timestamp": "2026-09-10T09:59:00+05:30",
            },
            "trend_momentum_bar_unconfirmed",
        ),
    ],
)
def test_research_neutral_slope_does_not_override_quality_guards(
    monkeypatch, side, updates, expected_reason
):
    monkeypatch.setenv("VWAP_RESEARCH_NEUTRAL_FUTURES_SLOPE", "true")
    strategy, admitted = _evaluate(
        monkeypatch, side=side, execution_mode="SHADOW", **updates
    )
    assert admitted is None
    assert strategy.last_no_vote_reason == expected_reason


def test_research_neutral_slope_needs_completed_option_momentum(monkeypatch):
    monkeypatch.setenv("VWAP_RESEARCH_NEUTRAL_FUTURES_SLOPE", "true")
    strategy, signal = _evaluate(
        monkeypatch,
        execution_mode="SHADOW",
        enabled=False,
        futures_vwap_slope=0.0001,
    )
    assert signal is None
    assert strategy.last_no_vote_reason == "futures_slope_not_aligned"


@pytest.mark.parametrize("side", ["CE", "PE"])
def test_live_neutral_slope_enables_quality_proven_entry_by_default(monkeypatch, side):
    monkeypatch.delenv("VWAP_LIVE_NEUTRAL_SLOPE_ENABLED", raising=False)
    near_neutral = 0.0001 if side == "CE" else -0.0001
    _, signal = _evaluate(
        monkeypatch,
        side=side,
        enabled=False,
        execution_mode="LIVE",
        futures_vwap_slope=near_neutral,
    )
    assert signal is not None
    assert signal.metadata["live_neutral_slope_enabled"] is True
    assert signal.metadata["live_neutral_slope_admitted"] is True
    assert signal.metadata["research_neutral_slope_admitted"] is False
    assert signal.metadata["vwap_finalized_bar_confirmed"] is True
    assert signal.metadata["vwap_neutral_slope_premium_proof_required"] is True
    assert signal.metadata["trend_momentum_completed_option_rvol"] >= 1.5
    assert signal.metadata["futures_slope_alignment"] is False
    assert "live_neutral_futures_slope" in signal.metadata["setup_reasons"]


@pytest.mark.parametrize("side", ["CE", "PE"])
def test_live_neutral_slope_rolls_back_with_one_switch(monkeypatch, side):
    monkeypatch.setenv("VWAP_LIVE_NEUTRAL_SLOPE_ENABLED", "false")
    near_neutral = 0.0001 if side == "CE" else -0.0001
    strategy, signal = _evaluate(
        monkeypatch,
        side=side,
        enabled=False,
        execution_mode="LIVE",
        futures_vwap_slope=near_neutral,
    )
    assert signal is None
    assert strategy.last_no_vote_reason == "futures_slope_not_aligned"


@pytest.mark.parametrize(
    "side, updates, reason",
    [
        ("CE", {"futures_vwap_slope": -0.0001}, "futures_slope_not_aligned"),
        ("PE", {"futures_vwap_slope": 0.0001}, "futures_slope_not_aligned"),
        ("CE", {"futures_vwap_slope": None}, "futures_slope_not_aligned"),
        ("CE", {"futures_vwap_slope": float("nan")}, "futures_slope_not_aligned"),
        (
            "CE",
            {"futures_vwap_slope": 0.0001, "regime": "RANGE"},
            "trend_momentum_regime_not_trend",
        ),
        (
            "CE",
            {"futures_vwap_slope": 0.0001, "underlying_direction_bias": "PE"},
            "underlying_direction_conflict",
        ),
        (
            "CE",
            {"futures_vwap_slope": 0.0001, "underlying_direction_confidence": 0.5},
            "trend_momentum_underlying_quality_unconfirmed",
        ),
        (
            "CE",
            {"futures_vwap_slope": 0.0001, "context_age_seconds": 8.0},
            "trend_momentum_underlying_quality_unconfirmed",
        ),
        (
            "CE",
            {"futures_vwap_slope": 0.0001, "history_volume": 300.0},
            "trend_momentum_option_volume_not_confirmed",
        ),
        (
            "CE",
            {"futures_vwap_slope": 0.0001, "futures_volume_ratio": 0.5},
            "futures_slope_not_aligned",
        ),
        (
            "CE",
            {"futures_vwap_slope": 0.0001, "history_high": 101.0},
            "trend_momentum_bar_unconfirmed",
        ),
        (
            "CE",
            {"futures_vwap_slope": 0.0001, "is_complete": False},
            "trend_momentum_bar_unconfirmed",
        ),
        (
            "CE",
            {"futures_vwap_slope": 0.0001, "open": 102.8},
            "trend_momentum_premium_not_confirmed",
        ),
        (
            "CE",
            {"futures_vwap_slope": 0.0001, "session_vwap": None},
            "trend_momentum_session_vwap_unavailable",
        ),
    ],
)
def test_live_neutral_slope_preserves_structural_guards(
    monkeypatch, side, updates, reason
):
    monkeypatch.delenv("VWAP_LIVE_NEUTRAL_SLOPE_ENABLED", raising=False)
    strategy, signal = _evaluate(
        monkeypatch, side=side, enabled=False, execution_mode="LIVE", **updates
    )
    assert signal is None
    assert strategy.last_no_vote_reason == reason


def test_live_strong_slope_baseline_is_not_tightened(monkeypatch):
    monkeypatch.delenv("VWAP_LIVE_NEUTRAL_SLOPE_ENABLED", raising=False)
    _, signal = _evaluate(
        monkeypatch,
        enabled=False,
        execution_mode="LIVE",
        futures_vwap_slope=0.01,
        is_complete=False,
        regime="RANGE",
    )
    assert signal is not None
    assert signal.metadata["live_neutral_slope_admitted"] is False
    assert signal.metadata["vwap_neutral_slope_premium_proof_required"] is False


@pytest.mark.parametrize("side", ["CE", "PE"])
def test_opt_in_trend_retest_extension_requires_completed_pullback_rebreak(
    monkeypatch, side
):
    inputs = dict(
        enabled=False,
        side=side,
        atr=1.2,
        open=102.0,
        high=105.0,
        low=101.0,
        close=104.0,
        current_price=104.0,
        prior_open=102.0,
        prior_close=101.0,
    )
    monkeypatch.setenv("VWAP_TREND_RETEST_EXTENSION_ENABLED", "false")
    baseline_strategy, baseline = _evaluate(monkeypatch, **inputs)
    assert baseline is None
    assert baseline_strategy.last_no_vote_reason == "distance_outside_band"

    monkeypatch.setenv("VWAP_TREND_RETEST_EXTENSION_ENABLED", "true")
    extension_strategy, extension = _evaluate(monkeypatch, **inputs)
    assert extension is not None
    assert extension.metadata["setup_type"] == "vwap_trend_pullback_retest"
    assert extension.metadata["vwap_trend_retest_extension_admitted"] is True
    assert extension.metadata["vwap_distance_atr"] > 3.0
    assert extension.metadata["requires_runner_execution_validation"] is True
    assert extension.metadata["trend_alignment"] is True


@pytest.mark.parametrize(
    "updates, expected",
    [
        (
            {"prior_open": 100.0, "prior_close": 101.0},
            "trend_retest_structure_unconfirmed",
        ),
        ({"prior_gap_minutes": 2}, "trend_retest_structure_unconfirmed"),
        ({"prior_volumes": [900.0] * 9 + [0.0]}, "trend_retest_structure_unconfirmed"),
        ({"low": 102.0}, "trend_momentum_weak_close"),
        ({"history_high": 101.0}, "trend_momentum_bar_unconfirmed"),
        ({"underlying_direction_bias": "PE"}, "distance_outside_band"),
        ({"context_age_seconds": 10.0}, "distance_outside_band"),
        ({"regime": "RANGE"}, "trend_momentum_regime_not_trend"),
        ({"futures_vwap_slope": -0.01}, "futures_slope_not_aligned"),
    ],
)
def test_trend_retest_extension_fails_closed_on_missing_structural_quality(
    monkeypatch, updates, expected
):
    monkeypatch.setenv("VWAP_TREND_RETEST_EXTENSION_ENABLED", "true")
    inputs = dict(
        enabled=False,
        atr=1.2,
        open=102.0,
        high=105.0,
        low=101.0,
        close=104.0,
        current_price=104.0,
        prior_open=102.0,
        prior_close=101.0,
    )
    inputs.update(updates)
    strategy, signal = _evaluate(monkeypatch, **inputs)
    assert signal is None
    assert strategy.last_no_vote_reason == expected


@pytest.mark.parametrize(
    ("configured_floor", "admitted"),
    [("0.90", False), ("0.80", True)],
)
def test_vwap_confidence_floor_counterfactual_is_only_a_unit_fixture(
    monkeypatch, configured_floor, admitted
):
    """Compare one unchanged completed bar; do not claim economic replay."""
    monkeypatch.setenv("VWAP_EARLY_TREND_MIN_CONTEXT_CONF", configured_floor)
    strategy, signal = _evaluate(monkeypatch, underlying_direction_confidence=0.80)
    assert strategy._early_trend_min_context_conf == float(configured_floor)
    assert (signal is not None) is admitted
    if not admitted:
        assert (
            strategy.last_no_vote_reason
            == "trend_momentum_underlying_quality_unconfirmed"
        )
    else:
        assert signal.metadata["trend_momentum_underlying_quality_confirmed"]


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"context_age_seconds": 3.0}, "trend_momentum_underlying_quality_unconfirmed"),
        ({"regime": "RANGE"}, "trend_momentum_regime_not_trend"),
        ({"is_complete": False}, "trend_momentum_bar_unconfirmed"),
    ],
)
def test_research_080_floor_still_requires_fresh_trend_and_final_bar(
    monkeypatch, changes, reason
):
    """An alternate confidence floor cannot replace independent proof."""
    monkeypatch.setenv("VWAP_EARLY_TREND_MIN_CONTEXT_CONF", "0.80")
    strategy, signal = _evaluate(
        monkeypatch, underlying_direction_confidence=0.80, **changes
    )
    assert signal is None
    assert strategy.last_no_vote_reason == reason


@pytest.mark.parametrize(
    ("confidence", "eligible"),
    [(0.79, False), (0.80, True), (0.85, True)],
)
def test_default_080_confidence_floor_accepts_080_but_not_weaker_context(
    monkeypatch, confidence, eligible
):
    """Exact live default boundary; do not replace independent quality checks."""
    monkeypatch.delenv("VWAP_EARLY_TREND_MIN_CONTEXT_CONF", raising=False)
    strategy, signal = _evaluate(
        monkeypatch, underlying_direction_confidence=confidence
    )
    assert strategy._early_trend_min_context_conf == 0.80
    assert (signal is not None) is eligible
    if not eligible:
        assert (
            strategy.last_no_vote_reason
            == "trend_momentum_underlying_quality_unconfirmed"
        )


def test_strict_continuation_retest_rebreak_is_eligible(monkeypatch):
    monkeypatch.setenv("VWAP_STRICT_CONTINUATION_ENABLED", "true")
    _, signal = _evaluate(
        monkeypatch,
        enabled=False,
        execution_mode="SHADOW",
        low=100.0,
        prior_high=102.0,
    )
    assert signal is not None
    assert signal.metadata["setup_type"] == "vwap_continuation"
    assert signal.metadata["vwap_continuation_proof"] == "retest_rebreak"
    assert signal.metadata["vwap_finalized_bar_required"] is True


def test_strict_continuation_rejects_no_breakout(monkeypatch):
    monkeypatch.setenv("VWAP_STRICT_CONTINUATION_ENABLED", "true")
    strategy, signal = _evaluate(
        monkeypatch,
        enabled=False,
        execution_mode="SHADOW",
        low=100.0,
        prior_high=105.0,
    )
    assert signal is None
    assert strategy.last_no_vote_reason == "vwap_continuation_quality_unconfirmed"


def test_strict_continuation_rejects_range_regime(monkeypatch):
    monkeypatch.setenv("VWAP_STRICT_CONTINUATION_ENABLED", "true")
    strategy, signal = _evaluate(
        monkeypatch,
        enabled=False,
        execution_mode="SHADOW",
        regime="RANGE",
        low=100.0,
    )
    assert signal is None
    assert strategy.last_no_vote_reason == "vwap_quality_regime_not_trend"


def test_strict_continuation_requires_finalized_candle(monkeypatch):
    monkeypatch.setenv("VWAP_STRICT_CONTINUATION_ENABLED", "true")
    strategy, signal = _evaluate(
        monkeypatch,
        enabled=False,
        execution_mode="SHADOW",
        low=100.0,
        is_complete=False,
    )
    assert signal is None
    assert strategy.last_no_vote_reason == "vwap_signal_bar_unconfirmed"


def test_strict_continuation_rejects_stale_thesis(monkeypatch):
    monkeypatch.setenv("VWAP_STRICT_CONTINUATION_ENABLED", "true")
    monkeypatch.setattr(
        VWAPProStrategy,
        "_recover_thesis_anchor_from_history",
        lambda self, *args, **kwargs: "2026-09-10T09:55:00+05:30",
    )
    strategy, signal = _evaluate(
        monkeypatch,
        enabled=False,
        execution_mode="SHADOW",
        low=100.0,
        prior_high=102.0,
    )
    assert signal is None
    assert strategy.last_no_vote_reason == "vwap_continuation_quality_unconfirmed"


def test_strict_continuation_rejects_untraded_prior_retest(monkeypatch):
    monkeypatch.setenv("VWAP_STRICT_CONTINUATION_ENABLED", "true")
    prior_volumes = [900.0] * 9 + [0.0]
    strategy, signal = _evaluate(
        monkeypatch,
        enabled=False,
        execution_mode="SHADOW",
        low=100.0,
        prior_high=102.0,
        prior_volumes=prior_volumes,
    )
    assert signal is None
    assert strategy.last_no_vote_reason == "vwap_continuation_quality_unconfirmed"


def test_strict_continuation_is_default_on(monkeypatch):
    monkeypatch.delenv("VWAP_STRICT_CONTINUATION_ENABLED", raising=False)
    strategy, signal = _evaluate(
        monkeypatch,
        enabled=False,
        execution_mode="SHADOW",
        low=100.0,
        prior_high=105.0,
    )
    assert signal is None
    assert strategy._strict_continuation_enabled is True
    assert strategy.last_no_vote_reason == "vwap_continuation_quality_unconfirmed"


def test_strict_continuation_explicit_rollback_retains_legacy_candidate(monkeypatch):
    monkeypatch.setenv("VWAP_STRICT_CONTINUATION_ENABLED", "false")
    _, signal = _evaluate(
        monkeypatch,
        enabled=False,
        execution_mode="SHADOW",
        low=100.0,
        prior_high=105.0,
    )
    assert signal is not None
    assert signal.metadata["vwap_strict_continuation_enabled"] is False


def test_strict_continuation_does_not_restrict_vwap_reclaim(monkeypatch):
    monkeypatch.setenv("VWAP_STRICT_CONTINUATION_ENABLED", "true")
    _, signal = _evaluate(
        monkeypatch,
        enabled=False,
        execution_mode="SHADOW",
        is_complete=False,
    )
    assert signal is not None
    assert signal.metadata["setup_type"] == "vwap_reclaim_momentum"
    assert signal.metadata["vwap_finalized_bar_required"] is False
