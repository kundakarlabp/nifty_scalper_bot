from __future__ import annotations

import pytest

from nifty_scalper_bot.core.adaptive_calibration import (
    AdaptiveParameterStore,
    ChronologicalWalkForward,
    WalkForwardOptimizer,
)


def test_adaptive_recalibration_trigger() -> None:
    opt = WalkForwardOptimizer(recalibrate_every=3)
    assert opt.should_recalibrate("s1") is False
    assert opt.should_recalibrate("s1") is False
    assert opt.should_recalibrate("s1") is True


def test_regime_change_blend() -> None:
    store = AdaptiveParameterStore(window_trades=10)
    stats = store.record_trade("s1", 10.0)
    opt = WalkForwardOptimizer(recalibrate_every=1, allow_parameter_updates=True)
    current = {
        "momentum_z_threshold": 1.0,
        "microvol_percentile": 60.0,
        "spread_threshold_pct": 0.2,
    }
    tuned = opt.optimize("s1", "trend", stats, current)
    shifted = opt.on_regime_change("s1", "trend", current)
    assert shifted.keys() == tuned.keys()


def test_optimizer_is_research_only_by_default() -> None:
    stats = AdaptiveParameterStore(window_trades=10).record_trade("s1", 10.0)
    current = {
        "momentum_z_threshold": 1.0,
        "microvol_percentile": 60.0,
        "spread_threshold_pct": 0.2,
    }
    opt = WalkForwardOptimizer(recalibrate_every=1)

    assert opt.optimize("s1", "trend", stats, current) == current
    assert opt._params == {}
    assert opt._regime_params == {}


def test_disabled_optimizer_does_not_load_cached_regime_parameters() -> None:
    stats = AdaptiveParameterStore(window_trades=10).record_trade("s1", 10.0)
    current = {
        "momentum_z_threshold": 1.0,
        "microvol_percentile": 60.0,
        "spread_threshold_pct": 0.2,
    }
    opt = WalkForwardOptimizer(allow_parameter_updates=True)
    opt.optimize("s1", "trend", stats, current)
    opt.allow_parameter_updates = False

    assert opt.on_regime_change("s1", "trend", current) == current


def test_negative_strategy_does_not_freeze_other_research_strategy() -> None:
    store = AdaptiveParameterStore(window_trades=10)
    store.record_trade("loser", -5.0)
    losing = store.record_trade("loser", -10.0)
    winning = store.record_trade("winner", 10.0)
    current = {
        "momentum_z_threshold": 1.0,
        "microvol_percentile": 60.0,
        "spread_threshold_pct": 0.2,
    }
    opt = WalkForwardOptimizer(recalibrate_every=1, allow_parameter_updates=True)

    assert opt.optimize("loser", "trend", losing, current) == current
    tuned = opt.optimize("winner", "trend", winning, current)

    assert "loser" in opt._frozen_strategies
    assert "winner" not in opt._frozen_strategies
    assert tuned != current


def test_chronological_walk_forward_keeps_test_untouched() -> None:
    records = [
        {
            "timestamp": float(index),
            "baseline": 1.0,
            "stable": 2.0,
            "overfit": 3.0 if index < 6 else -5.0,
        }
        for index in range(10)
    ]
    seen: list[tuple[str, float, float]] = []

    def evaluator(name: str):
        def _evaluate(fit, evaluation):
            seen.append((name, fit[-1]["timestamp"], evaluation[0]["timestamp"]))
            return [row[name] for row in evaluation]

        return _evaluate

    result = ChronologicalWalkForward(
        train_size=4,
        validation_size=2,
        test_size=2,
    ).evaluate(
        records,
        baseline=evaluator("baseline"),
        candidates={
            "stable": evaluator("stable"),
            "overfit": evaluator("overfit"),
        },
    )

    assert len(result.folds) == 2
    assert result.folds[0].selected_candidate == "overfit"
    assert result.folds[0].candidate_test.total_net_pnl == -10.0
    assert result.folds[0].baseline_test.total_net_pnl == 2.0
    assert result.folds[1].selected_candidate == "stable"
    assert result.folds[1].candidate_test.total_net_pnl == 4.0
    assert result.aggregate_candidate.total_net_pnl == -6.0
    assert result.aggregate_baseline.total_net_pnl == 4.0
    assert all(fit_end < evaluation_start for _, fit_end, evaluation_start in seen)


def test_chronological_walk_forward_rejects_unsorted_records() -> None:
    records = [{"timestamp": 2.0}, {"timestamp": 1.0}]

    def evaluator(_fit, evaluation):
        return [0.0 for _ in evaluation]

    with pytest.raises(ValueError, match="chronological"):
        ChronologicalWalkForward(1, 1, 1).evaluate(
            records,
            baseline=evaluator,
            candidates={"candidate": evaluator},
        )
