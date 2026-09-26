from __future__ import annotations

import pytest

from nifty_scalper_bot.core.adaptive_calibration import (
    AdaptiveParameterStore,
    ChronologicalWalkForward,
    PerformanceSummary,
    TradeStats,
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
    tuned = opt.optimize(
        "s1",
        "trend",
        stats,
        current,
        candidate_evaluator=lambda params: -abs(params["momentum_z_threshold"] - 0.9),
    )
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
    opt.optimize(
        "s1",
        "trend",
        stats,
        current,
        candidate_evaluator=lambda params: -params["spread_threshold_pct"],
    )
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

    assert (
        opt.optimize(
            "loser",
            "trend",
            losing,
            current,
            candidate_evaluator=lambda _params: 0.0,
        )
        == current
    )
    tuned = opt.optimize(
        "winner",
        "trend",
        winning,
        current,
        candidate_evaluator=lambda params: -abs(params["momentum_z_threshold"] - 0.9),
    )

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
        min_validation_trades=1,
        min_validation_expectancy_improvement=0.0,
    ).evaluate(
        records,
        baseline=evaluator("baseline"),
        candidates={
            "stable": evaluator("stable"),
            "overfit": evaluator("overfit"),
        },
    )

    assert len(result.folds) == 2
    assert result.folds[0].baseline_validation.total_net_pnl == 2.0
    assert result.folds[0].validation_candidate_count == 2
    assert result.folds[0].eligible_validation_candidate_count == 2
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
        ChronologicalWalkForward(
            1,
            1,
            1,
            min_validation_trades=1,
            min_validation_expectancy_improvement=0.0,
        ).evaluate(
            records,
            baseline=evaluator,
            candidates={"candidate": evaluator},
        )


def test_enabled_optimizer_still_fails_closed_without_candidate_evaluator() -> None:
    stats = AdaptiveParameterStore(window_trades=10).record_trade("s1", 10.0)
    current = {
        "momentum_z_threshold": 1.0,
        "microvol_percentile": 60.0,
        "spread_threshold_pct": 0.2,
    }
    opt = WalkForwardOptimizer(recalibrate_every=1, allow_parameter_updates=True)

    assert opt.optimize("s1", "trend", stats, current) == current
    assert opt._params == {}
    assert opt._regime_params == {}
    assert opt._frozen_strategies == set()
    assert opt.risk_scale == 1.0


def test_optimizer_uses_candidate_specific_evaluator() -> None:
    stats = AdaptiveParameterStore(window_trades=10).record_trade("s1", 10.0)
    current = {
        "momentum_z_threshold": 1.0,
        "microvol_percentile": 60.0,
        "spread_threshold_pct": 0.2,
    }
    seen: list[tuple[float, float, float]] = []
    opt = WalkForwardOptimizer(
        recalibrate_every=1,
        alpha=1.0,
        allow_parameter_updates=True,
    )

    def candidate_evaluator(params):
        seen.append(
            (
                params["momentum_z_threshold"],
                params["microvol_percentile"],
                params["spread_threshold_pct"],
            )
        )
        return -(
            abs(params["momentum_z_threshold"] - 0.9)
            + abs(params["microvol_percentile"] - 55.0) / 10.0
            + abs(params["spread_threshold_pct"] - 0.15)
        )

    tuned = opt.optimize(
        "s1",
        "trend",
        stats,
        current,
        candidate_evaluator=candidate_evaluator,
    )

    assert len(seen) == 27
    assert tuned == {
        "momentum_z_threshold": pytest.approx(0.9),
        "microvol_percentile": pytest.approx(55.0),
        "spread_threshold_pct": pytest.approx(0.15),
    }


def test_optimizer_does_not_drift_on_equal_candidate_scores() -> None:
    stats = AdaptiveParameterStore(window_trades=10).record_trade("s1", 10.0)
    current = {
        "momentum_z_threshold": 1.0,
        "microvol_percentile": 60.0,
        "spread_threshold_pct": 0.2,
        "unrelated_research_field": 42.0,
    }
    opt = WalkForwardOptimizer(
        recalibrate_every=1,
        alpha=1.0,
        allow_parameter_updates=True,
    )

    tuned = opt.optimize(
        "s1",
        "trend",
        stats,
        current,
        candidate_evaluator=lambda _params: 1.0,
    )

    assert tuned == current
    assert opt._params == {}
    assert opt._regime_params == {}


def test_walk_forward_rejects_zero_trade_validation_candidate() -> None:
    records = [{"timestamp": float(index)} for index in range(6)]
    baseline_test_calls = 0
    empty_test_calls = 0

    def baseline(_fit, evaluation):
        nonlocal baseline_test_calls
        if evaluation[0]["timestamp"] >= 4.0:
            baseline_test_calls += 1
        return [-1.0 for _ in evaluation]

    def empty_candidate(_fit, evaluation):
        nonlocal empty_test_calls
        if evaluation[0]["timestamp"] >= 4.0:
            empty_test_calls += 1
        return []

    result = ChronologicalWalkForward(
        train_size=2,
        validation_size=2,
        test_size=2,
        min_validation_trades=1,
        min_validation_expectancy_improvement=0.0,
    ).evaluate(
        records,
        baseline=baseline,
        candidates={"empty": empty_candidate},
    )

    assert result.folds[0].selected_candidate == "baseline"
    assert result.folds[0].baseline_validation.trade_count == 2
    assert result.folds[0].validation_candidate_count == 1
    assert result.folds[0].eligible_validation_candidate_count == 0
    assert result.folds[0].selected_validation.trade_count == 2
    assert result.folds[0].candidate_test == result.folds[0].baseline_test
    assert result.aggregate_candidate == result.aggregate_baseline
    assert baseline_test_calls == 1
    assert empty_test_calls == 0


def test_walk_forward_requires_validation_improvement_over_baseline() -> None:
    records = [{"timestamp": float(index)} for index in range(6)]
    weaker_test_calls = 0

    def baseline(_fit, evaluation):
        return [1.0 for _ in evaluation]

    def weaker_candidate(_fit, evaluation):
        nonlocal weaker_test_calls
        if evaluation[0]["timestamp"] >= 4.0:
            weaker_test_calls += 1
            return [100.0 for _ in evaluation]
        return [0.5 for _ in evaluation]

    result = ChronologicalWalkForward(
        train_size=2,
        validation_size=2,
        test_size=2,
        min_validation_trades=1,
        min_validation_expectancy_improvement=0.0,
    ).evaluate(
        records,
        baseline=baseline,
        candidates={"weaker": weaker_candidate},
    )

    assert result.folds[0].selected_candidate == "baseline"
    assert result.folds[0].selected_validation.expectancy == 1.0
    assert result.folds[0].candidate_test.total_net_pnl == 2.0
    assert result.aggregate_candidate.total_net_pnl == 2.0
    assert weaker_test_calls == 0


def test_walk_forward_rejects_overlapping_test_windows() -> None:
    with pytest.raises(ValueError, match="step_size must be at least test_size"):
        ChronologicalWalkForward(
            train_size=4,
            validation_size=2,
            test_size=3,
            min_validation_trades=1,
        min_validation_expectancy_improvement=0.0,
            step_size=2,
        )


def test_walk_forward_respects_declared_validation_trade_floor() -> None:
    records = [{"timestamp": float(index)} for index in range(6)]
    sparse_test_calls = 0

    def baseline(_fit, evaluation):
        return [1.0 for _ in evaluation]

    def sparse_candidate(_fit, evaluation):
        nonlocal sparse_test_calls
        if evaluation[0]["timestamp"] >= 4.0:
            sparse_test_calls += 1
            return [100.0]
        return [10.0]

    result = ChronologicalWalkForward(
        train_size=2,
        validation_size=2,
        test_size=2,
        min_validation_trades=2,
        min_validation_expectancy_improvement=0.0,
    ).evaluate(
        records,
        baseline=baseline,
        candidates={"sparse": sparse_candidate},
    )

    assert result.folds[0].selected_candidate == "baseline"
    assert result.folds[0].validation_candidate_count == 1
    assert result.folds[0].eligible_validation_candidate_count == 0
    assert result.folds[0].candidate_test == result.folds[0].baseline_test
    assert sparse_test_calls == 0


def test_walk_forward_rejects_nonpositive_validation_trade_floor() -> None:
    with pytest.raises(ValueError, match="min_validation_trades must be positive"):
        ChronologicalWalkForward(
            train_size=2,
            validation_size=2,
            test_size=2,
            min_validation_trades=0,
        min_validation_expectancy_improvement=0.0,
        )


@pytest.mark.parametrize("bad_pnl", [float("nan"), float("inf"), float("-inf")])
def test_performance_summary_rejects_nonfinite_pnl(bad_pnl: float) -> None:
    with pytest.raises(ValueError, match="pnl values must be finite"):
        PerformanceSummary.from_pnl([1.0, bad_pnl])


@pytest.mark.parametrize("bad_pnl", [float("nan"), float("inf"), float("-inf")])
def test_adaptive_store_rejects_nonfinite_pnl(bad_pnl: float) -> None:
    store = AdaptiveParameterStore(window_trades=10)

    with pytest.raises(ValueError, match="pnl must be finite"):
        store.record_trade("s1", bad_pnl)

    assert store.get_stats("s1") == TradeStats()


def test_optimizer_fails_closed_on_nonfinite_candidate_score() -> None:
    stats = TradeStats(
        win_rate=0.6,
        rolling_sharpe=1.0,
        max_drawdown=10.0,
    )
    current = {
        "momentum_z_threshold": 1.0,
        "microvol_percentile": 60.0,
        "spread_threshold_pct": 0.2,
    }
    opt = WalkForwardOptimizer(
        alpha=1.0,
        drawdown_threshold=1.0,
        allow_parameter_updates=True,
    )

    tuned = opt.optimize(
        "s1",
        "trend",
        stats,
        current,
        candidate_evaluator=lambda _params: float("nan"),
    )

    assert tuned == current
    assert opt._params == {}
    assert opt._regime_params == {}
    assert opt.risk_scale == 1.0


def test_walk_forward_requires_declared_expectancy_margin() -> None:
    records = [{"timestamp": float(index)} for index in range(6)]
    candidate_test_calls = 0

    def baseline(_fit, evaluation):
        return [1.0 for _ in evaluation]

    def marginal_candidate(_fit, evaluation):
        nonlocal candidate_test_calls
        if evaluation[0]["timestamp"] >= 4.0:
            candidate_test_calls += 1
            return [100.0 for _ in evaluation]
        return [1.04 for _ in evaluation]

    result = ChronologicalWalkForward(
        train_size=2,
        validation_size=2,
        test_size=2,
        min_validation_trades=1,
        min_validation_expectancy_improvement=0.05,
    ).evaluate(
        records,
        baseline=baseline,
        candidates={"marginal": marginal_candidate},
    )

    assert result.folds[0].selected_candidate == "baseline"
    assert result.folds[0].candidate_test == result.folds[0].baseline_test
    assert candidate_test_calls == 0


def test_walk_forward_selects_candidate_clearing_expectancy_margin() -> None:
    records = [{"timestamp": float(index)} for index in range(6)]

    def baseline(_fit, evaluation):
        return [1.0 for _ in evaluation]

    def candidate(_fit, evaluation):
        if evaluation[0]["timestamp"] >= 4.0:
            return [2.0 for _ in evaluation]
        return [1.06 for _ in evaluation]

    result = ChronologicalWalkForward(
        train_size=2,
        validation_size=2,
        test_size=2,
        min_validation_trades=1,
        min_validation_expectancy_improvement=0.05,
    ).evaluate(
        records,
        baseline=baseline,
        candidates={"candidate": candidate},
    )

    assert result.folds[0].selected_candidate == "candidate"
    assert result.folds[0].candidate_test.total_net_pnl == 4.0


@pytest.mark.parametrize(
    "bad_margin",
    [-0.01, float("nan"), float("inf"), float("-inf")],
)
def test_walk_forward_rejects_invalid_expectancy_margin(bad_margin: float) -> None:
    with pytest.raises(
        ValueError,
        match="min_validation_expectancy_improvement must be finite and non-negative",
    ):
        ChronologicalWalkForward(
            train_size=2,
            validation_size=2,
            test_size=2,
            min_validation_trades=1,
            min_validation_expectancy_improvement=bad_margin,
        )
