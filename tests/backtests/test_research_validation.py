from __future__ import annotations

import pytest

from nifty_scalper_bot.backtesting.research_validation import (
    bootstrap_mean_interval,
    combinatorial_purged_pbo,
    deflated_sharpe_ratio,
    expected_maximum_sharpe,
)


def test_bootstrap_mean_interval_is_deterministic_and_contains_estimate() -> None:
    first = bootstrap_mean_interval([1.0, 2.0, 3.0, 4.0], samples=500, seed=17)
    repeated = bootstrap_mean_interval([1.0, 2.0, 3.0, 4.0], samples=500, seed=17)

    assert first == repeated
    assert first.lower <= first.estimate <= first.upper
    assert first.estimate == 2.5


def test_deflated_sharpe_penalizes_multiple_trials() -> None:
    returns = [1.0, 0.5, -0.2, 1.2, 0.4, 0.8, -0.1, 1.1, 0.3, 0.7]

    report = deflated_sharpe_ratio(returns, trials=25)

    assert report.expected_maximum_sharpe > 0.0
    assert (
        0.0
        <= report.deflated_sharpe_ratio
        <= report.probabilistic_sharpe_ratio
        <= 1.0
    )
    assert report.observations == len(returns)
    assert report.trials == 25


def test_expected_maximum_sharpe_is_zero_for_single_trial() -> None:
    assert expected_maximum_sharpe(trials=1, sharpe_std=0.2) == 0.0


def test_combinatorial_pbo_detects_train_winner_that_reverses_oos() -> None:
    report = combinatorial_purged_pbo(
        {
            "first_half": [5.0, 5.0, 5.0, -5.0, -5.0, -5.0],
            "second_half": [-5.0, -5.0, -5.0, 5.0, 5.0, 5.0],
        },
        n_groups=6,
        n_test_groups=3,
    )

    assert report.fold_count == 20
    assert report.probability_backtest_overfitting == 1.0
    assert all(fold.oos_rank_logit < 0.0 for fold in report.folds)


def test_combinatorial_pbo_applies_purge_and_embargo_to_training_only() -> None:
    report = combinatorial_purged_pbo(
        {
            "a": [float(index) for index in range(12)],
            "b": [float(12 - index) for index in range(12)],
        },
        n_groups=6,
        n_test_groups=2,
        purge_observations=1,
        embargo_observations=1,
    )

    assert report.fold_count > 0
    assert all(fold.test_observations == 4 for fold in report.folds)
    assert all(fold.train_observations < 8 for fold in report.folds)


def test_research_validation_rejects_mismatched_candidate_lengths() -> None:
    with pytest.raises(ValueError, match="equal length"):
        combinatorial_purged_pbo(
            {"a": [1.0, 2.0], "b": [1.0]},
            n_groups=2,
            n_test_groups=1,
        )
