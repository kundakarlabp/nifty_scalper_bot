"""Research-only overfitting and uncertainty diagnostics.

These helpers never enter the live trading path. They quantify selection risk and
sampling uncertainty around already post-cost candidate return series.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
import math
import random
from statistics import NormalDist, mean, stdev


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    samples: int


@dataclass(frozen=True, slots=True)
class DeflatedSharpeReport:
    observations: int
    trials: int
    sharpe_ratio: float
    benchmark_sharpe: float
    expected_maximum_sharpe: float
    probabilistic_sharpe_ratio: float
    deflated_sharpe_ratio: float
    skewness: float
    kurtosis: float


@dataclass(frozen=True, slots=True)
class PBOFold:
    test_groups: tuple[int, ...]
    train_observations: int
    test_observations: int
    selected_candidate: str
    selected_train_score: float
    selected_test_score: float
    oos_rank_percentile: float
    oos_rank_logit: float


@dataclass(frozen=True, slots=True)
class PBOReport:
    candidate_count: int
    observation_count: int
    group_count: int
    test_groups_per_fold: int
    fold_count: int
    probability_backtest_overfitting: float
    folds: tuple[PBOFold, ...]


def _finite_values(values: Sequence[float], *, field: str) -> list[float]:
    resolved = [float(value) for value in values]
    if not resolved or any(not math.isfinite(value) for value in resolved):
        raise ValueError(f"{field} must contain finite values")
    return resolved


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapInterval:
    """Return a deterministic percentile-bootstrap interval for the mean."""

    resolved = _finite_values(values, field="values")
    count = int(samples)
    level = float(confidence)
    if count <= 0:
        raise ValueError("samples must be positive")
    if not 0.0 < level < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if len(resolved) == 1:
        value = resolved[0]
        return BootstrapInterval(value, value, value, level, count)

    rng = random.Random(int(seed))
    width = len(resolved)
    estimates = sorted(
        mean(resolved[rng.randrange(width)] for _ in range(width))
        for _ in range(count)
    )
    tail = (1.0 - level) / 2.0

    def percentile(probability: float) -> float:
        position = probability * (count - 1)
        lower_index = int(math.floor(position))
        upper_index = int(math.ceil(position))
        if lower_index == upper_index:
            return estimates[lower_index]
        weight = position - lower_index
        return (
            estimates[lower_index] * (1.0 - weight)
            + estimates[upper_index] * weight
        )

    return BootstrapInterval(
        estimate=round(mean(resolved), 8),
        lower=round(percentile(tail), 8),
        upper=round(percentile(1.0 - tail), 8),
        confidence=level,
        samples=count,
    )


def _distribution_moments(values: Sequence[float]) -> tuple[float, float]:
    resolved = _finite_values(values, field="returns")
    if len(resolved) < 3:
        return 0.0, 3.0
    center = mean(resolved)
    variance = sum((value - center) ** 2 for value in resolved) / len(resolved)
    if variance <= 0:
        return 0.0, 3.0
    sigma = math.sqrt(variance)
    skewness = mean(((value - center) / sigma) ** 3 for value in resolved)
    kurtosis = mean(((value - center) / sigma) ** 4 for value in resolved)
    return skewness, kurtosis


def _sample_sharpe(values: Sequence[float]) -> float:
    resolved = _finite_values(values, field="returns")
    if len(resolved) < 2:
        raise ValueError("at least two returns are required")
    dispersion = stdev(resolved)
    if dispersion <= 0:
        raise ValueError("returns must have non-zero dispersion")
    return mean(resolved) / dispersion


def probabilistic_sharpe_ratio(
    returns: Sequence[float],
    *,
    benchmark_sharpe: float = 0.0,
) -> float:
    """Probability that the observed non-annualised Sharpe exceeds a benchmark."""

    resolved = _finite_values(returns, field="returns")
    if len(resolved) < 3:
        raise ValueError("at least three returns are required")
    observed = _sample_sharpe(resolved)
    benchmark = float(benchmark_sharpe)
    if not math.isfinite(benchmark):
        raise ValueError("benchmark_sharpe must be finite")
    skewness, kurtosis = _distribution_moments(resolved)
    denominator = 1.0 - skewness * observed + ((kurtosis - 1.0) / 4.0) * (
        observed**2
    )
    if denominator <= 0:
        raise ValueError("probabilistic Sharpe denominator is non-positive")
    z_score = (observed - benchmark) * math.sqrt(len(resolved) - 1) / math.sqrt(
        denominator
    )
    return NormalDist().cdf(z_score)


def expected_maximum_sharpe(
    *,
    trials: int,
    sharpe_std: float,
) -> float:
    """Expected best Sharpe under repeated independent null trials."""

    count = int(trials)
    dispersion = float(sharpe_std)
    if count <= 1:
        return 0.0
    if not math.isfinite(dispersion) or dispersion <= 0:
        raise ValueError("sharpe_std must be finite and positive")
    normal = NormalDist()
    euler_gamma = 0.5772156649015329
    first = normal.inv_cdf(1.0 - 1.0 / count)
    second = normal.inv_cdf(1.0 - 1.0 / (count * math.e))
    return dispersion * ((1.0 - euler_gamma) * first + euler_gamma * second)


def deflated_sharpe_ratio(
    returns: Sequence[float],
    *,
    trials: int,
    trial_sharpe_std: float | None = None,
) -> DeflatedSharpeReport:
    """Return a multiple-testing/non-normality adjusted Sharpe diagnostic."""

    resolved = _finite_values(returns, field="returns")
    if len(resolved) < 3:
        raise ValueError("at least three returns are required")
    trial_count = int(trials)
    if trial_count <= 0:
        raise ValueError("trials must be positive")
    observed = _sample_sharpe(resolved)
    sharpe_std = (
        float(trial_sharpe_std)
        if trial_sharpe_std is not None
        else 1.0 / math.sqrt(len(resolved) - 1)
    )
    benchmark = expected_maximum_sharpe(
        trials=trial_count,
        sharpe_std=sharpe_std,
    )
    psr = probabilistic_sharpe_ratio(resolved, benchmark_sharpe=0.0)
    dsr = probabilistic_sharpe_ratio(resolved, benchmark_sharpe=benchmark)
    skewness, kurtosis = _distribution_moments(resolved)
    return DeflatedSharpeReport(
        observations=len(resolved),
        trials=trial_count,
        sharpe_ratio=round(observed, 8),
        benchmark_sharpe=round(benchmark, 8),
        expected_maximum_sharpe=round(benchmark, 8),
        probabilistic_sharpe_ratio=round(psr, 8),
        deflated_sharpe_ratio=round(dsr, 8),
        skewness=round(skewness, 8),
        kurtosis=round(kurtosis, 8),
    )


def _contiguous_groups(length: int, groups: int) -> tuple[tuple[int, ...], ...]:
    base, remainder = divmod(length, groups)
    result: list[tuple[int, ...]] = []
    cursor = 0
    for group in range(groups):
        width = base + (1 if group < remainder else 0)
        result.append(tuple(range(cursor, cursor + width)))
        cursor += width
    return tuple(result)


def _mean_score(values: Sequence[float]) -> float:
    return mean(values)


def _average_rank_percentile(
    selected_score: float,
    scores: Sequence[float],
) -> float:
    ordered = sorted(float(score) for score in scores)
    lower = sum(score < selected_score for score in ordered)
    equal = sum(score == selected_score for score in ordered)
    average_rank = lower + (equal + 1.0) / 2.0
    return average_rank / (len(ordered) + 1.0)


def combinatorial_purged_pbo(
    candidate_returns: Mapping[str, Sequence[float]],
    *,
    n_groups: int = 6,
    n_test_groups: int = 3,
    purge_observations: int = 0,
    embargo_observations: int = 0,
) -> PBOReport:
    """Estimate PBO from contiguous combinatorial train/test group splits.

    Purge/embargo are observation gaps around test blocks. For overlapping
    forward labels, callers should set them to at least the label horizon.
    """

    if len(candidate_returns) < 2:
        raise ValueError("at least two candidate return series are required")
    resolved = {
        str(name): _finite_values(values, field=f"candidate:{name}")
        for name, values in candidate_returns.items()
    }
    lengths = {len(values) for values in resolved.values()}
    if len(lengths) != 1:
        raise ValueError("candidate return series must have equal length")
    observation_count = next(iter(lengths))
    groups = int(n_groups)
    test_groups = int(n_test_groups)
    purge = int(purge_observations)
    embargo = int(embargo_observations)
    if groups < 2 or groups > observation_count:
        raise ValueError("n_groups must be between 2 and the observation count")
    if test_groups <= 0 or test_groups >= groups:
        raise ValueError("n_test_groups must be between 1 and n_groups - 1")
    if purge < 0 or embargo < 0:
        raise ValueError("purge_observations and embargo_observations must be non-negative")

    group_indices = _contiguous_groups(observation_count, groups)
    folds: list[PBOFold] = []
    candidates = tuple(sorted(resolved))
    all_indices = set(range(observation_count))
    for chosen in combinations(range(groups), test_groups):
        test_indices = {
            index for group in chosen for index in group_indices[group]
        }
        forbidden = set(test_indices)
        for group in chosen:
            block = group_indices[group]
            if not block:
                continue
            start, end = block[0], block[-1]
            forbidden.update(
                range(max(0, start - purge), min(observation_count, end + purge + 1))
            )
            forbidden.update(
                range(
                    end + 1,
                    min(observation_count, end + embargo + 1),
                )
            )
        train_indices = sorted(all_indices - forbidden)
        ordered_test = sorted(test_indices)
        if not train_indices or not ordered_test:
            continue

        train_scores = {
            candidate: _mean_score(
                [resolved[candidate][index] for index in train_indices]
            )
            for candidate in candidates
        }
        selected = max(candidates, key=lambda name: (train_scores[name], name))
        test_scores = {
            candidate: _mean_score(
                [resolved[candidate][index] for index in ordered_test]
            )
            for candidate in candidates
        }
        percentile = _average_rank_percentile(
            test_scores[selected],
            list(test_scores.values()),
        )
        logit = math.log(percentile / (1.0 - percentile))
        folds.append(
            PBOFold(
                test_groups=tuple(chosen),
                train_observations=len(train_indices),
                test_observations=len(ordered_test),
                selected_candidate=selected,
                selected_train_score=round(train_scores[selected], 8),
                selected_test_score=round(test_scores[selected], 8),
                oos_rank_percentile=round(percentile, 8),
                oos_rank_logit=round(logit, 8),
            )
        )

    if not folds:
        raise ValueError("purge/embargo removed every train/test fold")
    overfit = sum(fold.oos_rank_logit <= 0.0 for fold in folds)
    return PBOReport(
        candidate_count=len(candidates),
        observation_count=observation_count,
        group_count=groups,
        test_groups_per_fold=test_groups,
        fold_count=len(folds),
        probability_backtest_overfitting=round(overfit / len(folds), 8),
        folds=tuple(folds),
    )


__all__ = [
    "BootstrapInterval",
    "DeflatedSharpeReport",
    "PBOFold",
    "PBOReport",
    "bootstrap_mean_interval",
    "combinatorial_purged_pbo",
    "deflated_sharpe_ratio",
    "expected_maximum_sharpe",
    "probabilistic_sharpe_ratio",
]
