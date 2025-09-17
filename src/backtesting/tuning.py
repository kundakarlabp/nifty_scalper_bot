"""Hyper-parameter tuning utilities built on top of the backtest engine."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from src.backtesting.backtest_engine import BacktestEngine
from src.backtesting.data_feed import SpotFeed
from src.backtesting.metrics import reject
from src.strategies.parameters import (
    ParameterBound,
    StrategyParameterSpace,
    StrategyParameters,
)


def _normal_pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class BayesianOptimizer:
    """Lightweight sequential model-based optimiser using an RBF kernel."""

    def __init__(
        self,
        bounds: Sequence[ParameterBound],
        *,
        n_initial: int = 5,
        random_state: Optional[int] = None,
        xi: float = 0.01,
        length_scale: float = 0.25,
        noise: float = 1e-6,
        n_candidates: int = 128,
    ) -> None:
        self.bounds = list(bounds)
        self.dim = len(self.bounds)
        self.n_initial = max(1, n_initial)
        self.xi = float(xi)
        self.length_scale = float(length_scale)
        self.noise = float(noise)
        self.n_candidates = max(16, int(n_candidates))
        self._rng = np.random.default_rng(random_state)
        self._xs: list[np.ndarray] = []
        self._ys: list[float] = []
        self._L: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None

    def suggest(self) -> np.ndarray:
        if len(self._xs) < self.n_initial:
            return self._random()
        self._fit()
        candidates = self._rng.uniform(0.0, 1.0, size=(self.n_candidates, self.dim))
        scores = [self._expected_improvement(x) for x in candidates]
        best_idx = int(np.argmax(scores))
        if scores[best_idx] <= 0:
            return self._random()
        return candidates[best_idx]

    def observe(self, x: Iterable[float], y: float) -> None:
        arr = np.clip(np.asarray(list(x), dtype=float), 0.0, 1.0)
        if arr.shape != (self.dim,):
            raise ValueError("Observation dimensionality mismatch")
        self._xs.append(arr)
        self._ys.append(float(y))

    def _random(self) -> np.ndarray:
        return self._rng.uniform(0.0, 1.0, size=self.dim)

    def _fit(self) -> None:
        if not self._xs:
            return
        X = np.vstack(self._xs)
        y = np.asarray(self._ys)
        K = self._kernel(X, X)
        K.flat[:: K.shape[0] + 1] += self.noise
        self._L = np.linalg.cholesky(K)
        self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, y))

    def _kernel(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        sqdist = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2)
        denom = 2.0 * (self.length_scale ** 2)
        return np.exp(-sqdist / denom)

    def _posterior(self, x: np.ndarray) -> tuple[float, float]:
        if not self._xs:
            return 0.0, 1.0
        assert self._L is not None and self._alpha is not None
        K_trans = self._kernel(np.vstack(self._xs), x.reshape(1, -1)).flatten()
        mean = float(K_trans @ self._alpha)
        v = np.linalg.solve(self._L, K_trans)
        var = max(1e-12, 1.0 - float(v @ v))
        return mean, math.sqrt(var)

    def _expected_improvement(self, x: np.ndarray) -> float:
        mean, sigma = self._posterior(x)
        best = max(self._ys) if self._ys else 0.0
        if sigma <= 1e-9:
            return 0.0
        delta = mean - best - self.xi
        z = delta / sigma
        return max(delta * _normal_cdf(z) + sigma * _normal_pdf(z), 0.0)


@dataclass
class TrialResult:
    params: StrategyParameters
    score: float
    summary: Dict[str, Any]


@dataclass
class TuningResult:
    best: TrialResult
    trials: List[TrialResult] = field(default_factory=list)


class BacktestTuner:
    """Couple a parameter space with a backtest-based objective."""

    def __init__(
        self,
        space: StrategyParameterSpace,
        objective: Callable[[StrategyParameters], tuple[float, Dict[str, Any]]],
        *,
        maximize: bool = True,
        random_state: Optional[int] = None,
        initial_samples: int = 5,
        xi: float = 0.01,
    ) -> None:
        self.space = space
        self.objective = objective
        self.maximize = maximize
        self._optimizer = BayesianOptimizer(
            space.bounds,
            n_initial=initial_samples,
            random_state=random_state,
            xi=xi,
        )
        self._trials: List[TrialResult] = []

    def tune(self, n_trials: int) -> TuningResult:
        best: Optional[TrialResult] = None
        for _ in range(max(1, n_trials)):
            proposal = self._optimizer.suggest()
            params = self.space.from_vector(proposal)
            norm = self.space.to_vector(params)
            score, summary = self.objective(params)
            signed_score = score if self.maximize else -score
            self._optimizer.observe(norm, signed_score)
            trial = TrialResult(params=params, score=score, summary=summary)
            self._trials.append(trial)
            if best is None:
                best = trial
            else:
                better = score > best.score if self.maximize else score < best.score
                if better:
                    best = trial
        if best is None:
            raise RuntimeError("No trials executed")
        return TuningResult(best=best, trials=list(self._trials))


@dataclass
class WalkForwardSplit:
    train_start: Optional[str]
    train_end: Optional[str]
    test_start: Optional[str]
    test_end: Optional[str]


@dataclass
class WalkForwardResult:
    split: WalkForwardSplit
    tuning: TuningResult
    test_summary: Dict[str, Any]


class WalkForwardValidator:
    """Run walk-forward optimisation using :class:`BacktestTuner`."""

    def __init__(
        self,
        feed: SpotFeed,
        space: StrategyParameterSpace,
        engine_factory: Callable[[SpotFeed, str, bool], BacktestEngine],
        *,
        metric: str = "PF",
        maximize: bool = True,
    ) -> None:
        self.feed = feed
        self.space = space
        self.engine_factory = engine_factory
        self.metric = metric
        self.maximize = maximize

    def run(
        self,
        splits: Sequence[WalkForwardSplit],
        *,
        trials: int,
        outdir: str,
        tuner_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[WalkForwardResult]:
        os.makedirs(outdir, exist_ok=True)
        results: List[WalkForwardResult] = []
        for idx, split in enumerate(splits, 1):
            fold_dir = os.path.join(outdir, f"wf_{idx:02d}")
            os.makedirs(fold_dir, exist_ok=True)
            train_feed = self.feed.window(split.train_start, split.train_end)
            train_dir = os.path.join(fold_dir, "train")
            os.makedirs(train_dir, exist_ok=True)

            def _objective(params: StrategyParameters) -> tuple[float, Dict[str, Any]]:
                engine = self.engine_factory(train_feed, train_dir, False)
                summary = engine.run(
                    start=split.train_start,
                    end=split.train_end,
                    params=params,
                )
                if reject(summary):
                    return -1e6, summary
                score = float(summary.get(self.metric, 0.0))
                return score, summary

            tuner = BacktestTuner(
                self.space,
                _objective,
                maximize=self.maximize,
                **(tuner_kwargs or {}),
            )
            tuning_result = tuner.tune(trials)

            test_feed = self.feed.window(split.test_start, split.test_end)
            test_dir = os.path.join(fold_dir, "test")
            os.makedirs(test_dir, exist_ok=True)
            engine = self.engine_factory(test_feed, test_dir, True)
            summary = engine.run(
                start=split.test_start,
                end=split.test_end,
                params=tuning_result.best.params,
            )
            results.append(
                WalkForwardResult(
                    split=split,
                    tuning=tuning_result,
                    test_summary=summary,
                )
            )
        return results
