"""Adaptive walk-forward calibration utilities."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from math import sqrt
from statistics import mean, pstdev
from typing import Any, Deque

WalkForwardEvaluator = Callable[
    [Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]],
    Sequence[float],
]


@dataclass(frozen=True, slots=True)
class PerformanceSummary:
    """Net performance for one untouched walk-forward window."""

    trade_count: int
    total_net_pnl: float
    expectancy: float
    win_rate: float
    profit_factor: float | None
    max_drawdown: float

    @classmethod
    def from_pnl(cls, pnl: Sequence[float]) -> "PerformanceSummary":
        values = [float(value) for value in pnl]
        wins = [value for value in values if value > 0]
        losses = [value for value in values if value < 0]
        equity = 0.0
        peak = 0.0
        max_drawdown = 0.0
        for value in values:
            equity += value
            peak = max(peak, equity)
            max_drawdown = max(max_drawdown, peak - equity)
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        return cls(
            trade_count=len(values),
            total_net_pnl=round(sum(values), 2),
            expectancy=round(mean(values), 4) if values else 0.0,
            win_rate=round(len(wins) / len(values), 4) if values else 0.0,
            profit_factor=(
                round(gross_profit / gross_loss, 4) if gross_loss > 0 else None
            ),
            max_drawdown=round(max_drawdown, 2),
        )


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    """One ordered train/validation/test comparison."""

    fold: int
    train_start: Any
    train_end: Any
    validation_start: Any
    validation_end: Any
    test_start: Any
    test_end: Any
    selected_candidate: str
    selected_validation: PerformanceSummary
    baseline_test: PerformanceSummary
    candidate_test: PerformanceSummary


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    """All untouched fold comparisons and their aggregate evidence."""

    folds: tuple[WalkForwardFold, ...]
    aggregate_baseline: PerformanceSummary
    aggregate_candidate: PerformanceSummary


@dataclass(frozen=True, slots=True)
class ChronologicalWalkForward:
    """Evaluate candidates without random splits or test-window look-ahead."""

    train_size: int
    validation_size: int
    test_size: int
    step_size: int | None = None
    timestamp_field: str = "timestamp"

    def __post_init__(self) -> None:
        for name in ("train_size", "validation_size", "test_size"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.step_size is not None and int(self.step_size) <= 0:
            raise ValueError("step_size must be positive")

    def evaluate(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        baseline: WalkForwardEvaluator,
        candidates: Mapping[str, WalkForwardEvaluator],
    ) -> WalkForwardResult:
        """Select on validation and compare only on the following test slice."""

        observations = list(records)
        if not candidates:
            raise ValueError("at least one candidate is required")
        timestamps: list[Any] = []
        for record in observations:
            timestamp = record.get(self.timestamp_field)
            if timestamp is None:
                raise ValueError(f"every record requires {self.timestamp_field}")
            timestamps.append(timestamp)
        if any(
            current >= following
            for current, following in zip(timestamps, timestamps[1:])
        ):
            raise ValueError("records must be in strict chronological order")

        fold_results: list[WalkForwardFold] = []
        aggregate_baseline: list[float] = []
        aggregate_candidate: list[float] = []
        start = 0
        step = int(self.step_size or self.test_size)
        required = self.train_size + self.validation_size + self.test_size
        while start + required <= len(observations):
            train_end = start + self.train_size
            validation_end = train_end + self.validation_size
            test_end = validation_end + self.test_size
            train = observations[start:train_end]
            validation = observations[train_end:validation_end]
            test = observations[validation_end:test_end]

            validation_results = {
                name: (
                    PerformanceSummary.from_pnl(evaluator(train, validation)),
                    evaluator,
                )
                for name, evaluator in sorted(candidates.items())
            }
            selected_name, (selected_validation, selected_evaluator) = max(
                validation_results.items(),
                key=lambda item: (
                    item[1][0].expectancy,
                    item[1][0].total_net_pnl,
                    item[0],
                ),
            )
            fit_for_test = [*train, *validation]
            baseline_test_pnl = list(baseline(fit_for_test, test))
            candidate_test_pnl = list(selected_evaluator(fit_for_test, test))
            aggregate_baseline.extend(float(value) for value in baseline_test_pnl)
            aggregate_candidate.extend(float(value) for value in candidate_test_pnl)
            fold_results.append(
                WalkForwardFold(
                    fold=len(fold_results) + 1,
                    train_start=timestamps[start],
                    train_end=timestamps[train_end - 1],
                    validation_start=timestamps[train_end],
                    validation_end=timestamps[validation_end - 1],
                    test_start=timestamps[validation_end],
                    test_end=timestamps[test_end - 1],
                    selected_candidate=selected_name,
                    selected_validation=selected_validation,
                    baseline_test=PerformanceSummary.from_pnl(baseline_test_pnl),
                    candidate_test=PerformanceSummary.from_pnl(candidate_test_pnl),
                )
            )
            start += step

        if not fold_results:
            raise ValueError("insufficient records for one walk-forward fold")
        return WalkForwardResult(
            folds=tuple(fold_results),
            aggregate_baseline=PerformanceSummary.from_pnl(aggregate_baseline),
            aggregate_candidate=PerformanceSummary.from_pnl(aggregate_candidate),
        )


@dataclass(slots=True)
class TradeStats:
    """Rolling trade metrics. Args: None. Returns: None. Raises: None."""

    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    rolling_sharpe: float = 0.0
    max_drawdown: float = 0.0
    signal_frequency: float = 0.0


@dataclass(slots=True)
class AdaptiveParameterStore:
    """Store rolling per-strategy trade statistics."""

    window_trades: int = 200
    _pnl: dict[str, Deque[float]] = field(default_factory=dict)
    _stats: dict[str, TradeStats] = field(default_factory=dict)

    def record_trade(self, strategy: str, pnl: float) -> TradeStats:
        """Record strategy P&L and return its updated statistics."""

        bucket = self._pnl.setdefault(
            strategy, deque(maxlen=max(1, self.window_trades))
        )
        bucket.append(float(pnl))
        values = list(bucket)
        wins = [v for v in values if v > 0]
        losses = [abs(v) for v in values if v < 0]
        win_rate = len(wins) / max(len(values), 1)
        avg_win = mean(wins) if wins else 0.0
        avg_loss = mean(losses) if losses else 0.0
        sharpe = 0.0
        if len(values) >= 2:
            std = pstdev(values)
            if std > 0:
                sharpe = (mean(values) / std) * sqrt(len(values))
        eq = 0.0
        peak = 0.0
        dd = 0.0
        for v in values:
            eq += v
            peak = max(peak, eq)
            dd = max(dd, peak - eq)
        stats = TradeStats(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            rolling_sharpe=sharpe,
            max_drawdown=dd,
            signal_frequency=len(values) / max(float(self.window_trades), 1.0),
        )
        self._stats[strategy] = stats
        return stats

    def get_stats(self, strategy: str) -> TradeStats:
        """Get current stats. Args: strategy. Returns: TradeStats. Raises: None."""

        return self._stats.get(strategy, TradeStats())


@dataclass(slots=True)
class WalkForwardOptimizer:
    """Research-only parameter tuner; live updates are disabled by default."""

    recalibrate_every: int = 50
    alpha: float = 0.2
    drawdown_threshold: float = 0.0
    allow_parameter_updates: bool = False
    _trade_count: dict[str, int] = field(default_factory=dict)
    _params: dict[str, dict[str, float]] = field(default_factory=dict)
    _regime_params: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    _frozen_strategies: set[str] = field(default_factory=set)
    risk_scale: float = 1.0

    def should_recalibrate(self, strategy: str) -> bool:
        """Check cadence. Args: strategy. Returns: bool. Raises: None."""

        count = self._trade_count.get(strategy, 0) + 1
        self._trade_count[strategy] = count
        return count % max(1, self.recalibrate_every) == 0

    def optimize(
        self,
        strategy: str,
        regime: str,
        stats: TradeStats,
        current: dict[str, float],
    ) -> dict[str, float]:
        """Return research-tuned parameters when updates are explicitly enabled."""

        if not self.allow_parameter_updates:
            return current
        if strategy in self._frozen_strategies or stats.rolling_sharpe < 0:
            self._frozen_strategies.add(strategy)
            return current
        if self.drawdown_threshold > 0 and stats.max_drawdown > self.drawdown_threshold:
            self.risk_scale = min(self.risk_scale, 0.5)

        mz = current.get("momentum_z_threshold", 0.5)
        mv = current.get("microvol_percentile", 60.0)
        sp = current.get("spread_threshold_pct", 0.3)
        candidates: list[dict[str, float]] = []
        for dm in (-0.1, 0.0, 0.1):
            for dv in (-5.0, 0.0, 5.0):
                for ds in (-0.05, 0.0, 0.05):
                    candidates.append(
                        {
                            "momentum_z_threshold": max(0.1, mz + dm),
                            "microvol_percentile": min(95.0, max(5.0, mv + dv)),
                            "spread_threshold_pct": max(0.01, sp + ds),
                        }
                    )
        best = max(candidates, key=lambda p: self._objective(stats, p))
        prev = self._params.get(strategy, current)
        blended = {
            key: (1.0 - self.alpha) * float(prev.get(key, value))
            + self.alpha * float(value)
            for key, value in best.items()
        }
        self._params[strategy] = blended
        self._regime_params.setdefault(strategy, {})[regime] = blended
        return blended

    def on_regime_change(
        self, strategy: str, regime: str, current: dict[str, float]
    ) -> dict[str, float]:
        """Load research parameters for a regime when explicitly enabled."""

        if not self.allow_parameter_updates:
            return current
        target = self._regime_params.get(strategy, {}).get(regime)
        if not target:
            return current
        return {
            key: (1.0 - self.alpha) * float(current.get(key, value))
            + self.alpha * float(value)
            for key, value in target.items()
        }

    @staticmethod
    def _objective(stats: TradeStats, params: dict[str, float]) -> float:
        """Objective score. Args: stats,params. Returns: float. Raises: None."""

        spread_penalty = params["spread_threshold_pct"] * 0.2
        return stats.rolling_sharpe + stats.win_rate - spread_penalty
