from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd

from src.backtesting.data_feed import SpotFeed
from src.backtesting.tuning import WalkForwardSplit, WalkForwardValidator, BacktestTuner
from src.strategies.parameters import (
    ParameterBound,
    StrategyParameterSpace,
    StrategyParameters,
)


def test_strategy_parameter_space_clamps_bounds() -> None:
    bounds = [
        ParameterBound("ema_fast", 5, 20, True),
        ParameterBound("ema_slow", 10, 40, True),
        ParameterBound("atr_period", 5, 20, True),
        ParameterBound("confidence_threshold", 30.0, 90.0, False),
        ParameterBound("min_signal_score", 2, 6, True),
        ParameterBound("atr_sl_multiplier", 0.5, 1.5, False),
        ParameterBound("atr_tp_multiplier", 1.6, 2.6, False),
    ]
    space = StrategyParameterSpace(bounds=bounds)
    candidate = StrategyParameters(
        ema_fast=50,
        ema_slow=20,
        atr_period=3,
        confidence_threshold=110.0,
        min_signal_score=1,
        atr_sl_multiplier=3.0,
        atr_tp_multiplier=1.0,
    )
    result = space.ensure_valid(candidate)
    assert result.ema_fast < result.ema_slow
    assert 0.0 <= result.confidence_threshold <= 100.0
    assert result.atr_tp_multiplier > result.atr_sl_multiplier


def test_backtest_tuner_finds_high_score() -> None:
    bounds = [
        ParameterBound("ema_fast", 4, 8, True),
        ParameterBound("ema_slow", 10, 20, True),
        ParameterBound("atr_period", 10, 14, True),
        ParameterBound("confidence_threshold", 40.0, 60.0, False),
        ParameterBound("min_signal_score", 3, 5, True),
        ParameterBound("atr_sl_multiplier", 0.8, 1.2, False),
        ParameterBound("atr_tp_multiplier", 1.8, 2.4, False),
    ]
    space = StrategyParameterSpace(bounds=bounds)

    def objective(params: StrategyParameters) -> tuple[float, dict[str, float]]:
        score = 3.0 - abs(params.ema_fast - 6) * 0.5
        return score, {"PF": score, "trades": 50, "Win%": 60, "AvgR": 0.5, "MaxDD_R": 3.0}

    tuner = BacktestTuner(space, objective, random_state=1, initial_samples=3)
    result = tuner.tune(8)
    assert len(result.trials) == 8
    assert result.best.params.ema_fast in {5, 6, 7}


def test_walkforward_validator_runs() -> None:
    bounds = [
        ParameterBound("ema_fast", 4, 6, True),
        ParameterBound("ema_slow", 8, 12, True),
        ParameterBound("atr_period", 10, 12, True),
        ParameterBound("confidence_threshold", 40.0, 60.0, False),
        ParameterBound("min_signal_score", 3, 4, True),
        ParameterBound("atr_sl_multiplier", 0.9, 1.1, False),
        ParameterBound("atr_tp_multiplier", 1.8, 2.2, False),
    ]
    space = StrategyParameterSpace(bounds=bounds)

    idx = pd.date_range("2024-01-01", periods=5, freq="T")
    df = pd.DataFrame(
        {
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 0,
        },
        index=idx,
    )
    feed = SpotFeed(df=df, tz=ZoneInfo("Asia/Kolkata"))

    def engine_factory(segment: SpotFeed, outdir: str, write_results: bool):
        del segment, outdir, write_results

        class StubEngine:
            def run(
                self,
                *,
                start: str | None = None,
                end: str | None = None,
                params: StrategyParameters | None = None,
            ) -> dict[str, float]:
                del start, end
                score = 2.5
                if params is not None:
                    score -= abs(params.ema_fast - 5) * 0.2
                return {
                    "PF": score,
                    "trades": 40,
                    "Win%": 55,
                    "AvgR": 0.4,
                    "MaxDD_R": 2.0,
                }

        return StubEngine()

    validator = WalkForwardValidator(
        feed,
        space,
        engine_factory,
    )

    splits = [
        WalkForwardSplit(
            train_start="2024-01-01T00:00:00",
            train_end="2024-01-01T00:02:00",
            test_start="2024-01-01T00:02:00",
            test_end="2024-01-01T00:04:00",
        )
    ]

    results = validator.run(splits, trials=4, outdir="/tmp")
    assert len(results) == 1
    assert results[0].tuning.best.params.ema_fast in {4, 5, 6}
