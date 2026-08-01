"""A crashed strategy must not look like a strategy that declined (P2)."""

from __future__ import annotations

import logging

import pytest

from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    EliteStrategyConfig,
)

SYMBOL = "NFO:NIFTY2680424400CE"


class _Boom(EliteStrategy):
    def _evaluate_signal(self, symbol="", indicators=None, current_price=0.0, position=None):
        raise TypeError("must be real number, not NoneType")


class _Quiet(EliteStrategy):
    def _evaluate_signal(self, symbol="", indicators=None, current_price=0.0, position=None):
        return None


class _Good(EliteStrategy):
    def _evaluate_signal(self, symbol="", indicators=None, current_price=0.0, position=None):
        return EliteSignal(
            symbol=symbol,
            signal="BUY",
            confidence=0.9,
            entry_price=100.0,
            stop_loss=90.0,
            target=120.0,
            metadata={"latest_bar_ts": 1_785_000_000.0},
        )


def _build(cls) -> EliteStrategy:
    return cls(config=EliteStrategyConfig(), indicator_engine=None)


def _run(strategy: EliteStrategy):
    return strategy.generate_signal(SYMBOL, {"latest_bar_ts": 1_785_000_000.0}, 100.0)


def test_crash_is_logged_as_a_distinct_event(caplog) -> None:
    strategy = _build(_Boom)
    with caplog.at_level(logging.ERROR):
        assert _run(strategy) is None

    assert any(
        getattr(record, "event", "") == "STRATEGY_EVALUATION_FAILED"
        for record in caplog.records
    )


def test_consecutive_failures_are_counted() -> None:
    strategy = _build(_Boom)
    for _ in range(3):
        _run(strategy)

    health = strategy.evaluation_health
    assert health["consecutive_evaluation_failures"] == 3
    assert health["healthy"] is False
    assert "TypeError" in health["last_evaluation_error"]
    assert strategy.last_no_vote_reason == "evaluation_failed"


def test_a_declining_strategy_stays_healthy() -> None:
    strategy = _build(_Quiet)
    _run(strategy)

    assert strategy.evaluation_health["healthy"] is True
    assert strategy.evaluation_health["consecutive_evaluation_failures"] == 0


def test_a_successful_evaluation_clears_the_counter() -> None:
    strategy = _build(_Good)
    strategy._consecutive_evaluation_failures = 4

    assert _run(strategy) is not None
    assert strategy.evaluation_health["consecutive_evaluation_failures"] == 0


def test_health_is_exposed_in_stats() -> None:
    strategy = _build(_Boom)
    _run(strategy)

    assert strategy.get_stats()["healthy"] is False


def test_percentage_confidence_threshold_is_normalised() -> None:
    from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
        _as_confidence_fraction,
    )

    assert _as_confidence_fraction(70.0) == pytest.approx(0.70)
    assert _as_confidence_fraction(100.0) == pytest.approx(1.0)


def test_fractional_confidence_threshold_is_not_divided_again() -> None:
    from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
        _as_confidence_fraction,
    )

    # 0.6 previously became 0.006, silently disabling the gate.
    assert _as_confidence_fraction(0.6) == pytest.approx(0.6)
    assert _as_confidence_fraction(1.0) == pytest.approx(1.0)


def test_invalid_confidence_threshold_does_not_block() -> None:
    from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
        _as_confidence_fraction,
    )

    assert _as_confidence_fraction(None) == 0.0
    assert _as_confidence_fraction(-5.0) == 0.0


def test_signal_below_configured_confidence_is_rejected() -> None:
    strategy = _build(_Good)
    strategy._config.min_confidence = 95.0

    assert _run(strategy) is None
    assert strategy.last_no_vote_reason == "below_strategy_min_confidence"
