from __future__ import annotations

from types import SimpleNamespace

import pytest

from nifty_scalper_bot.core.strategy_manager import (
    StrategyManager,
    StrategyScoreWeights,
)


def test_breakeven_trade_is_neither_win_nor_loss_but_stays_in_denominator() -> None:
    manager = StrategyManager([], None, None)

    for pnl in (100.0, 0.0, -50.0):
        manager.record_trade_result("VWAPPro", pnl, metadata={"regime": "trend"})

    performance = manager._performance["VWAPPro"]
    adaptive = manager._adaptive_store.get_stats("VWAPPro")
    regime = performance.regime_buckets["trend"].snapshot()

    assert performance.trades == 3
    assert performance.wins == 1
    assert performance.losses == 1
    assert performance.win_rate() == pytest.approx(1 / 3)
    assert performance.snapshot()["trades"] == 3.0

    assert regime["trades"] == 3.0
    assert regime["win_rate"] == pytest.approx(1 / 3)
    assert adaptive.win_rate == pytest.approx(1 / 3)


@pytest.mark.parametrize("bad_pnl", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_trade_feedback_is_rejected_before_any_state_mutates(
    bad_pnl: float,
) -> None:
    manager = StrategyManager([], None, None)

    with pytest.raises(ValueError, match="pnl must be finite"):
        manager.record_trade_result(
            "VWAPPro",
            bad_pnl,
            metadata={"regime": "trend"},
        )

    assert "VWAPPro" not in manager._performance
    assert manager._adaptive_store.get_stats("VWAPPro").win_rate == 0.0


def _score_test_manager() -> StrategyManager:
    manager = StrategyManager(
        [],
        None,
        None,
        score_weights=StrategyScoreWeights(
            pnl=1.0,
            sharpe=0.0,
            win_rate=0.0,
            drawdown=0.0,
        ),
    )
    manager._strategies = [
        SimpleNamespace(name="A"),
        SimpleNamespace(name="B"),
    ]
    manager.refresh_regime_state = lambda: None
    manager._regime_state.regime = "trend"
    manager._regime_state.confidence = 1.0
    manager.configure_score_thresholds(min_trades=2)
    return manager


def test_known_regime_never_borrows_unknown_performance_bucket() -> None:
    manager = _score_test_manager()
    manager._record_performance_observation("A", 100.0, regime_label=None)
    manager._record_performance_observation("A", 50.0, regime_label=None)

    score = manager._recompute_scores()["A"]

    assert manager._performance["A"].regime_buckets["unknown"].trades == 2
    assert score.active_regime_stats["trades"] == 0.0


def test_regime_performance_stays_neutral_until_same_regime_floor() -> None:
    manager = _score_test_manager()

    manager._record_performance_observation("A", 210.0, regime_label="range")
    manager._record_performance_observation("A", -10.0, regime_label="trend")
    manager._record_performance_observation("B", -210.0, regime_label="range")
    manager._record_performance_observation("B", 10.0, regime_label="trend")

    sparse_scores = manager._recompute_scores()
    assert sparse_scores["A"].score > sparse_scores["B"].score
    assert sparse_scores["A"].active_regime_stats["trades"] == 1.0
    assert sparse_scores["B"].active_regime_stats["trades"] == 1.0

    manager._record_performance_observation("A", -10.0, regime_label="trend")
    manager._record_performance_observation("B", 10.0, regime_label="trend")

    mature_scores = manager._recompute_scores()
    assert mature_scores["B"].score > mature_scores["A"].score
