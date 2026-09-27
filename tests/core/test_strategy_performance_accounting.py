from __future__ import annotations

import pytest

from nifty_scalper_bot.core.strategy_manager import StrategyManager


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


def test_hydrate_trade_results_rebuilds_performance_without_current_regime_leak(
) -> None:
    manager = StrategyManager([], None, None)
    manager._regime_state.regime = "volatile"

    hydrated = manager.hydrate_trade_results(
        [
            {
                "strategy_name": "VWAPPro",
                "net_pnl": 100.0,
                "regime": "TREND",
            },
            {
                "strategy_name": "VWAPPro",
                "net_pnl": -25.0,
                "regime": None,
            },
        ]
    )

    performance = manager._performance["VWAPPro"]
    assert hydrated == 2
    assert performance.trades == 2
    assert performance.total_pnl == 75.0
    assert performance.regime_buckets["trend"].trades == 1
    assert performance.regime_buckets["unknown"].trades == 1
    assert "volatile" not in performance.regime_buckets
    assert manager._adaptive_store.get_stats("VWAPPro").win_rate == 0.5
