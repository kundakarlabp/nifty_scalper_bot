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


def test_restore_trade_results_is_idempotent_and_rebuilds_regime_stats() -> None:
    manager = StrategyManager([], None, None)
    records = [
        {
            "trade_id": "trade-1",
            "strategy": "VWAPPro",
            "net_pnl": 100.0,
            "regime": "TREND",
        },
        {
            "trade_id": "trade-2",
            "strategy": "VWAPPro",
            "net_pnl": -40.0,
            "regime": "RANGE",
        },
        {
            "trade_id": "trade-3",
            "strategy": "SMC",
            "net_pnl": 25.0,
            "regime": None,
        },
    ]

    assert manager.restore_trade_results(records) == 3
    assert manager.restore_trade_results(records) == 3

    vwap = manager._performance["VWAPPro"]
    assert vwap.trades == 2
    assert vwap.total_pnl == 60.0
    assert vwap.regime_buckets["trend"].trades == 1
    assert vwap.regime_buckets["range"].trades == 1
    assert manager._adaptive_store.get_stats("VWAPPro").win_rate == 0.5

    smc = manager._performance["SMC"]
    assert smc.trades == 1
    assert smc.regime_buckets["unknown"].trades == 1


def test_restore_trade_results_fails_atomically_on_invalid_record() -> None:
    manager = StrategyManager([], None, None)
    manager.record_trade_result("VWAPPro", 10.0, metadata={"regime": "trend"})

    with pytest.raises(ValueError, match="strategy and finite net_pnl are required"):
        manager.restore_trade_results(
            [
                {"strategy": "SMC", "net_pnl": 20.0, "regime": "range"},
                {"strategy": "", "net_pnl": 5.0, "regime": "trend"},
            ]
        )

    assert set(manager._performance) == {"VWAPPro"}
    assert manager._performance["VWAPPro"].trades == 1
