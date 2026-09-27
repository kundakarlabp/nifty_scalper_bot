from __future__ import annotations

import json
import sqlite3

import pytest

from nifty_scalper_bot.core.strategy_manager import StrategyManager
from nifty_scalper_bot.journal.trade_ledger import (
    ensure_trade_ledger_schema,
    load_completed_strategy_history,
)


def _insert_trade(
    connection: sqlite3.Connection,
    *,
    trade_id: str,
    strategy: str | None,
    net_pnl: float | None,
    closed_at: float,
    ledger_complete: int,
    regime: str | None,
) -> None:
    outcome = {"regime": regime} if regime is not None else {}
    connection.execute(
        """
        INSERT INTO trade_ledger (
            trade_id, strategy, state, state_rank, net_pnl, ledger_complete,
            closed_at, created_at, updated_at, last_event_name, outcome_json
        ) VALUES (?, ?, 'CLOSED', 100, ?, ?, ?, ?, ?, 'trade.closed', ?)
        """,
        (
            trade_id,
            strategy,
            net_pnl,
            ledger_complete,
            closed_at,
            closed_at,
            closed_at,
            json.dumps(outcome),
        ),
    )


def test_completed_strategy_history_is_bounded_and_chronological(tmp_path) -> None:
    db_path = tmp_path / "trades.db"
    with sqlite3.connect(db_path) as connection:
        ensure_trade_ledger_schema(connection)
        _insert_trade(
            connection,
            trade_id="v1",
            strategy="VWAPPro",
            net_pnl=10.0,
            closed_at=1.0,
            ledger_complete=1,
            regime="TREND",
        )
        _insert_trade(
            connection,
            trade_id="v2",
            strategy="VWAPPro",
            net_pnl=-5.0,
            closed_at=2.0,
            ledger_complete=1,
            regime="RANGE",
        )
        _insert_trade(
            connection,
            trade_id="v3",
            strategy="VWAPPro",
            net_pnl=20.0,
            closed_at=3.0,
            ledger_complete=1,
            regime="TREND",
        )
        _insert_trade(
            connection,
            trade_id="s1",
            strategy="SMC",
            net_pnl=7.0,
            closed_at=1.5,
            ledger_complete=1,
            regime="TREND",
        )
        _insert_trade(
            connection,
            trade_id="incomplete",
            strategy="SMC",
            net_pnl=999.0,
            closed_at=4.0,
            ledger_complete=0,
            regime="TREND",
        )

    history = load_completed_strategy_history(
        db_path,
        strategy_names=["VWAPPro", "SMC"],
        per_strategy_limit=2,
    )

    assert [(row["strategy"], row["net_pnl"]) for row in history] == [
        ("SMC", 7.0),
        ("VWAPPro", -5.0),
        ("VWAPPro", 20.0),
    ]
    assert [row["regime"] for row in history] == ["TREND", "RANGE", "TREND"]


def test_strategy_manager_restores_history_once_without_live_side_effects() -> None:
    manager = StrategyManager([], None, None)
    history = [
        {"strategy": "VWAPPro", "net_pnl": 100.0, "regime": "TREND"},
        {"strategy": "VWAPPro", "net_pnl": 0.0, "regime": "TREND"},
        {"strategy": "VWAPPro", "net_pnl": -50.0, "regime": "RANGE"},
    ]

    restored = manager.restore_performance_history(history)

    performance = manager._performance["VWAPPro"]
    assert restored == 3
    assert performance.trades == 3
    assert performance.wins == 1
    assert performance.losses == 1
    assert performance.win_rate() == pytest.approx(1 / 3)
    assert manager._adaptive_store.get_stats("VWAPPro").win_rate == pytest.approx(1 / 3)

    with pytest.raises(RuntimeError, match="performance history already initialised"):
        manager.restore_performance_history(history)
