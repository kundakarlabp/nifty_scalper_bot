from __future__ import annotations

from types import SimpleNamespace

import pytest

from nifty_scalper_bot.execution.position_manager import PositionManager


def _broker_snapshot(
    *,
    realized: float = 0.0,
    unrealized: float = 0.0,
    day_rows: int = 0,
) -> dict[str, object]:
    return {
        "account_realized": realized,
        "account_unrealized": unrealized,
        "account_total": realized + unrealized,
        "strategy_day_marked_gross": 0.0 if day_rows else None,
        "strategy_day_closed_gross": 0.0 if day_rows else None,
        "strategy_day_rows": day_rows,
        "source": "zerodha_margins_m2m",
        "positions_source": "zerodha_positions_day",
        "positions_error": None,
        "observed_at": "2026-09-11T12:30:00+00:00",
    }


def test_broker_zero_day_repairs_poisoned_today_local_carryover(tmp_path) -> None:
    """The exact Sep-11 incident: stale ₹858 must become today's ₹0."""

    state_path = tmp_path / "positions.json"
    manager = PositionManager(state_file=str(state_path))
    today = manager._trading_date_ist()
    with manager._lock:
        manager._local_realized_pnl = 858.0
        manager._local_provisional_realized_pnl = 0.0
        # Previous buggy broker patch had already poisoned this stale value by
        # stamping it with today's trading date.
        manager._pnl_trading_date = today
        manager._session_opening_realized_baseline = 0.0
        manager._baseline_source = "zerodha_margins_m2m"
        manager._refresh_realized_pnl_locked()

    manager.set_broker_client(
        SimpleNamespace(get_pnl_snapshot=lambda: _broker_snapshot())
    )

    snapshot = manager.refresh_broker_pnl_diagnostic(force=True)

    assert manager.get_realized_pnl() == pytest.approx(0.0)
    assert manager.get_strategy_realized_pnl() == pytest.approx(0.0)
    assert manager.get_broker_account_realized_pnl() == pytest.approx(0.0)
    assert snapshot["strategy_realized"] == pytest.approx(0.0)
    assert snapshot["account_realized"] == pytest.approx(0.0)
    assert snapshot["difference"] == pytest.approx(0.0)
    assert snapshot["status"] == "matched"
    assert manager._pnl_trading_date == today

    restarted = PositionManager(state_file=str(state_path))
    assert restarted.get_realized_pnl() == pytest.approx(0.0)
    assert restarted._pnl_trading_date == today


def test_broker_day_activity_prevents_zero_day_cleanup(tmp_path) -> None:
    """Do not erase a same-day ledger merely because account net P&L is zero."""

    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    today = manager._trading_date_ist()
    with manager._lock:
        manager._local_realized_pnl = 125.0
        manager._pnl_trading_date = today
        manager._session_opening_realized_baseline = 0.0
        manager._baseline_source = "zerodha_margins_m2m"
        manager._refresh_realized_pnl_locked()

    manager.set_broker_client(
        SimpleNamespace(
            get_pnl_snapshot=lambda: _broker_snapshot(
                realized=0.0,
                unrealized=0.0,
                day_rows=1,
            )
        )
    )

    snapshot = manager.refresh_broker_pnl_diagnostic(force=True)

    assert manager.get_realized_pnl() == pytest.approx(125.0)
    assert snapshot["strategy_realized"] == pytest.approx(125.0)
    assert snapshot["difference"] == pytest.approx(-125.0)
    assert snapshot["status"] == "mismatch"


def test_explicit_old_trading_date_resets_local_session_on_broker_refresh(tmp_path) -> None:
    """A dated prior-session local P&L never carries into a new broker day."""

    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    with manager._lock:
        manager._local_realized_pnl = -450.0
        manager._local_provisional_realized_pnl = -25.0
        manager._pnl_trading_date = "2026-09-10"
        manager._session_opening_realized_baseline = 0.0
        manager._refresh_realized_pnl_locked()

    manager.set_broker_client(
        SimpleNamespace(
            get_pnl_snapshot=lambda: _broker_snapshot(
                realized=50.0,
                unrealized=0.0,
                day_rows=1,
            )
        )
    )

    snapshot = manager.refresh_broker_pnl_diagnostic(force=True)

    assert manager.get_realized_pnl() == pytest.approx(0.0)
    assert snapshot["strategy_realized"] == pytest.approx(0.0)
    assert snapshot["account_realized"] == pytest.approx(50.0)
    assert snapshot["difference"] == pytest.approx(50.0)
    assert manager._pnl_trading_date == manager._trading_date_ist()
