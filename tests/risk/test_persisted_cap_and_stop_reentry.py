"""Regression tests for restart-safe daily limits and stop-loss re-entry."""

from __future__ import annotations

from types import SimpleNamespace

from nifty_scalper_bot.execution.position_manager import PositionManager
from nifty_scalper_bot.execution.position_risk_state_patch import _option_thesis
from nifty_scalper_bot.risk.entry_guard_patch import (
    _daily_limit_block_reason,
    _stop_reentry_block_reason,
)


def _manager(pm: PositionManager, limit: int = 6) -> SimpleNamespace:
    return SimpleNamespace(
        settings=SimpleNamespace(max_trades_per_day=limit, max_open_positions=0),
        position_manager=pm,
    )


def test_daily_entry_count_survives_intraday_restart(tmp_path) -> None:
    state_file = tmp_path / "positions.json"
    first = PositionManager(state_file=str(state_file))
    for _ in range(6):
        with first._lock:
            first._increment_trades_today_locked()
    first.save_state()

    restarted = PositionManager(state_file=str(state_file))

    assert restarted.trades_today() == 6
    reason = _daily_limit_block_reason(_manager(restarted))
    assert reason is not None
    assert reason[1] == "MAX_TRADES:6/6"


def test_previous_trading_date_count_is_not_restored(tmp_path) -> None:
    state_file = tmp_path / "positions.json"
    first = PositionManager(state_file=str(state_file))
    first._trades_today_date = "2000-01-01"
    first._trades_today_count = 99
    first.save_state()

    restarted = PositionManager(state_file=str(state_file))

    assert restarted.trades_today() == 0


def test_option_thesis_is_strike_independent() -> None:
    assert _option_thesis("NFO:NIFTY2680424400PE") == ("NIFTY", "PE")
    assert _option_thesis("NFO:NIFTY2680424350PE") == ("NIFTY", "PE")
    assert _option_thesis("NFO:NIFTY2680424400CE") == ("NIFTY", "CE")


def test_stop_loss_blocks_same_side_across_strikes(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("STOP_LOSS_REENTRY_COOLDOWN_SECONDS", "300")
    pm = PositionManager(state_file=str(tmp_path / "positions.json"))
    pm.open_position(
        symbol="NFO:NIFTY2680424400PE",
        side="LONG",
        quantity=65,
        entry_price=93.35,
    )
    pm.close_position(
        "NFO:NIFTY2680424400PE",
        exit_price=91.40,
        reason="STOP_LOSS",
    )

    same_thesis = SimpleNamespace(symbol="NFO:NIFTY2680424350PE", side="BUY")
    opposite_side = SimpleNamespace(symbol="NFO:NIFTY2680424400CE", side="BUY")

    assert "stop-loss thesis cooldown" in pm.stop_reentry_block_reason(same_thesis)
    assert pm.stop_reentry_block_reason(opposite_side) is None


def test_stop_reentry_lock_survives_restart(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("STOP_LOSS_REENTRY_COOLDOWN_SECONDS", "300")
    state_file = tmp_path / "positions.json"
    first = PositionManager(state_file=str(state_file))
    first.open_position(
        symbol="NFO:NIFTY2680424400PE",
        side="LONG",
        quantity=65,
        entry_price=93.35,
    )
    first.close_position(
        "NFO:NIFTY2680424400PE",
        exit_price=91.40,
        reason="SL",
    )

    restarted = PositionManager(state_file=str(state_file))
    signal = SimpleNamespace(symbol="NFO:NIFTY2680424350PE", side="BUY")

    assert _stop_reentry_block_reason(restarted, signal) is not None


def test_non_stop_exit_does_not_create_reentry_lock(tmp_path) -> None:
    pm = PositionManager(state_file=str(tmp_path / "positions.json"))
    pm.open_position(
        symbol="NFO:NIFTY2680424400PE",
        side="LONG",
        quantity=65,
        entry_price=93.35,
    )
    pm.close_position(
        "NFO:NIFTY2680424400PE",
        exit_price=96.00,
        reason="TAKE_PROFIT",
    )

    signal = SimpleNamespace(symbol="NFO:NIFTY2680424350PE", side="BUY")
    assert pm.stop_reentry_block_reason(signal) is None
