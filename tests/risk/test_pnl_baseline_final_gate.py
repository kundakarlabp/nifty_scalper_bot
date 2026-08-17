from __future__ import annotations

import pytest

from nifty_scalper_bot.config.settings import RiskSettings
from nifty_scalper_bot.execution.position_manager import PositionManager
from nifty_scalper_bot.risk.risk_manager import OrderSignal, RiskManager


SYMBOL = "NFO:NIFTY2681824250CE"


def _risk(manager: PositionManager) -> RiskManager:
    return RiskManager(
        settings=RiskSettings(
            per_trade_risk_pct=1.0,
            daily_loss_pct=2.0,
            cooldown_on_reject_seconds=0.0,
        ),
        position_manager=manager,
        account_balance=50_000.0,
    )


def _flat_row(*, pnl: float, realised: float = 0.0) -> dict[str, object]:
    return {
        "symbol": SYMBOL,
        "product": "MIS",
        "quantity": 0,
        "average_price": 0.0,
        "last_price": 100.0,
        "multiplier": 1,
        "pnl": pnl,
        "realised": realised,
    }


def test_closed_position_uses_broker_pnl_when_realised_field_is_zero(tmp_path) -> None:
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    manager.establish_pnl_session_baseline(0.0)

    manager.synchronize_with_broker([_flat_row(pnl=-234.0, realised=0.0)])

    snapshot = manager.pnl_reconciliation_snapshot()
    assert manager.get_realized_pnl() == pytest.approx(-234.0)
    assert snapshot["broker_realized_snapshot"] == pytest.approx(-234.0)
    assert snapshot["local_confirmed_realized"] == pytest.approx(-234.0)


def test_fresh_live_session_bootstraps_missing_baseline_from_broker_day_pnl(tmp_path) -> None:
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    manager.require_pnl_session_baseline(True)

    manager.synchronize_with_broker([_flat_row(pnl=-234.0, realised=0.0)])

    snapshot = manager.pnl_reconciliation_snapshot()
    assert snapshot["session_opening_realized_baseline"] == pytest.approx(-234.0)
    assert manager.get_realized_pnl() == pytest.approx(0.0)
    assert manager.current_pnl_reconciliation_blocker() is None


def test_ambiguous_same_day_missing_baseline_stays_fail_closed(tmp_path) -> None:
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    manager.require_pnl_session_baseline(True)
    manager._pnl_trading_date = manager._trading_date_ist()
    manager._local_realized_pnl = -120.0
    with manager._lock:
        manager._refresh_realized_pnl_locked()

    manager.synchronize_with_broker([_flat_row(pnl=-234.0, realised=0.0)])

    snapshot = manager.pnl_reconciliation_snapshot()
    assert snapshot["session_opening_realized_baseline"] is None
    assert manager.get_realized_pnl() == pytest.approx(-120.0)
    assert manager.current_pnl_reconciliation_blocker() == "pnl_baseline_uninitialized"


def test_empty_broker_day_clears_stale_unscoped_local_pnl_and_bootstraps_zero(tmp_path) -> None:
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    manager.require_pnl_session_baseline(True)
    manager._local_realized_pnl = -1200.67
    with manager._lock:
        manager._refresh_realized_pnl_locked()

    manager.synchronize_with_broker([])

    snapshot = manager.pnl_reconciliation_snapshot()
    assert snapshot["session_opening_realized_baseline"] == pytest.approx(0.0)
    assert snapshot["local_confirmed_realized"] == pytest.approx(0.0)
    assert manager.get_realized_pnl() == pytest.approx(0.0)
    assert manager.current_pnl_reconciliation_blocker() is None


def test_final_entry_gate_blocks_uninitialized_pnl_baseline(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PAPER_MODE", "true")
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    manager.require_pnl_session_baseline(True)
    risk = _risk(manager)
    signal = OrderSignal(
        symbol=SYMBOL,
        side="BUY",
        quantity=25,
        price=100.0,
        stop_loss=95.0,
        take_profit=115.0,
    )

    allowed, reason = risk.check_order(signal, live_enabled=True)

    assert allowed is False
    assert reason == "pnl_baseline_uninitialized"
    assert risk._last_rejection == "PNL_BASELINE_UNINITIALIZED"
    assert risk._breaker_tripped is False


def test_pnl_baseline_guard_does_not_block_reducing_order(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PAPER_MODE", "true")
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    manager.require_pnl_session_baseline(True)
    manager.open_position(
        symbol=SYMBOL,
        side="LONG",
        quantity=25,
        entry_price=100.0,
    )
    risk = _risk(manager)
    signal = OrderSignal(
        symbol=SYMBOL,
        side="SELL",
        quantity=25,
        price=100.0,
        stop_loss=105.0,
        take_profit=90.0,
    )

    allowed, reason = risk.check_order(signal, live_enabled=True)

    assert allowed is True
    assert reason == ""
