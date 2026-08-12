from __future__ import annotations

import pytest

from nifty_scalper_bot.execution.position_manager import PositionManager

SYMBOL = "NFO:NIFTY2681824400PE"
ENTRY_ID = "2087403339074953216"


def _broker_day_position(*, last_price: float = 124.30) -> dict[str, object]:
    return {
        "symbol": SYMBOL,
        "product": "MIS",
        "quantity": 65,
        # Zerodha day-position average after two same-contract buys:
        # (103.35 + 124.45) / 2 = 113.90. This is not the new trade's basis.
        "average_price": 113.90,
        "last_price": last_price,
    }


def _reentry_manager(tmp_path) -> PositionManager:
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    manager.add_pending_order(
        ENTRY_ID,
        SYMBOL,
        "BUY",
        65,
        124.45,
        "MARKET",
        intent="ENTRY",
        bracket_id=ENTRY_ID,
    )
    return manager


def test_broker_sync_before_reentry_fill_yields_to_confirmed_order_basis(tmp_path) -> None:
    """The local broker-confirmed fill owns lifecycle basis, not day-position average."""
    manager = _reentry_manager(tmp_path)

    # Reproduce the live race: periodic broker position sync lands before the
    # local COMPLETE callback and creates the quantity with Zerodha's day average.
    manager.synchronize_with_broker([_broker_day_position()])
    before_fill = manager.get_position(SYMBOL)
    assert before_fill is not None
    assert before_fill.quantity == 65
    assert before_fill.entry_price == pytest.approx(113.90)
    assert before_fill.order_id is None

    manager.apply_broker_order_update(
        ENTRY_ID,
        {
            "status": "COMPLETE",
            "filled_quantity": 65,
            "average_price": 124.45,
        },
    )

    position = manager.get_position(SYMBOL)
    assert position is not None
    assert position.quantity == 65  # broker quantity must not be double-applied
    assert position.order_id == ENTRY_ID
    assert position.entry_price == pytest.approx(124.45)
    assert position.current_price == pytest.approx(124.30)


def test_owned_reentry_basis_survives_day_sync_and_drives_exit_pnl(tmp_path) -> None:
    """Later broker day snapshots cannot overwrite an owned trade's fill basis."""
    manager = _reentry_manager(tmp_path)
    manager.synchronize_with_broker([_broker_day_position()])
    manager.apply_broker_order_update(
        ENTRY_ID,
        {
            "status": "COMPLETE",
            "filled_quantity": 65,
            "average_price": 124.45,
        },
    )
    manager.confirm_entry_protection(ENTRY_ID, ENTRY_ID, 65)

    # The next periodic reconciliation still reports 113.90 for the day-level
    # position. Quantity/side/mark remain broker truth, lifecycle basis must not.
    manager.synchronize_with_broker([_broker_day_position(last_price=122.05)])
    position = manager.get_position(SYMBOL)
    assert position is not None
    assert position.entry_price == pytest.approx(124.45)
    assert position.current_price == pytest.approx(122.05)

    manager.add_pending_order(
        "2087404505003384833",
        SYMBOL,
        "SELL",
        65,
        114.00,
        "MARKET",
        intent="EXIT",
        bracket_id=ENTRY_ID,
    )
    manager.apply_broker_order_update(
        "2087404505003384833",
        {
            "status": "COMPLETE",
            "filled_quantity": 65,
            "average_price": 114.00,
        },
    )

    assert manager.get_position(SYMBOL) is None
    assert manager.get_realized_pnl() == pytest.approx((114.00 - 124.45) * 65)
