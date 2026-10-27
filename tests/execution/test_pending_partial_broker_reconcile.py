"""Broker pending-entry reconciliation must protect genuine partial fills."""

from __future__ import annotations

import time
from types import SimpleNamespace

from nifty_scalper_bot.execution.bracket_manager import BracketManager

SYMBOL = "NFO:NIFTY26OCT22700CE"


class _Broker:
    def __init__(self):
        self.order = {
            "order_id": "entry-pending",
            "status": "OPEN",
            "filled_quantity": 0,
            "average_price": 0,
        }
        self.positions = []

    def get_order_status(self, order_id):
        assert order_id == "entry-pending"
        return dict(self.order)

    def get_positions(self):
        return list(self.positions)


def _fixture(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRACKET_FILL_LEDGER_PATH", str(tmp_path / "fills.db"))
    broker = _Broker()
    manager = BracketManager(order_manager=SimpleNamespace(_broker=broker))
    manager._running = False
    manager._watchdog_thread.join(timeout=1.0)
    manager.register_virtual_bracket(
        order_id="entry-pending",
        symbol=SYMBOL,
        side="BUY",
        qty=65,
        price=100.0,
        sl=95.0,
        tp=120.0,
        activate_immediately=False,
    )
    bracket = manager.get_bracket("entry-pending")
    assert bracket is not None
    bracket.created_at = time.time() - 10.0
    manager._pending_entry_reconcile_after_sec = 0.0
    return manager, broker, bracket


def test_cancelled_entry_with_executed_quantity_activates_partial_protection(
    monkeypatch, tmp_path
):
    manager, broker, bracket = _fixture(monkeypatch, tmp_path)
    broker.order.update(
        status="CANCELLED", filled_quantity=30, average_price=101.25
    )
    broker.positions = [{"symbol": SYMBOL, "quantity": 30}]
    manager._reconcile_pending_entry(bracket)

    assert manager.get_bracket("entry-pending") is bracket
    assert bracket.entry_confirmed is True
    assert bracket.active is True
    assert bracket.quantity == 30
    assert bracket.requested_entry_quantity == 65
    assert bracket.remaining_quantity == 30
    assert bracket.entry_price == 101.25
    assert "entry-pending" in manager._partial_entry_terminal_orders

    manager._reconcile_pending_entry(bracket)
    assert bracket.remaining_quantity == 30


def test_open_partial_then_later_cumulative_fill_reconciles_without_reset(
    monkeypatch, tmp_path
):
    manager, broker, bracket = _fixture(monkeypatch, tmp_path)
    broker.order.update(
        status="PARTIALLY FILLED", filled_quantity=30, average_price=101.0
    )
    manager._reconcile_pending_entry(bracket)
    assert bracket.entry_confirmed is True
    assert bracket.quantity == 30
    first_fill_ts = bracket.entry_fill_ts
    bracket.highest_ltp = 110.0
    bracket.sl_trigger_price = 103.0
    bracket.trail_revision = 2

    broker.order.update(status="COMPLETE", filled_quantity=65, average_price=102.0)
    manager._partial_entry_probe_next_at.clear()
    manager._reconcile_pending_entry(bracket)

    assert bracket.quantity == 65
    assert bracket.remaining_quantity == 65
    assert bracket.entry_price == 102.0
    assert bracket.entry_fill_ts == first_fill_ts
    assert bracket.highest_ltp == 110.0
    assert bracket.sl_trigger_price >= 103.0
    assert bracket.trail_revision == 2
    assert manager._fill_ledger is not None
    fills = manager._fill_ledger.load_fills(bracket.bracket_id)
    assert len(fills) == 1
    assert fills[0].quantity == 65
    assert fills[0].price == 102.0


def test_broker_open_unfilled_does_not_optimistically_activate(
    monkeypatch, tmp_path
):
    manager, broker, bracket = _fixture(monkeypatch, tmp_path)
    broker.order.update(status="OPEN", filled_quantity=0, average_price=0)
    manager._reconcile_pending_entry(bracket)
    assert bracket.entry_confirmed is False
    assert bracket.active is False


def test_terminal_partial_with_invalid_price_remains_pending_and_owned(
    monkeypatch, tmp_path
):
    manager, broker, bracket = _fixture(monkeypatch, tmp_path)
    broker.order.update(
        status="CANCELLED", filled_quantity=30, average_price=float("inf")
    )
    broker.positions = [{"symbol": SYMBOL, "quantity": 30}]
    manager._reconcile_pending_entry(bracket)
    assert bracket.entry_confirmed is False
    assert manager.get_bracket("entry-pending") is bracket


def test_unfilled_rejected_entry_and_flat_position_can_be_removed(
    monkeypatch, tmp_path
):
    manager, broker, bracket = _fixture(monkeypatch, tmp_path)
    broker.order.update(status="REJECTED", filled_quantity=0, average_price=0)
    manager._reconcile_pending_entry(bracket)
    assert manager.get_bracket("entry-pending") is None
