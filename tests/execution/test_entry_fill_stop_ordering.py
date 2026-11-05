"""Regression: a broker-confirmed fill must exist locally before SL admission.

The 2026-11-05 09:39:22 IST live incident sent the exchange SL to the
reduce-only gate before PositionManager applied the filled 65-unit entry.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nifty_scalper_bot.execution.order_manager_core import (
    OrderDetails,
    OrderManager,
    OrderStatus,
    OrderType,
)
from nifty_scalper_bot.execution.position_manager import PositionManager

SYMBOL = "NFO:NIFTY26N0924350CE"
ENTRY_ID = "entry-fill-stop-sync"


def _runner(tmp_path):
    positions = PositionManager(state_file=str(tmp_path / "positions.json"))
    positions.add_pending_order(
        ENTRY_ID, SYMBOL, "BUY", 65, 154.45, "LIMIT", intent="ENTRY"
    )
    manager = OrderManager.__new__(OrderManager)
    manager._orders = {
        ENTRY_ID: OrderDetails(
            order_id=ENTRY_ID,
            symbol=SYMBOL,
            side="BUY",
            quantity=65,
            order_type=OrderType.LIMIT,
            status=OrderStatus.PENDING,
            price=154.45,
            stop_loss=145.90,
            take_profit=171.55,
            intent="ENTRY",
            timestamp=datetime.now(timezone.utc),
        )
    }
    manager._lock = threading.RLock()
    manager._positions = positions
    manager._bracket_manager = SimpleNamespace()
    manager._broker = SimpleNamespace(query_positions=lambda: {})
    manager._logger = logging.getLogger(__name__)
    manager._lot_size_for_symbol = lambda _symbol: 65
    manager.save_orders = Mock()
    manager._notify_failed_entry_terminal = Mock()
    manager._confirm_position_protection_for_fill = Mock()
    return manager, positions


@pytest.mark.parametrize(
    ("status", "quantity"),
    [("COMPLETE", 65), ("PARTIALLY FILLED", 30)],
)
def test_confirmed_entry_position_precedes_bracket_and_exchange_sl(
    tmp_path, status, quantity
):
    manager, positions = _runner(tmp_path)
    observed = []

    def on_bracket(order, *, source):
        position = positions.get_position(SYMBOL)
        assert position is not None
        assert position.quantity == quantity
        observed.append((source, position.quantity))
        if quantity == 65:
            # The exact pre-fix failure was exit_without_open_position.
            assert (
                manager._validate_option_exit_quantity(
                    SYMBOL, "SELL", 65, "EXIT"
                )
                is None
            )

    manager._register_virtual_bracket_for_fill = on_bracket

    update = {
        "order_id": ENTRY_ID,
        "status": status,
        "filled_quantity": quantity,
        "average_price": 154.45,
    }
    manager._apply_broker_order_update(update)

    assert observed == [("order_update", quantity)]
    assert positions.get_position(SYMBOL).quantity == quantity

    if status == "COMPLETE":
        manager._apply_broker_order_update(update)
        assert observed == [("order_update", quantity)]
        assert positions.get_position(SYMBOL).quantity == quantity


def test_position_sync_failure_still_activates_software_bracket_and_fails_closed(
    tmp_path, monkeypatch
):
    manager, positions = _runner(tmp_path)
    bracket_attempts = []

    def failed_sync(*_args, **_kwargs):
        raise RuntimeError("position ledger unavailable")

    monkeypatch.setattr(positions, "apply_broker_order_update", failed_sync)

    def on_bracket(order, *, source):
        bracket_attempts.append(order.order_id)
        # No position evidence: exchange stop must NOT bypass reduce-only.
        decision = manager._validate_option_exit_quantity(
            SYMBOL, "SELL", 65, "EXIT"
        )
        assert decision is not None
        assert decision["reason"] == "exit_without_open_position"

    manager._register_virtual_bracket_for_fill = on_bracket

    with pytest.raises(RuntimeError, match="position ledger unavailable"):
        manager._apply_broker_order_update(
            {
                "order_id": ENTRY_ID,
                "status": "COMPLETE",
                "filled_quantity": 65,
                "average_price": 154.45,
            }
        )

    assert bracket_attempts == [ENTRY_ID]
    assert positions.get_position(SYMBOL) is None
