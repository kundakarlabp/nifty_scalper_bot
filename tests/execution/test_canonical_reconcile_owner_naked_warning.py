"""Broker/local quantity matching must use canonical bracket ownership."""

from __future__ import annotations

from unittest.mock import MagicMock

from nifty_scalper_bot.execution.order_manager_core import OrderManager

SYMBOL = "NFO:NIFTY26OCT22650CE"


def _manager(*, managed: bool, local_present: bool = True) -> MagicMock:
    manager = MagicMock()
    manager._broker.get_positions.return_value = [
        {
            "tradingsymbol": "NIFTY26OCT22650CE",
            "exchange": "NFO",
            "quantity": 65,
            "average_price": 146.65,
        }
    ]
    manager._positions.get_open_positions.return_value = (
        [MagicMock(symbol=SYMBOL, quantity=65, entry_price=146.65)]
        if local_present
        else []
    )
    manager._bracket_manager.is_symbol_managed.return_value = managed
    manager._bracket_manager.has_active_bracket.return_value = managed
    manager._brackets = {}  # Legacy map intentionally empty.
    manager._orders = {}
    manager._entries_in_flight = {}
    manager._recent_orphans = set()
    manager._missing_counts = {}
    manager.ENTRY_INFLIGHT_TTL_SEC = 30.0
    manager.guard_orphan_position.return_value = True
    return manager


def test_matching_managed_position_does_not_trigger_false_naked_guard() -> None:
    manager = _manager(managed=True)
    OrderManager._reconcile_positions(manager)
    manager._bracket_manager.is_symbol_managed.assert_called_with(SYMBOL)
    manager.guard_orphan_position.assert_not_called()
    assert manager._missing_counts[SYMBOL] == 0
    assert not any(
        "Naked Position Detected" in str(call)
        for call in manager._logger.warning.call_args_list
    )


def test_matching_unmanaged_position_still_triggers_guard() -> None:
    manager = _manager(managed=False)
    OrderManager._reconcile_positions(manager)
    manager.guard_orphan_position.assert_called_once_with(
        SYMBOL, 65, 146.65, consume_existing=True
    )


def test_broker_visible_managed_position_is_not_latched_as_orphan() -> None:
    manager = _manager(managed=True, local_present=False)
    OrderManager._reconcile_positions(manager)
    manager.guard_orphan_position.assert_not_called()
    assert SYMBOL not in manager._recent_orphans


def test_unmanaged_orphan_is_adopted_once() -> None:
    manager = _manager(managed=False, local_present=False)
    OrderManager._reconcile_positions(manager)
    OrderManager._reconcile_positions(manager)
    manager.guard_orphan_position.assert_called_once_with(SYMBOL, 65, 146.65)


def test_inflight_entry_does_not_permanently_suppress_orphan_rescue() -> None:
    manager = _manager(managed=False, local_present=False)
    manager._entries_in_flight[SYMBOL] = float("inf")
    OrderManager._reconcile_positions(manager)
    manager.guard_orphan_position.assert_not_called()
    assert SYMBOL not in manager._recent_orphans

    manager._entries_in_flight.clear()
    OrderManager._reconcile_positions(manager)
    manager.guard_orphan_position.assert_called_once_with(SYMBOL, 65, 146.65)
