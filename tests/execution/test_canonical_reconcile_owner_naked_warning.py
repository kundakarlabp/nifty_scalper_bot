"""Guard broker/local position reconciliation with canonical bracket ownership.

The local OrderManager _brackets dictionary is not the live BracketManager
registry; a matching broker quantity is not evidence of an orphan.
"""

from __future__ import annotations

import time
from threading import Lock, RLock
from types import SimpleNamespace
from unittest.mock import Mock

from nifty_scalper_bot.execution.order_manager_core import OrderManager


SYMBOL = "NFO:NIFTY26OCT22650CE"


def _manager(*, managed: bool, local_present: bool = True) -> SimpleNamespace:
    canonical_brackets = SimpleNamespace(
        _reconcile_lock=Lock(),
        is_symbol_managed=Mock(return_value=managed),
        has_active_bracket=Mock(return_value=managed),
    )
    positions = (
        [SimpleNamespace(symbol=SYMBOL, quantity=65, entry_price=146.65)]
        if local_present
        else []
    )
    return SimpleNamespace(
        _broker=SimpleNamespace(
            get_positions=lambda: [
                {
                    "tradingsymbol": "NIFTY26OCT22650CE",
                    "exchange": "NFO",
                    "quantity": 65,
                    "average_price": 146.65,
                }
            ]
        ),
        _positions=SimpleNamespace(get_open_positions=lambda: positions),
        _bracket_manager=canonical_brackets,
        _brackets={},  # Legacy registry intentionally empty on the live path.
        _lock=RLock(),
        _missing_counts={},
        _orders={},
        _entries_in_flight={},
        ENTRY_INFLIGHT_TTL_SEC=30.0,
        _recent_orphans=set(),
        _logger=Mock(),
        guard_orphan_position=Mock(return_value=True),
    )


def test_quantity_match_with_canonical_owner_is_not_naked() -> None:
    manager = _manager(managed=True)
    OrderManager._reconcile_positions(manager)
    manager._bracket_manager.is_symbol_managed.assert_called_with(SYMBOL)
    manager.guard_orphan_position.assert_not_called()
    assert manager._missing_counts[SYMBOL] == 0
    assert not any(
        "Naked Position Detected" in str(call)
        for call in manager._logger.warning.call_args_list
    )


def test_quantity_match_without_canonical_owner_still_activates_guard() -> None:
    manager = _manager(managed=False)
    OrderManager._reconcile_positions(manager)
    manager.guard_orphan_position.assert_called_once_with(
        SYMBOL, 65, 146.65, consume_existing=True
    )
    assert any(
        "Naked Position Detected" in str(call)
        for call in manager._logger.warning.call_args_list
    )


def test_broker_visible_pending_owner_is_not_permanently_marked_orphan() -> None:
    manager = _manager(managed=True, local_present=False)
    OrderManager._reconcile_positions(manager)
    manager.guard_orphan_position.assert_not_called()
    assert SYMBOL not in manager._recent_orphans


def test_genuine_orphan_is_adopted_only_once_per_reconcile_session() -> None:
    manager = _manager(managed=False, local_present=False)
    OrderManager._reconcile_positions(manager)
    OrderManager._reconcile_positions(manager)
    manager.guard_orphan_position.assert_called_once_with(
        SYMBOL, 65, 146.65
    )


def test_recent_pending_entry_is_not_latched_as_orphan() -> None:
    manager = _manager(managed=False, local_present=False)
    manager._entries_in_flight[SYMBOL] = time.time()
    OrderManager._reconcile_positions(manager)
    manager.guard_orphan_position.assert_not_called()
    assert SYMBOL not in manager._recent_orphans

    manager._entries_in_flight.clear()
    OrderManager._reconcile_positions(manager)
    manager.guard_orphan_position.assert_called_once_with(
        SYMBOL, 65, 146.65
    )
