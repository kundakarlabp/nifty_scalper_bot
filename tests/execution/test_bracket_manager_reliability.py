"""Reliability tests for virtual bracket exit execution."""

from __future__ import annotations

import time
from unittest.mock import Mock

from nifty_scalper_bot.execution.bracket_manager import BracketManager


def _build_manager() -> tuple[BracketManager, Mock]:
    """Create manager with mock dependencies. Args: none; Returns: tuple; Raises: none."""
    order_manager = Mock()
    order_manager.is_live_mode.return_value = False
    order_manager.wait_for_fill.return_value = True
    manager = BracketManager(order_manager=order_manager)
    manager.attach_exit_executor(lambda symbol, qty: f"exit-{symbol}-{qty}")
    return manager, order_manager


def _wait_for_exit(manager: BracketManager, entry_id: str, timeout_s: float = 1.0):
    deadline = time.time() + timeout_s
    bracket = manager.get_bracket(entry_id)
    while time.time() < deadline and bracket is not None and not bracket.exit_executed:
        time.sleep(0.01)
        bracket = manager.get_bracket(entry_id)
    return bracket


def test_stop_loss_cross_detection_on_jump() -> None:
    """Ensure SL triggers on jump across threshold."""
    manager, _ = _build_manager()
    manager.register_virtual_bracket(
        order_id="entry-1",
        symbol="NFO:NIFTYTEST",
        side="BUY",
        qty=1,
        price=105.0,
        sl=100.0,
        tp=120.0,
    )
    manager.confirm_entry_fill("entry-1", 105.0)

    manager.on_tick("NFO:NIFTYTEST", 105.0)
    manager.on_tick("NFO:NIFTYTEST", 104.0)
    manager.on_tick("NFO:NIFTYTEST", 97.0)

    bracket = _wait_for_exit(manager, "entry-1")
    assert bracket is not None
    assert bracket.exit_executed is True
    assert bracket.active is False


def test_trailing_stop_never_moves_backwards() -> None:
    """Ensure manual trailing updates are monotonic."""
    manager, _ = _build_manager()
    manager.register_virtual_bracket(
        order_id="entry-2",
        symbol="NFO:NIFTYTEST2",
        side="BUY",
        qty=1,
        price=100.0,
        sl=95.0,
        tp=120.0,
    )
    manager.confirm_entry_fill("entry-2", 100.0)
    manager.update_trailing_sl("NFO:NIFTYTEST2", 98.0)
    manager.update_trailing_sl("NFO:NIFTYTEST2", 96.0)
    bracket = manager.get_bracket("entry-2")
    assert bracket is not None
    assert bracket.sl_trigger_price == 98.0


def test_fallback_market_exit_executes_after_retry_exhaustion() -> None:
    """Ensure fallback market exit runs if retries fail."""
    order_manager = Mock()
    order_manager.is_live_mode.return_value = False
    order_manager.place_order.return_value = "fallback-1"
    order_manager.wait_for_fill.return_value = True
    manager = BracketManager(order_manager=order_manager)
    manager.attach_exit_executor(lambda _symbol, _qty: None)
    manager.register_virtual_bracket(
        order_id="entry-3",
        symbol="NFO:NIFTYTEST3",
        side="BUY",
        qty=1,
        price=100.0,
        sl=99.0,
        tp=120.0,
    )
    manager.confirm_entry_fill("entry-3", 100.0)

    manager.on_tick("NFO:NIFTYTEST3", 98.0)

    deadline = time.time() + 1.0
    while time.time() < deadline and not order_manager.place_order.called:
        time.sleep(0.01)

    assert order_manager.place_order.called
    bracket = _wait_for_exit(
        manager, "entry-3", timeout_s=max(0.0, deadline - time.time())
    )
    assert bracket is not None
    assert bracket.exit_executed is True


def test_exit_state_mutates_only_after_broker_confirmation() -> None:
    """Ensure state mutates only after broker confirmation."""
    order_manager = Mock()
    order_manager.is_live_mode.return_value = False
    order_manager.wait_for_fill.return_value = False
    order_manager.place_order.return_value = None
    manager = BracketManager(order_manager=order_manager)
    manager.attach_exit_executor(lambda _symbol, _qty: "exit-pending")
    manager.register_virtual_bracket(
        order_id="entry-4",
        symbol="NFO:NIFTYTEST4",
        side="BUY",
        qty=1,
        price=100.0,
        sl=99.0,
        tp=120.0,
    )
    manager.confirm_entry_fill("entry-4", 100.0)

    manager.on_tick("NFO:NIFTYTEST4", 98.0)
    bracket = manager.get_bracket("entry-4")
    assert bracket is not None
    assert bracket.exit_executed is False
    assert bracket.active is True


def test_websocket_tick_jump_scenario_triggers_sl() -> None:
    """Ensure websocket jump sequence triggers SL."""
    manager, _ = _build_manager()
    manager.register_virtual_bracket(
        order_id="entry-5",
        symbol="NFO:NIFTYTEST5",
        side="BUY",
        qty=1,
        price=105.0,
        sl=100.0,
        tp=120.0,
    )
    manager.confirm_entry_fill("entry-5", 105.0)

    for tick in [105.0, 104.0, 97.0]:
        manager.on_tick_event({"symbol": "NFO:NIFTYTEST5", "ltp": tick})

    bracket = _wait_for_exit(manager, "entry-5")
    assert bracket is not None
    assert bracket.exit_executed is True
