"""Restart safety for consumed deterministic signal ids (P0-2)."""

from __future__ import annotations

import logging
from threading import RLock
from types import MethodType
from collections import deque
from types import SimpleNamespace

from nifty_scalper_bot.execution.order_manager_core import OrderManager
from nifty_scalper_bot.execution.position_manager import PositionManager


def _om(position_manager: PositionManager) -> SimpleNamespace:
    """Minimal stand-in exposing exactly the state the helpers touch."""
    stub = SimpleNamespace(
        _positions=position_manager,
        _lock=RLock(),
        _logger=logging.getLogger("test.consumed_signals"),
        _seen_signal_ids=set(),
        _signal_history=deque(maxlen=10_000),
    )
    for name in (
        "_consumed_signal_state_path",
        "_trading_date_key",
        "_restore_consumed_signal_ids",
        "_persist_consumed_signal_ids",
        "_remember_signal",
        "_is_duplicate_signal",
    ):
        setattr(stub, name, MethodType(getattr(OrderManager, name), stub))
    return stub


def test_consumed_signal_survives_restart(tmp_path) -> None:
    state_file = str(tmp_path / "positions.json")
    first = PositionManager(state_file=state_file)
    first.save_state()

    writer = _om(first)
    writer._remember_signal("abc123")
    assert writer._is_duplicate_signal("abc123") is True

    restarted = _om(PositionManager(state_file=state_file))
    restarted._restore_consumed_signal_ids()

    assert restarted._is_duplicate_signal("abc123") is True
    assert restarted._is_duplicate_signal("other") is False


def test_previous_trading_day_ids_are_not_restored(tmp_path) -> None:
    state_file = str(tmp_path / "positions.json")
    pm = PositionManager(state_file=state_file)
    pm.save_state()

    writer = _om(pm)
    writer._remember_signal("abc123")

    restarted = _om(PositionManager(state_file=state_file))
    restarted._positions._trading_date_ist = lambda: "2000-01-01"
    restarted._restore_consumed_signal_ids()

    assert restarted._is_duplicate_signal("abc123") is False


def test_missing_state_file_is_a_no_op(tmp_path) -> None:
    pm = PositionManager(state_file=str(tmp_path / "absent.json"))
    stub = _om(pm)

    stub._persist_consumed_signal_ids()
    stub._restore_consumed_signal_ids()

    assert stub._seen_signal_ids == set()
