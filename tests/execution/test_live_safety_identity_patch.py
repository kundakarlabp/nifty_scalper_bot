from __future__ import annotations

from types import SimpleNamespace
import threading

import pytest

from nifty_scalper_bot.execution import live_safety_identity
from nifty_scalper_bot.execution.bracket_core import BracketExitLifecycle, BracketManager
from nifty_scalper_bot.execution.position_manager import PositionManager


class FakeOrderManager:
    def __init__(self, order_id: str = "EXIT-1") -> None:
        self.order_id = order_id
        self.place_order_calls: list[dict] = []
        self.cancelled: list[str] = []
        self._last_order_decision = {}

    def place_order(self, **kwargs):
        self.place_order_calls.append(dict(kwargs))
        return self.order_id

    def cancel_order(self, order_id: str) -> None:
        self.cancelled.append(order_id)


def _fake_bracket(**overrides):
    values = {
        "entry_order_id": "ENTRY-1",
        "trade_lifecycle_id": "LIFE-1",
        "bracket_id": "BRKT-1",
        "symbol": "NFO:NIFTY2670724250PE",
        "side": "BUY",
        "remaining_quantity": 65,
        "exit_state": BracketExitLifecycle.OPEN_ACTIVE.value,
        "entry_status": BracketExitLifecycle.OPEN_ACTIVE.value,
        "exit_pending": False,
        "exit_order_id": None,
        "pending_exit_order_id": None,
        "exit_attempt_count": 1,
        "last_exit_error": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _manager_for_submit(bracket):
    manager = object.__new__(BracketManager)
    manager.order_manager = FakeOrderManager()
    manager.get_bracket = lambda bracket_id: bracket
    manager._price_exit_order = lambda **kwargs: (
        "LIMIT",
        88.05,
        {"mode": "PROFIT_LIMIT", "bid": 88.3, "ask": 88.55, "ltp": 88.55, "fallback": False},
    )
    manager._is_fatal_exit_error = lambda message: False
    return manager


def test_submit_exit_order_carries_immutable_exit_identity():
    live_safety_identity.apply_patches()
    bracket = _fake_bracket()
    manager = _manager_for_submit(bracket)

    result = manager.submit_exit_order(
        symbol="NIFTY2670724250PE",
        qty=65,
        reason="HARD_TP_BREACH",
        bracket_id="BRKT-1",
    )

    assert result.accepted is True
    [call] = manager.order_manager.place_order_calls
    assert call["symbol"] == "NFO:NIFTY2670724250PE"
    assert call["side"] == "SELL"
    assert call["intent"] == "EXIT"
    assert call["linked_entry_order_id"] == "ENTRY-1"
    assert call["trade_lifecycle_id"] == "LIFE-1"
    assert call["bracket_id"] == "BRKT-1"


def test_forced_market_escalation_carries_exit_identity():
    live_safety_identity.apply_patches()
    bracket = _fake_bracket(
        exit_state=BracketExitLifecycle.EXIT_REJECTED_FATAL.value,
        pending_exit_order_id="STALE-EXIT",
        exit_order_id=None,
    )
    manager = object.__new__(BracketManager)
    manager.order_manager = FakeOrderManager("EXIT-MKT-1")
    manager._lock = threading.RLock()
    manager._exit_force_market_on_escalation = True
    manager._notify_event = lambda *args, **kwargs: None

    manager._escalate_exit_locked(bracket, "fatal_or_retry_exhausted")

    assert manager.order_manager.cancelled == ["STALE-EXIT"]
    [call] = manager.order_manager.place_order_calls
    assert call["order_type"] == "MARKET"
    assert call["intent"] == "EXIT"
    assert call["linked_entry_order_id"] == "ENTRY-1"
    assert call["trade_lifecycle_id"] == "LIFE-1"
    assert call["bracket_id"] == "BRKT-1"
    assert bracket.exit_order_id == "EXIT-MKT-1"


def test_position_manager_canonicalizes_bare_and_nfo_symbols(tmp_path):
    live_safety_identity.apply_patches()
    manager = PositionManager(str(tmp_path / "positions.json"))

    manager.open_position("NIFTY2670724250PE", "LONG", 65, 75.0)

    assert manager.has_position("NFO:NIFTY2670724250PE") is True
    assert manager.has_position("nfo:nifty2670724250pe") is True
    assert manager.is_flat("NFO:NIFTY2670724250PE") is False
    position = manager.get_position("nfo:nifty2670724250pe")
    assert position is not None
    assert position.symbol == "NFO:NIFTY2670724250PE"

    manager.update_position_price("NFO:NIFTY2670724250PE", 88.55)
    assert manager.get_position("NIFTY2670724250PE").current_price == pytest.approx(88.55)


def test_position_manager_does_not_double_quantity_on_alias_collision(tmp_path):
    live_safety_identity.apply_patches()
    manager = PositionManager(str(tmp_path / "positions.json"))
    manager.open_position("NIFTY2670724250PE", "LONG", 65, 75.0)

    # Simulate old persisted/runtime state that already had both aliases. The
    # canonicalizer must collapse aliases without summing quantity to 130.
    existing = manager.get_position("NFO:NIFTY2670724250PE")
    duplicate = SimpleNamespace(**existing.to_dict())
    duplicate.symbol = "NFO:NIFTY2670724250PE"
    duplicate.quantity = 65
    manager._positions["NIFTY2670724250PE"] = existing
    manager._positions["NFO:NIFTY2670724250PE"] = duplicate

    manager.save_state()

    assert sorted(manager._positions) == ["NFO:NIFTY2670724250PE"]
    assert manager.get_position("NIFTY2670724250PE").quantity == 65
