"""Canonical bracket ownership must exclude legacy direct-broker trailing."""

from __future__ import annotations

from threading import RLock
from types import SimpleNamespace
from unittest.mock import MagicMock

from nifty_scalper_bot.execution import order_manager_core
from nifty_scalper_bot.execution.order_manager_core import OrderManager

SYMBOL = "NFO:NIFTY26OCT22700CE"


def _owner(monkeypatch, *, canonical: bool) -> OrderManager:
    manager = object.__new__(OrderManager)
    manager._bracket_manager = MagicMock()
    manager._bracket_manager.is_symbol_managed.return_value = canonical
    manager._logger = MagicMock()
    manager._market_data = MagicMock()
    manager._data_hub = MagicMock()
    manager._indicator_engine = None
    manager._trailing_journal = MagicMock()
    manager._trailing = {}
    manager._lock = RLock()
    manager._broker = MagicMock()

    class LegacyTrailing:
        def __init__(self, **kwargs):
            self.order_id = kwargs["sl_order_id"]
            self.symbol = kwargs["symbol"]
            self.on_tick = MagicMock()

    monkeypatch.setattr(order_manager_core, "TrailingStopController", LegacyTrailing)
    return manager


def _attach(manager: OrderManager, stop_id: str = "old-stop") -> str:
    return manager.attach_trailing_stop(
        entry_order_id="entry-1",
        sl_order_id=stop_id,
        symbol=SYMBOL,
        side="BUY",
        entry_price=150.0,
        spec=MagicMock(),
    )


def test_canonical_owner_never_attaches_legacy_broker_modifier(monkeypatch) -> None:
    manager = _owner(monkeypatch, canonical=True)
    manager._market_data = None

    assert _attach(manager) == "entry-1"
    assert manager._trailing == {}
    manager._data_hub.subscribe_ticks.assert_not_called()
    manager._broker.modify_order.assert_not_called()


def test_legacy_listener_uses_symmetric_datahub_subscribe_unsubscribe(
    monkeypatch,
) -> None:
    manager = _owner(monkeypatch, canonical=False)
    assert _attach(manager) == "entry-1"
    manager._data_hub.subscribe_ticks.assert_called_once()
    manager._market_data.subscribe.assert_not_called()

    assert manager.stop_trailing("entry-1") is True
    manager._data_hub.unsubscribe_ticks.assert_called_once()
    subscribed = manager._data_hub.subscribe_ticks.call_args.args[1]
    detached = manager._data_hub.unsubscribe_ticks.call_args.args[1]
    assert detached is subscribed
    assert manager.stop_trailing("entry-1") is False


def test_repeated_attach_is_idempotent_and_new_stop_replaces_listener(
    monkeypatch,
) -> None:
    manager = _owner(monkeypatch, canonical=False)
    _attach(manager)
    _attach(manager)
    assert manager._data_hub.subscribe_ticks.call_count == 1

    _attach(manager, stop_id="new-stop")
    assert manager._data_hub.unsubscribe_ticks.call_count == 1
    assert manager._data_hub.subscribe_ticks.call_count == 2
    assert manager._trailing["entry-1"][0].order_id == "new-stop"


def test_canonical_ownership_transition_removes_old_legacy_listener(
    monkeypatch,
) -> None:
    manager = _owner(monkeypatch, canonical=False)
    _attach(manager)
    manager._bracket_manager.is_symbol_managed.return_value = True
    assert _attach(manager) == "entry-1"
    assert manager._trailing == {}
    manager._data_hub.unsubscribe_ticks.assert_called_once()


def test_legacy_resurrection_skips_canonical_bracket_owner(monkeypatch) -> None:
    manager = _owner(monkeypatch, canonical=True)
    manager._brackets = {
        "entry-1": SimpleNamespace(
            entry_id="entry-1",
            symbol=SYMBOL,
            side="BUY",
            entry_price=150.0,
            remaining_position=lambda: 65,
            stop_order_id="old-stop",
            trailing_spec=MagicMock(),
        )
    }
    manager.attach_trailing_stop = MagicMock()

    OrderManager._resurrect_trailing_stops(manager)
    manager.attach_trailing_stop.assert_not_called()
