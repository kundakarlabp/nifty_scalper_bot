from __future__ import annotations

from unittest.mock import Mock

from nifty_scalper_bot.execution.bracket_manager import BracketManager

SYMBOL = "NFO:NIFTY2680724500CE"


def _manager(monkeypatch, tmp_path) -> BracketManager:
    monkeypatch.setenv("BRACKET_FILL_LEDGER_PATH", str(tmp_path / "fills.db"))
    manager = BracketManager(order_manager=Mock())
    manager._running = False
    manager._watchdog_thread.join(timeout=1.0)
    manager.register_virtual_bracket(
        "entry-partial",
        SYMBOL,
        "BUY",
        130,
        100.0,
        95.0,
        110.0,
        activate_immediately=False,
    )
    return manager


def _entry_fill(manager: BracketManager):
    bracket = manager.get_bracket("entry-partial")
    assert bracket is not None
    assert manager._fill_ledger is not None
    fills = manager._fill_ledger.load_fills(bracket.bracket_id)
    entries = [fill for fill in fills if fill.kind == "ENTRY"]
    assert len(entries) == 1
    return bracket, entries[0]


def test_partial_fill_reconciles_bracket_and_entry_ledger(monkeypatch, tmp_path) -> None:
    manager = _manager(monkeypatch, tmp_path)

    manager.confirm_entry_fill("entry-partial", 100.0, 65)

    bracket, entry = _entry_fill(manager)
    assert bracket.quantity == 65
    assert bracket.remaining_quantity == 65
    assert bracket.active is True
    assert entry.quantity == 65


def test_late_same_price_callback_reconciles_smaller_quantity(monkeypatch, tmp_path) -> None:
    manager = _manager(monkeypatch, tmp_path)
    manager.confirm_entry_fill("entry-partial", 100.0)
    before, entry_before = _entry_fill(manager)
    assert before.quantity == 130
    assert entry_before.quantity == 130

    manager.confirm_entry_fill("entry-partial", 100.0, 65)

    bracket, entry = _entry_fill(manager)
    assert bracket.quantity == 65
    assert bracket.remaining_quantity == 65
    assert entry.quantity == 65


def test_identical_duplicate_callback_remains_idempotent(monkeypatch, tmp_path) -> None:
    manager = _manager(monkeypatch, tmp_path)
    manager.confirm_entry_fill("entry-partial", 100.0, 65)

    manager.confirm_entry_fill("entry-partial", 100.0, 65)

    bracket, entry = _entry_fill(manager)
    assert bracket.quantity == 65
    assert entry.quantity == 65


def test_later_cumulative_fill_grows_protection_up_to_original_request(
    monkeypatch, tmp_path
) -> None:
    manager = _manager(monkeypatch, tmp_path)
    manager.confirm_entry_fill("entry-partial", 100.0, 30)

    manager.confirm_entry_fill("entry-partial", 100.0, 65)

    bracket, entry = _entry_fill(manager)
    assert bracket.requested_entry_quantity == 130
    assert bracket.quantity == 65
    assert bracket.remaining_quantity == 65
    assert entry.quantity == 65


def test_cumulative_fill_growth_preserves_already_exited_quantity(
    monkeypatch, tmp_path
) -> None:
    manager = _manager(monkeypatch, tmp_path)
    manager.confirm_entry_fill("entry-partial", 100.0, 30)
    bracket = manager.get_bracket("entry-partial")
    assert bracket is not None
    bracket.remaining_quantity = 20

    manager.confirm_entry_fill("entry-partial", 100.0, 65)

    bracket, entry = _entry_fill(manager)
    assert bracket.quantity == 65
    assert bracket.remaining_quantity == 55
    assert entry.quantity == 65


def test_overreported_quantity_never_grows_bracket_or_ledger(monkeypatch, tmp_path) -> None:
    manager = _manager(monkeypatch, tmp_path)
    manager.confirm_entry_fill("entry-partial", 100.0, 260)

    bracket, entry = _entry_fill(manager)
    assert bracket.quantity == 130
    assert bracket.remaining_quantity == 130
    assert entry.quantity == 130


def test_live_late_cumulative_fill_resizes_existing_exchange_stop(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("EXCHANGE_PROTECTIVE_STOP_ENABLED", "true")
    manager = _manager(monkeypatch, tmp_path)
    monkeypatch.setattr(manager, "_is_live_execution", lambda: True)
    manager.order_manager.place_order.return_value = "stop-1"
    manager.order_manager.modify_order.return_value = True

    manager.confirm_entry_fill("entry-partial", 100.0, 30)
    bracket = manager.get_bracket("entry-partial")
    assert bracket is not None
    assert bracket.exchange_stop_quantity == 30
    assert bracket.exchange_stop_order_id == "stop-1"

    manager.confirm_entry_fill("entry-partial", 100.0, 65)

    assert bracket.quantity == 65
    assert bracket.remaining_quantity == 65
    assert bracket.exchange_stop_quantity == 65
    assert bracket.exchange_stop_order_id == "stop-1"
    assert manager.order_manager.place_order.call_count == 1
    assert manager.order_manager.modify_order.call_count == 1
    args, kwargs = manager.order_manager.modify_order.call_args
    assert args == ("stop-1",)
    assert kwargs["quantity"] == 65
    assert kwargs["trigger_price"] == 95.0

    manager.confirm_entry_fill("entry-partial", 100.0, 65)
    assert manager.order_manager.place_order.call_count == 1
    assert manager.order_manager.modify_order.call_count == 1


def test_failed_live_stop_resize_retains_original_quantity_and_degradation(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("EXCHANGE_PROTECTIVE_STOP_ENABLED", "true")
    manager = _manager(monkeypatch, tmp_path)
    monkeypatch.setattr(manager, "_is_live_execution", lambda: True)
    manager.order_manager.place_order.return_value = "stop-1"
    manager.order_manager.modify_order.return_value = False

    manager.confirm_entry_fill("entry-partial", 100.0, 30)
    manager.confirm_entry_fill("entry-partial", 100.0, 65)

    bracket = manager.get_bracket("entry-partial")
    assert bracket is not None
    assert bracket.exchange_stop_quantity == 30
    assert bracket.remaining_quantity == 65
    assert bracket.exchange_stop_degraded_reason == "modify_failed"
    assert manager.order_manager.place_order.call_count == 1


def test_live_quantity_only_resize_never_loosens_exchange_stop(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("EXCHANGE_PROTECTIVE_STOP_ENABLED", "true")
    manager = _manager(monkeypatch, tmp_path)
    monkeypatch.setattr(manager, "_is_live_execution", lambda: True)
    manager.order_manager.place_order.return_value = "stop-1"
    manager.order_manager.modify_order.return_value = True
    manager.confirm_entry_fill("entry-partial", 100.0, 30)
    bracket = manager.get_bracket("entry-partial")
    assert bracket is not None
    bracket.sl_trigger_price = 90.0  # stale virtual trail; broker stop remains at 95

    manager.confirm_entry_fill("entry-partial", 100.0, 65)

    assert bracket.exchange_stop_trigger_price == 95.0
    assert manager.order_manager.modify_order.call_args.kwargs["trigger_price"] == 95.0
