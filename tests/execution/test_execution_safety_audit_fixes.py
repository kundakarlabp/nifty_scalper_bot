from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
import threading
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.execution import BracketManager
from nifty_scalper_bot.execution import bracket_core
from nifty_scalper_bot.execution.bracket_core import BracketState
from nifty_scalper_bot.execution.execution_policy import ExecutionPolicy
from nifty_scalper_bot.execution.margin_engine import MarginInputs
from nifty_scalper_bot.execution.position_manager import (
    Order,
    Position,
    PositionManager,
    normalize_broker_order_status,
)
from nifty_scalper_bot.data.rest.zerodha_client import ZerodhaKiteClient
from nifty_scalper_bot.utils.errors import BrokerError
from nifty_scalper_bot.utils.errors import OrderPlacementError


SYMBOL = "NFO:NIFTY2662324050PE"


def _position() -> Position:
    return Position(
        symbol=SYMBOL,
        side="LONG",
        quantity=65,
        entry_price=100.0,
        entry_time=datetime.now(timezone.utc),
        current_price=101.0,
    )


def _position_manager(tmp_path) -> PositionManager:
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    manager._schedule_retry_after_failure = lambda *_args, **_kwargs: None
    manager._positions[SYMBOL] = _position()
    return manager


def test_none_broker_snapshot_fails_closed_and_preserves_position(tmp_path) -> None:
    manager = _position_manager(tmp_path)
    manager.set_broker_client(SimpleNamespace(get_positions=lambda: None))

    assert manager.reconcile_now() is False
    assert manager.get_position(SYMBOL) is not None


def test_malformed_broker_snapshot_fails_closed_and_preserves_position(tmp_path) -> None:
    manager = _position_manager(tmp_path)
    manager.set_broker_client(
        SimpleNamespace(
            get_positions=lambda: [
                {
                    "symbol": SYMBOL,
                    "product": "MIS",
                    "quantity": "not-a-number",
                    "average_price": 100.0,
                    "last_price": 101.0,
                }
            ]
        )
    )

    assert manager.reconcile_now() is False
    assert manager.get_position(SYMBOL) is not None


def test_explicit_empty_snapshot_is_authoritative_flat(tmp_path) -> None:
    manager = _position_manager(tmp_path)
    flattened: list[list[str]] = []
    manager.set_on_symbols_flat(lambda symbols: flattened.append(list(symbols)))
    manager.set_broker_client(SimpleNamespace(get_positions=lambda: []))

    assert manager.reconcile_now() is True
    assert manager.get_position(SYMBOL) is None
    assert flattened == [[SYMBOL]]


def test_broker_realised_field_updates_daily_realised_without_using_total_pnl(tmp_path) -> None:
    manager = _position_manager(tmp_path)
    manager.establish_pnl_session_baseline(-100.0)
    manager.synchronize_with_broker(
        [
            {
                "symbol": SYMBOL,
                "product": "MIS",
                "quantity": 0,
                "realised": -125.5,
                "pnl": 9999.0,
                "m2m": 9999.0,
            }
        ]
    )
    assert manager.get_realized_pnl() == pytest.approx(-25.5)


def test_update_from_order_uses_fill_price_and_existing_fill_lifecycle(tmp_path) -> None:
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    order = Order(
        order_id="entry-1",
        symbol=SYMBOL,
        side="BUY",
        order_type="MARKET",
        quantity=65,
        price=100.0,
        status="FILLED",
        filled_quantity=65,
        fill_price=101.0,
        intent="ENTRY",
    )
    manager.update_from_order(order)
    position = manager.get_position(SYMBOL)
    assert position is not None
    assert position.entry_price == pytest.approx(101.0)
    assert position.quantity == 65


def test_exit_sell_fill_after_reconciled_flat_does_not_open_short(tmp_path) -> None:
    manager = _position_manager(tmp_path)
    manager.add_pending_order(
        "exit-1", SYMBOL, "SELL", 65, 116.0, "MARKET", intent="EXIT"
    )
    manager.set_broker_client(SimpleNamespace(get_positions=lambda: []))
    assert manager.reconcile_now() is True
    assert manager.get_position(SYMBOL) is None

    manager.update_order_status("exit-1", "COMPLETE", fill_price=116.0)

    assert manager.get_position(SYMBOL) is None
    assert manager.get_open_positions() == []
    assert "exit-1" in manager._processed_order_ids


def test_duplicate_exit_update_is_idempotent_across_restart(tmp_path) -> None:
    state_path = tmp_path / "positions.json"
    manager = _position_manager(tmp_path)
    manager.add_pending_order(
        "exit-dup", SYMBOL, "SELL", 65, 116.0, "MARKET", intent="EXIT"
    )
    manager.set_broker_client(SimpleNamespace(get_positions=lambda: []))
    assert manager.reconcile_now() is True
    manager.update_order_status("exit-dup", "COMPLETE", fill_price=116.0)
    first_realized = manager.get_realized_pnl()

    restarted = PositionManager(state_file=str(state_path))
    restarted._schedule_retry_after_failure = lambda *_args, **_kwargs: None
    restarted.add_pending_order(
        "exit-dup", SYMBOL, "SELL", 65, 116.0, "MARKET", intent="EXIT"
    )

    assert "exit-dup" in restarted._processed_order_ids
    assert restarted.get_position(SYMBOL) is None
    assert restarted.get_realized_pnl() == pytest.approx(first_realized)


def test_unknown_legacy_sell_while_flat_never_opens_short(tmp_path) -> None:
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    order = Order(
        order_id="legacy-sell",
        symbol=SYMBOL,
        side="SELL",
        order_type="MARKET",
        quantity=65,
        price=116.0,
        status="FILLED",
        filled_quantity=65,
        fill_price=116.0,
    )
    manager._orders[order.order_id] = order

    manager.update_order_status(order.order_id, "COMPLETE", fill_price=116.0)

    assert manager.get_position(SYMBOL) is None


def test_explicit_short_entry_still_opens_short(tmp_path) -> None:
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    order = Order(
        order_id="short-entry",
        symbol=SYMBOL,
        side="SELL",
        order_type="MARKET",
        quantity=65,
        price=116.0,
        status="FILLED",
        filled_quantity=65,
        fill_price=116.0,
        intent="ENTRY",
    )
    manager.update_from_order(order)

    position = manager.get_position(SYMBOL)
    assert position is not None
    assert position.side == "SHORT"
    assert position.quantity == 65
    assert manager.current_entry_protection_blocker(SYMBOL) == "entry_protection_incomplete"
    manager.confirm_entry_protection("short-entry", "short-bracket", 65)
    assert manager.current_entry_protection_blocker(SYMBOL) is None


def test_broker_status_normalization_accepts_zerodha_statuses() -> None:
    assert normalize_broker_order_status("SUBMITTED") == "PENDING"
    assert normalize_broker_order_status("OPEN PENDING") == "OPEN"
    assert normalize_broker_order_status("TRIGGER PENDING") == "OPEN"
    assert normalize_broker_order_status("PUT ORDER REQ RECEIVED") == "PENDING"
    assert normalize_broker_order_status("PUT ORDER REQUEST RECEIVED") == "PENDING"
    assert normalize_broker_order_status("COMPLETE") == "FILLED"


def test_confirmed_incident_delayed_entry_after_exit_stays_flat(tmp_path) -> None:
    symbol = "NFO:NIFTY2670724050CE"
    entry_order_id = "2072238244200112128"
    exit_order_id = "2072245044739760128"
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    manager.add_pending_order(
        entry_order_id, symbol, "BUY", 65, 145.15, "MARKET", intent="ENTRY"
    )
    manager.open_position(
        symbol=symbol,
        side="LONG",
        quantity=65,
        entry_price=145.15,
        order_id=entry_order_id,
    )
    manager.add_pending_order(
        exit_order_id, symbol, "SELL", 65, 144.05, "MARKET", intent="EXIT"
    )
    manager.set_broker_client(SimpleNamespace(get_positions=lambda: []))
    assert manager.reconcile_now() is True

    for _ in range(3):
        manager.apply_broker_order_update(
            exit_order_id,
            {
                "status": "COMPLETE",
                "filled_quantity": 65,
                "average_price": 144.05,
            },
        )
        manager.apply_broker_order_update(
            entry_order_id,
            {
                "status": "COMPLETE",
                "filled_quantity": 65,
                "average_price": 145.15,
            },
        )

    restarted = PositionManager(state_file=str(tmp_path / "positions.json"))
    restarted.add_pending_order(
        exit_order_id, symbol, "SELL", 65, 144.05, "MARKET", intent="EXIT"
    )
    restarted.add_pending_order(
        entry_order_id, symbol, "BUY", 65, 145.15, "MARKET", intent="ENTRY"
    )
    restarted.apply_broker_order_update(
        exit_order_id,
        {"status": "COMPLETE", "filled_quantity": 65, "average_price": 144.05},
    )
    restarted.apply_broker_order_update(
        entry_order_id,
        {"status": "COMPLETE", "filled_quantity": 65, "average_price": 145.15},
    )

    assert restarted.get_position(symbol) is None
    assert restarted.get_open_positions() == []
    assert restarted.get_realized_pnl() == pytest.approx((144.05 - 145.15) * 65)
    assert entry_order_id in restarted._processed_order_ids
    assert exit_order_id in restarted._terminal_orders


def test_partial_fills_apply_only_cumulative_delta(tmp_path) -> None:
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    manager.add_pending_order("partial-entry", SYMBOL, "BUY", 65, 100.0, "MARKET", intent="ENTRY")

    manager.apply_broker_order_update(
        "partial-entry",
        {"status": "PARTIALLY FILLED", "filled_quantity": 25, "average_price": 100.0},
    )
    partial_position = manager.get_position(SYMBOL)
    assert partial_position is not None
    assert partial_position.quantity == 25

    manager.apply_broker_order_update(
        "partial-entry",
        {"status": "COMPLETE", "filled_quantity": 65, "average_price": 100.0},
    )

    position = manager.get_position(SYMBOL)
    assert position is not None
    assert position.quantity == 65


def test_partial_exit_fill_reduces_only_cumulative_delta(tmp_path) -> None:
    manager = _position_manager(tmp_path)
    manager.add_pending_order(
        "partial-exit", SYMBOL, "SELL", 65, 99.0, "MARKET", intent="EXIT"
    )

    manager.apply_broker_order_update(
        "partial-exit",
        {"status": "PARTIALLY FILLED", "filled_quantity": 25, "average_price": 99.0},
    )
    partial_position = manager.get_position(SYMBOL)
    assert partial_position is not None
    assert partial_position.quantity == 40

    manager.apply_broker_order_update(
        "partial-exit",
        {"status": "COMPLETE", "filled_quantity": 65, "average_price": 99.0},
    )

    assert manager.get_position(SYMBOL) is None
    assert manager.get_realized_pnl() == pytest.approx((99.0 - 100.0) * 65)


def test_simultaneous_duplicate_complete_update_is_single_lifecycle_mutation(tmp_path) -> None:
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    manager.add_pending_order(
        "duplicate-entry", SYMBOL, "BUY", 65, 100.0, "MARKET", intent="ENTRY"
    )
    payload = {"status": "COMPLETE", "filled_quantity": 65, "average_price": 100.0}

    threads = [
        threading.Thread(
            target=manager.apply_broker_order_update,
            args=("duplicate-entry", payload),
        )
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2.0)

    position = manager.get_position(SYMBOL)
    assert position is not None
    assert position.quantity == 65
    assert manager._terminal_orders["duplicate-entry"].fill_recorded is True
    assert manager._terminal_orders["duplicate-entry"].lifecycle_resolved is False


def test_unresolved_terminal_exit_is_retained_for_reconciliation(tmp_path) -> None:
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    manager.add_pending_order(
        "unresolved-exit", SYMBOL, "SELL", 65, 99.0, "MARKET", intent="EXIT"
    )

    manager.apply_broker_order_update(
        "unresolved-exit",
        {"status": "COMPLETE", "filled_quantity": 65, "average_price": 99.0},
    )

    assert manager.get_position(SYMBOL) is None
    assert "unresolved-exit" in manager._orders
    assert "unresolved-exit" in manager._unresolved_terminal_orders
    metadata = manager._terminal_orders["unresolved-exit"]
    assert metadata.lifecycle_resolved is False
    assert metadata.accounting_finalized is False
    assert manager.unresolved_terminal_summary()["count"] == 1


def test_variable_average_partial_entry_uses_incremental_notional(tmp_path) -> None:
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    manager.add_pending_order("avg-entry", SYMBOL, "BUY", 65, 100.0, "MARKET", intent="ENTRY")

    manager.apply_broker_order_update(
        "avg-entry",
        {"status": "PARTIALLY FILLED", "filled_quantity": 25, "average_price": 100.0},
    )
    manager.apply_broker_order_update(
        "avg-entry",
        {"status": "COMPLETE", "filled_quantity": 65, "average_price": 102.0},
    )

    position = manager.get_position(SYMBOL)
    assert position is not None
    assert position.quantity == 65
    assert position.entry_price == pytest.approx(102.0)
    order = manager._terminal_orders["avg-entry"]
    assert order.cumulative_filled_quantity == 65


def test_variable_average_partial_exit_uses_incremental_notional(tmp_path) -> None:
    manager = _position_manager(tmp_path)
    manager.add_pending_order(
        "avg-exit", SYMBOL, "SELL", 65, 99.0, "MARKET", intent="EXIT"
    )

    manager.apply_broker_order_update(
        "avg-exit",
        {"status": "PARTIALLY FILLED", "filled_quantity": 25, "average_price": 99.0},
    )
    manager.apply_broker_order_update(
        "avg-exit",
        {"status": "COMPLETE", "filled_quantity": 65, "average_price": 98.0},
    )

    assert manager.get_position(SYMBOL) is None
    assert manager.get_realized_pnl() == pytest.approx((98.0 - 100.0) * 65)


def test_new_entry_same_symbol_after_previous_trade_uses_distinct_lifecycle(tmp_path) -> None:
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    manager.add_pending_order("entry-a", SYMBOL, "BUY", 65, 100.0, "MARKET", intent="ENTRY")
    manager.apply_broker_order_update(
        "entry-a",
        {"status": "COMPLETE", "filled_quantity": 65, "average_price": 100.0},
    )
    manager.add_pending_order("exit-a", SYMBOL, "SELL", 65, 101.0, "MARKET", intent="EXIT")
    manager.apply_broker_order_update(
        "exit-a",
        {"status": "COMPLETE", "filled_quantity": 65, "average_price": 101.0},
    )
    assert manager.get_position(SYMBOL) is None

    manager.add_pending_order("entry-b", SYMBOL, "BUY", 65, 102.0, "MARKET", intent="ENTRY")
    manager.apply_broker_order_update(
        "entry-b",
        {"status": "COMPLETE", "filled_quantity": 65, "average_price": 102.0},
    )

    position = manager.get_position(SYMBOL)
    assert position is not None
    assert position.order_id == "entry-b"
    assert position.quantity == 65
    assert manager._orders.get("entry-b") is not None
    assert manager._terminal_orders["entry-a"].trade_lifecycle_id != manager._terminal_orders["entry-b"].trade_lifecycle_id


def test_status_regression_after_filled_is_noop(tmp_path) -> None:
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    manager.add_pending_order(
        "regression-entry", SYMBOL, "BUY", 65, 100.0, "MARKET", intent="ENTRY"
    )
    manager.apply_broker_order_update(
        "regression-entry",
        {"status": "COMPLETE", "filled_quantity": 65, "average_price": 100.0},
    )

    manager.apply_broker_order_update(
        "regression-entry",
        {"status": "OPEN", "filled_quantity": 65, "average_price": 100.0},
    )

    position = manager.get_position(SYMBOL)
    assert position is not None
    assert position.quantity == 65
    assert manager._terminal_orders["regression-entry"].normalized_status == "FILLED"



def test_duplicate_and_invalid_cumulative_updates_do_not_mutate_position(tmp_path) -> None:
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    manager.add_pending_order("bad-partial", SYMBOL, "BUY", 65, 100.0, "MARKET", intent="ENTRY")
    manager.apply_broker_order_update(
        "bad-partial",
        {"status": "PARTIALLY FILLED", "filled_quantity": 25, "average_price": 100.0},
    )
    manager.apply_broker_order_update(
        "bad-partial",
        {"status": "PARTIALLY FILLED", "filled_quantity": 25, "average_price": 100.0},
    )
    manager.apply_broker_order_update(
        "bad-partial",
        {"status": "PARTIALLY FILLED", "filled_quantity": 20, "average_price": 100.0},
    )
    manager.apply_broker_order_update(
        "bad-partial",
        {"status": "PARTIALLY FILLED", "filled_quantity": 30, "average_price": -1.0},
    )

    position = manager.get_position(SYMBOL)
    assert position is not None
    assert position.quantity == 25
    assert position.entry_price == pytest.approx(100.0)


def test_resolved_terminal_eviction_preserves_unresolved_records(tmp_path) -> None:
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    manager._max_terminal_orders = 1
    manager.add_pending_order("unresolved", SYMBOL, "SELL", 65, 99.0, "MARKET", intent="EXIT")
    manager.apply_broker_order_update(
        "unresolved",
        {"status": "COMPLETE", "filled_quantity": 65, "average_price": 99.0},
    )
    manager._terminal_orders["resolved-a"] = manager._terminal_orders["unresolved"]
    manager._terminal_orders["resolved-a"].lifecycle_resolved = True

    manager._evict_old_terminal_orders()

    assert "unresolved" in manager._terminal_orders
    assert "unresolved" in manager._unresolved_terminal_orders


def test_pnl_reconciliation_mismatch_exposes_entry_blocker(tmp_path) -> None:
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    manager.establish_pnl_session_baseline(-150.0)
    manager._local_realized_pnl = 100.0
    manager._broker_realized_pnl = -60.0
    with manager._lock:
        manager._refresh_realized_pnl_locked()

    snapshot = manager.pnl_reconciliation_snapshot()
    assert snapshot["pnl_reconciliation_status"] == "mismatch"
    assert snapshot["local_confirmed_realized"] == pytest.approx(100.0)
    assert snapshot["broker_realized_snapshot"] == pytest.approx(-60.0)
    assert snapshot["broker_session_realized"] == pytest.approx(90.0)
    assert manager.current_pnl_reconciliation_blocker() == "pnl_reconciliation_mismatch"


def test_entry_bracket_id_alone_does_not_resolve_without_protection_ack(tmp_path) -> None:
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    manager.add_pending_order(
        "entry-protect",
        SYMBOL,
        "BUY",
        65,
        100.0,
        "MARKET",
        intent="ENTRY",
        bracket_id="metadata-only",
    )

    manager.apply_broker_order_update(
        "entry-protect",
        {"status": "COMPLETE", "filled_quantity": 65, "average_price": 100.0},
    )

    metadata = manager._terminal_orders["entry-protect"]
    assert metadata.bracket_applied is False
    assert metadata.lifecycle_resolved is False
    assert metadata.protection_confirmed is False
    assert manager.current_entry_protection_blocker(SYMBOL) == "entry_protection_incomplete"

    with pytest.raises(ValueError):
        manager.confirm_entry_protection("entry-protect", "metadata-only", 25)

    manager.confirm_entry_protection("entry-protect", "metadata-only", 65)

    resolved = manager._terminal_orders["entry-protect"]
    assert resolved.bracket_applied is True
    assert resolved.protected_quantity == 65
    assert resolved.protection_confirmed is True
    assert resolved.lifecycle_resolved is True
    assert manager.current_entry_protection_blocker(SYMBOL) is None


def test_session_pnl_baseline_survives_restart_and_resets_by_ist_day(tmp_path) -> None:
    state_path = tmp_path / "positions.json"
    manager = PositionManager(state_file=str(state_path))
    established = manager.establish_pnl_session_baseline(
        -1000.0,
        snapshot_at=datetime(2026, 7, 2, 3, 45, tzinfo=timezone.utc),
    )
    assert established is True
    manager._broker_realized_pnl = -1100.0
    with manager._lock:
        manager._refresh_realized_pnl_locked()
    assert manager.broker_session_realized_pnl() == pytest.approx(-100.0)
    assert manager.get_realized_pnl() == pytest.approx(-100.0)
    manager.save_state()

    restarted = PositionManager(state_file=str(state_path))
    same_day_replaced = restarted.establish_pnl_session_baseline(
        -1100.0,
        snapshot_at=datetime(2026, 7, 2, 5, 0, tzinfo=timezone.utc),
    )
    assert same_day_replaced is False
    assert restarted.pnl_reconciliation_snapshot()[
        "session_opening_realized_baseline"
    ] == pytest.approx(-1000.0)

    new_day = restarted.establish_pnl_session_baseline(
        -1200.0,
        snapshot_at=datetime(2026, 7, 3, 4, 0, tzinfo=timezone.utc),
    )
    assert new_day is True
    assert restarted.pnl_reconciliation_snapshot()[
        "session_opening_realized_baseline"
    ] == pytest.approx(-1200.0)

def test_exit_order_id_cannot_register_entry_bracket(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    manager = BracketManager(order_manager=SimpleNamespace())
    _stop(manager)

    manager.register_virtual_bracket(
        order_id="exit-order",
        symbol=SYMBOL,
        side="SELL",
        qty=65,
        price=100.0,
        sl=110.0,
        tp=80.0,
        tag="exit",
        intent="EXIT",
    )

    assert manager.get_bracket("exit-order") is None


def test_exit_tag_text_is_not_authoritative_for_entry_bracket(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    manager = BracketManager(order_manager=SimpleNamespace())
    _stop(manager)

    manager.register_virtual_bracket(
        order_id="entry-tagged-exit",
        symbol=SYMBOL,
        side="BUY",
        qty=65,
        price=100.0,
        sl=90.0,
        tp=120.0,
        tag="strategy-text-mentions-exit",
        intent="ENTRY",
    )
    manager.confirm_entry_fill("entry-tagged-exit", 100.5)

    bracket = manager.get_bracket("entry-tagged-exit")
    assert bracket is not None
    assert bracket.entry_order_intent == "ENTRY"
    assert bracket.active is True


def _stop(manager: BracketManager) -> None:
    manager._running = False
    manager._watchdog_thread.join(timeout=1.0)


def test_direct_long_bracket_registration_normalizes_and_triggers_sl(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    manager = BracketManager(order_manager=SimpleNamespace())
    _stop(manager)
    manager.register_virtual_bracket(
        order_id="entry-1",
        symbol=SYMBOL,
        side="LONG",
        qty=65,
        price=100.0,
        sl=90.0,
        tp=120.0,
        activate_immediately=True,
    )
    bracket = manager.get_bracket("entry-1")
    assert bracket is not None
    assert bracket.side == "BUY"
    action = manager._evaluate_exit_fast(bracket, 89.0)
    assert action is not None
    assert action["type"] == "SL"


def test_bracket_state_is_written_and_restored_with_ledger_recovery_fields(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    manager = BracketManager(order_manager=SimpleNamespace())
    _stop(manager)
    manager.register_virtual_bracket(
        order_id="entry-1",
        symbol=SYMBOL,
        side="BUY",
        qty=65,
        price=100.0,
        sl=90.0,
        tp=120.0,
        activate_immediately=True,
    )
    bracket = manager.get_bracket("entry-1")
    assert bracket is not None
    bracket._ledger_pending_exit_order_id = "exit-1"
    bracket._ledger_pending_exit_quantity = 65
    bracket._ledger_pending_exit_price = 89.5
    manager.save_state()

    restored = BracketManager(order_manager=SimpleNamespace())
    _stop(restored)
    restored_bracket = restored.get_bracket("entry-1")
    assert restored_bracket is not None
    assert restored_bracket._ledger_pending_exit_order_id == "exit-1"
    assert restored_bracket._ledger_pending_exit_quantity == 65
    assert restored_bracket._ledger_pending_exit_price == pytest.approx(89.5)


def test_confirmed_fill_remains_active_when_snapshot_persistence_fails(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    manager = BracketManager(order_manager=SimpleNamespace())
    _stop(manager)
    manager.register_virtual_bracket(
        order_id="entry-1",
        symbol=SYMBOL,
        side="BUY",
        qty=65,
        price=100.0,
        sl=90.0,
        tp=120.0,
    )
    monkeypatch.setattr(manager, "save_state", lambda: (_ for _ in ()).throw(OSError("disk")))
    manager.confirm_entry_fill("entry-1", 101.0)
    bracket = manager.get_bracket("entry-1")
    assert bracket is not None
    assert bracket.active is True
    assert bracket.entry_confirmed is True


def test_metrics_failure_does_not_undo_registered_protection(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    manager = BracketManager(order_manager=SimpleNamespace())
    _stop(manager)

    class _Counter:
        def inc(self) -> None:
            raise RuntimeError("metrics down")

    monkeypatch.setattr(bracket_core, "METRICS_AVAILABLE", True)
    monkeypatch.setattr(bracket_core, "METRICS", SimpleNamespace(brackets_created=_Counter()))
    manager.register_virtual_bracket(
        order_id="entry-1",
        symbol=SYMBOL,
        side="BUY",
        qty=65,
        price=100.0,
        sl=90.0,
        tp=120.0,
        activate_immediately=True,
    )
    assert manager.get_bracket("entry-1") is not None


def test_margin_inputs_are_immutable() -> None:
    inputs = MarginInputs(
        symbol=SYMBOL,
        side="BUY",
        price=100.0,
        stop_loss=90.0,
        atr=5.0,
        requested_qty=65,
        product="MIS",
        lot_size=65,
        balance=100000.0,
        per_trade_risk_pct=1.0,
        per_trade_cap_pct=10.0,
        margin_factor=1.0,
        margin_buffer=0.95,
        contract_multiplier=1.0,
        ist_now=datetime.now(timezone.utc),
        min_lots_per_trade=1,
        max_lots_per_trade=2,
        atr_multiple=1.5,
    )
    with pytest.raises(FrozenInstanceError):
        inputs.balance = 1.0


class _Hub:
    def __init__(self, quote):
        self.quote = quote

    def get_quote(self, symbol: str, allow_pull: bool = True):
        return dict(self.quote)


def test_zero_spread_limit_is_strict_and_none_explicitly_disables_guard() -> None:
    quote = {"best_bid": 100.0, "best_ask": 101.0}
    with pytest.raises(OrderPlacementError):
        ExecutionPolicy(_Hub(quote), max_spread_pct=0.0).build_plan(SYMBOL, "BUY")
    plan = ExecutionPolicy(_Hub(quote), max_spread_pct=None).build_plan(SYMBOL, "BUY")
    assert plan.spread_pct > 0


def test_invalid_later_row_does_not_partially_mutate_existing_position(tmp_path) -> None:
    manager = _position_manager(tmp_path)
    original = manager.get_position(SYMBOL)
    assert original is not None
    original_price = original.current_price
    with pytest.raises(ValueError):
        manager.synchronize_with_broker(
            [
                {
                    "symbol": SYMBOL,
                    "product": "MIS",
                    "quantity": 65,
                    "average_price": 100.0,
                    "last_price": 150.0,
                },
                {
                    "symbol": "NFO:NIFTY2662324000CE",
                    "product": "MIS",
                    "quantity": "invalid",
                    "average_price": 100.0,
                    "last_price": 101.0,
                },
            ]
        )
    preserved = manager.get_position(SYMBOL)
    assert preserved is not None
    assert preserved.current_price == pytest.approx(original_price)


def _zerodha_positions_client(response):
    client = object.__new__(ZerodhaKiteClient)
    client._GENERAL_BUCKET = "general"
    client._positions_cache = None
    client._log_time_fn = lambda: 0.0
    client._acquire_bucket = lambda *_args, **_kwargs: None
    client._make_request = lambda *_args, **_kwargs: response
    client._ensure_json = lambda payload: payload
    client._build_retry_handlers = lambda **_kwargs: (lambda *_args, **_kw: False, None)
    client._execute_with_retry = lambda **kwargs: kwargs["operation"]()
    client._load_rest_cache = lambda *_args, **_kwargs: None
    return client


def test_zerodha_missing_net_snapshot_raises_instead_of_returning_flat() -> None:
    client = _zerodha_positions_client({"status": "success", "data": {}})
    with pytest.raises(BrokerError):
        client.get_positions()


def test_zerodha_authoritative_empty_net_does_not_fall_back_to_day_rows() -> None:
    client = _zerodha_positions_client(
        {
            "status": "success",
            "data": {
                "net": [],
                "day": [{"symbol": SYMBOL, "quantity": 65}],
            },
        }
    )
    assert client.get_positions() == []


def test_flat_verification_rejects_missing_and_malformed_snapshots() -> None:
    for response in (None, [None], [{"symbol": SYMBOL}], [{"quantity": 0}]):
        broker = SimpleNamespace(get_positions=lambda response=response: response)
        manager = BracketManager(order_manager=SimpleNamespace(_broker=broker))
        _stop(manager)
        assert manager._verify_position_closed(SYMBOL) is False


def test_flat_verification_accepts_only_valid_explicit_flat_snapshot() -> None:
    broker = SimpleNamespace(
        get_positions=lambda: [
            {"symbol": SYMBOL, "quantity": 0},
            {"symbol": "NFO:NIFTY2662324000CE", "quantity": 65},
        ]
    )
    manager = BracketManager(order_manager=SimpleNamespace(_broker=broker))
    _stop(manager)
    assert manager._verify_position_closed(SYMBOL) is True

    broker.get_positions = lambda: [{"symbol": SYMBOL, "quantity": 65}]
    assert manager._verify_position_closed(SYMBOL) is False


def test_duplicate_managed_position_rows_reject_snapshot_atomically(tmp_path) -> None:
    manager = _position_manager(tmp_path)
    original = manager.get_position(SYMBOL)
    assert original is not None
    original_price = original.current_price
    duplicate = {
        "symbol": SYMBOL,
        "product": "MIS",
        "quantity": 65,
        "average_price": 100.0,
        "last_price": 150.0,
    }
    with pytest.raises(ValueError, match="duplicate broker position"):
        manager.synchronize_with_broker([duplicate, dict(duplicate)])
    preserved = manager.get_position(SYMBOL)
    assert preserved is not None
    assert preserved.current_price == pytest.approx(original_price)
