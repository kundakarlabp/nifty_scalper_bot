from __future__ import annotations

import threading
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.data.rest.zerodha_client import ZerodhaKiteClient
from nifty_scalper_bot.execution import BracketManager, bracket_core
from nifty_scalper_bot.execution.execution_policy import ExecutionPolicy
from nifty_scalper_bot.execution.margin_engine import MarginInputs
from nifty_scalper_bot.execution.position_manager import (
    Order,
    Position,
    PositionManager,
    normalize_broker_order_status,
)
from nifty_scalper_bot.utils.errors import BrokerError, OrderPlacementError

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


def test_malformed_broker_snapshot_fails_closed_and_preserves_position(
    tmp_path,
) -> None:
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


def test_broker_realised_field_updates_daily_realised_without_using_total_pnl(
    tmp_path,
) -> None:
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


def test_update_from_order_uses_fill_price_and_existing_fill_lifecycle(
    tmp_path,
) -> None:
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
    assert (
        manager.current_entry_protection_blocker(SYMBOL)
        == "entry_protection_incomplete"
    )
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
    manager.add_pending_order(
        "partial-entry", SYMBOL, "BUY", 65, 100.0, "MARKET", intent="ENTRY"
    )

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


def test_simultaneous_duplicate_complete_update_is_single_lifecycle_mutation(
    tmp_path,
) -> None:
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
    manager.add_pending_order(
        "avg-entry", SYMBOL, "BUY", 65, 100.0, "MARKET", intent="ENTRY"
    )

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


def test_new_entry_same_symbol_after_previous_trade_uses_distinct_lifecycle(
    tmp_path,
) -> None:
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    manager.add_pending_order(
        "entry-a", SYMBOL, "BUY", 65, 100.0, "MARKET", intent="ENTRY"
    )
    manager.apply_broker_order_update(
        "entry-a",
        {"status": "COMPLETE", "filled_quantity": 65, "average_price": 100.0},
    )
    manager.add_pending_order(
        "exit-a", SYMBOL, "SELL", 65, 101.0, "MARKET", intent="EXIT"
    )
    manager.apply_broker_order_update(
        "exit-a",
        {"status": "COMPLETE", "filled_quantity": 65, "average_price": 101.0},
    )
    assert manager.get_position(SYMBOL) is None

    manager.add_pending_order(
        "entry-b", SYMBOL, "BUY", 65, 102.0, "MARKET", intent="ENTRY"
    )
    manager.apply_broker_order_update(
        "entry-b",
        {"status": "COMPLETE", "filled_quantity": 65, "average_price": 102.0},
    )

    position = manager.get_position(SYMBOL)
    assert position is not None
    assert position.order_id == "entry-b"
    assert position.quantity == 65
    assert manager._orders.get("entry-b") is not None
    assert (
        manager._terminal_orders["entry-a"].trade_lifecycle_id
        != manager._terminal_orders["entry-b"].trade_lifecycle_id
    )


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


def test_duplicate_and_invalid_cumulative_updates_do_not_mutate_position(
    tmp_path,
) -> None:
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    manager.add_pending_order(
        "bad-partial", SYMBOL, "BUY", 65, 100.0, "MARKET", intent="ENTRY"
    )
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
    manager.add_pending_order(
        "unresolved", SYMBOL, "SELL", 65, 99.0, "MARKET", intent="EXIT"
    )
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


def test_entry_bracket_id_alone_does_not_resolve_without_protection_ack(
    tmp_path,
) -> None:
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
    assert (
        manager.current_entry_protection_blocker(SYMBOL)
        == "entry_protection_incomplete"
    )

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
    restarted._local_realized_pnl = -4000.0
    with restarted._lock:
        restarted._refresh_realized_pnl_locked()

    new_day = restarted.establish_pnl_session_baseline(
        -1200.0,
        snapshot_at=datetime(2026, 7, 3, 4, 0, tzinfo=timezone.utc),
    )
    assert new_day is True
    assert restarted.pnl_reconciliation_snapshot()[
        "session_opening_realized_baseline"
    ] == pytest.approx(-1200.0)
    assert restarted.pnl_reconciliation_snapshot()[
        "local_confirmed_realized"
    ] == pytest.approx(0.0)
    assert restarted.get_realized_pnl() == pytest.approx(0.0)


def test_manual_daily_pnl_reset_clears_local_ledger_and_preserves_zero_baseline(
    tmp_path,
) -> None:
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    manager._broker_realized_pnl = -1000.0
    manager._local_realized_pnl = -4000.0
    with manager._lock:
        manager._refresh_realized_pnl_locked()

    manager.reset_daily_pnl()

    snapshot = manager.pnl_reconciliation_snapshot()
    assert snapshot["local_confirmed_realized"] == pytest.approx(0.0)
    assert snapshot["broker_session_realized"] == pytest.approx(0.0)
    assert manager.get_realized_pnl() == pytest.approx(0.0)


def test_late_same_day_broker_baseline_preserves_local_fills(tmp_path) -> None:
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    trading_date = "2026-07-03"
    manager._pnl_trading_date = trading_date
    manager._local_realized_pnl = -50.0

    manager.establish_pnl_session_baseline(
        -1000.0,
        trading_date=trading_date,
    )

    assert manager.get_realized_pnl() == pytest.approx(-50.0)


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


def test_exit_tag_text_is_not_authoritative_for_entry_bracket(
    tmp_path, monkeypatch
) -> None:
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


def test_direct_long_bracket_registration_normalizes_and_triggers_sl(
    tmp_path, monkeypatch
) -> None:
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


def test_bracket_state_is_written_and_restored_with_ledger_recovery_fields(
    tmp_path, monkeypatch
) -> None:
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

    monkeypatch.setenv("BRACKET_AUTO_RESTORE", "true")
    restored = BracketManager(order_manager=SimpleNamespace())
    _stop(restored)
    restored_bracket = restored.get_bracket("entry-1")
    assert restored_bracket is not None
    assert restored_bracket._ledger_pending_exit_order_id == "exit-1"
    assert restored_bracket._ledger_pending_exit_quantity == 65
    assert restored_bracket._ledger_pending_exit_price == pytest.approx(89.5)


def test_confirmed_fill_remains_active_when_snapshot_persistence_fails(
    tmp_path, monkeypatch
) -> None:
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
    monkeypatch.setattr(
        manager, "save_state", lambda: (_ for _ in ()).throw(OSError("disk"))
    )
    manager.confirm_entry_fill("entry-1", 101.0)
    bracket = manager.get_bracket("entry-1")
    assert bracket is not None
    assert bracket.active is True
    assert bracket.entry_confirmed is True


def test_metrics_failure_does_not_undo_registered_protection(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    manager = BracketManager(order_manager=SimpleNamespace())
    _stop(manager)

    class _Counter:
        def inc(self) -> None:
            raise RuntimeError("metrics down")

    monkeypatch.setattr(bracket_core, "METRICS_AVAILABLE", True)
    monkeypatch.setattr(
        bracket_core, "METRICS", SimpleNamespace(brackets_created=_Counter())
    )
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


def test_invalid_later_row_does_not_partially_mutate_existing_position(
    tmp_path,
) -> None:
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


def test_incident_single_lot_lifecycle_no_false_exit_no_second_entry(
    monkeypatch, tmp_path
) -> None:
    """Regression for the one-lot incident (NFO:NIFTY2670724050CE, qty 65).

    Invariants: exactly one bracket per entry, a replayed pre-fill tick never
    fires an immediate false exit right after activation, a fresh tick still
    evaluates the SL, a second entry on another strike is rejected atomically
    at the order-submission choke point, and exposure never exceeds 65.
    """
    from nifty_scalper_bot.execution.order_manager import OrderManager, OrderType

    ce = "NFO:NIFTY2670724050CE"
    pe = "NFO:NIFTY2670724050PE"

    class _IncidentBroker:
        def place_order(self, **_kwargs):
            return {"order_id": "E1"}

        def get_orders(self):
            return []

        def get_positions(self):
            return []

    class _IncidentPositions:
        def has_open_position(self, _symbol):
            return False

        def get_open_positions(self):
            return []

    om = OrderManager(_IncidentBroker(), _IncidentPositions(), object())
    monkeypatch.setattr(om, "_lot_size_for_symbol", lambda _s: 65)
    bm = BracketManager(order_manager=om)
    om.set_bracket_manager(bm)
    try:
        entry_id = om.place_order(
            symbol=ce,
            side="BUY",
            quantity=65,
            order_type=OrderType.LIMIT,
            price=145.15,
            stop_loss=144.50,
            take_profit=152.00,
            intent="ENTRY",
            check_risk=False,
            signal_id="incident-ce-1",
        )
        assert entry_id, "incident entry must be accepted"

        # Exactly one bracket, qty 65 — no duplicate registration.
        ce_brackets = [b for b in bm._brackets.values() if b.symbol == ce]
        assert len(ce_brackets) == 1
        bracket = ce_brackets[0]
        assert bracket.quantity == 65

        # Reservation released once the local order record owns the symbol.
        assert ce not in om._entries_in_flight

        bm.confirm_entry_fill(entry_id, 145.15)
        assert bracket.active and bracket.entry_confirmed
        assert bracket.entry_fill_ts is not None

        # Replayed pre-fill tick (exchange ts before fill) at 144.05 — below
        # SL 144.50 — must NOT fire a false immediate exit.
        bm.on_tick(ce, 144.05, exchange_ts=bracket.entry_fill_ts - 2.0)
        assert bracket.exit_pending is False
        assert bracket.exit_executed is False
        assert bracket.remaining_quantity == 65

        # A genuine (fresh) tick at the same price still evaluates the SL.
        action = bm._evaluate_exit_fast(bracket, 144.05)
        assert action is not None and action.get("type") == "SL"

        # Second entry on the other strike is rejected at the choke point.
        pe_id = om.place_order(
            symbol=pe,
            side="BUY",
            quantity=65,
            order_type=OrderType.LIMIT,
            price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            intent="ENTRY",
            check_risk=False,
            signal_id="incident-pe-1",
        )
        assert pe_id is None, "single-position gate must reject the second strike"
        assert not [b for b in bm._brackets.values() if b.symbol == pe]

        # Exposure never exceeded one lot across all brackets.
        assert sum(b.quantity for b in bm._brackets.values()) == 65
    finally:
        bm._running = False


def test_bracket_lifecycle_trailing_and_exits_on_live_class(
    monkeypatch, tmp_path
) -> None:
    """End-to-end lifecycle on the LIVE production class (full MRO), exercising
    the fallback trailing path (_apply_trailing_math): SL must ratchet up on a
    rally, stay monotonic on pullback, stay tick-rounded, fire the trailed-SL
    exit exactly once, and TP1 must fire a single partial exit."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    placed: list = []

    class _OM:
        def place_reduce_only_exit(self, intent):
            placed.append(intent)
            return "X1"

        def place_order(self, **kwargs):
            placed.append(kwargs)
            return "X1"

        def get_order_status(self, _oid):
            return {"status": "COMPLETE", "average_price": 151.0}

    sym = "NFO:NIFTYLIFECYCLE24050CE"
    bm = BracketManager(order_manager=_OM())
    class _Positions:
        def __init__(self):
            self.orders = {}
        def add_pending_order(self, order_id, symbol, side, qty, price, order_type, **kwargs):
            self.orders[str(order_id)] = dict(symbol=symbol, side=side, qty=qty, **kwargs)
        def bind_pending_order_id(self, provisional_order_id, final_order_id):
            self.orders[str(final_order_id)] = self.orders.pop(str(provisional_order_id))
        def remove_pending_order(self, order_id):
            self.orders.pop(str(order_id), None)
        def is_exit_converging(self, _symbol):
            return False
    bm.order_manager._positions = _Positions()
    bm._running = False
    try:
        bm.register_virtual_bracket(
            order_id="lc-1",
            symbol=sym,
            side="BUY",
            qty=65,
            price=145.15,
            sl=143.0,
            tp=152.0,
            activate_immediately=False,
        )
        bm.confirm_entry_fill("lc-1", 145.15)
        bracket = bm.get_bracket("lc-1")
        sl0 = bracket.sl_trigger_price

        for px in (146.0, 147.5, 149.0, 150.5):  # rally -> ratchet
            bm.on_tick(sym, px)
        trailed = bracket.sl_trigger_price
        assert trailed > sl0, "trailing must raise SL on a rally"
        assert round(trailed / 0.05, 6) % 1 == 0, "SL must stay tick-rounded"

        bm.on_tick(sym, 147.0)  # pullback below trailed SL
        assert bracket.sl_trigger_price >= trailed, "SL must never move down"
        assert bracket.exit_pending is True
        assert "SL" in str(bracket.exit_reason or "").upper()
        assert len(placed) == 1, "trailed-SL exit must fire exactly once"

        bm.on_tick(sym, 152.4)  # further ticks must not double-fire
        assert len(placed) == 1
    finally:
        bm._running = False


def test_bracket_tp1_partial_fires_once_on_live_class(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    placed: list = []

    class _OM:
        def place_reduce_only_exit(self, intent):
            placed.append(intent)
            return "X2"

        def place_order(self, **kwargs):
            placed.append(kwargs)
            return "X2"

        def get_order_status(self, _oid):
            return {"status": "COMPLETE", "average_price": 149.1}

    sym = "NFO:NIFTYLIFECYCLE24050PE"
    bm = BracketManager(order_manager=_OM())
    bm._running = False
    try:
        bm.register_virtual_bracket(
            order_id="lc-2",
            symbol=sym,
            side="BUY",
            qty=65,
            price=145.15,
            sl=143.0,
            tp=152.0,
            tp1_price=149.0,
            tp1_qty=25,
            activate_immediately=False,
        )
        bm.confirm_entry_fill("lc-2", 145.15)
        bracket = bm.get_bracket("lc-2")
        assert bracket.tp_levels == []

        bm.on_tick(sym, 149.1)
        assert bracket.exit_pending is False
        assert len(placed) == 0, "one-lot brackets must not fire sub-lot TP1 exits"
    finally:
        bm._running = False


def test_fill_reanchor_and_controller_trailing_on_executed_range(
    monkeypatch, tmp_path
) -> None:
    """Signal->executed coordination on the live class: a slipped broker fill
    must re-anchor entry/SL/TP1 to the executed range (tick-rounded), sync the
    adaptive controller's anchor, and controller-path trailing must ratchet on
    a rally even with degraded/unavailable ATR (the old absolute 20.0-point
    fallback trail distance left trailing silently dead on option premiums)."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    class _OM:
        def place_reduce_only_exit(self, intent):
            return "X"

        def place_order(self, **kwargs):
            return "X"

        def get_order_status(self, _oid):
            return {"status": "COMPLETE"}

    class _IndicatorEngine:  # bare: SafeATRProvider yields degraded ATR
        pass

    sym = "NFO:NIFTYEXECRANGE24050CE"
    bm = BracketManager(order_manager=_OM(), indicator_engine=_IndicatorEngine())
    bm._running = False
    try:
        bm.register_virtual_bracket(
            order_id="ex-1",
            symbol=sym,
            side="BUY",
            qty=65,
            price=145.15,
            sl=143.15,
            tp=160.0,
            activate_immediately=False,
            trailing_atr_mult=1.5,
        )
        controller = bm._trailing_controllers.get("ex-1")
        assert controller is not None, "adaptive controller must attach"

        bm.confirm_entry_fill("ex-1", 146.00)  # +0.85 slippage vs signal
        bracket = bm.get_bracket("ex-1")

        # Re-anchored to broker fill, tick-rounded.
        assert bracket.entry_price == 146.00
        assert bracket.entry_fill_price == 146.00
        assert bracket.sl_trigger_price > 143.15
        assert round(bracket.sl_trigger_price / 0.05, 6) % 1 == 0
        # Controller anchor synced to executed range.
        assert float(controller.entry_price) == 146.00
        assert float(controller.current_sl) == bracket.sl_trigger_price

        # Controller-path trailing ratchets on a rally despite degraded ATR.
        sl_path = []
        for px in (147.0, 149.0, 151.0, 153.0, 151.5, 155.0):
            bm.on_tick(sym, px)
            sl_path.append(bracket.sl_trigger_price)
        assert sl_path[-1] > 146.00, "trailing must lock profit above entry"
        assert all(b >= a for a, b in zip(sl_path, sl_path[1:]))
        assert all(round(v / 0.05, 6) % 1 == 0 for v in sl_path)
    finally:
        bm._running = False


def test_live_entry_registers_pending_order_with_position_manager(
    monkeypatch, tmp_path
) -> None:
    """2026-07-10 incident: place_order registered the order locally and with
    the bracket manager but NOT with the PositionManager, so the broker fill
    for the bot's OWN live entry arrived as an unknown order (intent=UNKNOWN),
    was quarantined every sync, blocked all new entries, and left the position
    on a wide guard bracket. The pending-order sync must happen at submit."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from nifty_scalper_bot.execution.order_manager import OrderManager, OrderType

    class _Broker:
        def place_order(self, **_k):
            return {"order_id": "LIVE-ENTRY-1"}

        def get_orders(self):
            return []

        def get_positions(self):
            return []

    class _Positions:
        def __init__(self):
            self.pending: dict = {}

        def has_open_position(self, _s):
            return False

        def get_open_positions(self):
            return []

        def add_pending_order(self, **kwargs):
            self.pending[kwargs["order_id"]] = kwargs

    om = OrderManager(_Broker(), _Positions(), object())
    monkeypatch.setattr(om, "_lot_size_for_symbol", lambda _s: 65)
    bm = BracketManager(order_manager=om)
    bm._running = False
    om.set_bracket_manager(bm)
    try:
        oid = om.place_order(
            symbol="NFO:NIFTY2671424100CE",
            side="BUY",
            quantity=65,
            order_type=OrderType.LIMIT,
            price=160.45,
            stop_loss=151.20,
            take_profit=177.10,
            intent="ENTRY",
            check_risk=False,
            signal_id="sig-0710",
        )
        assert oid
        pending = om._positions.pending.get(oid)
        assert pending is not None, "pending order must reach PositionManager"
        assert str(pending["intent"]).upper() == "ENTRY"
        assert pending["qty"] == 65
        # The freshly registered bracket is pending entry — the ghost sweeper's
        # skip condition (entry_confirmed False) must hold for it.
        bracket = bm.get_bracket(oid)
        assert bracket is not None and bracket.entry_confirmed is False
    finally:
        bm._running = False


def test_system_exit_does_not_poll_for_fill_on_protection_path(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from nifty_scalper_bot.execution.order_manager import OrderManager, OrderType

    class _Broker:
        def place_order(self, **_kwargs):
            return {"order_id": "SYSTEM-EXIT-1"}

        def get_orders(self):
            return []

        def get_positions(self):
            return []

    class _Positions:
        def add_pending_order(self, **_kwargs):
            return None

        def get_position(self, _symbol):
            return SimpleNamespace(quantity=65, side="LONG")

    manager = OrderManager(_Broker(), _Positions(), object())
    monkeypatch.setattr(
        manager,
        "_confirm_fill_fast",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("system exit must not synchronously poll")
        ),
    )

    order_id = manager.place_order(
        symbol="NFO:NIFTY2671424100CE",
        side="SELL",
        quantity=65,
        order_type=OrderType.MARKET,
        intent="EXIT",
        tag="stop_exit",
        check_risk=False,
    )

    assert order_id == "SYSTEM-EXIT-1"


def test_non_incremental_fill_warning_dedupes_per_snapshot(tmp_path) -> None:
    """2026-07-10 RCA: the same terminal fill replayed by every reconcile
    cycle flooded logs/Telegram with identical warnings. The idempotency drop
    stays; the warning fires once per (order, cumulative snapshot)."""
    import logging

    from nifty_scalper_bot.execution.position_manager import PositionManager

    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    manager.add_pending_order(
        "OID-1", "NFO:NIFTY2671424100CE", "BUY", 65, 160.45, "LIMIT", intent="ENTRY"
    )
    manager.apply_broker_order_update(
        "OID-1", {"status": "COMPLETE", "filled_quantity": 65, "average_price": 160.45}
    )

    records: list = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Capture()
    manager._logger.addHandler(handler)
    try:
        for _ in range(5):  # replayed terminal fill, five reconcile cycles
            manager.apply_broker_order_update(
                "OID-1",
                {"status": "COMPLETE", "filled_quantity": 65, "average_price": 160.45},
            )
    finally:
        manager._logger.removeHandler(handler)
    warnings = [m for m in records if "non-incremental" in m]
    assert len(warnings) == 1, warnings
    assert manager.get_position("NFO:NIFTY2671424100CE").quantity == 65


def test_order_update_side_effect_failure_keeps_fill_delta_replayable(
    monkeypatch, tmp_path
):
    from nifty_scalper_bot.execution.order_manager import (
        OrderDetails,
        OrderManager,
        OrderStatus,
        OrderType,
    )

    class Broker:
        is_simulated_adapter = True

    class Positions:
        def __init__(self):
            self.calls = 0
            self.fail = True

        def add_pending_order(self, **_kwargs):
            return None

        def update_order_status(self, *_args, **_kwargs):
            return None

        def apply_broker_order_update(self, *_args, **_kwargs):
            self.calls += 1
            if self.fail:
                raise RuntimeError("position side effect failed")

        def get_open_positions(self):
            return []

    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    positions = Positions()
    manager = OrderManager(Broker(), positions, object())
    order = OrderDetails(
        order_id="OID-FAIL",
        symbol="NFO:NIFTY2671423950CE",
        side="BUY",
        quantity=130,
        order_type=OrderType.LIMIT,
        status=OrderStatus.SUBMITTED,
        price=100.0,
        fill_price=100.0,
        filled_quantity=65,
    )
    order.applied_filled_quantity = 65
    manager._orders[order.order_id] = order
    monkeypatch.setattr(
        manager, "_register_virtual_bracket_for_fill", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        manager, "_confirm_position_protection_for_fill", lambda *_a, **_k: None
    )

    with pytest.raises(RuntimeError, match="position side effect failed"):
        manager.apply_broker_order_update(
            "OID-FAIL",
            {
                "status": "PARTIALLY FILLED",
                "filled_quantity": 130,
                "average_price": 100.0,
            },
        )

    assert order.filled_quantity == 130
    assert order.applied_filled_quantity == 65

    positions.fail = False
    manager.apply_broker_order_update(
        "OID-FAIL",
        {"status": "PARTIALLY FILLED", "filled_quantity": 130, "average_price": 100.0},
    )

    assert positions.calls == 2
    assert order.applied_filled_quantity == 130


def test_bracket_registration_is_idempotent_when_fill_replayed_after_position_failure(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace

    from nifty_scalper_bot.execution.order_manager import (
        OrderDetails,
        OrderManager,
        OrderStatus,
        OrderType,
    )

    class Broker:
        is_simulated_adapter = True

    class Positions:
        def __init__(self):
            self.calls = 0
            self.fail = True

        def add_pending_order(self, **_kwargs):
            return None

        def update_order_status(self, *_args, **_kwargs):
            return None

        def apply_broker_order_update(self, *_args, **_kwargs):
            self.calls += 1
            if self.fail:
                raise RuntimeError("position side effect failed")

        def get_open_positions(self):
            return []

        def confirm_entry_protection(self, *_args):
            return None

    class RecordingBracketManager:
        def __init__(self):
            self.bracket = None
            self.register_calls = 0
            self.confirm_calls = 0

        def get_bracket(self, order_id):
            return (
                self.bracket
                if self.bracket and self.bracket.order_id == order_id
                else None
            )

        def has_active_bracket(self, _symbol):
            return self.bracket is not None

        def register_virtual_bracket(self, **kwargs):
            self.register_calls += 1
            self.bracket = SimpleNamespace(
                order_id=kwargs["order_id"],
                bracket_id=kwargs["order_id"],
                quantity=kwargs["qty"],
                protected_quantity=kwargs["qty"],
                active=False,
                entry_confirmed=False,
                sl_trigger_price=kwargs["sl"],
            )

        def confirm_entry_fill(self, order_id, _entry_price):
            assert self.bracket is not None and self.bracket.order_id == order_id
            self.confirm_calls += 1
            self.bracket.quantity = 130
            self.bracket.protected_quantity = 130
            self.bracket.active = True
            self.bracket.entry_confirmed = True

    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    positions = Positions()
    manager = OrderManager(Broker(), positions, object())
    brackets = RecordingBracketManager()
    manager.set_bracket_manager(brackets)
    order = OrderDetails(
        order_id="OID-BRACKET",
        symbol="NFO:NIFTY2671423950CE",
        side="BUY",
        quantity=130,
        order_type=OrderType.LIMIT,
        status=OrderStatus.SUBMITTED,
        price=100.0,
        fill_price=100.0,
        filled_quantity=65,
        stop_loss=95.0,
        take_profit=110.0,
        intent="ENTRY",
    )
    order.applied_filled_quantity = 65
    manager._orders[order.order_id] = order

    with pytest.raises(RuntimeError, match="position side effect failed"):
        manager.apply_broker_order_update(
            "OID-BRACKET",
            {
                "status": "PARTIALLY FILLED",
                "filled_quantity": 130,
                "average_price": 100.0,
            },
        )

    assert brackets.register_calls == 1
    assert order.applied_filled_quantity == 65

    positions.fail = False
    manager.apply_broker_order_update(
        "OID-BRACKET",
        {"status": "PARTIALLY FILLED", "filled_quantity": 130, "average_price": 100.0},
    )

    assert brackets.register_calls == 1
    assert brackets.bracket.protected_quantity == 130
    assert positions.calls == 2
    assert order.applied_filled_quantity == 130


def test_order_details_applied_fill_persists_across_restart_and_replay(
    monkeypatch, tmp_path
):
    from nifty_scalper_bot.execution.order_manager import (
        OrderDetails,
        OrderManager,
        OrderStatus,
        OrderType,
    )

    class Broker:
        is_simulated_adapter = True

    class Positions:
        def __init__(self):
            self.calls = 0

        def add_pending_order(self, **_kwargs):
            return None

        def update_order_status(self, *_args, **_kwargs):
            return None

        def apply_broker_order_update(self, *_args, **_kwargs):
            self.calls += 1

        def get_open_positions(self):
            return []

    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    first = OrderManager(Broker(), Positions(), object())
    order = OrderDetails(
        order_id="OID-PERSIST",
        symbol="NFO:NIFTY2671423950CE",
        side="BUY",
        quantity=130,
        order_type=OrderType.LIMIT,
        status=OrderStatus.PARTIALLY_FILLED,
        price=100.0,
        fill_price=100.0,
        filled_quantity=130,
        applied_filled_quantity=65,
        stop_loss=95.0,
        take_profit=110.0,
        intent="ENTRY",
    )
    first._orders[order.order_id] = order
    first.save_orders()

    positions = Positions()
    restarted = OrderManager(Broker(), positions, object())
    restored = restarted._orders["OID-PERSIST"]
    assert restored.filled_quantity == 130
    assert restored.applied_filled_quantity == 65
    monkeypatch.setattr(
        restarted, "_register_virtual_bracket_for_fill", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        restarted, "_confirm_position_protection_for_fill", lambda *_a, **_k: None
    )

    restarted.apply_broker_order_update(
        "OID-PERSIST",
        {"status": "PARTIALLY FILLED", "filled_quantity": 130, "average_price": 100.0},
    )

    assert positions.calls == 1
    assert restored.applied_filled_quantity == 130
