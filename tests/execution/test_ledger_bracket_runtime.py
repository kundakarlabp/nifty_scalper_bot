from __future__ import annotations

import time
from typing import Any

from nifty_scalper_bot.execution.bracket_manager import BracketExitLifecycle
from nifty_scalper_bot.execution.ledger_bracket_manager import LedgerBracketManager

SYMBOL = "NFO:NIFTY2662324050PE"


class _Broker:
    def __init__(self) -> None:
        self.statuses: dict[str, dict[str, Any]] = {}
        self.positions: list[dict[str, Any]] = [{"symbol": SYMBOL, "quantity": 65}]

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        return dict(self.statuses.get(order_id, {"status": ""}))

    def get_positions(self) -> list[dict[str, Any]]:
        return list(self.positions)

    def cancel_order(self, order_id: str, *args: Any, **kwargs: Any) -> bool:
        payload = dict(self.statuses.get(order_id, {}))
        payload["status"] = "CANCELLED"
        self.statuses[order_id] = payload
        return True


class _OrderManager:
    def __init__(self, broker: _Broker) -> None:
        self._broker = broker
        self._last_order_decision: dict[str, Any] = {}
        self.place_calls: list[dict[str, Any]] = []

    def place_order(self, **kwargs: Any) -> str:
        self.place_calls.append(dict(kwargs))
        order_id = f"exit-{len(self.place_calls)}"
        self._broker.statuses[order_id] = {"status": "OPEN"}
        return order_id

    def cancel_order(self, order_id: str, *args: Any, **kwargs: Any) -> bool:
        return self._broker.cancel_order(order_id, *args, **kwargs)

    def set_last_skip_reason(self, _reason: str) -> None:
        return None


def _manager(
    monkeypatch, tmp_path
) -> tuple[LedgerBracketManager, _OrderManager, _Broker]:
    monkeypatch.setenv("BRACKET_FILL_LEDGER_PATH", str(tmp_path / "lifecycle.db"))
    broker = _Broker()
    order_manager = _OrderManager(broker)
    manager = LedgerBracketManager(order_manager=order_manager)
    manager._running = False
    manager._watchdog_thread.join(timeout=1.0)
    manager._filled_position_sync_grace_seconds = 0.0
    manager._exit_reconcile_interval_seconds = 0.0
    manager.register_virtual_bracket(
        order_id="entry-1",
        symbol=SYMBOL,
        side="BUY",
        qty=130,
        price=100.0,
        sl=90.0,
        tp=120.0,
        tp1_price=110.0,
        tp1_qty=65,
        activate_immediately=False,
    )
    return manager, order_manager, broker


def _mark_filled_exit(
    manager: LedgerBracketManager,
    broker: _Broker,
    *,
    order_id: str,
    reason: str,
    price: float,
    residual: int,
) -> Any:
    bracket = manager.get_bracket("entry-1")
    assert bracket is not None
    bracket.exit_pending = True
    bracket.exit_reason = reason
    bracket.exit_state = BracketExitLifecycle.EXIT_ORDER_SUBMITTED.value
    bracket.entry_status = bracket.exit_state
    bracket.exit_order_id = order_id
    bracket.pending_exit_order_id = order_id
    bracket.exit_triggered_at = time.time()
    broker.statuses[order_id] = {"status": "COMPLETE", "average_price": price}
    broker.positions = (
        [] if residual == 0 else [{"symbol": SYMBOL, "quantity": residual}]
    )
    return bracket


def test_scaled_fills_persist_and_close_uses_exact_weighted_pnl(
    monkeypatch, tmp_path
) -> None:
    manager, _order_manager, broker = _manager(monkeypatch, tmp_path)
    manager.confirm_entry_fill("entry-1", 100.0)

    bracket = _mark_filled_exit(
        manager,
        broker,
        order_id="tp1-order",
        reason="TP1 Hit (110.00)",
        price=110.0,
        residual=65,
    )
    assert manager._reconcile_exit_state(bracket, requested_by="tp1") is False
    assert bracket.remaining_quantity == 65
    assert bracket.exit_state == BracketExitLifecycle.OPEN_ACTIVE.value

    ordering: list[str] = []
    manager.save_state = lambda: ordering.append("save")  # type: ignore[method-assign]
    manager.attach_on_exit_complete(lambda _symbol: ordering.append("hook"))
    bracket = _mark_filled_exit(
        manager,
        broker,
        order_id="final-order",
        reason="HARD_SL_BREACH",
        price=95.0,
        residual=0,
    )
    assert manager._reconcile_exit_state(bracket, requested_by="final") is True

    assert manager._fill_ledger is not None
    fills = manager._fill_ledger.load_fills(bracket.bracket_id)
    assert [(fill.kind, fill.quantity, fill.price) for fill in fills] == [
        ("ENTRY", 130, 100.0),
        ("EXIT", 65, 110.0),
        ("EXIT", 65, 95.0),
    ]
    pnl = manager._fill_ledger.realized_pnl(bracket.bracket_id)
    assert pnl.gross_pnl == 325.0
    assert pnl.complete is True
    assert ordering == ["save", "hook"]
    assert bracket.ledger_realized_pnl["gross_pnl"] == 325.0
    assert manager.has_unresolved_exit() is False


def test_completed_trade_outcome_preserves_provenance_and_net_costs(
    monkeypatch, tmp_path
) -> None:
    manager, _order_manager, broker = _manager(monkeypatch, tmp_path)
    bracket = manager.get_bracket("entry-1")
    assert bracket is not None
    manager.attach_trade_provenance(
        "entry-1",
        {
            "strategy_name": "VWAPPro",
            "setup_name": "continuation_pullback",
            "regime": "TREND",
            "signal_id": "sig-1",
            "trace_id": "trace-1",
            "strategy_profile_version": "2026-07-30",
        },
    )
    manager.confirm_entry_fill("entry-1", 100.0)
    bracket.highest_ltp = 112.0
    bracket.lowest_ltp = 96.0
    bracket = _mark_filled_exit(
        manager,
        broker,
        order_id="final-outcome",
        reason="HARD_TP_BREACH",
        price=110.0,
        residual=0,
    )
    completed: list[str] = []
    manager.attach_on_exit_complete(completed.append)

    manager._close_bracket(bracket, close_source="broker_fill", exit_price=110.0)

    assert len(completed) == 1
    assert completed[0] == SYMBOL
    outcome = manager.get_completed_trade_outcome(SYMBOL)
    assert outcome is not None
    assert outcome["strategy_name"] == "VWAPPro"
    assert outcome["setup_name"] == "continuation_pullback"
    assert outcome["regime"] == "TREND"
    assert outcome["gross_pnl"] == 1300.0
    assert outcome["estimated_costs"]["total"] > 0
    assert outcome["net_pnl"] < outcome["gross_pnl"]
    assert outcome["mfe_pnl"] == 1560.0
    assert outcome["mae_pnl"] == 520.0
    assert outcome["exit_reason"] == "HARD_TP_BREACH"
    assert outcome["ledger_complete"] is True

    restored = manager._decode_restored_bracket(
        bracket.entry_order_id, bracket.to_dict()
    )
    assert restored.trade_provenance == bracket.trade_provenance


def test_completed_trade_outcome_preserves_execution_loss(
    monkeypatch, tmp_path
) -> None:
    manager, _order_manager, broker = _manager(monkeypatch, tmp_path)
    bracket = manager.get_bracket("entry-1")
    assert bracket is not None
    manager.attach_trade_provenance(
        "entry-1",
        {
            "entry_arrival_price": 99.5,
            "entry_quote_bid": 99.0,
            "entry_quote_ask": 100.0,
            "decision_ts": 1000.0,
            "entry_submit_ts": 1001.0,
        },
    )
    manager.confirm_entry_fill("entry-1", 100.0)
    bracket.entry_fill_ts = 1002.0
    bracket = _mark_filled_exit(
        manager,
        broker,
        order_id="execution-loss-outcome",
        reason="HARD_TP_BREACH",
        price=110.0,
        residual=0,
    )
    bracket.exit_arrival_price = 110.2
    bracket.exit_quote_bid = 110.0
    bracket.exit_quote_ask = 110.4
    bracket.exit_triggered_at = 1008.0
    bracket.exit_submitted_at = 1009.0
    bracket.exit_order_type = "MARKET"
    bracket.exit_market_fallback = True
    bracket.exit_rejected_attempts = 1
    monkeypatch.setattr(
        "nifty_scalper_bot.execution.ledger_bracket_manager.time.time",
        lambda: 1010.0,
    )

    manager._close_bracket(bracket, close_source="broker_fill", exit_price=110.0)

    outcome = manager.get_completed_trade_outcome(SYMBOL)
    assert outcome is not None
    quality = outcome["execution_quality"]
    assert quality["entry_slippage_points"] == 0.5
    assert quality["entry_slippage_cost"] == 65.0
    assert quality["entry_spread_points"] == 1.0
    assert quality["decision_to_entry_fill_seconds"] == 2.0
    assert quality["entry_submit_to_fill_seconds"] == 1.0
    assert quality["exit_slippage_points"] == 0.2
    assert quality["exit_slippage_cost"] == 26.0
    assert quality["exit_spread_points"] == 0.4
    assert quality["exit_trigger_to_fill_seconds"] == 2.0
    assert quality["exit_submit_to_fill_seconds"] == 1.0
    assert quality["exit_order_type"] == "MARKET"
    assert quality["exit_market_fallback"] is True
    assert quality["exit_rejected_attempts"] == 1

    restored = manager._decode_restored_bracket(
        bracket.entry_order_id, bracket.to_dict()
    )
    assert restored.exit_arrival_price == bracket.exit_arrival_price
    assert restored.exit_quote_bid == bracket.exit_quote_bid
    assert restored.exit_quote_ask == bracket.exit_quote_ask
    assert restored.exit_submitted_at == bracket.exit_submitted_at
    assert restored.exit_rejected_attempts == 1


def test_duplicate_entry_callback_is_idempotent(monkeypatch, tmp_path) -> None:
    manager, _order_manager, _broker = _manager(monkeypatch, tmp_path)
    manager.confirm_entry_fill("entry-1", 100.0)
    manager.confirm_entry_fill("entry-1", 100.0)
    assert manager._fill_ledger is not None
    fills = manager._fill_ledger.load_fills("entry-1")
    assert len(fills) == 1
    assert fills[0].fill_id == "ENTRY:entry-1"


def test_partial_exit_persistence_failure_keeps_residual_protected_and_blocks_entries(
    monkeypatch, tmp_path
) -> None:
    manager, _order_manager, broker = _manager(monkeypatch, tmp_path)
    manager.confirm_entry_fill("entry-1", 100.0)

    class _FailingLedger:
        def record_fill(self, _leg: Any) -> bool:
            raise OSError("disk unavailable")

    manager._fill_ledger = _FailingLedger()  # type: ignore[assignment]
    bracket = _mark_filled_exit(
        manager,
        broker,
        order_id="tp1-failed-ledger",
        reason="TP1 Hit (110.00)",
        price=110.0,
        residual=65,
    )
    assert manager._reconcile_exit_state(bracket, requested_by="tp1-failure") is False
    assert bracket.remaining_quantity == 65
    assert bracket.exit_state == BracketExitLifecycle.OPEN_ACTIVE.value
    assert bracket.sl_trigger_price == bracket.entry_price
    assert manager.has_unresolved_exit() is True
    assert bracket.bracket_id in manager._ledger_blocked


def test_final_close_does_not_release_runner_when_ledger_is_incomplete(
    monkeypatch, tmp_path
) -> None:
    manager, _order_manager, broker = _manager(monkeypatch, tmp_path)
    released: list[str] = []
    manager.attach_on_exit_complete(released.append)
    bracket = manager.get_bracket("entry-1")
    assert bracket is not None
    bracket.entry_confirmed = True
    bracket.active = True
    bracket.entry_status = "ACTIVE"

    bracket = _mark_filled_exit(
        manager,
        broker,
        order_id="final-without-entry-ledger",
        reason="HARD_SL_BREACH",
        price=95.0,
        residual=0,
    )
    assert manager._reconcile_exit_state(bracket, requested_by="missing-entry") is True
    assert bracket.exit_state == BracketExitLifecycle.CLOSED.value
    assert released == []
    assert manager.has_unresolved_exit() is True


def test_release_block_survives_manager_restart(monkeypatch, tmp_path) -> None:
    path = tmp_path / "restart.db"
    monkeypatch.setenv("BRACKET_FILL_LEDGER_PATH", str(path))
    broker = _Broker()
    first = LedgerBracketManager(order_manager=_OrderManager(broker))
    first._running = False
    first._watchdog_thread.join(timeout=1.0)
    first.register_virtual_bracket(
        order_id="entry-restart",
        symbol=SYMBOL,
        side="BUY",
        qty=65,
        price=100.0,
        sl=90.0,
        tp=120.0,
        activate_immediately=True,
    )
    bracket = first.get_bracket("entry-restart")
    assert bracket is not None
    first._block_ledger_release(bracket, reason="test_restart", payload={})

    second = LedgerBracketManager(order_manager=_OrderManager(broker))
    second._running = False
    second._watchdog_thread.join(timeout=1.0)
    assert bracket.bracket_id in second._ledger_blocked
    assert second.has_unresolved_exit() is True


def test_duplicate_final_close_does_not_latch_ledger_block(monkeypatch, tmp_path) -> None:
    """P0: closing an already-accounted bracket twice must be a no-op.

    Production symptom: a second _close_bracket() on a successfully accounted
    bracket re-entered final accounting with remaining_quantity already 0,
    raised "final exit identity, quantity or fill price unavailable", and
    latched ledger_blocked=True with
    fill_ledger_degraded:final_exit_accounting_failed.

    That latch does not self-heal -- _retry_ledger_block() sees a pending exit
    marker with quantity 0 and returns early -- so entries stayed blocked with
    ENTRY_BLOCKED_NATIVE_GATE reason=unresolved_exit_position even after
    position reconciliation succeeded.
    """
    manager, _order_manager, broker = _manager(monkeypatch, tmp_path)
    manager.confirm_entry_fill("entry-1", 100.0)
    bracket = _mark_filled_exit(
        manager,
        broker,
        order_id="final-1",
        reason="SL Hit (95.00)",
        price=95.0,
        residual=0,
    )

    manager._close_bracket(bracket, close_source="first")
    assert bracket.exit_executed is True
    assert bracket.remaining_quantity == 0
    assert bracket.bracket_id not in manager._ledger_blocked

    # Second closure: previously latched the block.
    manager._close_bracket(bracket, close_source="duplicate")

    assert bracket.bracket_id not in manager._ledger_blocked
    assert bracket.remaining_quantity == 0
    assert bracket.exit_executed is True
    assert manager.has_unresolved_exit() is False

    # Exactly one entry and one exit fill -- accounting was not repeated.
    assert manager._fill_ledger is not None
    fills = manager._fill_ledger.load_fills(bracket.bracket_id)
    assert len([f for f in fills if f.fill_id.startswith("ENTRY:")]) == 1


def test_concurrent_final_close_accounts_once(monkeypatch, tmp_path) -> None:
    """Two callers beginning final accounting at once must not double-account.

    Watchdog, tick processing and reconciliation can all reach closure; an
    idempotency check alone does not stop two threads entering together.
    """
    import threading

    manager, _order_manager, broker = _manager(monkeypatch, tmp_path)
    manager.confirm_entry_fill("entry-1", 100.0)
    bracket = _mark_filled_exit(
        manager,
        broker,
        order_id="final-1",
        reason="SL Hit (95.00)",
        price=95.0,
        residual=0,
    )

    barrier = threading.Barrier(2)

    def _close(tag: str) -> None:
        barrier.wait(timeout=5.0)
        manager._close_bracket(bracket, close_source=tag)

    threads = [
        threading.Thread(target=_close, args=("concurrent-a",)),
        threading.Thread(target=_close, args=("concurrent-b",)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert bracket.bracket_id not in manager._ledger_blocked
    assert bracket.exit_executed is True
    assert bracket.remaining_quantity == 0
    assert manager.has_unresolved_exit() is False
