from __future__ import annotations

from typing import Any

from nifty_scalper_bot.execution.bracket_manager import BracketExitLifecycle
from nifty_scalper_bot.execution.ledger_bracket_manager import LedgerBracketManager


SYMBOL = "NFO:NIFTYTESTCE"


class _Broker:
    def __init__(self) -> None:
        self.statuses: dict[str, dict[str, Any]] = {}
        self.positions: list[dict[str, Any]] = [{"symbol": SYMBOL, "quantity": 130}]

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

    def place_order(self, **_kwargs: Any) -> str:
        return "unused-exit-order"

    def cancel_order(self, order_id: str, *args: Any, **kwargs: Any) -> bool:
        return self._broker.cancel_order(order_id, *args, **kwargs)

    def set_last_skip_reason(self, _reason: str) -> None:
        return None


def _mark_exit(
    manager: LedgerBracketManager,
    broker: _Broker,
    *,
    order_id: str,
    reason: str,
    price: float,
    residual: int,
):
    bracket = manager.get_bracket("entry-1")
    assert bracket is not None
    bracket.exit_pending = True
    bracket.exit_reason = reason
    bracket.exit_state = BracketExitLifecycle.EXIT_ORDER_SUBMITTED.value
    bracket.entry_status = bracket.exit_state
    bracket.exit_order_id = order_id
    bracket.pending_exit_order_id = order_id
    broker.statuses[order_id] = {"status": "COMPLETE", "average_price": price}
    broker.positions = [] if residual == 0 else [{"symbol": SYMBOL, "quantity": residual}]
    return bracket


def test_incomplete_final_ledger_never_fabricates_full_quantity_completed_pnl(
    monkeypatch, tmp_path
) -> None:
    """Strict LIVE accounting must prefer unresolved PnL to invented economics."""
    monkeypatch.setenv("BRACKET_FILL_LEDGER_PATH", str(tmp_path / "ledger.db"))
    broker = _Broker()
    manager = LedgerBracketManager(order_manager=_OrderManager(broker))
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
    manager.confirm_entry_fill("entry-1", 100.0)

    bracket = _mark_exit(
        manager,
        broker,
        order_id="tp1-order",
        reason="TP1 Hit (110.00)",
        price=110.0,
        residual=65,
    )
    assert manager._reconcile_exit_state(bracket, requested_by="tp1") is False
    assert bracket.remaining_quantity == 65
    assert manager._fill_ledger is not None
    partial = manager._fill_ledger.realized_pnl(bracket.bracket_id)
    assert partial.complete is False
    assert partial.gross_pnl == 650.0

    bracket = _mark_exit(
        manager,
        broker,
        order_id="final-order",
        reason="HARD_SL_BREACH",
        price=95.0,
        residual=0,
    )
    original_record_exit = manager._record_exit_fill

    def _fail_final_fill(*args: Any, **kwargs: Any) -> None:
        if kwargs.get("order_id") == "final-order":
            raise OSError("simulated final-fill persistence failure")
        original_record_exit(*args, **kwargs)

    monkeypatch.setattr(manager, "_record_exit_fill", _fail_final_fill)
    released: list[str] = []
    manager.attach_on_exit_complete(released.append)

    manager._close_bracket(bracket, close_source="broker_fill", exit_price=95.0)

    outcome = manager.get_completed_trade_outcome(SYMBOL)
    assert outcome is not None
    assert outcome["ledger_complete"] is False
    assert outcome["gross_pnl"] is None
    assert outcome["net_pnl"] is None
    assert released == []
    assert manager.has_unresolved_exit() is True
