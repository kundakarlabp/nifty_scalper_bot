from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from nifty_scalper_bot.execution.bracket_manager import (
    BracketExitLifecycle,
    BracketManager,
)
from nifty_scalper_bot.execution.native_entry_gate import NO_BLOCK
from nifty_scalper_bot.execution.order_manager import OrderManager
from nifty_scalper_bot.execution.position_manager import PositionManager
from nifty_scalper_bot.utils.rate_limiter import RateLimiter

SYMBOL = "NFO:NIFTY2662324050PE"


class _Broker:
    def __init__(self) -> None:
        self.orders: dict[str, dict[str, Any]] = {}
        self.positions: list[dict[str, Any]] = []
        self.counter = 0
        self.is_connected = True

    def place_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.counter += 1
        order_id = f"ORD-{self.counter}"
        self.orders[order_id] = {
            "order_id": order_id,
            "status": "OPEN",
            "quantity": int(payload.get("quantity") or 0),
            "filled_quantity": 0,
            "average_price": payload.get("price"),
            "payload": dict(payload),
        }
        return {"order_id": order_id, "status": "OPEN"}

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        return dict(self.orders.get(order_id, {"status": ""}))

    def get_positions(self) -> list[dict[str, Any]]:
        return list(self.positions)

    def cancel_order(self, order_id: str, *args: Any, **kwargs: Any) -> bool:
        if order_id in self.orders:
            self.orders[order_id]["status"] = "CANCELLED"
        return True

    def modify_order(self, order_id: str, **kwargs: Any) -> dict[str, Any]:
        if order_id in self.orders:
            self.orders[order_id].update(kwargs)
        return {"order_id": order_id}


class _Resolver:
    def lot_size_for_symbol(self, _symbol: str) -> int:
        return 65


def _composition(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRACKET_FILL_LEDGER_PATH", str(tmp_path / "bo.db"))
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("ENABLE_LIVE", "false")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "false")
    monkeypatch.setenv("ORDERS__ENABLE_LIVE", "false")
    broker = _Broker()
    positions = PositionManager(state_file=str(tmp_path / "positions.json"))
    limiter = RateLimiter()
    limiter.configure_bucket("orders", capacity=10, refill_rate_per_sec=10)
    order_manager = OrderManager(
        broker,
        positions,
        limiter,
        history_path=tmp_path / "orders.json",
    )
    monkeypatch.setattr(order_manager, "is_live_mode", lambda: False, raising=False)
    order_manager.set_instrument_resolver(_Resolver())
    bracket_manager = BracketManager(order_manager=order_manager)
    bracket_manager._running = False
    bracket_manager._watchdog_thread.join(timeout=1.0)
    bracket_manager._filled_position_sync_grace_seconds = 0.0
    bracket_manager._exit_reconcile_interval_seconds = 0.0
    return broker, order_manager, bracket_manager


def _register(bracket_manager: BracketManager) -> Any:
    bracket_manager.register_virtual_bracket(
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
    bracket_manager.confirm_entry_fill("entry-1", 100.0)
    bracket = bracket_manager.get_bracket("entry-1")
    assert bracket is not None
    return bracket


def _filled_exit(
    broker: _Broker,
    bracket: Any,
    *,
    order_id: str,
    reason: str,
    price: float,
    residual: int,
) -> None:
    bracket.exit_pending = True
    bracket.exit_reason = reason
    bracket.exit_state = BracketExitLifecycle.EXIT_ORDER_SUBMITTED.value
    bracket.entry_status = bracket.exit_state
    bracket.exit_order_id = order_id
    bracket.pending_exit_order_id = order_id
    bracket.exit_triggered_at = time.time()
    broker.orders[order_id] = {
        "order_id": order_id,
        "status": "COMPLETE",
        "average_price": price,
    }
    broker.positions = (
        [] if residual == 0 else [{"tradingsymbol": SYMBOL, "quantity": residual}]
    )


def test_public_composition_runs_entry_tp1_final_pnl_and_release(
    monkeypatch,
    tmp_path,
) -> None:
    broker, order_manager, bracket_manager = _composition(monkeypatch, tmp_path)
    assert order_manager._unresolved_exit_provider is bracket_manager
    bracket = _register(bracket_manager)

    events: list[str] = []
    releases: list[str] = []
    bracket_manager.set_notifier(lambda event, _payload: events.append(event))
    bracket_manager.attach_on_exit_complete(releases.append)

    _filled_exit(
        broker,
        bracket,
        order_id="tp1-order",
        reason="TP1 Hit (110.00)",
        price=110.0,
        residual=65,
    )
    assert (
        bracket_manager._reconcile_exit_state(bracket, requested_by="e2e_tp1") is False
    )
    assert bracket.remaining_quantity == 65
    assert bracket.tp_levels[0].executed is True
    assert bracket.sl_trigger_price == bracket.entry_price
    assert bracket.exit_state == BracketExitLifecycle.OPEN_ACTIVE.value
    assert releases == []

    _filled_exit(
        broker,
        bracket,
        order_id="final-order",
        reason="HARD_SL_BREACH",
        price=95.0,
        residual=0,
    )
    assert (
        bracket_manager._reconcile_exit_state(bracket, requested_by="e2e_final") is True
    )
    assert bracket.exit_state == BracketExitLifecycle.CLOSED.value
    assert bracket.position_flat_confirmed is True
    assert releases == [SYMBOL]

    assert bracket_manager._fill_ledger is not None
    pnl = bracket_manager._fill_ledger.realized_pnl(bracket.bracket_id)
    assert pnl.entry_quantity == 130
    assert pnl.exit_quantity == 130
    assert pnl.gross_pnl == 325.0
    assert pnl.complete is True
    assert "PARTIAL_EXIT_CONFIRMED" in events
    assert "BRACKET_CLOSED" in events
    assert (
        order_manager._blocked(
            "submit_trade_plan",
            tuple(),
            {"symbol": SYMBOL},
        )
        is NO_BLOCK
    )
