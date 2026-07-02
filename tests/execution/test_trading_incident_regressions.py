from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.execution import BracketManager
from nifty_scalper_bot.execution.order_manager_core import (
    OrderDetails,
    OrderManager,
    OrderPreflightResult,
    OrderStatus,
    OrderType,
    TradePlan,
)
from nifty_scalper_bot.execution.position_manager import PositionManager
from nifty_scalper_bot.strategies.runner import _ranked_candidate_for_symbol


SYMBOL = "NFO:NIFTY2670724050CE"
ENTRY_ID = "2072238244200112128"
EXIT_ID = "2072245044739760128"


def _stop_bracket_manager(manager: BracketManager) -> None:
    manager._running = False
    manager._watchdog_thread.join(timeout=1.0)


def test_final_symbol_rebinds_to_its_own_ranked_candidate() -> None:
    rejected = SimpleNamespace(symbol="NFO:NIFTY2670724100CE", entry_price=152.10)
    original = SimpleNamespace(symbol=SYMBOL, entry_price=125.10)

    selected = _ranked_candidate_for_symbol([rejected, original], SYMBOL)

    assert selected is original
    assert selected.entry_price == pytest.approx(125.10)


def test_abnormal_entry_reprice_is_rejected_before_broker(monkeypatch) -> None:
    monkeypatch.delenv("MAX_ENTRY_REPRICE_DEVIATION_PCT", raising=False)
    manager = SimpleNamespace(
        _logger=SimpleNamespace(
            warning=lambda *_a, **_k: None,
            error=lambda *_a, **_k: None,
        ),
        is_kill_switch_active=lambda: False,
        _validate_trade_plan=lambda _plan: OrderPreflightResult(True),
        _protected_limit_price=lambda _plan: 125.10,
        _reanchor_bracket_to_price=lambda plan, _price: plan,
    )
    plan = TradePlan(
        symbol=SYMBOL,
        side="BUY",
        quantity=65,
        entry_price=152.10,
        stop_loss=145.00,
        take_profit=166.00,
        intent="ENTRY",
    )

    result = OrderManager.submit_trade_plan_result(manager, plan)

    assert result.accepted is False
    assert result.reason == "entry_price_deviation_exceeded"
    assert result.broker_attempted is False
    assert result.details["deviation_pct"] > 8.0


def test_pending_entry_bracket_is_owned_not_orphan(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRACKET_AUTO_RESTORE", "false")
    manager = BracketManager(order_manager=SimpleNamespace())
    _stop_bracket_manager(manager)
    manager.register_virtual_bracket(
        order_id=ENTRY_ID,
        symbol=SYMBOL,
        side="BUY",
        qty=65,
        price=145.15,
        sl=140.00,
        tp=155.00,
        intent="ENTRY",
        activate_immediately=False,
    )

    bracket = manager.get_bracket(ENTRY_ID)
    assert bracket is not None
    assert bracket.active is False
    assert bracket.entry_confirmed is False
    assert manager.is_symbol_managed(SYMBOL) is True


class _ExactPendingBracketManager:
    def __init__(self) -> None:
        self.bracket = SimpleNamespace(
            bracket_id=ENTRY_ID,
            entry_confirmed=False,
            active=False,
            sl_trigger_price=140.0,
        )
        self.confirmed = False

    def get_bracket(self, order_id: str):
        return self.bracket if order_id == ENTRY_ID else None

    def has_active_bracket(self, _symbol: str) -> bool:
        raise AssertionError("exact entry bracket must be checked before symbol guard")

    def confirm_entry_fill(self, order_id: str, _price: float) -> None:
        assert order_id == ENTRY_ID
        self.bracket.entry_confirmed = True
        self.bracket.active = True
        self.confirmed = True


def test_exact_pending_bracket_activates_before_symbol_duplicate_guard() -> None:
    bracket_manager = _ExactPendingBracketManager()
    manager = OrderManager.__new__(OrderManager)
    manager._bracket_manager = bracket_manager
    manager._notifier = None
    manager._logger = SimpleNamespace(
        debug=lambda *_a, **_k: None,
        info=lambda *_a, **_k: None,
        warning=lambda *_a, **_k: None,
        error=lambda *_a, **_k: None,
    )
    order = OrderDetails(
        order_id=ENTRY_ID,
        symbol=SYMBOL,
        side="BUY",
        quantity=65,
        order_type=OrderType.LIMIT,
        status=OrderStatus.FILLED,
        price=145.15,
        fill_price=145.15,
        average_price=145.15,
        filled_quantity=65,
        stop_loss=140.0,
        take_profit=155.0,
        intent="ENTRY",
        timestamp=datetime.now(timezone.utc),
    )

    OrderManager._register_virtual_bracket_for_fill(manager, order, source="test")

    assert bracket_manager.confirmed is True
    assert bracket_manager.bracket.active is True


def test_one_broker_buy_and_one_exit_never_become_local_two_lots(tmp_path) -> None:
    state_path = tmp_path / "positions.json"
    manager = PositionManager(state_file=str(state_path))
    manager.add_pending_order(
        ENTRY_ID, SYMBOL, "BUY", 65, 145.15, "LIMIT", intent="ENTRY"
    )

    # Zerodha's authoritative position snapshot already reflects the one-lot BUY.
    manager.synchronize_with_broker(
        [
            {
                "symbol": SYMBOL,
                "product": "MIS",
                "quantity": 65,
                "average_price": 145.15,
                "last_price": 145.15,
                "realised": 0.0,
            }
        ]
    )
    assert manager.get_position(SYMBOL).quantity == 65

    # The delayed COMPLETE event is cumulative confirmation of the same BUY, not
    # another incremental lot. Local quantity must remain 65.
    manager.apply_broker_order_update(
        ENTRY_ID,
        {"status": "COMPLETE", "filled_quantity": 65, "average_price": 145.15},
    )
    position = manager.get_position(SYMBOL)
    assert position is not None
    assert position.quantity == 65
    assert position.order_id == ENTRY_ID
    assert manager._orders[ENTRY_ID].applied_filled_quantity == 65

    manager.confirm_entry_protection(ENTRY_ID, ENTRY_ID, 65)
    assert manager.current_entry_protection_blocker(SYMBOL) is None

    manager.add_pending_order(
        EXIT_ID, SYMBOL, "SELL", 65, 144.05, "MARKET", intent="EXIT"
    )
    manager.apply_broker_order_update(
        EXIT_ID,
        {"status": "COMPLETE", "filled_quantity": 65, "average_price": 144.05},
    )

    assert manager.get_position(SYMBOL) is None
    assert manager.get_realized_pnl() == pytest.approx(-71.50)

    # Restart and replay remain idempotent.
    restarted = PositionManager(state_file=str(state_path))
    restarted.apply_broker_order_update(
        ENTRY_ID,
        {"status": "COMPLETE", "filled_quantity": 65, "average_price": 145.15},
    )
    restarted.apply_broker_order_update(
        EXIT_ID,
        {"status": "COMPLETE", "filled_quantity": 65, "average_price": 144.05},
    )
    assert restarted.get_position(SYMBOL) is None
    assert restarted.get_realized_pnl() == pytest.approx(-71.50)


def test_verified_bracket_acknowledges_position_protection(tmp_path) -> None:
    positions = PositionManager(state_file=str(tmp_path / "positions.json"))
    positions.add_pending_order(
        ENTRY_ID, SYMBOL, "BUY", 65, 145.15, "LIMIT", intent="ENTRY"
    )
    positions.apply_broker_order_update(
        ENTRY_ID,
        {"status": "COMPLETE", "filled_quantity": 65, "average_price": 145.15},
    )
    bracket = SimpleNamespace(
        bracket_id=ENTRY_ID,
        entry_confirmed=True,
        active=True,
        sl_trigger_price=140.0,
    )
    manager = OrderManager.__new__(OrderManager)
    manager._positions = positions
    manager._bracket_manager = SimpleNamespace(get_bracket=lambda _oid: bracket)
    manager._logger = SimpleNamespace(error=lambda *_a, **_k: None)
    order = OrderDetails(
        order_id=ENTRY_ID,
        symbol=SYMBOL,
        side="BUY",
        quantity=65,
        order_type=OrderType.LIMIT,
        status=OrderStatus.FILLED,
        price=145.15,
        fill_price=145.15,
        filled_quantity=65,
        intent="ENTRY",
    )

    OrderManager._confirm_position_protection_for_fill(manager, order)

    assert positions.current_entry_protection_blocker(SYMBOL) is None
    assert positions._terminal_orders[ENTRY_ID].protection_confirmed is True
