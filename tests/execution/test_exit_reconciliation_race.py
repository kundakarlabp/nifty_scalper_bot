from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from nifty_scalper_bot.execution.bracket_manager import BracketManager
from nifty_scalper_bot.execution.position_manager import PositionManager
from nifty_scalper_bot.strategies.runner import StrategyRunner

SYMBOL = "NFO:NIFTY2660923100CE"
QTY = 65


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))


class _Broker:
    def __init__(
        self, *, status: str = "COMPLETE", positions: list[dict[str, Any]] | None = None
    ) -> None:
        self.status = status
        self.positions = positions if positions is not None else []

    def get_order_status(self, _order_id: str) -> dict[str, Any]:
        return {"status": self.status, "average_price": 120.0}

    def get_positions(self) -> list[dict[str, Any]]:
        return list(self.positions)


class _OrderManager:
    def __init__(self, broker: _Broker) -> None:
        self._broker = broker
        self.calls: list[dict[str, Any]] = []

    def place_order(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return "exit-1"


def _position_manager(tmp_path) -> PositionManager:
    pm = PositionManager(str(tmp_path / "positions.json"))
    pm.open_position(SYMBOL, "LONG", QTY, 100.0, order_id="entry-1")
    pm.add_pending_order(
        "exit-1",
        SYMBOL,
        "SELL",
        QTY,
        120.0,
        "MARKET",
        intent="EXIT",
        bracket_id="entry-1",
    )
    return pm


def _bracket_manager() -> BracketManager:
    broker = _Broker(status="COMPLETE", positions=[])
    bm = BracketManager(order_manager=_OrderManager(broker))
    bm.register_virtual_bracket(
        order_id="entry-1",
        symbol=SYMBOL,
        side="BUY",
        qty=QTY,
        price=100.0,
        sl=90.0,
        tp=120.0,
    )
    bm.confirm_entry_fill("entry-1", 100.0)
    return bm


def test_exit_fill_reconciliation_flat_does_not_leave_orphan_bracket(tmp_path) -> None:
    """Terminal SELL fill makes broker, PM and bracket ownership flat immediately."""
    pm = _position_manager(tmp_path)
    bm = _bracket_manager()

    pm.update_order_status("exit-1", "FILLED", 120.0)
    removed = bm.reconcile_symbol_flat(SYMBOL)

    assert pm.get_position(SYMBOL) is None
    assert pm.get_all_positions() == []
    assert pm.unresolved_terminal_summary()["count"] == 0
    assert removed == 1
    assert bm.is_symbol_managed(SYMBOL) is False
    assert bm.get_bracket("entry-1") is None

    runner = StrategyRunner.__new__(StrategyRunner)
    runner._position_manager = pm
    runner._bracket_manager = bm
    runner._active_orphan_guards = set()
    runner._orphan_retry_count = {}
    runner._orphan_retry_last_attempt = {}
    runner._logger = SimpleNamespace(
        debug=lambda *a, **k: None,
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
        exception=lambda *a, **k: None,
    )

    runner._adopt_orphan_positions()

    assert bm.is_symbol_managed(SYMBOL) is False
    assert bm.get_bracket(f"orphan_{SYMBOL}") is None


def test_stale_snapshot_after_exit_fill_is_not_orphan_adopted_then_zero_clears(
    tmp_path,
) -> None:
    """A stale non-zero broker snapshot after a terminal exit cannot re-open/adopt."""
    pm = _position_manager(tmp_path)
    bm = _bracket_manager()
    adopt_calls: list[dict[str, Any]] = []
    bm.attach_orphan_position = lambda **kwargs: adopt_calls.append(kwargs) or "orphan"  # type: ignore[method-assign]

    pm.update_order_status("exit-1", "FILLED", 120.0)
    bm.reconcile_symbol_flat(SYMBOL)

    pm.synchronize_with_broker(
        [
            {
                "symbol": SYMBOL,
                "quantity": QTY,
                "average_price": 100.0,
                "last_price": 120.0,
                "product": "MIS",
            }
        ]
    )

    runner = StrategyRunner.__new__(StrategyRunner)
    runner._position_manager = pm
    runner._bracket_manager = bm
    runner._active_orphan_guards = set()
    runner._orphan_retry_count = {}
    runner._orphan_retry_last_attempt = {}
    runner._logger = SimpleNamespace(
        debug=lambda *a, **k: None,
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
        exception=lambda *a, **k: None,
    )

    runner._adopt_orphan_positions()

    assert pm.get_all_positions() == []
    assert adopt_calls == []
    assert SYMBOL in pm._recently_flat_exit_until_monotonic

    pm.synchronize_with_broker([])

    assert pm.get_all_positions() == []
    assert SYMBOL not in pm._recently_flat_exit_until_monotonic
    assert bm.is_symbol_managed(SYMBOL) is False


def test_terminal_exit_order_not_pending_for_bracket_ghost_or_orphan(tmp_path) -> None:
    pm = _position_manager(tmp_path)
    bm = _bracket_manager()

    pm.update_order_status("exit-1", "FILLED", 120.0)
    bm.reconcile_symbol_flat(SYMBOL)

    assert "exit-1" not in pm._orders
    assert pm._terminal_orders["exit-1"].lifecycle_resolved is True
    assert pm.unresolved_terminal_summary()["count"] == 0
    assert bm.has_unresolved_exit() is False
    assert bm.is_symbol_managed(SYMBOL) is False
    assert bm.get_bracket("entry-1") is None
    assert bm.get_bracket(f"orphan_{SYMBOL}") is None


def test_exit_reconciliation_grace_configuration(monkeypatch) -> None:
    from nifty_scalper_bot.execution import position_manager as module

    monkeypatch.delenv("EXIT_RECONCILIATION_SETTLEMENT_GRACE_SECONDS", raising=False)
    assert module._resolve_exit_reconciliation_grace_seconds() == 2.0
    monkeypatch.setenv("EXIT_RECONCILIATION_SETTLEMENT_GRACE_SECONDS", "bad")
    assert module._resolve_exit_reconciliation_grace_seconds() == 2.0
    monkeypatch.setenv("EXIT_RECONCILIATION_SETTLEMENT_GRACE_SECONDS", "nan")
    assert module._resolve_exit_reconciliation_grace_seconds() == 2.0
    monkeypatch.setenv("EXIT_RECONCILIATION_SETTLEMENT_GRACE_SECONDS", "0.1")
    assert module._resolve_exit_reconciliation_grace_seconds() == 0.25
    monkeypatch.setenv("EXIT_RECONCILIATION_SETTLEMENT_GRACE_SECONDS", "10")
    assert module._resolve_exit_reconciliation_grace_seconds() == 5.0


def _broker_row(quantity: int = QTY) -> dict[str, Any]:
    return {
        "symbol": SYMBOL,
        "quantity": quantity,
        "average_price": 100.0,
        "last_price": 120.0,
        "product": "MIS",
    }


def test_repeated_stale_snapshots_do_not_extend_fixed_exit_deadline(tmp_path) -> None:
    pm = _position_manager(tmp_path)
    pm._recently_flat_exit_grace_seconds = 1.0
    pm.update_order_status("exit-1", "FILLED", 120.0)
    first_deadline = pm._recently_flat_exit_until_monotonic[SYMBOL]

    pm.synchronize_with_broker([_broker_row()])
    pm.synchronize_with_broker([_broker_row()])

    assert pm.get_position(SYMBOL) is None
    assert pm._recently_flat_exit_until_monotonic[SYMBOL] == first_deadline
    guard = pm._recently_flat_exit_metadata[SYMBOL]
    assert guard.stale_snapshot_count == 2
    assert guard.last_stale_quantity == QTY


def test_zero_or_missing_snapshot_clears_exit_guard(tmp_path) -> None:
    pm = _position_manager(tmp_path)
    pm.update_order_status("exit-1", "FILLED", 120.0)

    pm.synchronize_with_broker([_broker_row(0)])
    assert SYMBOL not in pm._recently_flat_exit_until_monotonic
    assert SYMBOL not in pm._recently_flat_exit_metadata

    pm = _position_manager(tmp_path / "second")
    pm.update_order_status("exit-1", "FILLED", 120.0)
    pm.synchronize_with_broker([])
    assert SYMBOL not in pm._recently_flat_exit_until_monotonic
    assert SYMBOL not in pm._recently_flat_exit_metadata


def test_persistent_non_zero_position_is_restored_after_exit_grace_expiry(
    tmp_path,
) -> None:
    pm = _position_manager(tmp_path)
    pm._recently_flat_exit_grace_seconds = 0.25
    pm.update_order_status("exit-1", "FILLED", 120.0)
    pm._recently_flat_exit_until_monotonic[SYMBOL] = 0.0
    pm._recently_flat_exit_metadata[SYMBOL].grace_until_monotonic = 0.0

    pm.synchronize_with_broker([_broker_row()])

    restored = pm.get_position(SYMBOL)
    assert restored is not None
    assert restored.quantity == QTY
    assert SYMBOL not in pm._recently_flat_exit_until_monotonic


def test_new_entry_clears_old_exit_guard(tmp_path) -> None:
    pm = _position_manager(tmp_path)
    pm.update_order_status("exit-1", "FILLED", 120.0)

    pm.open_position(SYMBOL, "LONG", QTY, 121.0, order_id="entry-2")

    assert SYMBOL not in pm._recently_flat_exit_until_monotonic
    assert SYMBOL not in pm._recently_flat_exit_metadata
    assert pm.get_position(SYMBOL) is not None


def test_partial_exit_does_not_create_flat_guard(tmp_path) -> None:
    pm = PositionManager(str(tmp_path / "positions.json"))
    pm.open_position(SYMBOL, "LONG", QTY * 2, 100.0, order_id="entry-1")
    pm.add_pending_order(
        "exit-partial",
        SYMBOL,
        "SELL",
        QTY,
        120.0,
        "MARKET",
        intent="REDUCE",
        bracket_id="entry-1",
    )

    pm.update_order_status("exit-partial", "FILLED", 120.0)

    assert pm.get_position(SYMBOL) is not None
    assert SYMBOL not in pm._recently_flat_exit_until_monotonic


def test_duplicate_terminal_callbacks_remain_idempotent(tmp_path) -> None:
    pm = _position_manager(tmp_path)

    pm.update_order_status("exit-1", "FILLED", 120.0)
    pm.update_order_status("exit-1", "FILLED", 120.0)

    assert pm.get_position(SYMBOL) is None
    assert pm.unresolved_terminal_summary()["count"] == 0
    assert pm._terminal_orders["exit-1"].lifecycle_applied is True
