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
