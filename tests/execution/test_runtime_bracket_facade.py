from __future__ import annotations

from datetime import datetime, timezone
import importlib
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import nifty_scalper_bot.execution as execution
from nifty_scalper_bot.execution import bracket_core, bracket_manager
from nifty_scalper_bot.execution.ownership import BoundBracketManager
from nifty_scalper_bot.execution.runtime_bracket_manager import RuntimeBracketManager


class _OrderManager:
    def __init__(self) -> None:
        self._broker = None
        self.provider: Any | None = None
        self._unresolved_exit_guard_installed = False

    def set_unresolved_exit_provider(self, provider: Any | None) -> None:
        self.provider = provider
        self._unresolved_exit_guard_installed = provider is not None

    def place_order(self, **_kwargs: Any) -> str:
        return "test-order"

    def cancel_order(self, _order_id: str, *args: Any, **kwargs: Any) -> bool:
        return True


def test_public_bracket_import_has_one_runtime_identity() -> None:
    assert bracket_manager.BracketManager is BoundBracketManager
    assert execution.BracketManager is BoundBracketManager
    assert execution.CanonicalBracketManager is BoundBracketManager
    assert issubclass(BoundBracketManager, RuntimeBracketManager)
    assert issubclass(BoundBracketManager, bracket_core.BracketManager)
    assert bracket_manager.BracketExitLifecycle is bracket_core.BracketExitLifecycle
    assert not hasattr(bracket_manager, "LegacyBracketManager")


def test_importing_execution_package_does_not_replace_bracket_class() -> None:
    before = bracket_manager.BracketManager
    imported = importlib.import_module("nifty_scalper_bot.execution")
    assert imported.BracketManager is before
    assert bracket_manager.BracketManager is before


def test_runtime_manager_binds_provider_without_replacing_order_methods(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("BRACKET_FILL_LEDGER_PATH", str(tmp_path / "facade.db"))
    order_manager = _OrderManager()
    before = order_manager.place_order.__func__
    manager = bracket_manager.BracketManager(order_manager=order_manager)
    manager._running = False
    manager._watchdog_thread.join(timeout=1.0)
    assert isinstance(manager, BoundBracketManager)
    assert order_manager.provider is manager
    assert order_manager.place_order.__func__ is before


def test_bracket_module_is_safe_when_imported_before_package() -> None:
    code = r"""
import json
import importlib
bm = importlib.import_module("nifty_scalper_bot.execution.bracket_manager")
before = id(bm.BracketManager)
execution = importlib.import_module("nifty_scalper_bot.execution")
print(json.dumps({
    "before": before,
    "after": id(bm.BracketManager),
    "package": id(execution.BracketManager),
    "module": bm.BracketManager.__module__,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                filter(None, ["src", os.environ.get("PYTHONPATH", "")])
            ),
        },
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["before"] == payload["after"] == payload["package"]
    assert payload["module"] == "nifty_scalper_bot.execution.ownership"


def test_runtime_bracket_reconciliation_comes_from_canonical_without_patch(
    tmp_path, monkeypatch
) -> None:
    from nifty_scalper_bot.execution.canonical_bracket_manager import (
        CanonicalBracketManager,
    )

    monkeypatch.setenv(
        "BRACKET_FILL_LEDGER_PATH", str(tmp_path / "facade-canonical.db")
    )
    reconcile_before = BoundBracketManager._reconcile_exit_state
    rescue_before = BoundBracketManager._rescue_stale_exit_order
    order_manager = _OrderManager()
    manager = bracket_manager.BracketManager(order_manager=order_manager)
    manager._running = False
    manager._watchdog_thread.join(timeout=1.0)

    assert isinstance(manager, CanonicalBracketManager)
    assert BoundBracketManager._reconcile_exit_state is reconcile_before
    assert BoundBracketManager._rescue_stale_exit_order is rescue_before
    assert "CanonicalBracketManager" in rescue_before.__qualname__


def test_identical_entry_fill_callback_does_not_reactivate_bracket(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("BRACKET_FILL_LEDGER_PATH", str(tmp_path / "idempotent.db"))
    manager = bracket_manager.BracketManager(order_manager=_OrderManager())
    manager._running = False
    manager._watchdog_thread.join(timeout=1.0)
    manager.register_virtual_bracket(
        order_id="entry-1",
        symbol="NFO:NIFTY26JUL23950CE",
        side="BUY",
        qty=65,
        price=90.05,
        sl=87.35,
        tp=95.20,
        activate_immediately=False,
    )

    manager.confirm_entry_fill("entry-1", 88.65)
    bracket = manager.get_bracket("entry-1")
    assert bracket is not None
    activated_at = bracket.entry_fill_ts
    bracket.trail_revision = 7

    assert manager.confirm_entry_fill("entry-1", 88.65) is True
    assert bracket.entry_fill_ts == activated_at
    assert bracket.trail_revision == 7


def test_tick_epoch_uses_explicit_receipt_time_when_exchange_time_missing() -> None:
    received_at = datetime(2026, 7, 27, 7, 49, tzinfo=timezone.utc)
    assert bracket_core.tick_exchange_epoch({"received_at": received_at}) == received_at.timestamp()


class _ExitBroker:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.statuses: dict[str, dict[str, Any]] = {}
        self.positions: list[dict[str, Any]] = [
            {"symbol": symbol, "quantity": 65, "average_price": 100.0}
        ]

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        return dict(self.statuses.get(order_id, {"status": ""}))

    def get_positions(self) -> list[dict[str, Any]]:
        return list(self.positions)


class _ExitOrderManager(_OrderManager):
    def __init__(self, broker: _ExitBroker) -> None:
        super().__init__()
        self._broker = broker


def _exit_manager(tmp_path, monkeypatch):
    symbol = "NFO:NIFTY2681124500CE"
    monkeypatch.setenv("BRACKET_FILL_LEDGER_PATH", str(tmp_path / "exit-correlation.db"))
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("ENABLE_LIVE", "false")
    broker = _ExitBroker(symbol)
    manager = bracket_manager.BracketManager(order_manager=_ExitOrderManager(broker))
    manager._running = False
    manager._watchdog_thread.join(timeout=1.0)
    manager._filled_position_sync_grace_seconds = 0.0
    manager.register_virtual_bracket(
        order_id="ENTRY-1",
        symbol=symbol,
        side="BUY",
        qty=65,
        price=100.0,
        sl=90.0,
        tp=120.0,
        activate_immediately=False,
    )
    manager.confirm_entry_fill("ENTRY-1", 100.0)
    return manager, broker, symbol


def test_managed_filled_exit_terminalizes_linked_bracket(tmp_path, monkeypatch) -> None:
    manager, broker, symbol = _exit_manager(tmp_path, monkeypatch)
    bracket = manager.get_bracket("ENTRY-1")
    assert bracket is not None
    bracket.exit_reason = "HARD_SL_BREACH"
    broker.statuses["EXIT-1"] = {"status": "COMPLETE", "average_price": 95.0}
    broker.positions = []
    order = SimpleNamespace(
        order_id="EXIT-1",
        symbol=symbol,
        side="SELL",
        quantity=65,
        filled_quantity=65,
        fill_price=95.0,
        intent="EXIT",
        bracket_id="ENTRY-1",
        linked_entry_order_id="ENTRY-1",
        trade_lifecycle_id="ENTRY-1",
    )

    assert manager.reconcile_filled_exit_order(order, broker.statuses["EXIT-1"]) is True
    assert bracket.exit_state == bracket_core.BracketExitLifecycle.CLOSED.value
    assert bracket.exit_reason == "HARD_SL_BREACH"
    assert bracket.exit_order_id == "EXIT-1"
    assert "EXIT-1" in bracket.linked_exit_order_ids
    assert manager.has_unresolved_exit() is False


def test_external_flatten_closes_only_unique_flat_bracket(tmp_path, monkeypatch) -> None:
    manager, broker, symbol = _exit_manager(tmp_path, monkeypatch)
    bracket = manager.get_bracket("ENTRY-1")
    assert bracket is not None
    broker.statuses["MANUAL-1"] = {"status": "COMPLETE", "average_price": 97.0}
    broker.positions = []
    order = SimpleNamespace(
        order_id="MANUAL-1",
        symbol=symbol,
        side="SELL",
        quantity=65,
        filled_quantity=65,
        fill_price=97.0,
        intent="EXIT",
        bracket_id=None,
        linked_entry_order_id=None,
        trade_lifecycle_id=None,
    )

    assert manager.reconcile_filled_exit_order(order, broker.statuses["MANUAL-1"]) is True
    assert bracket.exit_state == bracket_core.BracketExitLifecycle.CLOSED.value
    assert bracket.exit_reason == "BROKER_EXTERNAL"
    assert bracket.close_source == "broker_fill"


def test_external_exit_does_not_steal_bracket_while_broker_nonflat(
    tmp_path, monkeypatch
) -> None:
    manager, broker, symbol = _exit_manager(tmp_path, monkeypatch)
    bracket = manager.get_bracket("ENTRY-1")
    assert bracket is not None
    broker.statuses["MANUAL-1"] = {"status": "COMPLETE", "average_price": 97.0}
    order = SimpleNamespace(
        order_id="MANUAL-1",
        symbol=symbol,
        side="SELL",
        quantity=65,
        filled_quantity=65,
        fill_price=97.0,
        intent="EXIT",
        bracket_id=None,
        linked_entry_order_id=None,
        trade_lifecycle_id=None,
    )

    assert manager.reconcile_filled_exit_order(order, broker.statuses["MANUAL-1"]) is False
    assert bracket.exit_state == bracket_core.BracketExitLifecycle.OPEN_ACTIVE.value
    assert bracket.exit_order_id is None
    assert bracket.exit_reason is None
