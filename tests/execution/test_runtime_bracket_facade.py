from __future__ import annotations

from datetime import datetime, timezone
import importlib
import json
import os
import subprocess
import sys
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
