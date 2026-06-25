from __future__ import annotations

import importlib
import json
import subprocess
import sys
from typing import Any

import nifty_scalper_bot.execution as execution
from nifty_scalper_bot.execution import bracket_manager
from nifty_scalper_bot.execution import legacy_bracket_manager
from nifty_scalper_bot.execution.runtime_bracket_manager import RuntimeBracketManager


class _OrderManager:
    def __init__(self) -> None:
        self._broker = None

    def place_order(self, **_kwargs: Any) -> str:
        return "test-order"

    def cancel_order(self, _order_id: str, *args: Any, **kwargs: Any) -> bool:
        return True


def test_public_bracket_import_has_one_runtime_identity() -> None:
    assert bracket_manager.BracketManager is RuntimeBracketManager
    assert execution.BracketManager is RuntimeBracketManager
    assert execution.CanonicalBracketManager is RuntimeBracketManager
    assert bracket_manager.LegacyBracketManager is legacy_bracket_manager.BracketManager
    assert bracket_manager.BracketExitLifecycle is legacy_bracket_manager.BracketExitLifecycle


def test_importing_execution_package_does_not_replace_bracket_class() -> None:
    before = bracket_manager.BracketManager
    imported = importlib.import_module("nifty_scalper_bot.execution")
    assert imported.BracketManager is before
    assert bracket_manager.BracketManager is before


def test_runtime_manager_constructs_end_to_end_from_public_facade(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BRACKET_FILL_LEDGER_PATH", str(tmp_path / "facade.db"))
    manager = bracket_manager.BracketManager(order_manager=_OrderManager())
    manager._running = False
    manager._watchdog_thread.join(timeout=1.0)
    assert isinstance(manager, RuntimeBracketManager)
    assert manager.order_manager is not None


def test_bracket_module_is_safe_when_imported_before_package() -> None:
    code = r'''
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
    "legacy_module": bm.LegacyBracketManager.__module__,
}))
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["before"] == payload["after"] == payload["package"]
    assert payload["module"] == "nifty_scalper_bot.execution.runtime_bracket_manager"
    assert payload["legacy_module"] == "nifty_scalper_bot.execution.legacy_bracket_manager"
