from __future__ import annotations

import ast
import importlib
from pathlib import Path

import nifty_scalper_bot.execution as execution
from nifty_scalper_bot.execution.adaptive_trailing import AdaptiveTrailingController
from nifty_scalper_bot.execution.adaptive_trailing_core import (
    AdaptiveTrailingController as CoreAdaptiveTrailingController,
)
from nifty_scalper_bot.execution.bracket_core import BracketManager as CoreBracketManager
from nifty_scalper_bot.execution.bracket_manager import BracketManager
from nifty_scalper_bot.execution.order_manager import OrderManager
from nifty_scalper_bot.execution.order_manager_core import OrderManager as CoreOrderManager


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "nifty_scalper_bot"
EXECUTION = SRC / "execution"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _strategy_runner_execute_order_calls() -> set[str]:
    path = SRC / "strategies" / "runner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_execute_order":
            return {
                call.func.attr
                for call in ast.walk(node)
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            }
    raise AssertionError("StrategyRunner._execute_order not found")


def test_public_runtime_has_one_owner_per_lifecycle_domain() -> None:
    assert OrderManager.__module__ == "nifty_scalper_bot.execution.runtime_order_manager"
    assert BracketManager.__module__ == "nifty_scalper_bot.execution.ownership"
    assert AdaptiveTrailingController.__module__ == (
        "nifty_scalper_bot.execution.hardened_adaptive_trailing"
    )
    assert execution.OrderManager is OrderManager
    assert execution.BracketManager is BracketManager
    assert execution.AdaptiveTrailingController is AdaptiveTrailingController
    assert issubclass(OrderManager, CoreOrderManager)
    assert issubclass(BracketManager, CoreBracketManager)
    assert issubclass(AdaptiveTrailingController, CoreAdaptiveTrailingController)


def test_execution_package_import_does_not_install_or_replace_runtime_methods() -> None:
    order_method = OrderManager.submit_trade_plan_result
    bracket_class = BracketManager
    trailing_class = AdaptiveTrailingController
    package = importlib.import_module("nifty_scalper_bot.execution")
    assert package.OrderManager.submit_trade_plan_result is order_method
    assert package.BracketManager is bracket_class
    assert package.AdaptiveTrailingController is trailing_class
    source = (EXECUTION / "__init__.py").read_text(encoding="utf-8")
    assert "install_entry_recovery(" not in source
    assert "_bracket_module.BracketManager" not in source
    assert "LegacyOrderManager" not in source
    assert "LegacyBracketManager" not in source
    assert "LegacyAdaptiveTrailingController" not in source


def test_runner_uses_only_canonical_entry_api() -> None:
    calls = _strategy_runner_execute_order_calls()
    assert "submit_trade_plan_result" in calls
    assert "execute_market_order" not in calls
    assert "place_order" not in calls
    assert "attach_dynamic_tp" not in calls
    assert "stop_dynamic_tp" not in calls


def test_runtime_classes_own_required_methods_through_explicit_mro() -> None:
    for name in (
        "submit_trade_plan_result",
        "place_managed_order_result",
        "place_order",
        "_update_from_response",
        "set_unresolved_exit_provider",
    ):
        assert callable(getattr(OrderManager, name, None)), name
    for name in (
        "register_virtual_bracket",
        "confirm_entry_fill",
        "confirm_partial_entry_fill",
        "_evaluate_exit_fast",
        "_reconcile_exit_state",
        "_close_bracket",
        "has_unresolved_exit",
    ):
        assert callable(getattr(BracketManager, name, None)), name


def test_core_modules_are_imported_only_by_canonical_facades() -> None:
    allowed = {
        "nifty_scalper_bot.execution.order_manager_core": {
            "order_manager.py",
            "runtime_order_manager.py",
        },
        "nifty_scalper_bot.execution.bracket_core": {
            "bracket_manager.py",
            "ownership.py",
        },
        "nifty_scalper_bot.execution.adaptive_trailing_core": {
            "adaptive_trailing.py",
        },
    }
    offenders: list[str] = []
    for path in SRC.rglob("*.py"):
        imported = _imports(path)
        relative = str(path.relative_to(ROOT))
        for module, allowed_names in allowed.items():
            if module in imported and path.name not in allowed_names:
                offenders.append(f"{relative} imports {module}")
    assert not offenders, offenders


def test_retired_duplicate_bo_modules_are_absent() -> None:
    retired = {
        "order_manager_legacy.py",
        "legacy_bracket_manager.py",
        "adaptive_trailing_legacy.py",
        "dynamic_tp.py",
        "order_executor.py",
        "order_processor.py",
        "entry_price.py",
    }
    present = sorted(name for name in retired if (EXECUTION / name).exists())
    assert not present, f"Retired BO modules still present: {present}"
    assert (EXECUTION / "options_policy.py").exists()


def test_startup_compatibility_adapters_do_not_own_execution_logic() -> None:
    safe_tree = ast.parse((EXECUTION / "safe_order_manager.py").read_text(encoding="utf-8"))
    lifecycle_tree = ast.parse((EXECUTION / "lifecycle_manager.py").read_text(encoding="utf-8"))
    safe_methods = {
        node.name
        for node in ast.walk(safe_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    lifecycle_methods = {
        node.name
        for node in ast.walk(lifecycle_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "_chase_fill" not in safe_methods
    assert "_check_rate_limit" not in safe_methods
    assert "_monitor_loop" not in safe_methods
    assert "_evaluate_tick" not in lifecycle_methods
    assert "_monitor_loop" not in lifecycle_methods
    assert "_execute_exit" not in lifecycle_methods

def test_live_capable_strategies_do_not_bypass_trade_plan_execution() -> None:
    offenders: list[str] = []
    strategies_root = SRC / "strategies"
    for path in strategies_root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if ".place_order(" not in source:
            continue
        if path.name == "premium_decay.py" and "LIVE_CAPABLE = False" in source:
            continue
        offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"Live-capable strategy bypasses canonical TradePlan path: {offenders}"
