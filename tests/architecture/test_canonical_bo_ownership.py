from __future__ import annotations

import ast
import importlib
from pathlib import Path

import nifty_scalper_bot.execution as execution
from nifty_scalper_bot.execution.adaptive_trailing import (
    AdaptiveTrailingController,
    LegacyAdaptiveTrailingController,
)
from nifty_scalper_bot.execution.bracket_manager import (
    BracketManager,
    LegacyBracketManager,
)
from nifty_scalper_bot.execution.order_manager import (
    LegacyOrderManager,
    OrderManager,
)


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
    assert issubclass(OrderManager, LegacyOrderManager)
    assert issubclass(BracketManager, LegacyBracketManager)
    assert issubclass(AdaptiveTrailingController, LegacyAdaptiveTrailingController)


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
    assert "AdaptiveTrailingController = HardenedAdaptiveTrailingController" not in source
    assert "_bracket_module.BracketManager" not in source


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


def test_compatibility_modules_are_imported_only_by_explicit_facades() -> None:
    allowed = {
        "nifty_scalper_bot.execution.order_manager_legacy": {
            "order_manager.py",
            "runtime_order_manager.py",
        },
        "nifty_scalper_bot.execution.legacy_bracket_manager": {
            "bracket_manager.py",
            "ownership.py",
        },
        "nifty_scalper_bot.execution.adaptive_trailing_legacy": {
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


def test_experimental_lifecycle_manager_is_not_in_production_import_graph() -> None:
    forbidden = "nifty_scalper_bot.execution.lifecycle_manager"
    offenders: list[str] = []
    for path in SRC.rglob("*.py"):
        if path.name == "lifecycle_manager.py":
            continue
        if forbidden in _imports(path):
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, offenders


def test_safe_order_manager_is_not_constructed_in_production_source() -> None:
    offenders: list[str] = []
    for path in SRC.rglob("*.py"):
        if path.name == "safe_order_manager.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "SafeOrderManager":
                    offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, offenders
