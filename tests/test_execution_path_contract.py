from __future__ import annotations

import ast
from pathlib import Path

from nifty_scalper_bot.execution.bracket_manager import BracketManager
from nifty_scalper_bot.execution.order_manager import OrderManager


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src" / "nifty_scalper_bot"


def test_no_stale_execution_import_strings() -> None:
    banned = (
        "order_execution_hub",
        "execution_router",
        "preflight_validator",
        "OrderExecutionHub",
        "ExecutionRouter",
        "PreflightValidator",
    )
    offenders: list[str] = []
    for path in SRC_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(token in text for token in banned):
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"Banned execution references present: {offenders}"


def test_runner_has_single_bracket_exit_callback_definition() -> None:
    runner_path = SRC_ROOT / "strategies" / "runner.py"
    tree = ast.parse(runner_path.read_text(encoding="utf-8"))
    names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    assert names.count("_on_bracket_exit_complete") == 1


def test_runtime_bracket_owner_exposes_required_exit_methods() -> None:
    required = {
        "on_tick",
        "_evaluate_exit_fast",
        "_fire_exits_batch",
        "_execute_exit",
        "_market_fallback_exit",
        "_reconcile_exit_state",
        "_close_bracket",
    }
    missing = {name for name in required if not callable(getattr(BracketManager, name, None))}
    assert not missing, f"Bracket runtime owner missing: {sorted(missing)}"


def test_runtime_order_owner_exposes_required_entry_methods() -> None:
    required = {
        "submit_trade_plan_result",
        "submit_trade_plan",
        "place_managed_order_result",
        "place_managed_order",
        "place_order",
        "set_unresolved_exit_provider",
    }
    missing = {name for name in required if not callable(getattr(OrderManager, name, None))}
    assert not missing, f"Order runtime owner missing: {sorted(missing)}"


def test_architecture_doc_declares_canonical_path_and_compatibility_layers() -> None:
    doc = (ROOT / "ARCHITECTURE_TRADING_PATH.md").read_text(encoding="utf-8")
    assert "StrategyRunner" in doc
    assert "RuntimeOrderManager" in doc
    assert "BoundBracketManager" in doc
    assert "HardenedAdaptiveTrailingController" in doc
    assert "Compatibility-only modules" in doc
    assert "Forbidden runtime layers" in doc


def test_runner_has_single_execute_order_definition() -> None:
    runner_path = SRC_ROOT / "strategies" / "runner.py"
    tree = ast.parse(runner_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "StrategyRunner":
            names = [
                n.name
                for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            assert names.count("_execute_order") == 1
            return
    raise AssertionError("StrategyRunner class not found")


def test_market_data_manager_has_single_depth_coercer() -> None:
    mdm_path = SRC_ROOT / "data" / "market_data_manager.py"
    tree = ast.parse(mdm_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "MarketDataManager":
            names = [
                n.name
                for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            assert names.count("_coerce_from_depth") == 1
            return
    raise AssertionError("MarketDataManager class not found")


def test_runner_execute_order_does_not_bypass_canonical_entry_api() -> None:
    runner_path = SRC_ROOT / "strategies" / "runner.py"
    tree = ast.parse(runner_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_execute_order":
            calls = {
                call.func.attr
                for call in ast.walk(node)
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            }
            assert "submit_trade_plan_result" in calls
            assert "place_order" not in calls
            assert "execute_market_order" not in calls
            return
    raise AssertionError("StrategyRunner._execute_order not found")
