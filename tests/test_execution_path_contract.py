from __future__ import annotations

import ast
from pathlib import Path


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


def test_bracket_manager_owns_required_exit_methods() -> None:
    bracket_path = SRC_ROOT / "execution" / "bracket_manager.py"
    tree = ast.parse(bracket_path.read_text(encoding="utf-8"))
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    required = {"on_tick", "_evaluate_exit_fast", "_fire_exits_batch", "_execute_exit", "_market_fallback_exit"}
    assert required.issubset(names)


def test_order_manager_owns_required_entry_methods() -> None:
    order_path = SRC_ROOT / "execution" / "order_manager.py"
    tree = ast.parse(order_path.read_text(encoding="utf-8"))
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    required = {"submit_trade_plan", "place_managed_order", "place_order"}
    assert required.issubset(names)


def test_architecture_doc_mentions_core_path_and_forbidden_layers() -> None:
    doc = (ROOT / "ARCHITECTURE_TRADING_PATH.md").read_text(encoding="utf-8")
    assert "StrategyRunner" in doc
    assert "OrderManager" in doc
    assert "BracketManager" in doc
    assert "Forbidden removed layers" in doc


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


def test_bracket_exit_callback_is_full_close_only() -> None:
    bracket_path = SRC_ROOT / "execution" / "bracket_manager.py"
    text = bracket_path.read_text(encoding="utf-8")
    assert "fully_closed = bracket.remaining_quantity <= 0" in text
    assert "if fully_closed:" in text
    assert "BRACKET_EXIT_COMPLETE_HOOK_FAILED" in text


async def test_runner_live_entry_uses_trade_plan_path_not_legacy_price_ladder() -> None:
    """Prevent audits or refactors from confusing a side path with live entry."""

    runner_path = SRC_ROOT / "strategies" / "runner.py"
    tree = ast.parse(runner_path.read_text(encoding="utf-8"))
    execute_order: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "StrategyRunner":
            continue
        execute_order = next(
            (
                child
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and child.name == "_execute_order"
            ),
            None,
        )
        break

    assert execute_order is not None
    called_attributes = {
        call.func.attr
        for call in ast.walk(execute_order)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
    }
    assert "submit_trade_plan_result" in called_attributes
    assert "execute_market_order" not in called_attributes


async def test_unused_lifecycle_engines_are_not_runtime_imports() -> None:
    """Keep test-only lifecycle prototypes out of the production import graph."""

    forbidden_modules = {
        "nifty_scalper_bot.execution.dynamic_tp",
        "nifty_scalper_bot.execution.lifecycle_manager",
    }
    offenders: list[str] = []
    ignored = {
        SRC_ROOT / "execution" / "dynamic_tp.py",
        SRC_ROOT / "execution" / "lifecycle_manager.py",
    }

    for path in SRC_ROOT.rglob("*.py"):
        if path in ignored:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported = {str(node.module or "")}
            else:
                continue
            if imported.intersection(forbidden_modules):
                offenders.append(str(path.relative_to(ROOT)))
                break

    assert not offenders, f"Test-only lifecycle engines imported at runtime: {offenders}"
