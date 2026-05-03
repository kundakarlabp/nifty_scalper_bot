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
