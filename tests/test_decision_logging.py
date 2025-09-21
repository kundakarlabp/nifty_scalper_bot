from __future__ import annotations

import ast
import logging
from pathlib import Path

import pytest

from src.strategies.runner import StrategyRunner


class DummyTelegram:
    def send_message(self, msg: str) -> None:  # pragma: no cover - interface stub
        pass


def test_decisive_event_logging_includes_shared_fields(
    caplog: pytest.LogCaptureFixture,
) -> None:
    runner = StrategyRunner(telegram_controller=DummyTelegram())
    runner.last_eval_ts = "2024-01-01T00:00:00+00:00"
    runner.eval_count = 42
    plan = {
        "action": "BUY",
        "option_type": "CE",
        "symbol": "NIFTY24APR18000CE",
        "qty_lots": 2,
        "score": 1.2,
        "atr_pct": 0.45,
        "atr_pct_raw": 0.451,
        "rr": 2.5,
        "sl": 80.0,
        "tp1": 120.0,
        "tp2": 150.0,
        "micro": {"spread_pct": 0.12, "depth_ok": True, "block_reason": "micro"},
        "reason_block": "micro",
        "reasons": ["micro:depth"],
    }

    with caplog.at_level(logging.INFO, logger="src.strategies.runner"):
        runner._log_decisive_event(
            label="blocked",
            signal=dict(plan),
            reason_block="micro",
        )

    record = next((rec for rec in caplog.records if rec.message == "decision"), None)
    assert record is not None, "Expected decision log entry"

    assert getattr(record, "label", None) == "blocked"
    assert getattr(record, "reason_block", None) == "micro"

    plan_payload = getattr(record, "plan", None)
    assert isinstance(plan_payload, dict)
    assert plan_payload["action"] == "BUY"
    assert plan_payload["option_type"] == "CE"
    assert plan_payload["qty_lots"] == 2
    assert plan_payload["symbol"] == "NIFTY24APR18000CE"
    assert plan_payload["atr_pct_raw"] == pytest.approx(0.451)
    assert plan_payload["eval_count"] == 42
    assert plan_payload["last_eval_ts"] == "2024-01-01T00:00:00+00:00"
    assert plan_payload["micro"]["spread_pct"] == pytest.approx(0.12)
    assert plan_payload["micro"]["depth_ok"] is True
    assert plan_payload["reasons"] == ["micro:depth"]


def test_process_tick_returns_emit_decisive_logs() -> None:
    module = ast.parse(Path("src/strategies/runner.py").read_text())

    parents: dict[ast.AST, ast.AST] = {}

    def _walk(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            parents[child] = node
            _walk(child)

    _walk(module)

    runner_class = next(
        (node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "StrategyRunner"),
        None,
    )
    assert runner_class is not None, "StrategyRunner definition not found"

    process_tick = next(
        (
            node
            for node in runner_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "process_tick"
        ),
        None,
    )
    assert process_tick is not None, "process_tick definition not found"

    def _enclosing_function(node: ast.AST) -> ast.FunctionDef | None:
        current = node
        while current in parents:
            current = parents[current]
            if isinstance(current, ast.FunctionDef):
                return current
        return None

    def _statements_for_parent(parent: ast.AST, child: ast.stmt) -> list[ast.stmt] | None:
        for field in ("body", "orelse", "finalbody"):
            stmts = getattr(parent, field, None)
            if isinstance(stmts, list) and child in stmts:
                return stmts
        if isinstance(parent, ast.ExceptHandler):
            if child in parent.body:
                return parent.body
        return None

    missing_logs: list[int] = []

    for node in ast.walk(process_tick):
        if not isinstance(node, ast.Return):
            continue
        enclosing = _enclosing_function(node)
        if enclosing is not process_tick:
            continue
        parent = parents.get(node)
        while parent is not None and _statements_for_parent(parent, node) is None:
            parent = parents.get(parent)
        if parent is None:
            continue
        statements = _statements_for_parent(parent, node)
        if not statements:
            continue
        idx = statements.index(node)
        prior = statements[:idx]
        if not any(
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Attribute)
            and stmt.value.func.attr == "_log_decisive_event"
            for stmt in reversed(prior)
        ):
            missing_logs.append(node.lineno)

    assert not missing_logs, f"Returns without decisive event logging: {missing_logs}"
