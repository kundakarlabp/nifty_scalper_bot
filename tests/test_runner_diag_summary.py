"""Tests for StrategyRunner.get_compact_diag_summary risk gate messages."""

from typing import Any

from src.strategies.runner import StrategyRunner


class DummyTelegram:
    def send_message(self, msg: str, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - minimal stub
        pass


def _make_runner() -> StrategyRunner:
    return StrategyRunner(kite=None, telegram_controller=DummyTelegram())


def test_risk_gates_no_eval() -> None:
    runner = _make_runner()
    runner._last_flow_debug = {}
    summary = runner.get_compact_diag_summary()
    assert summary["status_messages"]["risk_gates"] == "no-eval"


def test_risk_gates_ok() -> None:
    runner = _make_runner()
    runner._last_flow_debug = {"risk_gates": {"a": True, "b": True}}
    summary = runner.get_compact_diag_summary()
    assert summary["status_messages"]["risk_gates"] == "ok"


def test_risk_gates_blocked() -> None:
    runner = _make_runner()
    runner._last_flow_debug = {"risk_gates": {"a": True, "b": False}}
    summary = runner.get_compact_diag_summary()
    assert summary["status_messages"]["risk_gates"] == "blocked"


def test_risk_gates_skipped() -> None:
    runner = _make_runner()
    runner._last_flow_debug = {"risk_gates": {"skipped": True, "a": False}}
    summary = runner.get_compact_diag_summary()
    assert summary["status_messages"]["risk_gates"] == "skipped"
