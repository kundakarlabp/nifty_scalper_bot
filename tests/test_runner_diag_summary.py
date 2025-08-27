"""Diagnostics around StrategyRunner's summary endpoints."""

from __future__ import annotations

from src.strategies.runner import StrategyRunner
from src.config import settings


class DummyTelegram:
    def send_message(self, msg: str) -> None:  # pragma: no cover - no behavior
        pass


def test_skipped_gates_consistency(monkeypatch) -> None:
    """Skipped risk gates appear as skipped in both summary endpoints."""

    runner = StrategyRunner(telegram_controller=DummyTelegram())
    monkeypatch.setattr(runner, "_within_trading_window", lambda: False)
    monkeypatch.setattr(settings, "allow_offhours_testing", False, raising=False)

    runner.process_tick(None)

    bundle = runner._build_diag_bundle()
    summary = runner.get_compact_diag_summary()

    risk_check = next(c for c in bundle["checks"] if c["name"] == "Risk gates")
    assert risk_check["ok"] is True
    assert risk_check["detail"] == "skipped"
    assert summary["status_messages"]["risk_gates"] == "skipped"


def test_non_dict_risk_gates_handled() -> None:
    """Non-dict risk gate info should not break summary generation."""

    runner = StrategyRunner(telegram_controller=DummyTelegram())
    runner._last_flow_debug = {"risk_gates": ["bad"], "bars": 0}

    summary = runner.get_compact_diag_summary()

    assert summary["status_messages"]["risk_gates"] == "no-eval"

