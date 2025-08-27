"""Test StrategyRunner behavior outside trading hours."""

from __future__ import annotations

from src.strategies.runner import StrategyRunner
from src.config import settings


class DummyTelegram:
    """Minimal telegram controller stub for StrategyRunner."""

    def send_message(self, msg: str) -> None:  # pragma: no cover - no behavior
        pass


def test_offhours_skips_risk_gates(monkeypatch) -> None:
    """Risk gates are marked as skipped when outside trading hours."""

    runner = StrategyRunner(telegram_controller=DummyTelegram())

    # Simulate being outside the trading window and ensure off-hours testing is disabled
    monkeypatch.setattr(runner, "_within_trading_window", lambda: False)
    monkeypatch.setattr(settings, "allow_offhours_testing", False, raising=False)

    runner.process_tick(None)
    summary = runner.get_compact_diag_summary()
    flow = runner.get_last_flow_debug()

    assert summary["status_messages"]["risk_gates"] == "skipped"
    assert flow["reason_block"] == "off_hours"

