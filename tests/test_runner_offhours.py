"""Test StrategyRunner behavior outside trading hours."""

from __future__ import annotations

from src.strategies.runner import StrategyRunner
from src.config import settings


class DummyTelegram:
    """Minimal telegram controller stub for StrategyRunner."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def send_message(self, msg: str) -> None:  # pragma: no cover - capture only
        self.messages.append(msg)


def test_offhours_skips_risk_gates(monkeypatch) -> None:
    """Risk gates are marked as skipped when outside trading hours."""

    telegram = DummyTelegram()
    runner = StrategyRunner(telegram_controller=telegram)

    # Simulate being outside the trading window and ensure off-hours testing is disabled
    monkeypatch.setattr(runner, "_within_trading_window", lambda: False)
    monkeypatch.setattr(settings, "allow_offhours_testing", False, raising=False)

    runner.process_tick(None)
    summary = runner.get_compact_diag_summary()
    flow = runner.get_last_flow_debug()

    assert summary["status_messages"]["risk_gates"] == "skipped"
    assert flow["reason_block"] == "off_hours"
    assert any("outside trading window" in m for m in telegram.messages)


def test_offhours_notifies_only_once(monkeypatch) -> None:
    """Repeated ticks outside window trigger a single notification."""

    telegram = DummyTelegram()
    runner = StrategyRunner(telegram_controller=telegram)

    monkeypatch.setattr(runner, "_within_trading_window", lambda: False)
    monkeypatch.setattr(settings, "allow_offhours_testing", False, raising=False)

    runner.process_tick(None)
    runner.process_tick(None)

    messages = [m for m in telegram.messages if "outside trading window" in m]
    assert len(messages) == 1

