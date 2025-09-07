"""Tests for StrategyRunner._within_trading_window respecting config times."""

from datetime import datetime

from src.strategies.runner import StrategyRunner


class DummyTelegram:
    def send_message(self, msg: str) -> None:  # pragma: no cover - not used
        pass


def test_within_window_respects_config(monkeypatch) -> None:
    runner = StrategyRunner(telegram_controller=DummyTelegram())

    # Before configured start time -> False
    monkeypatch.setattr(runner, "_now_ist", lambda: datetime(2024, 1, 1, 9, 14))
    assert runner._within_trading_window() is False

    # Inside configured window -> True
    monkeypatch.setattr(runner, "_now_ist", lambda: datetime(2024, 1, 1, 9, 16))
    assert runner._within_trading_window() is True

    # After configured end time -> False
    monkeypatch.setattr(runner, "_now_ist", lambda: datetime(2024, 1, 1, 15, 31))
    assert runner._within_trading_window() is False
