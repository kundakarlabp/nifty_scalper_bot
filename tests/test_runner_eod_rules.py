from datetime import datetime
from zoneinfo import ZoneInfo

from src.strategies.runner import StrategyRunner
from src.config import settings


class DummyTelegram:
    """Minimal telegram controller stub for StrategyRunner."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def send_message(self, msg: str) -> None:  # pragma: no cover - capture only
        self.messages.append(msg)


TZ = ZoneInfo("Asia/Kolkata")


def test_no_entries_after_1520(monkeypatch) -> None:
    telegram = DummyTelegram()
    runner = StrategyRunner(telegram_controller=telegram)
    monkeypatch.setattr(settings, "enable_live_trading", True, raising=False)

    monkeypatch.setattr(
        runner,
        "_now_ist",
        lambda: datetime(2024, 1, 1, 15, 21, tzinfo=TZ),
    )

    runner.process_tick(None)
    flow = runner.get_last_flow_debug()
    assert flow["reason_block"] == "after_1520"


def test_eod_close_triggers(monkeypatch) -> None:
    telegram = DummyTelegram()
    runner = StrategyRunner(telegram_controller=telegram)
    monkeypatch.setattr(settings, "enable_live_trading", True, raising=False)

    monkeypatch.setattr(
        runner,
        "_now_ist",
        lambda: datetime(2024, 1, 1, 15, 25, tzinfo=TZ),
    )
    monkeypatch.setattr(runner.order_executor, "open_count", 1, raising=False)

    called = {"n": 0}

    def fake_close() -> None:
        called["n"] += 1

    monkeypatch.setattr(
        runner.order_executor, "close_all_positions_eod", fake_close, raising=False
    )

    runner.process_tick(None)
    flow = runner.get_last_flow_debug()
    assert flow["reason_block"] == "after_1525"
    assert called["n"] == 1
