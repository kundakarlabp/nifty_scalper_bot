from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from src.strategies.runner import StrategyRunner
from src.config import settings


class DummyTelegram:
    """Minimal telegram controller stub for StrategyRunner."""

    def __init__(self) -> None:
        self.messages: list[str] = []
        self.eod_calls = 0

    def send_message(self, msg: str) -> None:  # pragma: no cover - capture only
        self.messages.append(msg)

    def send_eod_summary(self) -> None:  # pragma: no cover - capture only
        self.eod_calls += 1


TZ = ZoneInfo("Asia/Kolkata")


def _tz_datetime_for(hhmm: str) -> datetime:
    hour, minute = map(int, hhmm.split(":"))
    return datetime(2024, 1, 1, hour, minute, tzinfo=TZ)


def test_no_entries_after_1520(monkeypatch) -> None:
    telegram = DummyTelegram()
    runner = StrategyRunner(telegram_controller=telegram)
    monkeypatch.setattr(settings, "enable_live_trading", True, raising=False)

    cutoff_dt = _tz_datetime_for(settings.risk.no_new_after_hhmm)
    monkeypatch.setattr(
        runner,
        "_now_ist",
        lambda: cutoff_dt + timedelta(minutes=1),
    )

    runner.process_tick(None)
    flow = runner.get_last_flow_debug()
    expected_tag = f"after_{cutoff_dt.strftime('%H%M')}"
    assert flow["reason_block"] == expected_tag


def test_eod_close_triggers(monkeypatch) -> None:
    telegram = DummyTelegram()
    runner = StrategyRunner(telegram_controller=telegram)
    monkeypatch.setattr(settings, "enable_live_trading", True, raising=False)

    flatten_dt = _tz_datetime_for(settings.risk.eod_flatten_hhmm)
    monkeypatch.setattr(
        runner,
        "_now_ist",
        lambda: flatten_dt,
    )
    monkeypatch.setattr(runner.order_executor, "open_count", 1, raising=False)

    called = {"close": 0, "cancel": 0}

    def fake_close() -> None:
        called["close"] += 1

    def fake_cancel() -> None:
        called["cancel"] += 1

    monkeypatch.setattr(
        runner.order_executor, "close_all_positions_eod", fake_close, raising=False
    )
    monkeypatch.setattr(
        runner.order_executor, "cancel_all_orders", fake_cancel, raising=False
    )

    runner.process_tick(None)
    flow = runner.get_last_flow_debug()
    expected_tag = f"after_{flatten_dt.strftime('%H%M')}"
    assert flow["reason_block"] == expected_tag
    assert called["close"] == 1
    assert called["cancel"] == 1
    assert telegram.eod_calls == 1
    assert any("EOD" in m for m in telegram.messages)
