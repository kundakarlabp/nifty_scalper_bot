from datetime import datetime
from zoneinfo import ZoneInfo

import nifty_scalper_bot.operating_control as control

IST = ZoneInfo("Asia/Kolkata")


def _env(mode="AUTO"):
    return {
        "BOT_OPERATING_MODE": mode,
        "BOT_AUTO_START_IST": "08:55",
        "BOT_AUTO_STOP_IST": "15:40",
    }


def test_auto_window_weekday(monkeypatch):
    monkeypatch.setattr(control, "_read_env", lambda: _env())
    assert control.engine_should_run(datetime(2026, 9, 16, 9, 0, tzinfo=IST)) is True
    assert control.engine_should_run(datetime(2026, 9, 16, 7, 0, tzinfo=IST)) is False
    assert control.engine_should_run(datetime(2026, 9, 16, 15, 40, tzinfo=IST)) is False


def test_auto_is_quiet_on_weekend(monkeypatch):
    monkeypatch.setattr(control, "_read_env", lambda: _env())
    assert control.engine_should_run(datetime(2026, 9, 19, 10, 0, tzinfo=IST)) is False


def test_manual_modes_override_schedule(monkeypatch):
    monkeypatch.setattr(control, "_read_env", lambda: _env("ACTIVE"))
    assert control.engine_should_run(datetime(2026, 9, 20, 2, 0, tzinfo=IST)) is True
    monkeypatch.setattr(control, "_read_env", lambda: _env("QUIET"))
    assert control.engine_should_run(datetime(2026, 9, 16, 10, 0, tzinfo=IST)) is False
