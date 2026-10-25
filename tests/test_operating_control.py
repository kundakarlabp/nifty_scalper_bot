from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

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



def test_auto_watchdog_retries_rejected_restart_then_exits_on_accepted_queue(
    monkeypatch,
) -> None:
    """One failed sudo/systemd trigger must not miss the whole weekday window."""
    created = []
    sleeps = []
    attempts = []

    class FakeThread:
        def __init__(self, *, target, name, daemon):
            assert name == "operating-mode-watchdog"
            assert daemon is True
            self.run_watch = target
            created.append(self)

        def start(self):
            pass

    def fake_sleep(seconds):
        sleeps.append(seconds)
        assert len(sleeps) <= 2, "watchdog did not return after accepted queue"

    def fake_restart():
        attempts.append(True)
        return SimpleNamespace(ok=len(attempts) >= 2)

    monkeypatch.setattr(control.threading, "Thread", FakeThread)
    monkeypatch.setattr(control.time, "sleep", fake_sleep)
    monkeypatch.setattr(control, "operating_mode", lambda: "AUTO")
    monkeypatch.setattr(control, "engine_should_run", lambda: True)
    monkeypatch.setattr(control, "_restart_service", fake_restart)

    thread = control.start_transition_watchdog(False, interval_seconds=30)
    assert thread is created[0]
    thread.run_watch()
    assert sleeps == [30, 30]
    assert len(attempts) == 2


def test_auto_watchdog_never_queues_when_mode_does_not_transition(monkeypatch):
    """Heartbeat ticks with unchanged mode must not cause restart requests."""
    class StopWatch(Exception):
        pass

    class FakeThread:
        def __init__(self, *, target, **_kwargs):
            self.run_watch = target

        def start(self):
            pass

    ticks = []
    monkeypatch.setattr(control.threading, "Thread", FakeThread)
    monkeypatch.setattr(control, "operating_mode", lambda: "AUTO")
    monkeypatch.setattr(control, "engine_should_run", lambda: False)
    monkeypatch.setattr(
        control,
        "_restart_service",
        lambda: pytest.fail("restart must not be requested"),
    )

    def fake_sleep(seconds):
        ticks.append(seconds)
        if len(ticks) == 3:
            raise StopWatch()

    monkeypatch.setattr(control.time, "sleep", fake_sleep)
    with pytest.raises(StopWatch):
        control.start_transition_watchdog(False).run_watch()
    assert ticks == [30, 30, 30]
