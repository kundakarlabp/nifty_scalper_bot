from __future__ import annotations

from src.main import _make_cmd_handler
from src.notifications.telegram_commands import TelegramCommands


class DummyRunner:
    """Minimal runner with pause/resume for command tests."""

    def __init__(self) -> None:
        self.paused = False

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False


def test_cmd_handler_pause_resume() -> None:
    runner = DummyRunner()
    handler = _make_cmd_handler(runner)
    handler("/pause", "")
    assert runner.paused is True
    handler("/resume", "")
    assert runner.paused is False


def test_telegram_commands_start_without_creds() -> None:
    tg = TelegramCommands(None, None)
    tg.start()
    assert tg._running is False
    tg.stop()
