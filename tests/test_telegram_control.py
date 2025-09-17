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


class _DummyTelegramController:
    def __init__(self) -> None:
        self._chat_id = 7
        self.calls: list[dict] = []

    def _handle_update(self, update):
        self.calls.append(update)


class RunnerWithController(DummyRunner):
    def __init__(self) -> None:
        super().__init__()
        self.telegram_controller = _DummyTelegramController()


def test_cmd_handler_delegates_to_telegram_controller() -> None:
    runner = RunnerWithController()
    handler = _make_cmd_handler(runner)

    handler("/risk", "now")

    assert runner.telegram_controller.calls == [
        {"message": {"chat": {"id": 7}, "text": "/risk now"}}
    ]
