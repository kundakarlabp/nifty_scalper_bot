from src.notifications.telegram_commands import TelegramCommands


def test_stop_joins_and_clears_thread() -> None:
    tc = TelegramCommands(bot_token="t", chat_id=1)
    joined: list[float | None] = []

    class DummyThread:
        def join(self, timeout: float | None = None) -> None:
            joined.append(timeout)

    tc._th = DummyThread()  # type: ignore[assignment]
    tc._running = True
    tc.stop()
    assert not tc._running
    assert joined == [1]
    assert tc._th is None
