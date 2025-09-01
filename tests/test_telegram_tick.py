from types import SimpleNamespace

from src.notifications.telegram_controller import TelegramController
from src.config import settings


def test_tick_runs_runner_and_l1(monkeypatch) -> None:
    monkeypatch.setattr(
        settings,
        "telegram",
        SimpleNamespace(bot_token="t", chat_id=1, enabled=True, extra_admin_ids=[]),
        raising=False,
    )
    called: dict[str, bool] = {"tick": False}

    def fake_tick(*, dry: bool = False) -> dict[str, bool]:
        called["tick"] = True
        return {"ok": True}

    tc = TelegramController(
        status_provider=lambda: {}, runner_tick=fake_tick, l1_provider=lambda: {"bid": 1}
    )
    sent: list[str] = []
    tc._send = lambda text, parse_mode=None: sent.append(text)

    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/tick"}})
    assert called["tick"] is True
    assert sent and "Tick executed" in sent[0]

    sent.clear()
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/l1"}})
    assert sent and sent[0].startswith("L1: ")
