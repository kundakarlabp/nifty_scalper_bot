from types import SimpleNamespace

from src.strategies.scoring import compute_score
from src.notifications.telegram_controller import TelegramController
from src.strategies.runner import StrategyRunner
from src.config import settings


class DummyRunner:
    def __init__(self) -> None:
        self._score_items = {"a": 1.0, "b": -0.5, "c": 0.25}
        self._score_total = 0.75
        self.strategy_cfg = SimpleNamespace(min_signal_score=0.5)


def test_compute_score() -> None:
    si = compute_score({"a": 1.0, "b": 0.5}, {"a": 2.0, "b": 4.0, "c": 10.0})
    assert si.items == {"a": 2.0, "b": 2.0}
    assert si.total == 4.0


def test_telegram_score_breakdown(monkeypatch) -> None:
    monkeypatch.setattr(
        settings,
        "telegram",
        SimpleNamespace(bot_token="t", chat_id=1, enabled=True, extra_admin_ids=[]),
    )
    dummy = DummyRunner()
    monkeypatch.setattr(StrategyRunner, "get_singleton", classmethod(lambda cls: dummy))
    tc = TelegramController(status_provider=lambda: {})
    sent: list[str] = []
    tc._send = lambda text, parse_mode=None: sent.append(text)
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/score"}})
    msg = sent[0]
    assert "Score breakdown" in msg
    assert "total=0.75" in msg
    assert "a: 1.0" in msg
    assert "b: -0.5" in msg
