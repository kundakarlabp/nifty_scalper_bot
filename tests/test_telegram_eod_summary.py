from datetime import datetime
from types import SimpleNamespace

from src.config import settings
from src.notifications import telegram_controller as mod


def _prep(monkeypatch) -> None:
    monkeypatch.setattr(
        settings,
        "telegram",
        SimpleNamespace(bot_token="t", chat_id=1, enabled=True, extra_admin_ids=[]),
        raising=False,
    )


def test_eod_summary_sent_once_per_day(monkeypatch) -> None:
    _prep(monkeypatch)

    monkeypatch.setattr(
        mod,
        "daily_summary",
        lambda now: {
            "R": 1,
            "hit_rate": 55,
            "avg_R": 0.8,
            "slippage_bps": 3,
        },
    )

    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return cls(2024, 1, 1, 15, 45, tzinfo=tz)

    monkeypatch.setattr(mod, "datetime", _FixedDatetime)

    tc = mod.TelegramController(status_provider=lambda: {})
    sent: list[str] = []
    tc._send = lambda text, parse_mode=None: sent.append(text)

    tc.send_eod_summary()
    tc.send_eod_summary()

    assert sent == [
        "\U0001F514 EOD flat R=1 Hit=55% AvgR=0.8 Slip=3bps",
    ]

