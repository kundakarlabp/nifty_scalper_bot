from types import SimpleNamespace

from src.notifications.telegram_controller import TelegramController
from src.config import settings


def test_diag_shows_status_and_signal(monkeypatch):
    monkeypatch.setattr(
        settings,
        "telegram",
        SimpleNamespace(bot_token="t", chat_id=1, enabled=True, extra_admin_ids=[]),
    )
    status = {
        "market_open": True,
        "within_window": True,
        "daily_dd_hit": False,
        "cooloff_until": "-",
        "trades_today": 1,
        "consecutive_losses": 0,
    }
    plan = {
        "action": "BUY",
        "option_type": "CE",
        "strike": "100",
        "qty_lots": 1,
        "regime": "TREND",
        "score": 5,
        "rr": 2.0,
        "atr_pct": 0.5,
        "micro": {"spread_pct": 0.1, "depth_ok": True},
        "entry": 100.0,
        "sl": 95.0,
        "tp1": 102.0,
        "tp2": 104.0,
        "reason_block": None,
        "reasons": ["test"],
        "ts": "2024-01-01T00:00:00",
    }
    tc = TelegramController(
        status_provider=lambda: status,
        last_signal_provider=lambda: plan,
    )
    sent = []
    tc._send = lambda text, parse_mode=None: sent.append(text)
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/diag"}})
    msg = sent[0]
    assert "Status" in msg and "Signal" in msg
    assert "market_open: True" in msg
    assert "action: BUY" in msg
