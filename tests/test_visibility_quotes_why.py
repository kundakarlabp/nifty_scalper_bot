def test_quotes_why_show_micro_fields():
    plan = {
        "micro": {
            "source": "ltp_fallback",
            "ltp": 101,
            "bid": 100.5,
            "ask": 101.5,
            "spread_pct": 0.25,
            "bid5": 0,
            "ask5": 0,
            "depth_ok": True,
        },
        "last_bar_ts": "2025-08-29T09:40:00",
        "last_bar_lag_s": 42,
        "tsym": "NIFTY25AUG22500CE",
    }
    from src.notifications.telegram_controller import _fmt_micro

    s = _fmt_micro(plan["tsym"], plan["micro"], plan["last_bar_ts"], plan["last_bar_lag_s"])
    assert "src=ltp_fallback" in s and "spread%=0.25" in s and "lag_s=42" in s


def test_why_handles_missing_atr_and_score(monkeypatch):
    from types import SimpleNamespace

    from src.notifications.telegram_controller import TelegramController
    from src.config import settings

    monkeypatch.setattr(
        settings,
        "telegram",
        SimpleNamespace(bot_token="t", chat_id=1, enabled=True, extra_admin_ids=[]),
    )
    status = {
        "within_window": True,
        "daily_dd_hit": False,
        "cooloff_until": "-",
        "day_realized_loss": 0,
    }
    plan = {
        "bar_count": 5,
        "last_bar_ts": "2025-08-29T09:40:00",
        "regime": "TREND",
        "atr_min": 0.2,
    }
    tc = TelegramController(
        status_provider=lambda: status,
        last_signal_provider=lambda: plan,
    )
    sent: list[str] = []
    tc._send = lambda text, parse_mode=None: sent.append(text)
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/why gates"}})
    msg = sent[0]
    assert "atr_pct: FAIL N/A" in msg and "score: FAIL N/A" in msg
