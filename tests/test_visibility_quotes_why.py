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
