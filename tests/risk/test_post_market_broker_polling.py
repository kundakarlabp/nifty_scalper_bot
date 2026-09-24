from __future__ import annotations

import nifty_scalper_bot.risk.risk_manager as risk_module


def test_balance_refresher_keeps_open_market_interval(monkeypatch) -> None:
    monkeypatch.setattr(risk_module, "post_market_quiet_mode_enabled", lambda: True)
    monkeypatch.setattr(risk_module, "get_runtime_market_mode", lambda: "OPEN")
    monkeypatch.setattr(
        risk_module, "post_market_broker_refresh_seconds", lambda: 3600.0
    )

    assert risk_module._balance_refresher_sleep_seconds(60.0) == 60.0


def test_balance_refresher_throttles_only_post_market(monkeypatch) -> None:
    monkeypatch.setattr(risk_module, "post_market_quiet_mode_enabled", lambda: True)
    monkeypatch.setattr(risk_module, "get_runtime_market_mode", lambda: "POST_MARKET")
    monkeypatch.setattr(
        risk_module, "post_market_broker_refresh_seconds", lambda: 3600.0
    )

    assert risk_module._balance_refresher_sleep_seconds(60.0) == 3600.0


def test_balance_refresher_keeps_pre_market_interval(monkeypatch) -> None:
    monkeypatch.setattr(risk_module, "post_market_quiet_mode_enabled", lambda: True)
    monkeypatch.setattr(risk_module, "get_runtime_market_mode", lambda: "PRE_MARKET")
    monkeypatch.setattr(
        risk_module, "post_market_broker_refresh_seconds", lambda: 3600.0
    )

    assert risk_module._balance_refresher_sleep_seconds(60.0) == 60.0
