from __future__ import annotations

from nifty_scalper_bot import main


def test_health_reports_degraded_when_startup_error_exists() -> None:
    """Verify health endpoint reports degraded mode after startup failures."""
    main.app.state.bot = None
    main.app.state.bot_error = 'boom'

    payload = main.health()

    assert payload['status'] == 'degraded'
    assert payload['error'] == 'boom'


def test_trading_status_reports_degraded_when_startup_error_exists(monkeypatch) -> None:
    """Verify trading status surfaces degraded bot state without crashing."""
    monkeypatch.setenv('ENABLE_LIVE', 'true')
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    main.app.state.bot = None
    main.app.state.bot_error = 'startup failed'

    payload = main.trading_status()

    assert payload['bot_status'] == 'degraded'
    assert payload['will_trade'] is True
