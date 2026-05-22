import logging

from nifty_scalper_bot.core import app


def test_hydration_market_open_defaults_closed_on_exception(monkeypatch, caplog):
    def _raise() -> bool:
        raise RuntimeError("boom")

    monkeypatch.setattr(app, "is_market_open_session", _raise)

    with caplog.at_level(logging.WARNING):
        state = app._resolve_hydration_market_open_state(app.LOGGER)

    assert state is False
    assert any("MARKET_OPEN_CHECK_FAILED defaulting_closed" in r.message for r in caplog.records)


def test_hydration_market_open_before_session_does_not_crash(monkeypatch):
    monkeypatch.setattr(app, "is_market_open_session", lambda: False)
    state = app._resolve_hydration_market_open_state(app.LOGGER)
    assert state is False
