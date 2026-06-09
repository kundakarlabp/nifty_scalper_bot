from types import SimpleNamespace

from nifty_scalper_bot.core import app


def test_post_market_basket_refresh_skips_without_clearing_active_basket(monkeypatch) -> None:
    ctx = SimpleNamespace(
        active_contract_basket={"selected_ce": "NFO:NIFTY2660923250CE", "selected_pe": "NFO:NIFTY2660923250PE"},
        active_trading_universe={"selected_ce": "NFO:NIFTY2660923250CE", "selected_pe": "NFO:NIFTY2660923250PE"},
        basket_build_last_completed_mono=1000.0,
    )
    monkeypatch.setattr(app, "get_runtime_market_mode", lambda: "POST_MARKET")
    monkeypatch.setattr(app, "post_market_quiet_mode_enabled", lambda: True)
    monkeypatch.setattr(app, "post_market_basket_refresh_seconds", lambda: 900.0)
    monkeypatch.setattr(app.time_module, "monotonic", lambda: 1100.0)

    skip, remaining = app._should_skip_post_market_basket_refresh(ctx)

    assert skip is True
    assert int(remaining) == 800
    assert ctx.active_contract_basket["selected_ce"] == "NFO:NIFTY2660923250CE"


def test_post_market_basket_refresh_allows_after_interval(monkeypatch) -> None:
    ctx = SimpleNamespace(active_contract_basket={"selected_ce": "CE"}, basket_build_last_completed_mono=1000.0)
    monkeypatch.setattr(app, "get_runtime_market_mode", lambda: "POST_MARKET")
    monkeypatch.setattr(app, "post_market_quiet_mode_enabled", lambda: True)
    monkeypatch.setattr(app, "post_market_basket_refresh_seconds", lambda: 900.0)
    monkeypatch.setattr(app.time_module, "monotonic", lambda: 2000.0)

    skip, _remaining = app._should_skip_post_market_basket_refresh(ctx)

    assert skip is False
