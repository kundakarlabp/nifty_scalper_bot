from __future__ import annotations

from types import SimpleNamespace

from nifty_scalper_bot.core.history_readiness import _get_cached_quote


def test_readiness_prefers_canonical_mdm_cached_tick_over_stale_datahub_copy() -> None:
    symbol = "NFO:NIFTY26AUGFUT"
    stale_datahub = {
        "symbol": symbol,
        "ltp": 25000.0,
        "timestamp": "2026-08-21T06:40:00+00:00",
        "marker": "stale_datahub",
    }
    fresh_mdm = {
        "symbol": symbol,
        "ltp": 25005.0,
        "timestamp": "2026-08-21T06:52:00+00:00",
        "marker": "fresh_mdm",
    }
    ctx = SimpleNamespace(
        data_hub=SimpleNamespace(
            get_quote=lambda _symbol, allow_pull=False: dict(stale_datahub)
        ),
        market_data_manager=SimpleNamespace(
            get_latest_tick=lambda _symbol: dict(fresh_mdm)
        ),
    )

    quote = _get_cached_quote(ctx, symbol)

    assert quote["marker"] == "fresh_mdm"


def test_readiness_falls_back_to_datahub_when_mdm_has_no_cached_tick() -> None:
    symbol = "NFO:NIFTY26AUGFUT"
    datahub_quote = {
        "symbol": symbol,
        "ltp": 25000.0,
        "timestamp": "2026-08-21T06:52:00+00:00",
        "marker": "datahub_fallback",
    }
    ctx = SimpleNamespace(
        data_hub=SimpleNamespace(
            get_quote=lambda _symbol, allow_pull=False: dict(datahub_quote)
        ),
        market_data_manager=SimpleNamespace(get_latest_tick=lambda _symbol: None),
    )

    quote = _get_cached_quote(ctx, symbol)

    assert quote["marker"] == "datahub_fallback"
