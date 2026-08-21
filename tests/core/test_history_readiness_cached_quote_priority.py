from __future__ import annotations

from types import SimpleNamespace

from nifty_scalper_bot.core.history_readiness import _get_cached_quote
from nifty_scalper_bot.execution.readiness import resolve_quote_age_seconds


def test_readiness_uses_canonical_mdm_tick_with_datahub_identity_stamp() -> None:
    symbol = "NFO:NIFTY26AUGFUT"
    stale_datahub = {
        "symbol": symbol,
        "ltp": 25000.0,
        "tick_age_ms": 120_000.0,
        "marker": "stale_datahub",
    }
    fresh_mdm = {
        "symbol": symbol,
        "ltp": 25005.0,
        "timestamp": "2026-08-21T06:52:00+00:00",
        "marker": "fresh_mdm",
    }
    stamp_calls: list[str] = []

    def stamp(_symbol: str, quote: dict) -> dict:
        stamp_calls.append(_symbol)
        return {**quote, "tick_age_ms": 25.0, "quote_identity_source": "test"}

    ctx = SimpleNamespace(
        data_hub=SimpleNamespace(
            get_quote=lambda _symbol, allow_pull=False: dict(stale_datahub),
            _stamp_quote_identity=stamp,
        ),
        market_data_manager=SimpleNamespace(
            get_latest_tick=lambda _symbol: dict(fresh_mdm)
        ),
    )

    quote = _get_cached_quote(ctx, symbol)

    assert quote["marker"] == "fresh_mdm"
    assert stamp_calls == [symbol]
    assert resolve_quote_age_seconds(quote) == 0.025


def test_readiness_does_not_call_mdm_get_quote_broker_surface() -> None:
    symbol = "NFO:NIFTY26AUGFUT"

    def forbidden_get_quote(_symbol: str) -> dict:
        raise AssertionError("readiness must remain cache-only")

    ctx = SimpleNamespace(
        data_hub=None,
        market_data_manager=SimpleNamespace(
            get_latest_tick=lambda _symbol: {
                "symbol": symbol,
                "ltp": 25005.0,
                "tick_age_s": 0.1,
            },
            get_quote=forbidden_get_quote,
        ),
    )

    quote = _get_cached_quote(ctx, symbol)

    assert quote["ltp"] == 25005.0


def test_readiness_falls_back_to_datahub_when_mdm_has_no_cached_tick() -> None:
    symbol = "NFO:NIFTY26AUGFUT"
    datahub_quote = {
        "symbol": symbol,
        "ltp": 25000.0,
        "tick_age_ms": 25.0,
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
