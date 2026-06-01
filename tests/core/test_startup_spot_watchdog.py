from __future__ import annotations

from types import SimpleNamespace

from nifty_scalper_bot.core import app


class _RestFallbackMDM:
    def get_cached_ltp(self, *_args, **_kwargs):
        return 0

    def get_ltp(self, *_args, **_kwargs):
        raise RuntimeError("broker temporarily unavailable")

    def refresh_quote_now(self, _symbol: str):
        return {"last_price": 25000.0}


def test_rest_fallback_get_ltp_runtime_error_does_not_crash(caplog):
    ctx = SimpleNamespace(market_data_manager=_RestFallbackMDM())

    price = app._resolve_startup_rest_spot_ltp(ctx)

    assert price == 25000.0
    assert "STARTUP_SPOT_REST_FALLBACK_FAILED" in caplog.text
    assert "stage=get_ltp" in caplog.text
