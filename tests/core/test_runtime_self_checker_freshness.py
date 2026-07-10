from types import SimpleNamespace

from nifty_scalper_bot.core import app


class _Hub:
    def __init__(self) -> None:
        self._quotes = {
            "NSE:NIFTY": {},
            "NFO:NIFTY26JUL24000CE": {},
            "NFO:NIFTY26JUL24000PE": {},
            "NFO:NIFTY26JUL24500CE": {},
        }
        self._fresh = {
            "NSE:NIFTY",
            "NFO:NIFTY26JUL24000CE",
            "NFO:NIFTY26JUL24000PE",
        }

    def is_fresh(self, symbol: str, *, threshold_ms: float | None = None):
        ok = symbol in self._fresh
        return ok, {
            "reason": None if ok else "stale",
            "symbol": symbol,
            "threshold_ms": threshold_ms,
        }

    def get_active_contract_basket(self):
        return {
            "spot_symbol": "NSE:NIFTY",
            "selected_ce": "NFO:NIFTY26JUL24000CE",
            "selected_pe": "NFO:NIFTY26JUL24000PE",
            "option_symbols": list(self._quotes),
        }


def test_runtime_self_checker_ignores_stale_non_selected_option(monkeypatch) -> None:
    monkeypatch.setattr(app, "get_market_state", lambda: app.MarketState.OPEN)
    runner = SimpleNamespace(set_data_freshness_backoff=lambda *args, **kwargs: None)
    ctx = SimpleNamespace(
        data_hub=_Hub(),
        streamer=SimpleNamespace(_interval_s=0.7),
        market_data_manager=SimpleNamespace(hard_ready=lambda: True),
        strategy_runner=runner,
        active_contract_basket={
            "spot_symbol": "NSE:NIFTY",
            "selected_ce": "NFO:NIFTY26JUL24000CE",
            "selected_pe": "NFO:NIFTY26JUL24000PE",
        },
    )

    ok, detail, meta = app.RuntimeSelfChecker(ctx)._check_data_freshness()

    assert ok is True
    assert detail == "partial_stale_ignored"
    assert meta["stale_symbols"] == 1
    assert meta["critical_symbols"] == 3
