from types import SimpleNamespace

from nifty_scalper_bot.core import app


class _Hub:
    def __init__(self, fresh: set[str] | None = None) -> None:
        self._quotes = {
            "NSE:NIFTY": {},
            "NFO:NIFTY26JUL24000CE": {},
            "NFO:NIFTY26JUL24000PE": {},
            "NFO:NIFTY26JUL24500CE": {},
        }
        self._fresh = (
            {
                "NSE:NIFTY",
                "NFO:NIFTY26JUL24000CE",
                "NFO:NIFTY26JUL24000PE",
            }
            if fresh is None
            else set(fresh)
        )

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


def _ctx_for(hub: _Hub, *, hard_ready: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        data_hub=hub,
        streamer=SimpleNamespace(_interval_s=0.7),
        market_data_manager=SimpleNamespace(hard_ready=lambda: hard_ready),
        strategy_runner=SimpleNamespace(
            set_data_freshness_backoff=lambda *args, **kwargs: None
        ),
        active_contract_basket={
            "spot_symbol": "NSE:NIFTY",
            "selected_ce": "NFO:NIFTY26JUL24000CE",
            "selected_pe": "NFO:NIFTY26JUL24000PE",
        },
    )


def test_fresh_non_selected_option_cannot_mask_stale_selected_contracts(
    monkeypatch,
) -> None:
    """A stale selected contract must not be excused by an unrelated fresh symbol.

    The verdict previously came from ``fresh[0]`` over every DataHub-tracked
    symbol, so a far strike that happened to tick reported the whole feed
    healthy while both traded contracts and spot were stale.
    """
    monkeypatch.setattr(app, "get_market_state", lambda: app.MarketState.OPEN)
    hub = _Hub(fresh={"NFO:NIFTY26JUL24500CE"})

    ok, detail, meta = app.RuntimeSelfChecker(_ctx_for(hub))._check_data_freshness()

    assert ok is False
    assert detail == "stale"
    assert meta["symbol_checked"] in {
        "NSE:NIFTY",
        "NFO:NIFTY26JUL24000CE",
        "NFO:NIFTY26JUL24000PE",
    }


def test_stale_verdict_reports_every_critical_symbol_age(monkeypatch) -> None:
    """A stale verdict must name the critical symbols and thresholds it used."""
    monkeypatch.setattr(app, "get_market_state", lambda: app.MarketState.OPEN)
    hub = _Hub(fresh=set())

    ok, _detail, meta = app.RuntimeSelfChecker(_ctx_for(hub))._check_data_freshness()

    assert ok is False
    critical_detail = meta["critical_detail"]
    assert isinstance(critical_detail, list)
    reported = {str(item["symbol"]) for item in critical_detail}
    assert reported == {
        "NSE:NIFTY",
        "NFO:NIFTY26JUL24000CE",
        "NFO:NIFTY26JUL24000PE",
    }
    for item in critical_detail:
        assert "threshold_ms" in item
        assert "fresh" in item


def test_failed_check_log_names_the_symbols_it_judged(monkeypatch, caplog) -> None:
    """The failure line operators read must carry the per-symbol evidence."""
    monkeypatch.setattr(app, "get_market_state", lambda: app.MarketState.OPEN)
    hub = _Hub(fresh=set())

    with caplog.at_level("ERROR"):
        app.RuntimeSelfChecker(_ctx_for(hub)).run_full_check()

    failures = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "RUNTIME_SELF_CHECK_FAILED"
        and getattr(record, "check", None) == "data_freshness"
    ]
    assert failures
    evidence = getattr(failures[0], "evidence", "")
    assert "NFO:NIFTY26JUL24000CE" in evidence
    assert "stale" in evidence
    assert "evidence=%s" in failures[0].msg


def test_partial_stale_tolerated_without_mdm_hard_ready(monkeypatch) -> None:
    """Partial staleness tolerance must not depend on the MDM hard_ready flag."""
    monkeypatch.setattr(app, "get_market_state", lambda: app.MarketState.OPEN)
    hub = _Hub()

    ok, detail, _meta = app.RuntimeSelfChecker(
        _ctx_for(hub, hard_ready=False)
    )._check_data_freshness()

    assert ok is True
    assert detail == "partial_stale_ignored"
