from nifty_scalper_bot.strategies.runner import StrategyRunner


def test_runner_option_freshness_uses_canonical_quote_age_schema(monkeypatch) -> None:
    runner = object.__new__(StrategyRunner)
    runner._market_data = None
    runner._data_hub = None
    monkeypatch.setattr(runner, "_is_tradable_symbol", lambda symbol: True)
    monkeypatch.setattr(
        runner,
        "_get_cached_quote_for_live_entry",
        lambda symbol: {"quote_age_s": 0.5},
    )
    monkeypatch.setattr(runner, "get_quote", lambda symbol: {"quote_age_s": 0.5})

    assert runner._is_option_symbol_tick_fresh("NFO:NIFTY26MAY24000CE", max_age_s=1.0)
    assert runner._quote_fresh_for_symbol(
        "NFO:NIFTY26MAY24000CE", {"quote_age_s": 0.5}
    )
