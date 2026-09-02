from types import SimpleNamespace

from nifty_scalper_bot.core.strategy_runner_dynamic_universe_safety import apply_patches
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


def test_live_option_freshness_prefers_genuine_ws_age_over_fresh_cached_quote(
    monkeypatch,
) -> None:
    """A poll/cache refresh must not make a stale genuine WS option tick fresh."""

    apply_patches()
    runner = object.__new__(StrategyRunner)
    runner._data_hub = None
    runner._market_data = SimpleNamespace(
        time_since_last_live_ws_tick=lambda _symbol: 120.0,
        time_since_last_tick=lambda _symbol: 0.1,
    )
    monkeypatch.setattr(runner, "_is_tradable_symbol", lambda symbol: True)
    monkeypatch.setattr(
        runner,
        "_resolve_execution_mode_snapshot",
        lambda: SimpleNamespace(is_live_mode=True),
    )
    monkeypatch.setattr(
        runner,
        "_get_cached_quote_for_live_entry",
        lambda symbol: {
            "quote_age_s": 0.1,
            "ltp": 100.0,
            "bid": 99.5,
            "ask": 100.5,
            "source": "poll",
        },
    )

    assert (
        runner._is_option_symbol_tick_fresh(
            "NFO:NIFTY26MAY24000CE", max_age_s=60.0
        )
        is False
    )
