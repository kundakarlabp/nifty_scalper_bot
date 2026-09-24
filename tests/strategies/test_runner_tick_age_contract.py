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
    assert runner._quote_fresh_for_symbol("NFO:NIFTY26MAY24000CE", {"quote_age_s": 0.5})


def test_runtime_quote_age_prefers_genuine_live_ws_age(monkeypatch) -> None:
    runner = object.__new__(StrategyRunner)
    runner._data_hub = None
    runner._market_data = SimpleNamespace(
        time_since_last_live_ws_tick=lambda _symbol: 0.125,
    )
    monkeypatch.setattr(
        runner,
        "_resolve_execution_mode_snapshot",
        lambda: SimpleNamespace(is_live_mode=True),
    )

    assert (
        runner._runtime_quote_age_seconds(
            "NFO:NIFTY26MAY24000CE",
            {"bid": 99.5, "ask": 100.5},
            {},
        )
        == 0.125
    )


def test_runtime_quote_age_live_ws_missing_fails_closed(monkeypatch) -> None:
    runner = object.__new__(StrategyRunner)
    runner._data_hub = None
    runner._market_data = SimpleNamespace(
        time_since_last_live_ws_tick=lambda _symbol: None,
    )
    monkeypatch.setattr(
        runner,
        "_resolve_execution_mode_snapshot",
        lambda: SimpleNamespace(is_live_mode=True),
    )

    assert (
        runner._runtime_quote_age_seconds(
            "NFO:NIFTY26MAY24000CE",
            {"quote_age_s": 0.1},
            {},
        )
        is None
    )


def test_runtime_quote_age_shadow_uses_canonical_quote_age(monkeypatch) -> None:
    runner = object.__new__(StrategyRunner)
    runner._data_hub = None
    runner._market_data = None
    monkeypatch.setattr(
        runner,
        "_resolve_execution_mode_snapshot",
        lambda: SimpleNamespace(is_live_mode=False),
    )

    assert (
        runner._runtime_quote_age_seconds(
            "NFO:NIFTY26MAY24000CE",
            {"quote_age_s": 0.25},
            {},
        )
        == 0.25
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
        runner._is_option_symbol_tick_fresh("NFO:NIFTY26MAY24000CE", max_age_s=60.0)
        is False
    )



def test_runner_get_quote_never_allows_implicit_pull() -> None:
    symbol = "NFO:NIFTY26SEP23100CE"

    class _Hub:
        def __init__(self) -> None:
            self.allow_pull_calls: list[bool] = []

        def get_quote(
            self,
            _symbol: str,
            allow_pull: bool = True,
        ) -> dict[str, object]:
            self.allow_pull_calls.append(bool(allow_pull))
            return {
                "symbol": symbol,
                "ltp": 100.0,
                "bid": 99.9,
                "ask": 100.1,
                "depth": {
                    "buy": [{"price": 99.9, "quantity": 65}],
                    "sell": [{"price": 100.1, "quantity": 65}],
                },
            }

    hub = _Hub()
    runner = object.__new__(StrategyRunner)
    runner._data_hub = hub
    runner._market_data = None

    quote = runner.get_quote(symbol)

    assert quote is not None
    assert quote["depth"]["buy"][0]["quantity"] == 65
    assert hub.allow_pull_calls == [False]
