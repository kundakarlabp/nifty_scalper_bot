from nifty_scalper_bot.core.strategy_manager import StrategyManager


def test_strategy_manager_context_age_uses_canonical_quote_age_schema() -> None:
    assert StrategyManager._context_tick_age_seconds({"quote_age_s": 0.25}) == 0.25
    assert StrategyManager._context_tick_age_seconds({"tick_age_ms": 250}) == 0.25
    assert StrategyManager._context_tick_age_seconds({"tick_age_ms": "invalid"}) is None
