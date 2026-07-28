from nifty_scalper_bot.core.strategy_manager import StrategyManager


def test_strategy_manager_context_age_uses_canonical_quote_age_schema() -> None:
    assert StrategyManager._context_tick_age_seconds({"quote_age_s": 0.25}) == 0.25
    assert StrategyManager._context_tick_age_seconds({"tick_age_ms": 250}) == 0.25
    assert StrategyManager._context_tick_age_seconds({"tick_age_ms": "invalid"}) is None



def test_futures_context_neutral_values_do_not_create_direction() -> None:
    from nifty_scalper_bot.core.strategy_manager import StrategyManager

    manager = object.__new__(StrategyManager)
    manager._latest_context_snapshots = {}
    manager._update_context_snapshot(
        symbol="NFO:NIFTY26JULFUT",
        indicators={"close": 25000.0, "vwap": 25000.0, "ema_fast": 25000.0,
                    "ema_slow": 25000.0, "ema_50": 25000.0, "vwap_slope": 0.0,
                    "ema_slope": 0.0},
        role="futures_context",
    )
    snapshot = manager._latest_context_snapshots["futures_context"]
    assert snapshot["vwap_slope"] == 0.0
    assert snapshot["direction_bias"] is None


def test_futures_context_uses_same_evaluation_slope_only() -> None:
    from nifty_scalper_bot.core.strategy_manager import StrategyManager

    manager = object.__new__(StrategyManager)
    manager._latest_context_snapshots = {"futures_context": {"vwap": 100.0}}
    manager._update_context_snapshot(
        symbol="NFO:NIFTY26JULFUT",
        indicators={"close": 101.0, "vwap": 101.0, "vwap_slope": 0.001},
        role="futures_context",
    )
    snapshot = manager._latest_context_snapshots["futures_context"]
    assert snapshot["vwap_slope"] == 0.001
    assert snapshot["direction_bias"] == "CE"
    manager._update_context_snapshot(
        symbol="NFO:NIFTY26JULFUT",
        indicators={"close": 102.0, "vwap": 102.0},
        role="futures_context",
    )
    assert manager._latest_context_snapshots["futures_context"]["vwap_slope"] is None
