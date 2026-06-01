from nifty_scalper_bot.strategies.elite_strategies.config_models import OrderFlowStrategyConfig
from nifty_scalper_bot.strategies.elite_strategies.order_flow import OrderFlowStrategy


def _ind(score_high: bool):
    buy_qty = 300 if score_high else 180
    sell_qty = 100
    return {
        "bid": 100.0,
        "ask": 100.5,
        "spread_pct": 0.5,
        "depth": {"buy": [{"quantity": buy_qty}], "sell": [{"quantity": sell_qty}]},
        "tick_direction": "UP",
        "direction_bias": "PE",
        "atr": 2.0,
        "data_age_seconds": 0.1,
        "context_age_seconds": 1.0,
        "tick_age_ms": 100,
        "quote_depth_valid": True,
        "tradable_quote": True,
        "is_selected_option": True,
        "strike_distance_from_atm": 0,
    }


def test_orderflow_direction_conflict_score_eight_remains_blocked(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    strategy = OrderFlowStrategy(OrderFlowStrategyConfig(enabled=True, quantity=1), indicator_engine=None)

    signal = strategy._evaluate_signal("NFO:NIFTY26MAY24000CE", _ind(score_high=False), current_price=100.2)

    assert signal is not None
    assert signal.metadata["trigger_conditions_met"] is False
    assert signal.metadata["trigger_block_reason"] == "direction_bias_conflict"
    assert signal.metadata["strategy_score"] == 8.0


def test_orderflow_high_conviction_conflict_override_requires_depth_tick_and_near_atm(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    strategy = OrderFlowStrategy(OrderFlowStrategyConfig(enabled=True, quantity=1), indicator_engine=None)

    signal = strategy._evaluate_signal("NFO:NIFTY26MAY24000CE", _ind(score_high=True), current_price=100.2)

    assert signal is not None
    assert signal.metadata["strategy_score"] >= 9.0
    assert signal.metadata["trigger_conditions_met"] is True
    assert signal.metadata["orderflow_conflict_override"] is True
    assert signal.metadata["conflict_override_reason"] == "high_conviction_depth_spread_tick_near_atm"


def test_orderflow_override_requested_vs_applied(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    strategy = OrderFlowStrategy(OrderFlowStrategyConfig(enabled=True, quantity=1), indicator_engine=None)
    indicators = _ind(score_high=True)
    indicators["tradable_quote"] = False

    signal = strategy._evaluate_signal("NFO:NIFTY26MAY24000CE", indicators, current_price=100.2)

    assert signal is not None
    assert signal.metadata["orderflow_conflict_override_requested"] is True
    assert signal.metadata["orderflow_conflict_override_applied"] is False
    assert signal.metadata["orderflow_conflict_override"] is False
    assert signal.metadata["conflict_override_reason"] == ""
    assert signal.metadata["trigger_conditions_met"] is False
    assert signal.metadata["trigger_block_reason"] == "tradable_quote_false"
