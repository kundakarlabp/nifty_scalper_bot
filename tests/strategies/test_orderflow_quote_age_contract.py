from __future__ import annotations

from nifty_scalper_bot.strategies.elite_strategies.config_models import OrderFlowStrategyConfig
from nifty_scalper_bot.strategies.elite_strategies.order_flow import OrderFlowStrategy


def test_orderflow_accepts_quote_age_seconds_schema_in_live_mode(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    strategy = OrderFlowStrategy(OrderFlowStrategyConfig(enabled=True, quantity=1), indicator_engine=None)
    indicators = {
        "bid": 100.0,
        "ask": 100.25,
        "spread_pct": 0.24,
        "depth": {"buy": [{"quantity": 400}], "sell": [{"quantity": 80}]},
        "tick_direction": "UP",
        "direction_bias": "CE",
        "atr": 2.0,
        "quote_age_s": 0.10,
        "context_age_seconds": 0.10,
        "quote_depth_valid": True,
        "tradable_quote": True,
        "is_selected_option": True,
        "strike_distance_from_atm": 0,
        "quote_update_version": 1,
    }

    signal = strategy._evaluate_signal("NFO:NIFTY26MAY24000CE", indicators, current_price=100.1)

    assert signal is not None
    assert signal.metadata["quote_readiness_reason"] == "ready"
    assert signal.metadata["tick_age_ms"] == 100
    assert signal.metadata["trigger_block_reason"] != "tick_age_missing"
    assert signal.metadata["trigger_conditions_met"] is True


def test_orderflow_ltp_fallback_rejects_unknown_quote_age(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("ORDERFLOW_ALLOW_LTP_TICK_FALLBACK", "true")
    monkeypatch.setenv("ORDERFLOW_ALLOW_LTP_FALLBACK_TRIGGER", "true")
    strategy = OrderFlowStrategy(OrderFlowStrategyConfig(enabled=True, quantity=1), indicator_engine=None)
    indicators = {
        "bid": 100.0,
        "ask": 100.25,
        "spread_pct": 0.24,
        "depth": {},
        "tick_direction": "UP",
        "direction_bias": "CE",
        "atr": 2.0,
    }

    signal = strategy._evaluate_signal("NFO:NIFTY26MAY24000CE", indicators, current_price=100.1)

    assert signal is None
    assert strategy.last_no_vote_reason == "stale_tick_for_ltp_fallback"
