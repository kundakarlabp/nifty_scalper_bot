from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    OrderFlowStrategyConfig,
)
from nifty_scalper_bot.strategies.elite_strategies.order_flow import (
    OrderFlowStrategy,
    _context_confirmation_score,
)


def _strategy() -> OrderFlowStrategy:
    return OrderFlowStrategy(
        OrderFlowStrategyConfig(enabled=True, quantity=1), indicator_engine=None
    )


def test_orderflow_no_missing_depth_when_bid_ask_and_depth_exist() -> None:
    strategy = _strategy()
    indicators = {
        "bid": 100.0,
        "ask": 101.0,
        "spread_pct": 0.99,
        "depth": {"buy": [{"quantity": 200}], "sell": [{"quantity": 100}]},
        "tick_direction": "UP",
        "direction_bias": "CE",
        "atr": 2.0,
        "data_age_seconds": 0.2,
    }
    signal = strategy._evaluate_signal(
        "NFO:NIFTY26MAY24000CE", indicators, current_price=100.5
    )
    assert signal is not None
    assert getattr(strategy, "last_no_vote_reason", None) != "missing_depth"
    assert signal.metadata["context_evidence_score"] == 6.0
    assert signal.metadata["context_bonus_score"] == 3.0
    assert signal.metadata["depth_supports_side"] is True


def test_option_book_ask_pressure_does_not_earn_depth_confirmation() -> None:
    strategy = _strategy()
    indicators = {
        "bid": 100.0,
        "ask": 101.0,
        "spread_pct": 0.99,
        "depth": {"buy": [{"quantity": 100}], "sell": [{"quantity": 400}]},
        "tick_direction": "UP",
        "direction_bias": "CE",
        "atr": 2.0,
        "data_age_seconds": 0.2,
    }

    signal = strategy._evaluate_signal(
        "NFO:NIFTY26MAY24000CE", indicators, current_price=100.5
    )

    assert signal is not None
    assert signal.metadata["depth_imbalance"] < 0
    assert signal.metadata["depth_supports_side"] is False
    assert signal.metadata["depth_score"] == 0.0
    assert "depth_imbalance_support" not in signal.metadata["score_reasons"]
    assert "strong_depth_imbalance_support" not in signal.metadata["score_reasons"]
    assert signal.metadata["strategy_score"] == 8.0
    assert signal.metadata["context_evidence_score"] == 4.0
    assert signal.metadata["context_bonus_score"] == 2.0


def test_live_unready_quote_cannot_contribute_orderflow_context(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    strategy = _strategy()
    indicators = {
        "bid": 100.0,
        "ask": 101.0,
        "spread_pct": 0.99,
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
        "stale_data_used": False,
    }

    signal = strategy._evaluate_signal(
        "NFO:NIFTY26MAY24000CE", indicators, current_price=100.5
    )

    assert signal is not None
    assert signal.metadata["quote_readiness_allowed"] is False
    assert signal.metadata["quote_readiness_reason"] == "spread_too_wide"
    assert signal.metadata["context_quality_eligible"] is False
    assert signal.metadata["context_bonus_score"] == 0.0
    assert signal.metadata["context_veto_score"] == 0.0
    assert signal.metadata["trigger_conditions_met"] is False
    assert signal.metadata["trigger_block_reason"] == "context_only_role"


def test_orderflow_context_floor_does_not_reinforce_a_trigger() -> None:
    assert _context_confirmation_score(4.0, 4.0) == (0.0, 0.0)
    assert _context_confirmation_score(6.0, 4.0) == (2.0, 1.0)
    assert _context_confirmation_score(10.0, 4.0) == (6.0, 3.0)


from nifty_scalper_bot.strategies.indicators import IndicatorEngine


def test_runtime_context_depth_keys_are_preserved() -> None:
    engine = IndicatorEngine()
    engine.set_runtime_context(
        "NFO:NIFTY26MAY24000CE",
        {
            "depth": {
                "buy": [{"price": 10.0}],
                "sell": [{"price": 11.0}],
            },
            "depth_available": True,
            "bid": 10.0,
            "ask": 11.0,
        },
    )
    ctx = engine.get_runtime_context("NFO:NIFTY26MAY24000CE")
    assert ctx["depth_available"] is True
    assert ctx["depth"]["buy"][0]["price"] == 10.0
