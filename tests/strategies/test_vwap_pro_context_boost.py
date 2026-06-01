from nifty_scalper_bot.strategies.elite_strategies.config_models import VWAPProStrategyConfig
from nifty_scalper_bot.strategies.elite_strategies.vwap_pro import VWAPProStrategy


def test_vwap_pro_trend_context_boost_avoids_weak_score(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    strategy = VWAPProStrategy(VWAPProStrategyConfig(enabled=True, quantity=1), indicator_engine=None)
    indicators = {
        "vwap": 100.0,
        "open": 108.9,
        "high": 109.4,
        "low": 108.5,
        "close": 109.0,
        "atr": 2.0,
        "volume": 0,
        "avg_volume": 0,
        "direction_bias": "CE",
        "underlying_direction_bias": "CE",
        "underlying_direction_confidence": 0.95,
        "context_age_seconds": 1.0,
        "spread_pct": 0.29,
    }

    signal = strategy._evaluate_signal("NFO:NIFTY26MAY24000CE", indicators, current_price=109.0)

    assert signal is not None
    assert "trend_context_boost" in signal.metadata["score_reasons"]
    assert signal.metadata["strategy_score"] >= 5.5
