from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    VWAPProStrategyConfig,
)
from nifty_scalper_bot.strategies.elite_strategies.vwap_pro import VWAPProStrategy


def test_vwap_pro_trend_context_boost_avoids_weak_score(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    strategy = VWAPProStrategy(
        VWAPProStrategyConfig(enabled=True, quantity=1), indicator_engine=None
    )
    symbol = "NFO:NIFTY26MAY24000CE"

    # A live VWAP thesis is edge-triggered. Establish the finalized reset first
    # rather than manufacturing a mid-session entry from an already-extended
    # above-VWAP state.
    assert (
        strategy._evaluate_signal(
            symbol,
            {
                "vwap": 100.0,
                "open": 100.2,
                "high": 100.4,
                "low": 99.0,
                "close": 99.5,
                "atr": 2.0,
                "direction_bias": "CE",
                "underlying_direction_bias": "CE",
                "latest_bar_ts": 1_785_000_000.0,
            },
            current_price=99.5,
        )
        is None
    )
    assert strategy.last_no_vote_reason == "vwap_thesis_reset"

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
        "latest_bar_ts": 1_785_000_060.0,
    }

    signal = strategy._evaluate_signal(symbol, indicators, current_price=109.0)

    assert signal is not None
    assert "trend_context_boost" in signal.metadata["score_reasons"]
    assert signal.metadata["strategy_score"] >= 5.5
    assert signal.metadata["vwap_domain"] == "option_premium"
    assert signal.metadata["invalidation_level_domain"] == "option_premium"
    assert signal.metadata["setup_invalidation_premium"] == 107.0
    assert "underlying_invalidation_level" not in signal.metadata
