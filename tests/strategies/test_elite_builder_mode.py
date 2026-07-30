from __future__ import annotations

from nifty_scalper_bot.strategies.elite_strategies.builder import build_elite_strategies
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    EliteStrategiesSettings,
    ORBProStrategyConfig,
)
from nifty_scalper_bot.strategies.elite_strategies.orb_pro import ORBProStrategy


def test_directional_mode_disables_gamma_theta_and_context(monkeypatch) -> None:
    monkeypatch.setenv("STRATEGY_MODE", "directional_scalp")
    monkeypatch.setenv("ALLOW_EXPIRY_GAMMA_STRATEGIES", "false")

    strategies = build_elite_strategies(
        settings=EliteStrategiesSettings(),
        indicator_engine=None,
    )
    names = {strategy.name for strategy in strategies}

    assert "GammaScalping" not in names
    assert "EliteTuesdayGammaBuyer" not in names
    assert "StraddleTheta" not in names
    assert "OIMaxPain" not in names
    assert "BBSqueeze" not in names
    assert {"SMC", "VWAPPro", "OrderFlow", "ORBPro"}.issubset(names)
    assert "CPRBreakout" not in names


def test_orb_direction_is_consumed_only_after_entry_acceptance(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ENABLE_ORB_STRATEGY", "true")
    strategy = ORBProStrategy(ORBProStrategyConfig(), indicator_engine=None)
    indicators = {
        "orb_ready": True,
        "orb_high": 105.0,
        "orb_low": 95.0,
        "open": 104.0,
        "high": 111.0,
        "low": 103.0,
        "close": 110.0,
        "atr": 5.0,
        "volume": 1000.0,
        "avg_volume": 900.0,
        "direction_bias": "CE",
        "regime": "TREND_UP",
    }

    assert strategy._evaluate_signal("NFO:NIFTYCE", indicators, 110.0) is not None
    assert strategy._evaluate_signal("NFO:NIFTYCE", indicators, 110.0) is not None

    strategy.notify_entry_accepted("CE")

    assert strategy._evaluate_signal("NFO:NIFTYCE", indicators, 110.0) is None
    assert strategy.last_no_vote_reason == "direction_already_traded_today"
