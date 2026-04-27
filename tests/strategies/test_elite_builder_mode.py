from __future__ import annotations

from nifty_scalper_bot.strategies.elite_strategies.builder import build_elite_strategies
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    EliteStrategiesSettings,
)


def test_directional_mode_disables_gamma_theta_and_context(monkeypatch) -> None:
    monkeypatch.setenv('STRATEGY_MODE', 'directional_scalp')
    monkeypatch.setenv('ALLOW_EXPIRY_GAMMA_STRATEGIES', 'false')

    strategies = build_elite_strategies(
        settings=EliteStrategiesSettings(),
        indicator_engine=None,
    )
    names = {strategy.name for strategy in strategies}

    assert 'GammaScalping' not in names
    assert 'EliteTuesdayGammaBuyer' not in names
    assert 'StraddleTheta' not in names
    assert 'OIMaxPain' not in names
    assert {'SMC', 'VWAPPro', 'CPRBreakout', 'OrderFlow', 'BBSqueeze', 'ORBPro'}.issubset(
        names
    )
