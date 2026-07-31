from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from nifty_scalper_bot.config.settings import (
    LiquiditySettings,
    OrderLifecycleSettings,
    OrderSettings,
    RiskSettings,
)
from nifty_scalper_bot.strategies.elite_strategies.builder import (
    build_elite_strategies,
    build_production_strategy_profile,
)
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


def test_production_profile_is_stable_and_changes_with_material_settings(
    monkeypatch,
) -> None:
    monkeypatch.setenv("STRATEGY_MODE", "directional_scalp")
    monkeypatch.setenv("ORDERFLOW_ALLOW_TRIGGER_ROLE", "false")
    elite = EliteStrategiesSettings()
    strategies = build_elite_strategies(elite, indicator_engine=None)
    runtime = SimpleNamespace(
        execution_mode="LIVE",
        orders=OrderSettings(lifecycle=OrderLifecycleSettings()),
        liquidity=LiquiditySettings(max_spread_pct=30.0),
        risk=RiskSettings(per_trade_risk_pct=5.0),
    )
    mode_profile = {
        "mode": "LIVE",
        "allow_context_promotion": False,
        "allow_single_vote": True,
        "min_trade_quality": 7.0,
    }

    first = build_production_strategy_profile(
        settings=runtime,
        strategies=strategies,
        mode_profile=mode_profile,
        global_min_confidence=0.35,
    )
    repeated = build_production_strategy_profile(
        settings=runtime,
        strategies=strategies,
        mode_profile=mode_profile,
        global_min_confidence=0.35,
    )

    assert first == repeated
    assert first["version"].startswith("production-v1-")
    assert first["execution_mode"] == "LIVE"
    assert "OrderFlow" in first["strategies"]["context_only"]
    assert {"SMC", "VWAPPro", "ORBPro"}.issubset(
        first["strategies"]["trigger_capable"]
    )
    assert first["score_thresholds"]["global_min_confidence"] == 0.35

    changed_runtime = SimpleNamespace(
        **{
            **runtime.__dict__,
            "risk": replace(runtime.risk, per_trade_risk_pct=4.0),
        }
    )
    changed = build_production_strategy_profile(
        settings=changed_runtime,
        strategies=strategies,
        mode_profile=mode_profile,
        global_min_confidence=0.35,
    )

    assert changed["risk"]["per_trade_risk_pct"] == 4.0
    assert changed["version"] != first["version"]
