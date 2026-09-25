from __future__ import annotations

from datetime import datetime, timezone

from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    ORBProStrategyConfig,
    SMCStrategyConfig,
)
from nifty_scalper_bot.strategies.elite_strategies.orb_pro import ORBProStrategy
from nifty_scalper_bot.strategies.elite_strategies.smc_liquidity import SMCStrategy
from nifty_scalper_bot.strategies.signal_quality import score_signal_metadata

from .test_strategy_quality_contract_v2 import (
    _orb_snapshot,
    _smc_event,
    _smc_snapshot,
)


def test_orb_direction_and_setup_alpha_are_independent(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    strategy = ORBProStrategy(ORBProStrategyConfig(), indicator_engine=None)
    event_time = datetime(2026, 9, 11, 4, 29, tzinfo=timezone.utc)

    signal = strategy._build_signal(
        symbol="NFO:NIFTY26SEP25000CE",
        side="CE",
        current_price=100.0,
        indicators={
            "atr": 2.0,
            "stale_data_used": False,
            "underlying_direction_bias": "CE",
            "underlying_direction_confidence": 0.72,
            "context_fresh": True,
        },
        snapshot=_orb_snapshot(high=105.0, low=100.0, current_close=105.1),
        event={
            "boundary": 105.0,
            "breakout_timestamp": event_time,
            "body_ratio": 0.7,
            "recovered_from_history": False,
        },
        branch="momentum",
        retest_timestamp=None,
    )

    assert signal is not None
    assert signal.metadata["strategy_score"] == 7.0
    assert signal.metadata["direction_score"] == 7.2
    assert signal.metadata["independent_setup_score"] == 6.0
    assert signal.metadata["direction_score"] != signal.metadata["strategy_score"]


def test_smc_direction_and_setup_alpha_are_independent(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    strategy = SMCStrategy(SMCStrategyConfig(), indicator_engine=None)
    now = datetime(2026, 9, 11, 5, 0, tzinfo=timezone.utc)
    snapshot = _smc_snapshot(now)
    monkeypatch.setattr(strategy, "_underlying_snapshot", lambda indicators: snapshot)
    strategy._events[("NFO:NIFTY26SEPFUT", "CE")] = _smc_event(
        now, volume_confirmation=True
    )

    signal = strategy._evaluate_signal(
        "NFO:NIFTY26SEP25000CE",
        {
            "direction_bias": "CE",
            "underlying_direction_bias": "CE",
            "underlying_direction_confidence": 0.81,
            "context_fresh": True,
            "atr": 2.0,
            "data_age_seconds": 0.1,
            "stale_data_used": False,
        },
        current_price=100.0,
    )

    assert signal is not None
    assert signal.metadata["strategy_score"] == 6.5
    assert signal.metadata["direction_score"] == 8.1
    assert signal.metadata["independent_setup_score"] == 5.0
    assert signal.metadata["direction_score"] != signal.metadata["strategy_score"]


def test_runner_quality_uses_independent_setup_for_orb_and_smc(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")

    for strategy_name in ("ORBPro", "SMC"):
        quality = score_signal_metadata(
            {
                "direction_score": 8.5,
                "strategy_score": 8.0,
                "independent_setup_score": 5.5,
                "option_score": 10.0,
                "data_score": 10.0,
                "rr_score": 10.0,
            },
            strategy_name=strategy_name,
        )
        assert quality.strategy_score == 5.5
        assert quality.components["strategy_score"] == 5.5
