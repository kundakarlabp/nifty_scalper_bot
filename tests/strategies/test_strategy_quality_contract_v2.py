from __future__ import annotations

from datetime import datetime, timedelta, timezone

from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    ORBProStrategyConfig,
    OrderFlowStrategyConfig,
    SMCStrategyConfig,
    VWAPProStrategyConfig,
)
from nifty_scalper_bot.strategies.elite_strategies.orb_pro import ORBProStrategy
from nifty_scalper_bot.strategies.elite_strategies.order_flow import OrderFlowStrategy
from nifty_scalper_bot.strategies.elite_strategies.smc_liquidity import SMCStrategy
from nifty_scalper_bot.strategies.elite_strategies.vwap_pro import VWAPProStrategy


def test_orderflow_native_score_uses_config_without_unconditional_point() -> None:
    strategy = OrderFlowStrategy(
        OrderFlowStrategyConfig(enabled=True, quantity=1), indicator_engine=None
    )
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
        "NFO:NIFTY26SEP25000CE", indicators, current_price=100.5
    )

    assert signal is not None
    assert signal.metadata["strategy_score"] == 8.0
    assert signal.metadata["raw_setup_score"] == 8.0
    assert signal.metadata["context_evidence_score"] == 4.0
    assert signal.metadata["context_bonus_score"] == 2.0
    assert signal.metadata["depth_support_threshold"] == 0.15
    assert 0.47 < signal.metadata["strong_depth_support_threshold"] < 0.48
    assert signal.metadata["role"] == "context"


def test_orderflow_native_large_order_threshold_changes_depth_score() -> None:
    strategy = OrderFlowStrategy(
        OrderFlowStrategyConfig(
            enabled=True,
            quantity=1,
            large_order_threshold_pct=30.0,
        ),
        indicator_engine=None,
    )
    indicators = {
        "bid": 100.0,
        "ask": 100.5,
        "spread_pct": 0.50,
        "depth": {"buy": [{"quantity": 150}], "sell": [{"quantity": 100}]},
        "tick_direction": "UP",
        "direction_bias": "CE",
        "atr": 2.0,
        "data_age_seconds": 0.2,
    }

    signal = strategy._evaluate_signal(
        "NFO:NIFTY26SEP25000CE", indicators, current_price=100.25
    )

    assert signal is not None
    assert signal.metadata["depth_imbalance"] == 0.2
    assert signal.metadata["depth_support_threshold"] == 0.3
    assert signal.metadata["depth_supports_side"] is False
    assert signal.metadata["depth_score"] == 0.0
    assert signal.metadata["strategy_score"] == 6.0


def _smc_snapshot(now: datetime) -> dict[str, object]:
    return {
        "source": "futures",
        "symbol": "NFO:NIFTY26SEPFUT",
        "rows": [],
        "current": {
            "timestamp": now,
            "open": 100.0,
            "high": 104.0,
            "low": 99.0,
            "close": 103.0,
            "volume": 1000.0,
        },
        "current_ts": now,
        "atr": 10.0,
        "volume_ratio": 1.0,
        "pivot_low": None,
        "pivot_high": None,
        "history_count": 50,
    }


def _smc_event(now: datetime, *, volume_confirmation: bool) -> dict[str, object]:
    return {
        "side": "CE",
        "sweep_ts": now - timedelta(minutes=1),
        "sweep_level": 100.0,
        "sweep_extreme": 95.0,
        "sweep_bar_high": 102.0,
        "sweep_bar_low": 95.0,
        "depth_points": 8.0,
        "depth_atr": 0.8,
        "reclaim_points": 2.0,
        "reclaim_atr": 0.2,
        "volume_ratio": 2.2 if volume_confirmation else 1.0,
        "volume_confirmation": volume_confirmation,
        "effective_min_sweep_points": 0.8,
        "recovered_from_history": False,
    }


def test_smc_native_live_core_plus_direction_does_not_auto_pass(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    strategy = SMCStrategy(SMCStrategyConfig(), indicator_engine=None)
    now = datetime(2026, 9, 11, 5, 0, tzinfo=timezone.utc)
    snapshot = _smc_snapshot(now)
    monkeypatch.setattr(strategy, "_underlying_snapshot", lambda indicators: snapshot)
    strategy._events[("NFO:NIFTY26SEPFUT", "CE")] = _smc_event(
        now, volume_confirmation=False
    )

    signal = strategy._evaluate_signal(
        "NFO:NIFTY26SEP25000CE",
        {
            "direction_bias": "CE",
            "underlying_direction_bias": "CE",
            "atr": 2.0,
            "data_age_seconds": 0.1,
            "stale_data_used": False,
        },
        current_price=100.0,
    )

    assert signal is None
    assert strategy.last_no_vote_reason == "smc_low_score"


def test_smc_native_independent_confirmation_reaches_live_floor(monkeypatch) -> None:
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
            "atr": 2.0,
            "data_age_seconds": 0.1,
            "stale_data_used": False,
        },
        current_price=100.0,
    )

    assert signal is not None
    assert signal.metadata["strategy_score"] == 6.5
    assert signal.metadata["setup_min"] == 6.5
    assert signal.metadata["setup_pass"] is True
    assert signal.metadata["smc_quality_independent_confirmation"] is True


def _ready_vwap_indicators() -> dict[str, object]:
    return {
        "exchange_vwap": 100.0,
        "atr": 2.0,
        "close": 100.10,
        "open": 99.8,
        "high": 100.3,
        "low": 99.2,
        "volume": 1200.0,
        "avg_volume": 1000.0,
        "spread_pct": 0.5,
        "direction_bias": "CE",
        "underlying_direction_bias": "CE",
        "underlying_direction_confidence": 0.95,
        "context_age_seconds": 0.5,
        "futures_vwap_slope": 0.01,
        "futures_volume_ratio": 1.1,
        "stale_data_used": False,
        "data_age_seconds": 0.2,
        "latest_bar_ts": "2026-09-11T05:00:00+00:00",
        "session_date": "2026-09-11",
    }


def test_vwap_native_consumes_proximity_config(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    strategy = VWAPProStrategy(
        VWAPProStrategyConfig(proximity_pct=0.15), indicator_engine=None
    )
    strategy._thesis_anchor_by_scope[
        ("NFO:NIFTY26SEP25000CE", "2026-09-11")
    ] = "2026-09-11T04:59:00+00:00"

    signal = strategy._evaluate_signal(
        "NFO:NIFTY26SEP25000CE",
        _ready_vwap_indicators(),
        current_price=100.10,
    )

    assert signal is not None
    assert signal.metadata["vwap_configured_proximity_pct"] == 0.15
    assert signal.metadata["vwap_configured_proximity_pass"] is True
    assert signal.metadata["vwap_distance_atr"] == 0.05
    assert signal.metadata["setup_score"] == signal.metadata["strategy_score"]


def test_vwap_native_rejects_atr_overextension(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    strategy = VWAPProStrategy(
        VWAPProStrategyConfig(proximity_pct=0.15), indicator_engine=None
    )
    indicators = _ready_vwap_indicators()
    indicators.update(
        {
            "close": 120.0,
            "open": 119.0,
            "high": 121.0,
            "low": 118.0,
            "atr": 5.0,
        }
    )

    signal = strategy._evaluate_signal(
        "NFO:NIFTY26SEP25000CE", indicators, current_price=120.0
    )

    assert signal is None
    assert strategy.last_no_vote_reason == "distance_outside_band"


def _orb_snapshot(*, high: float, low: float, current_close: float) -> dict[str, object]:
    now = datetime(2026, 9, 11, 4, 30, tzinfo=timezone.utc)
    return {
        "source": "futures",
        "symbol": "NFO:NIFTY26SEPFUT",
        "session_date": "2026-09-11",
        "orb_minutes": 15,
        "orb_high": high,
        "orb_low": low,
        "current": {
            "timestamp": now,
            "open": current_close - 0.2,
            "high": current_close + 0.2,
            "low": current_close - 0.3,
            "close": current_close,
            "volume": 1000.0,
        },
        "current_ts": now,
        "atr": 10.0,
        "body_ratio": 0.7,
        "volume_ratio": 1.0,
    }


def test_orb_native_opening_range_quality_controls_admission(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    strategy = ORBProStrategy(ORBProStrategyConfig(), indicator_engine=None)
    event_time = datetime(2026, 9, 11, 4, 29, tzinfo=timezone.utc)
    event = {
        "boundary": 101.0,
        "breakout_timestamp": event_time,
        "body_ratio": 0.7,
        "recovered_from_history": False,
    }
    indicators = {"atr": 2.0, "stale_data_used": False}

    weak = strategy._build_signal(
        symbol="NFO:NIFTY26SEP25000CE",
        side="CE",
        current_price=100.0,
        indicators=indicators,
        snapshot=_orb_snapshot(high=101.0, low=100.9, current_close=101.1),
        event=event,
        branch="momentum",
        retest_timestamp=None,
    )
    assert weak is None
    assert strategy.last_no_vote_reason == "orb_quality_below_minimum"

    event["boundary"] = 105.0
    accepted = strategy._build_signal(
        symbol="NFO:NIFTY26SEP25000CE",
        side="CE",
        current_price=100.0,
        indicators=indicators,
        snapshot=_orb_snapshot(high=105.0, low=100.0, current_close=105.1),
        event=event,
        branch="momentum",
        retest_timestamp=None,
    )
    assert accepted is not None
    assert accepted.metadata["opening_range_width_atr"] == 0.5
    assert accepted.metadata["opening_range_balanced"] is True
    assert accepted.metadata["strategy_score"] == 6.0
    assert accepted.metadata["setup_min"] == 6.0
