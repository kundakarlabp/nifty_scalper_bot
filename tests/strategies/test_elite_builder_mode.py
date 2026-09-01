from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
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


_FUTURE = "NFO:NIFTY26SEPFUT"
_OPEN = datetime(2026, 9, 1, 3, 45, tzinfo=timezone.utc)


class _OrbIndicatorEngine:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def get_history(self, symbol: str, count=None, *, field: str = "close"):
        rows = list(self.rows) if symbol == _FUTURE else []
        if count is not None:
            rows = rows[-count:]
        if field == "bars":
            return rows
        return [float(row["close"]) for row in rows]


def _orb_rows(*, side: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for minute in range(15):
        rows.append(
            {
                "timestamp": _OPEN + timedelta(minutes=minute),
                "open": 24_000.0,
                "high": 24_020.0 if minute == 8 else 24_012.0,
                "low": 23_980.0 if minute == 4 else 23_992.0,
                "close": 24_000.0,
                "volume": 1_000.0,
                "is_complete": True,
                "is_provisional": False,
            }
        )
    if side == "CE":
        breakout = {
            "timestamp": _OPEN + timedelta(minutes=15),
            "open": 24_008.0,
            "high": 24_034.0,
            "low": 24_006.0,
            "close": 24_030.0,
            "volume": 3_000.0,
            "is_complete": True,
            "is_provisional": False,
        }
    else:
        breakout = {
            "timestamp": _OPEN + timedelta(minutes=15),
            "open": 23_990.0,
            "high": 23_992.0,
            "low": 23_955.0,
            "close": 23_962.0,
            "volume": 3_000.0,
            "is_complete": True,
            "is_provisional": False,
        }
    rows.append(breakout)
    return rows


def _orb_indicators(side: str, latest_ts: datetime) -> dict[str, object]:
    return {
        "history_count": 100,
        "orb_ready": True,
        "orb_high": 105.0,
        "orb_low": 95.0,
        "open": 48.0,
        "high": 52.0,
        "low": 47.0,
        "close": 51.0,
        "atr": 4.0,
        "volume": 1000.0,
        "avg_volume": 900.0,
        "direction_bias": side,
        "underlying_direction_bias": side,
        "regime": "TREND_UP" if side == "CE" else "TREND_DOWN",
        "spread_pct": 0.5,
        "bid": 49.5,
        "ask": 50.0,
        "quote_depth_valid": True,
        "tradable_quote": True,
        "stale_data_used": False,
        "futures_symbol": _FUTURE,
        "futures_price": 24_030.0 if side == "CE" else 23_962.0,
        "futures_vwap_slope": 1.0 if side == "CE" else -1.0,
        "latest_bar_ts": latest_ts.timestamp(),
        "session_date": "2026-09-01",
    }


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


def test_orb_enabled_config_is_not_overridden_by_legacy_env(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_ORB_STRATEGY", "false")
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    rows = _orb_rows(side="CE")
    strategy = ORBProStrategy(
        ORBProStrategyConfig(min_confidence=0.0, orb_minutes=15),
        indicator_engine=_OrbIndicatorEngine(rows),
    )
    indicators = _orb_indicators("CE", rows[-1]["timestamp"])

    signal = strategy.generate_signal("NFO:NIFTY26SEP24050CE", indicators, 50.0)

    assert signal is not None
    assert signal.metadata["setup_id"].startswith("orbv2:")
    assert signal.metadata["opening_range_source"] == "futures"
    assert signal.metadata["orb_window_minutes"] == 15


def test_orb_pe_underlying_breakout_uses_contract_side(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    rows = _orb_rows(side="PE")
    strategy = ORBProStrategy(
        ORBProStrategyConfig(min_confidence=0.0, orb_minutes=15),
        indicator_engine=_OrbIndicatorEngine(rows),
    )
    indicators = _orb_indicators("PE", rows[-1]["timestamp"])

    signal = strategy.generate_signal("NFO:NIFTY26SEP24050PE", indicators, 48.0)

    assert signal is not None
    assert signal.metadata["contract_side"] == "PE"
    assert signal.metadata["breakout_side"] == "PE"
    assert signal.stop_loss is not None and 0 < signal.stop_loss < 48.0
    assert signal.metadata["underlying_invalidation"] > 23_962.0


def test_orb_option_premium_breakdown_does_not_create_buy_vote() -> None:
    strategy = ORBProStrategy(
        ORBProStrategyConfig(min_confidence=0.0), indicator_engine=None
    )
    indicators = {
        "orb_ready": True,
        "orb_high": 105.0,
        "orb_low": 95.0,
        "open": 96.0,
        "high": 97.0,
        "low": 89.0,
        "close": 90.0,
        "atr": 5.0,
        "volume": 1000.0,
        "avg_volume": 900.0,
        "underlying_direction_bias": "PE",
        "regime": "TREND_DOWN",
        "latest_bar_ts": 1_785_000_000.0,
    }

    signal = strategy.generate_signal("NFO:NIFTY2680724500PE", indicators, 90.0)

    assert signal is None


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
