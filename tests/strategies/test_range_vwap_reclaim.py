"""Causal / broker-safe tests for the range VWAP reclaim research model."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from nifty_scalper_bot.config.strategy_taxonomy import (
    canonical_strategy_role,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    RangeVWAPReclaimStrategyConfig,
)
from nifty_scalper_bot.strategies.elite_strategies.range_vwap_reclaim import (
    RangeVWAPReclaimStrategy,
    assess_option_cost_risk,
    assess_range_vwap_reclaim,
)

IST = ZoneInfo("Asia/Kolkata")
FUTURE = "NFO:NIFTY26NOVFUT"
OPTION = "NFO:NIFTY26NOV25250CE"


def bars(*, base: float = 25250.0) -> list[dict]:
    start = datetime(2026, 11, 18, 10, 0, tzinfo=IST)
    result = []
    for i in range(40):
        close = base + 7.0 * math.sin(i * math.pi * 2 / 13.0)
        result.append(
            {
                "timestamp": (start + timedelta(minutes=i)).isoformat(),
                "open": close - 0.5,
                "high": close + 2.0,
                "low": close - 2.0,
                "close": close,
                "volume": 1000.0,
                "is_complete": True,
            }
        )
    result[-2].update(
        open=base - 11.8, high=base - 10.2,
        low=base - 14.5, close=base - 11.5,
    )
    result[-1].update(
        open=base - 11.2, high=base - 6.1,
        low=base - 12.5, close=base - 7.0,
    )
    return result


def decision_at() -> datetime:
    return datetime(2026, 11, 18, 10, 40, 5, tzinfo=IST)


class History:
    def __init__(self, history: list[dict]) -> None:
        self.history = history
        self.calls = 0

    def get_history(self, symbol: str, *, count: int, field: str) -> list[dict]:
        assert symbol == FUTURE
        assert field == "bars"
        assert count >= 40
        self.calls += 1
        return self.history[-count:]


def indicators(**overrides: object) -> dict:
    return {
        "regime": "range",
        "underlying_direction_bias": "CE",
        "underlying_direction_state": "BULLISH",
        "is_selected_option": True,
        "tradable_quote": True,
        "stale_data_used": False,
        "context_age_seconds": 0.03,
        "data_age_seconds": 0.03,
        "active_futures_symbol": FUTURE,
        "decision_at": decision_at().isoformat(),
        "bid": 99.80,
        "ask": 100.0,
        "atr": 6.0,
        "lot_size": 65,
        **overrides,
    }


def test_range_proof_passes_two_sided_futures_bounce() -> None:
    result = assess_range_vwap_reclaim(
        bars(), decision_at=decision_at(),
        config=RangeVWAPReclaimStrategyConfig(),
    )
    assert result["qualified"] is True, result
    assert result["side"] == "CE"
    assert result["source"] == "verified_completed_futures_1m"


def test_missing_and_provisional_bars_fail_closed() -> None:
    cfg = RangeVWAPReclaimStrategyConfig()
    missing = bars()
    missing[-1]["is_complete"] = False
    assert not assess_range_vwap_reclaim(
        missing, decision_at=decision_at(), config=cfg
    )["qualified"]
    gap = bars()
    gap.pop(-10)
    assert assess_range_vwap_reclaim(
        gap, decision_at=decision_at(), config=cfg
    )["reason"] == "insufficient_verified_futures_history"


def test_gap_in_oversupplied_completed_bars_fails() -> None:
    history = bars()
    history[15]["timestamp"] = (
        datetime(2026, 11, 18, 10, 15, tzinfo=IST)
        + timedelta(seconds=30)
    ).isoformat()
    assert assess_range_vwap_reclaim(
        history, decision_at=decision_at(),
        config=RangeVWAPReclaimStrategyConfig(),
    )["reason"] == "futures_history_gap"


def test_repeated_conflicting_candle_not_used() -> None:
    history = bars()
    fake = dict(history[-1])
    fake["close"] += 0.25
    history.append(fake)
    assert assess_range_vwap_reclaim(
        history, decision_at=decision_at(),
        config=RangeVWAPReclaimStrategyConfig(),
    )["reason"] == "conflicting_futures_bar"


def test_stale_futures_and_directional_trends_do_not_qualify() -> None:
    cfg = RangeVWAPReclaimStrategyConfig()
    stale = assess_range_vwap_reclaim(
        bars(),
        decision_at=decision_at() + timedelta(minutes=3),
        config=cfg,
    )
    assert stale["reason"] == "futures_completed_bar_stale"
    trend = bars()
    for i, row in enumerate(trend):
        close = 25200.0 + i * 15
        row.update(
            open=close - 2, high=close + 3,
            low=close - 3, close=close,
        )
    assert assess_range_vwap_reclaim(
        trend, decision_at=decision_at(), config=cfg
    )["qualified"] is False


def test_option_cost_includes_actual_fees_and_stressed_slippage() -> None:
    cfg = RangeVWAPReclaimStrategyConfig()
    a = assess_option_cost_risk(
        {"bid": 99.80, "ask": 100.0, "atr": 6.0, "lot_size": 65}, config=cfg
    )
    assert a["qualified"] is True, a
    assert a["worst_stop_loss_including_costs"] <= 773
    assert a["net_reward_risk"] >= 1.5
    high_stress = RangeVWAPReclaimStrategyConfig(
        slippage_stress_bps=200,
    )
    b = assess_option_cost_risk(
        {"bid": 99.80, "ask": 100.0, "atr": 6.0, "lot_size": 65},
        config=high_stress,
    )
    assert b["net_reward_risk"] < a["net_reward_risk"]


def test_overspread_and_high_cash_risk_abstain() -> None:
    cfg = RangeVWAPReclaimStrategyConfig()
    assert assess_option_cost_risk(
        {"bid": 95, "ask": 100, "atr": 6, "lot_size": 65}, config=cfg
    )["reason"] == "option_spread_too_wide"
    assert assess_option_cost_risk(
        {"bid": 200, "ask": 200.2, "atr": 25, "lot_size": 65}, config=cfg
    )["reason"] == "cash_risk_over_cap"


def test_context_only_taxonomy_blocks_broker_trigger() -> None:
    assert canonical_strategy_role("range_vwap") == "context"
    assert canonical_strategy_role("RangeVWAPReclaim") == "context"


def test_shadow_generates_one_nontrigger_context_vote(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    engine = History(bars())
    strategy = RangeVWAPReclaimStrategy(RangeVWAPReclaimStrategyConfig(), engine)
    signal = strategy._evaluate_signal(OPTION, indicators(), 100.0)
    assert signal is not None
    assert signal.metadata["role"] == "context"
    assert signal.metadata["can_trigger"] is False
    assert signal.metadata["research_only"] is True
    assert strategy._evaluate_signal(OPTION, indicators(), 100.0) is None
    assert engine.calls == 2


def test_live_mode_never_returns_eligible_signal(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    strategy = RangeVWAPReclaimStrategy(
        RangeVWAPReclaimStrategyConfig(), History(bars())
    )
    assert strategy._evaluate_signal(OPTION, indicators(), 100.0) is None


def test_stale_quote_trend_mode_or_countertrend_are_not_candidates(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    for override in (
        {"data_age_seconds": 10},
        {"regime": "trend"},
        {"underlying_direction_bias": "PE"},
        {"underlying_direction_state": "TRANSITION"},
        {"tradable_quote": False},
        {"is_selected_option": False},
        {"active_futures_symbol": None},
    ):
        strategy = RangeVWAPReclaimStrategy(
            RangeVWAPReclaimStrategyConfig(), History(bars())
        )
        assert strategy._evaluate_signal(
            OPTION, indicators(**override), 100.0
        ) is None
