"""Causal session-VWAP reclaim, cash risk and no-live-trigger tests.

These fixtures assert *logic*, not executable historical NIFTY profitability.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from nifty_scalper_bot.config.strategy_taxonomy import canonical_strategy_role
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
    """Complete IST day from 09:15 with an exterior candle + interior reclaim."""
    start = datetime(2026, 11, 18, 9, 15, tzinfo=IST)
    result = []
    for i in range(46):
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
        open=base - 11.8,
        high=base - 10.2,
        low=base - 14.5,
        close=base - 11.5,
    )
    result[-1].update(
        open=base - 8.0,
        high=base - 4.0,
        low=base - 12.5,
        close=base - 4.7,
    )
    return result


def decision_at() -> datetime:
    return datetime(2026, 11, 18, 10, 1, 5, tzinfo=IST)


class History:
    def __init__(self, history: list[dict]) -> None:
        self.history = history
        self.calls = 0

    def get_history(self, symbol: str, *, count: int, field: str) -> list[dict]:
        assert symbol == FUTURE
        assert field == "bars"
        assert count >= 375
        self.calls += 1
        return self.history[-count:]


def indicators(**overrides: object) -> dict:
    return {
        "regime": "RANGING",
        # Deliberately opposite: futures structure determines research side.
        "underlying_direction_bias": "PE",
        "underlying_direction_state": "BEARISH",
        "direction_transition": False,
        "is_selected_option": True,
        "tradable_quote": True,
        "stale_data_used": False,
        "context_age_seconds": 0.03,
        "data_age_seconds": 0.03,
        "active_futures_symbol": FUTURE,
        "decision_at": decision_at().isoformat(),
        "futures_ltp": 25245.3,
        "bid": 99.80,
        "ask": 100.0,
        "atr": 6.0,
        "lot_size": 65,
        "days_to_expiry": 8,
        "delta": 0.55,
        "ofi_ready": True,
        "ofi_3s_complete": True,
        "ofi_update_count_3s": 4,
        "ofi_3s_span_seconds": 1.8,
        "ofi_3s_normalized": 0.30,
        "ofi_1s_normalized": 0.20,
        **overrides,
    }


def test_true_session_anchor_and_rolling_secondary_agree() -> None:
    report = assess_range_vwap_reclaim(
        bars(),
        decision_at=decision_at(),
        config=RangeVWAPReclaimStrategyConfig(),
    )
    assert report["qualified"] is True, report
    assert report["side"] == "CE"
    assert report["source"] == "contiguous_session_futures_1m"
    assert report["primary_vwap_source"].startswith("session_0915")
    assert report["rolling_20m_secondary_center"] != 0
    assert report["lower_band_before_confirmation"] < 25245.3
    assert report["session_vwap_before_confirmation"] > 25245.3


def test_partial_intraday_history_cannot_impersonate_session_vwap() -> None:
    data = bars()[5:]
    proof = assess_range_vwap_reclaim(
        data,
        decision_at=decision_at(),
        config=RangeVWAPReclaimStrategyConfig(min_history_bars=40),
    )
    assert proof["reason"] == "session_open_anchor_missing"


def test_reclaim_must_close_inside_band_precomputed_before_confirmation() -> None:
    data = bars()
    weak = data[-1]
    weak.update(open=25238.0, close=25243.0, high=25244.0, low=25237.5)
    proof = assess_range_vwap_reclaim(
        data,
        decision_at=decision_at(),
        config=RangeVWAPReclaimStrategyConfig(),
    )
    assert proof["qualified"] is False
    assert proof["reason"] == "no_completed_band_reentry"
    assert proof["lower_band_before_confirmation"] > weak["close"]


def test_confirmation_candle_cannot_move_its_own_outer_band() -> None:
    original = assess_range_vwap_reclaim(
        bars(),
        decision_at=decision_at(),
        config=RangeVWAPReclaimStrategyConfig(),
    )
    modified = bars()
    modified[-1]["volume"] = 100000.0
    modified[-1]["high"] += 1.0
    other = assess_range_vwap_reclaim(
        modified,
        decision_at=decision_at(),
        config=RangeVWAPReclaimStrategyConfig(),
    )
    for key in (
        "session_vwap_before_confirmation",
        "rolling_20m_secondary_center",
        "lower_band_before_confirmation",
        "upper_band_before_confirmation",
    ):
        assert original[key] == other[key]


def test_missing_provisional_gaps_conflicts_and_stale_fail_closed() -> None:
    cfg = RangeVWAPReclaimStrategyConfig()
    missing = bars()
    missing[-1]["is_complete"] = False
    assert not assess_range_vwap_reclaim(
        missing, decision_at=decision_at(), config=cfg
    )["qualified"]
    gap = bars()
    gap.pop(-10)
    assert (
        assess_range_vwap_reclaim(gap, decision_at=decision_at(), config=cfg)["reason"]
        == "futures_history_gap"
    )
    noncanonical = bars()
    noncanonical[15]["timestamp"] = (
        datetime(2026, 11, 18, 9, 30, 30, tzinfo=IST)
    ).isoformat()
    assert (
        assess_range_vwap_reclaim(noncanonical, decision_at=decision_at(), config=cfg)[
            "reason"
        ]
        == "noncanonical_minute_timestamp"
    )
    dup = bars()
    fake = dict(dup[-1])
    fake["close"] += 0.25
    dup.append(fake)
    assert (
        assess_range_vwap_reclaim(dup, decision_at=decision_at(), config=cfg)["reason"]
        == "conflicting_futures_bar"
    )
    stale = assess_range_vwap_reclaim(
        bars(), decision_at=decision_at() + timedelta(minutes=4), config=cfg
    )
    assert stale["reason"] == "futures_completed_bar_stale"


def test_drift_and_center_disagreement_exclude_trends() -> None:
    trend = bars()
    for i, row in enumerate(trend):
        close = 25200.0 + i * 15
        row.update(open=close - 2, high=close + 3, low=close - 3, close=close)
    assert not assess_range_vwap_reclaim(
        trend,
        decision_at=decision_at(),
        config=RangeVWAPReclaimStrategyConfig(),
    )["qualified"]
    assert not assess_range_vwap_reclaim(
        bars(),
        decision_at=decision_at(),
        config=RangeVWAPReclaimStrategyConfig(max_secondary_center_deviation_atr=0.001),
    )["qualified"]


def test_reachable_option_reward_is_futures_delta_theta_and_cost_limited() -> None:
    cfg = RangeVWAPReclaimStrategyConfig()
    quote = {
        "bid": 99.8,
        "ask": 100.0,
        "atr": 6.0,
        "lot_size": 65,
        "delta": 0.55,
        "theta": -15.0,
    }
    a = assess_option_cost_risk(
        quote,
        config=cfg,
        side="CE",
        futures_target_distance=35,
        futures_stop_distance=3,
    )
    assert a["qualified"] is True, a
    assert a["worst_stop_loss_including_costs"] <= 773
    assert a["net_reward_risk"] >= 1.5
    assert a["projected_premium_gain_points"] < 0.55 * 35
    assert a["theta_cost_source"] == "observed_theta_daily_375minute_stress"
    b = assess_option_cost_risk(
        quote,
        config=RangeVWAPReclaimStrategyConfig(slippage_stress_bps=200),
        side="CE",
        futures_target_distance=35,
        futures_stop_distance=3,
    )
    assert b.get("net_reward_risk", 0) < a["net_reward_risk"]


def test_missing_delta_expiry_side_wide_spread_and_high_cash_risk_reject() -> None:
    cfg = RangeVWAPReclaimStrategyConfig()
    quote = {"bid": 99.8, "ask": 100, "atr": 6, "lot_size": 65, "delta": 0.55}

    def assess(updated: dict, side: str = "CE", gap: float = 3) -> str:
        return str(
            assess_option_cost_risk(
                updated,
                config=cfg,
                side=side,
                futures_target_distance=35,
                futures_stop_distance=gap,
            )["reason"]
        )

    assert assess({**quote, "delta": None}) == "option_delta_unverified"
    assert assess({**quote, "delta": -0.55}) == "option_delta_side_conflict"
    assert assess({**quote, "bid": 95}) == "option_spread_too_wide"
    assert assess(quote, gap=60) == "cash_risk_over_cap"
    assert assess_option_cost_risk(quote, config=cfg)["reason"] == (
        "option_contract_side_missing"
    )


def test_context_only_taxonomy_blocks_broker_trigger() -> None:
    assert canonical_strategy_role("range_vwap") == "context"
    assert canonical_strategy_role("RangeVWAPReclaim") == "context"


def test_shadow_opposing_trend_bias_still_uses_structural_reversal(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    engine = History(bars()[:-1])
    strategy = RangeVWAPReclaimStrategy(
        RangeVWAPReclaimStrategyConfig(min_net_reward_risk=0.0), engine
    )
    earlier = decision_at() - timedelta(minutes=1)
    assert (
        strategy._evaluate_signal(
            OPTION, indicators(decision_at=earlier.isoformat()), 100.0
        )
        is None
    )
    assert strategy._regime_bars[FUTURE][1] == 1
    # The *next* chronological finalized candle confirms sustained RANGE.
    engine.history = bars()
    signal = strategy._evaluate_signal(OPTION, indicators(), 100.0)
    assert signal is not None
    assert signal.metadata["role"] == "context"
    assert signal.metadata["can_trigger"] is False
    assert signal.metadata["research_only"] is True
    assert signal.metadata["prior_trend_bias"] == "PE"
    assert strategy._evaluate_signal(OPTION, indicators(), 100.0) is None
    assert engine.calls == 3


def test_live_mode_never_emits_entry_trigger(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    engine = History(bars()[:-1])
    strategy = RangeVWAPReclaimStrategy(
        RangeVWAPReclaimStrategyConfig(min_net_reward_risk=0.0), engine
    )
    strategy._evaluate_signal(
        OPTION,
        indicators(decision_at=(decision_at() - timedelta(minutes=1)).isoformat()),
        100.0,
    )
    engine.history = bars()
    assert strategy._evaluate_signal(OPTION, indicators(), 100.0) is None


def test_stale_regime_transition_expiry_flow_and_option_selection_reject(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    for override in (
        {"data_age_seconds": 10},
        {"regime": "trend"},
        {"regime": "CHOPPY", "regime_transition_pending": True},
        {"direction_transition": True},
        {"underlying_direction_state": "TRANSITION"},
        {"tradable_quote": False},
        {"is_selected_option": False, "strike_distance_from_atm": 150},
        {"days_to_expiry": 0},
        {"days_to_expiry": None},
        {"ofi_ready": False},
        {"ofi_3s_normalized": -0.3},
        {"depth_imbalance": -0.30},
        {"active_futures_symbol": None},
        {"futures_ltp": 25285},
    ):
        strategy = RangeVWAPReclaimStrategy(
            RangeVWAPReclaimStrategyConfig(min_net_reward_risk=0.0),
            History(bars()),
        )
        strategy._regime_bar_confirmed(FUTURE, bars()[-2]["timestamp"])
        assert strategy._evaluate_signal(OPTION, indicators(**override), 100.0) is None


def test_bearish_pe_reclaim_mirrors_bullish_futures_without_old_bias(monkeypatch):
    """Both CE and PE require positive *option premium* OFI."""
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    midpoint = 25250.0
    bearish: list[dict] = []
    for item in bars():
        mirrored = dict(item)
        mirrored.update(
            open=2 * midpoint - item["open"],
            high=2 * midpoint - item["low"],
            low=2 * midpoint - item["high"],
            close=2 * midpoint - item["close"],
        )
        bearish.append(mirrored)
    proof = assess_range_vwap_reclaim(
        bearish,
        decision_at=decision_at(),
        config=RangeVWAPReclaimStrategyConfig(),
    )
    assert proof["qualified"] is True, proof
    assert proof["side"] == "PE"
    engine = History(bearish[:-1])
    strategy = RangeVWAPReclaimStrategy(
        RangeVWAPReclaimStrategyConfig(min_net_reward_risk=0.0), engine
    )
    early = decision_at() - timedelta(minutes=1)
    strategy._evaluate_signal(
        "NFO:NIFTY26NOV25250PE",
        indicators(decision_at=early.isoformat()),
        100.0,
    )
    engine.history = bearish
    result = strategy._evaluate_signal(
        "NFO:NIFTY26NOV25250PE",
        indicators(
            underlying_direction_bias="CE",
            futures_ltp=25254.7,
            delta=-0.55,
        ),
        100.0,
    )
    assert result is not None
    assert result.metadata["side"] == "PE"
    assert result.metadata["role"] == "context"
    assert result.metadata["can_trigger"] is False


def test_regime_confirmation_never_counts_duplicate_or_backward_bar():
    strategy = RangeVWAPReclaimStrategy(
        RangeVWAPReclaimStrategyConfig(), History(bars())
    )
    t = bars()[-2]["timestamp"]
    newer = bars()[-1]["timestamp"]
    assert strategy._regime_bar_confirmed(FUTURE, t) is False
    assert strategy._regime_bar_confirmed(FUTURE, t) is False
    assert strategy._regime_bar_confirmed(FUTURE, newer) is True
    assert strategy._regime_bar_confirmed(FUTURE, t) is False


def test_same_excursion_requires_real_vwap_reset():
    strategy = RangeVWAPReclaimStrategy(
        RangeVWAPReclaimStrategyConfig(), History(bars())
    )
    key = (FUTURE, "CE", "2026-11-18")
    strategy._consumed_excursions.add(key)
    strategy._maybe_reset_excursions(
        FUTURE,
        {
            "underlying_latest_close": 25230.0,
            "session_vwap_before_confirmation": 25250.0,
            "futures_atr": 6.0,
        },
    )
    assert key in strategy._consumed_excursions
    strategy._maybe_reset_excursions(
        FUTURE,
        {
            "underlying_latest_close": 25249.0,
            "session_vwap_before_confirmation": 25250.0,
            "futures_atr": 6.0,
        },
    )
    assert key not in strategy._consumed_excursions
