"""Causal session-VWAP reclaim, cash risk and guarded live-pilot tests.

These fixtures assert *logic*, not executable historical NIFTY profitability.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from nifty_scalper_bot.config.strategy_taxonomy import canonical_strategy_role
from nifty_scalper_bot.execution.position_manager import PositionManager
from nifty_scalper_bot.strategies.elite_strategies import (
    range_vwap_reclaim as range_module,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    RangeVWAPReclaimStrategyConfig,
)
from nifty_scalper_bot.strategies.elite_strategies.range_vwap_reclaim import (
    RangeVWAPReclaimStrategy,
    _restore_authenticated_session_bars,
    assess_option_cost_risk,
    assess_range_vwap_reclaim,
    has_ist_session_anchor,
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
    result = {
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
        "futures_vwap": 25250.23,
        "futures_vwap_source": "active_future_quote_vwap",
        "bid": 99.80,
        "ask": 100.0,
        "atr": 6.0,
        "lot_size": 65,
        "days_to_expiry": 8,
        "delta": 0.55,
        # Synthetic known theta; a missing-theta haircut would correctly
        # reject this narrow 5-point futures-to-VWAP opportunity.
        "theta": -1.0,
        "theta_period": "trading_day",
        "delta_underlying": "NIFTY_SPOT",
        "vega_per_iv_point": 0.01,
        "vega_convention": "per_iv_percentage_point",
        "range_spot_ltp": 25230.0,
        "quote_update_version": 123,
        "ofi_ready": True,
        "ofi_3s_complete": True,
        "ofi_update_count_3s": 4,
        "ofi_3s_span_seconds": 1.8,
        "ofi_3s_normalized": 0.30,
        "ofi_1s_normalized": 0.20,
    }
    result.update(overrides)
    result.setdefault(
        "range_orderflow_confirmation",
        {
            "candidate_symbol": OPTION,
            "quote_update_version": result.get("quote_update_version"),
            "quote_update_version_source": "quote_update_version",
            "vote_timestamp": decision_at().timestamp(),
            "quote_readiness_allowed": True,
            "ofi_reversal_holdoff_evaluated": True,
            "ofi_reversal_holdoff_active": False,
            "range_research_holdoff_active": False,
            **{
                key: result.get(key)
                for key in (
                    "ofi_ready",
                    "ofi_3s_complete",
                    "ofi_update_count_3s",
                    "ofi_3s_span_seconds",
                    "ofi_3s_normalized",
                    "ofi_1s_normalized",
                    "depth_imbalance",
                )
            },
        },
    )
    return result


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


def test_volatility_pressure_uses_only_preconfirmation_futures_history() -> None:
    baseline = assess_range_vwap_reclaim(
        bars(), decision_at=decision_at(), config=RangeVWAPReclaimStrategyConfig()
    )
    other = bars()
    other[-1]["open"] += 0.1
    other[-1]["high"] += 0.1
    other[-1]["low"] += 0.1
    other[-1]["close"] += 0.1
    changed = assess_range_vwap_reclaim(
        other, decision_at=decision_at(), config=RangeVWAPReclaimStrategyConfig()
    )
    for key in (
        "prior_5m_to_14m_true_range_ratio",
        "prior_5m_directional_efficiency",
    ):
        assert baseline[key] == changed[key]
        assert math.isfinite(baseline[key])
    assert baseline["prior_5m_to_14m_true_range_ratio"] > 0.0
    assert 0 <= baseline["prior_5m_directional_efficiency"] <= 1


def test_live_quote_rebreak_of_frozen_futures_band_is_not_traded(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    proof = assess_range_vwap_reclaim(
        bars(), decision_at=decision_at(), config=RangeVWAPReclaimStrategyConfig()
    )
    strategy = RangeVWAPReclaimStrategy(
        RangeVWAPReclaimStrategyConfig(), History(bars())
    )
    strategy._regime_bar_confirmed(FUTURE, bars()[-2]["timestamp"])
    strategy._evaluate_signal(
        OPTION,
        indicators(futures_ltp=proof["lower_band_before_confirmation"] - 0.01),
        100.0,
    )
    assert strategy.last_no_vote_reason == "range_futures_rebroke_outer_band"


def test_option_reward_repriced_to_current_futures_not_completed_candle(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    observed = {}

    def option_model(_quote, **kwargs):
        observed.update(kwargs)
        return {
            "qualified": True,
            "worst_stop_loss_including_costs": 650.0,
            "net_reward_risk": 1.6,
            "stop_price_model": 94.0,
            "vwap_limited_target_price_model": 112.0,
        }

    monkeypatch.setattr(range_module, "assess_option_cost_risk", option_model)
    strategy = RangeVWAPReclaimStrategy(
        RangeVWAPReclaimStrategyConfig(), History(bars())
    )
    strategy._regime_bar_confirmed(FUTURE, bars()[-2]["timestamp"])
    current_future = 25247.0
    signal = strategy._evaluate_signal(
        OPTION, indicators(futures_ltp=current_future), 100.0
    )
    assert signal is not None, strategy.last_no_vote_reason
    vwap = signal.metadata["exchange_session_vwap_at_decision"]
    completed_close = signal.metadata["range_structure"]["underlying_entry_close"]
    assert observed["futures_target_distance"] == min(
        abs(vwap - completed_close), abs(vwap - current_future)
    )
    assert observed["futures_stop_distance"] == max(
        abs(completed_close - signal.metadata["range_futures_invalidation"]),
        abs(current_future - signal.metadata["range_futures_invalidation"]),
    )
    assert (
        signal.metadata["option_cost_risk"]["target_points_removed_due_to_quote_drift"]
        > 0
    )


def test_excursion_research_records_original_regime_rejection_only_once() -> None:
    cfg = RangeVWAPReclaimStrategyConfig(max_secondary_center_deviation_atr=0.001)
    structural = assess_range_vwap_reclaim(
        bars(), decision_at=decision_at(), config=cfg
    )
    assert structural["qualified"] is False
    assert structural["reason"] == "session_rolling_center_disagree"
    assert structural["excursion_side"] == "CE"
    assert structural["excursion_first_bar_timestamp"] == bars()[-2]["timestamp"]
    strategy = RangeVWAPReclaimStrategy(cfg, History(bars()))
    context = indicators(regime="TREND", is_selected_option=True)
    strategy.observe_research_excursion(
        OPTION, context, blocked_reason="strategy_regime_or_direction_ineligible"
    )
    first = strategy.get_research_snapshot()["opportunities"]
    assert len(first) == 1
    assert first[0]["dominant_rejection"] == "strategy_regime_or_direction_ineligible"
    assert first[0]["center_gap_atr"] > 0
    assert first[0]["volatility_ratio"] > 0
    assert first[0]["quote"]["ask"] == 100.0
    assert first[0]["status"] == "rejected"
    strategy.observe_research_excursion(
        OPTION, context, blocked_reason="strategy_regime_or_direction_ineligible"
    )
    assert len(strategy.get_research_snapshot()["opportunities"]) == 1
    assert first[0]["observed_decisions"] == 1


def test_qualified_excursion_reuses_first_futures_identity_not_option_bar() -> None:
    strategy = RangeVWAPReclaimStrategy(
        RangeVWAPReclaimStrategyConfig(), History(bars())
    )
    context = indicators()
    strategy._evaluate_signal(OPTION, context, 100.0)
    records = strategy.get_research_snapshot()["opportunities"]
    assert len(records) == 1
    record = records[0]
    assert record["excursion_first_bar"] == bars()[-2]["timestamp"]
    assert record["identity"].startswith(FUTURE + ":CE:")
    assert record["symbol"] == OPTION
    assert record["option_fills_verified"] is False


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
        "theta_period": "trading_day",
        "delta_underlying": "NIFTY_SPOT",
        "spot_ltp": 25230.0,
        "futures_ltp": 25245.0,
        "futures_atr": 7.0,
    }
    a = assess_option_cost_risk(
        quote,
        config=cfg,
        side="CE",
        futures_target_distance=35,
        futures_stop_distance=3,
    )
    assert a["qualified"] is True, a
    # Bid-based stop activation and adverse liquidation are distinct prices.
    assert (
        0 < a["modeled_stressed_stop_fill_price"] < a["stop_price_model"] < quote["ask"]
    )
    assert a["worst_stop_loss_including_costs"] <= 773
    assert a["net_reward_risk"] >= 1.5
    assert a["projected_premium_gain_points"] < 0.55 * 35
    assert a["theta_cost_source"] == "observed_theta_trading_day_linear_stress"
    assert (
        a["pricing_scenarios"]["at_vwap_fixed_basis"]["net_rr"] >= a["net_reward_risk"]
    )
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
    quote = {
        "bid": 99.8,
        "ask": 100,
        "atr": 6,
        "lot_size": 65,
        "delta": 0.55,
        "delta_underlying": "NIFTY_SPOT",
        "spot_ltp": 25230,
        "futures_ltp": 25245,
        "futures_atr": 7,
    }

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
    assert signal is not None, strategy.last_no_vote_reason
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
        {"futures_vwap": None},
        {"futures_vwap": 25300},
        {"futures_vwap_source": "derived_index_vwap"},
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
            range_orderflow_confirmation={
                **indicators()["range_orderflow_confirmation"],
                "candidate_symbol": "NFO:NIFTY26NOV25250PE",
            },
        ),
        100.0,
    )
    assert result is not None, strategy.last_no_vote_reason
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


def test_unresolved_broker_reservation_cannot_rearm_on_vwap_touch() -> None:
    strategy = RangeVWAPReclaimStrategy(
        RangeVWAPReclaimStrategyConfig(), History(bars())
    )
    key = (FUTURE, "CE", "2026-11-18")
    setup_id = "range_session_vwap:unknown-order"
    strategy._seen.add(setup_id)
    strategy._consumed_excursions.add(key)
    strategy._pending_excursions[setup_id] = key
    strategy._maybe_reset_excursions(
        FUTURE,
        {
            "underlying_latest_close": 25249.0,
            "session_vwap_before_confirmation": 25250.0,
            "futures_atr": 6.0,
        },
    )
    assert key in strategy._consumed_excursions


def test_reclaim_does_not_require_redundant_remaining_gap() -> None:
    cfg = RangeVWAPReclaimStrategyConfig(min_remaining_center_gap_atr=999.0)
    proof = assess_range_vwap_reclaim(bars(), decision_at=decision_at(), config=cfg)
    assert proof["qualified"] is True, proof
    assert proof["remaining_vwap_distance_atr"] > 0
    assert proof["candidate_session_anchored_band_points"] > 0


def test_orderflow_absence_is_not_false_holdoff(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    strategy = RangeVWAPReclaimStrategy(
        RangeVWAPReclaimStrategyConfig(min_net_reward_risk=0), History(bars())
    )
    strategy._regime_bar_confirmed(FUTURE, bars()[-2]["timestamp"])
    assert (
        strategy._evaluate_signal(
            OPTION, indicators(range_orderflow_confirmation=None), 100.0
        )
        is None
    )
    assert strategy.last_no_vote_reason == "range_orderflow_handoff_missing"
    report = strategy.get_research_snapshot()
    assert report["completed_band_reclaims"] == 1
    assert report["shadow_net_expectancy"] is None


def test_orderflow_mismatched_quote_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    strategy = RangeVWAPReclaimStrategy(
        RangeVWAPReclaimStrategyConfig(min_net_reward_risk=0), History(bars())
    )
    strategy._regime_bar_confirmed(FUTURE, bars()[-2]["timestamp"])
    proof = {
        **indicators()["range_orderflow_confirmation"],
        "quote_update_version": 999,
    }
    assert (
        strategy._evaluate_signal(
            OPTION, indicators(range_orderflow_confirmation=proof), 100.0
        )
        is None
    )
    assert strategy.last_no_vote_reason == (
        "range_orderflow_quote_or_holdoff_unverified"
    )


def test_spot_basis_and_delta_provenance_required() -> None:
    cfg = RangeVWAPReclaimStrategyConfig()
    quote = {
        "bid": 99.8,
        "ask": 100,
        "atr": 6,
        "lot_size": 65,
        "delta": 0.55,
        "futures_atr": 7,
    }
    inputs = {
        "config": cfg,
        "side": "CE",
        "futures_target_distance": 35,
        "futures_stop_distance": 3,
    }
    assert assess_option_cost_risk(quote, **inputs)["reason"] == (
        "spot_futures_basis_unverified"
    )
    quote.update(spot_ltp=25230, futures_ltp=25245)
    assert assess_option_cost_risk(quote, **inputs)["reason"] == (
        "option_delta_underlying_unverified"
    )
    quote["delta_underlying"] = "NIFTY_SPOT"
    result = assess_option_cost_risk(quote, **inputs)
    assert result["model_only"] is True
    assert result["observed_basis_points"] == 15
    assert result["pricing_scenarios"]["stalled_time_stop"]["net_reward"] < 0


def test_authenticated_history_recovery_preserves_open_anchor() -> None:
    live = bars()[5:]
    recovered = []
    for row in bars()[:5]:
        bar = dict(row)
        start = datetime.fromisoformat(bar["timestamp"])
        bar.update(
            symbol=FUTURE,
            source="local_persisted_1m_futures",
            available_at=(start + timedelta(minutes=1)).isoformat(),
        )
        recovered.append(bar)
    merged, status = _restore_authenticated_session_bars(
        live, recovered, future=FUTURE, decision_at=decision_at()
    )
    assert status == "authenticated_local_session_recovery"
    assert assess_range_vwap_reclaim(
        merged, decision_at=decision_at(), config=RangeVWAPReclaimStrategyConfig()
    )["qualified"]
    recovered[0]["source"] = "unverified_broker_snapshot"
    _, status = _restore_authenticated_session_bars(
        live, recovered, future=FUTURE, decision_at=decision_at()
    )
    assert status == "unverified_recovered_bar"


def test_live_range_pilot_emits_executable_stop_target_only_with_proof(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("ENABLE_RANGE_VWAP_LIVE", "true")
    monkeypatch.setenv("PAPER__ENABLED", "false")
    monkeypatch.setenv("PAPER_MODE", "false")
    monkeypatch.setenv("SHADOW_MODE", "false")
    # Isolate the approval plumbing from the separate option cost-model tests.
    monkeypatch.setattr(
        range_module,
        "assess_option_cost_risk",
        lambda *_args, **_kwargs: {
            "qualified": True,
            "worst_stop_loss_including_costs": 650.0,
            "net_reward_risk": 1.6,
            "stop_price_model": 94.0,
            "vwap_limited_target_price_model": 112.0,
            "model_only": True,
        },
    )
    engine = History(bars()[:-1])
    strategy = RangeVWAPReclaimStrategy(RangeVWAPReclaimStrategyConfig(), engine)
    monkeypatch.setattr(
        PositionManager,
        "_trading_date_ist",
        staticmethod(lambda now=None: "2026-11-18"),
    )
    strategy.attach_persistent_guard(PositionManager(str(tmp_path / "positions.json")))
    strategy._evaluate_signal(
        OPTION,
        indicators(decision_at=(decision_at() - timedelta(minutes=1)).isoformat()),
        100.0,
    )
    engine.history = bars()
    signal = strategy._evaluate_signal(OPTION, indicators(), 100.0)
    assert signal is not None, strategy.last_no_vote_reason
    assert signal.metadata["role"] == "trigger"
    assert signal.metadata["can_trigger"] is True
    assert signal.metadata["research_only"] is False
    assert signal.metadata["range_option_flow_verified"] is True
    assert signal.stop_loss == 94.0
    assert signal.target == 112.0


def test_live_range_pilot_rejects_unpriced_or_overbudget_signal(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("ENABLE_RANGE_VWAP_LIVE", "true")
    monkeypatch.setenv("PAPER__ENABLED", "false")
    monkeypatch.setenv("PAPER_MODE", "false")
    monkeypatch.setenv("SHADOW_MODE", "false")
    for risk, stop, target in (
        (774.0, 94.0, 112.0),
        (650.0, 105.0, 112.0),
        (650.0, 94.0, 99.0),
    ):
        monkeypatch.setattr(
            range_module,
            "assess_option_cost_risk",
            lambda *_args, **_kwargs: {
                "qualified": True,
                "worst_stop_loss_including_costs": risk,
                "net_reward_risk": 1.6,
                "stop_price_model": stop,
                "vwap_limited_target_price_model": target,
                "model_only": True,
            },
        )
        strategy = RangeVWAPReclaimStrategy(
            RangeVWAPReclaimStrategyConfig(), History(bars())
        )
        strategy._regime_bar_confirmed(FUTURE, bars()[-2]["timestamp"])
        assert strategy._evaluate_signal(OPTION, indicators(), 100.0) is None


def test_range_setup_lifecycle_reuses_manager_broker_callbacks() -> None:
    strategy = RangeVWAPReclaimStrategy(
        RangeVWAPReclaimStrategyConfig(), History(bars())
    )
    setup_id = "range_session_vwap:NFO:NIFTY26NOVFUT:CE:2026-11-18T10:01:00+05:30"
    key = (FUTURE, "CE", "2026-11-18")
    opportunity = "one"
    strategy._seen.add(setup_id)
    strategy._consumed_excursions.add(key)
    strategy._pending_excursions[setup_id] = key
    strategy._opportunities[opportunity] = {
        "status": "cost_eligible",
        "modeled_cost_eligible": True,
        "setup_id": setup_id,
    }

    # Wrong side, ambiguous broker rejection and stale setup are never released.
    strategy.notify_entry_rejected(
        "PE", setup_id=setup_id, reason="no_affordable_execution_candidate"
    )
    strategy.notify_entry_rejected(
        "CE", setup_id="other", reason="no_affordable_execution_candidate"
    )
    strategy.notify_entry_rejected(
        "CE", setup_id=setup_id, reason="broker_submit_unknown"
    )
    assert setup_id in strategy._seen and key in strategy._consumed_excursions

    # Runner invokes this only after a deterministic no-broker affordability veto.
    strategy.notify_entry_rejected(
        "CE", setup_id=setup_id, reason="no_affordable_execution_candidate"
    )
    assert setup_id not in strategy._seen
    assert key not in strategy._consumed_excursions
    assert setup_id not in strategy._pending_excursions
    report = strategy.get_research_snapshot()
    assert report["cost_eligible_candidates"] == 1
    assert report["prebroker_capacity_rejections"] == 1
    assert report["broker_orders_accepted"] == 0

    # Accepted order claims remain consumed even on subsequent rejection calls.
    strategy._seen.add(setup_id)
    strategy._consumed_excursions.add(key)
    strategy._pending_excursions[setup_id] = key
    strategy.notify_entry_accepted("CE", setup_id=setup_id)
    strategy.notify_entry_rejected(
        "CE", setup_id=setup_id, reason="no_affordable_execution_candidate"
    )
    assert setup_id in strategy._seen
    assert key in strategy._consumed_excursions
    assert report["option_fills_verified"] is False
    assert strategy.get_research_snapshot()["broker_orders_accepted"] == 1


def test_unarmed_range_pilot_does_not_consume_setup(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_RANGE_VWAP_LIVE", "false")
    engine = History(bars())
    strategy = RangeVWAPReclaimStrategy(
        RangeVWAPReclaimStrategyConfig(min_net_reward_risk=0.0), engine
    )
    strategy._regime_bar_confirmed(FUTURE, bars()[-2]["timestamp"])
    assert strategy._evaluate_signal(OPTION, indicators(), 100.0) is None
    assert not strategy._seen
    assert not strategy._consumed_excursions


def test_session_open_anchor_supports_utc_and_rejects_stale_or_naive() -> None:
    observed = [{"timestamp": "2026-11-19T03:45:00Z", "is_complete": True}]
    decision = datetime(2026, 11, 19, 10, 0, tzinfo=IST)
    assert has_ist_session_anchor(observed, decision_at=decision)
    assert has_ist_session_anchor(
        [{"timestamp": "2026-11-19T09:15:00+05:30"}],
        decision_at=decision,
    )
    assert not has_ist_session_anchor(
        [{"timestamp": "2026-11-18T03:45:00Z"}],
        decision_at=decision,
    )
    assert not has_ist_session_anchor(
        [{"timestamp": "2026-11-19T09:15:00"}],
        decision_at=decision,
    )
    assert not has_ist_session_anchor(
        [{"timestamp": "2026-11-19T03:46:00Z"}],
        decision_at=decision,
    )
    assert not has_ist_session_anchor([], decision_at=decision)
    assert not has_ist_session_anchor(observed, decision_at="bad_timestamp")


def test_live_range_durable_guard_survives_restart_and_preserves_unknown_order(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("ENABLE_RANGE_VWAP_LIVE", "true")
    monkeypatch.setenv("PAPER__ENABLED", "false")
    monkeypatch.setenv("PAPER_MODE", "false")
    monkeypatch.setenv("SHADOW_MODE", "false")
    monkeypatch.setattr(
        PositionManager,
        "_trading_date_ist",
        staticmethod(lambda now=None: "2026-11-18"),
    )
    file = tmp_path / "positions.json"
    owner = PositionManager(str(file))
    strategy = RangeVWAPReclaimStrategy(
        RangeVWAPReclaimStrategyConfig(), History(bars())
    )
    strategy.attach_persistent_guard(owner)
    assert strategy._load_persistent_guard(decision_at())
    setup_id = "range_session_vwap:NFO:NIFTY26NOVFUT:CE:2026-11-18T09:59:00+05:30"
    excursion = (FUTURE, "CE", "2026-11-18")
    strategy._seen.add(setup_id)
    strategy._consumed_excursions.add(excursion)
    strategy._pending_excursions[setup_id] = excursion
    assert strategy._persist_guard()

    # Reload the exact canonical position snapshot; no separate guard file.
    restored_owner = PositionManager(str(file))
    restored = RangeVWAPReclaimStrategy(
        RangeVWAPReclaimStrategyConfig(), History(bars())
    )
    restored.attach_persistent_guard(restored_owner)
    assert restored._load_persistent_guard(decision_at())
    assert setup_id in restored._seen
    assert excursion in restored._consumed_excursions
    restored.notify_entry_rejected(
        "CE", setup_id=setup_id, reason="broker_submit_unknown"
    )
    assert setup_id in restored._seen
    assert excursion in restored._consumed_excursions

    # Deterministic PRE-broker rejection may release, and release is durable.
    restored.notify_entry_rejected(
        "CE", setup_id=setup_id, reason="no_affordable_execution_candidate"
    )
    again = RangeVWAPReclaimStrategy(RangeVWAPReclaimStrategyConfig(), History(bars()))
    again.attach_persistent_guard(PositionManager(str(file)))
    assert again._load_persistent_guard(decision_at())
    assert setup_id not in again._seen
    assert excursion not in again._consumed_excursions


def test_range_guard_fails_closed_without_persisted_owner(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("ENABLE_RANGE_VWAP_LIVE", "true")
    monkeypatch.setenv("PAPER__ENABLED", "false")
    monkeypatch.setenv("PAPER_MODE", "false")
    monkeypatch.setenv("SHADOW_MODE", "false")
    strategy = RangeVWAPReclaimStrategy(
        RangeVWAPReclaimStrategyConfig(), History(bars())
    )
    assert not strategy._load_persistent_guard(decision_at())
    assert not strategy._persist_guard()


def test_range_guard_does_not_authorize_memory_only_write(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("ENABLE_RANGE_VWAP_LIVE", "true")
    monkeypatch.setenv("PAPER__ENABLED", "false")
    monkeypatch.setenv("PAPER_MODE", "false")
    monkeypatch.setenv("SHADOW_MODE", "false")
    monkeypatch.setattr(
        PositionManager,
        "_trading_date_ist",
        staticmethod(lambda now=None: "2026-11-18"),
    )
    owner = PositionManager(str(tmp_path / "positions.json"))
    strategy = RangeVWAPReclaimStrategy(
        RangeVWAPReclaimStrategyConfig(), History(bars())
    )
    strategy.attach_persistent_guard(owner)
    assert strategy._load_persistent_guard(decision_at())

    def no_disk_write(*_args, **_kwargs) -> None:
        raise OSError("test simulated disk failure")

    monkeypatch.setattr(owner, "save_state", no_disk_write)
    strategy._seen.add("range_session_vwap:NFO:NIFTY26NOVFUT:CE:test")
    strategy._consumed_excursions.add((FUTURE, "CE", "2026-11-18"))
    assert not strategy._persist_guard()
