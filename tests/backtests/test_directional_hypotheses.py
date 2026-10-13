"""Deterministic evidence and safety checks for non-live strategy hypotheses."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from nifty_scalper_bot.backtesting.directional_hypotheses import (
    ArchivedDirectionalHypothesis,
    ORBRangeScaledResearch,
    classify_completed_futures_bars,
    run_directional_hypothesis_research,
)
from nifty_scalper_bot.risk.cost_model import (
    estimate_round_trip_cost,
    evaluate_net_reward_risk,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    ORBProStrategyConfig,
)
from nifty_scalper_bot.strategies.elite_strategies.orb_pro import ORBProStrategy

_START = datetime(2026, 10, 1, 9, 30, tzinfo=timezone(timedelta(hours=5, minutes=30)))
_FUTURE = "NFO:NIFTY26OCTFUT"
_CE = "NFO:NIFTY26O0624000CE"
_PE = "NFO:NIFTY26O0624000PE"


def _bar(index: int, close: float, *, volume: float = 1000.0) -> dict[str, Any]:
    return {
        "timestamp": _START + timedelta(minutes=index),
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": volume,
        "is_complete": True,
        "is_provisional": False,
    }


def _momentum_bars(side: str = "CE") -> list[dict[str, Any]]:
    sign = 1 if side == "CE" else -1
    rows = [_bar(i, 24000 + sign * i * 2) for i in range(45)]
    rows[-1] = _bar(44, 24000 + sign * (43 * 2 + 12), volume=2000)
    return rows


def test_momentum_acceleration_respects_futures_side_and_completed_bar() -> None:
    for side in ("CE", "PE"):
        rows = _momentum_bars(side)
        assert classify_completed_futures_bars(rows)["momentum_acceleration"] == side

        rows[-1] = {**rows[-1], "is_complete": False}
        assert classify_completed_futures_bars(rows) == {}


def test_compression_is_based_on_prior_bands_not_breakout_candle() -> None:
    rows = [_bar(i, 24000.0) for i in range(45)]
    rows[-1] = _bar(44, 24005.0, volume=2000.0)
    assert classify_completed_futures_bars(rows)["compression_breakout"] == "CE"
    rows[-1] = _bar(44, 24000.0, volume=2000.0)
    assert "compression_breakout" not in classify_completed_futures_bars(rows)


def test_trend_pullback_requires_reclaim_after_retracement() -> None:
    rows = [_bar(i, 24000.0 + 2.0 * i) for i in range(45)]
    rows[-2] = {
        **_bar(43, 24076.0),
        "open": 24083.0,
        "high": 24084.0,
        "low": 24075.0,
    }
    rows[-1] = {
        **_bar(44, 24095.0, volume=1500.0),
        "open": 24077.0,
        "low": 24076.0,
    }
    assert classify_completed_futures_bars(rows)["trend_pullback"] == "CE"
    assert "trend_pullback" not in classify_completed_futures_bars(rows[:-1])


def test_disconnected_or_ambiguous_history_fails_closed() -> None:
    rows = _momentum_bars()
    assert classify_completed_futures_bars(rows[:-1]) == {}
    bad = list(rows)
    bad[20] = {**bad[20], "timestamp": bad[20]["timestamp"] + timedelta(minutes=2)}
    assert classify_completed_futures_bars(bad) == {}
    bad = list(rows)
    bad[-1] = {**bad[-1], "timestamp": bad[-1]["timestamp"].replace(tzinfo=None)}
    assert classify_completed_futures_bars(bad) == {}
    bad = list(rows)
    bad[-1] = {**bad[-1], "close": float("nan")}
    assert classify_completed_futures_bars(bad) == {}
    bad = list(rows)
    bad[-1] = {**bad[-1], "is_provisional": True}
    assert classify_completed_futures_bars(bad) == {}


class _HistoryEngine:
    def __init__(
        self,
        bars: list[dict[str, Any]],
        future_symbol: str = _FUTURE,
    ) -> None:
        self.bars = bars
        self.future_symbol = future_symbol

    def get_history(
        self, symbol: str, count: int, *, field: str
    ) -> list[dict[str, Any]]:
        assert field == "bars"
        return self.bars[-count:] if symbol == self.future_symbol else []

    def get_atr(self, symbol: str) -> float | None:
        return 2.0 if symbol in {_CE, _PE} else None


def test_archived_adapter_only_generates_option_buy_with_next_bar_proxy() -> None:
    rows = _momentum_bars()
    candidate = ArchivedDirectionalHypothesis(
        "momentum_acceleration", _HistoryEngine(rows)
    )
    context = {"futures_symbol": _FUTURE, "bar_timestamp": rows[-1]["timestamp"]}

    assert candidate.generate_signal(_PE, context, 60.0) is None
    assert candidate.generate_signal("NSE:NIFTY 50", context, 60.0) is None
    assert (
        candidate.generate_signal(
            _CE, {**context, "bar_timestamp": rows[-2]["timestamp"]}, 60.0
        )
        is None
    )

    signal = candidate.generate_signal(_CE, context, 60.0)
    assert signal is not None
    assert signal.action == "BUY"
    assert signal.stop_loss == pytest.approx(57.6)
    assert signal.take_profit == pytest.approx(64.8)
    assert signal.metadata["signal_family"] == "research_only"
    assert signal.metadata["contract_side"] == "CE"
    assert signal.metadata["setup_id"].startswith("research:momentum_acceleration:")


def _archive(directory: Path) -> None:
    candles = directory / "candles"
    candles.mkdir()
    for symbol, kind in (
        ("NSE:NIFTY 50", "EQ"),
        (_FUTURE, "FUT"),
        (_CE, "CE"),
        (_PE, "PE"),
    ):
        payload = {
            "symbol": symbol,
            "timestamp_convention": "bar_start",
            "instrument": {
                "instrument_type": kind,
                "name": "NIFTY",
                "lot_size": 75,
                "expiry": "2026-10-06",
            },
            "candles": [
                [
                    (_START + timedelta(minutes=i)).isoformat(),
                    100.0,
                    102.0,
                    98.0,
                    100.0,
                    1000.0,
                ]
                for i in range(4)
            ],
        }
        (candles / f"{kind}.json").write_text(json.dumps(payload))


def test_hypothesis_research_reuses_canonical_costed_scenarios(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    _archive(tmp_path)

    result = run_directional_hypothesis_research(tmp_path)
    assert result["live_equivalent"] is False
    assert result["selected_for_live"] is None
    assert result["promotion_eligible"] is False
    assert result["evidence_label"] == "RESEARCH_CANDIDATE"
    assert len(result["scenarios"]) == 3
    assert [row["slippage_bps_per_side"] for row in result["scenarios"]] == [
        10.0,
        25.0,
        50.0,
    ]
    for row in result["scenarios"]:
        assert set(row["strategies"]) == {
            "PullbackResearch",
            "CompressionResearch",
            "MomentumResearch",
        }
        assert all(
            value["metrics"]["trade_count"] == 0 for value in row["strategies"].values()
        )

    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    with pytest.raises(ValueError, match="research_requires_shadow_process"):
        run_directional_hypothesis_research(tmp_path)


def test_exact_public_archive_future_context_is_research_only() -> None:
    """Monthly public archives use a continuous context future identifier."""
    rows = _momentum_bars()
    historical_symbol = "NFO:NIFTY_F1_CONTEXT"
    candidate = ArchivedDirectionalHypothesis(
        "momentum_acceleration",
        _HistoryEngine(rows, future_symbol=historical_symbol),
    )
    context = {
        "futures_symbol": historical_symbol,
        "bar_timestamp": rows[-1]["timestamp"],
    }
    assert candidate.generate_signal(_CE, context, 60.0) is not None
    assert (
        candidate.generate_signal(
            _CE, {**context, "futures_symbol": "NFO:NIFTY_OTHER_CONTEXT"}, 60.0
        )
        is None
    )


def test_fixed_hypothesis_summary_preserves_costs_periods_and_missing_exits() -> None:
    from scripts.research_orb_multiyear import summarize_fixed_hypothesis_days

    names = (
        "ORBNetGateBaseline",
        "ORBRangeScaledResearch",
        "PullbackResearch",
        "CompressionResearch",
        "MomentumResearch",
    )

    def trade(day: str) -> dict[str, Any]:
        return {
            "entry_time": day + "T09:40:00+05:30",
            "exit_time": day + "T09:45:00+05:30",
            "entry_price": 100.0,
            "exit_price": 105.0,
            "quantity": 75,
            "gross_pnl": 375.0,
            "fees": 75.0,
            "net_pnl": 300.0,
            "duration_minutes": 5.0,
            "exit_reason": "target",
        }

    sessions = []
    for day in ("2017-10-02", "2018-10-02", "2019-10-02", "2020-10-02"):
        scenarios = []
        for bps in (10.0, 25.0, 50.0):
            strategies = {}
            for name in names:
                is_reference = name == "ORBNetGateBaseline"
                unresolved = (
                    [{"entry_time": day, "reason": "missing_exit"}]
                    if name == "MomentumResearch" and day.startswith("2020")
                    else []
                )
                strategies[name] = {
                    "trades": [trade(day)] if is_reference else [],
                    "unresolved_trades": unresolved,
                    "worst_case_stress_trades": [trade(day)] if is_reference else [],
                    "no_vote_reasons": {"no_setup": 3},
                }
            scenarios.append({"slippage_bps_per_side": bps, "strategies": strategies})
        sessions.append({"day": day, "scenarios": scenarios})

    results = summarize_fixed_hypothesis_days(sessions)
    assert len(results) == 45
    development = next(
        row
        for row in results
        if row["phase"] == "development"
        and row["candidate"] == "ORBNetGateBaseline"
        and row["slippage_bps_per_side"] == 10.0
    )
    assert development["session_count"] == 2
    assert development["metrics"]["trade_count"] == 2
    assert development["metrics"]["gross_pnl"] == 750.0
    assert development["metrics"]["fees"] == 150.0
    assert development["metrics"]["net_pnl"] == 600.0
    holdout = next(
        row
        for row in results
        if row["phase"] == "final"
        and row["candidate"] == "MomentumResearch"
        and row["slippage_bps_per_side"] == 10.0
    )
    assert holdout["metrics"]["trade_count"] == 0
    assert holdout["data_quality"]["unresolved_exit_count"] == 1
    assert holdout["data_quality"]["primary_metrics_complete"] is False
    with pytest.raises(ValueError, match="hypothesis_duplicate_session"):
        summarize_fixed_hypothesis_days(sessions + [sessions[0]])
    corrupted = [
        (
            row
            if row["day"] != "2020-10-02"
            else {
                **row,
                "scenarios": [
                    scenario
                    for scenario in row["scenarios"]
                    if scenario["slippage_bps_per_side"] != 50.0
                ],
            }
        )
        for row in sessions
    ]
    with pytest.raises(ValueError, match="hypothesis_scenario_incomplete"):
        summarize_fixed_hypothesis_days(corrupted)


def test_fixed_cohort_slippage_uses_identical_trades_without_reselection() -> None:
    """Stress realized 10-bps trades without re-running the entry RR gate."""
    from scripts.research_orb_multiyear import fixed_cohort_slippage_stress

    base_entry = 100.0
    base_exit = 104.0
    entry = base_entry * 1.001
    exit_price = base_exit * 0.999
    qty = 75
    costs = estimate_round_trip_cost(
        entry_price=entry, exit_price=exit_price, quantity=qty
    )
    trade = {
        "entry_time": "2020-01-02T10:01:00+05:30",
        "exit_time": "2020-01-02T10:05:00+05:30",
        "entry_price": entry,
        "exit_price": exit_price,
        "quantity": qty,
        "gross_pnl": (exit_price - entry) * qty,
        "fees": costs.total,
        "net_pnl": (exit_price - entry) * qty - costs.total,
        "duration_minutes": 4.0,
        "cost_source": "canonical_model_estimate",
        "exit_reason": "target",
    }
    base = {
        "phase": "final",
        "candidate": "CompressionResearch",
        "slippage_bps_per_side": 10.0,
        "trades": [trade],
        "unresolved_trades": [{"unresolved_reason": "history_gap_unresolved"}],
    }
    stress = fixed_cohort_slippage_stress([base])
    assert len(stress) == 3
    assert [row["slippage_bps_per_side"] for row in stress] == [10.0, 25.0, 50.0]
    assert all(row["metrics"]["trade_count"] == 1 for row in stress)
    assert all(row["unresolved_exit_count"] == 1 for row in stress)
    assert stress[0]["metrics"]["net_pnl"] == pytest.approx(trade["net_pnl"])
    assert (
        stress[0]["metrics"]["net_pnl"]
        > stress[1]["metrics"]["net_pnl"]
        > stress[2]["metrics"]["net_pnl"]
    )
    assert stress[1]["metrics"]["fees"] != pytest.approx(trade["fees"])
    assert all(row["method"] == "same_10bps_resolved_trade_cohort" for row in stress)
    with pytest.raises(ValueError, match="research_cost_source_unverified"):
        fixed_cohort_slippage_stress(
            [{**base, "trades": [{**trade, "cost_source": "unverified"}]}]
        )


def test_orb_15minute_range_scaling_research_only() -> None:
    """Expose an ATR horizon mismatch, preserving every other structural gate."""
    config = ORBProStrategyConfig(orb_minutes=15)
    engine = _HistoryEngine(_momentum_bars())
    live_reference = ORBProStrategy(config, engine)
    research = ORBRangeScaledResearch(config, engine)
    evidence = {
        "side": "CE",
        "direction": "CE",
        "penetration_atr": 0.3,
        "volume_ratio": 1.5,
        "opening_range_atr": 3.5,
        "indicators": {"futures_vwap_slope": 0.05},
    }
    assert live_reference._structural_evidence(**evidence)[0] is False
    passed, reasons, balanced, _, _ = research._structural_evidence(**evidence)
    assert passed is True
    assert balanced is True
    assert "balanced_opening_range" in reasons
    assert research.research_gate_counts["evaluated"] == 1
    assert research.research_gate_counts["raw_range_pass"] == 0
    assert research.research_gate_counts["scaled_range_pass"] == 1

    blocked, *_ = research._structural_evidence(**{**evidence, "volume_ratio": 0.4})
    assert blocked is False
    assert research.research_gate_counts["failed_underlying_volume_confirmation"] == 1
    assert research.name == "ORBPro"
    assert live_reference._structural_evidence(**evidence)[0] is False


def test_exit_lifecycle_ablation_preserves_original_and_paired_entries() -> None:
    """Different holding durations must not masquerade as matched trade counts."""
    from scripts.research_orb_multiyear import summarize_exit_lifecycle_ablation

    families = ("PullbackResearch", "CompressionResearch", "MomentumResearch")

    def make_trade(entry: str, price: float) -> dict[str, Any]:
        return {
            "symbol": _CE,
            "entry_time": entry,
            "exit_time": "2020-01-02T11:01:00+05:30",
            "entry_price": 100.0,
            "exit_price": 101.0,
            "quantity": 75,
            "gross_pnl": price + 70.0,
            "fees": 70.0,
            "net_pnl": price,
            "duration_minutes": 2.0,
            "exit_reason": "stop",
        }

    first = make_trade("2020-01-02T10:00:00+05:30", -80.0)
    second = make_trade("2020-01-02T10:30:00+05:30", 100.0)
    longer = {**first, "net_pnl": 130.0, "gross_pnl": 200.0}
    no_trailing = {
        family: {
            "trades": [longer],
            "unresolved_trades": [{"reason": "history_gap_unresolved"}],
        }
        for family in families
    }
    daily = [
        {
            "day": "2020-01-02",
            "scenarios": [
                {
                    "slippage_bps_per_side": 10.0,
                    "strategies": {
                        family: {"trades": [first, second], "unresolved_trades": []}
                        for family in families
                    },
                }
            ],
            "exit_ablation": no_trailing,
        }
    ]
    summary = summarize_exit_lifecycle_ablation(daily)
    assert len(summary) == 9
    final = next(
        row
        for row in summary
        if row["phase"] == "final" and row["candidate"] == "PullbackResearch"
    )
    assert final["original"]["metrics"]["trade_count"] == 2
    assert final["no_trailing"]["metrics"]["trade_count"] == 1
    assert final["identical_entry_count"] == 1
    assert final["identical_entry_original_metrics"]["net_pnl"] == -80.0
    assert final["identical_entry_no_trailing_metrics"]["net_pnl"] == 130.0
    assert final["no_trailing"]["unresolved_exit_count"] == 1
    with pytest.raises(ValueError, match="exit_ablation_scenario_incomplete"):
        summarize_exit_lifecycle_ablation([{**daily[0], "exit_ablation": {}}])
    with pytest.raises(ValueError, match="exit_ablation_duplicate_trade_identity"):
        corrupted = [
            {
                **daily[0],
                "exit_ablation": {
                    **no_trailing,
                    "PullbackResearch": {
                        "trades": [longer, longer],
                        "unresolved_trades": [],
                    },
                },
            }
        ]
        summarize_exit_lifecycle_ablation(corrupted)



def test_orb_target_cost_feasibility_without_weakening_rr_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same 8% stop can fail at 1.8R and pass at a farther 2.2R target."""
    monkeypatch.setenv("MIN_NET_REWARD_RISK", "1.5")
    entry = 55.0
    stop = entry * 0.92
    risk = entry - stop
    first = evaluate_net_reward_risk(
        entry_price=entry,
        stop_price=stop,
        target_price=entry + risk * 1.8,
        quantity=65,
    )
    second = evaluate_net_reward_risk(
        entry_price=entry,
        stop_price=stop,
        target_price=entry + risk * 2.2,
        quantity=65,
    )
    assert first.allowed is False
    assert second.allowed is True
    assert first.minimum == second.minimum == 1.5
    assert first.target_cost.total > 0
    assert second.net_rr > first.net_rr


def test_orb_target_sensitivity_distinguishes_cost_rejects_and_shared_trades() -> None:
    """Never attribute altered trade admission to an improved held-out edge."""
    from scripts.research_orb_multiyear import summarize_orb_target_sensitivity

    def trade(entry_time: str, pnl: float) -> dict[str, Any]:
        return {
            "symbol": _CE,
            "entry_time": entry_time,
            "exit_time": "2020-01-02T10:59:00+05:30",
            "entry_price": 55.0,
            "exit_price": 56.0,
            "quantity": 65,
            "gross_pnl": pnl + 60.0,
            "fees": 60.0,
            "net_pnl": pnl,
            "duration_minutes": 3.0,
            "exit_reason": "stop",
        }

    shared = trade("2020-01-02T10:00:00+05:30", -90.0)
    other = trade("2020-01-02T10:30:00+05:30", 200.0)
    changed_shared = {**shared, "net_pnl": 110.0, "gross_pnl": 170.0}
    original = {
        "trades": [shared, other],
        "unresolved_trades": [],
        "no_vote_reasons": {"cost_net_rr_rejected": 15},
    }
    changed = {
        "trades": [changed_shared],
        "unresolved_trades": [{"reason": "ambiguous_intrabar_stop_target"}],
        "no_vote_reasons": {"cost_net_rr_rejected": 2},
    }
    day = {
        "day": "2020-01-02",
        "scenarios": [{
            "slippage_bps_per_side": 10.0,
            "strategies": {"ORBRangeScaledResearch": original},
        }],
        "orb_target_2_2": changed,
    }
    result = summarize_orb_target_sensitivity([day])
    assert len(result) == 3
    final = next(row for row in result if row["phase"] == "final")
    assert final["baseline"]["metrics"]["trade_count"] == 2
    assert final["target_2_2"]["metrics"]["trade_count"] == 1
    assert final["baseline"]["cost_net_rr_rejections"] == 15
    assert final["target_2_2"]["cost_net_rr_rejections"] == 2
    assert final["target_2_2"]["unresolved_exit_count"] == 1
    assert final["identical_entry_count"] == 1
    assert final["identical_entry_baseline"]["net_pnl"] == -90.0
    assert final["identical_entry_target_2_2"]["net_pnl"] == 110.0
    with pytest.raises(ValueError, match="orb_target_sensitivity_incomplete"):
        summarize_orb_target_sensitivity([{**day, "orb_target_2_2": None}])
    with pytest.raises(ValueError, match="orb_target_duplicate_entry"):
        summarize_orb_target_sensitivity([
            {**day, "orb_target_2_2": {
                **changed,
                "trades": [changed_shared, changed_shared],
            }}
        ])
