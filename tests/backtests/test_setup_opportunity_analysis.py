from __future__ import annotations

import pytest

from nifty_scalper_bot.backtesting.setup_opportunity_analysis import (
    ExperimentRecord,
    canonicalize_setup_opportunities,
    compare_score_policy,
    label_forward_option_buy_path,
    validate_experiment_record,
)


def _decision(
    timestamp: str,
    *,
    event_name: str = "candidate.blocked",
    setup_id: str = "vwap:CE:anchor",
    symbol: str = "NFO:NIFTYCE",
    reason: str = "alpha_below_threshold",
) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "event_name": event_name,
        "reason_code": reason,
        "direction": "CE",
        "selected_candidate": symbol,
        "meta": {
            "research_context": {
                "strategy": "VWAPPro",
                "setup_id": setup_id,
                "setup_name": "premium_vwap_reclaim",
                "contract_side": "CE",
                "regime": "TREND",
                "raw_setup_score": 6.4,
            }
        },
    }


def test_opportunity_canonicalization_deduplicates_repeated_setup_evaluations() -> None:
    rows = [
        _decision("2026-09-30T04:00:00Z"),
        _decision("2026-09-30T04:00:05Z"),
        _decision(
            "2026-09-30T04:00:10Z",
            event_name="candidate.approved",
            reason="order_submitted",
        ),
    ]

    opportunities = canonicalize_setup_opportunities(rows)

    assert len(opportunities) == 1
    assert opportunities[0].decision_count == 3
    assert opportunities[0].approved is True
    assert opportunities[0].final_reason == "order_submitted"
    assert opportunities[0].setup_id == "vwap:CE:anchor"


def test_opportunity_canonicalization_never_invents_missing_setup_identity() -> None:
    row = _decision("2026-09-30T04:00:00Z")
    row["meta"]["research_context"].pop("setup_id")  # type: ignore[index,union-attr]

    assert canonicalize_setup_opportunities([row]) == ()


def test_opportunity_canonicalization_fails_closed_on_conflicting_symbol() -> None:
    rows = [
        _decision("2026-09-30T04:00:00Z"),
        _decision("2026-09-30T04:00:05Z", symbol="NFO:NIFTYOTHERCE"),
    ]

    assert canonicalize_setup_opportunities(rows) == ()


def test_forward_path_uses_executable_bid_and_reports_mfe_mae_in_r() -> None:
    label = label_forward_option_buy_path(
        [
            {"timestamp": 100.0, "bid": 100.0, "ltp": 102.0},
            {"timestamp": 110.0, "bid": 112.0, "ltp": 115.0},
            {"timestamp": 120.0, "bid": 94.0, "ltp": 98.0},
            {"timestamp": 130.0, "bid": 108.0, "ltp": 111.0},
        ],
        decision_ts=100.0,
        entry_price=100.0,
        risk_points=10.0,
        horizon_seconds=30.0,
    )

    assert label is not None
    assert label.mfe_r == 1.2
    assert label.mae_r == 0.6
    assert label.time_to_mfe_seconds == 10.0
    assert label.time_to_mae_seconds == 20.0
    assert label.terminal_r == 0.8
    assert label.hit_positive_1r is True
    assert label.hit_negative_1r is False


def test_forward_path_ignores_out_of_horizon_and_invalid_prices() -> None:
    label = label_forward_option_buy_path(
        [
            {"timestamp": 99.0, "bid": 200.0},
            {"timestamp": 105.0, "bid": 105.0},
            {"timestamp": 106.0, "bid": 0.0},
            {"timestamp": 200.0, "bid": 250.0},
        ],
        decision_ts=100.0,
        entry_price=100.0,
        risk_points=10.0,
        horizon_seconds=20.0,
    )

    assert label is not None
    assert label.observation_count == 1
    assert label.mfe_r == 0.5
    assert label.terminal_r == 0.5


def test_score_policy_can_compare_regime_weight_with_neutral_without_selecting_winner() -> None:
    samples = [
        {"raw_setup_score": 6.0, "regime_weight": 1.2, "outcome_r": 1.0},
        {"raw_setup_score": 6.5, "regime_weight": 0.8, "outcome_r": -0.5},
        {"raw_setup_score": 7.0, "regime_weight": 1.0, "outcome_r": 0.2},
    ]

    weighted = compare_score_policy(samples, threshold=6.5, use_regime_weight=True)
    neutral = compare_score_policy(samples, threshold=6.5, use_regime_weight=False)

    assert weighted.policy == "regime_weighted"
    assert weighted.selected == 2
    assert weighted.mean_r == 0.6
    assert neutral.policy == "neutral_weight"
    assert neutral.selected == 2
    assert neutral.mean_r == -0.15


def test_experiment_record_requires_strictly_later_oos_window_and_unique_id() -> None:
    record = ExperimentRecord(
        experiment_id="vwap-domain-001",
        hypothesis="Underlying VWAP confirmation improves post-cost OOS R.",
        parameters={"variant": "underlying_confirmation"},
        train_start="2026-01-01T00:00:00Z",
        train_end="2026-06-30T23:59:59Z",
        test_start="2026-07-01T00:00:00Z",
        test_end="2026-08-31T23:59:59Z",
        cost_model="broker_effective_costs_v1",
    )

    assert validate_experiment_record(record) is record
    with pytest.raises(ValueError, match="duplicate experiment_id"):
        validate_experiment_record(record, existing_ids=("vwap-domain-001",))


def test_experiment_record_rejects_train_test_overlap() -> None:
    record = ExperimentRecord(
        experiment_id="exit-001",
        hypothesis="Alternative time stop changes post-cost R.",
        parameters={},
        train_start="2026-01-01T00:00:00Z",
        train_end="2026-06-30T23:59:59Z",
        test_start="2026-06-30T23:59:59Z",
        test_end="2026-07-31T23:59:59Z",
        cost_model="broker_effective_costs_v1",
    )

    with pytest.raises(ValueError, match="strictly later"):
        validate_experiment_record(record)
