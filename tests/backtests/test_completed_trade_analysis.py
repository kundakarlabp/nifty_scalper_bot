from __future__ import annotations

import pytest

from nifty_scalper_bot.backtesting.completed_trade_analysis import (
    attribution_readiness,
    calibrate_signal_scores,
    canonicalize_completed_trades,
    chronological_post_cost_blocks,
    chronological_walk_forward,
    execution_data_quality,
    post_cost_attribution_groups,
    summarize_completed_trades,
    summarize_candidate_decisions,
    walk_forward_stability,
)


def _trade(
    trade_id: str,
    closed_at: float,
    *,
    strategy: str = "VWAPPro",
    gross_pnl: float = 100.0,
    estimated_costs: float = 20.0,
    net_pnl: float = 80.0,
    ledger_complete: bool = True,
    state: str = "CLOSED",
    signal_quality: dict[str, float] | None = None,
    exit_reason: str = "HARD_SL_BREACH src=ltp sl=90.0",
) -> dict[str, object]:
    outcome: dict[str, object] = {
        "cost_source": "broker_virtual_contract_note",
        "effective_costs": {"total": estimated_costs},
    }
    if signal_quality is not None:
        outcome["signal_quality"] = signal_quality
        outcome.update(
            {
                "strategy_key": str(strategy).lower(),
                "strategy_role": "trigger",
                "signal_family": "directional_trigger",
                "setup_name": "continuation_pullback",
                "regime": "TREND",
                "approval_path": "multi_trigger",
                "score_contract_version": 1,
                "score_lineage": {
                    "raw_setup_score": 7.5,
                    "regime_weight": 1.0,
                    "regime_adjusted_setup_score": 7.5,
                    "context_confirmation_bonus": 0.0,
                    "context_veto_penalty": 0.0,
                    "manager_reference_score": 7.5,
                    "manager_reference_threshold": 7.0,
                    "manager_reference_pass": True,
                    "final_numeric_gate_owner": "runner_final_execution_score",
                },
                "confirming_trigger_strategies": [strategy],
                "context_confirmation_strategies": [],
            }
        )
    return {
        "trade_id": trade_id,
        "closed_at": closed_at,
        "strategy": strategy,
        "gross_pnl": gross_pnl,
        "estimated_costs": estimated_costs,
        "net_pnl": net_pnl,
        "ledger_complete": ledger_complete,
        "state": state,
        "exit_reason": exit_reason,
        "outcome": outcome,
    }


def test_canonicalize_filters_and_orders_complete_closed_rows() -> None:
    rows = [
        _trade("later", 30.0),
        _trade("open", 20.0, state="OPEN"),
        _trade("incomplete", 10.0, ledger_complete=False),
        _trade("earlier", 5.0),
    ]

    result = canonicalize_completed_trades(rows)

    assert [trade.trade_id for trade in result] == ["earlier", "later"]


def test_canonicalize_completed_trades_rejects_duplicate_trade_identity() -> None:
    rows = [_trade("duplicate", 1.0), _trade("duplicate", 2.0)]

    with pytest.raises(ValueError, match="duplicate trade_id"):
        canonicalize_completed_trades(rows)


def test_canonicalize_completed_trades_rejects_invalid_post_cost_identity() -> None:
    rows = [
        _trade(
            "bad-costs",
            1.0,
            gross_pnl=100.0,
            estimated_costs=20.0,
            net_pnl=60.0,
        )
    ]

    with pytest.raises(ValueError, match="gross_pnl - effective_costs"):
        canonicalize_completed_trades(rows)


def test_canonicalize_rejects_estimated_only_costs_by_default() -> None:
    row = _trade("estimated", 1.0)
    row["outcome"] = {"cost_source": "estimated_model"}

    with pytest.raises(ValueError, match="broker-calculated costs required"):
        canonicalize_completed_trades([row])


def test_canonicalize_can_explicitly_include_estimated_costs_for_legacy_audit() -> None:
    row = _trade("estimated", 1.0)
    row["outcome"] = {"cost_source": "estimated_model"}

    result = canonicalize_completed_trades([row], allow_estimated_costs=True)

    assert result[0].cost_source == "estimated_model"
    assert result[0].effective_costs == 20.0


def test_chronological_post_cost_blocks_are_contiguous_and_non_overlapping() -> None:
    trades = canonicalize_completed_trades(
        [
            _trade(
                f"t{index}",
                float(index),
                net_pnl=float(index),
                gross_pnl=float(index + 20),
            )
            for index in range(1, 7)
        ]
    )

    blocks = chronological_post_cost_blocks(trades, block_size=2)

    assert [(block.start_trade_id, block.end_trade_id) for block in blocks] == [
        ("t1", "t2"),
        ("t3", "t4"),
        ("t5", "t6"),
    ]
    assert [block.summary.trade_count for block in blocks] == [2, 2, 2]


def test_walk_forward_uses_expanding_train_and_strictly_later_test_windows() -> None:
    trades = canonicalize_completed_trades(
        [
            _trade(
                f"t{index}",
                float(index),
                gross_pnl=float(index + 20),
                net_pnl=float(index),
            )
            for index in range(1, 9)
        ]
    )

    folds = chronological_walk_forward(
        trades,
        min_train_trades=4,
        test_trades=2,
    )

    assert [
        (
            fold.train_start_trade_id,
            fold.train_end_trade_id,
            fold.test_start_trade_id,
            fold.test_end_trade_id,
        )
        for fold in folds
    ] == [
        ("t1", "t4", "t5", "t6"),
        ("t1", "t6", "t7", "t8"),
    ]
    assert [fold.test_summary.net_pnl for fold in folds] == [11.0, 15.0]


def test_walk_forward_fails_closed_when_sample_is_too_small() -> None:
    trades = canonicalize_completed_trades([_trade("t1", 1.0), _trade("t2", 2.0)])

    assert chronological_walk_forward(trades, min_train_trades=2, test_trades=1) == ()


def test_walk_forward_rejects_non_positive_window_sizes() -> None:
    with pytest.raises(ValueError, match="min_train_trades"):
        chronological_walk_forward((), min_train_trades=0, test_trades=1)
    with pytest.raises(ValueError, match="test_trades"):
        chronological_walk_forward((), min_train_trades=1, test_trades=0)


def test_walk_forward_stability_requires_minimum_oos_fold_coverage() -> None:
    trades = canonicalize_completed_trades(
        [
            _trade(
                f"t{index}",
                float(index),
                gross_pnl=float(index + 20),
                net_pnl=float(index),
            )
            for index in range(1, 7)
        ]
    )
    folds = chronological_walk_forward(trades, min_train_trades=4, test_trades=2)

    stability = walk_forward_stability(folds, minimum_folds=2)

    assert stability.ready is False
    assert stability.fold_count == 1
    assert stability.blockers == ("insufficient_oos_folds:1<2",)


def test_walk_forward_stability_reports_oos_consistency_without_verdict() -> None:
    trades = canonicalize_completed_trades(
        [
            _trade("t1", 1.0, gross_pnl=21.0, net_pnl=1.0),
            _trade("t2", 2.0, gross_pnl=22.0, net_pnl=2.0),
            _trade("t3", 3.0, gross_pnl=23.0, net_pnl=3.0),
            _trade("t4", 4.0, gross_pnl=24.0, net_pnl=4.0),
            _trade("t5", 5.0, gross_pnl=30.0, net_pnl=10.0),
            _trade("t6", 6.0, gross_pnl=30.0, net_pnl=10.0),
            _trade("t7", 7.0, gross_pnl=10.0, net_pnl=-10.0),
            _trade("t8", 8.0, gross_pnl=10.0, net_pnl=-10.0),
        ]
    )
    folds = chronological_walk_forward(trades, min_train_trades=4, test_trades=2)

    stability = walk_forward_stability(folds, minimum_folds=2)

    assert stability.ready is True
    assert stability.fold_count == 2
    assert stability.positive_oos_folds == 1
    assert stability.positive_oos_fraction == 0.5
    assert stability.aggregate_oos.trade_count == 4
    assert stability.aggregate_oos.net_pnl == 0.0


def test_walk_forward_stability_rejects_invalid_minimum_folds() -> None:
    with pytest.raises(ValueError, match="minimum_folds"):
        walk_forward_stability((), minimum_folds=0)


def test_summary_uses_post_cost_net_pnl() -> None:
    trades = canonicalize_completed_trades(
        [
            _trade(
                "winner",
                1.0,
                gross_pnl=120.0,
                estimated_costs=20.0,
                net_pnl=100.0,
            ),
            _trade(
                "loser",
                2.0,
                gross_pnl=-30.0,
                estimated_costs=20.0,
                net_pnl=-50.0,
            ),
        ]
    )

    summary = summarize_completed_trades(trades)

    assert summary.gross_pnl == 90.0
    assert summary.estimated_costs == 40.0
    assert summary.effective_costs == 40.0
    assert summary.broker_cost_trade_count == 2
    assert summary.net_pnl == 50.0
    assert summary.expectancy == 25.0
    assert summary.win_rate == 0.5
    assert summary.profit_factor == 2.0


def test_attribution_readiness_fails_closed_for_missing_component_or_quality() -> None:
    trades = canonicalize_completed_trades(
        [
            _trade("vwap", 1.0, strategy="VWAPPro"),
            _trade(
                "smc",
                2.0,
                strategy="SMC",
                signal_quality={"alpha_score": 8.0, "strategy_score": 7.5},
            ),
        ]
    )

    readiness = attribution_readiness(
        trades,
        required_components=("ORBPro", "SMC", "VWAPPro"),
    )

    assert readiness.ready is False
    assert readiness.coverage["ORBPro"].completed_trades == 0
    assert readiness.coverage["VWAPPro"].with_signal_quality == 0
    assert "missing_completed_trades:ORBPro" in readiness.blockers
    assert "missing_signal_quality:VWAPPro" in readiness.blockers


def test_attribution_readiness_accepts_complete_component_evidence() -> None:
    quality = {"alpha_score": 8.0, "strategy_score": 7.5}
    trades = canonicalize_completed_trades(
        [
            _trade("orb", 1.0, strategy="ORBPro", signal_quality=quality),
            _trade("smc", 2.0, strategy="SMC", signal_quality=quality),
            _trade("vwap", 3.0, strategy="VWAPPro", signal_quality=quality),
        ]
    )

    readiness = attribution_readiness(
        trades,
        required_components=("ORBPro", "SMC", "VWAPPro"),
    )

    assert readiness.ready is True
    assert readiness.blockers == ()


def test_attribution_readiness_fails_closed_when_score_lineage_is_missing() -> None:
    quality = {"alpha_score": 8.0, "strategy_score": 7.5}
    rows = [_trade("vwap", 1.0, strategy="VWAPPro", signal_quality=quality)]
    rows[0]["outcome"].pop("score_lineage")

    readiness = attribution_readiness(
        canonicalize_completed_trades(rows),
        required_components=("VWAPPro",),
    )

    assert readiness.ready is False
    assert readiness.coverage["VWAPPro"].with_signal_quality == 1
    assert readiness.coverage["VWAPPro"].with_attribution_provenance == 0
    assert "missing_attribution_provenance:VWAPPro" in readiness.blockers


def test_attribution_readiness_requires_canonical_strategy_identity() -> None:
    quality = {"alpha_score": 8.0, "strategy_score": 7.5}
    rows = [_trade("vwap", 1.0, strategy="VWAPPro", signal_quality=quality)]
    rows[0]["outcome"].pop("strategy_key")

    readiness = attribution_readiness(
        canonicalize_completed_trades(rows),
        required_components=("VWAPPro",),
    )

    assert readiness.ready is False
    assert readiness.coverage["VWAPPro"].with_attribution_provenance == 0
    assert "missing_attribution_provenance:VWAPPro" in readiness.blockers


def test_post_cost_attribution_groups_regime_setup_and_confirmation() -> None:
    quality = {"alpha_score": 8.0, "strategy_score": 7.5}
    rows = [
        _trade("context", 1.0, signal_quality=quality, net_pnl=80.0),
        _trade(
            "multi",
            2.0,
            signal_quality=quality,
            gross_pnl=-20.0,
            estimated_costs=20.0,
            net_pnl=-40.0,
        ),
    ]
    rows[0]["outcome"]["context_confirmation_strategies"] = ["OrderFlow"]
    rows[1]["outcome"]["confirming_trigger_strategies"] = ["VWAPPro", "ORBPro"]

    groups = post_cost_attribution_groups(canonicalize_completed_trades(rows))

    assert [group.confirmation_type for group in groups] == [
        "multi_trigger",
        "single_trigger_context_confirmed",
    ]
    assert [group.summary.net_pnl for group in groups] == [-40.0, 80.0]


def test_execution_data_quality_flags_explicit_stale_quote_exits() -> None:
    trades = canonicalize_completed_trades(
        [
            _trade(
                "stale",
                1.0,
                exit_reason="HARD_SL_BREACH src=ltp_stale_quote sl=90.0",
            ),
            _trade("live", 2.0),
        ]
    )

    quality = execution_data_quality(trades)

    assert quality.total_trades == 2
    assert quality.known_stale_quote_exit_trades == 1
    assert quality.known_stale_quote_exit_fraction == 0.5
    assert quality.blockers == ("known_stale_quote_exit_trades:1",)


def test_score_calibration_reports_post_cost_expectancy_and_r_uncertainty() -> None:
    rows = [
        _trade(
            "s1",
            1.0,
            gross_pnl=100.0,
            net_pnl=80.0,
            signal_quality={"alpha_score": 6.2, "strategy_score": 6.0},
        ),
        _trade(
            "s2",
            2.0,
            gross_pnl=60.0,
            net_pnl=40.0,
            signal_quality={"alpha_score": 6.8, "strategy_score": 6.5},
        ),
        _trade(
            "s3",
            3.0,
            gross_pnl=-20.0,
            net_pnl=-40.0,
            signal_quality={"alpha_score": 7.2, "strategy_score": 7.0},
        ),
        _trade(
            "s4",
            4.0,
            gross_pnl=120.0,
            net_pnl=100.0,
            signal_quality={"alpha_score": 7.8, "strategy_score": 7.5},
        ),
    ]
    for row, r_multiple in zip(rows, (0.8, 0.4, -0.4, 1.0)):
        row["outcome"]["r_multiple"] = r_multiple

    report = calibrate_signal_scores(
        canonicalize_completed_trades(rows),
        score_key="alpha_score",
        bin_width=1.0,
        minimum_trades_per_bin=2,
        bootstrap_samples=300,
        seed=9,
    )

    assert report.scored_trades == 4
    assert report.r_scored_trades == 4
    assert len(report.bins) == 2
    assert report.bins[0].lower == 6.0
    assert report.bins[0].net_expectancy == 60.0
    assert report.bins[0].mean_r == 0.6
    assert report.bins[0].evidence_ready is True
    assert report.bins[0].net_expectancy_ci_lower <= 60.0
    assert report.bins[0].net_expectancy_ci_upper >= 60.0
    assert report.blockers == ()


def test_score_calibration_marks_underpowered_bins() -> None:
    row = _trade(
        "s1",
        1.0,
        signal_quality={"alpha_score": 8.1, "strategy_score": 8.0},
    )

    report = calibrate_signal_scores(
        canonicalize_completed_trades([row]),
        minimum_trades_per_bin=10,
        bootstrap_samples=20,
    )

    assert report.bins[0].evidence_ready is False
    assert report.blockers == ("underpowered_bins:1",)


def test_candidate_decision_funnel_preserves_block_reasons_and_quality_coverage() -> None:
    rows = [
        {
            "event_name": "candidate.blocked",
            "reason_code": "alpha_below_threshold",
            "meta": {
                "research_context": {
                    "signal_quality": {"alpha_score": 6.8, "final_score": 7.6}
                }
            },
        },
        {
            "event_name": "candidate.blocked",
            "reason_code": "risk_capacity_unavailable",
            "meta": {},
        },
        {
            "event_name": "candidate.approved",
            "reason_code": "order_submitted",
            "meta": {},
        },
    ]

    summary = summarize_candidate_decisions(rows)

    assert summary.total_decisions == 3
    assert summary.approved == 1
    assert summary.blocked == 2
    assert summary.approval_fraction == 0.3333
    assert summary.with_research_context == 1
    assert summary.with_signal_quality == 1
    assert summary.blocked_by_reason == {
        "alpha_below_threshold": 1,
        "risk_capacity_unavailable": 1,
    }
