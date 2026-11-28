"""Analyze canonical completed trades without modifying live strategy parameters.

Responsibilities: Isolate canonical Range VWAP trade outcomes, observed
paired bid/ask execution-quality coverage, pre-entry volatility/payoff
cohorts and 10/25/50-bps stress scenarios.
Safety: No missing premium/quantity, broker-cost identity or independent-trade
sample may be inferred as zero loss. Calibration is read-only and cannot
automatically promote any parameter. Sensitivities are not fill-path replays;
selection diagnostics require complete finite R-multiple coverage.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from nifty_scalper_bot.backtesting.completed_trade_analysis import (  # noqa: E402, I001
    attribution_readiness,
    canonicalize_completed_trades,
    chronological_post_cost_blocks,
    chronological_walk_forward,
    execution_data_quality,
    post_cost_attribution_groups,
    post_cost_outcome_evidence,
    summarize_candidate_decisions,
    summarize_gate_effectiveness,
    summarize_completed_trades,
    walk_forward_stability,
)
from nifty_scalper_bot.backtesting.setup_opportunity_analysis import (  # noqa: E402
    canonicalize_setup_opportunities,
    summarize_contextual_label_coverage,
)
from nifty_scalper_bot.backtesting.research_validation import (  # noqa: E402
    combinatorial_purged_pbo,
    deflated_sharpe_ratio,
)
from nifty_scalper_bot.backtesting.broker_tradebook_reconciliation import (  # noqa: E402
    read_broker_tradebook_csv,
    summarize_broker_tradebook_reconciliation,
)
from nifty_scalper_bot.backtesting.ledger_integrity import (  # noqa: E402
    summarize_ledger_integrity,
)
from nifty_scalper_bot.config.strategy_taxonomy import (  # noqa: E402
    normalize_strategy_name,
)
from nifty_scalper_bot.utils.market_hours import IST  # noqa: E402
from nifty_scalper_bot.journal.trade_ledger import (  # noqa: E402
    load_candidate_decision_rows,
    load_trade_ledger_rows as load_canonical_trade_ledger_rows,
)


def load_trade_ledger_rows(db_path: Path) -> list[dict[str, Any]]:
    """Read completed trades through the shared canonical ledger read path."""

    resolved = db_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Trade journal not found: {resolved}")
    return load_canonical_trade_ledger_rows(
        resolved,
        limit=None,
        closed_only=True,
    )


def build_analysis(
    rows: list[dict[str, Any]],
    *,
    block_size: int,
    components: tuple[str, ...],
    allow_estimated_costs: bool = False,
    selection_trials: int | None = None,
    walk_forward_min_train: int = 20,
    walk_forward_test_trades: int = 10,
    walk_forward_min_folds: int = 2,
    strategy: str | None = None,
) -> dict[str, Any]:
    """Build a machine-readable evidence report from canonical ledger rows."""

    selected_rows = (
        [
            row
            for row in rows
            if normalize_strategy_name(str(row.get("strategy") or ""))
            == normalize_strategy_name(strategy)
        ]
        if strategy
        else rows
    )
    trades = canonicalize_completed_trades(
        selected_rows,
        allow_estimated_costs=allow_estimated_costs,
    )
    overall = summarize_completed_trades(trades)
    blocks = chronological_post_cost_blocks(trades, block_size=block_size)
    readiness = attribution_readiness(trades, required_components=components)
    execution_quality = execution_data_quality(trades)
    attribution_groups = post_cost_attribution_groups(trades)
    walk_forward_folds = chronological_walk_forward(
        trades,
        min_train_trades=walk_forward_min_train,
        test_trades=walk_forward_test_trades,
    )
    walk_forward = walk_forward_stability(
        walk_forward_folds,
        minimum_folds=walk_forward_min_folds,
    )
    r_values: list[float] = []
    for trade in trades:
        raw_r = trade.outcome.get("r_multiple")
        try:
            resolved_r = float(raw_r)
        except (TypeError, ValueError):
            continue
        if math.isfinite(resolved_r):
            r_values.append(resolved_r)
    r_multiple_coverage = {
        "finite_observations": len(r_values),
        "canonical_completed_trades": len(trades),
        "coverage_fraction": round(len(r_values) / len(trades), 4) if trades else 0.0,
    }
    dsr: dict[str, Any]
    if selection_trials is None:
        dsr = {
            "ready": False,
            "reason": "selection_trial_count_not_supplied",
        }
    elif len(r_values) != len(trades):
        dsr = {
            "ready": False,
            "reason": "incomplete_r_multiple_coverage",
            "observations": len(r_values),
            "required_observations": len(trades),
        }
    elif len(r_values) < 3:
        dsr = {
            "ready": False,
            "reason": "insufficient_r_multiple_observations",
            "observations": len(r_values),
        }
    else:
        dsr = {
            "ready": True,
            **asdict(
                deflated_sharpe_ratio(
                    r_values,
                    trials=selection_trials,
                )
            ),
        }
    return {
        "dataset": {
            "canonical_completed_trades": len(trades),
            "strategy_filter": normalize_strategy_name(strategy) if strategy else None,
            "input_closed_trade_rows": len(selected_rows),
            "first_closed_at": trades[0].closed_at if trades else None,
            "last_closed_at": trades[-1].closed_at if trades else None,
        },
        "overall_post_cost": asdict(overall),
        "chronological_blocks": [
            {
                "index": block.index,
                "start_trade_id": block.start_trade_id,
                "end_trade_id": block.end_trade_id,
                "start_closed_at": block.start_closed_at,
                "end_closed_at": block.end_closed_at,
                "summary": asdict(block.summary),
            }
            for block in blocks
        ],
        "walk_forward": {
            "configuration": {
                "min_train_trades": walk_forward_min_train,
                "test_trades": walk_forward_test_trades,
                "minimum_folds": walk_forward_min_folds,
            },
            "folds": [
                {
                    "index": fold.index,
                    "train_start_trade_id": fold.train_start_trade_id,
                    "train_end_trade_id": fold.train_end_trade_id,
                    "test_start_trade_id": fold.test_start_trade_id,
                    "test_end_trade_id": fold.test_end_trade_id,
                    "train_summary": asdict(fold.train_summary),
                    "test_summary": asdict(fold.test_summary),
                }
                for fold in walk_forward_folds
            ],
            "stability": asdict(walk_forward),
        },
        "realized_post_cost_evidence": {
            f"by_{dimension}": [
                asdict(item)
                for item in post_cost_outcome_evidence(trades, dimension=dimension)
            ]
            for dimension in (
                "strategy",
                "setup",
                "regime",
                "confirmation_type",
                "entry_hour_ist",
                "days_to_expiry",
                "target_adjustment",
                "exit_reason",
            )
        },
        "execution_data_quality": asdict(execution_quality),
        "cost_evidence": {
            "broker_cost_trades": overall.broker_cost_trade_count,
            "total_trades": overall.trade_count,
            "all_broker_costed": (
                overall.broker_cost_trade_count == overall.trade_count
            ),
        },
        "deflated_sharpe": dsr,
        "r_multiple_coverage": r_multiple_coverage,
        "attribution": {
            "ready": readiness.ready,
            "groups": [
                {
                    "regime": group.regime,
                    "setup_name": group.setup_name,
                    "confirmation_type": group.confirmation_type,
                    "summary": asdict(group.summary),
                }
                for group in attribution_groups
            ],
            "coverage": {
                name: asdict(coverage) for name, coverage in readiness.coverage.items()
            },
            "blockers": list(readiness.blockers),
        },
        "parameter_change": {
            "allowed": False,
            "reason": (
                "candidate_walk_forward_required"
                if readiness.ready
                else "component_attribution_not_ready"
            ),
        },
    }


def range_vwap_slippage_stress(
    rows: list[dict[str, Any]], *, allow_estimated_costs: bool = False
) -> dict[str, Any]:
    """Extra per-side fill slippage stress; never an executable counterfactual."""
    selected = [
        row
        for row in rows
        if normalize_strategy_name(str(row.get("strategy") or ""))
        == "range_vwap_reclaim"
    ]
    trades = canonicalize_completed_trades(
        selected, allow_estimated_costs=allow_estimated_costs
    )
    evidence = {
        "policy": "extra_slippage_per_side_on_observed_fills",
        "executable_quote_path_verified": False,
        "option_profitability_proven": False,
        "minimum_trades_for_parameter_promotion_met": False,
    }
    if not trades:
        return {
            **evidence,
            "ready": False,
            "reason": "no_verified_closed_range_trades",
            "scenarios": [],
        }
    raw = {
        str(row["trade_id"]): row
        for row in selected
        if str(row.get("state") or row.get("status") or "").upper() == "CLOSED"
        and row.get("ledger_complete") in (True, 1, "true", "1")
    }
    premiums: dict[str, float] = {}
    for trade in trades:
        row = raw[trade.trade_id]
        try:
            entry = float(row["entry_price"])
            exit_price = float(row["exit_price"])
            quantity = float(row["quantity"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return {
                **evidence,
                "ready": False,
                "reason": "premium_or_quantity_missing",
                "scenarios": [],
            }
        if (
            not all(math.isfinite(v) and v > 0 for v in (entry, exit_price, quantity))
            or not quantity.is_integer()
            or trade.cost_source != "broker_virtual_contract_note"
        ):
            return {
                **evidence,
                "ready": False,
                "reason": "premium_or_broker_cost_unverified",
                "scenarios": [],
            }
        premiums[trade.trade_id] = (entry + exit_price) * quantity

    scenarios = []
    for bps in (10, 25, 50):
        stressed = [
            replace(
                trade,
                gross_pnl=trade.gross_pnl - premiums[trade.trade_id] * bps / 10000.0,
                net_pnl=trade.net_pnl - premiums[trade.trade_id] * bps / 10000.0,
            )
            for trade in trades
        ]
        summary = summarize_completed_trades(stressed)
        scenarios.append(
            {
                "extra_slippage_bps_each_side": bps,
                "net_pnl": summary.net_pnl,
                "net_expectancy": summary.expectancy,
                "profit_factor": summary.profit_factor,
                "max_drawdown": summary.max_drawdown,
            }
        )
    return {
        **evidence,
        "ready": True,
        "reason": "descriptive_stress_only_not_fill_replay",
        "completed_trades": len(trades),
        "scenarios": scenarios,
    }


def range_vwap_execution_calibration(
    rows: list[dict[str, Any]], *, allow_estimated_costs: bool = False
) -> dict[str, Any]:
    """Read observed paired option fills; never adjust live risk from samples."""
    selected = [
        row
        for row in rows
        if normalize_strategy_name(str(row.get("strategy") or ""))
        == "range_vwap_reclaim"
    ]
    trades = canonicalize_completed_trades(
        selected, allow_estimated_costs=allow_estimated_costs
    )
    pairs: list[tuple[float, float]] = []
    dates: set[str] = set()
    for trade in trades:
        q = trade.execution_quality
        try:
            entry = float(q["entry_slippage_bps"])
            exit_slip = float(q["exit_slippage_bps"])
            entry_ask = float(q["entry_quote_ask"])
            exit_bid = float(q["exit_quote_bid"])
            exit_age_ms = float(q["exit_quote_age_ms"])
        except (KeyError, ValueError, TypeError, OverflowError):
            continue
        if (
            trade.cost_source != "broker_virtual_contract_note"
            or q.get("exit_quote_evidence_source") != "cached_bid_observation"
            or not all(
                math.isfinite(value)
                for value in (entry, exit_slip, entry_ask, exit_bid, exit_age_ms)
            )
            or min(entry_ask, exit_bid) <= 0
            or not 0 <= exit_age_ms <= 2000
        ):
            continue
        pairs.append((entry, exit_slip))
        dates.add(datetime.fromtimestamp(trade.closed_at, IST).date().isoformat())

    def percentile(values: list[float], pct: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        position = (len(ordered) - 1) * pct
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return round(
            ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower),
            3,
        )

    entry_adverse = [max(0.0, pair[0]) for pair in pairs]
    exit_adverse = [max(0.0, pair[1]) for pair in pairs]
    complete = len(trades)
    coverage = len(pairs) / complete if complete else 0.0
    ready = len(pairs) >= 30 and len(dates) >= 10 and coverage >= 0.9
    return {
        "ready_for_research_calibration": ready,
        "live_parameter_change_allowed": False,
        "completed_range_trades": complete,
        "paired_measured_quote_fill_trades": len(pairs),
        "paired_coverage_fraction": round(coverage, 4),
        "independent_trading_sessions": len(dates),
        "minimum_required": {
            "paired_trades": 30,
            "sessions": 10,
            "paired_fraction": 0.9,
        },
        "entry_adverse_slippage_median_bps": (
            round(median(entry_adverse), 3) if entry_adverse else None
        ),
        "exit_adverse_slippage_median_bps": (
            round(median(exit_adverse), 3) if exit_adverse else None
        ),
        "entry_adverse_slippage_p95_bps": percentile(entry_adverse, 0.95),
        "exit_adverse_slippage_p95_bps": percentile(exit_adverse, 0.95),
        "reason": (
            "sample_only_not_live_fill_or_regime_parity"
            if ready
            else "insufficient_paired_verified_quotes_or_sessions"
        ),
    }


def range_vwap_transition_economics(
    rows: list[dict[str, Any]], *, allow_estimated_costs: bool = False
) -> dict[str, Any]:
    """Compare causal entry volatility and modeled target with realized option P&L.

    This is a selected-trade diagnostic, NOT proof of reversals or a
    counterfactual assessment of vetoed/missed candidates.
    """
    selected = [
        row
        for row in rows
        if normalize_strategy_name(str(row.get("strategy") or ""))
        == "range_vwap_reclaim"
    ]
    trades = canonicalize_completed_trades(
        selected, allow_estimated_costs=allow_estimated_costs
    )
    cohorts: dict[str, list[tuple[Any, float, bool]]] = {}
    for trade in trades:
        structure = trade.outcome.get("range_structure")
        economics = trade.outcome.get("option_cost_risk")
        if (
            trade.cost_source != "broker_virtual_contract_note"
            or not isinstance(structure, dict)
            or not isinstance(economics, dict)
        ):
            continue
        try:
            vol_ratio = float(structure["prior_5m_to_14m_true_range_ratio"])
            short_eff = float(structure["prior_5m_directional_efficiency"])
            long_eff = float(structure["directional_efficiency"])
            target_reward = float(economics["target_net_reward_after_costs"])
            cash_risk = float(economics["worst_stop_loss_including_costs"])
        except (KeyError, ValueError, TypeError, OverflowError):
            continue
        if not (
            all(
                math.isfinite(v)
                for v in (vol_ratio, short_eff, long_eff, target_reward, cash_risk)
            )
            and vol_ratio >= 0
            and 0 <= short_eff <= 1
            and 0 <= long_eff <= 1
            and target_reward > 0
            and 0 < cash_risk <= 773
        ):
            continue
        label = (
            "recent_true_range_expansion"
            if vol_ratio > 1.0
            else "recent_true_range_not_expanding"
        )
        cohorts.setdefault(label, []).append(
            (trade, target_reward, short_eff > long_eff)
        )

    results = []
    for label, observations in sorted(cohorts.items()):
        sample = [trade for trade, _, _ in observations]
        summary = summarize_completed_trades(sample)
        sessions = {
            datetime.fromtimestamp(t.closed_at, IST).date().isoformat() for t in sample
        }
        results.append(
            {
                "preentry_volatility_cohort": label,
                "completed_trades": len(sample),
                "independent_sessions": len(sessions),
                "recent_directional_pressure_rising_fraction": round(
                    sum(rising for _, _, rising in observations) / len(sample), 4
                ),
                "modeled_target_net_reward_mean": round(
                    sum(reward for _, reward, _ in observations) / len(sample), 2
                ),
                "realized_net_expectancy": summary.expectancy,
                "realized_win_rate": summary.win_rate,
                "realized_net_pnl": summary.net_pnl,
                "research_sample_sufficient": (
                    len(sample) >= 20 and len(sessions) >= 10
                ),
            }
        )
    return {
        "completed_range_trades": len(trades),
        "trades_with_causal_volatility_and_economic_provenance": sum(
            len(items) for items in cohorts.values()
        ),
        "cohorts": results,
        "live_filter_or_parameter_promotion_allowed": False,
        "interpretation": (
            "descriptive_selected_trades_only; modeled_target_is_not_expectancy;"
            " no_assessment_of_blocked_setup_profitability"
        ),
    }


def load_candidate_returns(path: Path) -> dict[str, list[float]]:
    """Load aligned candidate post-cost return/R series from JSON."""

    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("candidate returns JSON must be an object")
    candidates: dict[str, list[float]] = {}
    for name, values in payload.items():
        if not isinstance(values, list):
            raise ValueError(f"candidate {name!r} must be a JSON array")
        candidates[str(name)] = [float(value) for value in values]
    return candidates


def load_opportunity_outcomes(path: Path) -> dict[str, float]:
    """Load independent setup opportunity -> post-cost counterfactual R labels."""

    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("opportunity outcomes JSON must be an object")
    result: dict[str, float] = {}
    for opportunity_id, value in payload.items():
        resolved = float(value)
        if not math.isfinite(resolved):
            raise ValueError(f"non-finite post-cost R for {opportunity_id!r}")
        result[str(opportunity_id)] = resolved
    return result


def build_candidate_validation(
    candidates: dict[str, list[float]],
    *,
    n_groups: int,
    n_test_groups: int,
    purge_observations: int,
    embargo_observations: int,
) -> dict[str, Any]:
    """Build PBO and candidate-wise DSR evidence from aligned alternatives."""

    pbo = combinatorial_purged_pbo(
        candidates,
        n_groups=n_groups,
        n_test_groups=n_test_groups,
        purge_observations=purge_observations,
        embargo_observations=embargo_observations,
    )
    trial_count = len(candidates)
    dsr = {
        name: asdict(deflated_sharpe_ratio(values, trials=trial_count))
        for name, values in sorted(candidates.items())
        if len(values) >= 3
    }
    return {
        "pbo": asdict(pbo),
        "deflated_sharpe_by_candidate": dsr,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trades-db", type=Path, required=True)
    parser.add_argument(
        "--strategy",
        help="Optional canonical strategy (e.g. range_vwap_reclaim). Other "
        "strategies are excluded from the completed-trade analysis.",
    )
    parser.add_argument(
        "--broker-trades-csv",
        type=Path,
        help=(
            "Explicit read-only Kite tradebook export. Match exact entry/exit "
            "order IDs and fill prices; does not verify final contract-note charges."
        ),
    )
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--walk-forward-min-train", type=int, default=20)
    parser.add_argument("--walk-forward-test-trades", type=int, default=10)
    parser.add_argument("--walk-forward-min-folds", type=int, default=2)
    parser.add_argument(
        "--allow-estimated-costs",
        action="store_true",
        help=(
            "Explicit legacy-audit override; canonical research requires broker costs."
        ),
    )
    parser.add_argument(
        "--selection-trials",
        type=int,
        help="Actual number of parameter/strategy trials used for DSR deflation.",
    )
    parser.add_argument(
        "--opportunity-outcomes-json",
        type=Path,
        help=(
            "JSON object mapping canonical opportunity_id to independently "
            "labelled post-cost counterfactual R for gate-effectiveness analysis."
        ),
    )
    parser.add_argument(
        "--candidate-returns-json",
        type=Path,
        help="JSON object of aligned candidate post-cost return/R series for PBO.",
    )
    parser.add_argument("--pbo-groups", type=int, default=6)
    parser.add_argument("--pbo-test-groups", type=int, default=3)
    parser.add_argument("--pbo-purge-observations", type=int, default=0)
    parser.add_argument("--pbo-embargo-observations", type=int, default=0)
    parser.add_argument(
        "--component",
        action="append",
        dest="components",
        help="Required attribution component; repeat as needed.",
    )
    args = parser.parse_args(argv)
    components = tuple(args.components or ("ORBPro", "SMC", "VWAPPro"))
    try:
        rows = load_trade_ledger_rows(args.trades_db)
        # Closed-trade performance and historical lifecycle completeness are
        # different populations. Read both from the same canonical database.
        all_ledger_rows = load_canonical_trade_ledger_rows(
            args.trades_db,
            limit=None,
            closed_only=False,
        )
        report = build_analysis(
            rows,
            block_size=args.block_size,
            components=components,
            allow_estimated_costs=args.allow_estimated_costs,
            selection_trials=args.selection_trials,
            walk_forward_min_train=args.walk_forward_min_train,
            walk_forward_test_trades=args.walk_forward_test_trades,
            walk_forward_min_folds=args.walk_forward_min_folds,
            strategy=args.strategy,
        )
        if normalize_strategy_name(args.strategy) == "range_vwap_reclaim":
            report["range_vwap_slippage_stress"] = range_vwap_slippage_stress(
                rows,
                allow_estimated_costs=args.allow_estimated_costs,
            )
            report["range_vwap_execution_calibration"] = (
                range_vwap_execution_calibration(
                    rows, allow_estimated_costs=args.allow_estimated_costs
                )
            )
            report["range_vwap_transition_economics"] = range_vwap_transition_economics(
                rows, allow_estimated_costs=args.allow_estimated_costs
            )
        report["selected_strategy_note"] = (
            "Closed-trade metrics and candidate decisions are strategy-filtered; "
            "historical journal integrity and broker reconciliation are global."
            if args.strategy
            else None
        )
        report["ledger_integrity"] = summarize_ledger_integrity(all_ledger_rows)
        report["broker_tradebook_reconciliation"] = (
            summarize_broker_tradebook_reconciliation(
                all_ledger_rows,
                read_broker_tradebook_csv(args.broker_trades_csv),
            )
            if args.broker_trades_csv is not None
            else {
                "ready": False,
                "reason": "broker_tradebook_csv_not_provided",
                "final_net_pnl_broker_reconciled": False,
                "live_positions_inferred": False,
            }
        )
        candidate_decisions = load_candidate_decision_rows(args.trades_db)
        if args.strategy:
            candidate_decisions = [
                row
                for row in candidate_decisions
                if normalize_strategy_name(str(row.get("strategy") or ""))
                == normalize_strategy_name(args.strategy)
            ]
        decision_funnel = summarize_candidate_decisions(candidate_decisions)
        report["candidate_decision_funnel"] = asdict(decision_funnel)
        setup_opportunities = canonicalize_setup_opportunities(candidate_decisions)
        report["setup_opportunities"] = {
            "independent_opportunities": len(setup_opportunities),
            "approved": sum(item.approved for item in setup_opportunities),
            "blocked": sum(not item.approved for item in setup_opportunities),
            "decision_rows_with_setup_identity": sum(
                item.decision_count for item in setup_opportunities
            ),
            "decision_rows_total": decision_funnel.total_decisions,
            "identity_coverage_fraction": (
                round(
                    sum(item.decision_count for item in setup_opportunities)
                    / decision_funnel.total_decisions,
                    4,
                )
                if decision_funnel.total_decisions
                else 0.0
            ),
            "ready_for_forward_labelling": bool(setup_opportunities),
            # Counts are unique setups, not option/bar evaluations. Only
            # stages explicitly persisted in decision provenance are reported.
            "observed_stage_unique_setups": {
                stage: sum(
                    stage in item.observed_stages for item in setup_opportunities
                )
                for stage in sorted(
                    {
                        stage
                        for item in setup_opportunities
                        for stage in item.observed_stages
                    }
                )
            },
            "blocked_reason_unique_setups": {
                reason: sum(
                    reason in item.blocked_reasons for item in setup_opportunities
                )
                for reason in sorted(
                    {
                        reason
                        for item in setup_opportunities
                        for reason in item.blocked_reasons
                    }
                )
            },
            "stage_provenance_coverage_fraction": (
                round(
                    sum(bool(item.observed_stages) for item in setup_opportunities)
                    / len(setup_opportunities),
                    4,
                )
                if setup_opportunities
                else 0.0
            ),
            "broker_acceptance_inferred": False,
        }
        outcome_labels = (
            load_opportunity_outcomes(args.opportunity_outcomes_json)
            if args.opportunity_outcomes_json is not None
            else {}
        )
        gate_rows = [
            {
                "approved": item.approved,
                "final_reason": item.final_reason,
                "post_cost_r": outcome_labels[item.opportunity_id],
            }
            for item in setup_opportunities
            if item.opportunity_id in outcome_labels
        ]
        gate_report = summarize_gate_effectiveness(gate_rows)
        report["contextual_counterfactual_coverage"] = (
            summarize_contextual_label_coverage(setup_opportunities, outcome_labels)
        )
        report["gate_effectiveness"] = {
            **asdict(gate_report),
            "label_coverage_fraction": (
                round(len(gate_rows) / len(setup_opportunities), 4)
                if setup_opportunities
                else 0.0
            ),
            "interpretation": (
                "descriptive_counterfactual_post_cost_r_by_final_gate_reason"
                if gate_report.ready
                else "insufficient_counterfactual_labels_no_gate_effect_claim"
            ),
        }
        report["candidate_validation"] = (
            build_candidate_validation(
                load_candidate_returns(args.candidate_returns_json),
                n_groups=args.pbo_groups,
                n_test_groups=args.pbo_test_groups,
                purge_observations=args.pbo_purge_observations,
                embargo_observations=args.pbo_embargo_observations,
            )
            if args.candidate_returns_json is not None
            else {
                "ready": False,
                "reason": "candidate_returns_not_supplied",
            }
        )
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 1

    print(json.dumps({"ok": True, **report}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
