"""Research-only analysis for canonical completed trades.

The live trading path does not import this module. It accepts already-materialized
trade-ledger rows, keeps only economically complete CLOSED trades, validates the
post-cost identity, and exposes chronological summaries plus attribution
readiness. Parameter selection remains owned by the walk-forward research layer.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from nifty_scalper_bot.utils.market_hours import IST


@dataclass(frozen=True, slots=True)
class CanonicalCompletedTrade:
    """One economically complete trade in chronological research form."""

    trade_id: str
    closed_at: float
    strategy: str
    gross_pnl: float
    estimated_costs: float
    effective_costs: float
    cost_source: str
    net_pnl: float
    exit_reason: str
    outcome: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CompletedTradeSummary:
    """Post-cost performance summary for one chronological sample."""

    trade_count: int
    gross_pnl: float
    estimated_costs: float
    effective_costs: float
    broker_cost_trade_count: int
    net_pnl: float
    expectancy: float
    win_rate: float
    average_win: float
    average_loss: float
    profit_factor: float | None
    max_drawdown: float


@dataclass(frozen=True, slots=True)
class ChronologicalBlock:
    """One contiguous, non-overlapping chronological trade block."""

    index: int
    start_trade_id: str
    end_trade_id: str
    start_closed_at: float
    end_closed_at: float
    summary: CompletedTradeSummary


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    """One expanding-history, strictly later out-of-sample evaluation fold."""

    index: int
    train_start_trade_id: str
    train_end_trade_id: str
    test_start_trade_id: str
    test_end_trade_id: str
    train_summary: CompletedTradeSummary
    test_summary: CompletedTradeSummary


@dataclass(frozen=True, slots=True)
class WalkForwardStability:
    """Descriptive OOS stability evidence; never a parameter-selection verdict."""

    ready: bool
    fold_count: int
    positive_oos_folds: int
    positive_oos_fraction: float
    aggregate_oos: CompletedTradeSummary
    blockers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ComponentCoverage:
    """Observed completed-trade coverage for one trigger strategy."""

    completed_trades: int
    with_structural_provenance: int


@dataclass(frozen=True, slots=True)
class AttributionReadiness:
    """Whether completed trades contain enough fields for component attribution."""

    ready: bool
    coverage: Mapping[str, ComponentCoverage]
    blockers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AttributionGroup:
    """Post-cost outcome summary for one canonical decision cohort."""

    regime: str
    setup_name: str
    confirmation_type: str
    summary: CompletedTradeSummary


@dataclass(frozen=True, slots=True)
class CandidateDecisionSummary:
    """Selection-funnel coverage from persisted runner decisions."""

    total_decisions: int
    approved: int
    blocked: int
    approval_fraction: float
    with_research_context: int
    with_structural_provenance: int
    blocked_by_reason: Mapping[str, int]


def summarize_candidate_decisions(
    rows: Sequence[Mapping[str, Any]],
) -> CandidateDecisionSummary:
    """Summarize accepted/rejected candidates without inventing outcomes."""

    approved = 0
    blocked = 0
    with_context = 0
    with_structural = 0
    reasons: dict[str, int] = {}
    for row in rows:
        event_name = str(row.get("event_name") or "").strip()
        if event_name == "candidate.approved":
            approved += 1
        elif event_name == "candidate.blocked":
            blocked += 1
            reason = str(row.get("reason_code") or "unknown").strip() or "unknown"
            reasons[reason] = reasons.get(reason, 0) + 1
        else:
            continue
        meta = row.get("meta")
        if not isinstance(meta, Mapping):
            continue
        research = meta.get("research_context")
        if not isinstance(research, Mapping):
            continue
        with_context += 1
        contracts = (
            research.get("direction_contract"),
            research.get("setup_contract"),
            research.get("confirmation_contract"),
        )
        if all(isinstance(item, Mapping) and item for item in contracts):
            with_structural += 1

    total = approved + blocked
    return CandidateDecisionSummary(
        total_decisions=total,
        approved=approved,
        blocked=blocked,
        approval_fraction=round(approved / total, 4) if total else 0.0,
        with_research_context=with_context,
        with_structural_provenance=with_structural,
        blocked_by_reason=dict(sorted(reasons.items())),
    )


@dataclass(frozen=True, slots=True)
class OutcomeEvidenceGroup:
    """Post-cost realized evidence for one strategy or setup cohort."""

    dimension: str
    value: str
    trade_count: int
    net_expectancy: float
    mean_r_multiple: float | None
    mean_mfe_r: float | None
    mean_mae_r: float | None
    mean_mfe_capture_ratio: float | None


@dataclass(frozen=True, slots=True)
class GateOutcomeGroup:
    """Counterfactual post-cost R evidence for one final decision outcome."""

    decision: str
    reason: str
    opportunity_count: int
    mean_post_cost_r: float
    positive_fraction: float


@dataclass(frozen=True, slots=True)
class GateEffectivenessReport:
    """Descriptive gate evidence; never a causal or parameter-change verdict."""

    ready: bool
    labelled_opportunities: int
    approved_labelled: int
    blocked_labelled: int
    groups: tuple[GateOutcomeGroup, ...]
    blockers: tuple[str, ...]


def _optional_outcome_mean(
    trades: Sequence[CanonicalCompletedTrade],
    key: str,
) -> float | None:
    values: list[float] = []
    for trade in trades:
        raw_value = trade.outcome.get(key)
        if raw_value is None:
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return round(sum(values) / len(values), 6) if values else None


def _mean_mfe_capture_ratio(
    trades: Sequence[CanonicalCompletedTrade],
) -> float | None:
    """Return mean realised net-R divided by available MFE-R."""

    values: list[float] = []
    for trade in trades:
        try:
            realised_r = float(trade.outcome.get("r_multiple"))
            mfe_r = float(trade.outcome.get("mfe_r"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(realised_r) and math.isfinite(mfe_r) and mfe_r > 0:
            values.append(realised_r / mfe_r)
    return round(sum(values) / len(values), 6) if values else None


def post_cost_outcome_evidence(
    trades: Sequence[CanonicalCompletedTrade],
    *,
    dimension: str,
) -> tuple[OutcomeEvidenceGroup, ...]:
    """Summarize realized post-cost expectancy and R excursions by cohort."""

    if dimension not in {
        "strategy",
        "setup",
        "entry_hour_ist",
        "days_to_expiry",
        "target_adjustment",
        "exit_reason",
    }:
        raise ValueError("unsupported outcome evidence dimension")
    grouped: dict[str, list[CanonicalCompletedTrade]] = {}
    for trade in trades:
        if dimension == "strategy":
            value = str(trade.strategy or "").strip() or "unknown"
        elif dimension == "setup":
            value = str(trade.outcome.get("setup_name") or "").strip() or "unknown"
        elif dimension == "exit_reason":
            value = (
                str(trade.exit_reason or "").strip().split(maxsplit=1)[0]
                or "unknown"
            )
        elif dimension == "target_adjustment":
            adjusted = trade.outcome.get("premium_cost_target_adjusted")
            value = (
                "adjusted"
                if adjusted is True
                else "original" if adjusted is False else "unknown"
            )
        else:
            value = "unknown"
            try:
                decision_ts = _closed_timestamp(trade.outcome.get("decision_ts"))
                decision = datetime.fromtimestamp(decision_ts, IST)
                if decision_ts > 0 and decision_ts <= trade.closed_at:
                    if dimension == "entry_hour_ist":
                        value = decision.strftime("%H")
                    else:
                        expiry = datetime.fromisoformat(
                            str(trade.outcome.get("contract_expiry"))
                        ).date()
                        days = (expiry - decision.date()).days
                        if days >= 0:
                            value = str(days)
            except (ValueError, TypeError, OverflowError, OSError):
                pass
        grouped.setdefault(value, []).append(trade)
    result: list[OutcomeEvidenceGroup] = []
    for value, sample in sorted(grouped.items()):
        summary = summarize_completed_trades(sample)
        result.append(
            OutcomeEvidenceGroup(
                dimension=dimension,
                value=value,
                trade_count=summary.trade_count,
                net_expectancy=summary.expectancy,
                mean_r_multiple=_optional_outcome_mean(sample, "r_multiple"),
                mean_mfe_r=_optional_outcome_mean(sample, "mfe_r"),
                mean_mae_r=_optional_outcome_mean(sample, "mae_r"),
                mean_mfe_capture_ratio=_mean_mfe_capture_ratio(sample),
            )
        )
    return tuple(result)


def summarize_gate_effectiveness(
    rows: Sequence[Mapping[str, Any]],
) -> GateEffectivenessReport:
    """Summarize independently labelled approved/blocked setup opportunities."""

    grouped: dict[tuple[str, str], list[float]] = {}
    approved_labelled = 0
    blocked_labelled = 0
    for row in rows:
        try:
            value = float(row["post_cost_r"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        approved = bool(row.get("approved"))
        decision = "approved" if approved else "blocked"
        reason = str(row.get("final_reason") or "unknown").strip() or "unknown"
        grouped.setdefault((decision, reason), []).append(value)
        if approved:
            approved_labelled += 1
        else:
            blocked_labelled += 1
    groups = tuple(
        GateOutcomeGroup(
            decision=decision,
            reason=reason,
            opportunity_count=len(values),
            mean_post_cost_r=round(sum(values) / len(values), 6),
            positive_fraction=round(
                sum(value > 0 for value in values) / len(values), 6
            ),
        )
        for (decision, reason), values in sorted(grouped.items())
    )
    blockers: list[str] = []
    if not approved_labelled:
        blockers.append("missing_approved_counterfactual_labels")
    if not blocked_labelled:
        blockers.append("missing_blocked_counterfactual_labels")
    return GateEffectivenessReport(
        ready=not blockers,
        labelled_opportunities=approved_labelled + blocked_labelled,
        approved_labelled=approved_labelled,
        blocked_labelled=blocked_labelled,
        groups=groups,
        blockers=tuple(blockers),
    )


@dataclass(frozen=True, slots=True)
class ExecutionDataQuality:
    """Known execution-evidence caveats in the realized historical sample."""

    total_trades: int
    known_stale_quote_exit_trades: int
    known_stale_quote_exit_fraction: float
    blockers: tuple[str, ...]


_STALE_QUOTE_EXIT_MARKER = "src=ltp_stale_quote"

_STRATEGY_ALIASES = {
    "ORB": "ORBPro",
    "ORBPRO": "ORBPro",
    "SMC": "SMC",
    "SMCLITE": "SMC",
    "VWAP": "VWAPPro",
    "VWAPPRO": "VWAPPro",
}


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(resolved):
        raise ValueError(f"{field} must be a finite number")
    return resolved


def _closed_timestamp(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _finite_number(value, field="closed_at")
    text = str(value or "").strip()
    if not text:
        raise ValueError("closed_at is required")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("closed_at must be an epoch or ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _ledger_complete(value: Any) -> bool:
    if value is True or value == 1:
        return True
    return str(value or "").strip().lower() in {"true", "yes", "1"}


def _outcome(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("outcome")
    if isinstance(value, Mapping):
        return dict(value)
    value = row.get("outcome_json")
    if isinstance(value, Mapping):
        return dict(value)
    if value in (None, ""):
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _canonical_strategy(value: Any) -> str:
    text = str(value or "").strip()
    key = "".join(character for character in text.upper() if character.isalnum())
    return _STRATEGY_ALIASES.get(key, text or "UNKNOWN")


def _has_attribution_provenance(outcome: Mapping[str, Any]) -> bool:
    """Return whether a trade can support structural decision attribution."""

    for field in ("strategy_key", "strategy_role", "signal_family"):
        if not str(outcome.get(field) or "").strip():
            return False
    if str(outcome.get("regime") or "").strip().upper() in {"", "UNKNOWN"}:
        return False
    if not str(outcome.get("setup_name") or "").strip():
        return False
    if not str(outcome.get("approval_path") or "").strip():
        return False
    for field in ("direction_contract", "setup_contract", "confirmation_contract"):
        contract = outcome.get(field)
        if not isinstance(contract, Mapping) or contract.get("passed") is not True:
            return False
    if not isinstance(outcome.get("confirming_trigger_strategies"), list):
        return False
    if not isinstance(outcome.get("context_confirmation_strategies"), list):
        return False
    return True


def _confirmation_type(outcome: Mapping[str, Any]) -> str:
    triggers = outcome.get("confirming_trigger_strategies")
    contexts = outcome.get("context_confirmation_strategies")
    trigger_count = len(triggers) if isinstance(triggers, list) else 0
    context_count = len(contexts) if isinstance(contexts, list) else 0
    if trigger_count >= 2:
        return "multi_trigger"
    if trigger_count == 1 and context_count > 0:
        return "single_trigger_context_confirmed"
    if trigger_count == 1:
        return "single_trigger_unconfirmed"
    return "unknown"


def post_cost_attribution_groups(
    trades: Sequence[CanonicalCompletedTrade],
) -> tuple[AttributionGroup, ...]:
    """Summarize canonical trades by regime, setup and confirmation type."""

    grouped: dict[tuple[str, str, str], list[CanonicalCompletedTrade]] = {}
    for trade in trades:
        if not _has_attribution_provenance(trade.outcome):
            continue
        key = (
            str(trade.outcome["regime"]).strip().upper(),
            str(trade.outcome["setup_name"]).strip(),
            _confirmation_type(trade.outcome),
        )
        grouped.setdefault(key, []).append(trade)

    return tuple(
        AttributionGroup(
            regime=regime,
            setup_name=setup_name,
            confirmation_type=confirmation_type,
            summary=summarize_completed_trades(sample),
        )
        for (regime, setup_name, confirmation_type), sample in sorted(grouped.items())
    )


def canonicalize_completed_trades(
    rows: Sequence[Mapping[str, Any]],
    *,
    net_identity_tolerance: float = 0.02,
    allow_estimated_costs: bool = False,
) -> tuple[CanonicalCompletedTrade, ...]:
    """Return one validated row per CLOSED, ledger-complete economic trade.

    Incomplete/open rows are intentionally ignored. Selected completed rows fail
    closed when identity, timestamps, or post-cost arithmetic are ambiguous.
    """

    tolerance = _finite_number(net_identity_tolerance, field="net_identity_tolerance")
    if tolerance < 0:
        raise ValueError("net_identity_tolerance must be non-negative")

    trades: list[CanonicalCompletedTrade] = []
    seen_trade_ids: set[str] = set()
    for row in rows:
        state = str(row.get("state") or row.get("status") or "").strip().upper()
        if state != "CLOSED" or not _ledger_complete(row.get("ledger_complete")):
            continue

        trade_id = str(row.get("trade_id") or "").strip()
        if not trade_id:
            raise ValueError("completed trade requires trade_id")
        if trade_id in seen_trade_ids:
            raise ValueError(f"duplicate trade_id: {trade_id}")
        seen_trade_ids.add(trade_id)

        closed_at = _closed_timestamp(row.get("closed_at"))
        gross_pnl = _finite_number(row.get("gross_pnl"), field="gross_pnl")
        estimated_costs = _finite_number(
            row.get("estimated_costs"),
            field="estimated_costs",
        )
        outcome = _outcome(row)
        cost_source = str(outcome.get("cost_source") or "estimated_model").strip()
        effective_payload = outcome.get("effective_costs")
        if cost_source == "broker_virtual_contract_note":
            if not isinstance(effective_payload, Mapping):
                raise ValueError(
                    f"broker-calculated costs missing from completed trade: {trade_id}"
                )
            effective_costs = _finite_number(
                effective_payload.get("total"),
                field="effective_costs.total",
            )
        elif allow_estimated_costs:
            effective_costs = estimated_costs
            cost_source = "estimated_model"
        else:
            raise ValueError(
                "broker-calculated costs required for canonical research dataset: "
                f"{trade_id}"
            )

        net_pnl = _finite_number(row.get("net_pnl"), field="net_pnl")
        expected_net = gross_pnl - effective_costs
        if abs(expected_net - net_pnl) > tolerance:
            raise ValueError(
                "completed trade violates gross_pnl - effective_costs = net_pnl "
                f"within tolerance: {trade_id}"
            )

        trades.append(
            CanonicalCompletedTrade(
                trade_id=trade_id,
                closed_at=closed_at,
                strategy=_canonical_strategy(row.get("strategy")),
                gross_pnl=gross_pnl,
                estimated_costs=estimated_costs,
                effective_costs=effective_costs,
                cost_source=cost_source,
                net_pnl=net_pnl,
                exit_reason=str(row.get("exit_reason") or "").strip(),
                outcome=outcome,
            )
        )

    trades.sort(key=lambda trade: (trade.closed_at, trade.trade_id))
    return tuple(trades)


def summarize_completed_trades(
    trades: Sequence[CanonicalCompletedTrade],
) -> CompletedTradeSummary:
    """Summarize realized economics using the effective post-cost net P&L."""

    values = [float(trade.net_pnl) for trade in trades]
    wins = [value for value in values if value > 0]
    losses = [value for value in values if value < 0]

    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    count = len(values)
    return CompletedTradeSummary(
        trade_count=count,
        gross_pnl=round(sum(trade.gross_pnl for trade in trades), 2),
        estimated_costs=round(sum(trade.estimated_costs for trade in trades), 2),
        effective_costs=round(sum(trade.effective_costs for trade in trades), 2),
        broker_cost_trade_count=sum(
            trade.cost_source == "broker_virtual_contract_note" for trade in trades
        ),
        net_pnl=round(sum(values), 2),
        expectancy=round(sum(values) / count, 4) if count else 0.0,
        win_rate=round(len(wins) / count, 4) if count else 0.0,
        average_win=round(sum(wins) / len(wins), 4) if wins else 0.0,
        average_loss=round(sum(losses) / len(losses), 4) if losses else 0.0,
        profit_factor=round(gross_profit / gross_loss, 4) if gross_loss > 0 else None,
        max_drawdown=round(max_drawdown, 2),
    )


def chronological_post_cost_blocks(
    trades: Sequence[CanonicalCompletedTrade],
    *,
    block_size: int,
) -> tuple[ChronologicalBlock, ...]:
    """Split already-canonical trades into contiguous, non-overlapping blocks."""

    size = int(block_size)
    if size <= 0:
        raise ValueError("block_size must be positive")

    ordered = list(trades)
    if any(
        (current.closed_at, current.trade_id)
        >= (following.closed_at, following.trade_id)
        for current, following in zip(ordered, ordered[1:])
    ):
        raise ValueError("trades must be in strict chronological order")

    blocks: list[ChronologicalBlock] = []
    for start in range(0, len(ordered), size):
        sample = ordered[start : start + size]
        if not sample:
            continue
        blocks.append(
            ChronologicalBlock(
                index=len(blocks) + 1,
                start_trade_id=sample[0].trade_id,
                end_trade_id=sample[-1].trade_id,
                start_closed_at=sample[0].closed_at,
                end_closed_at=sample[-1].closed_at,
                summary=summarize_completed_trades(sample),
            )
        )
    return tuple(blocks)


def chronological_walk_forward(
    trades: Sequence[CanonicalCompletedTrade],
    *,
    min_train_trades: int,
    test_trades: int,
) -> tuple[WalkForwardFold, ...]:
    """Evaluate expanding history against strictly later post-cost test windows."""

    min_train = int(min_train_trades)
    test_size = int(test_trades)
    if min_train <= 0:
        raise ValueError("min_train_trades must be positive")
    if test_size <= 0:
        raise ValueError("test_trades must be positive")

    ordered = list(trades)
    if any(
        (current.closed_at, current.trade_id)
        >= (following.closed_at, following.trade_id)
        for current, following in zip(ordered, ordered[1:])
    ):
        raise ValueError("trades must be in strict chronological order")
    if len(ordered) < min_train + test_size:
        return ()

    folds: list[WalkForwardFold] = []
    train_end = min_train
    while train_end + test_size <= len(ordered):
        train = ordered[:train_end]
        test = ordered[train_end : train_end + test_size]
        folds.append(
            WalkForwardFold(
                index=len(folds) + 1,
                train_start_trade_id=train[0].trade_id,
                train_end_trade_id=train[-1].trade_id,
                test_start_trade_id=test[0].trade_id,
                test_end_trade_id=test[-1].trade_id,
                train_summary=summarize_completed_trades(train),
                test_summary=summarize_completed_trades(test),
            )
        )
        train_end += test_size
    return tuple(folds)


def walk_forward_stability(
    folds: Sequence[WalkForwardFold],
    *,
    minimum_folds: int = 2,
) -> WalkForwardStability:
    """Summarize OOS fold consistency and fail closed on inadequate coverage."""

    required = int(minimum_folds)
    if required <= 0:
        raise ValueError("minimum_folds must be positive")

    blockers: list[str] = []
    if len(folds) < required:
        blockers.append(f"insufficient_oos_folds:{len(folds)}<{required}")

    oos_trade_count = sum(fold.test_summary.trade_count for fold in folds)
    aggregate = CompletedTradeSummary(
        trade_count=oos_trade_count,
        gross_pnl=round(sum(fold.test_summary.gross_pnl for fold in folds), 2),
        estimated_costs=round(
            sum(fold.test_summary.estimated_costs for fold in folds), 2
        ),
        effective_costs=round(
            sum(fold.test_summary.effective_costs for fold in folds), 2
        ),
        broker_cost_trade_count=sum(
            fold.test_summary.broker_cost_trade_count for fold in folds
        ),
        net_pnl=round(sum(fold.test_summary.net_pnl for fold in folds), 2),
        expectancy=(
            round(
                sum(fold.test_summary.net_pnl for fold in folds) / oos_trade_count,
                4,
            )
            if oos_trade_count
            else 0.0
        ),
        win_rate=(
            round(
                sum(
                    fold.test_summary.win_rate * fold.test_summary.trade_count
                    for fold in folds
                )
                / oos_trade_count,
                4,
            )
            if oos_trade_count
            else 0.0
        ),
        average_win=0.0,
        average_loss=0.0,
        profit_factor=None,
        max_drawdown=max(
            (fold.test_summary.max_drawdown for fold in folds), default=0.0
        ),
    )
    positive = sum(fold.test_summary.net_pnl > 0 for fold in folds)
    return WalkForwardStability(
        ready=not blockers,
        fold_count=len(folds),
        positive_oos_folds=positive,
        positive_oos_fraction=round(positive / len(folds), 4) if folds else 0.0,
        aggregate_oos=aggregate,
        blockers=tuple(blockers),
    )


def attribution_readiness(
    trades: Sequence[CanonicalCompletedTrade],
    *,
    required_components: Sequence[str] = ("ORBPro", "SMC", "VWAPPro"),
) -> AttributionReadiness:
    """Fail closed unless every requested trigger has structural decision evidence."""

    coverage: dict[str, ComponentCoverage] = {}
    blockers: list[str] = []
    for raw_component in required_components:
        component = _canonical_strategy(raw_component)
        component_trades = [trade for trade in trades if trade.strategy == component]
        with_structural = sum(
            _has_attribution_provenance(trade.outcome) for trade in component_trades
        )
        coverage[component] = ComponentCoverage(
            completed_trades=len(component_trades),
            with_structural_provenance=with_structural,
        )
        if not component_trades:
            blockers.append(f"missing_completed_trades:{component}")
        else:
            if with_structural != len(component_trades):
                blockers.append(f"missing_structural_provenance:{component}")

    return AttributionReadiness(
        ready=not blockers,
        coverage=coverage,
        blockers=tuple(blockers),
    )


def execution_data_quality(
    trades: Sequence[CanonicalCompletedTrade],
) -> ExecutionDataQuality:
    """Flag explicit stale-quote exits without silently excluding realized trades."""

    total = len(trades)
    stale = sum(
        _STALE_QUOTE_EXIT_MARKER in trade.exit_reason.lower() for trade in trades
    )
    blockers = (f"known_stale_quote_exit_trades:{stale}",) if stale else ()
    return ExecutionDataQuality(
        total_trades=total,
        known_stale_quote_exit_trades=stale,
        known_stale_quote_exit_fraction=round(stale / total, 4) if total else 0.0,
        blockers=blockers,
    )


__all__ = [
    "AttributionReadiness",
    "CanonicalCompletedTrade",
    "CandidateDecisionSummary",
    "ChronologicalBlock",
    "CompletedTradeSummary",
    "ComponentCoverage",
    "WalkForwardFold",
    "WalkForwardStability",
    "ExecutionDataQuality",
    "attribution_readiness",
    "execution_data_quality",
    "canonicalize_completed_trades",
    "chronological_post_cost_blocks",
    "chronological_walk_forward",
    "walk_forward_stability",
    "summarize_completed_trades",
    "summarize_candidate_decisions",
]
