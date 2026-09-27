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

from nifty_scalper_bot.backtesting.research_validation import bootstrap_mean_interval


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
    with_signal_quality: int
    with_attribution_provenance: int


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
class ScoreCalibrationBin:
    """Observed post-cost outcomes for one canonical score interval."""

    lower: float
    upper: float
    trade_count: int
    r_trade_count: int
    mean_score: float
    net_expectancy: float
    net_expectancy_ci_lower: float
    net_expectancy_ci_upper: float
    mean_r: float | None
    mean_r_ci_lower: float | None
    mean_r_ci_upper: float | None
    win_rate: float
    evidence_ready: bool


@dataclass(frozen=True, slots=True)
class ScoreCalibrationReport:
    """Descriptive score-to-outcome calibration without threshold selection."""

    score_key: str
    total_trades: int
    scored_trades: int
    r_scored_trades: int
    minimum_trades_per_bin: int
    bins: tuple[ScoreCalibrationBin, ...]
    blockers: tuple[str, ...]


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


def _has_signal_quality(outcome: Mapping[str, Any]) -> bool:
    quality = outcome.get("signal_quality")
    if not isinstance(quality, Mapping):
        return False
    for field in ("alpha_score", "strategy_score"):
        value = quality.get(field)
        if value is None or isinstance(value, bool):
            return False
        try:
            resolved = float(value)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(resolved):
            return False
    return True


def _has_attribution_provenance(outcome: Mapping[str, Any]) -> bool:
    """Return whether a trade can support regime/setup/score attribution."""

    for field in ("strategy_key", "strategy_role", "signal_family"):
        if not str(outcome.get(field) or "").strip():
            return False
    if str(outcome.get("regime") or "").strip().upper() in {"", "UNKNOWN"}:
        return False
    if not str(outcome.get("setup_name") or "").strip():
        return False
    if not str(outcome.get("approval_path") or "").strip():
        return False
    if outcome.get("score_contract_version") != 1:
        return False
    lineage = outcome.get("score_lineage")
    if not isinstance(lineage, Mapping):
        return False
    required_lineage = {
        "raw_setup_score",
        "regime_weight",
        "regime_adjusted_setup_score",
        "context_confirmation_bonus",
        "context_veto_penalty",
        "manager_reference_score",
        "manager_reference_threshold",
        "manager_reference_pass",
        "final_numeric_gate_owner",
    }
    if not required_lineage.issubset(lineage):
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


def calibrate_signal_scores(
    trades: Sequence[CanonicalCompletedTrade],
    *,
    score_key: str = "alpha_score",
    bin_width: float = 1.0,
    minimum_trades_per_bin: int = 10,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> ScoreCalibrationReport:
    """Map canonical score bins to post-cost expectancy and R uncertainty."""

    width = float(bin_width)
    minimum = int(minimum_trades_per_bin)
    if not math.isfinite(width) or width <= 0 or width > 10:
        raise ValueError("bin_width must be within (0, 10]")
    if minimum <= 0:
        raise ValueError("minimum_trades_per_bin must be positive")

    grouped: dict[int, list[tuple[float, CanonicalCompletedTrade, float | None]]] = {}
    scored = 0
    r_scored = 0
    invalid_scores = 0
    for trade in trades:
        quality = trade.outcome.get("signal_quality")
        if not isinstance(quality, Mapping):
            continue
        try:
            score = float(quality.get(score_key))
        except (TypeError, ValueError):
            invalid_scores += 1
            continue
        if not math.isfinite(score) or not 0.0 <= score <= 10.0:
            invalid_scores += 1
            continue
        r_multiple = trade.outcome.get("r_multiple")
        try:
            r_value = float(r_multiple) if r_multiple is not None else None
        except (TypeError, ValueError):
            r_value = None
        if r_value is not None and not math.isfinite(r_value):
            r_value = None
        if r_value is not None:
            r_scored += 1
        scored += 1
        bin_index = min(int(score / width), max(0, math.ceil(10.0 / width) - 1))
        grouped.setdefault(bin_index, []).append((score, trade, r_value))

    bins: list[ScoreCalibrationBin] = []
    for bin_index, sample in sorted(grouped.items()):
        scores = [item[0] for item in sample]
        net_values = [item[1].net_pnl for item in sample]
        r_values = [item[2] for item in sample if item[2] is not None]
        net_interval = bootstrap_mean_interval(
            net_values,
            samples=bootstrap_samples,
            seed=seed + bin_index,
        )
        r_interval = (
            bootstrap_mean_interval(
                r_values,
                samples=bootstrap_samples,
                seed=seed + 10_000 + bin_index,
            )
            if r_values
            else None
        )
        lower = bin_index * width
        upper = min(10.0, lower + width)
        bins.append(
            ScoreCalibrationBin(
                lower=round(lower, 4),
                upper=round(upper, 4),
                trade_count=len(sample),
                r_trade_count=len(r_values),
                mean_score=round(sum(scores) / len(scores), 4),
                net_expectancy=round(net_interval.estimate, 4),
                net_expectancy_ci_lower=round(net_interval.lower, 4),
                net_expectancy_ci_upper=round(net_interval.upper, 4),
                mean_r=round(r_interval.estimate, 4) if r_interval else None,
                mean_r_ci_lower=round(r_interval.lower, 4) if r_interval else None,
                mean_r_ci_upper=round(r_interval.upper, 4) if r_interval else None,
                win_rate=round(
                    sum(value > 0 for value in net_values) / len(net_values),
                    4,
                ),
                evidence_ready=len(sample) >= minimum,
            )
        )

    blockers: list[str] = []
    if scored == 0:
        blockers.append(f"missing_score:{score_key}")
    if invalid_scores:
        blockers.append(f"invalid_score:{score_key}:{invalid_scores}")
    underpowered = sum(not item.evidence_ready for item in bins)
    if underpowered:
        blockers.append(f"underpowered_bins:{underpowered}")
    return ScoreCalibrationReport(
        score_key=score_key,
        total_trades=len(trades),
        scored_trades=scored,
        r_scored_trades=r_scored,
        minimum_trades_per_bin=minimum,
        bins=tuple(bins),
        blockers=tuple(blockers),
    )


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
    """Fail closed unless every requested trigger has decision-time score evidence."""

    coverage: dict[str, ComponentCoverage] = {}
    blockers: list[str] = []
    for raw_component in required_components:
        component = _canonical_strategy(raw_component)
        component_trades = [trade for trade in trades if trade.strategy == component]
        with_quality = sum(
            _has_signal_quality(trade.outcome) for trade in component_trades
        )
        with_attribution = sum(
            _has_attribution_provenance(trade.outcome) for trade in component_trades
        )
        coverage[component] = ComponentCoverage(
            completed_trades=len(component_trades),
            with_signal_quality=with_quality,
            with_attribution_provenance=with_attribution,
        )
        if not component_trades:
            blockers.append(f"missing_completed_trades:{component}")
        else:
            if with_quality != len(component_trades):
                blockers.append(f"missing_signal_quality:{component}")
            if with_attribution != len(component_trades):
                blockers.append(f"missing_attribution_provenance:{component}")

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
    "ChronologicalBlock",
    "CompletedTradeSummary",
    "ComponentCoverage",
    "WalkForwardFold",
    "WalkForwardStability",
    "ExecutionDataQuality",
    "ScoreCalibrationBin",
    "ScoreCalibrationReport",
    "attribution_readiness",
    "calibrate_signal_scores",
    "execution_data_quality",
    "canonicalize_completed_trades",
    "chronological_post_cost_blocks",
    "chronological_walk_forward",
    "walk_forward_stability",
    "summarize_completed_trades",
]
