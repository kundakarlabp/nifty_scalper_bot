"""Research-only setup opportunity and counterfactual outcome analysis.

Responsibilities: Deduplicate structural decisions and first-observed
RangeVWAP futures excursions across restarts; preserve their original contract,
decision context and chronological forward-path provenance.
Boundaries: Consumes persisted decisions and observed quotes; never places orders.
Counterfactual stratification uses one row per unique setup and never promotes
unlabelled or thinly labelled strata into profitability claims.
Safety: Later blocked candidates cannot overwrite an approved setup identity;
mixed or unverifiable option-contract quote paths cannot imply trade outcomes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from nifty_scalper_bot.utils.symbols import normalize_symbol


@dataclass(frozen=True, slots=True)
class CanonicalSetupOpportunity:
    """One independent structural setup, deduplicated across evaluation cycles."""

    opportunity_id: str
    setup_id: str
    strategy: str
    symbol: str
    observed_symbols: tuple[str, ...]
    side: str
    first_seen_ts: float
    last_seen_ts: float
    decision_ts: float
    decision_count: int
    approved: bool
    final_reason: str
    regime: str
    research_context: Mapping[str, Any]
    observed_stages: tuple[str, ...] = ()
    blocked_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ForwardPathLabel:
    """Observed-bid path diagnostic, not proof of executable broker fills."""

    observation_count: int
    horizon_seconds: float
    entry_price: float
    risk_points: float
    max_favorable_excursion: float
    max_adverse_excursion: float
    mfe_r: float
    mae_r: float
    time_to_mfe_seconds: float
    time_to_mae_seconds: float
    terminal_price: float
    terminal_r: float
    hit_positive_1r: bool
    hit_negative_1r: bool
    first_barrier: str | None = None
    time_to_first_barrier_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    """Immutable research provenance for one pre-declared comparison."""

    experiment_id: str
    hypothesis: str
    parameters: Mapping[str, Any]
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    cost_model: str


def _timestamp(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a timestamp")
    if isinstance(value, (int, float)):
        resolved = float(value)
        if math.isfinite(resolved):
            return resolved
        raise ValueError(f"{field} must be finite")
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} is required")
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _research_context(row: Mapping[str, Any]) -> dict[str, Any]:
    meta = row.get("meta")
    if isinstance(meta, Mapping) and isinstance(meta.get("research_context"), Mapping):
        return dict(meta["research_context"])
    value = row.get("research_context")
    return dict(value) if isinstance(value, Mapping) else {}


def canonicalize_setup_opportunities(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[CanonicalSetupOpportunity, ...]:
    """Collapse repeated candidate evaluations to one structural setup identity.

    Rows without a strategy-owned setup identity are ignored rather than assigned
    a synthetic identity. Conflicting strategy/side facts for the same setup
    fail closed because such rows cannot support trustworthy attribution.
    """

    grouped: dict[str, list[tuple[float, Mapping[str, Any], dict[str, Any]]]] = {}
    for row in rows:
        event = str(row.get("event_name") or "").strip()
        if event not in {"candidate.approved", "candidate.blocked"}:
            continue
        context = _research_context(row)
        setup_id = str(
            context.get("setup_id") or context.get("setup_structure_id") or ""
        ).strip()
        if not setup_id:
            continue
        ts_value = row.get("timestamp")
        if ts_value in (None, ""):
            ts_value = row.get("created_at")
        ts = _timestamp(ts_value, field="decision timestamp")
        strategy = str(context.get("strategy") or "").strip().lower()
        side = (
            str(
                context.get("contract_side")
                or context.get("trade_side")
                or row.get("direction")
                or ""
            )
            .strip()
            .upper()
        )
        if not strategy or side not in {"CE", "PE"}:
            continue
        key = f"{strategy}:{side}:{setup_id}"
        grouped.setdefault(key, []).append((ts, row, context))

    opportunities: list[CanonicalSetupOpportunity] = []
    for key, sample in grouped.items():
        sample.sort(key=lambda item: item[0])
        strategies = {
            str(item[2].get("strategy") or "").strip().lower() for item in sample
        }
        sides = {
            str(
                item[2].get("contract_side")
                or item[2].get("trade_side")
                or item[1].get("direction")
                or ""
            )
            .strip()
            .upper()
            for item in sample
        }
        symbols = {
            str(
                item[1].get("selected_candidate") or item[1].get("symbol") or ""
            ).strip()
            for item in sample
            if str(
                item[1].get("selected_candidate") or item[1].get("symbol") or ""
            ).strip()
        }
        if len(strategies) != 1 or len(sides) != 1:
            continue
        last_ts, _, last_context = sample[-1]
        approved_rows = [
            item for item in sample if item[1].get("event_name") == "candidate.approved"
        ]
        # Keep the latest approval's contract/context together with its outcome;
        # a later rejection on a rotated contract is not the approved trade.
        outcome_ts, outcome_row, outcome_context = (
            approved_rows[-1] if approved_rows else sample[-1]
        )
        setup_id = str(
            last_context.get("setup_id") or last_context.get("setup_structure_id") or ""
        ).strip()
        final_reason = str(
            outcome_row.get("reason_code")
            or outcome_row.get("final_reason")
            or "unknown"
        ).strip()
        observed_stage_values: set[str] = set()
        blocked_reason_values: set[str] = set()
        for _, row, context in sample:
            stage = str(
                context.get("decision_stage") or context.get("rejection_stage") or ""
            ).strip()
            if stage:
                observed_stage_values.add(stage)
            if row.get("event_name") == "candidate.blocked":
                reason = str(
                    row.get("reason_code") or row.get("final_reason") or "unknown"
                ).strip()
                blocked_reason_values.add(reason)
        observed_stages = tuple(sorted(observed_stage_values))
        blocked_reasons = tuple(sorted(blocked_reason_values))
        symbol = str(
            outcome_row.get("selected_candidate") or outcome_row.get("symbol") or ""
        ).strip()
        opportunities.append(
            CanonicalSetupOpportunity(
                opportunity_id=key,
                setup_id=setup_id,
                strategy=next(iter(strategies)),
                symbol=symbol,
                observed_symbols=tuple(sorted(symbols)),
                side=next(iter(sides)),
                first_seen_ts=sample[0][0],
                last_seen_ts=last_ts,
                decision_ts=outcome_ts,
                decision_count=len(sample),
                approved=bool(approved_rows),
                final_reason=final_reason,
                regime=str(outcome_context.get("regime") or "").strip().upper(),
                research_context=dict(outcome_context),
                observed_stages=observed_stages,
                blocked_reasons=blocked_reasons,
            )
        )
    opportunities.sort(key=lambda item: (item.first_seen_ts, item.opportunity_id))
    return tuple(opportunities)


def canonicalize_range_excursions(
    observations: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Use the first causal futures-excursion observation, across restarts.

    Do not cherry-pick a later option/quote after seeing a reversal. Only
    same-evaluation economic evidence can license a modeled comparison.
    """
    grouped: dict[str, list[tuple[float, Mapping[str, Any]]]] = {}
    for row in observations:
        identity = str(row.get("excursion_id") or "")
        future = str(row.get("futures_symbol") or "")
        side = str(row.get("side") or "")
        first_bar_identity = str(row.get("excursion_first_bar") or "")
        if (
            not future.startswith("NFO:")
            or not future.endswith("FUT")
            or side not in {"CE", "PE"}
            or not first_bar_identity
            or identity != f"{future}:{side}:{first_bar_identity}"
        ):
            continue
        try:
            ts = _timestamp(row.get("decision_at"), field="excursion decision")
            first_ts = _timestamp(first_bar_identity, field="first outside bar")
        except (TypeError, ValueError, OverflowError):
            continue
        if not first_ts < ts or ts - first_ts > 375 * 60:
            continue
        grouped.setdefault(identity, []).append((ts, row))

    decision_rows: dict[str, list[tuple[float, Mapping[str, Any]]]] = {}
    for row in decisions:
        identity = str(row.get("excursion_id") or "")
        if identity not in grouped:
            continue
        try:
            ts = _timestamp(row.get("decision_at"), field="stage decision")
        except (TypeError, ValueError, OverflowError):
            continue
        decision_rows.setdefault(identity, []).append((ts, row))

    unique: list[dict[str, Any]] = []
    for identity, samples in grouped.items():
        samples.sort(key=lambda item: item[0])
        first_ts, first = samples[0]
        side = str(first["side"])
        future = str(first["futures_symbol"])
        symbol = normalize_symbol(str(first.get("symbol") or ""))
        conflicts = any(
            str(row.get("futures_symbol")) != future
            or str(row.get("side")) != side
            or normalize_symbol(str(row.get("symbol") or "")) != symbol
            for _, row in samples
        )
        stage_events = sorted(decision_rows.get(identity, ()), key=lambda it: it[0])
        reasons = list(
            dict.fromkeys(
                str(row.get("reason") or "unavailable") for _, row in stage_events
            )
        )
        # Only the exact original evaluation timestamp may establish its risk.
        # Even a 1-second-later evaluation may use a different executable ask.
        same_decision = [
            (ts, row)
            for ts, row in stage_events
            if abs(ts - first_ts) < 0.001
            and normalize_symbol(str(row.get("symbol") or "")) == symbol
        ]
        economics = [
            row
            for _, row in same_decision
            if row.get("reason") in {"cost_eligible", "signal_emitted"}
            and row.get("modeled_stop_risk") is not None
            and row.get("modeled_net_rr") is not None
        ]
        risk = economics[-1] if economics else {}
        try:
            stop_risk = float(risk["modeled_stop_risk"])
            net_rr = float(risk["modeled_net_rr"])
            if not (math.isfinite(stop_risk) and math.isfinite(net_rr)):
                raise ValueError
        except (KeyError, TypeError, ValueError, OverflowError):
            stop_risk, net_rr = None, None
        initial_rejection = next(
            (
                str(row.get("reason"))
                for _, row in same_decision
                if row.get("reason") not in {"cost_eligible", "signal_emitted"}
            ),
            str(first.get("structural_reason") or "decision_not_captured"),
        )
        unique.append(
            {
                "excursion_id": identity,
                "decision_timestamp": first_ts,
                "futures_symbol": future,
                "side": side,
                "symbol": symbol,
                "regime": first.get("regime"),
                "regime_raw": first.get("regime_raw"),
                "regime_gate_allowed": first.get("regime_gate_allowed"),
                "shadow_transition_candidate": first.get("shadow_transition_candidate"),
                "local_range_candidate": first.get("local_range_candidate"),
                "local_range_state": first.get("local_range_state"),
                "local_range_evidence_fraction": first.get(
                    "local_range_evidence_fraction"
                ),
                "local_reclaim_side": first.get("local_reclaim_side"),
                "local_range_confirmed_in_trend": first.get(
                    "local_range_confirmed_in_trend"
                ),
                "local_rolling_center_slope_bps_per_min": first.get(
                    "local_rolling_center_slope_bps_per_min"
                ),
                "local_range_session_vwap_setup": first.get(
                    "local_range_session_vwap_setup"
                ),
                "structural_reason": first.get("structural_reason"),
                "structural_qualified": first.get("structural_qualified") is True,
                "first_rejection_reason": initial_rejection,
                "observed_decision_reasons": reasons,
                "same_decision_signal_emitted": any(
                    row.get("reason") == "signal_emitted" for _, row in same_decision
                ),
                "modeled_stop_risk": stop_risk,
                "modeled_net_rr": net_rr,
                "quote": first.get("quote"),
                "center_gap_atr": first.get("center_gap_atr"),
                "session_vwap": first.get("session_vwap"),
                "rolling_center": first.get("rolling_center"),
                "futures_atr": first.get("futures_atr"),
                "range_width_atr": first.get("range_width_atr"),
                "volatility_ratio": first.get("volatility_ratio"),
                "directional_efficiency": first.get("directional_efficiency"),
                "duplicate_restart_observations": len(samples) - 1,
                "contract_identity_conflicted": conflicts,
                "entry_evidence_from_first_observation_only": True,
            }
        )
    unique.sort(key=lambda row: (row["decision_timestamp"], row["excursion_id"]))
    return unique


def summarize_contextual_label_coverage(
    opportunities: Sequence[CanonicalSetupOpportunity],
    post_cost_labels: Mapping[str, float],
    *,
    min_labelled: int = 5,
) -> dict[str, Any]:
    """Stratify only independent setups; unresolved labels remain explicit."""
    from collections import defaultdict
    from zoneinfo import ZoneInfo

    if min_labelled < 1:
        raise ValueError("min_labelled must be positive")
    groups: dict[tuple[str, str], list[CanonicalSetupOpportunity]] = defaultdict(list)
    for item in opportunities:
        context = item.research_context
        hour = (
            datetime.fromtimestamp(item.first_seen_ts, timezone.utc)
            .astimezone(ZoneInfo("Asia/Kolkata"))
            .strftime("%H")
        )
        fields = {
            "strategy": item.strategy,
            "regime": item.regime or "UNKNOWN",
            "entry_hour_ist": hour,
            "volatility_regime": str(
                context.get("volatility_regime") or "UNKNOWN"
            ).upper(),
        }
        for key, value in fields.items():
            groups[(key, value)].append(item)

    results: list[dict[str, Any]] = []
    for (dimension, value), items in sorted(groups.items()):
        labelled: list[tuple[CanonicalSetupOpportunity, float]] = []
        for item in items:
            raw = post_cost_labels.get(item.opportunity_id)
            if raw is None or isinstance(raw, bool):
                continue
            try:
                observation = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(observation):
                labelled.append((item, observation))
        enough = len(labelled) >= min_labelled
        results.append(
            {
                "dimension": dimension,
                "value": value,
                "unique_setups": len(items),
                "labelled_setups": len(labelled),
                "unlabelled_setups": len(items) - len(labelled),
                "approved_labelled": sum(item.approved for item, _ in labelled),
                "blocked_labelled": sum(not item.approved for item, _ in labelled),
                "label_coverage_fraction": (
                    round(len(labelled) / len(items), 4) if items else 0.0
                ),
                "mean_post_cost_r": (
                    round(sum(label for _, label in labelled) / len(labelled), 6)
                    if enough
                    else None
                ),
                "descriptive_only": True,
                "reason": (
                    "descriptive_only_no_causal_gate_effect_claim"
                    if enough
                    else "insufficient_independent_post_cost_labels"
                ),
            }
        )
    return {
        "basis": "unique_setup_identity_not_repeated_votes",
        "source": "independently_supplied_post_cost_labels",
        "minimum_labels_for_descriptive_mean": min_labelled,
        "strata": results,
        "causal_gate_effect_verified": False,
    }


def label_forward_option_buy_path(
    observations: Sequence[Mapping[str, Any]],
    *,
    decision_ts: float,
    entry_price: float,
    risk_points: float,
    horizon_seconds: float,
    price_field: str = "bid",
    target_price: float | None = None,
    expected_symbol: str | None = None,
) -> ForwardPathLabel | None:
    """Label one option-bid path; reject mixed or mismatched contract quotes.

    This is a quote-level diagnostic, not broker fill or depth proof.
    """

    start = float(decision_ts)
    entry = float(entry_price)
    risk = float(risk_points)
    horizon = float(horizon_seconds)
    if not all(math.isfinite(value) for value in (start, entry, risk, horizon)):
        raise ValueError(
            "decision_ts, entry_price, risk_points and horizon must be finite"
        )
    if entry <= 0 or risk <= 0 or horizon <= 0:
        raise ValueError(
            "entry_price, risk_points and horizon_seconds must be positive"
        )
    target = float(target_price) if target_price is not None else None
    if target is not None and (not math.isfinite(target) or target <= entry):
        raise ValueError("target_price must be finite and above entry_price")
    expected = normalize_symbol(expected_symbol) if expected_symbol is not None else ""
    if expected_symbol is not None and not expected:
        raise ValueError("expected_symbol must identify an option contract")

    path: list[tuple[float, float]] = []
    observed_symbols: set[str] = set()
    for row in observations:
        try:
            ts = _timestamp(row.get("timestamp"), field="observation timestamp")
            raw_price = row.get(price_field)
            if raw_price is None:
                continue
            price = float(raw_price)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(price) or price <= 0 or ts < start or ts > start + horizon:
            continue
        observed_symbols.add(normalize_symbol(str(row.get("symbol") or "")))
        path.append((ts, price))
    if not path or len(observed_symbols) != 1:
        return None
    if expected_symbol is not None and observed_symbols != {expected}:
        return None
    path.sort(key=lambda item: item[0])
    favorable = [(price - entry, ts) for ts, price in path]
    adverse = [(entry - price, ts) for ts, price in path]
    mfe, mfe_ts = max(favorable, key=lambda item: item[0])
    mae, mae_ts = max(adverse, key=lambda item: item[0])
    terminal_price = path[-1][1]
    terminal_r = (terminal_price - entry) / risk
    first_barrier = None
    barrier_time = None
    if target is not None:
        for ts, price in path:
            if price <= entry - risk or price >= target:
                first_barrier = "stop" if price <= entry - risk else "target"
                if any(
                    observed_ts == ts
                    and (
                        observed_price >= target
                        if first_barrier == "stop"
                        else observed_price <= entry - risk
                    )
                    for observed_ts, observed_price in path
                ):
                    first_barrier = "ambiguous"
                barrier_time = round(ts - start, 6)
                break
    return ForwardPathLabel(
        observation_count=len(path),
        horizon_seconds=horizon,
        entry_price=entry,
        risk_points=risk,
        max_favorable_excursion=round(max(0.0, mfe), 6),
        max_adverse_excursion=round(max(0.0, mae), 6),
        mfe_r=round(max(0.0, mfe) / risk, 6),
        mae_r=round(max(0.0, mae) / risk, 6),
        time_to_mfe_seconds=round(max(0.0, mfe_ts - start), 6),
        time_to_mae_seconds=round(max(0.0, mae_ts - start), 6),
        terminal_price=round(terminal_price, 6),
        terminal_r=round(terminal_r, 6),
        hit_positive_1r=any((price - entry) >= risk for _, price in path),
        hit_negative_1r=any((entry - price) >= risk for _, price in path),
        first_barrier=first_barrier,
        time_to_first_barrier_seconds=barrier_time,
    )


def validate_experiment_record(
    record: ExperimentRecord,
    *,
    existing_ids: Sequence[str] = (),
) -> ExperimentRecord:
    """Validate immutable experiment provenance before a research run is recorded."""

    if not record.experiment_id.strip():
        raise ValueError("experiment_id is required")
    if record.experiment_id in set(existing_ids):
        raise ValueError(f"duplicate experiment_id:{record.experiment_id}")
    if not record.hypothesis.strip():
        raise ValueError("hypothesis is required")
    train_start = _timestamp(record.train_start, field="train_start")
    train_end = _timestamp(record.train_end, field="train_end")
    test_start = _timestamp(record.test_start, field="test_start")
    test_end = _timestamp(record.test_end, field="test_end")
    if train_start > train_end:
        raise ValueError("train window is reversed")
    if test_start > test_end:
        raise ValueError("test window is reversed")
    if test_start <= train_end:
        raise ValueError("test window must be strictly later than train window")
    if not record.cost_model.strip():
        raise ValueError("cost_model is required")
    return record


__all__ = [
    "CanonicalSetupOpportunity",
    "ExperimentRecord",
    "ForwardPathLabel",
    "canonicalize_range_excursions",
    "canonicalize_setup_opportunities",
    "label_forward_option_buy_path",
    "validate_experiment_record",
]
