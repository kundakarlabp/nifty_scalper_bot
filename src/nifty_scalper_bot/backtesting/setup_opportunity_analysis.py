"""Research-only setup opportunity and counterfactual outcome analysis.

This module is intentionally outside the live trading path. It consumes persisted
runner decisions and already-recorded market observations; it never generates a
signal, infers direction, changes a threshold, sizes an order, or routes execution.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


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
    decision_count: int
    approved: bool
    final_reason: str
    regime: str
    research_context: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ForwardPathLabel:
    """Executable-side counterfactual path label for one option-buy opportunity."""

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


@dataclass(frozen=True, slots=True)
class PolicyCounterfactual:
    """Descriptive threshold result; never a live parameter-selection verdict."""

    policy: str
    eligible: int
    selected: int
    mean_r: float | None
    positive_fraction: float | None


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
            ).strip().upper()
            for item in sample
        }
        symbols = {
            str(
                item[1].get("selected_candidate")
                or item[1].get("symbol")
                or ""
            ).strip()
            for item in sample
            if str(
                item[1].get("selected_candidate")
                or item[1].get("symbol")
                or ""
            ).strip()
        }
        if len(strategies) != 1 or len(sides) != 1:
            continue
        last_ts, last_row, last_context = sample[-1]
        setup_id = str(
            last_context.get("setup_id")
            or last_context.get("setup_structure_id")
            or ""
        ).strip()
        event = str(last_row.get("event_name") or "").strip()
        final_reason = str(
            last_row.get("reason_code")
            or last_row.get("final_reason")
            or "unknown"
        ).strip()
        symbol = str(
            last_row.get("selected_candidate") or last_row.get("symbol") or ""
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
                decision_count=len(sample),
                approved=event == "candidate.approved",
                final_reason=final_reason,
                regime=str(last_context.get("regime") or "").strip().upper(),
                research_context=dict(last_context),
            )
        )
    opportunities.sort(key=lambda item: (item.first_seen_ts, item.opportunity_id))
    return tuple(opportunities)


def label_forward_option_buy_path(
    observations: Sequence[Mapping[str, Any]],
    *,
    decision_ts: float,
    entry_price: float,
    risk_points: float,
    horizon_seconds: float,
    price_field: str = "bid",
) -> ForwardPathLabel | None:
    """Label a hypothetical long-option path using executable-side observations."""

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

    path: list[tuple[float, float]] = []
    for row in observations:
        try:
            ts = _timestamp(row.get("timestamp"), field="observation timestamp")
            price = float(row.get(price_field))
        except (TypeError, ValueError):
            continue
        if (
            not math.isfinite(price)
            or price <= 0
            or ts < start
            or ts > start + horizon
        ):
            continue
        path.append((ts, price))
    if not path:
        return None
    path.sort(key=lambda item: item[0])
    favorable = [(price - entry, ts) for ts, price in path]
    adverse = [(entry - price, ts) for ts, price in path]
    mfe, mfe_ts = max(favorable, key=lambda item: item[0])
    mae, mae_ts = max(adverse, key=lambda item: item[0])
    terminal_price = path[-1][1]
    terminal_r = (terminal_price - entry) / risk
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
    )


def compare_score_policy(
    samples: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    use_regime_weight: bool,
) -> PolicyCounterfactual:
    """Describe score-threshold selection using already-labelled opportunity rows."""

    cutoff = float(threshold)
    if not math.isfinite(cutoff):
        raise ValueError("threshold must be finite")
    selected_r: list[float] = []
    eligible = 0
    for row in samples:
        try:
            raw = float(row["raw_setup_score"])
            outcome_r = float(row["outcome_r"])
            weight = float(row.get("regime_weight", 1.0))
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in (raw, outcome_r, weight)):
            continue
        eligible += 1
        score = raw * weight if use_regime_weight else raw
        if score >= cutoff:
            selected_r.append(outcome_r)
    return PolicyCounterfactual(
        policy="regime_weighted" if use_regime_weight else "neutral_weight",
        eligible=eligible,
        selected=len(selected_r),
        mean_r=round(sum(selected_r) / len(selected_r), 6) if selected_r else None,
        positive_fraction=(
            round(sum(value > 0 for value in selected_r) / len(selected_r), 6)
            if selected_r
            else None
        ),
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
    "PolicyCounterfactual",
    "canonicalize_setup_opportunities",
    "compare_score_policy",
    "label_forward_option_buy_path",
    "validate_experiment_record",
]
