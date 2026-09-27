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


@dataclass(frozen=True, slots=True)
class CanonicalCompletedTrade:
    """One economically complete trade in chronological research form."""

    trade_id: str
    closed_at: float
    strategy: str
    gross_pnl: float
    estimated_costs: float
    net_pnl: float
    outcome: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CompletedTradeSummary:
    """Post-cost performance summary for one chronological sample."""

    trade_count: int
    gross_pnl: float
    estimated_costs: float
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
class ComponentCoverage:
    """Observed completed-trade coverage for one trigger strategy."""

    completed_trades: int
    with_signal_quality: int


@dataclass(frozen=True, slots=True)
class AttributionReadiness:
    """Whether completed trades contain enough fields for component attribution."""

    ready: bool
    coverage: Mapping[str, ComponentCoverage]
    blockers: tuple[str, ...]


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
        if isinstance(value, bool):
            return False
        try:
            resolved = float(value)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(resolved):
            return False
    return True


def canonicalize_completed_trades(
    rows: Sequence[Mapping[str, Any]],
    *,
    net_identity_tolerance: float = 0.02,
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
        net_pnl = _finite_number(row.get("net_pnl"), field="net_pnl")
        expected_net = gross_pnl - estimated_costs
        if abs(expected_net - net_pnl) > tolerance:
            raise ValueError(
                "completed trade violates gross_pnl - estimated_costs = net_pnl "
                f"within tolerance: {trade_id}"
            )

        trades.append(
            CanonicalCompletedTrade(
                trade_id=trade_id,
                closed_at=closed_at,
                strategy=_canonical_strategy(row.get("strategy")),
                gross_pnl=gross_pnl,
                estimated_costs=estimated_costs,
                net_pnl=net_pnl,
                outcome=_outcome(row),
            )
        )

    trades.sort(key=lambda trade: (trade.closed_at, trade.trade_id))
    return tuple(trades)


def summarize_completed_trades(
    trades: Sequence[CanonicalCompletedTrade],
) -> CompletedTradeSummary:
    """Summarize realized economics using net P&L after estimated costs."""

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
        coverage[component] = ComponentCoverage(
            completed_trades=len(component_trades),
            with_signal_quality=with_quality,
        )
        if not component_trades:
            blockers.append(f"missing_completed_trades:{component}")
        elif with_quality != len(component_trades):
            blockers.append(f"missing_signal_quality:{component}")

    return AttributionReadiness(
        ready=not blockers,
        coverage=coverage,
        blockers=tuple(blockers),
    )


__all__ = [
    "AttributionReadiness",
    "CanonicalCompletedTrade",
    "ChronologicalBlock",
    "CompletedTradeSummary",
    "ComponentCoverage",
    "attribution_readiness",
    "canonicalize_completed_trades",
    "chronological_post_cost_blocks",
    "summarize_completed_trades",
]
