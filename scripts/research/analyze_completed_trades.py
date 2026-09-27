"""Read-only evidence gate for canonical completed-trade research.

This utility deliberately stops short of parameter optimization. It consumes the
canonical one-row-per-trade ledger, validates post-cost economics, reports
chronological stability and descriptive strategy/component outcomes, and makes
counterfactual walk-forward evidence an explicit prerequisite for tuning.
"""

from __future__ import annotations

import argparse
import collections
import collections.abc
import json
import math
import sqlite3
from pathlib import Path
from typing import Any


_ECONOMIC_TOLERANCE_RUPEES = 0.02
_QUALITY_COMPONENTS = ("alpha_score", "direction_score", "strategy_score")
_TARGET_STRATEGIES = ("ORBPro", "SMC", "VWAPPro")
_STALE_EXIT_MARKER = "src=ltp_stale_quote"


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_object(raw: Any) -> dict[str, Any]:
    if isinstance(raw, collections.abc.Mapping):
        return dict(raw)
    if raw in (None, ""):
        return {}
    try:
        parsed = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(parsed) if isinstance(parsed, collections.abc.Mapping) else {}


def load_canonical_completed_trades(db_path: str | Path) -> list[dict[str, Any]]:
    """Load ledger-complete closed trades in deterministic chronological order."""

    resolved = Path(db_path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"trade ledger not found: {resolved}")
    uri = f"{resolved.as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute(
            """
            SELECT
                trade_id,
                strategy,
                symbol,
                side,
                gross_pnl,
                estimated_costs,
                net_pnl,
                entry_filled_at,
                closed_at,
                exit_reason,
                build_sha,
                outcome_json
            FROM trade_ledger
            WHERE state = 'CLOSED'
              AND ledger_complete = 1
            ORDER BY closed_at, trade_id
            """
        ).fetchall()

    trades = [
        {
            "trade_id": row[0],
            "strategy": row[1],
            "symbol": row[2],
            "side": row[3],
            "gross_pnl": row[4],
            "estimated_costs": row[5],
            "net_pnl": row[6],
            "entry_filled_at": row[7],
            "closed_at": row[8],
            "exit_reason": row[9],
            "build_sha": row[10],
            "outcome": _json_object(row[11]),
        }
        for row in rows
    ]
    validate_canonical_completed_trades(trades)
    return trades


def validate_canonical_completed_trades(
    trades: collections.abc.Sequence[collections.abc.Mapping[str, Any]],
) -> None:
    """Fail closed when canonical economics or chronological identity are invalid."""

    seen: set[str] = set()
    previous_key: tuple[float, str] | None = None
    for index, trade in enumerate(trades):
        trade_id = str(trade.get("trade_id") or "").strip()
        if not trade_id:
            raise ValueError(f"trade at index {index} is missing trade_id")
        if trade_id in seen:
            raise ValueError(f"duplicate canonical trade_id: {trade_id}")
        seen.add(trade_id)

        gross = _finite_number(trade.get("gross_pnl"))
        costs = _finite_number(trade.get("estimated_costs"))
        net = _finite_number(trade.get("net_pnl"))
        closed_at = _finite_number(trade.get("closed_at"))
        if gross is None or costs is None or net is None or closed_at is None:
            raise ValueError(f"incomplete canonical economics for {trade_id}")
        if costs < 0:
            raise ValueError(f"negative estimated costs for {trade_id}")
        identity_error = abs((gross - costs) - net)
        if identity_error > _ECONOMIC_TOLERANCE_RUPEES:
            raise ValueError(
                f"net pnl identity failed for {trade_id}: error={identity_error:.4f}"
            )

        current_key = (closed_at, trade_id)
        if previous_key is not None and current_key <= previous_key:
            raise ValueError(
                "canonical trades must be strictly chronologically ordered"
            )
        previous_key = current_key


def performance_summary(
    trades: collections.abc.Sequence[collections.abc.Mapping[str, Any]],
) -> dict[str, Any]:
    gross = [_finite_number(trade.get("gross_pnl")) or 0.0 for trade in trades]
    costs = [_finite_number(trade.get("estimated_costs")) or 0.0 for trade in trades]
    net = [_finite_number(trade.get("net_pnl")) or 0.0 for trade in trades]
    wins = [value for value in net if value > 0]
    losses = [value for value in net if value < 0]

    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for value in net:
        equity += value
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    trade_count = len(net)
    return {
        "trade_count": trade_count,
        "gross_pnl": round(sum(gross), 2),
        "estimated_costs": round(sum(costs), 2),
        "net_pnl": round(sum(net), 2),
        "expectancy": round(sum(net) / trade_count, 4) if trade_count else 0.0,
        "win_rate": round(len(wins) / trade_count, 4) if trade_count else 0.0,
        "average_win": round(sum(wins) / len(wins), 4) if wins else None,
        "average_loss": round(sum(losses) / len(losses), 4) if losses else None,
        "profit_factor": (
            round(gross_profit / gross_loss, 4) if gross_loss > 0 else None
        ),
        "max_drawdown": round(max_drawdown, 2),
    }


def chronological_blocks(
    trades: collections.abc.Sequence[collections.abc.Mapping[str, Any]],
    *,
    block_size: int,
) -> list[dict[str, Any]]:
    """Return non-overlapping chronological post-cost evidence blocks."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    rows: list[dict[str, Any]] = []
    for start in range(0, len(trades), block_size):
        block = list(trades[start : start + block_size])
        if not block:
            continue
        rows.append(
            {
                "block": len(rows) + 1,
                "complete_block": len(block) == block_size,
                "first_closed_at": block[0]["closed_at"],
                "last_closed_at": block[-1]["closed_at"],
                **performance_summary(block),
            }
        )
    return rows


def grouped_performance(
    trades: collections.abc.Sequence[collections.abc.Mapping[str, Any]],
    *,
    fields: collections.abc.Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[collections.abc.Mapping[str, Any]]] = collections.defaultdict(list)
    for trade in trades:
        outcome = trade.get("outcome")
        outcome_map = outcome if isinstance(outcome, collections.abc.Mapping) else {}
        values = []
        for field in fields:
            if field == "strategy_profile_version":
                value = outcome_map.get(field)
            else:
                value = trade.get(field)
            values.append(str(value or "UNKNOWN"))
        grouped[tuple(values)].append(trade)

    rows: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        row = {field: value for field, value in zip(fields, key)}
        row.update(performance_summary(group))
        rows.append(row)
    return rows


def component_score_summary(
    trades: collections.abc.Sequence[collections.abc.Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Describe outcome by recorded decision-time score; never claim causality."""

    grouped: dict[tuple[str, str, int], list[collections.abc.Mapping[str, Any]]] = collections.defaultdict(list)
    for trade in trades:
        outcome = trade.get("outcome")
        if not isinstance(outcome, collections.abc.Mapping):
            continue
        quality = outcome.get("signal_quality")
        if not isinstance(quality, collections.abc.Mapping):
            continue
        strategy = str(trade.get("strategy") or "UNKNOWN")
        for component in _QUALITY_COMPONENTS:
            score = _finite_number(quality.get(component))
            if score is None:
                continue
            grouped[(strategy, component, math.floor(score))].append(trade)

    rows: list[dict[str, Any]] = []
    for (strategy, component, lower), group in sorted(grouped.items()):
        rows.append(
            {
                "strategy": strategy,
                "component": component,
                "score_bucket": f"{lower}.0-<{lower + 1}.0",
                **performance_summary(group),
            }
        )
    return rows


def evidence_coverage(
    trades: collections.abc.Sequence[collections.abc.Mapping[str, Any]],
) -> dict[str, Any]:
    with_quality = 0
    with_final_score = 0
    with_context_confirmation = 0
    strategy_counts = {strategy: 0 for strategy in _TARGET_STRATEGIES}
    missing_entry_filled_at = 0

    for trade in trades:
        strategy = str(trade.get("strategy") or "")
        if strategy in strategy_counts:
            strategy_counts[strategy] += 1
        if _finite_number(trade.get("entry_filled_at")) is None:
            missing_entry_filled_at += 1

        outcome = trade.get("outcome")
        if not isinstance(outcome, collections.abc.Mapping):
            continue
        quality = outcome.get("signal_quality")
        if isinstance(quality, collections.abc.Mapping) and any(
            _finite_number(quality.get(component)) is not None
            for component in _QUALITY_COMPONENTS
        ):
            with_quality += 1
        if _finite_number(outcome.get("final_score")) is not None:
            with_final_score += 1
        evidence = outcome.get("context_confirmation_evidence")
        if isinstance(evidence, list) and evidence:
            with_context_confirmation += 1

    count = len(trades)

    def _coverage(value: int) -> float:
        return round(value / count, 4) if count else 0.0

    return {
        "trade_count": count,
        "missing_entry_filled_at": missing_entry_filled_at,
        "signal_quality_trades": with_quality,
        "signal_quality_coverage": _coverage(with_quality),
        "final_score_trades": with_final_score,
        "final_score_coverage": _coverage(with_final_score),
        "context_confirmation_trades": with_context_confirmation,
        "context_confirmation_coverage": _coverage(with_context_confirmation),
        "target_strategy_counts": strategy_counts,
    }


def execution_data_quality(
    trades: collections.abc.Sequence[collections.abc.Mapping[str, Any]],
) -> dict[str, Any]:
    """Report known degraded exit-price evidence without silently excluding it."""

    flagged = [
        trade
        for trade in trades
        if _STALE_EXIT_MARKER in str(trade.get("exit_reason") or "")
    ]
    not_flagged = [trade for trade in trades if trade not in flagged]
    count = len(trades)
    return {
        "known_stale_quote_exit_trades": len(flagged),
        "known_stale_quote_exit_fraction": (
            round(len(flagged) / count, 4) if count else 0.0
        ),
        "known_stale_quote_exit_performance": performance_summary(flagged),
        "not_explicitly_stale_exit_performance": performance_summary(not_flagged),
        "interpretation": (
            "not_explicitly_stale does not prove clean execution; the flag only "
            "identifies trades whose recorded exit reason names ltp_stale_quote"
        ),
    }


def build_evidence_report(
    trades: collections.abc.Sequence[collections.abc.Mapping[str, Any]],
    *,
    block_size: int = 20,
) -> dict[str, Any]:
    ordered = list(trades)
    validate_canonical_completed_trades(ordered)
    coverage = evidence_coverage(ordered)
    execution_quality = execution_data_quality(ordered)
    target_counts = coverage["target_strategy_counts"]
    missing_target_strategy = [
        strategy for strategy in _TARGET_STRATEGIES if target_counts[strategy] == 0
    ]
    component_ready = (
        coverage["signal_quality_coverage"] == 1.0 and not missing_target_strategy
    )
    blocks = chronological_blocks(ordered, block_size=block_size)
    complete_blocks = sum(bool(block["complete_block"]) for block in blocks)

    parameter_reasons = [
        "completed-trade outcomes are observational and do not provide "
        "counterfactual candidate trades",
        "parameter selection requires chronological replay with non-overlapping "
        "test windows",
    ]
    if not component_ready:
        parameter_reasons.append(
            "ORB/SMC/VWAP decision-time component coverage is incomplete"
        )
    if execution_quality["known_stale_quote_exit_trades"]:
        parameter_reasons.append(
            "historical outcomes include exits explicitly sourced from "
            "ltp_stale_quote"
        )

    return {
        "dataset": {
            "status": "READY" if ordered else "EMPTY",
            "canonical_trade_count": len(ordered),
            "economic_identity_tolerance_rupees": _ECONOMIC_TOLERANCE_RUPEES,
            "coverage": coverage,
            "execution_data_quality": execution_quality,
        },
        "post_cost_baseline": performance_summary(ordered),
        "chronological_blocks": blocks,
        "strategy_attribution": grouped_performance(ordered, fields=("strategy",)),
        "strategy_profile_attribution": grouped_performance(
            ordered,
            fields=("strategy", "strategy_profile_version"),
        ),
        "component_score_attribution": component_score_summary(ordered),
        "gates": {
            "canonical_dataset": "PASS" if ordered else "BLOCKED_EMPTY",
            "chronological_post_cost": (
                "PASS" if complete_blocks >= 1 else "BLOCKED_INSUFFICIENT_TRADES"
            ),
            "component_attribution": (
                "DESCRIPTIVE_READY"
                if component_ready
                else "BLOCKED_INCOMPLETE_DECISION_EVIDENCE"
            ),
            "counterfactual_walk_forward": "REQUIRED",
            "parameter_changes": "BLOCKED",
            "parameter_change_reasons": parameter_reasons,
        },
    }


def write_report(path: str | Path, report: collections.abc.Mapping[str, Any]) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(dict(report), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return resolved


def main(argv: collections.abc.Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trades-db", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    trades = load_canonical_completed_trades(args.trades_db)
    report = build_evidence_report(trades, block_size=args.block_size)
    if args.output is None:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        write_report(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
